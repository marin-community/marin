"""Regression tests for trajectory.json persistence to remote (UPath) log dirs.

Commit 3d170c8f coerced the agent ``logs_dir`` from a remote ``UPath``
(``gs://.../agent``) to a stdlib ``Path`` via ``Path(str(agent_dir))``. That
collapses the scheme (``gs://`` -> ``gs:/``) and turns the path into a local
filesystem path, so the trajectory dump silently wrote ``trajectory.json`` to a
bogus local directory instead of the trials_dir. The file never reached object
storage and downstream trace datasets came out empty.

These tests pin the contract that an agent given a remote-scheme ``logs_dir``
persists ``trajectory.json`` *to that scheme* via the background artifact
writer. ``memory://`` is an in-process fsspec backend, so it exercises the real
UPath write path with no network.
"""

import asyncio
import json
import threading
import uuid

import pytest
from harbor.utils import artifact_writer
from harbor.utils.artifact_writer import (
    ArtifactPersistenceError,
    ArtifactWriter,
    get_artifact_writer,
)
from upath import UPath

from tests.unit.agents.terminus_2.bare_agent import make_bare_terminus2


def test_persist_trajectory_writes_to_remote_upath():
    """A remote-scheme logs_dir must receive trajectory.json at that scheme."""
    logs_dir = UPath(f"memory://harbor-traj-{uuid.uuid4().hex}/agent")
    logs_dir.mkdir(parents=True, exist_ok=True)

    agent = make_bare_terminus2(logs_dir)
    agent._persist_trajectory(agent._summarization_count)
    get_artifact_writer().flush(prefix=str(logs_dir), timeout=30)

    trajectory_path = logs_dir / "trajectory.json"
    assert (
        trajectory_path.exists()
    ), "trajectory.json was not written to the remote logs_dir"

    data = json.loads(trajectory_path.read_text())
    assert data["session_id"] == agent._session_id
    assert len(data["steps"]) == 1
    assert data["steps"][0]["message"] == "hi"


async def test_final_trajectory_survives_optional_backlog_and_remains_latest(
    tmp_path, monkeypatch
):
    writer = ArtifactWriter(num_workers=1, queue_size=1)
    monkeypatch.setattr(artifact_writer, "_WRITER", writer)
    agent = make_bare_terminus2(tmp_path)
    trajectory_path = tmp_path / "trajectory.json"
    release = threading.Event()
    upload_started = asyncio.Event()
    flush_started = asyncio.Event()
    loop = asyncio.get_running_loop()
    flush = writer.flush_async

    async def observed_flush(*args, **kwargs):
        flush_started.set()
        return await flush(*args, **kwargs)

    monkeypatch.setattr(writer, "flush_async", observed_flush)

    def blocked_snapshot():
        loop.call_soon_threadsafe(upload_started.set)
        if not release.wait(timeout=10):
            raise TimeoutError("Test did not release the trajectory upload")
        return '{"steps": []}'

    writer.submit_optional(
        trajectory_path, blocked_snapshot, description="earlier snapshot"
    )
    await asyncio.wait_for(upload_started.wait(), 5)
    writer.submit_optional(
        trajectory_path, lambda: '{"steps": []}', description="queued stale snapshot"
    )
    finalizing = asyncio.create_task(agent._persist_final_trajectory())
    try:
        await asyncio.wait_for(flush_started.wait(), 5)
        assert not finalizing.done()
        release.set()
        await asyncio.wait_for(finalizing, 5)
        await writer.flush_async(timeout=5)
        data = json.loads(trajectory_path.read_text())
        assert data["steps"][0]["message"] == "hi"
        assert writer.optional_dropped_count == 0
    finally:
        release.set()
        await asyncio.gather(finalizing, return_exceptions=True)
        await writer.flush_async(timeout=5)


async def test_final_trajectory_repairs_previously_failed_optional_snapshot(
    tmp_path, monkeypatch
):
    writer = ArtifactWriter(num_workers=1)
    monkeypatch.setattr(artifact_writer, "_WRITER", writer)
    trajectory_path = tmp_path / "trajectory.json"

    def failed_snapshot():
        raise OSError("optional upload failed")

    earlier_write = writer.submit_optional(
        trajectory_path, failed_snapshot, description="failed optional snapshot"
    )
    with pytest.raises(OSError):
        await writer.await_write(earlier_write, timeout=5)
    agent = make_bare_terminus2(tmp_path)

    await agent._persist_final_trajectory()

    assert json.loads(trajectory_path.read_text())["steps"][0]["message"] == "hi"


async def test_final_trajectory_storage_failure_is_a_persistence_failure(
    tmp_path, monkeypatch
):
    writer = ArtifactWriter(num_workers=1)
    monkeypatch.setattr(artifact_writer, "_WRITER", writer)
    agent = make_bare_terminus2(tmp_path)
    trajectory_path = tmp_path / "trajectory.json"
    open_path = type(tmp_path).open

    def failed_upload(path, *args, **kwargs):
        if path == trajectory_path:
            raise OSError("trajectory upload failed")
        return open_path(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "open", failed_upload)

    with pytest.raises(ArtifactPersistenceError) as failure:
        await agent._persist_final_trajectory()

    assert isinstance(failure.value.__cause__, OSError)
    assert not trajectory_path.exists()


async def test_final_continuation_rejects_missing_initial_trajectory_segment(
    tmp_path, monkeypatch
):
    writer = ArtifactWriter(num_workers=1)
    monkeypatch.setattr(artifact_writer, "_WRITER", writer)
    agent = make_bare_terminus2(
        tmp_path, _linear_history=True, _continuation_segments_persisted=1
    )

    with pytest.raises(ArtifactPersistenceError):
        await agent._persist_final_trajectory()

    assert not (tmp_path / "trajectory.json").exists()
    assert not (tmp_path / "trajectory.cont-1.json").exists()
