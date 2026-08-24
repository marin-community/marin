# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import os
import tarfile
import threading
from pathlib import Path

import pytest
import zstandard
from iris.cluster.config import TaskOutputPolicy
from iris.cluster.runtime.output_capture import (
    ResolvedOutputDestination,
    TaskOutputLimits,
    capture_task_outputs,
    resolve_task_output_destination,
)
from iris.cluster.types import AttemptUid, JobName
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Deadline


def _capture(source: Path, destination: Path, *, max_bytes: int = 1024**2) -> job_pb2.TaskOutputArchive:
    return capture_task_outputs(
        source,
        ResolvedOutputDestination(
            path=StoragePath(str(destination)),
            retention=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_RETENTION_LOCAL_CLUSTER,
        ),
        TaskOutputLimits(max_bytes=max_bytes, max_entries=100),
        Deadline.from_seconds(30),
        threading.Event(),
    )


def test_capture_task_outputs_writes_deterministic_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.txt").write_text("second")
    (source / "nested").mkdir()
    (source / "nested" / "a.txt").write_text("first")
    (source / "link").symlink_to("nested/a.txt")

    first_path = tmp_path / "first.tar.zst"
    second_path = tmp_path / "second.tar.zst"
    first = _capture(source, first_path)
    second = _capture(source, second_path)

    assert first.state == job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_UPLOADED
    assert first.sha256 == second.sha256
    assert first_path.read_bytes() == second_path.read_bytes()

    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(first_path.read_bytes())) as reader:
        body = reader.read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:") as archive:
        assert archive.getnames() == ["b.txt", "link", "nested", "nested/a.txt"]
        assert archive.extractfile("nested/a.txt").read() == b"first"
        assert archive.getmember("link").linkname == "nested/a.txt"


def test_capture_task_outputs_skips_special_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "events")
    (source / "profile.heap").write_bytes(b"profile")

    result = _capture(source, tmp_path / "outputs.tar.zst")

    assert result.state == job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_UPLOADED
    assert result.skipped_count == 1
    assert [(entry.path, entry.reason) for entry in result.skipped_sample] == [("events", "special_file")]


def test_capture_task_outputs_preserves_no_partial_object_on_limit_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "large.bin").write_bytes(b"12345")
    destination = tmp_path / "outputs.tar.zst"

    result = _capture(source, destination, max_bytes=4)

    assert result.state == job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_FAILED
    assert result.error.startswith("too_large:")
    assert not destination.exists()


def test_capture_task_outputs_empty_tree_writes_no_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "outputs.tar.zst"

    result = _capture(source, destination)

    assert result.state == job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_EMPTY
    assert not destination.exists()


def test_resolve_task_output_destination_reports_effective_lifecycle_ttl(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "iris.cluster.runtime.output_capture.marin_temp_bucket",
        lambda *args, **kwargs: "gs://marin-us-east1/tmp/ttl=14d/iris/task-outputs",
    )

    result = resolve_task_output_destination(
        TaskOutputPolicy(ttl_days=10),
        JobName.from_wire("/user/job/0"),
        AttemptUid("0123456789abcdef"),
        local_root=tmp_path,
        source_prefix="gs://marin-us-east1/marin",
    )

    assert result.ttl_days == 14
    assert str(result.path).startswith("gs://marin-us-east1/tmp/ttl=14d/")


def test_resolve_task_output_destination_rejects_unmanaged_temp_prefix(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "iris.cluster.runtime.output_capture.marin_temp_bucket",
        lambda *args, **kwargs: "file:///tmp/marin/tmp/iris/task-outputs",
    )

    with pytest.raises(ValueError, match="no lifecycle TTL prefix"):
        resolve_task_output_destination(
            TaskOutputPolicy(),
            JobName.from_wire("/user/job/0"),
            AttemptUid("0123456789abcdef"),
            local_root=tmp_path,
            source_prefix=None,
        )
