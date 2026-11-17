"""Behavior tests for the bounded background artifact writer.

The writer is the hot-path persistence backbone: rollout code enqueues durable
artifact writes (trajectory dumps, result/exception files, episode logs) and a
small dedicated pool performs both the serialization and the storage write.
These tests use real local filesystems (tmp_path, fsspec ``memory://``) — not
behavior-encoding mocks — and pin the contract: required backpressure is
bounded, optional overflow is counted, per-destination ordering holds, flush
is a durability barrier, errors are surfaced, and cloud destinations get a
local spool copy that survives a failed upload.
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import uuid
from pathlib import Path

import pytest
from upath import UPath

from harbor.utils.artifact_writer import (
    CLOUD_UPLOAD_CHUNK_SIZE,
    ArtifactWriteTimeoutError,
    ArtifactUploadTimeoutError,
    ArtifactWriter,
    ArtifactWriterBacklogError,
    get_artifact_writer,
)


class _CloudUploadThatRetainsANonDaemonThread:
    protocol = "s3"

    def __str__(self) -> str:
        return "s3://bucket/job/result.json"

    def open(self, _mode: str):
        threading.Thread(target=threading.Event().wait, daemon=False).start()
        threading.Event().wait()


class _CloudDestination:
    protocol = "s3"

    def __init__(self, real_path: Path, spool_dir: Path, observed_spool: Path):
        self._real_path = real_path
        self._spool_dir = spool_dir
        self._observed_spool = observed_spool

    def __str__(self) -> str:
        return f"s3://bucket/trial/{self._real_path.name}"

    def open(self, mode: str):
        spool_files = list(self._spool_dir.iterdir())
        self._observed_spool.write_bytes(spool_files[0].read_bytes())
        return self._real_path.open(mode)


class _ChunkLimitedWriter:
    def __init__(self, path: Path, mode: str, max_chunk_size: int):
        self._file = path.open(mode)
        self._max_chunk_size = max_chunk_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._file.__exit__(exc_type, exc_value, traceback)

    def write(self, data: bytes) -> int:
        if len(data) > self._max_chunk_size:
            raise ValueError(f"write exceeded {self._max_chunk_size} bytes")
        return self._file.write(data)


class _ChunkLimitedCloudDestination:
    protocol = "s3"

    def __init__(self, real_path: Path, max_chunk_size: int):
        self._real_path = real_path
        self._max_chunk_size = max_chunk_size

    def __str__(self) -> str:
        return f"s3://bucket/trial/{self._real_path.name}"

    def open(self, mode: str):
        return _ChunkLimitedWriter(self._real_path, mode, self._max_chunk_size)


class _FailingCloudDestination:
    protocol = "s3"

    def __str__(self) -> str:
        return "s3://bucket/trial/trajectory.json"

    def open(self, _mode: str):
        raise OSError("503 SlowDown")


class _StalledCloudDestination:
    protocol = "s3"
    storage_options = {"client_kwargs": {"endpoint_url": "https://objects.example"}}

    def __str__(self) -> str:
        return "s3://bucket/stuck/trajectory.json"

    def open(self, _mode: str):
        threading.Event().wait()


def _run_stalled_cloud_upload(spool_dir: str) -> None:
    writer = ArtifactWriter(
        num_workers=1,
        spool_dir=Path(spool_dir),
        upload_timeout=0.1,
    )
    future = writer.submit_optional(
        _CloudUploadThatRetainsANonDaemonThread(),
        lambda: "job result",
        description="job result",
    )

    assert writer.flush(timeout=2) == 1
    assert isinstance(future.exception(), ArtifactUploadTimeoutError)


@pytest.mark.unit
def test_write_lands_on_local_filesystem_and_creates_parent(tmp_path) -> None:
    writer = ArtifactWriter(num_workers=2)
    destination = tmp_path / "nested" / "dir" / "artifact.json"

    writer.submit_optional(
        destination, lambda: '{"ok": 1}', description="test artifact"
    )
    assert writer.flush() == 0

    assert destination.read_text() == '{"ok": 1}'


@pytest.mark.unit
def test_serialization_and_write_run_off_the_caller_thread(tmp_path) -> None:
    writer = ArtifactWriter()
    caller_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def payload() -> str:
        seen["thread"] = threading.get_ident()
        return "data"

    writer.submit_optional(tmp_path / "f.txt", payload, description="thread probe")
    writer.flush()

    assert seen["thread"] != caller_thread, "payload ran ON the caller thread"


@pytest.mark.unit
def test_same_destination_writes_apply_in_submission_order(tmp_path) -> None:
    """Per-destination FIFO: with many workers, the LAST submitted content for
    a given file must win — out-of-order writes would resurrect stale
    trajectory snapshots."""
    writer = ArtifactWriter(num_workers=4)
    destination = tmp_path / "trajectory.json"

    for i in range(50):
        writer.submit_optional(
            destination, lambda i=i: f"version-{i}", description="ordering"
        )
    writer.flush()

    assert destination.read_text() == "version-49"


@pytest.mark.unit
def test_flush_timeout_raises(tmp_path) -> None:
    gate = threading.Event()
    writer = ArtifactWriter(num_workers=1)

    def slow_payload() -> str:
        gate.wait(timeout=30)
        return "done"

    writer.submit_optional(tmp_path / "slow.txt", slow_payload, description="slow")
    with pytest.raises(TimeoutError):
        writer.flush(timeout=0.05)
    gate.set()
    writer.flush()
    assert (tmp_path / "slow.txt").read_text() == "done"


@pytest.mark.unit
def test_flush_prefix_only_waits_for_matching_destinations(tmp_path) -> None:
    """A scoped flush must not hang on another scope's stuck writes."""
    gate = threading.Event()
    # Two workers so trial-b's stuck write cannot head-of-line block trial-a.
    writer = ArtifactWriter(num_workers=2)
    trial_a = tmp_path / "trial-a"
    trial_b = tmp_path / "trial-b"

    def stuck() -> str:
        gate.wait(timeout=30)
        return "b"

    # Find destinations that shard onto DIFFERENT workers so the stuck write
    # cannot serialize in front of trial-a's write.
    b_dest = trial_b / "result.json"
    b_shard = hash(str(b_dest)) % 2
    a_dest = next(
        trial_a / f"result-{i}.json"
        for i in range(64)
        if hash(str(trial_a / f"result-{i}.json")) % 2 != b_shard
    )
    writer.submit_optional(b_dest, stuck, description="stuck other trial")
    writer.submit_optional(a_dest, lambda: "a", description="this trial")

    assert writer.flush(prefix=str(trial_a), timeout=10) == 0
    assert a_dest.read_text() == "a"

    gate.set()
    writer.flush()


@pytest.mark.unit
def test_flush_prefix_preserves_symlinked_root_identity(tmp_path) -> None:
    """A scoped flush matches the lexical root used when the write was submitted."""
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    lexical_root = tmp_path / "mounted-root"
    lexical_root.symlink_to(real_root, target_is_directory=True)

    payload_started = threading.Event()
    release_payload = threading.Event()
    flush_finished = threading.Event()
    writer = ArtifactWriter(num_workers=1)

    def pending_payload() -> str:
        payload_started.set()
        release_payload.wait(timeout=30)
        return "result"

    destination = lexical_root / "trial-1" / "result.json"
    writer.submit_optional(destination, pending_payload, description="trial result")
    assert payload_started.wait(timeout=5)

    flush_error: list[BaseException] = []

    def flush_lexical_scope() -> None:
        try:
            writer.flush(prefix=str(lexical_root / "trial-1"), timeout=5)
        except BaseException as exc:
            flush_error.append(exc)
        finally:
            flush_finished.set()

    flush_thread = threading.Thread(target=flush_lexical_scope, daemon=True)
    flush_thread.start()
    assert not flush_finished.wait(timeout=0.05)

    release_payload.set()
    assert flush_finished.wait(timeout=5)
    flush_thread.join(timeout=5)

    assert flush_error == []
    assert destination.read_text() == "result"


@pytest.mark.unit
def test_failed_write_surfaces_future_exception_and_count(tmp_path) -> None:
    writer = ArtifactWriter()

    def boom() -> str:
        raise ValueError("cannot serialize")

    future = writer.submit_optional(tmp_path / "x.json", boom, description="failing")
    writer.flush()

    assert writer.failed_count == 1
    assert isinstance(future.exception(), ValueError)


@pytest.mark.unit
def test_memory_upath_destination_writes_through_fsspec() -> None:
    """fsspec-scheme destinations work end to end (real memory:// backend)."""
    writer = ArtifactWriter()
    destination = UPath(f"memory://aw-{uuid.uuid4().hex}/agent/trajectory.json")

    writer.submit_optional(
        destination, lambda: '{"steps": []}', description="remote write"
    )
    writer.flush()

    assert destination.read_text() == '{"steps": []}'


@pytest.mark.unit
def test_cloud_destination_spools_locally_and_cleans_up_on_success(tmp_path) -> None:
    """Cloud targets get a local spool copy BEFORE the upload; the spool file
    is removed once the upload lands."""
    spool_dir = tmp_path / "spool"
    writer = ArtifactWriter(num_workers=1, spool_dir=spool_dir)
    observed_spool = tmp_path / "observed-spool"
    destination = _CloudDestination(
        tmp_path / "uploaded.json", spool_dir, observed_spool
    )
    second_observed_spool = tmp_path / "second-observed-spool"
    second_destination = _CloudDestination(
        tmp_path / "second-uploaded.json", spool_dir, second_observed_spool
    )

    writer.submit_optional(destination, lambda: "payload", description="cloud write")
    writer.submit_optional(
        second_destination,
        lambda: "second payload",
        description="second cloud write",
    )
    assert writer.flush() == 0

    assert (tmp_path / "uploaded.json").read_text() == "payload"
    assert observed_spool.read_bytes() == b"payload"
    assert (tmp_path / "second-uploaded.json").read_text() == "second payload"
    assert second_observed_spool.read_bytes() == b"second payload"
    assert list(spool_dir.iterdir()) == [], "spool copy not cleaned up after upload"


@pytest.mark.unit
def test_cloud_destination_uploads_in_bounded_chunks(tmp_path) -> None:
    payload = b"x" * (2 * CLOUD_UPLOAD_CHUNK_SIZE + 17)
    destination_path = tmp_path / "uploaded.bin"
    destination = _ChunkLimitedCloudDestination(
        destination_path, CLOUD_UPLOAD_CHUNK_SIZE
    )
    writer = ArtifactWriter(num_workers=1, spool_dir=tmp_path / "spool")

    future = writer.submit_optional(
        destination, lambda: payload, description="cloud write"
    )

    assert writer.flush() == 0
    assert future.exception() is None
    assert destination_path.read_bytes() == payload


@pytest.mark.unit
def test_cloud_destination_failure_retains_spool_copy(tmp_path) -> None:
    """If the cloud upload fails, the serialized bytes survive in the spool."""
    spool_dir = tmp_path / "spool"
    writer = ArtifactWriter(spool_dir=spool_dir)

    future = writer.submit_optional(
        _FailingCloudDestination(),
        lambda: "precious trajectory",
        description="upload",
    )
    writer.flush()

    assert writer.failed_count == 1
    assert isinstance(future.exception(), OSError)
    spool_files = list(spool_dir.iterdir())
    assert len(spool_files) == 1
    assert spool_files[0].read_text() == "precious trajectory"


@pytest.mark.unit
def test_stalled_cloud_upload_is_bounded_and_frees_its_worker(tmp_path) -> None:
    """Property #1 of the MarinSkyRL #194 deadlock: _write_cloud did an
    UNTIMED destination.open("wb"), so a stalled object-store commit parked the
    worker forever. With upload_timeout, the stalled upload must FAIL (spool
    retained, failed_count incremented) and free the worker to drain the rest
    of its shard queue — here, a follow-up write sharing the single worker."""
    spool_dir = tmp_path / "spool"
    # One worker: the follow-up write shares the stalled worker's shard, so it
    # can only land once the stalled upload has been abandoned.
    writer = ArtifactWriter(num_workers=1, spool_dir=spool_dir, upload_timeout=0.2)
    destination = _StalledCloudDestination()

    stalled = writer.submit_optional(
        destination, lambda: "precious trajectory", description="stalled upload"
    )
    followup = writer.submit_optional(
        tmp_path / "followup.json", lambda: "ok", description="follow-up write"
    )

    # flush must return inside the bound: the stalled upload times out and the
    # worker moves on to the follow-up write.
    failed = writer.flush(timeout=2.0)
    assert failed >= 1
    assert writer.failed_count >= 1
    error = stalled.exception()
    assert isinstance(error, ArtifactUploadTimeoutError)
    assert error.destination == "s3://bucket/stuck/trajectory.json"
    assert error.byte_count == len(b"precious trajectory")
    assert error.endpoint == "provider-default"
    assert error.elapsed >= 0.2
    assert followup.exception() is None
    assert (tmp_path / "followup.json").read_text() == "ok"
    # The spool copy survives the timed-out upload (data is not lost).
    spool_files = list(spool_dir.iterdir())
    assert len(spool_files) == 1
    assert spool_files[0].read_text() == "precious trajectory"
    assert error.spool_path == spool_files[0]


@pytest.mark.unit
def test_stalled_cloud_upload_does_not_prevent_process_exit(tmp_path) -> None:
    process = multiprocessing.get_context("spawn").Process(
        target=_run_stalled_cloud_upload,
        args=(str(tmp_path / "spool"),),
    )
    process.start()
    process.join(timeout=5)

    exited = not process.is_alive()
    if not exited:
        process.kill()
        process.join(timeout=5)

    assert exited, "timed-out cloud upload kept the Harbor process alive"
    assert process.exitcode == 0


@pytest.mark.unit
def test_stalled_upload_degrades_only_its_destination(tmp_path) -> None:
    """The 'done when' from #194: a stalled commit degrades ONLY its own
    destination. Writes to a destination sharded onto a different worker must
    land promptly while the stalled upload is still wedged."""
    writer = ArtifactWriter(num_workers=4, upload_timeout=0.2)

    # Route the follow-up onto a DIFFERENT shard than the stalled cloud write.
    stalled_shard = hash(str(_StalledCloudDestination())) % 4
    other = next(
        tmp_path / f"other-{i}.json"
        for i in range(200)
        if hash(str(tmp_path / f"other-{i}.json")) % 4 != stalled_shard
    )

    writer.submit_optional(
        _StalledCloudDestination(), lambda: "stuck", description="stalled"
    )
    writer.submit_optional(other, lambda: "unaffected", description="other dest")

    # Lands without releasing the stall.
    assert writer.flush(prefix=str(other), timeout=2.0) == 0
    assert other.read_text() == "unaffected"

    assert writer.flush() == 1


@pytest.mark.unit
def test_optional_lane_drops_new_write_without_blocking(tmp_path) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    writer = ArtifactWriter(queue_size=1, num_workers=1)

    def stalled_optional() -> str:
        worker_started.set()
        release_worker.wait(timeout=30)
        return "first"

    writer.submit_optional(
        tmp_path / "first.txt", stalled_optional, description="stalled optional"
    )
    assert worker_started.wait(timeout=5)
    writer.submit_optional(
        tmp_path / "queued.txt", lambda: "second", description="queued optional"
    )

    dropped = writer.submit_optional(
        tmp_path / "dropped.txt", lambda: "third", description="dropped optional"
    )

    assert isinstance(dropped.exception(), ArtifactWriterBacklogError)
    assert writer.optional_dropped_count == 1
    assert not (tmp_path / "dropped.txt").exists()

    release_worker.set()
    assert writer.flush(timeout=5) == 0
    assert (tmp_path / "first.txt").read_text() == "first"
    assert (tmp_path / "queued.txt").read_text() == "second"


@pytest.mark.unit
async def test_required_admission_yields_while_required_lane_is_full(tmp_path) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    writer = ArtifactWriter(
        queue_size=1,
        num_workers=1,
        max_enqueue_wait=1,
    )

    def stalled_required() -> str:
        worker_started.set()
        release_worker.wait(timeout=30)
        return "first"

    first = await writer.submit_required(
        tmp_path / "first.txt", stalled_required, description="stalled required"
    )
    assert worker_started.wait(timeout=5)
    second = await writer.submit_required(
        tmp_path / "second.txt", lambda: "second", description="queued required"
    )

    admission = asyncio.create_task(
        writer.submit_required(
            tmp_path / "third.txt", lambda: "third", description="async required"
        )
    )
    await asyncio.sleep(0)
    assert not admission.done()

    # This coroutine can still run while required admission waits for capacity.
    release_worker.set()
    third = await asyncio.wait_for(admission, timeout=2)
    await asyncio.gather(
        asyncio.wrap_future(first),
        asyncio.wrap_future(second),
        asyncio.wrap_future(third),
    )

    assert (tmp_path / "third.txt").read_text() == "third"


@pytest.mark.unit
async def test_required_admission_raises_after_bounded_wait(tmp_path) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    writer = ArtifactWriter(queue_size=1, num_workers=1, max_enqueue_wait=0.05)

    def stalled_required() -> str:
        worker_started.set()
        release_worker.wait(timeout=30)
        return "first"

    await writer.submit_required(
        tmp_path / "first.txt", stalled_required, description="stalled required"
    )
    assert worker_started.wait(timeout=5)
    await writer.submit_required(
        tmp_path / "second.txt", lambda: "second", description="queued required"
    )

    with pytest.raises(ArtifactWriterBacklogError):
        await writer.submit_required(
            tmp_path / "third.txt", lambda: "third", description="overflow required"
        )

    release_worker.set()
    writer.flush(timeout=5)


@pytest.mark.unit
async def test_required_wait_timeout_does_not_cancel_worker_write(tmp_path) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    writer = ArtifactWriter(num_workers=1)

    def stalled_required() -> str:
        worker_started.set()
        release_worker.wait(timeout=30)
        return "result"

    future = await writer.submit_required(
        tmp_path / "result.json",
        stalled_required,
        description="required result",
    )
    assert worker_started.wait(timeout=5)

    with pytest.raises(ArtifactWriteTimeoutError):
        await writer.await_write(future, timeout=0.01)
    assert not future.cancelled()

    release_worker.set()
    await writer.await_write(future, timeout=5)
    assert (tmp_path / "result.json").read_text() == "result"


@pytest.mark.unit
async def test_required_wait_preserves_completed_upload_timeout(tmp_path) -> None:
    writer = ArtifactWriter(num_workers=1)
    upload_error = ArtifactUploadTimeoutError(
        destination="s3://bucket/result.json",
        byte_count=1,
        endpoint="https://objects.example",
        elapsed=1.0,
        timeout=1.0,
        spool_path=tmp_path / "spool.json",
    )
    future: concurrent.futures.Future[None] = concurrent.futures.Future()
    future.set_exception(upload_error)

    with pytest.raises(ArtifactUploadTimeoutError) as exc_info:
        await writer.await_write(future, timeout=5)

    assert exc_info.value is upload_error


@pytest.mark.unit
def test_get_artifact_writer_is_a_process_singleton() -> None:
    assert get_artifact_writer() is get_artifact_writer()
