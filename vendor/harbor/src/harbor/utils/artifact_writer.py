"""Bounded background writer for durable rollout artifacts.

Rollout hot paths (the per-turn agent loop, trial finalization, exception
recording) must never perform synchronous object-store writes on the shared
coordinator event loop, and must not push them onto asyncio's DEFAULT
``to_thread`` executor either: a live py-spy on production RL rollouts caught
coordinators with 12-30 default-executor threads blocked in s3fs uploads plus
CPU-bound trajectory JSON formatting, starving every other ``to_thread`` user
and collapsing per-step throughput ~6.7x.

This module provides a per-process singleton :class:`ArtifactWriter` with:

- a bounded queue (``HARBOR_ARTIFACT_WRITER_QUEUE_SIZE``, default 256) sharded
  over a small dedicated worker pool (``HARBOR_ARTIFACT_WRITER_THREADS``,
  default 4) that performs BOTH the serialization (the payload callable) and
  the storage write off the caller thread;
- per-destination FIFO within each durability lane: items are sharded by
  destination path, while required writes may overtake optional diagnostics;
- non-blocking optional admission: ``submit_optional`` abandons diagnostic
  artifacts immediately when their dedicated lane is full, while
  ``submit_required`` asynchronously admits reward-bearing records through a
  reserved lane without blocking the caller's event loop;
- a hard deadline on cloud uploads (``upload_timeout``, default 300s, env
  ``HARBOR_ARTIFACT_WRITER_UPLOAD_TIMEOUT``): a stalled object-store commit
  fails (spool copy retained) instead of parking a worker forever, so a slow
  destination degrades only itself and cannot deadlock the coordinator;
- a local spool for cloud destinations: the serialized bytes hit local disk
  before the cloud upload, and the spool copy is retained (and its path
  logged) if the upload fails;
- bounded single-write waits and synchronous ``flush`` for process shutdown or
  diagnostics, plus queryable failure and optional-drop counts.

Writes go through fsspec (``UPath.open``) uniformly, so local, ``memory://``,
and ``s3://``/``gs://`` destinations all work.

The writer owns only artifacts submitted through this API. Direct filesystem
access remains the caller's responsibility. Entry points must establish the
artifact root's mount authority before either path is used; draining this
writer is not an authority boundary.

Flush scopes deliberately compare submitted destination strings. They do not
resolve symlinks or mount aliases: a caller that submits through a lexical
root must flush through that same lexical root.
"""

import asyncio
import atexit
import concurrent.futures
import logging
import multiprocessing
import os
import pickle
import queue
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing.connection import Connection
from pathlib import Path

from upath import UPath

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256
DEFAULT_NUM_WORKERS = 4
DEFAULT_FLUSH_TIMEOUT = 600.0
# A single cloud upload may stall this long before being failed (spool copy
# retained). s3fs has no generic per-call deadline — the PUT happens on file
# close — so without this a slow commit parks a worker forever (the deadlock
# in MarinSkyRL #194).
DEFAULT_UPLOAD_TIMEOUT = 300.0
# Hard ceiling on asynchronous required-write admission under storage
# backpressure. Optional writes never wait for queue capacity.
DEFAULT_MAX_ENQUEUE_WAIT = 60.0
REQUIRED_ADMISSION_POLL_INTERVAL = 0.05
ATEXIT_DRAIN_TIMEOUT = 60.0
UPLOAD_PROCESS_STOP_TIMEOUT = 1.0
CLOUD_UPLOAD_CHUNK_SIZE = 1024 * 1024

QUEUE_SIZE_ENV = "HARBOR_ARTIFACT_WRITER_QUEUE_SIZE"
NUM_WORKERS_ENV = "HARBOR_ARTIFACT_WRITER_THREADS"
UPLOAD_TIMEOUT_ENV = "HARBOR_ARTIFACT_WRITER_UPLOAD_TIMEOUT"
MAX_ENQUEUE_WAIT_ENV = "HARBOR_ARTIFACT_WRITER_MAX_ENQUEUE_WAIT"

# Protocols where a direct write is already fast and durable-enough that no
# local spool copy is needed. Everything else is treated as a cloud target.
_LOCAL_PROTOCOLS = ("", "file", "local", "memory")


def _is_cloud_destination(destination: Path | UPath) -> bool:
    return str(getattr(destination, "protocol", "") or "") not in _LOCAL_PROTOCOLS


class ArtifactPersistenceError(TimeoutError):
    """Base class for failures in Harbor's artifact durability lane."""


class ArtifactWriterBacklogError(ArtifactPersistenceError):
    """A write could not enter its durability lane.

    The shard queue stayed full (the storage backend is not keeping up), so
    Required admission raises after ``max_enqueue_wait``. Optional admission
    abandons immediately. Both outcomes are counted and logged.
    """


class ArtifactUploadTimeoutError(ArtifactPersistenceError):
    """A cloud upload exceeded its hard deadline after its helper was stopped."""

    def __init__(
        self,
        *,
        destination: str,
        byte_count: int,
        endpoint: str,
        elapsed: float,
        timeout: float,
        spool_path: Path,
    ) -> None:
        self.destination = destination
        self.byte_count = byte_count
        self.endpoint = endpoint
        self.elapsed = elapsed
        self.timeout = timeout
        self.spool_path = spool_path
        super().__init__(
            f"Artifact upload timed out: destination={destination} "
            f"bytes={byte_count} endpoint={endpoint} elapsed={elapsed:.3f}s "
            f"deadline={timeout:.3f}s retained_spool={spool_path}"
        )


class ArtifactWriteTimeoutError(ArtifactPersistenceError):
    """A required artifact write did not finish within its wait deadline."""


@dataclass
class _WriteItem:
    destination: Path | UPath
    payload: Callable[[], str | bytes]
    description: str
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)


class _CloudUploadOutcome(Enum):
    COMPLETE = auto()
    DEADLINE_EXCEEDED = auto()


def _picklable_upload_exception(exc: BaseException) -> BaseException:
    try:
        pickle.dumps(exc)
    except (AttributeError, TypeError, pickle.PickleError):
        return RuntimeError(f"{type(exc).__name__}: {exc}")
    return exc


def _cloud_upload_process(connection: Connection) -> None:
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        destination, spool_path = request
        try:
            with (
                spool_path.open("rb") as source_file,
                destination.open("wb") as destination_file,
            ):
                while chunk := source_file.read(CLOUD_UPLOAD_CHUNK_SIZE):
                    destination_file.write(chunk)
        except BaseException as exc:
            connection.send(_picklable_upload_exception(exc))
        else:
            connection.send(None)


class _CloudUploader:
    """Reusable process boundary for one artifact-writer shard."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._connection: Connection | None = None
        self._process: multiprocessing.Process | None = None

    def upload(
        self,
        destination: Path | UPath,
        spool_path: Path,
        timeout: float,
    ) -> _CloudUploadOutcome:
        """Upload a spooled artifact or raise when its deadline expires."""
        self._ensure_started()
        if self._connection is None:
            raise RuntimeError("Cloud upload process did not start")
        try:
            self._connection.send((destination, spool_path))
        except BaseException:
            self._stop()
            raise
        if not self._connection.poll(timeout):
            self._stop()
            return _CloudUploadOutcome.DEADLINE_EXCEEDED
        try:
            error = self._connection.recv()
        except EOFError as exc:
            self._stop()
            raise RuntimeError("Cloud upload process exited without a result") from exc
        if error is not None:
            raise error
        return _CloudUploadOutcome.COMPLETE

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop()
        parent_connection, child_connection = self._context.Pipe()
        process = self._context.Process(
            target=_cloud_upload_process,
            args=(child_connection,),
            name="harbor-artifact-upload",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

    def _stop(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is None:
            return
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=UPLOAD_PROCESS_STOP_TIMEOUT)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=UPLOAD_PROCESS_STOP_TIMEOUT)
        if self._process.is_alive():
            raise RuntimeError("Cloud upload process did not stop")
        self._process.close()
        self._process = None


class ArtifactWriter:
    """Bounded background writer with required and optional durability lanes.

    ``num_workers`` single-threaded shards each own a bounded queue; an item is
    routed to a shard by hashing its destination path, so successive writes to
    the same file execute in submission order within one lane while distinct
    files still write in parallel. Required writes take priority over optional
    writes assigned to the same shard.
    """

    def __init__(
        self,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        spool_dir: Path | None = None,
        upload_timeout: float = DEFAULT_UPLOAD_TIMEOUT,
        max_enqueue_wait: float = DEFAULT_MAX_ENQUEUE_WAIT,
    ):
        if queue_size < 1 or num_workers < 1:
            raise ValueError(
                f"queue_size and num_workers must be >= 1, "
                f"got {queue_size=} {num_workers=}"
            )
        self._upload_timeout = upload_timeout
        self._max_enqueue_wait = max_enqueue_wait
        self._spool_dir = spool_dir or (
            Path(tempfile.gettempdir()) / f"harbor-artifact-spool-{os.getpid()}"
        )
        per_shard = max(1, queue_size // num_workers)
        self._queues: list[queue.Queue[_WriteItem]] = [
            queue.Queue(maxsize=per_shard) for _ in range(num_workers)
        ]
        self._optional_queues: list[queue.Queue[_WriteItem]] = [
            queue.Queue(maxsize=per_shard) for _ in range(num_workers)
        ]
        self._wake_events = [threading.Event() for _ in range(num_workers)]
        self._uploaders = [_CloudUploader() for _ in range(num_workers)]
        self._lock = threading.Lock()
        self._pending: dict[int, _WriteItem] = {}
        self._failed_count = 0
        self._optional_dropped_count = 0
        # Daemon workers, no explicit shutdown: the writer lives for the whole
        # process (per-process singleton) and the atexit drain flushes queued
        # writes; the threads themselves die with the interpreter.
        self._threads = [
            threading.Thread(
                target=self._worker_loop,
                args=(
                    shard_queue,
                    self._optional_queues[i],
                    self._wake_events[i],
                    self._uploaders[i],
                ),
                name=f"harbor-artifact-writer-{i}",
                daemon=True,
            )
            for i, shard_queue in enumerate(self._queues)
        ]
        for thread in self._threads:
            thread.start()

    @property
    def failed_count(self) -> int:
        """Number of writes that failed since this writer was created."""
        with self._lock:
            return self._failed_count

    @property
    def optional_dropped_count(self) -> int:
        """Number of optional writes rejected because their lane was full."""
        with self._lock:
            return self._optional_dropped_count

    def _shard_index(self, destination: Path | UPath) -> int:
        return hash(str(destination)) % len(self._queues)

    def submit_optional(
        self,
        destination: Path | UPath,
        payload: Callable[[], str | bytes],
        *,
        description: str,
    ) -> concurrent.futures.Future:
        """Try to enqueue a diagnostic write without blocking the caller.

        A full optional lane abandons the new write, resolves the returned
        future with :class:`ArtifactWriterBacklogError`, and records the drop.
        Existing queued writes retain per-destination FIFO ordering.
        """
        item = self._new_item(destination, payload, description)
        shard_index = self._shard_index(destination)
        shard_queue = self._optional_queues[shard_index]
        try:
            shard_queue.put_nowait(item)
        except queue.Full:
            drop_count = self._abandon_optional(item)
            if drop_count & (drop_count - 1) == 0:
                logger.warning(
                    "Artifact writer optional lane full; dropped %d optional "
                    "write(s). latest_destination=%s (%s)",
                    drop_count,
                    destination,
                    description,
                )
        else:
            self._wake_events[shard_index].set()
        return item.future

    async def submit_required(
        self,
        destination: Path | UPath,
        payload: Callable[[], str | bytes],
        *,
        description: str,
    ) -> concurrent.futures.Future:
        """Admit a required write without blocking the asyncio event loop."""
        item = self._new_item(destination, payload, description)
        shard_index = self._shard_index(destination)
        shard_queue = self._queues[shard_index]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_enqueue_wait
        warned = False
        try:
            while True:
                try:
                    shard_queue.put_nowait(item)
                    self._wake_events[shard_index].set()
                    return item.future
                except queue.Full:
                    if not warned:
                        warned = True
                        logger.warning(
                            "Artifact writer required lane full; waiting up to "
                            "%.0fs without blocking the event loop. destination=%s (%s)",
                            self._max_enqueue_wait,
                            destination,
                            description,
                        )
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        self._abandon(item)
                        raise ArtifactWriterBacklogError(
                            f"Artifact writer backlog for {destination} "
                            f"({description}) did not clear after "
                            f"{self._max_enqueue_wait:.0f}s; storage backend is "
                            "not keeping up. Required write NOT enqueued."
                        )
                    await asyncio.sleep(
                        min(REQUIRED_ADMISSION_POLL_INTERVAL, remaining)
                    )
        except asyncio.CancelledError:
            with self._lock:
                self._pending.pop(id(item), None)
            item.future.cancel()
            raise

    async def await_write(
        self,
        future: concurrent.futures.Future,
        *,
        timeout: float | None = DEFAULT_FLUSH_TIMEOUT,
    ) -> None:
        """Await one admitted write without cancelling its worker on timeout."""
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(future)),
                timeout,
            )
        except TimeoutError:
            if future.done():
                error = future.exception()
                if error is None:
                    return
                raise error
            raise ArtifactWriteTimeoutError(
                f"Artifact writer required write timed out after {timeout}s"
            ) from None

    def _new_item(
        self,
        destination: Path | UPath,
        payload: Callable[[], str | bytes],
        description: str,
    ) -> _WriteItem:
        item = _WriteItem(
            destination=destination,
            payload=payload,
            description=description,
        )
        with self._lock:
            self._pending[id(item)] = item
        return item

    def flush(
        self, prefix: str | None = None, timeout: float | None = DEFAULT_FLUSH_TIMEOUT
    ) -> int:
        """Wait until all currently-pending writes (under ``prefix``) land.

        Returns the number of matched writes that FAILED (already logged).
        Raises ``TimeoutError`` if writes are still pending after ``timeout``.
        """
        futures = self._matching_futures(prefix)
        if not futures:
            return 0
        done, not_done = concurrent.futures.wait(futures, timeout=timeout)
        if not_done:
            raise TimeoutError(
                f"Artifact writer flush timed out after {timeout}s with "
                f"{len(not_done)} write(s) still pending"
                + (f" under {prefix}" if prefix else "")
            )
        return self._count_and_log_failures(done, prefix)

    def _matching_futures(self, prefix: str | None) -> list[concurrent.futures.Future]:
        if prefix is None:
            with self._lock:
                return [item.future for item in self._pending.values()]
        # Match the prefix as a whole path component ("a/trial-1" must not
        # match "a/trial-10/result.json").
        dir_prefix = prefix.rstrip("/") + "/"
        with self._lock:
            return [
                item.future
                for item in self._pending.values()
                if str(item.destination) == prefix
                or str(item.destination).startswith(dir_prefix)
            ]

    def _count_and_log_failures(self, futures, prefix: str | None) -> int:
        failed = sum(1 for f in futures if f.exception() is not None)
        if failed:
            logger.error(
                "Artifact writer flush: %d write(s) failed%s (see earlier "
                "'Artifact write failed' errors for details)",
                failed,
                f" under {prefix}" if prefix else "",
            )
        return failed

    def _abandon(self, item: _WriteItem) -> None:
        """Drop an item that submit() could never enqueue.

        Pops it from the pending set (so flush stops waiting on it), counts it
        as failed, and resolves its future with the backlog error — any
        concurrent flush that already snapshotted the future sees a terminal
        state instead of hanging.
        """
        with self._lock:
            self._pending.pop(id(item), None)
            self._failed_count += 1
        item.future.set_exception(
            ArtifactWriterBacklogError(
                f"Artifact write to {item.destination} ({item.description}) "
                "abandoned: writer backlog did not clear"
            )
        )

    def _abandon_optional(self, item: _WriteItem) -> int:
        self._abandon(item)
        with self._lock:
            self._optional_dropped_count += 1
            return self._optional_dropped_count

    def _worker_loop(
        self,
        required_queue: "queue.Queue[_WriteItem]",
        optional_queue: "queue.Queue[_WriteItem]",
        wake_event: threading.Event,
        uploader: _CloudUploader,
    ) -> None:
        while True:
            wake_event.clear()
            try:
                item = required_queue.get_nowait()
                source_queue = required_queue
            except queue.Empty:
                try:
                    item = optional_queue.get_nowait()
                    source_queue = optional_queue
                except queue.Empty:
                    wake_event.wait()
                    continue
            try:
                self._write_item(item, uploader)
            except BaseException as exc:
                with self._lock:
                    self._failed_count += 1
                logger.error(
                    "Artifact write failed (%s -> %s): %s",
                    item.description,
                    item.destination,
                    exc,
                    exc_info=True,
                )
                if not item.future.cancelled():
                    item.future.set_exception(exc)
            else:
                if not item.future.cancelled():
                    item.future.set_result(None)
            finally:
                with self._lock:
                    self._pending.pop(id(item), None)
                source_queue.task_done()

    def _write_item(self, item: _WriteItem, uploader: _CloudUploader) -> None:
        data = item.payload()
        if isinstance(data, str):
            data = data.encode("utf-8")
        if _is_cloud_destination(item.destination):
            self._write_cloud(item, data, uploader)
            return
        # Local-ish target: direct fsspec write from the worker is fine.
        parent = item.destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        with item.destination.open("wb") as f:
            f.write(data)

    def _write_cloud(
        self, item: _WriteItem, data: bytes, uploader: _CloudUploader
    ) -> None:
        """Spool locally first, then upload; keep the spool copy on failure."""
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        spool_path = self._spool_dir / (
            f"{uuid.uuid4().hex}-{Path(str(item.destination)).name}"
        )
        spool_path.write_bytes(data)
        try:
            self._upload_with_timeout(item.destination, data, spool_path, uploader)
        except BaseException:
            logger.error(
                "Artifact upload to %s failed; spool copy retained at %s",
                item.destination,
                spool_path,
            )
            raise
        spool_path.unlink(missing_ok=True)

    def _upload_with_timeout(
        self,
        destination: Path | UPath,
        data: bytes,
        spool_path: Path,
        uploader: _CloudUploader,
    ) -> None:
        """Upload one cloud artifact without exceeding the hard deadline."""
        started_at = time.monotonic()
        outcome = uploader.upload(destination, spool_path, self._upload_timeout)
        if outcome is _CloudUploadOutcome.DEADLINE_EXCEEDED:
            elapsed = time.monotonic() - started_at
            storage_options = (
                destination.storage_options if isinstance(destination, UPath) else {}
            )
            client_kwargs = storage_options.get("client_kwargs", {})
            endpoint = client_kwargs.get("endpoint_url") or storage_options.get(
                "endpoint_url"
            )
            raise ArtifactUploadTimeoutError(
                destination=str(destination),
                byte_count=len(data),
                endpoint=str(endpoint or "provider-default"),
                elapsed=elapsed,
                timeout=self._upload_timeout,
                spool_path=spool_path,
            ) from None


_WRITER: ArtifactWriter | None = None
_WRITER_LOCK = threading.Lock()


def get_artifact_writer() -> ArtifactWriter:
    """Per-process singleton, configured from the environment on first use."""
    global _WRITER
    if _WRITER is None:
        with _WRITER_LOCK:
            if _WRITER is None:
                _WRITER = ArtifactWriter(
                    queue_size=int(os.environ.get(QUEUE_SIZE_ENV, DEFAULT_QUEUE_SIZE)),
                    num_workers=int(
                        os.environ.get(NUM_WORKERS_ENV, DEFAULT_NUM_WORKERS)
                    ),
                    upload_timeout=float(
                        os.environ.get(UPLOAD_TIMEOUT_ENV, DEFAULT_UPLOAD_TIMEOUT)
                    ),
                    max_enqueue_wait=float(
                        os.environ.get(MAX_ENQUEUE_WAIT_ENV, DEFAULT_MAX_ENQUEUE_WAIT)
                    ),
                )
                atexit.register(_drain_at_exit, _WRITER)
    return _WRITER


def _drain_at_exit(writer: ArtifactWriter) -> None:
    """Bounded best-effort drain so queued artifacts survive process exit."""
    try:
        writer.flush(timeout=ATEXIT_DRAIN_TIMEOUT)
    except TimeoutError as exc:
        logger.error("Artifact writer drain at process exit incomplete: %s", exc)
