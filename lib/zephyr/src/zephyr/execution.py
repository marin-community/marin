# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job-based execution engine for Zephyr pipelines.

The coordinator runs as a fray *job* that internally creates coordinator and
worker *actors* as child jobs. Workers pull tasks from the coordinator actor,
execute shard operations, and report results back. Because actors are children
of the coordinator job, Iris cascading termination automatically cleans them
up when the coordinator exits or is killed — preventing stale-coordinator
bugs where orphaned coordinators and workers consume resources indefinitely.
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
import logging
import os
import pickle
import re
from datetime import datetime, timezone
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

import cloudpickle
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from iris.marin_fs import open_url, url_to_fs
from fray.v2 import ActorConfig, ActorFuture, ActorHandle, Client, ResourceConfig
from fray.v2.client import JobHandle
from fray.v2.types import Entrypoint, JobRequest
from iris.marin_fs import marin_temp_bucket
from iris.time_utils import ExponentialBackoff

from zephyr.dataset import Dataset
from zephyr.plan import (
    ExecutionHint,
    Join,
    PhysicalOp,
    PhysicalPlan,
    Scatter,
    SourceItem,
    StageContext,
    StageResultChunk,
    StageType,
    compute_plan,
    run_stage,
)
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)


class Chunk(Protocol):
    def __iter__(self) -> Iterator: ...


_ZEPHYR_SHUFFLE_SHARD_IDX_COL = "shard_idx"
_ZEPHYR_SHUFFLE_CHUNK_IDX_COL = "chunk_idx"
_ZEPHYR_SHUFFLE_ITEM_COL = "item"
_ZEPHYR_SHUFFLE_PICKLED_COL = "pickled"


@dataclass(frozen=True)
class PickleDiskChunk:
    """Reference to a pickle chunk stored on disk.

    Each write goes to a UUID-unique path to avoid collisions when multiple
    workers race on the same shard.  No coordinator-side rename is needed;
    the winning result's paths are used directly and the entire execution
    directory is cleaned up after the pipeline completes.
    """

    path: str
    count: int

    def __iter__(self) -> Iterator:
        return iter(self.read())

    @classmethod
    def write(cls, path: str, data: list) -> PickleDiskChunk:
        """Write *data* to a UUID-unique path derived from *path*.

        The UUID suffix avoids collisions when multiple workers race on
        the same shard.  The resulting path is used directly for reads —
        no rename step is required.
        """
        from zephyr.writers import unique_temp_path

        ensure_parent_dir(path)
        data = list(data)
        count = len(data)

        unique_path = unique_temp_path(path)
        with open_url(unique_path, "wb") as f:
            pickle.dump(data, f)
        return cls(path=unique_path, count=count)

    def read(self) -> list:
        """Load chunk data from disk."""
        with open_url(self.path, "rb") as f:
            return pickle.load(f)


@dataclass(frozen=True)
class ParquetDiskChunk:
    """Slice of a shared Parquet scatter file, filtered by target shard and chunk.

    Multiple ParquetDiskChunk instances share the same file path but filter
    for different (shard_idx, chunk_idx) pairs. Each chunk is pre-sorted
    by key, preserving the invariant needed for k-way merge in Reduce.

    Items are stored in one of two envelope formats:

    * **Native** (``is_pickled=False``): ``{"shard_idx", "chunk_idx", "item": <data>}``
    * **Pickle** (``is_pickled=True``): ``{"shard_idx", "chunk_idx", "pickled": <bytes>}``

    The pickle envelope is used when items are not Arrow-serializable.
    """

    path: str
    filter_shard: int
    filter_chunk: int
    count: int
    is_pickled: bool = False

    def __iter__(self) -> Iterator:
        return iter(self.read())

    def read(self) -> list:
        """Load filtered chunk data from a Parquet file, unwrapping envelope."""
        col = _ZEPHYR_SHUFFLE_PICKLED_COL if self.is_pickled else _ZEPHYR_SHUFFLE_ITEM_COL
        table = pq.read_table(
            self.path,
            columns=[col],
            filters=(
                (pc.field(_ZEPHYR_SHUFFLE_SHARD_IDX_COL) == self.filter_shard)
                & (pc.field(_ZEPHYR_SHUFFLE_CHUNK_IDX_COL) == self.filter_chunk)
            ),
        )
        items = table.column(col).to_pylist()
        if self.is_pickled:
            return [pickle.loads(b) for b in items]
        return items


@dataclass
class MemChunk:
    """In-memory chunk."""

    items: list[Any]

    def __iter__(self) -> Iterator:
        return iter(self.items)


@dataclass
class Shard:
    """An ordered sequence of chunks assigned to a single worker."""

    chunks: list[Chunk]

    def __iter__(self) -> Iterator:
        """Flatten iteration over all items, loading chunks as needed."""
        for chunk in self.chunks:
            yield from chunk

    def iter_chunks(self) -> Iterator[list]:
        """Yield each chunk as a materialized list. Used by k-way merge in Reduce."""
        for chunk in self.chunks:
            yield list(chunk)


@dataclass
class ResultChunk:
    """Output chunk from a single worker task before resharding."""

    source_shard: int
    target_shard: int
    data: Chunk


@dataclass
class TaskResult:
    """Result of a single task."""

    chunks: list[ResultChunk]


def _generate_execution_id() -> str:
    """Generate unique ID for this execution to avoid conflicts."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _shared_data_path(prefix: str, execution_id: str, name: str) -> str:
    """Path for a shared data object: {prefix}/{execution_id}/shared/{name}.pkl"""
    return f"{prefix}/{execution_id}/shared/{name}.pkl"


def _cleanup_execution(prefix: str, execution_id: str) -> None:
    """Remove all chunk files for an execution."""
    exec_dir = f"{prefix}/{execution_id}"
    fs = url_to_fs(exec_dir)[0]

    # TODO: use log_time util when possible
    t_0 = time.monotonic()
    if fs.exists(exec_dir):
        try:
            fs.rm(exec_dir, recursive=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup chunks at {exec_dir}: {e}")
        finally:
            elapsed = time.monotonic() - t_0
            logger.info(f"Cleaned up execution directory {exec_dir} in {elapsed:.1f}s")


def _make_envelope(items: list, target_shard: int, chunk_idx: int) -> list[dict]:
    return [
        {
            _ZEPHYR_SHUFFLE_SHARD_IDX_COL: target_shard,
            _ZEPHYR_SHUFFLE_CHUNK_IDX_COL: chunk_idx,
            _ZEPHYR_SHUFFLE_ITEM_COL: item,
        }
        for item in items
    ]


def _make_pickle_envelope(items: list, target_shard: int, chunk_idx: int) -> list[dict]:
    """Wrap items as pickle-serialized bytes for Arrow-incompatible types."""
    return [
        {
            _ZEPHYR_SHUFFLE_SHARD_IDX_COL: target_shard,
            _ZEPHYR_SHUFFLE_CHUNK_IDX_COL: chunk_idx,
            _ZEPHYR_SHUFFLE_PICKLED_COL: cloudpickle.dumps(item),
        }
        for item in items
    ]


def _segment_path(base_path: str, seg_idx: int) -> str:
    """Return the file path for a given segment index.

    ``shard-0000.parquet`` → ``shard-0000-seg0000.parquet``
    """
    stem, ext = os.path.splitext(base_path)
    return f"{stem}-seg{seg_idx:04d}{ext}"


class _ChunkMetadata(NamedTuple):
    path: str
    target_shard: int
    chunk_idx: int
    cnt: int


def _write_parquet_scatter(
    stage_gen: Iterator[StageResultChunk],
    source_shard: int,
    parquet_path: str,
    pickled: bool = False,
) -> list[ResultChunk]:
    """Stream scatter chunks into Parquet files as row groups.

    Writes batches to a Parquet file until a schema mismatch is detected
    (e.g. a field evolves from null to a concrete type). On mismatch the
    current file is closed, the schema is unified via ``pa.unify_schemas``,
    and a new segment file is opened with the evolved schema.

    When ``pickled=True``, items are serialized via pickle into a binary
    ``pickled`` column instead of being stored natively in the ``item`` column.
    """
    chunk_results: list[_ChunkMetadata] = []
    per_shard_chunk_cnt: dict[int, int] = defaultdict(int)
    n_chunks_flushed = 0
    seg_idx = 0
    schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    seg_file = ""

    pending_chunk: pa.RecordBatch | None = None
    pending_chunk_metadata: _ChunkMetadata | None = None

    def _flush_pending():
        nonlocal n_chunks_flushed
        if pending_chunk is None:
            return
        writer.write_batch(pending_chunk)
        chunk_results.append(pending_chunk_metadata)
        n_chunks_flushed += 1
        if n_chunks_flushed % 10 == 0:
            logger.info(
                "[shard %d segment %d] Wrote %d parquet chunks so far (latest chunk size: %d items)",
                source_shard,
                seg_idx,
                n_chunks_flushed,
                pending_chunk_metadata.cnt,
            )

    for result in stage_gen:
        chunk_items = list(result.chunk)
        target_shard = result.target_shard
        shard_chunk_idx = per_shard_chunk_cnt[target_shard]
        per_shard_chunk_cnt[target_shard] += 1
        envelope_fn = _make_pickle_envelope if pickled else _make_envelope
        envelope = envelope_fn(chunk_items, target_shard, shard_chunk_idx)
        chunk_arrow = pa.RecordBatch.from_pylist(envelope)

        if schema is None:
            # First batch — initialize writer
            schema = chunk_arrow.schema
            seg_file = _segment_path(parquet_path, seg_idx)
            ensure_parent_dir(seg_file)
            writer = pq.ParquetWriter(seg_file, schema)
        elif chunk_arrow.schema != schema:
            # Schema evolved — flush pending, start new segment
            _flush_pending()
            writer.close()

            schema = pa.unify_schemas([schema, chunk_arrow.schema])
            seg_idx += 1
            seg_file = _segment_path(parquet_path, seg_idx)
            ensure_parent_dir(seg_file)
            writer = pq.ParquetWriter(seg_file, schema)
            logger.info(
                "[shard %d] Schema evolved after %d chunks; starting segment %d",
                source_shard,
                n_chunks_flushed,
                seg_idx,
            )
            chunk_arrow = chunk_arrow.cast(schema)
        else:
            _flush_pending()

        pending_chunk = chunk_arrow
        pending_chunk_metadata = _ChunkMetadata(
            path=seg_file, target_shard=target_shard, chunk_idx=shard_chunk_idx, cnt=len(chunk_items)
        )

    _flush_pending()
    if writer is not None:
        writer.close()

    return [
        ResultChunk(
            source_shard=source_shard,
            target_shard=rec.target_shard,
            data=ParquetDiskChunk(
                path=rec.path,
                filter_shard=rec.target_shard,
                filter_chunk=rec.chunk_idx,
                count=rec.cnt,
                is_pickled=pickled,
            ),
        )
        for rec in chunk_results
    ]


def _write_pickle_chunks(
    stage_gen: Iterator[StageResultChunk],
    source_shard: int,
    chunk_path_fn: Callable[[int], str],
) -> list[ResultChunk]:
    """Write stage output chunks as pickle files."""
    results: list[ResultChunk] = []
    for pidx, result in enumerate(stage_gen):
        items = list(result.chunk)
        chunk_ref = PickleDiskChunk.write(chunk_path_fn(pidx), items)
        results.append(ResultChunk(source_shard=source_shard, target_shard=result.target_shard, data=chunk_ref))
        if (pidx + 1) % 10 == 0:
            logger.info(
                "[shard %d] Wrote %d chunks so far (latest: %d items)",
                source_shard,
                pidx + 1,
                chunk_ref.count,
            )
    return results


def _write_stage_chunks(
    stage_gen: Iterator[StageResultChunk],
    source_shard: int,
    stage_dir: str,
    shard_idx: int,
    is_scatter: bool,
) -> list[ResultChunk]:
    """Write stage output chunks to disk.

    For scatter stages, attempts to stream all chunks into Parquet files
    with envelope wrapping. Falls back to pickle if Arrow conversion fails
    on the first chunk.

    For non-scatter stages, writes each chunk as a PickleDiskChunk.

    Args:
        stage_gen: Generator of StageResultChunks from run_stage
        source_shard: Source shard index
        stage_dir: Directory for this stage's output (``{prefix}/{execution_id}/{stage_name}``)
        shard_idx: Shard index (used to derive file paths)
        is_scatter: Whether this stage contains a scatter operation

    Returns:
        List of ResultChunks with ParquetDiskChunk or PickleDiskChunk data
    """
    first_result = next(stage_gen, None)
    if first_result is None:
        return []

    first_items = list(first_result.chunk)

    # Prepend the already-consumed first result back into the stream
    first_with_materialized_chunk = dataclasses.replace(first_result, chunk=first_items)
    full_gen = itertools.chain([first_with_materialized_chunk], stage_gen)

    if is_scatter:
        # Test Arrow serializability on the first chunk to decide native vs pickle envelope
        use_pickle_envelope = False
        try:
            test_envelope = _make_envelope(first_items, 0, 0)
            pa.RecordBatch.from_pylist(test_envelope)
            logger.info("Using Parquet for scatter serialization for shard %d", source_shard)
        except Exception:
            use_pickle_envelope = True
            logger.info(
                "Using Parquet with pickle envelope for scatter serialization for shard %d",
                source_shard,
            )

        parquet_path = f"{stage_dir}/shard-{shard_idx:04d}.parquet"
        return _write_parquet_scatter(full_gen, source_shard, parquet_path, pickled=use_pickle_envelope)

    def chunk_path_fn(idx: int) -> str:
        return f"{stage_dir}/shard-{shard_idx:04d}/chunk-{idx:04d}.pkl"

    return _write_pickle_chunks(full_gen, source_shard, chunk_path_fn)


class WorkerState(enum.Enum):
    INIT = "init"
    READY = "ready"
    BUSY = "busy"
    FAILED = "failed"
    DEAD = "dead"


@dataclass
class ShardTask:
    """Describes a unit of work for a worker: one shard through one stage."""

    shard_idx: int
    total_shards: int
    chunk_size: int
    shard: Shard
    operations: list[PhysicalOp]
    stage_name: str = "output"
    aux_shards: dict[int, Shard] | None = None


class ZephyrWorkerError(RuntimeError):
    """Raised when a worker encounters a fatal (non-transient) error."""


# Application errors that should never be retried by the execute() retry loop.
# These are deterministic errors (bad plan, invalid config, programming bugs)
# that would fail identically on every attempt. Infrastructure errors (OSError,
# RuntimeError from dead actors, Ray actor errors) are NOT listed here so they
# remain retryable.
_NON_RETRYABLE_ERRORS = (ZephyrWorkerError, ValueError, TypeError, KeyError, AttributeError)


# ---------------------------------------------------------------------------
# WorkerContext protocol — the public interface exposed to user task code
# ---------------------------------------------------------------------------


class WorkerContext(Protocol):
    def get_shared(self, name: str) -> Any: ...


_worker_ctx_var: ContextVar[ZephyrWorker | None] = ContextVar("zephyr_worker_ctx", default=None)


def zephyr_worker_ctx() -> WorkerContext:
    """Get the current worker's context. Only valid inside a worker task."""
    ctx = _worker_ctx_var.get()
    if ctx is None:
        raise RuntimeError("zephyr_worker_ctx() called outside of a worker task")
    return ctx


# ---------------------------------------------------------------------------
# ZephyrCoordinator
# ---------------------------------------------------------------------------


@dataclass
class JobStatus:
    stage: str
    completed: int
    total: int
    retries: int
    in_flight: int
    queue_depth: int
    done: bool
    fatal_error: str
    workers: dict[str, dict[str, Any]]


class ZephyrCoordinator:
    """Central coordinator actor that owns and manages the worker pool.

    The coordinator creates workers via current_client(), runs a background
    loop for discovery and heartbeat checking, and manages all pipeline
    execution internally. Workers poll the coordinator for tasks until
    receiving a SHUTDOWN signal.
    """

    def __init__(self):
        from fray.v2 import current_actor

        # Task management state
        self._task_queue: deque[ShardTask] = deque()
        self._results: dict[int, TaskResult] = {}
        self._worker_states: dict[str, WorkerState] = {}
        self._last_seen: dict[str, float] = {}
        self._stage_name: str = ""
        self._total_shards: int = 0
        self._completed_shards: int = 0
        self._retries: int = 0
        self._in_flight: dict[str, tuple[ShardTask, int]] = {}
        self._task_attempts: dict[int, int] = {}
        self._fatal_error: str | None = None
        self._chunk_prefix: str = ""
        self._execution_id: str = ""
        self._no_workers_timeout: float = 60.0

        # Worker management state (workers self-register via register_worker)
        self._worker_handles: dict[str, ActorHandle] = {}
        self._worker_group: Any = None  # ActorGroup, set via set_worker_group()
        self._coordinator_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._is_last_stage: bool = False
        self._initialized: bool = False
        self._pipeline_running: bool = False

        # Lock for accessing coordinator state from background thread
        self._lock = threading.Lock()

        actor_ctx = current_actor()
        self._name = f"{actor_ctx.group_name}"

    def initialize(
        self,
        chunk_prefix: str,
        coordinator_handle: ActorHandle,
        no_workers_timeout: float = 60.0,
    ) -> None:
        """Initialize coordinator for push-based worker registration.

        Workers register themselves by calling register_worker() when they start.
        This eliminates polling overhead and log spam from discover_new() calls.

        Args:
            chunk_prefix: Storage prefix for intermediate chunks
            coordinator_handle: Handle to this coordinator actor (passed from context)
            no_workers_timeout: Seconds to wait for at least one worker before failing.
        """
        self._chunk_prefix = chunk_prefix
        self._self_handle = coordinator_handle
        self._no_workers_timeout = no_workers_timeout

        logger.info("Coordinator initialized")

        # Start coordinator background loop (heartbeat checking only)
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop, daemon=True, name="zephyr-coordinator-loop"
        )
        self._coordinator_thread.start()
        self._initialized = True

    def set_worker_group(self, worker_group: Any) -> None:
        """Set the worker ActorGroup so the coordinator can detect permanent worker death."""
        self._worker_group = worker_group

    def register_worker(self, worker_id: str, worker_handle: ActorHandle) -> None:
        """Called by workers when they come online to register with coordinator.

        Handles re-registration from reconstructed workers (e.g. after node
        preemption) by updating the stale handle and resetting worker state.
        """
        with self._lock:
            if worker_id in self._worker_handles:
                logger.info("Worker %s re-registering (likely reconstructed), updating handle", worker_id)
                self._worker_handles[worker_id] = worker_handle
                self._worker_states[worker_id] = WorkerState.READY
                self._last_seen[worker_id] = time.monotonic()
                # NOTE: if there was a task assigned to the worker, there's a race condition between marking
                # the worker as unhealthy via heartbeat and re-registration. If we do not requeue we may silently
                # lose tasks.
                self._maybe_requeue_worker_task(worker_id)
                return

            self._worker_handles[worker_id] = worker_handle
            self._worker_states[worker_id] = WorkerState.READY
            self._last_seen[worker_id] = time.monotonic()

            logger.info("Worker %s registered, total: %d", worker_id, len(self._worker_handles))

    def _coordinator_loop(self) -> None:
        """Background loop for heartbeat checking and worker job monitoring."""
        last_log_time = 0.0

        while not self._shutdown_event.is_set():
            self.check_heartbeats()
            self._check_worker_group()

            now = time.monotonic()
            if self._has_active_execution() and now - last_log_time > 5.0:
                self._log_status()
                last_log_time = now

            self._shutdown_event.wait(timeout=0.5)

    def _check_worker_group(self) -> None:
        """Abort the pipeline if the worker job has permanently terminated."""
        if self._worker_group is None or self._fatal_error is not None:
            return
        try:
            if self._worker_group.is_done():
                self.abort(
                    "Worker job terminated permanently (all retries exhausted). "
                    "Workers likely crashed (OOM or other fatal error)."
                )
        except Exception:
            logger.debug("Failed to check worker group status", exc_info=True)

    def _has_active_execution(self) -> bool:
        return self._execution_id != "" and self._total_shards > 0 and self._completed_shards < self._total_shards

    def _log_status(self) -> None:
        alive = sum(1 for s in self._worker_states.values() if s in {WorkerState.READY, WorkerState.BUSY})
        dead = sum(1 for s in self._worker_states.values() if s in {WorkerState.FAILED, WorkerState.DEAD})
        logger.info(
            "[%s] [%s] %d/%d complete, %d in-flight, %d queued, %d/%d workers alive, %d dead",
            self._execution_id,
            self._stage_name,
            self._completed_shards,
            self._total_shards,
            len(self._in_flight),
            len(self._task_queue),
            alive,
            len(self._worker_handles),
            dead,
        )

    def _maybe_requeue_worker_task(self, worker_id: str) -> None:
        """If the worker has a task in-flight, re-queue it and mark the worker as failed."""
        task_and_attempt = self._in_flight.pop(worker_id, None)
        if task_and_attempt is not None:
            logger.info("Worker %s had an in-flight task, re-queuing", worker_id)
            task, _old_attempt = task_and_attempt
            self._task_attempts[task.shard_idx] += 1
            self._task_queue.append(task)
            self._retries += 1

    def _check_worker_heartbeats(self, timeout: float = 120.0) -> None:
        """Internal heartbeat check (called with lock held)."""
        now = time.monotonic()
        for worker_id, last in list(self._last_seen.items()):
            if now - last > timeout and self._worker_states.get(worker_id) not in {WorkerState.FAILED, WorkerState.DEAD}:
                logger.warning(f"Zephyr worker {worker_id} failed to heartbeat within timeout ({now - last:.1f}s)")
                self._worker_states[worker_id] = WorkerState.FAILED
                self._maybe_requeue_worker_task(worker_id)

    def pull_task(self, worker_id: str) -> tuple[ShardTask, int, dict] | str | None:
        """Called by workers to get next task.

        Returns:
            - (task, attempt, config) tuple if task available
            - "SHUTDOWN" string if coordinator is shutting down
            - None if no task available (worker should backoff and retry)
        """
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            self._worker_states[worker_id] = WorkerState.READY

            if self._shutdown_event.is_set():
                self._worker_states[worker_id] = WorkerState.DEAD
                return "SHUTDOWN"

            if self._fatal_error:
                return None

            if not self._task_queue:
                if self._is_last_stage:
                    self._worker_states[worker_id] = WorkerState.DEAD
                    return "SHUTDOWN"
                return None

            task = self._task_queue.popleft()
            attempt = self._task_attempts[task.shard_idx]
            self._in_flight[worker_id] = (task, attempt)
            self._worker_states[worker_id] = WorkerState.BUSY

            config = {
                "chunk_prefix": self._chunk_prefix,
                "execution_id": self._execution_id,
            }
            return (task, attempt, config)

    def _assert_in_flight_consistent(self, worker_id: str, shard_idx: int) -> None:
        """Assert _in_flight[worker_id], if present, matches the reported shard.

        Workers block on report_result/report_error before calling pull_task,
        so _in_flight can never point to a different shard. It may be absent
        if a heartbeat timeout already re-queued the task.
        """
        in_flight = self._in_flight.get(worker_id)
        if in_flight is not None:
            assert in_flight[0].shard_idx == shard_idx, (
                f"_in_flight mismatch for {worker_id}: reporting shard {shard_idx}, "
                f"but _in_flight tracks shard {in_flight[0].shard_idx}. "
                f"This indicates report_result/pull_task reordering — workers must block on report_result."
            )

    def report_result(self, worker_id: str, shard_idx: int, attempt: int, result: TaskResult) -> None:
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            self._assert_in_flight_consistent(worker_id, shard_idx)

            current_attempt = self._task_attempts.get(shard_idx, 0)
            if attempt != current_attempt:
                logger.warning(
                    f"Ignoring stale result from worker {worker_id} for shard {shard_idx} "
                    f"(attempt {attempt}, current {current_attempt})"
                )
                return

            self._results[shard_idx] = result
            self._completed_shards += 1
            self._in_flight.pop(worker_id, None)
            self._worker_states[worker_id] = WorkerState.READY

    def report_error(self, worker_id: str, shard_idx: int, error_info: str) -> None:
        """Worker reports a task failure. All errors are fatal."""
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            self._assert_in_flight_consistent(worker_id, shard_idx)
            self._in_flight.pop(worker_id, None)
            self._fatal_error = error_info
            self._worker_states[worker_id] = WorkerState.DEAD

    def heartbeat(self, worker_id: str) -> None:
        # No lock needed: _last_seen is only read by _check_worker_heartbeats
        # (which holds the lock), and monotonic float writes are atomic on CPython.
        self._last_seen[worker_id] = time.monotonic()

    def get_status(self) -> JobStatus:
        with self._lock:
            return JobStatus(
                stage=self._stage_name,
                completed=self._completed_shards,
                total=self._total_shards,
                retries=self._retries,
                in_flight=len(self._in_flight),
                queue_depth=len(self._task_queue),
                done=self._shutdown_event.is_set(),
                fatal_error=self._fatal_error,
                workers={
                    wid: {
                        "state": state.value,
                        "last_seen_ago": time.monotonic() - self._last_seen.get(wid, 0),
                    }
                    for wid, state in self._worker_states.items()
                },
            )

    def get_fatal_error(self) -> str | None:
        with self._lock:
            return self._fatal_error

    def abort(self, reason: str) -> None:
        """Set a fatal error that causes the current stage to fail immediately.

        Called by the external worker watchdog when the worker job terminates
        permanently (e.g. all retries exhausted after OOM).
        """
        with self._lock:
            if self._fatal_error is None:
                logger.error("Coordinator aborted: %s", reason)
                self._fatal_error = reason

    def _start_stage(self, stage_name: str, tasks: list[ShardTask], is_last_stage: bool = False) -> None:
        """Load a new stage's tasks into the queue."""
        with self._lock:
            self._task_queue = deque(tasks)
            self._results = {}
            self._stage_name = stage_name
            self._total_shards = len(tasks)
            self._completed_shards = 0
            self._retries = 0
            self._in_flight = {}
            self._task_attempts = {task.shard_idx: 0 for task in tasks}
            self._fatal_error = None
            self._is_last_stage = is_last_stage

    def _wait_for_stage(self) -> None:
        """Block until current stage completes or error occurs."""
        backoff = ExponentialBackoff(initial=0.1, maximum=1.0)
        last_log_completed = -1
        start_time = time.monotonic()
        all_dead_since: float | None = None
        no_workers_timeout = self._no_workers_timeout
        stage_done = threading.Event()

        while True:
            with self._lock:
                if self._fatal_error:
                    raise ZephyrWorkerError(self._fatal_error)

                completed = self._completed_shards
                total = self._total_shards

                if completed >= total:
                    return

                # Count alive workers (READY or BUSY), not just total registered.
                # Dead/failed workers stay in _worker_handles but can't make progress.
                alive_workers = sum(
                    1 for s in self._worker_states.values() if s in {WorkerState.READY, WorkerState.BUSY}
                )

                if alive_workers == 0:
                    now = time.monotonic()
                    elapsed = now - start_time

                    if all_dead_since is None:
                        all_dead_since = now
                        logger.warning("All workers are dead/failed. Waiting for workers to recover...")

                    dead_duration = now - all_dead_since
                    if dead_duration > no_workers_timeout:
                        raise ZephyrWorkerError(
                            f"No alive workers for {dead_duration:.1f}s "
                            f"(total elapsed {elapsed:.1f}s). "
                            f"All {len(self._worker_handles)} registered workers are dead/failed. "
                            "Check cluster resources and worker group configuration."
                        )
                else:
                    # Workers are alive — reset the dead timer
                    all_dead_since = None

            if completed != last_log_completed:
                logger.info("[%s] %d/%d tasks completed", self._stage_name, completed, total)
                last_log_completed = completed
                backoff.reset()

            stage_done.wait(timeout=backoff.next_interval())

    def _collect_results(self) -> dict[int, TaskResult]:
        """Return results for the completed stage."""
        with self._lock:
            return dict(self._results)

    def run_pipeline(
        self,
        plan: PhysicalPlan,
        execution_id: str,
        hints: ExecutionHint,
    ) -> list:
        """Run complete pipeline, blocking until done. Returns flattened results."""
        with self._lock:
            if self._pipeline_running:
                self._fatal_error = "run_pipeline called while another pipeline is already running"
                raise RuntimeError(self._fatal_error)
            self._pipeline_running = True
            self._execution_id = execution_id

        try:
            shards = _build_source_shards(plan.source_items)
            if not shards:
                return []

            # Identify the last stage that dispatches work to workers (non-RESHARD).
            # On that stage, idle workers receive SHUTDOWN once all tasks are
            # in-flight, so they exit eagerly instead of polling until
            # coordinator.shutdown().
            last_worker_stage_idx = max(
                (i for i, s in enumerate(plan.stages) if s.stage_type != StageType.RESHARD),
                default=-1,
            )

            for stage_idx, stage in enumerate(plan.stages):
                stage_label = f"stage{stage_idx}-{stage.stage_name(max_length=40)}"

                if stage.stage_type == StageType.RESHARD:
                    shards = _reshard_refs(shards, stage.output_shards or len(shards))
                    continue

                # Compute aux data for joins
                aux_per_shard = self._compute_join_aux(stage.operations, shards, hints, stage_idx)

                # Build and submit tasks
                tasks = _compute_tasks_from_shards(shards, stage, hints, aux_per_shard, stage_name=stage_label)
                logger.info("[%s] Starting stage %s with %d tasks", self._execution_id, stage_label, len(tasks))
                self._start_stage(stage_label, tasks, is_last_stage=(stage_idx == last_worker_stage_idx))

                # Wait for stage completion
                self._wait_for_stage()

                # Collect and regroup results for next stage
                result_refs = self._collect_results()
                shards = _regroup_result_refs(result_refs, len(shards), output_shard_count=stage.output_shards)

            # Flatten final results
            flat_result = []
            for shard in shards:
                for chunk in shard.chunks:
                    flat_result.extend(list(chunk))

            # Signal workers to shut down now that all stages are complete.
            self.shutdown()

            return flat_result
        finally:
            with self._lock:
                self._pipeline_running = False

    def _compute_join_aux(
        self,
        operations: list[PhysicalOp],
        shard_refs: list[Shard],
        hints: ExecutionHint,
        parent_stage_idx: int,
    ) -> list[dict[int, Shard]] | None:
        """Execute right sub-plans for join operations, returning aux refs per shard."""
        all_right_shard_refs: dict[int, list[Shard]] = {}

        for i, op in enumerate(operations):
            if not isinstance(op, Join) or op.right_plan is None:
                continue

            right_refs = _build_source_shards(op.right_plan.source_items)

            for stage_idx, right_stage in enumerate(op.right_plan.stages):
                if right_stage.stage_type == StageType.RESHARD:
                    right_refs = _reshard_refs(right_refs, right_stage.output_shards or len(right_refs))
                    continue

                join_stage_label = f"join-right-{parent_stage_idx}-{i}-stage{stage_idx}"
                right_tasks = _compute_tasks_from_shards(right_refs, right_stage, hints, stage_name=join_stage_label)
                self._start_stage(join_stage_label, right_tasks)
                self._wait_for_stage()
                raw = self._collect_results()
                right_refs = _regroup_result_refs(raw, len(right_refs), output_shard_count=right_stage.output_shards)

            if len(shard_refs) != len(right_refs):
                raise ValueError(
                    f"Sorted merge join requires equal shard counts. "
                    f"Left has {len(shard_refs)} shards, right has {len(right_refs)} shards."
                )
            all_right_shard_refs[i] = right_refs

        if not all_right_shard_refs:
            return None

        return [
            {op_idx: right_refs[shard_idx] for op_idx, right_refs in all_right_shard_refs.items()}
            for shard_idx in range(len(shard_refs))
        ]

    def __repr__(self) -> str:
        return f"ZephyrCoordinator(name={self._name})"

    def shutdown(self) -> None:
        """Signal workers to exit. Worker group is managed by ZephyrContext."""
        logger.info("[coordinator.shutdown] Starting shutdown")
        self._shutdown_event.set()

        # Wait for coordinator thread to exit
        if self._coordinator_thread is not None:
            self._coordinator_thread.join(timeout=5.0)

        logger.info("Coordinator shutdown complete")

    def set_chunk_config(self, prefix: str, execution_id: str) -> None:
        """Configure chunk storage for this execution."""
        with self._lock:
            self._chunk_prefix = prefix
            self._execution_id = execution_id

    def get_chunk_config(self) -> dict:
        """Return chunk storage configuration for workers."""
        with self._lock:
            return {
                "prefix": self._chunk_prefix,
                "execution_id": self._execution_id,
            }

    def start_stage(self, stage_name: str, tasks: list[ShardTask], is_last_stage: bool = False) -> None:
        """Load a new stage's tasks into the queue (legacy compat)."""
        self._start_stage(stage_name, tasks, is_last_stage=is_last_stage)

    def check_heartbeats(self, timeout: float = 120.0) -> None:
        """Marks stale workers as FAILED, re-queues their in-flight tasks."""
        with self._lock:
            self._check_worker_heartbeats(timeout)

    def collect_results(self) -> dict[int, TaskResult]:
        """Return results for the completed stage (legacy compat)."""
        return self._collect_results()

    def signal_done(self) -> None:
        """Signal workers that no more stages will be submitted (legacy compat)."""
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# ZephyrWorker
# ---------------------------------------------------------------------------


class ZephyrWorker:
    """Long-lived worker actor that polls coordinator for tasks.

    Workers register themselves with the coordinator on startup via
    register_worker(), then poll continuously until receiving a SHUTDOWN
    signal. Each task includes the execution config (shared data, chunk
    prefix, execution ID), so workers can handle multiple executions
    without restart.
    """

    def __init__(self, coordinator_handle: ActorHandle):
        from fray.v2 import current_actor

        self._coordinator = coordinator_handle
        self._shared_data_cache: dict[str, Any] = {}
        self._shutdown_event = threading.Event()
        self._chunk_prefix: str = ""
        self._execution_id: str = ""

        # Build descriptive worker ID from actor context
        actor_ctx = current_actor()
        self._worker_id = f"{actor_ctx.group_name}-{actor_ctx.index}"

        # Register with coordinator - wait is not stricly necessary, but it reduces the complexity
        self._coordinator.register_worker.remote(self._worker_id, actor_ctx.handle).result(timeout=60.0)

        # Start polling in a background thread
        self._polling_thread = threading.Thread(
            target=self._run_polling,
            args=(coordinator_handle,),
            daemon=True,
            name=f"zephyr-poll-{self._worker_id}",
        )
        self._polling_thread.start()

    def get_shared(self, name: str) -> Any:
        if name not in self._shared_data_cache:
            path = _shared_data_path(self._chunk_prefix, self._execution_id, name)
            logger.info("[%s] Loading shared data '%s' from %s", self._worker_id, name, path)
            t0 = time.monotonic()
            with open_url(path, "rb") as f:
                data = f.read()
            elapsed = time.monotonic() - t0
            self._shared_data_cache[name] = cloudpickle.loads(data)
            logger.info(
                "[%s] Loaded shared data '%s' in %.2fs (%d bytes)",
                self._worker_id,
                name,
                elapsed,
                len(data),
            )
        return self._shared_data_cache[name]

    def _run_polling(self, coordinator: ActorHandle) -> None:
        """Main polling loop. Runs in a background thread started by __init__."""
        logger.info("[%s] Starting polling loop", self._worker_id)

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator,),
            daemon=True,
            name=f"zephyr-hb-{self._worker_id}",
        )
        heartbeat_thread.start()

        try:
            self._poll_loop(coordinator)
        finally:
            self._shutdown_event.set()
            heartbeat_thread.join(timeout=5.0)
            logger.debug("[%s] Polling loop ended", self._worker_id)

    def _heartbeat_loop(
        self, coordinator: ActorHandle, interval: float = 5.0, max_consecutive_failures: int = 5
    ) -> None:
        logger.debug("[%s] Heartbeat loop starting", self._worker_id)
        heartbeat_count = 0
        consecutive_failures = 0
        while not self._shutdown_event.is_set():
            try:
                # Block on result to avoid congesting the coordinator RPC pipe
                # with fire-and-forget heartbeats.
                coordinator.heartbeat.remote(self._worker_id).result()
                heartbeat_count += 1
                consecutive_failures = 0
                if heartbeat_count % 10 == 1:
                    logger.debug("[%s] Sent heartbeat #%d", self._worker_id, heartbeat_count)
            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    "[%s] Heartbeat failed (%d/%d): %s",
                    self._worker_id,
                    consecutive_failures,
                    max_consecutive_failures,
                    e,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "[%s] %d consecutive heartbeat failures — coordinator is unreachable, shutting down",
                        self._worker_id,
                        consecutive_failures,
                    )
                    self._shutdown_event.set()
                    break
            self._shutdown_event.wait(timeout=interval)
        logger.debug("[%s] Heartbeat loop exiting after %d beats", self._worker_id, heartbeat_count)

    def _poll_loop(self, coordinator: ActorHandle) -> None:
        """Pure polling loop. Exits on SHUTDOWN signal, coordinator death, or shutdown event."""
        task_count = 0
        backoff = ExponentialBackoff(initial=0.1, maximum=1.0)

        future: ActorFuture | None = None
        future_start = 0.0
        warned = False

        while not self._shutdown_event.is_set():
            # Create a pull_task future if we don't have one in flight
            if future is None:
                future = coordinator.pull_task.remote(self._worker_id)
                future_start = time.monotonic()
                warned = False

            # Poll with a short timeout so we stay responsive to shutdown
            # without killing the worker (and its heartbeat thread) on slow
            # deserialization.
            try:
                response = future.result(timeout=0.5)
            except TimeoutError:
                elapsed = time.monotonic() - future_start
                if elapsed > 30 and not warned:
                    logger.warning("[%s] Waiting for coordinator pull_task response (%.0fs)", self._worker_id, elapsed)
                    warned = True
                continue
            except Exception as e:
                logger.info("[%s] pull_task failed (coordinator may be dead): %s", self._worker_id, e)
                break

            future = None  # consumed; next iteration will create a new one

            # SHUTDOWN signal - exit cleanly
            if response == "SHUTDOWN":
                logger.info("[%s] Received SHUTDOWN", self._worker_id)
                break

            # No task available - backoff and retry
            if response is None:
                time.sleep(backoff.next_interval())
                continue

            backoff.reset()

            # Unpack task and config
            task, attempt, config = response

            logger.info("[%s] Executing task for shard %d (attempt %d)", self._worker_id, task.shard_idx, attempt)
            try:
                t_0 = time.monotonic()
                result = self._execute_shard(task, config)
                logger.info(
                    "[%s] Task for shard %d completed in %.2f seconds",
                    self._worker_id,
                    task.shard_idx,
                    time.monotonic() - t_0,
                )
                # Block until coordinator records the result. This ensures
                # report_result is fully processed before the next pull_task,
                # preventing _in_flight tracking races.
                coordinator.report_result.remote(self._worker_id, task.shard_idx, attempt, result).result()
                task_count += 1
            except Exception as e:
                logger.error("Worker %s error on shard %d: %s", self._worker_id, task.shard_idx, e)
                import traceback

                coordinator.report_error.remote(
                    self._worker_id,
                    task.shard_idx,
                    "".join(traceback.format_exc()),
                ).result()

    def _execute_shard(self, task: ShardTask, config: dict) -> TaskResult:
        """Execute a stage's operations on a single shard.

        Returns list[TaskResult].
        """
        # Update config for this execution
        self._chunk_prefix = config["chunk_prefix"]
        self._execution_id = config["execution_id"]

        _worker_ctx_var.set(self)

        logger.info(
            "[%s] [shard %d/%d] Starting stage=%s, %d input chunks, %d ops",
            self._execution_id,
            task.shard_idx,
            task.total_shards,
            task.stage_name,
            len(task.shard.chunks),
            len(task.operations),
        )

        stage_ctx = StageContext(
            shard=task.shard,
            shard_idx=task.shard_idx,
            total_shards=task.total_shards,
            chunk_size=task.chunk_size,
            aux_shards=task.aux_shards,
        )

        stage_dir = f"{self._chunk_prefix}/{self._execution_id}/{task.stage_name}"
        is_scatter = any(isinstance(op, Scatter) for op in task.operations)

        results = _write_stage_chunks(
            run_stage(stage_ctx, task.operations),
            source_shard=task.shard_idx,
            stage_dir=stage_dir,
            shard_idx=task.shard_idx,
            is_scatter=is_scatter,
        )
        logger.info("[shard %d] Complete: %d chunks produced", task.shard_idx, len(results))
        return TaskResult(chunks=results)

    def __repr__(self) -> str:
        return f"ZephyrWorker(id={self._worker_id})"

    def shutdown(self) -> None:
        """Stop the worker's polling loop."""
        self._shutdown_event.set()
        if hasattr(self, "_polling_thread") and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5.0)


def _regroup_result_refs(
    result_refs: dict[int, TaskResult],
    input_shard_count: int,
    output_shard_count: int | None = None,
) -> list[Shard]:
    """Regroup worker output refs by output shard index without loading data."""
    output_by_shard: dict[int, list[Chunk]] = defaultdict(list)

    for _input_idx, result in result_refs.items():
        for chunk in result.chunks:
            output_by_shard[chunk.target_shard].append(chunk.data)

    num_output = max(max(output_by_shard.keys(), default=0) + 1, input_shard_count)
    if output_shard_count is not None:
        num_output = max(num_output, output_shard_count)
    return [Shard(chunks=output_by_shard.get(idx, [])) for idx in range(num_output)]


# ---------------------------------------------------------------------------
# Coordinator-as-Job infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CoordinatorJobConfig:
    """Serializable config for the coordinator job entrypoint."""

    plan: PhysicalPlan
    execution_id: str
    hints: ExecutionHint
    chunk_storage_prefix: str
    no_workers_timeout: float
    max_workers: int
    worker_resources: ResourceConfig
    name: str
    pipeline_id: int


def _run_coordinator_job(config: _CoordinatorJobConfig, result_path: str) -> None:
    """Entrypoint for the coordinator job.

    Hosts the coordinator actor in-process via host_actor(), creates
    worker actors as child jobs, runs the pipeline, and writes results
    to disk. The coordinator monitors worker job health directly in its
    maintenance loop (no separate watchdog thread).
    """
    from fray.v2.client import current_client

    logger.info(
        "Coordinator job starting: name=%s, execution_id=%s, pipeline=%d",
        config.name,
        config.execution_id,
        config.pipeline_id,
    )

    client = current_client()

    # Host coordinator actor in this process (no child job needed)
    coord_name = f"zephyr-{config.name}-p{config.pipeline_id}-coord"
    hosted = client.host_actor(
        ZephyrCoordinator,
        name=coord_name,
        actor_config=ActorConfig(max_concurrency=100),
    )
    coordinator = hosted.handle
    coordinator.initialize.remote(
        config.chunk_storage_prefix,
        coordinator,
        config.no_workers_timeout,
    ).result()

    # Create workers (child jobs)
    num_shards = config.plan.num_shards
    actual_workers = min(config.max_workers, num_shards) if num_shards > 0 else 0
    worker_group = None

    if actual_workers > 0:
        logger.info("Starting %d workers (max=%d, shards=%d)", actual_workers, config.max_workers, num_shards)
        worker_group = client.create_actor_group(
            ZephyrWorker,
            coordinator,
            name=f"zephyr-{config.name}-p{config.pipeline_id}-workers",
            count=actual_workers,
            resources=config.worker_resources,
            actor_config=ActorConfig(max_task_retries=10),
        )
        worker_group.wait_ready(count=1, timeout=3600.0)

        # Let the coordinator poll worker job health in its maintenance loop
        coordinator.set_worker_group.remote(worker_group).result()

    try:
        results = coordinator.run_pipeline.remote(config.plan, config.execution_id, config.hints).result()

        ensure_parent_dir(result_path)
        with open_url(result_path, "wb") as f:
            f.write(cloudpickle.dumps(results))
    except Exception as e:
        # Persist the exception so the caller can recover the original type
        # (important for non-retryable error detection).
        with suppress(Exception):
            ensure_parent_dir(result_path)
            with open_url(result_path, "wb") as f:
                f.write(cloudpickle.dumps(e))
        raise
    finally:
        if worker_group is not None:
            with suppress(Exception):
                worker_group.shutdown()
        with suppress(Exception):
            coordinator.shutdown.remote().result()
        with suppress(Exception):
            hosted.shutdown()


def _read_coordinator_result(result_path: str) -> Any:
    """Read the coordinator job's result file. Returns the deserialized object."""
    with open_url(result_path, "rb") as f:
        return cloudpickle.loads(f.read())


def _try_read_coordinator_result(result_path: str) -> Any:
    """Best-effort read of the result file. Returns None if unreadable.

    Used only in the retry error-recovery path where the coordinator job
    may have crashed before writing the file.
    """
    try:
        return _read_coordinator_result(result_path)
    except Exception:
        return None


@dataclass
class ZephyrContext:
    """Execution context for Zephyr pipelines.

    Each execute() call submits a coordinator *job* that internally creates
    coordinator and worker actors as child jobs. The coordinator job owns the
    full lifecycle: it boots workers, runs the pipeline, writes results to
    disk, and tears everything down. Iris cascading termination ensures that
    if the coordinator job dies, its children are cleaned up automatically.

    Args:
        client: The fray client to use. If None, auto-detects using current_client().
        max_workers: Upper bound on worker count. The actual count is
            min(max_workers, num_shards), computed at first execute(). If None,
            defaults to os.cpu_count() for LocalClient, or 128 for distributed clients.
        resources: Resource config per worker.
        chunk_storage_prefix: Storage prefix for intermediate chunks. If None, defaults
            to MARIN_PREFIX/tmp/zephyr or /tmp/zephyr.
        name: Descriptive name for this context, used in actor group names for debugging.
            Defaults to a random 8-character hex string.
        no_workers_timeout: Seconds to wait for at least one worker before failing a stage.
            Defaults to 600s.
        max_execution_retries: Maximum number of times to retry a pipeline execution after
            an infrastructure failure (e.g., coordinator VM preemption). Application errors
            (ZephyrWorkerError) are never retried. Defaults to 100.
    """

    client: Client | None = None
    max_workers: int | None = None
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="1g"))
    chunk_storage_prefix: str | None = None
    name: str = ""
    no_workers_timeout: float | None = None
    # NOTE: 100 is fairly aggressive but it fits the preemptible env better
    max_execution_retries: int = 100

    # Shared data staged by put(), uploaded to disk at the start of execute()
    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    # Handle to the coordinator job (for termination on retry/shutdown)
    _coordinator_job: JobHandle | None = field(default=None, repr=False)
    # NOTE: execute calls increment this at the very beginning
    _pipeline_id: int = field(default=-1, repr=False)

    def __post_init__(self):
        if self.client is None:
            from fray.v2.client import current_client

            self.client = current_client()

        if self.max_workers is None:
            from fray.v2.local_backend import LocalClient

            if isinstance(self.client, LocalClient):
                self.max_workers = os.cpu_count() or 1
            else:
                # Default to 128 for distributed, but allow override
                env_val = os.environ.get("ZEPHYR_MAX_WORKERS")
                self.max_workers = int(env_val) if env_val else 128

        if self.no_workers_timeout is None:
            self.no_workers_timeout = 6 * 60 * 60  # 6 hours

        if self.chunk_storage_prefix is None:
            self.chunk_storage_prefix = marin_temp_bucket(ttl_days=3, prefix="zephyr")

        # make sure each context is unique
        self.name = f"{self.name}-{uuid.uuid4().hex[:8]}"

    def put(self, name: str, obj: Any) -> None:
        """Stage shared data for workers to load on demand.

        Must be called before execute(). The object must be picklable.
        Workers access it via zephyr_worker_ctx().get_shared(name), which
        loads from disk on first access and caches locally.

        The actual serialization to disk happens at the start of execute(),
        once the execution_id is known, so each execution is isolated.
        """
        self._shared_data[name] = obj

    def _upload_shared_data(self, execution_id: str) -> None:
        """Serialize all staged shared data to disk under the execution directory."""
        for name, obj in self._shared_data.items():
            path = _shared_data_path(self.chunk_storage_prefix, execution_id, name)
            ensure_parent_dir(path)
            t0 = time.monotonic()
            data = cloudpickle.dumps(obj)
            elapsed = time.monotonic() - t0
            with open_url(path, "wb") as f:
                f.write(data)
            logger.info(
                "Shared data '%s' written to %s (serialized %d bytes in %.2fs)",
                name,
                path,
                len(data),
                elapsed,
            )

    def execute(
        self,
        dataset: Dataset,
        hints: ExecutionHint = ExecutionHint(),
        verbose: bool = False,
        dry_run: bool = False,
    ) -> Sequence:
        """Execute a dataset pipeline.

        Submits a coordinator *job* that creates coordinator and worker
        actors as child jobs, runs the pipeline, and writes results to
        disk. If the coordinator job dies (e.g., VM preemption), the
        pipeline is retried up to ``max_execution_retries`` times.
        Application errors (``ZephyrWorkerError``) are never retried.
        """
        plan = compute_plan(dataset, hints)
        if verbose or dry_run:
            _print_plan(dataset.operations, plan)
        if dry_run:
            return []

        # NOTE: pipeline ID incremented on clean completion only
        self._pipeline_id += 1
        last_exception: Exception | None = None
        # Backoff between retries to avoid hammering an overloaded controller.
        # Starts at 2s, caps at 60s. Resets on successful pipeline startup.
        backoff = ExponentialBackoff(initial=2.0, maximum=60.0, factor=2.0, jitter=0.1)
        for attempt in range(self.max_execution_retries + 1):
            execution_id = _generate_execution_id()
            logger.info(
                "Starting zephyr pipeline: %s (pipeline %d, attempt %d)", execution_id, self._pipeline_id, attempt
            )

            result_path = f"{self.chunk_storage_prefix}/{execution_id}/results.pkl"

            try:
                self._upload_shared_data(execution_id)

                config = _CoordinatorJobConfig(
                    plan=plan,
                    execution_id=execution_id,
                    hints=hints,
                    chunk_storage_prefix=self.chunk_storage_prefix,
                    no_workers_timeout=self.no_workers_timeout,
                    max_workers=self.max_workers,
                    worker_resources=self.resources,
                    name=self.name,
                    pipeline_id=self._pipeline_id,
                )

                job_name = f"zephyr-{self.name}-p{self._pipeline_id}-a{attempt}"
                # The wrapper job just blocks on child actors; real
                # resources are requested by the coordinator/worker children.
                # Set the context var so the coordinator job inherits self.client
                # instead of auto-detecting (which may pick a different backend).
                from fray.v2.client import set_current_client

                with set_current_client(self.client):
                    self._coordinator_job = self.client.submit(
                        JobRequest(
                            name=job_name,
                            entrypoint=Entrypoint.from_callable(
                                _run_coordinator_job,
                                args=(config, result_path),
                            ),
                            resources=ResourceConfig(cpu=1, ram="1g"),
                        )
                    )

                backoff.reset()
                logger.info("Coordinator job submitted: %s (job_id=%s)", job_name, self._coordinator_job.job_id)

                self._coordinator_job.wait(raise_on_failure=True)

                # Read results written by the coordinator job.
                # This must succeed — the job completed successfully.
                result = _read_coordinator_result(result_path)
                if isinstance(result, Exception):
                    raise result
                return result

            except _NON_RETRYABLE_ERRORS:
                raise

            except Exception as e:
                # The coordinator job may have persisted the original
                # exception before failing. Recover it so non-retryable
                # errors are detected correctly.
                result = _try_read_coordinator_result(result_path)
                if isinstance(result, _NON_RETRYABLE_ERRORS):
                    raise result from None

                last_exception = e
                if attempt >= self.max_execution_retries:
                    raise

                delay = backoff.next_interval()
                logger.warning(
                    "Pipeline attempt %d failed (%d retries left), retrying in %.1fs: %s",
                    attempt,
                    self.max_execution_retries - attempt,
                    delay,
                    e,
                )
                time.sleep(delay)

            finally:
                # Kill coordinator job (cascade kills child actors)
                self._terminate_coordinator_job()
                _cleanup_execution(self.chunk_storage_prefix, execution_id)

        # Should be unreachable, but just in case
        raise last_exception  # type: ignore[misc]

    def _terminate_coordinator_job(self) -> None:
        if self._coordinator_job is not None:
            with suppress(Exception):
                self._coordinator_job.terminate()
            self._coordinator_job = None

    def shutdown(self) -> None:
        """Shutdown the coordinator job and all child actors."""
        self._terminate_coordinator_job()


def _reshard_refs(shards: list[Shard], num_shards: int) -> list[Shard]:
    """Reshard shard refs by output shard index without loading data."""
    output_by_shard: dict[int, list[Chunk]] = defaultdict(list)
    output_idx = 0
    for shard in shards:
        for chunk in shard.chunks:
            output_by_shard[output_idx].append(chunk)
            output_idx = (output_idx + 1) % num_shards
    return [Shard(chunks=output_by_shard.get(idx, [])) for idx in range(num_shards)]


def _build_source_shards(source_items: list[SourceItem]) -> list[Shard]:
    """Build shard data from source items.

    Each source item becomes a single-element chunk in its assigned shard.
    """
    items_by_shard: dict[int, list] = defaultdict(list)
    for item in source_items:
        items_by_shard[item.shard_idx].append(item.data)

    num_shards = max(items_by_shard.keys()) + 1 if items_by_shard else 0
    shards = []
    for i in range(num_shards):
        shards.append(Shard(chunks=[MemChunk(items=items_by_shard.get(i, []))]))

    return shards


def _compute_tasks_from_shards(
    shard_refs: list[Shard],
    stage,
    hints: ExecutionHint,
    aux_per_shard: list[dict[int, Shard]] | None = None,
    stage_name: str | None = None,
) -> list[ShardTask]:
    """Convert shard references into ShardTasks for the coordinator."""
    total = len(shard_refs)
    tasks = []
    # Sanitize for use as a path component: replace non-alphanumeric runs with '-'
    raw_name = stage_name or stage.stage_name(max_length=60)
    output_stage_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_name).strip("-")

    for i, shard in enumerate(shard_refs):
        aux_shards = None
        if aux_per_shard and aux_per_shard[i]:
            aux_shards = aux_per_shard[i]

        tasks.append(
            ShardTask(
                shard_idx=i,
                total_shards=total,
                chunk_size=hints.chunk_size,
                shard=shard,
                operations=stage.operations,
                stage_name=output_stage_name,
                aux_shards=aux_shards,
            )
        )

    return tasks


def _print_plan(original_ops: list, plan: PhysicalPlan) -> None:
    """Print the physical plan showing shard count and operation fusion."""
    total_physical_ops = sum(len(stage.operations) for stage in plan.stages)

    logger.info("\n=== Physical Execution Plan ===\n")
    logger.info(f"Shards: {plan.num_shards}")
    logger.info(f"Original operations: {len(original_ops)}")
    logger.info(f"Stages: {len(plan.stages)}")
    logger.info(f"Physical ops: {total_physical_ops}\n")

    logger.info("Original pipeline:")
    for i, op in enumerate(original_ops, 1):
        logger.info(f"  {i}. {op}")

    logger.info("\nPhysical stages:")
    for i, stage in enumerate(plan.stages, 1):
        stage_desc = stage.stage_name()
        hint_parts = []
        if stage.stage_type == StageType.RESHARD:
            hint_parts.append(f"reshard→{stage.output_shards}")
        if any(isinstance(op, Join) for op in stage.operations):
            hint_parts.append("join")
        hint_str = f" [{', '.join(hint_parts)}]" if hint_parts else ""
        logger.info(f"  {i}. {stage_desc}{hint_str}")

    logger.info("\n=== End Plan ===\n")
