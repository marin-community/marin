# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor-based execution engine for Zephyr pipelines.

Workers pull tasks from the coordinator, execute shard operations, and report
results back. This enables persistent worker state (caches, loaded models),
transient error recovery, and backend-agnostic dispatch via fray v2's Client
protocol.
"""

from __future__ import annotations
import enum
import logging
import os
import pickle
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

import cloudpickle
import fsspec
from fray.v2 import ActorConfig, ActorHandle, Client, ResourceConfig
from iris.temp_buckets import get_temp_bucket_path
from iris.time_utils import ExponentialBackoff

from zephyr.dataset import Dataset
from zephyr.plan import (
    ExecutionHint,
    Join,
    PhysicalOp,
    PhysicalPlan,
    SourceItem,
    StageContext,
    StageType,
    compute_plan,
    run_stage,
)
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)


class Chunk(Protocol):
    def __iter__(self) -> Iterator: ...


@dataclass(frozen=True)
class DiskChunk:
    """Reference to a chunk stored on disk.

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
    def write(cls, path: str, data: list) -> DiskChunk:
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
        with fsspec.open(unique_path, "wb") as f:
            pickle.dump(data, f)
        return cls(path=unique_path, count=count)

    def read(self) -> list:
        """Load chunk data from disk."""
        with fsspec.open(self.path, "rb") as f:
            return pickle.load(f)


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
    return uuid.uuid4().hex[:12]


def _shared_data_path(prefix: str, execution_id: str, name: str) -> str:
    """Path for a shared data object: {prefix}/{execution_id}/shared/{name}.pkl"""
    return f"{prefix}/{execution_id}/shared/{name}.pkl"


def _chunk_path(
    prefix: str,
    execution_id: str,
    stage_name: str,
    shard_idx: int,
    chunk_idx: int,
) -> str:
    """Generate path for a chunk file.

    Format: {prefix}/{execution_id}/{stage_name}/shard-{shard_idx:04d}/chunk-{chunk_idx:04d}.pkl
    """
    return f"{prefix}/{execution_id}/{stage_name}/shard-{shard_idx:04d}/chunk-{chunk_idx:04d}.pkl"


def _cleanup_execution(prefix: str, execution_id: str) -> None:
    """Remove all chunk files for an execution."""
    exec_dir = f"{prefix}/{execution_id}"
    fs = fsspec.core.url_to_fs(exec_dir)[0]

    if fs.exists(exec_dir):
        try:
            fs.rm(exec_dir, recursive=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup chunks at {exec_dir}: {e}")


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
        self._coordinator_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._is_last_stage: bool = False
        self._initialized: bool = False

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
        """Background loop for heartbeat checking only.

        Workers register themselves via register_worker() - no polling needed.
        """
        last_log_time = 0.0

        while not self._shutdown:
            # Check heartbeats, re-queue stale tasks.
            # NOTE: we could use self._self_handle.check_heartbeats.remote() to
            # serialize this with worker RPCs (pull_task, report_result) via the
            # concurrency queue instead of running it inline.
            self.check_heartbeats()

            # Log status periodically during active execution
            now = time.monotonic()
            if self._has_active_execution() and now - last_log_time > 5.0:
                self._log_status()
                last_log_time = now

            time.sleep(0.5)

    def _has_active_execution(self) -> bool:
        return self._execution_id != "" and self._total_shards > 0 and self._completed_shards < self._total_shards

    def _log_status(self) -> None:
        alive = sum(1 for s in self._worker_states.values() if s in {WorkerState.READY, WorkerState.BUSY})
        dead = sum(1 for s in self._worker_states.values() if s in {WorkerState.FAILED, WorkerState.DEAD})
        logger.info(
            "[%s] %d/%d complete, %d in-flight, %d queued, %d/%d workers alive, %d dead",
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

    def _check_worker_heartbeats(self, timeout: float = 30.0) -> None:
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

            if self._shutdown:
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
                done=self._shutdown,
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
        backoff = ExponentialBackoff(initial=0.05, maximum=1.0)
        last_log_completed = -1
        start_time = time.monotonic()
        all_dead_since: float | None = None
        no_workers_timeout = self._no_workers_timeout

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

            time.sleep(backoff.next_interval())

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
            self._execution_id = execution_id

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
            logger.info("Starting stage %s with %d tasks", stage_label, len(tasks))
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
        self._shutdown = True

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

    def check_heartbeats(self, timeout: float = 30.0) -> None:
        """Marks stale workers as FAILED, re-queues their in-flight tasks."""
        with self._lock:
            self._check_worker_heartbeats(timeout)

    def collect_results(self) -> dict[int, TaskResult]:
        """Return results for the completed stage (legacy compat)."""
        return self._collect_results()

    def signal_done(self) -> None:
        """Signal workers that no more stages will be submitted (legacy compat)."""
        self._shutdown = True


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
            with fsspec.open(path, "rb") as f:
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

    def _heartbeat_loop(self, coordinator: ActorHandle, interval: float = 5.0) -> None:
        logger.debug("[%s] Heartbeat loop starting", self._worker_id)
        heartbeat_count = 0
        while not self._shutdown_event.is_set():
            try:
                # Block on result to avoid congesting the coordinator RPC pipe
                # with fire-and-forget heartbeats.
                coordinator.heartbeat.remote(self._worker_id).result()
                heartbeat_count += 1
                if heartbeat_count % 10 == 1:
                    logger.debug("[%s] Sent heartbeat #%d", self._worker_id, heartbeat_count)
            except Exception as e:
                logger.warning("[%s] Heartbeat failed: %s", self._worker_id, e)
            self._shutdown_event.wait(timeout=interval)
        logger.debug("[%s] Heartbeat loop exiting after %d beats", self._worker_id, heartbeat_count)

    def _poll_loop(self, coordinator: ActorHandle) -> None:
        """Pure polling loop. Exits on SHUTDOWN signal, coordinator death, or shutdown event."""
        loop_count = 0
        task_count = 0
        backoff = ExponentialBackoff(initial=0.05, maximum=1.0)

        while not self._shutdown_event.is_set():
            loop_count += 1
            if loop_count % 100 == 1:
                logger.debug("[%s] Poll iteration #%d, tasks completed: %d", self._worker_id, loop_count, task_count)

            try:
                response = coordinator.pull_task.remote(self._worker_id).result(timeout=30.0)
            except Exception as e:
                # Coordinator is dead or unreachable - exit gracefully
                logger.info("[%s] pull_task failed (coordinator may be dead): %s", self._worker_id, e)
                break

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
            "[shard %d/%d] Starting stage=%s, %d input chunks, %d ops",
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

        results: list[ResultChunk] = []
        chunk_idx = 0

        for stage_output in run_stage(stage_ctx, task.operations):
            chunk_path = _chunk_path(
                self._chunk_prefix,
                self._execution_id,
                task.stage_name,
                task.shard_idx,
                chunk_idx,
            )
            chunk = list(stage_output.chunk)
            chunk_ref = DiskChunk.write(chunk_path, chunk)
            results.append(
                ResultChunk(
                    source_shard=stage_output.source_shard,
                    target_shard=stage_output.target_shard,
                    data=chunk_ref,
                )
            )
            chunk_idx += 1
            if chunk_idx % 10 == 0:
                logger.info(
                    "[shard %d] Wrote %d chunks so far (latest: %d items)",
                    task.shard_idx,
                    chunk_idx,
                    len(stage_output.chunk),
                )

        logger.info("[shard %d] Complete: %d chunks produced", task.shard_idx, chunk_idx)
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


@dataclass
class ZephyrContext:
    """Execution context for Zephyr pipelines.

    Each execute() call creates a fresh coordinator and worker pool, runs
    the pipeline, then tears everything down. Workers are sized to
    min(max_workers, plan.num_shards) to avoid over-provisioning. Shared
    data registered via put() is serialized to disk once and loaded lazily
    by workers on first access.

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
            (ZephyrWorkerError) are never retried. Defaults to 3.
    """

    client: Client | None = None
    max_workers: int | None = None
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="1g"))
    chunk_storage_prefix: str | None = None
    name: str = ""
    no_workers_timeout: float | None = None
    max_execution_retries: int = 3

    # Shared data staged by put(), uploaded to disk at the start of execute()
    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    _coordinator: ActorHandle | None = field(default=None, repr=False)
    _coordinator_group: Any = field(default=None, repr=False)
    _worker_group: Any = field(default=None, repr=False)
    _worker_count: int = field(default=0, repr=False)
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
            temp_prefix = get_temp_bucket_path(ttl_days=3, prefix="zephyr")
            if temp_prefix is None:
                marin_prefix = os.environ.get("MARIN_PREFIX")
                if not marin_prefix:
                    raise RuntimeError(
                        "MARIN_PREFIX must be set when using a distributed backend.\n"
                        "  Example: export MARIN_PREFIX=gs://marin-us-central2"
                    )
                temp_prefix = f"{marin_prefix}/tmp/zephyr"

            self.chunk_storage_prefix = temp_prefix

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
            with fsspec.open(path, "wb") as f:
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

        Each call creates a fresh coordinator and worker pool, runs the
        pipeline, then tears everything down. If the coordinator dies
        mid-execution (e.g., VM preemption), the pipeline is retried
        with fresh actors up to ``max_execution_retries`` times.
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
        for attempt in range(self.max_execution_retries + 1):
            execution_id = _generate_execution_id()
            logger.info(
                "Starting zephyr pipeline: %s (pipeline %d, attempt %d)", execution_id, self._pipeline_id, attempt
            )

            try:
                self._upload_shared_data(execution_id)
                self._create_coordinator(attempt)
                self._create_workers(plan.num_shards, attempt)

                # Run pipeline on coordinator (blocking call).
                # run_pipeline() calls coordinator.shutdown() at the end,
                # which causes workers to receive SHUTDOWN on their next
                # pull_task() call.
                results = self._coordinator.run_pipeline.remote(plan, execution_id, hints).result()

                return results

            except _NON_RETRYABLE_ERRORS:
                raise

            except Exception as e:
                last_exception = e
                if attempt >= self.max_execution_retries:
                    raise

                logger.warning(
                    "Pipeline attempt %d failed (%d retries left), retrying: %s",
                    attempt,
                    self.max_execution_retries - attempt,
                    e,
                )

            finally:
                # Tear down coordinator and workers for this pipeline
                self.shutdown()
                # Clean up chunks for this execution
                _cleanup_execution(self.chunk_storage_prefix, execution_id)

        # Should be unreachable, but just in case
        raise last_exception  # type: ignore[misc]

    def _create_coordinator(self, attempt: int = 0) -> None:
        """Create a fresh coordinator actor."""
        # max_concurrency allows workers to call pull_task/report_result
        # while run_pipeline blocks.
        logger.info("Starting coordinator for %s (pipeline %d, attempt %d)", self.name, self._pipeline_id, attempt)
        coordinator_resources = ResourceConfig(cpu=1, ram="2g")
        coordinator_actor_config = ActorConfig(max_concurrency=100)
        self._coordinator_group = self.client.create_actor_group(
            ZephyrCoordinator,
            name=f"zephyr-{self.name}-p{self._pipeline_id}-a{attempt}-coord",
            count=1,
            resources=coordinator_resources,
            actor_config=coordinator_actor_config,
        )
        self._coordinator = self._coordinator_group.wait_ready()[0]

        self._coordinator.initialize.remote(
            self.chunk_storage_prefix,
            self._coordinator,
            self.no_workers_timeout,
        ).result()

        logger.info("Coordinator initialized for %s", self.name)

    def _create_workers(self, num_shards: int, attempt: int = 0) -> None:
        """Create a fresh worker pool sized to demand.

        The worker count is min(max_workers, num_shards) to avoid
        over-provisioning when there are fewer shards than the cap.
        """
        if num_shards == 0:
            return

        assert self.max_workers is not None  # set by __post_init__
        actual_workers = min(self.max_workers, num_shards)
        logger.info(
            "Starting worker group: %d workers (max_workers=%d, num_shards=%d, attempt=%d)",
            actual_workers,
            self.max_workers,
            num_shards,
            attempt,
        )
        self._worker_group = self.client.create_actor_group(
            ZephyrWorker,
            self._coordinator,  # Pass coordinator handle as init arg
            name=f"zephyr-{self.name}-p{self._pipeline_id}-a{attempt}-workers",
            count=actual_workers,
            resources=self.resources,
        )

        self._worker_count = actual_workers

        # Wait for at least one worker to be ready before proceeding
        self._worker_group.wait_ready(count=1, timeout=3600.0)

        logger.info("ZephyrContext initialized with coordinator and %d workers", actual_workers)

    def shutdown(self) -> None:
        """Shutdown coordinator and all workers."""
        # Terminate worker group
        if self._worker_group is not None:
            with suppress(Exception):
                self._worker_group.shutdown()

        # Terminate coordinator actor group
        if self._coordinator_group is not None:
            with suppress(Exception):
                self._coordinator_group.shutdown()

        self._coordinator = None
        self._coordinator_group = None
        self._worker_group = None


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
    output_stage_name = stage_name or stage.stage_name(max_length=60)

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
