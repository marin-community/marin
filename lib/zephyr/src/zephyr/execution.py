# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

import fsspec
from fray.v2 import ActorHandle, Client, ResourceConfig
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
    """Reference to a chunk stored on disk."""

    path: str
    count: int

    def __iter__(self) -> Iterator:
        return iter(self.read())

    @classmethod
    def write(cls, path: str, data: list) -> DiskChunk:
        """Write data to path using temp-file pattern.

        Uses a .tmp suffix and rename/move to reduce (but not eliminate)
        the risk of partial writes. On object stores like GCS/S3, the
        move is not atomic but minimizes corruption windows.
        """
        ensure_parent_dir(path)
        temp_path = f"{path}.tmp"
        fs = fsspec.core.url_to_fs(path)[0]
        data = list(data)
        count = len(data)
        try:
            with fsspec.open(temp_path, "wb") as f:
                pickle.dump(data, f)
            fs.mv(temp_path, path)
        except Exception:
            with suppress(Exception):
                if fs.exists(temp_path):
                    fs.rm(temp_path)
            raise
        return cls(path=path, count=count)

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


# ---------------------------------------------------------------------------
# shard_ctx() — worker-side context access
# ---------------------------------------------------------------------------

_shard_ctx_var: ContextVar[ZephyrWorker | None] = ContextVar("zephyr_shard_ctx", default=None)


def shard_ctx() -> ZephyrWorker:
    """Get the current worker's context. Only valid inside a worker task."""
    ctx = _shard_ctx_var.get()
    if ctx is None:
        raise RuntimeError("shard_ctx() called outside of a worker task")
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
        # Task management state
        self._task_queue: deque[ShardTask] = deque()
        self._results: dict[int, TaskResult] = {}
        self._worker_states: dict[str, WorkerState] = {}
        self._last_seen: dict[str, float] = {}
        self._shared_data: dict[str, Any] = {}
        self._stage_name: str = ""
        self._total_shards: int = 0
        self._completed_shards: int = 0
        self._retries: int = 0
        self._in_flight: dict[str, tuple[ShardTask, int]] = {}
        self._task_attempts: dict[int, int] = {}
        self._fatal_error: str | None = None
        self._chunk_prefix: str = ""
        self._execution_id: str = ""

        # Worker management state
        self._worker_group: Any = None
        self._worker_handles: list[ActorHandle] = []
        self._worker_futures: list = []
        self._coordinator_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._initialized: bool = False

        # Lock for accessing coordinator state from background thread
        self._lock = threading.Lock()

    def initialize(
        self,
        chunk_prefix: str,
        coordinator_handle: ActorHandle,
        worker_group: Any,  # ActorGroup - use Any to avoid import issues
    ) -> None:
        """Initialize coordinator with worker group for continuous discovery.

        The worker group is created by ZephyrContext using the appropriate client,
        then passed here. Coordinator owns the group and polls it for new workers
        via discover_new() in the background loop - we do NOT block waiting for
        all workers to be ready upfront.

        Args:
            chunk_prefix: Storage prefix for intermediate chunks
            coordinator_handle: Handle to this coordinator actor (passed from context)
            worker_group: ActorGroup with workers - coordinator will poll discover_new()
        """
        self._chunk_prefix = chunk_prefix
        self._self_handle = coordinator_handle
        self._worker_group = worker_group

        # Don't block on wait_ready() - let the coordinator loop discover workers
        # as they become available. This allows execution to start immediately
        # and work with whatever workers are ready.

        logger.info("Coordinator initialized, discovering workers asynchronously")

        # Start coordinator background loop (handles discovery + heartbeats)
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop, daemon=True, name="zephyr-coordinator-loop"
        )
        self._coordinator_thread.start()
        self._initialized = True

    def _coordinator_loop(self) -> None:
        """Background loop for discovery and heartbeat checking."""
        last_log_time = 0.0

        while not self._shutdown:
            # Discover newly available workers
            if self._worker_group:
                new_handles = self._worker_group.discover_new()
                if new_handles:
                    with self._lock:
                        for handle in new_handles:
                            self._worker_handles.append(handle)
                            self._worker_futures.append(handle.start_polling.remote(self._self_handle))
                    logger.info("Discovered %d new workers, total: %d", len(new_handles), len(self._worker_handles))

            # Check heartbeats, re-queue stale tasks
            with self._lock:
                self._check_heartbeats_internal()

            # Log status periodically during active execution
            now = time.monotonic()
            if self._has_active_execution() and now - last_log_time > 5.0:
                self._log_status()
                last_log_time = now

            time.sleep(0.5)

    def _has_active_execution(self) -> bool:
        return self._execution_id != "" and self._total_shards > 0 and self._completed_shards < self._total_shards

    def _log_status(self) -> None:
        logger.info(
            "[%s] %d/%d complete, %d in-flight, %d queued, %d workers",
            self._stage_name,
            self._completed_shards,
            self._total_shards,
            len(self._in_flight),
            len(self._task_queue),
            len(self._worker_handles),
        )

    def _check_heartbeats_internal(self, timeout: float = 30.0) -> None:
        """Internal heartbeat check (called with lock held)."""
        now = time.monotonic()
        for worker_id, last in list(self._last_seen.items()):
            if now - last > timeout and self._worker_states.get(worker_id) != WorkerState.DEAD:
                logger.warning(f"Worker {worker_id} heartbeat timeout ({now - last:.1f}s)")
                self._worker_states[worker_id] = WorkerState.FAILED
                task_and_attempt = self._in_flight.pop(worker_id, None)
                if task_and_attempt is not None:
                    task, _old_attempt = task_and_attempt
                    self._task_attempts[task.shard_idx] += 1
                    self._task_queue.append(task)
                    self._retries += 1

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
                return "SHUTDOWN"

            if self._fatal_error:
                return None

            if not self._task_queue:
                return None

            task = self._task_queue.popleft()
            attempt = self._task_attempts[task.shard_idx]
            self._in_flight[worker_id] = (task, attempt)
            self._worker_states[worker_id] = WorkerState.BUSY

            config = {
                "shared_data": self._shared_data,
                "chunk_prefix": self._chunk_prefix,
                "execution_id": self._execution_id,
            }
            return (task, attempt, config)

    def report_result(self, worker_id: str, shard_idx: int, attempt: int, result: TaskResult) -> None:
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()

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
            self._in_flight.pop(worker_id, None)
            self._fatal_error = error_info
            self._worker_states[worker_id] = WorkerState.DEAD

    def heartbeat(self, worker_id: str) -> None:
        with self._lock:
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

    def _start_stage(self, stage_name: str, tasks: list[ShardTask]) -> None:
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

    def _wait_for_stage(self, no_workers_timeout: float = 60.0) -> None:
        """Block until current stage completes or error occurs.

        Args:
            no_workers_timeout: Seconds to wait for at least one worker before failing.
                If no workers are discovered within this time, raises ZephyrWorkerError.
        """
        backoff = ExponentialBackoff(initial=0.05, maximum=1.0)
        last_log_completed = -1
        start_time = time.monotonic()
        warned_no_workers = False

        while True:
            with self._lock:
                if self._fatal_error:
                    raise ZephyrWorkerError(self._fatal_error)

                num_workers = len(self._worker_handles)
                completed = self._completed_shards
                total = self._total_shards

                if completed >= total:
                    return

                # Fail fast if no workers appear within timeout
                if num_workers == 0:
                    elapsed = time.monotonic() - start_time
                    if elapsed > no_workers_timeout:
                        raise ZephyrWorkerError(
                            f"No workers available after {elapsed:.1f}s. "
                            "Check cluster resources and worker group configuration."
                        )
                    if not warned_no_workers and elapsed > 5.0:
                        logger.warning(
                            "No workers available yet after %.1fs, waiting for discovery...",
                            elapsed,
                        )
                        warned_no_workers = True

            if completed != last_log_completed:
                logger.info("[%s] %d/%d tasks completed", self._stage_name, completed, total)
                last_log_completed = completed
                backoff.reset()

            time.sleep(backoff.next_interval())

    def _collect_results(self) -> dict[int, TaskResult]:
        """Return results for the completed stage."""
        with self._lock:
            return dict(self._results)

    def set_execution_config(self, shared_data: dict[str, Any], execution_id: str) -> None:
        """Set config for the current execution."""
        with self._lock:
            self._shared_data = shared_data
            self._execution_id = execution_id

    def run_pipeline(
        self,
        plan: PhysicalPlan,
        shared_data: dict[str, Any],
        execution_id: str,
        hints: ExecutionHint,
    ) -> list:
        """Run complete pipeline, blocking until done. Returns flattened results."""
        with self._lock:
            self._shared_data = shared_data
            self._execution_id = execution_id

        shards = _build_source_shards(plan.source_items)
        if not shards:
            return []

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
            self._start_stage(stage_label, tasks)

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

    def shutdown(self) -> None:
        """Signal workers to exit and terminate the worker group."""
        self._shutdown = True

        # Wait for coordinator thread to exit
        if self._coordinator_thread is not None:
            self._coordinator_thread.join(timeout=5.0)

        # Wait for workers to exit gracefully
        for f in self._worker_futures:
            with suppress(Exception):
                f.result(timeout=10.0)

        # Terminate worker group
        if self._worker_group is not None:
            with suppress(Exception):
                self._worker_group.shutdown()

        logger.info("Coordinator shutdown complete")

    # Legacy compatibility methods (used by tests)
    def set_shared_data(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._shared_data = data

    def get_shared_data(self) -> dict[str, Any]:
        with self._lock:
            return self._shared_data

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

    def start_stage(self, stage_name: str, tasks: list[ShardTask]) -> None:
        """Load a new stage's tasks into the queue (legacy compat)."""
        self._start_stage(stage_name, tasks)

    def check_heartbeats(self, timeout: float = 30.0) -> None:
        """Marks stale workers as FAILED, re-queues their in-flight tasks."""
        with self._lock:
            self._check_heartbeats_internal(timeout)

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

    Workers are created by the coordinator and poll continuously until
    receiving a SHUTDOWN signal. Each task includes the execution config
    (shared data, chunk prefix, execution ID), so workers can handle
    multiple executions without restart.
    """

    def __init__(self):
        self._shared_data: dict[str, Any] = {}
        self._shutdown_event = threading.Event()
        self._chunk_prefix: str = ""
        self._execution_id: str = ""

    def get_shared(self, name: str) -> Any:
        return self._shared_data[name]

    def start_polling(self, coordinator: ActorHandle) -> None:
        """Start polling loop. Called once per worker lifetime."""
        worker_id = f"worker-{id(self)}-{os.getpid()}"
        logger.info("[%s] Starting polling loop", worker_id)

        self._shutdown_event.clear()

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator, worker_id),
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            self._poll_loop(coordinator, worker_id)
        finally:
            self._shutdown_event.set()
            heartbeat_thread.join(timeout=5.0)
            logger.info("[%s] Polling loop ended", worker_id)

    def _heartbeat_loop(self, coordinator: ActorHandle, worker_id: str, interval: float = 5.0) -> None:
        logger.info("[%s] Heartbeat loop starting", worker_id)
        heartbeat_count = 0
        while not self._shutdown_event.is_set():
            try:
                coordinator.heartbeat.remote(worker_id)
                heartbeat_count += 1
                if heartbeat_count % 10 == 1:
                    logger.debug("[%s] Sent heartbeat #%d", worker_id, heartbeat_count)
            except Exception as e:
                logger.warning("[%s] Heartbeat failed: %s", worker_id, e)
            self._shutdown_event.wait(timeout=interval)
        logger.info("[%s] Heartbeat loop exiting after %d beats", worker_id, heartbeat_count)

    def _poll_loop(self, coordinator: ActorHandle, worker_id: str) -> None:
        """Pure polling loop. Exits only on SHUTDOWN signal."""
        loop_count = 0
        task_count = 0
        backoff = ExponentialBackoff(initial=0.05, maximum=1.0)

        while True:
            loop_count += 1
            if loop_count % 100 == 1:
                logger.debug("[%s] Poll iteration #%d, tasks completed: %d", worker_id, loop_count, task_count)

            response = coordinator.pull_task.remote(worker_id).result()

            # SHUTDOWN signal - exit cleanly
            if response == "SHUTDOWN":
                logger.info("[%s] Received SHUTDOWN", worker_id)
                break

            # No task available - backoff and retry
            if response is None:
                time.sleep(backoff.next_interval())
                continue

            backoff.reset()

            # Unpack task and config
            task, attempt, config = response

            # Update config for this execution
            self._shared_data = config["shared_data"]
            self._chunk_prefix = config["chunk_prefix"]
            self._execution_id = config["execution_id"]

            logger.info("[%s] Executing task for shard %d (attempt %d)", worker_id, task.shard_idx, attempt)
            try:
                result = self._execute_shard(task)
                logger.info("[%s] Task complete, reporting result for shard %d", worker_id, task.shard_idx)
                coordinator.report_result.remote(worker_id, task.shard_idx, attempt, result)
                task_count += 1
            except Exception as e:
                logger.error("Worker %s error on shard %d: %s", worker_id, task.shard_idx, e)
                import traceback

                coordinator.report_error.remote(
                    worker_id,
                    task.shard_idx,
                    "".join(traceback.format_exc()),
                )

    def _execute_shard(self, task: ShardTask) -> TaskResult:
        """Execute a stage's operations on a single shard.

        Returns list of TaskResult.
        """
        _shard_ctx_var.set(self)

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

        results: list[TaskResult] = []
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

    def shutdown(self) -> None:
        self._shutdown_event.set()


def _regroup_result_refs(
    result_refs: dict[int, TaskResult],
    input_shard_count: int,
    output_shard_count: int | None = None,
) -> list[Shard]:
    """Regroup worker output refs by output shard index without loading data."""
    output_by_shard: dict[int, list[DiskChunk]] = defaultdict(list)

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

    Creates a coordinator actor on __enter__ which owns and manages the worker
    pool. Workers persist across pipeline stages and execute() calls, allowing
    cached state (tokenizers, models) to be reused. Shared data broadcast via
    put() is delivered to workers with each task.

    Args:
        client: The fray client to use. If None, auto-detects using current_client().
        num_workers: Number of workers. If None, defaults to os.cpu_count() for LocalClient,
            or 128 for distributed clients.
        resources: Resource config per worker.
        chunk_storage_prefix: Storage prefix for intermediate chunks. If None, defaults
            to MARIN_PREFIX/tmp/zephyr or /tmp/zephyr.
    """

    client: Client | None = None
    num_workers: int | None = None
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="16g"))
    chunk_storage_prefix: str | None = None

    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    _coordinator: ActorHandle | None = field(default=None, repr=False)
    _coordinator_group: Any = field(default=None, repr=False)
    _instance_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8], repr=False)

    def __post_init__(self):
        if self.client is None:
            from fray.v2.client import current_client

            self.client = current_client()

        if self.num_workers is None:
            from fray.v2.local_backend import LocalClient

            if isinstance(self.client, LocalClient):
                self.num_workers = os.cpu_count() or 1
            else:
                self.num_workers = 128

        if self.chunk_storage_prefix is None:
            marin_prefix = os.environ.get("MARIN_PREFIX", "")
            if marin_prefix:
                self.chunk_storage_prefix = f"{marin_prefix}/tmp/zephyr"
            else:
                self.chunk_storage_prefix = "/tmp/zephyr"

    def put(self, name: str, obj: Any) -> None:
        """Register shared data to broadcast to all workers.

        Must be called before execute(). The object must be picklable.
        Workers access it via shard_ctx().get_shared(name).
        """
        self._shared_data[name] = obj

    def execute(
        self,
        dataset: Dataset,
        hints: ExecutionHint = ExecutionHint(),
        verbose: bool = False,
        dry_run: bool = False,
    ) -> Sequence:
        """Execute a dataset pipeline on the worker pool.

        Workers persist across execute() calls, so cached state (tokenizers,
        models) will be reused.
        """
        plan = compute_plan(dataset, hints)
        if dry_run:
            _print_plan(dataset.operations, plan)
            return []

        execution_id = _generate_execution_id()

        try:
            # Ensure coordinator is initialized
            self._ensure_coordinator()

            # Run pipeline on coordinator (blocking call)
            results = self._coordinator.run_pipeline.remote(plan, self._shared_data, execution_id, hints).result()

            return results

        except Exception:
            # Re-raise without cleanup - workers will be reused
            raise

        finally:
            # Clean up chunks for this execution only (workers persist)
            _cleanup_execution(self.chunk_storage_prefix, execution_id)

    def _ensure_coordinator(self) -> None:
        """Create coordinator and workers if not already initialized."""
        if self._coordinator is not None:
            return

        # Create coordinator actor with high max_concurrency to allow
        # workers to call pull_task/report_result while run_pipeline blocks
        coordinator_resources = ResourceConfig(cpu=1, ram="2g", max_concurrency=100)
        self._coordinator_group = self.client.create_actor_group(
            ZephyrCoordinator,
            name=f"zephyr-coordinator-{self._instance_id}",
            count=1,
            resources=coordinator_resources,
        )
        self._coordinator = self._coordinator_group.wait_ready()[0]

        # Create worker group using the context's client (avoids auto-detection issues)
        worker_group = self.client.create_actor_group(
            ZephyrWorker,
            name=f"zephyr-workers-{self._instance_id}",
            count=self.num_workers,
            resources=self.resources,
        )

        # Pass worker group to coordinator - it will poll discover_new() for new workers
        self._coordinator.initialize.remote(
            self.chunk_storage_prefix,
            self._coordinator,
            worker_group,
        ).result()

        logger.info("ZephyrContext initialized with coordinator and %d workers", self.num_workers)

    def shutdown(self) -> None:
        """Shutdown coordinator and all workers."""
        if self._coordinator is not None:
            # Tell coordinator to shutdown workers gracefully
            with suppress(Exception):
                self._coordinator.shutdown.remote().result()

            # Terminate coordinator actor group
            if self._coordinator_group is not None:
                with suppress(Exception):
                    self._coordinator_group.shutdown()

            self._coordinator = None
            self._coordinator_group = None

    def __enter__(self) -> ZephyrContext:
        # Eagerly initialize coordinator and workers on context entry
        self._ensure_coordinator()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


def _reshard_refs(shards: list[Shard], num_shards: int) -> list[Shard]:
    """Reshard shard refs by output shard index without loading data."""
    output_by_shard: dict[int, list[Chunk]] = defaultdict(list)
    output_idx = 0
    for shard in shards:
        for chunk in shard.chunks:
            output_idx = (output_idx + 1) % num_shards
            output_by_shard[output_idx].append(chunk)
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


_default_zephyr_context: ContextVar[ZephyrContext | None] = ContextVar("zephyr_context", default=None)


@contextmanager
def default_zephyr_context(ctx: ZephyrContext):
    """Set the default ZephyrContext for the duration of a with-block."""
    old = _default_zephyr_context.get()
    _default_zephyr_context.set(ctx)
    try:
        yield ctx
    finally:
        _default_zephyr_context.set(old)


def get_default_zephyr_context() -> ZephyrContext:
    """Get the current default ZephyrContext, creating one if unset."""
    ctx = _default_zephyr_context.get()
    if ctx is None:
        ctx = ZephyrContext()
    return ctx


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
