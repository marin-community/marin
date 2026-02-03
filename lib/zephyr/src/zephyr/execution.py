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
    aux_shards: dict[int, list[Chunk]]


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
    """Central coordinator actor. Workers pull tasks from it.

    All state mutations happen through actor method calls, which are serialized
    by the actor framework — no concurrent access, no locks needed.
    """

    def __init__(self):
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
        self._done: bool = False
        self._fatal_error: str | None = None
        self._chunk_prefix: str = ""
        self._execution_id: str = ""

    # These are actors in order to allow remotely pushing the shared data context to the coordinator
    def set_shared_data(self, data: dict[str, Any]) -> None:
        self._shared_data = data

    def get_shared_data(self) -> dict[str, Any]:
        return self._shared_data

    def set_chunk_config(self, prefix: str, execution_id: str) -> None:
        """Configure chunk storage for this execution."""
        self._chunk_prefix = prefix
        self._execution_id = execution_id

    def get_chunk_config(self) -> dict:
        """Return chunk storage configuration for workers."""
        return {
            "prefix": self._chunk_prefix,
            "execution_id": self._execution_id,
        }

    def start_stage(self, stage_name: str, tasks: list[ShardTask]) -> None:
        """Load a new stage's tasks into the queue."""
        self._task_queue = deque(tasks)
        self._results = defaultdict(list)
        self._stage_name = stage_name
        self._total_shards = len(tasks)
        self._completed_shards = 0
        self._retries = 0
        self._in_flight = {}
        self._task_attempts = {task.shard_idx: 0 for task in tasks}
        self._done = False
        self._fatal_error = None

    def pull_task(self, worker_id: str) -> tuple[ShardTask, int] | None:
        """Called by workers to get next task. Returns (task, attempt) or None."""
        self._last_seen[worker_id] = time.monotonic()
        self._worker_states[worker_id] = WorkerState.READY

        if self._done or self._fatal_error:
            return None
        if not self._task_queue:
            return None

        task = self._task_queue.popleft()
        attempt = self._task_attempts[task.shard_idx]
        self._in_flight[worker_id] = (task, attempt)
        self._worker_states[worker_id] = WorkerState.BUSY
        return (task, attempt)

    def report_result(self, worker_id: str, shard_idx: int, attempt: int, result: TaskResult) -> None:
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
        self._last_seen[worker_id] = time.monotonic()
        self._in_flight.pop(worker_id, None)
        self._fatal_error = error_info
        self._worker_states[worker_id] = WorkerState.DEAD

    def heartbeat(self, worker_id: str) -> None:
        self._last_seen[worker_id] = time.monotonic()

    def check_heartbeats(self, timeout: float = 30.0) -> None:
        """Marks stale workers as FAILED, re-queues their in-flight tasks."""
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

    def get_status(self) -> JobStatus:
        return JobStatus(
            stage=self._stage_name,
            completed=self._completed_shards,
            total=self._total_shards,
            retries=self._retries,
            in_flight=len(self._in_flight),
            queue_depth=len(self._task_queue),
            done=self._done,
            fatal_error=self._fatal_error,
            workers={
                wid: {
                    "state": state.value,
                    "last_seen_ago": time.monotonic() - self._last_seen.get(wid, 0),
                }
                for wid, state in self._worker_states.items()
            },
        )

    def collect_results(self) -> dict[int, TaskResult]:
        """Return results for the completed stage."""
        return self._results

    def signal_done(self) -> None:
        """Signal workers that no more stages will be submitted."""
        self._done = True

    def reset(self) -> None:
        """Reset coordinator state for reuse across multiple execute() calls."""
        self._done = False
        self._fatal_error = None


# ---------------------------------------------------------------------------
# ZephyrWorker
# ---------------------------------------------------------------------------


class ZephyrWorker:
    """Long-lived worker actor. Pulls tasks from coordinator, executes, reports."""

    def __init__(self):
        self._shared_data: dict[str, Any] = {}
        self._shutdown_event = threading.Event()
        self._chunk_prefix: str = ""
        self._execution_id: str = ""

    def get_shared(self, name: str) -> Any:
        return self._shared_data[name]

    def run_loop(self, coordinator: ActorHandle) -> None:
        """Main worker loop. Pulls tasks from coordinator until done."""
        worker_id = f"worker-{id(self)}-{os.getpid()}"

        # Clear shutdown event in case this worker is being reused for a new stage
        self._shutdown_event.clear()

        logger.info(f"[{worker_id}] run_loop starting, fetching shared data...")

        try:
            self._shared_data = coordinator.get_shared_data.remote().result()
            logger.info(f"[{worker_id}] Got shared data, fetching chunk config...")
        except Exception as e:
            logger.error(f"[{worker_id}] Failed to get shared data: {e}")
            raise

        try:
            chunk_config = coordinator.get_chunk_config.remote().result()
            self._chunk_prefix = chunk_config["prefix"]
            self._execution_id = chunk_config["execution_id"]
            logger.info(f"[{worker_id}] Got chunk config, starting heartbeat thread...")
        except Exception as e:
            logger.error(f"[{worker_id}] Failed to get chunk config: {e}")
            raise

        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator, worker_id),
            daemon=True,
        )
        heartbeat_thread.start()
        logger.info(f"[{worker_id}] Heartbeat thread started, entering work loop...")

        try:
            self._work_loop(coordinator, worker_id)
        finally:
            self._shutdown_event.set()
            heartbeat_thread.join(timeout=5.0)
            logger.info(f"[{worker_id}] run_loop finished")

    def _heartbeat_loop(self, coordinator: ActorHandle, worker_id: str, interval: float = 5.0) -> None:
        logger.info(f"[{worker_id}] Heartbeat loop starting...")
        heartbeat_count = 0
        while not self._shutdown_event.is_set():
            try:
                coordinator.heartbeat.remote(worker_id)
                heartbeat_count += 1
                if heartbeat_count % 10 == 1:  # Log every 10th heartbeat
                    logger.info(f"[{worker_id}] Sent heartbeat #{heartbeat_count}")
            except Exception as e:
                logger.warning(f"[{worker_id}] Heartbeat failed: {e}")
            self._shutdown_event.wait(timeout=interval)
        logger.info(f"[{worker_id}] Heartbeat loop exiting after {heartbeat_count} beats")

    def _work_loop(self, coordinator: ActorHandle, worker_id: str) -> None:
        loop_count = 0
        task_count = 0
        logger.info(f"[{worker_id}] _work_loop starting, shutdown_event is set: {self._shutdown_event.is_set()}")
        while not self._shutdown_event.is_set():
            loop_count += 1
            if loop_count % 100 == 1:
                logger.info(f"[{worker_id}] Work loop iteration #{loop_count}, tasks completed: {task_count}")

            logger.debug(f"[{worker_id}] Pulling task...")
            task_and_attempt = coordinator.pull_task.remote(worker_id).result()
            logger.debug(f"[{worker_id}] Pull returned: {task_and_attempt is not None}")

            if task_and_attempt is None:
                status = coordinator.get_status.remote().result()
                logger.info(f"[{worker_id}] No task available, status: {status}")
                if status.done:
                    logger.info(f"[{worker_id}] Stage done (done={status.done}, error={status.fatal_error}), exiting")
                    break
                time.sleep(0.1)
                continue

            task, attempt = task_and_attempt
            logger.info(f"[{worker_id}] Executing task for shard {task.shard_idx} (attempt {attempt})")
            try:
                result = self._execute_shard(task)
                logger.info(f"[{worker_id}] Task complete, reporting result for shard {task.shard_idx}")
                coordinator.report_result.remote(worker_id, task.shard_idx, attempt, result)
                logger.info(f"[{worker_id}] Result reported for shard {task.shard_idx}")
                task_count += 1
            except Exception as e:
                logger.error(f"Worker {worker_id} error on shard {task.shard_idx}: {e}")
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


# ---------------------------------------------------------------------------
# _run_stage_on_coordinator — shared poll-loop for executing a stage
# ---------------------------------------------------------------------------


def _run_stage_on_coordinator(
    coordinator: ActorHandle,
    stage_name: str,
    tasks: list[ShardTask],
) -> dict[int, list[TaskResult]]:
    """Submit tasks to coordinator, poll until complete, return raw results."""
    coordinator.start_stage.remote(stage_name, tasks).result()
    last_log_completed = -1
    while True:
        coordinator.check_heartbeats.remote()
        status = coordinator.get_status.remote().result()
        if status.fatal_error:
            raise ZephyrWorkerError(status.fatal_error)
        if status.completed != last_log_completed:
            logger.info("[%s] %d/%d tasks completed", stage_name, status.completed, status.total)
            last_log_completed = status.completed
        if status.completed >= status.total:
            break
        time.sleep(0.1)
    return coordinator.collect_results.remote().result()


def _regroup_result_refs(
    result_refs: dict[int, TaskResult],
    input_shard_count: int,
) -> list[Shard]:
    """Regroup worker output refs by output shard index without loading data."""
    output_by_shard: dict[int, list[DiskChunk]] = defaultdict(list)

    for _input_idx, result in result_refs.items():
        for chunk in result.chunks:
            output_by_shard[chunk.target_shard].append(chunk.data)

    num_output = max(max(output_by_shard.keys(), default=0) + 1, input_shard_count)
    return [Shard(chunks=output_by_shard.get(idx, [])) for idx in range(num_output)]


@dataclass
class ZephyrContext:
    """Execution context for Zephyr pipelines.

    Creates and manages a coordinator actor and a pool of long-lived worker
    actors. Workers persist across pipeline stages, allowing cached state
    (tokenizers, models) to be reused. Shared data broadcast via put() is
    delivered to workers through the coordinator.
    """

    client: Client
    num_workers: int
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="16g"))
    max_parallelism: int = 1024
    chunk_storage_prefix: str | None = None

    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    _coordinator: ActorHandle | None = field(default=None, repr=False)
    _workers: list[ActorHandle] = field(default_factory=list, repr=False)
    _worker_futures: list = field(default_factory=list, repr=False)
    _instance_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8], repr=False)
    _coordinator_group: Any = field(default=None, repr=False)
    _worker_group: Any = field(default=None, repr=False)

    def __post_init__(self):
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
        """Execute a dataset pipeline on the worker pool."""
        plan = compute_plan(dataset, hints)
        if dry_run:
            _print_plan(dataset.operations, plan)
            return []

        execution_id = _generate_execution_id()

        try:
            coordinator = self._get_or_create_coordinator()
            coordinator.set_chunk_config.remote(self.chunk_storage_prefix, execution_id).result()
            coordinator.set_shared_data.remote(self._shared_data).result()

            # Reset coordinator done flag in case this context is being reused
            coordinator.reset.remote().result()

            # Start run loops for whatever workers are ready now
            self._worker_futures = [w.run_loop.remote(coordinator) for w in self._workers]

            shards = _build_source_shards(plan.source_items)

            for stage_idx, stage in enumerate(plan.stages):
                # Pick up any newly-available workers and start their run loops
                for new_worker in self._discover_workers():
                    self._worker_futures.append(new_worker.run_loop.remote(coordinator))
                stage_label = f"stage{stage_idx}-{stage.stage_name(max_length=40)}"

                if stage.stage_type == StageType.RESHARD:
                    shards = _reshard_refs(shards, stage.output_shards or len(shards))
                    continue

                aux_per_shard = _compute_join_aux(
                    stage.operations, shards, coordinator, self.chunk_storage_prefix, hints, execution_id
                )

                tasks = _compute_tasks_from_shards(shards, stage, hints, aux_per_shard, stage_name=stage_label)
                logger.info("Starting stage %s with %d tasks", stage_label, len(tasks))

                result_refs = _run_stage_on_coordinator(coordinator, stage_label, tasks)
                shards = _regroup_result_refs(result_refs, len(shards))

            coordinator.signal_done.remote().result()

            for f in self._worker_futures:
                with suppress(Exception):
                    f.result(timeout=10.0)
            self._worker_futures = []
            flat_result = []
            for shard in shards:
                for chunk in shard.chunks:
                    flat_result.extend(list(chunk))
            return flat_result

        finally:
            _cleanup_execution(self.chunk_storage_prefix, execution_id)

    def _get_or_create_coordinator(self) -> ActorHandle:
        if self._coordinator is None:
            coordinator_resources = ResourceConfig(cpu=1, ram="2g")
            coordinator_group = self.client.create_actor_group(
                ZephyrCoordinator,
                name=f"zephyr-controller-{self._instance_id}",
                count=1,
                resources=coordinator_resources,
            )
            self._coordinator = coordinator_group.wait_ready()[0]
            self._coordinator_group = coordinator_group

            worker_group = self.client.create_actor_group(
                ZephyrWorker,
                name=f"zephyr-worker-{self._instance_id}",
                count=self.num_workers,
                resources=self.resources,
            )
            self._workers = worker_group.wait_ready()
            self._worker_group = worker_group
        return self._coordinator

    def _discover_workers(self) -> list[ActorHandle]:
        """Discover newly available workers and return them."""
        group = self._worker_group
        if group is None:
            return []
        new_handles = group.discover_new()
        if new_handles:
            self._workers.extend(new_handles)
            logger.info("Discovered %d new workers, total now %d", len(new_handles), len(self._workers))
        return new_handles

    def shutdown(self) -> None:
        # TODO: terminate the jobs isntead of the signals
        if self._coordinator is not None:
            with suppress(Exception):
                self._coordinator.signal_done.remote()
            for w in self._workers:
                with suppress(Exception):
                    w.shutdown.remote()

            for f in self._worker_futures:
                with suppress(Exception):
                    f.result(timeout=5.0)

            if self._coordinator_group is not None:
                with suppress(Exception):
                    self._coordinator_group.shutdown()
            if self._worker_group is not None:
                with suppress(Exception):
                    self._worker_group.shutdown()

            self._coordinator = None
            self._workers = []
            self._worker_futures = []
            self._coordinator_group = None
            self._worker_group = None

    def __enter__(self) -> ZephyrContext:
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
    aux_per_shard: list[dict[int, list[Chunk]]] | None = None,
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


def _compute_join_aux(
    operations: list[PhysicalOp],
    shard_refs: list[Shard],
    coordinator: ActorHandle,
    chunk_storage_prefix: str,
    hints: ExecutionHint,
    execution_id: str,
) -> list[dict[int, list[list[DiskChunk]]]] | None:
    """Execute right sub-plans for join operations, returning aux refs per shard."""
    all_right_shard_refs: dict[int, list[list[Chunk]]] = {}

    for i, op in enumerate(operations):
        if not isinstance(op, Join) or op.right_plan is None:
            continue

        right_refs = _build_source_shards(op.right_plan.source_items)
        # now run and produce the results for this

        for stage_idx, right_stage in enumerate(op.right_plan.stages):
            if right_stage.stage_type == StageType.RESHARD:
                right_refs = _reshard_refs(right_refs, right_stage.output_shards or len(right_refs))
                continue

            join_stage_label = f"join-right-{i}-stage{stage_idx}"
            right_tasks = _compute_tasks_from_shards(right_refs, right_stage, hints, stage_name=join_stage_label)
            raw = _run_stage_on_coordinator(coordinator, join_stage_label, right_tasks)
            right_refs = _regroup_result_refs(raw, len(right_refs))

        if len(shard_refs) != len(right_refs):
            raise ValueError(
                f"Sorted merge join requires equal shard counts. "
                f"Left has {len(shard_refs)} shards, right has {len(right_refs)} shards."
            )
        all_right_shard_refs[i] = right_refs

    if not all_right_shard_refs:
        return None

    return [
        {op_idx: [right_refs.shards[shard_idx]] for op_idx, right_refs in all_right_shard_refs.items()}
        for shard_idx in range(len(shard_refs))
    ]


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
        from fray.v2.client import current_client

        ctx = ZephyrContext(client=current_client(), num_workers=os.cpu_count() or 1)
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


class Backend:
    """Shim preserving the old Backend.execute() calling convention.

    Delegates to ZephyrContext under the hood. Callers can continue writing
    ``Backend.execute(pipeline)`` during the v1 -> v2 migration.
    """

    @staticmethod
    def execute(
        dataset: Dataset,
        hints: ExecutionHint = ExecutionHint(),
        verbose: bool = False,
        max_parallelism: int = 1024,
        dry_run: bool = False,
    ) -> Sequence:
        ctx = get_default_zephyr_context()
        ctx.max_parallelism = max_parallelism
        return ctx.execute(dataset, hints=hints, verbose=verbose, dry_run=dry_run)
