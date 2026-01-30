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

Replaces the stateless Backend.execute() dispatch with long-lived worker actors
managed through a coordinator. Workers pull tasks from the coordinator, execute
shard operations, and report results back. This enables persistent worker state
(caches, loaded models), transient error recovery, and backend-agnostic dispatch
via fray v2's Client protocol.
"""

from __future__ import annotations

import enum
import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fray.v2 import ActorHandle, Client, ResourceConfig
from zephyr.dataset import Dataset
from zephyr.plan import (
    ChunkHeader,
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


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
    shard_data: Any  # serializable shard description (list of chunk data)
    operations: list[PhysicalOp]
    aux_data: dict[int, list[list]] | None = None  # op_index -> list of right shard chunk lists


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
# _SerializableShard — lightweight wrapper matching the Shard interface
# ---------------------------------------------------------------------------


class _SerializableShard:
    """Wraps chunk data lists to match the interface run_stage expects.

    run_stage does `iter(ctx.shard)` for flat iteration and
    `ctx.shard.iter_chunks()` for Reduce operations.
    """

    def __init__(self, chunks: list[list]):
        self._chunks = chunks

    def iter_chunks(self):
        yield from self._chunks

    def __iter__(self):
        for chunk in self._chunks:
            yield from chunk


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def _is_transient_error(error: Exception) -> bool:
    """Classify whether an error is transient (recoverable) or permanent.

    We enumerate transient cases and treat everything else as permanent.
    """
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True

    ray_transient = {
        "NodeDiedError",
        "ActorDiedError",
        "OwnerDiedError",
        "WorkerCrashedError",
        "RayActorError",
    }
    if type(error).__name__ in ray_transient:
        return True

    cause = error.__cause__
    if cause is not None and type(cause).__name__ in ray_transient:
        return True

    return False


def _serialize_error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# ZephyrCoordinator
# ---------------------------------------------------------------------------


class ZephyrCoordinator:
    """Central coordinator actor. Workers pull tasks from it.

    All state mutations happen through actor method calls, which are serialized
    by the actor framework — no concurrent access, no locks needed.
    """

    def __init__(self):
        self._task_queue: deque[ShardTask] = deque()
        self._results: dict[int, list] = defaultdict(list)
        self._worker_states: dict[str, WorkerState] = {}
        self._last_seen: dict[str, float] = {}
        self._shared_data: dict[str, Any] = {}
        self._stage_name: str = ""
        self._total_shards: int = 0
        self._completed_shards: int = 0
        self._retries: int = 0
        self._in_flight: dict[str, ShardTask] = {}
        self._done: bool = False
        self._fatal_error: str | None = None

    def set_shared_data(self, data: dict[str, Any]) -> None:
        self._shared_data = data

    def get_shared_data(self) -> dict[str, Any]:
        return self._shared_data

    def start_stage(self, stage_name: str, tasks: list[ShardTask]) -> None:
        """Load a new stage's tasks into the queue."""
        self._task_queue = deque(tasks)
        self._results = defaultdict(list)
        self._stage_name = stage_name
        self._total_shards = len(tasks)
        self._completed_shards = 0
        self._retries = 0
        self._in_flight = {}
        self._done = False
        self._fatal_error = None

    def pull_task(self, worker_id: str) -> ShardTask | None:
        """Called by workers to get next task. Returns None when no work available."""
        self._last_seen[worker_id] = time.monotonic()
        self._worker_states[worker_id] = WorkerState.READY

        if self._done or self._fatal_error:
            return None
        if not self._task_queue:
            return None

        task = self._task_queue.popleft()
        self._in_flight[worker_id] = task
        self._worker_states[worker_id] = WorkerState.BUSY
        return task

    def report_result(self, worker_id: str, shard_idx: int, result: list) -> None:
        self._last_seen[worker_id] = time.monotonic()
        self._results[shard_idx].extend(result)
        self._completed_shards += 1
        self._in_flight.pop(worker_id, None)
        self._worker_states[worker_id] = WorkerState.READY

    def report_error(self, worker_id: str, shard_idx: int, error_info: str, is_transient: bool) -> None:
        """Worker reports a task failure.

        Transient errors re-queue the shard. Application errors set fatal_error
        so the user process can raise immediately on next poll.
        """
        self._last_seen[worker_id] = time.monotonic()
        task = self._in_flight.pop(worker_id, None)

        if is_transient:
            if task is not None:
                self._task_queue.append(task)
                self._retries += 1
            self._worker_states[worker_id] = WorkerState.READY
        else:
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
                task = self._in_flight.pop(worker_id, None)
                if task is not None:
                    self._task_queue.append(task)
                    self._retries += 1

    def get_status(self) -> dict:
        return {
            "stage": self._stage_name,
            "completed": self._completed_shards,
            "total": self._total_shards,
            "retries": self._retries,
            "in_flight": len(self._in_flight),
            "queue_depth": len(self._task_queue),
            "done": self._done,
            "fatal_error": self._fatal_error,
            "workers": {
                wid: {
                    "state": state.value,
                    "last_seen_ago": time.monotonic() - self._last_seen.get(wid, 0),
                }
                for wid, state in self._worker_states.items()
            },
        }

    def collect_results(self) -> dict[int, list]:
        """Return results for the completed stage."""
        return dict(self._results)

    def signal_done(self) -> None:
        """Signal workers that no more stages will be submitted."""
        self._done = True


# ---------------------------------------------------------------------------
# ZephyrWorker
# ---------------------------------------------------------------------------


class ZephyrWorker:
    """Long-lived worker actor. Pulls tasks from coordinator, executes, reports."""

    def __init__(self):
        self._shared_data: dict[str, Any] = {}
        self._shutdown_event = threading.Event()

    def get_shared(self, name: str) -> Any:
        return self._shared_data[name]

    def run_loop(self, coordinator: ActorHandle) -> None:
        """Main worker loop. Pulls tasks from coordinator until done."""
        self._shared_data = coordinator.get_shared_data.remote().result()
        worker_id = f"worker-{id(self)}-{os.getpid()}"

        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator, worker_id),
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            self._work_loop(coordinator, worker_id)
        finally:
            self._shutdown_event.set()
            heartbeat_thread.join(timeout=5.0)

    def _heartbeat_loop(self, coordinator: ActorHandle, worker_id: str, interval: float = 5.0) -> None:
        while not self._shutdown_event.is_set():
            with suppress(Exception):
                coordinator.heartbeat.remote(worker_id)
            self._shutdown_event.wait(timeout=interval)

    def _work_loop(self, coordinator: ActorHandle, worker_id: str) -> None:
        while not self._shutdown_event.is_set():
            task = coordinator.pull_task.remote(worker_id).result()
            if task is None:
                status = coordinator.get_status.remote().result()
                if status.get("done") or status.get("fatal_error"):
                    break
                time.sleep(0.1)
                continue

            try:
                result = self._execute_shard(task)
                coordinator.report_result.remote(worker_id, task.shard_idx, result)
            except Exception as e:
                coordinator.report_error.remote(
                    worker_id,
                    task.shard_idx,
                    _serialize_error(e),
                    is_transient=_is_transient_error(e),
                )

    def _execute_shard(self, task: ShardTask) -> list[tuple[dict, list]]:
        """Execute a stage's operations on a single shard.

        Returns list of (header_dict, data) pairs — one per output chunk.
        """
        _shard_ctx_var.set(self)

        shard = _SerializableShard(task.shard_data)

        # Reconstruct aux shards for joins
        aux_shards: dict[int, list[Any]] = {}
        if task.aux_data:
            for op_idx, right_chunk_lists in task.aux_data.items():
                aux_shards[op_idx] = [_SerializableShard(chunks) for chunks in right_chunk_lists]

        stage_ctx = StageContext(
            shard=shard,
            shard_idx=task.shard_idx,
            total_shards=task.total_shards,
            chunk_size=task.chunk_size,
            aux_shards=aux_shards,
        )

        results: list[tuple[dict, list]] = []
        current_header: ChunkHeader | None = None
        for item in run_stage(stage_ctx, task.operations):
            if isinstance(item, ChunkHeader):
                current_header = item
            else:
                assert current_header is not None
                results.append(
                    (
                        {"shard_idx": current_header.shard_idx, "count": current_header.count},
                        item,
                    )
                )

        return results

    def shutdown(self) -> None:
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# ZephyrContext — user-facing entry point
# ---------------------------------------------------------------------------


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
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    max_parallelism: int = 1024

    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    _coordinator: ActorHandle | None = field(default=None, repr=False)
    _workers: list[ActorHandle] = field(default_factory=list, repr=False)
    _worker_futures: list = field(default_factory=list, repr=False)

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

        coordinator = self._get_or_create_coordinator()
        coordinator.set_shared_data.remote(self._shared_data)

        # Start worker run loops (non-blocking via .remote())
        self._worker_futures = [w.run_loop.remote(coordinator) for w in self._workers]

        # Execute stages sequentially
        shards = _build_source_shards(plan.source_items)
        for stage in plan.stages:
            if stage.stage_type == StageType.RESHARD:
                shards = _reshard(shards, stage.output_shards or len(shards))
                continue

            # Handle joins: execute right sub-plans first
            aux_per_shard = _compute_join_aux(stage.operations, shards, self, hints)

            tasks = _shards_to_tasks(shards, stage, hints, aux_per_shard)
            coordinator.start_stage.remote(stage.stage_name(), tasks)

            # Poll until stage completes
            while True:
                coordinator.check_heartbeats.remote()
                status = coordinator.get_status.remote().result()

                if status.get("fatal_error"):
                    raise ZephyrWorkerError(status["fatal_error"])
                if status["completed"] >= status["total"]:
                    break
                time.sleep(0.1)

            raw_results = coordinator.collect_results.remote().result()
            shards = _results_to_shards(raw_results, len(shards))

        # Signal workers we're done so they exit their run_loop
        coordinator.signal_done.remote()

        # Wait for workers to finish so the context can be reused for the next execute()
        for f in self._worker_futures:
            with suppress(Exception):
                f.result(timeout=10.0)
        self._worker_futures = []
        # Reset coordinator so next execute() creates fresh actors
        self._coordinator = None
        self._workers = []

        return _materialize(shards)

    def _get_or_create_coordinator(self) -> ActorHandle:
        if self._coordinator is None:
            self._coordinator = self.client.create_actor(
                ZephyrCoordinator,
                name="zephyr-controller",
                resources=ResourceConfig(),
            )
            group = self.client.create_actor_group(
                ZephyrWorker,
                name="zephyr-worker",
                count=self.num_workers,
                resources=self.resources,
            )
            self._workers = group.wait_ready()
        return self._coordinator

    def shutdown(self) -> None:
        if self._coordinator is not None:
            self._coordinator.signal_done.remote()
            for w in self._workers:
                with suppress(Exception):
                    w.shutdown.remote()
            # Wait briefly for worker futures to complete
            for f in self._worker_futures:
                with suppress(Exception):
                    f.result(timeout=5.0)
            self._coordinator = None
            self._workers = []
            self._worker_futures = []

    def __enter__(self) -> ZephyrContext:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_source_shards(source_items: list[SourceItem]) -> list[list[list]]:
    """Build shard data from source items.

    Returns a list of shards, where each shard is a list of chunks,
    and each chunk is a list of items. For source items, each item
    becomes a single-element chunk (matching current Backend behavior).
    """
    items_by_shard: dict[int, list] = defaultdict(list)
    for item in source_items:
        items_by_shard[item.shard_idx].append(item.data)

    num_shards = max(items_by_shard.keys()) + 1 if items_by_shard else 0
    shards: list[list[list]] = []
    for i in range(num_shards):
        chunks = [[item] for item in items_by_shard.get(i, [])]
        shards.append(chunks)

    return shards


def _reshard(shards: list[list[list]], num_shards: int) -> list[list[list]]:
    """Redistribute chunks across target number of shards."""
    if not shards:
        return []

    all_chunks = [chunk for shard in shards for chunk in shard]
    if not all_chunks:
        return []

    chunk_groups = np.array_split(all_chunks, num_shards)
    return [list(group) for group in chunk_groups if len(group) > 0]


def _shards_to_tasks(
    shards: list[list[list]],
    stage,
    hints: ExecutionHint,
    aux_per_shard: list[dict[int, list[list[list]]]] | None = None,
) -> list[ShardTask]:
    """Convert shards into ShardTasks for the coordinator."""
    total = len(shards)
    tasks = []
    for i, shard_chunks in enumerate(shards):
        aux_data = None
        if aux_per_shard and aux_per_shard[i]:
            aux_data = aux_per_shard[i]

        tasks.append(
            ShardTask(
                shard_idx=i,
                total_shards=total,
                chunk_size=hints.chunk_size,
                shard_data=shard_chunks,
                operations=stage.operations,
                aux_data=aux_data,
            )
        )
    return tasks


def _compute_join_aux(
    operations: list[PhysicalOp],
    shards: list[list[list]],
    ctx: ZephyrContext,
    hints: ExecutionHint,
) -> list[dict[int, list[list[list]]]] | None:
    """Execute right sub-plans for join operations, returning aux data per shard."""
    all_right_shards: dict[int, list[list[list]]] = {}

    for i, op in enumerate(operations):
        if isinstance(op, Join) and op.right_plan is not None:
            right_source = _build_source_shards(op.right_plan.source_items)
            # Execute right plan stages
            for right_stage in op.right_plan.stages:
                if right_stage.stage_type == StageType.RESHARD:
                    right_source = _reshard(right_source, right_stage.output_shards or len(right_source))
                    continue
                # For right sub-plans, execute directly via coordinator
                right_tasks = _shards_to_tasks(right_source, right_stage, hints)
                ctx._coordinator.start_stage.remote(f"join-right-{i}", right_tasks)  # type: ignore[union-attr]

                while True:
                    ctx._coordinator.check_heartbeats.remote()  # type: ignore[union-attr]
                    status = ctx._coordinator.get_status.remote().result()  # type: ignore[union-attr]
                    if status.get("fatal_error"):
                        raise ZephyrWorkerError(status["fatal_error"])
                    if status["completed"] >= status["total"]:
                        break
                    time.sleep(0.1)

                raw = ctx._coordinator.collect_results.remote().result()  # type: ignore[union-attr]
                right_source = _results_to_shards(raw, len(right_source))

            if len(shards) != len(right_source):
                raise ValueError(
                    f"Sorted merge join requires equal shard counts. "
                    f"Left has {len(shards)} shards, right has {len(right_source)} shards."
                )
            all_right_shards[i] = right_source

    if not all_right_shards:
        return None

    return [
        {op_idx: [right_shards[shard_idx]] for op_idx, right_shards in all_right_shards.items()}
        for shard_idx in range(len(shards))
    ]


def _results_to_shards(raw_results: dict[int, list], input_shard_count: int) -> list[list[list]]:
    """Convert coordinator results back into shard format.

    raw_results maps input_shard_idx -> list of (header_dict, data) tuples.
    Each header_dict contains the actual output shard_idx (which may differ
    from the input shard_idx when a Scatter operation redistributes items).
    """
    # Regroup by the header's shard_idx (the output shard)
    output_by_shard: dict[int, list[list]] = defaultdict(list)
    for _input_idx, result_pairs in raw_results.items():
        for header, data in result_pairs:
            output_shard = header["shard_idx"]
            output_by_shard[output_shard].append(data)

    num_output_shards = input_shard_count
    if output_by_shard:
        max_idx = max(output_by_shard.keys())
        if max_idx >= num_output_shards:
            num_output_shards = max_idx + 1

    shards: list[list[list]] = []
    for idx in range(num_output_shards):
        shards.append(output_by_shard.get(idx, []))
    return shards


def _materialize(shards: list[list[list]]) -> list:
    """Flatten all shard chunks into a single list of results."""
    results = []
    for shard_chunks in shards:
        for chunk in shard_chunks:
            results.extend(chunk)
    return results


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
