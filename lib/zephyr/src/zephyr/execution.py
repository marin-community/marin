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

import atexit
import enum
import logging
import os
import pickle
import threading
import time
import uuid
import weakref
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import fsspec
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
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)

# Track live ZephyrContext instances for atexit cleanup via weak references,
# so contexts that are garbage-collected don't prevent the process from exiting.
_live_context_refs: list[weakref.ref[ZephyrContext]] = []


def _atexit_cleanup() -> None:
    """Kill orphaned actor jobs when the process exits."""
    for ref in _live_context_refs:
        ctx = ref()
        if ctx is not None:
            try:
                ctx.shutdown()
            except Exception:
                pass
    _live_context_refs.clear()


atexit.register(_atexit_cleanup)


# ---------------------------------------------------------------------------
# Type wrappers — explicit types for shard/chunk data at different lifecycle
# stages, replacing the old list[list[list]] / list[list[ChunkRef]] duality.
# ---------------------------------------------------------------------------

StageResult = tuple[dict, "ChunkRef"]


@dataclass(frozen=True)
class ChunkRef:
    """Reference to a chunk stored on disk.

    Owns its serialization: use the `write` classmethod to write data
    to disk with a temp-file pattern (best-effort durability), and the
    `read` instance method to load it back.
    """

    path: str
    count: int

    @classmethod
    def write(cls, path: str, data: list) -> ChunkRef:
        """Write data to path using temp-file pattern.

        Uses a .tmp suffix and rename/move to reduce (but not eliminate)
        the risk of partial writes. On object stores like GCS/S3, the
        move is not atomic but minimizes corruption windows.
        """
        ensure_parent_dir(path)
        temp_path = f"{path}.tmp"
        fs = fsspec.core.url_to_fs(path)[0]
        try:
            with fsspec.open(temp_path, "wb") as f:
                pickle.dump(data, f)
            fs.mv(temp_path, path)
        except Exception:
            with suppress(Exception):
                if fs.exists(temp_path):
                    fs.rm(temp_path)
            raise
        return cls(path=path, count=len(data))

    def read(self) -> list:
        """Load chunk data from disk."""
        with fsspec.open(self.path, "rb") as f:
            return pickle.load(f)


@dataclass
class ShardRefs:
    """All shards as disk references — the primary inter-stage representation.

    Data stays on disk between stages. Call `load()` only when you actually
    need in-memory access (reshard, final materialization).
    """

    shards: list[list[ChunkRef]]

    def __len__(self) -> int:
        return len(self.shards)

    def load(self) -> ShardedData:
        """Load all shard data from disk into memory."""
        return ShardedData(
            shards=[Shard(chunks=[Chunk(items=ref.read()) for ref in chunk_refs]) for chunk_refs in self.shards]
        )


@dataclass
class Chunk:
    """A list of items that form one unit of processing."""

    items: list[Any]


@dataclass
class Shard:
    """An ordered sequence of chunks assigned to a single worker."""

    chunks: list[Chunk]


@dataclass
class ShardedData:
    """All shards of in-memory data."""

    shards: list[Shard]

    def __len__(self) -> int:
        return len(self.shards)

    def materialize(self) -> list:
        """Flatten all shard chunks into a single list of results."""
        results = []
        for shard in self.shards:
            for chunk in shard.chunks:
                results.extend(chunk.items)
        return results

    def write_to_disk(self, prefix: str, execution_id: str, stage_name: str) -> ShardRefs:
        """Write all shard chunks to disk, return references."""
        shard_refs = []
        for shard_idx, shard in enumerate(self.shards):
            chunk_refs = []
            for chunk_idx, chunk in enumerate(shard.chunks):
                path = _chunk_path(prefix, execution_id, stage_name, shard_idx, chunk_idx)
                chunk_refs.append(ChunkRef.write(path, chunk.items))
            shard_refs.append(chunk_refs)
        return ShardRefs(shards=shard_refs)

    def reshard(self, num_shards: int) -> ShardedData:
        """Redistribute chunks across target number of shards."""
        all_chunks = [chunk for shard in self.shards for chunk in shard.chunks]
        if not all_chunks:
            return ShardedData(shards=[])
        chunk_groups = np.array_split(all_chunks, num_shards)
        return ShardedData(shards=[Shard(chunks=list(group)) for group in chunk_groups if len(group) > 0])


# ---------------------------------------------------------------------------
# ShardProtocol — contract for objects passed as StageContext.shard
# ---------------------------------------------------------------------------


@runtime_checkable
class ShardProtocol(Protocol):
    """Protocol for shard objects consumed by run_stage.

    run_stage iterates via `iter(shard)` for flat access and calls
    `shard.iter_chunks()` for Reduce operations that need chunk boundaries.
    """

    def iter_chunks(self) -> Iterator[list]: ...

    def __iter__(self) -> Iterator: ...


# ---------------------------------------------------------------------------
# Chunk path generation and cleanup
# ---------------------------------------------------------------------------


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
    chunk_refs: list[ChunkRef]
    operations: list[PhysicalOp]
    stage_name: str = "output"
    aux_refs: dict[int, list[list[ChunkRef]]] | None = None


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
# _SerializableShard — lightweight wrapper matching the ShardProtocol
# ---------------------------------------------------------------------------


class _SerializableShard:
    """Wraps chunk references and streams data one chunk at a time from disk.

    This class implements true chunk-by-chunk streaming to minimize memory pressure.
    Chunks are loaded one at a time and discarded after iteration, rather than cached.
    Multiple iterations will re-read from disk, which is an acceptable trade-off for
    memory efficiency.

    run_stage does `iter(ctx.shard)` for flat iteration and
    `ctx.shard.iter_chunks()` for Reduce operations.
    """

    def __init__(self, chunk_refs: list[ChunkRef]):
        self._chunk_refs = chunk_refs

    def iter_chunks(self) -> Iterator[list]:
        """Iterate over chunks, loading one at a time from disk."""
        for ref in self._chunk_refs:
            yield ref.read()

    def __iter__(self) -> Iterator:
        """Flatten iteration over all items, loading chunks as needed."""
        for ref in self._chunk_refs:
            chunk_data = ref.read()
            yield from chunk_data


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
        self._results: dict[int, list[StageResult]] = defaultdict(list)
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

    def report_result(self, worker_id: str, shard_idx: int, attempt: int, result: list[StageResult]) -> None:
        self._last_seen[worker_id] = time.monotonic()

        current_attempt = self._task_attempts.get(shard_idx, 0)
        if attempt != current_attempt:
            logger.warning(
                f"Ignoring stale result from worker {worker_id} for shard {shard_idx} "
                f"(attempt {attempt}, current {current_attempt})"
            )
            return

        self._results[shard_idx].extend(result)
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

    def collect_results(self) -> dict[int, list[StageResult]]:
        """Return results for the completed stage."""
        return dict(self._results)

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
                logger.debug(f"[{worker_id}] No task available, status: {status}")
                if status.get("done") or status.get("fatal_error"):
                    done = status.get("done")
                    error = status.get("fatal_error")
                    logger.info(f"[{worker_id}] Stage done (done={done}, error={error}), exiting")
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
                coordinator.report_error.remote(
                    worker_id,
                    task.shard_idx,
                    _serialize_error(e),
                )

    def _execute_shard(self, task: ShardTask) -> list[StageResult]:
        """Execute a stage's operations on a single shard.

        Returns list of (header_dict, ChunkRef) pairs — one per output chunk.
        """
        _shard_ctx_var.set(self)

        logger.info(
            "[shard %d/%d] Starting stage=%s, %d input chunks, %d ops",
            task.shard_idx,
            task.total_shards,
            task.stage_name,
            len(task.chunk_refs),
            len(task.operations),
        )

        shard = _SerializableShard(task.chunk_refs)

        aux_shards: dict[int, list[Any]] = {}
        if task.aux_refs:
            for op_idx, right_shard_refs in task.aux_refs.items():
                aux_shards[op_idx] = [_SerializableShard(shard_chunk_refs) for shard_chunk_refs in right_shard_refs]

        stage_ctx = StageContext(
            shard=shard,
            shard_idx=task.shard_idx,
            total_shards=task.total_shards,
            chunk_size=task.chunk_size,
            aux_shards=aux_shards,
        )

        results: list[StageResult] = []
        current_header: ChunkHeader | None = None
        chunk_idx = 0

        for item in run_stage(stage_ctx, task.operations):
            if isinstance(item, ChunkHeader):
                current_header = item
            else:
                assert current_header is not None

                chunk_path = _chunk_path(
                    self._chunk_prefix,
                    self._execution_id,
                    task.stage_name,
                    task.shard_idx,
                    chunk_idx,
                )

                chunk_ref = ChunkRef.write(chunk_path, item)
                results.append(
                    (
                        {"shard_idx": current_header.shard_idx, "count": current_header.count},
                        chunk_ref,
                    )
                )
                chunk_idx += 1
                if chunk_idx % 10 == 0:
                    logger.info(
                        "[shard %d] Wrote %d chunks so far (latest: %d items)",
                        task.shard_idx,
                        chunk_idx,
                        current_header.count,
                    )

        logger.info("[shard %d] Complete: %d chunks produced", task.shard_idx, chunk_idx)
        return results

    def shutdown(self) -> None:
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# _run_stage_on_coordinator — shared poll-loop for executing a stage
# ---------------------------------------------------------------------------


def _run_stage_on_coordinator(
    coordinator: ActorHandle,
    stage_name: str,
    tasks: list[ShardTask],
) -> dict[int, list[StageResult]]:
    """Submit tasks to coordinator, poll until complete, return raw results."""
    coordinator.start_stage.remote(stage_name, tasks).result()
    last_log_completed = -1
    while True:
        coordinator.check_heartbeats.remote()
        status = coordinator.get_status.remote().result()
        if status.get("fatal_error"):
            raise ZephyrWorkerError(status["fatal_error"])
        completed = status["completed"]
        total = status["total"]
        if completed != last_log_completed:
            logger.info("[%s] %d/%d tasks completed", stage_name, completed, total)
            last_log_completed = completed
        if completed >= total:
            break
        time.sleep(1.0)
    return coordinator.collect_results.remote().result()


# ---------------------------------------------------------------------------
# _regroup_result_refs — regroup worker output by output shard without loading
# ---------------------------------------------------------------------------


def _regroup_result_refs(
    result_refs: dict[int, list[StageResult]],
    input_shard_count: int,
) -> ShardRefs:
    """Regroup worker output refs by output shard index without loading data."""
    output_by_shard: dict[int, list[ChunkRef]] = defaultdict(list)

    for _input_idx, result_pairs in result_refs.items():
        for header, chunk_ref in result_pairs:
            output_by_shard[header["shard_idx"]].append(chunk_ref)

    num_output = max(max(output_by_shard.keys(), default=0) + 1, input_shard_count)
    return ShardRefs(shards=[output_by_shard.get(idx, []) for idx in range(num_output)])


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
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="16g"))
    max_parallelism: int = 1024
    chunk_storage_prefix: str | None = None
    preserve_chunks: bool = False

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
        _live_context_refs.append(weakref.ref(self))

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

            # Build source data and immediately write to disk as refs
            source_data = _build_source_shards(plan.source_items)
            shard_refs = source_data.write_to_disk(self.chunk_storage_prefix, execution_id, "source")

            for stage_idx, stage in enumerate(plan.stages):
                # Pick up any newly-available workers and start their run loops
                for new_worker in self._discover_workers():
                    self._worker_futures.append(new_worker.run_loop.remote(coordinator))
                stage_label = f"stage{stage_idx}-{stage.stage_name(max_length=40)}"

                if stage.stage_type == StageType.RESHARD:
                    loaded = shard_refs.load()
                    resharded = loaded.reshard(stage.output_shards or len(shard_refs))
                    shard_refs = resharded.write_to_disk(
                        self.chunk_storage_prefix,
                        execution_id,
                        f"{stage_label}-reshard",
                    )
                    continue

                aux_per_shard = _compute_join_aux(
                    stage.operations, shard_refs, coordinator, self.chunk_storage_prefix, hints, execution_id
                )

                tasks = _shard_refs_to_tasks(shard_refs, stage, hints, aux_per_shard, stage_name=stage_label)
                logger.info(f"Starting stage {stage_label} with {len(tasks)} tasks")

                result_refs = _run_stage_on_coordinator(coordinator, stage_label, tasks)
                shard_refs = _regroup_result_refs(result_refs, len(shard_refs))

            coordinator.signal_done.remote().result()

            for f in self._worker_futures:
                with suppress(Exception):
                    f.result(timeout=10.0)
            self._worker_futures = []

            return shard_refs.load().materialize()

        finally:
            if not self.preserve_chunks:
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
        """Discover newly available workers and return them.

        For backends that don't support incremental discovery (e.g. LocalClient),
        this is a no-op since all workers are returned from wait_ready().
        """
        group = self._worker_group
        if group is None:
            return []
        new_handles = group.discover_new()
        if new_handles:
            self._workers.extend(new_handles)
            logger.info("Discovered %d new workers, total now %d", len(new_handles), len(self._workers))
        return new_handles

    def shutdown(self) -> None:
        # Remove our weak ref from the atexit list
        _live_context_refs[:] = [r for r in _live_context_refs if r() is not None and r() is not self]

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_source_shards(source_items: list[SourceItem]) -> ShardedData:
    """Build shard data from source items.

    Each source item becomes a single-element chunk in its assigned shard.
    """
    items_by_shard: dict[int, list] = defaultdict(list)
    for item in source_items:
        items_by_shard[item.shard_idx].append(item.data)

    num_shards = max(items_by_shard.keys()) + 1 if items_by_shard else 0
    shards = []
    for i in range(num_shards):
        chunks = [Chunk(items=[item]) for item in items_by_shard.get(i, [])]
        shards.append(Shard(chunks=chunks))

    return ShardedData(shards=shards)


def _shard_refs_to_tasks(
    shard_refs: ShardRefs,
    stage,
    hints: ExecutionHint,
    aux_per_shard: list[dict[int, list[list[ChunkRef]]]] | None = None,
    stage_name: str | None = None,
) -> list[ShardTask]:
    """Convert shard references into ShardTasks for the coordinator."""
    total = len(shard_refs)
    tasks = []
    output_stage_name = stage_name or stage.stage_name(max_length=60)

    for i, chunk_refs in enumerate(shard_refs.shards):
        aux_refs = None
        if aux_per_shard and aux_per_shard[i]:
            aux_refs = aux_per_shard[i]

        tasks.append(
            ShardTask(
                shard_idx=i,
                total_shards=total,
                chunk_size=hints.chunk_size,
                chunk_refs=chunk_refs,
                operations=stage.operations,
                stage_name=output_stage_name,
                aux_refs=aux_refs,
            )
        )

    return tasks


def _compute_join_aux(
    operations: list[PhysicalOp],
    shard_refs: ShardRefs,
    coordinator: ActorHandle,
    chunk_storage_prefix: str,
    hints: ExecutionHint,
    execution_id: str,
) -> list[dict[int, list[list[ChunkRef]]]] | None:
    """Execute right sub-plans for join operations, returning aux refs per shard."""
    all_right_shard_refs: dict[int, ShardRefs] = {}

    for i, op in enumerate(operations):
        if not isinstance(op, Join) or op.right_plan is None:
            continue

        right_source_data = _build_source_shards(op.right_plan.source_items)
        right_refs = right_source_data.write_to_disk(
            chunk_storage_prefix,
            execution_id,
            f"join-right-{i}-source",
        )

        for stage_idx, right_stage in enumerate(op.right_plan.stages):
            if right_stage.stage_type == StageType.RESHARD:
                right_data = right_refs.load()
                right_data = right_data.reshard(right_stage.output_shards or len(right_refs))
                right_refs = right_data.write_to_disk(
                    chunk_storage_prefix,
                    execution_id,
                    f"join-right-{i}-stage{stage_idx}-reshard",
                )
                continue

            join_stage_label = f"join-right-{i}-stage{stage_idx}"
            right_tasks = _shard_refs_to_tasks(right_refs, right_stage, hints, stage_name=join_stage_label)
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
