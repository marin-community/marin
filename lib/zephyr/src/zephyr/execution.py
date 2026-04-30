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

import enum
import itertools
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
import threading
import time
import traceback
import uuid
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

import cloudpickle
from rigging.filesystem import open_url, url_to_fs
from fray import ActorConfig, ActorFuture, ActorHandle, Client, ResourceConfig
from fray.client import JobHandle
from fray.types import Entrypoint, JobRequest
from rigging.filesystem import marin_temp_bucket
from rigging.timing import ExponentialBackoff, log_time

from iris.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from zephyr.dataset import Dataset
from zephyr.plan import (
    Join,
    PhysicalOp,
    PhysicalPlan,
    PhysicalStage,
    Scatter,
    Shard,
    SourceItem,
    StageType,
    compute_plan,
)
from zephyr.shuffle import ListShard, MemChunk, _write_scatter
from zephyr.writers import INTERMEDIATE_CHUNK_SIZE, ensure_parent_dir

logger = logging.getLogger(__name__)

# Max explicit task errors (report_error) per shard before aborting.
MAX_SHARD_FAILURES = 3

# Max infra failures observed *while the same shard was in flight* on the
# crashing worker before treating that shard as a deterministic crasher and
# aborting. Genuine preemption distributes across all in-flight shards, so
# the same shard hitting this cap is strong evidence the shard payload is
# what's killing the worker (e.g. native SIGSEGV from Arrow / JAX, or an
# OOM that brings the host down). Set well above realistic preemption
# storms for any one shard in a multi-shard pipeline.
MAX_SHARD_INFRA_FAILURES = 20

ZEPHYR_STAGE_ITEM_COUNT_KEY = "zephyr/stage/{stage_name}/item_count"
ZEPHYR_STAGE_BYTES_PROCESSED_KEY = "zephyr/stage/{stage_name}/bytes_processed"

# Typical status text for a 6-stage pipeline is ~300 chars.
MAX_STATUS_TEXT_LENGTH = 1000


class ShardFailureKind(enum.StrEnum):
    """TASK failures count toward MAX_SHARD_FAILURES; INFRA failures (preemption) do not."""

    TASK = enum.auto()
    INFRA = enum.auto()


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


# ---------------------------------------------------------------------------
# Task result
# ---------------------------------------------------------------------------


@dataclass
class CounterSnapshot:
    """Bundled counter values and monotonically increasing generation tag.

    The generation increments on every snapshot, so each heartbeat and
    report_result carries a unique tag.  The coordinator uses strict
    ordering (>) to discard stale or out-of-order updates.
    """

    counters: dict[str, int]
    generation: int

    @staticmethod
    def empty(generation: int = 0) -> CounterSnapshot:
        return CounterSnapshot(counters={}, generation=generation)


@dataclass
class TaskResult:
    """Result of a single worker task.

    Always contains a ListShard. For non-scatter stages, refs are
    PickleDiskChunks. For scatter stages, refs contain file paths
    (the actual metadata lives in ``.scatter_meta`` sidecar files
    read lazily by reducers).
    """

    shard: ListShard


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

    with log_time(f"Cleaning up execution directory {exec_dir}"):
        if fs.exists(exec_dir):
            try:
                fs.rm(exec_dir, recursive=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup chunks at {exec_dir}: {e}")


def _write_pickle_chunks(
    items: Iterator,
    source_shard: int,
    chunk_path_fn: Callable[[int], str],
) -> ListShard:
    """Batch a plain item stream into pickle chunk files.

    Returns a ListShard containing PickleDiskChunk references.
    """
    chunk_size = INTERMEDIATE_CHUNK_SIZE
    chunks: list[Iterable] = []
    batch: list = []
    pidx = 0

    for item in items:
        batch.append(item)
        if chunk_size > 0 and len(batch) >= chunk_size:
            chunk_ref = PickleDiskChunk.write(chunk_path_fn(pidx), batch)
            chunks.append(chunk_ref)
            pidx += 1
            batch = []
            if pidx % 10 == 0:
                logger.info(
                    "[shard %d] Wrote %d pickle chunks so far (latest: %d items)",
                    source_shard,
                    pidx,
                    chunk_ref.count,
                )

    if batch:
        chunks.append(PickleDiskChunk.write(chunk_path_fn(pidx), batch))

    return ListShard(refs=chunks)


def _write_stage_output(
    stage_gen: Iterator,
    source_shard: int,
    stage_dir: str,
    shard_idx: int,
    scatter_op: Scatter | None,
    total_shards: int,
) -> TaskResult:
    """Write stage output to disk.

    For scatter stages (``scatter_op`` is set), writes Parquet with envelope
    wrapping and ``.scatter_meta`` sidecars. Returns TaskResult with compact
    scatter metadata.

    For non-scatter stages, batches items into pickle chunk files. Returns
    TaskResult with a ListShard.
    """
    if scatter_op is not None:
        first_item = next(stage_gen, None)
        if first_item is None:
            return TaskResult(shard=ListShard(refs=[]))

        full_gen = itertools.chain([first_item], stage_gen)

        num_output_shards = scatter_op.num_output_shards if scatter_op.num_output_shards > 0 else total_shards
        data_path = f"{stage_dir}/shard-{shard_idx:04d}.shuffle"
        shard = _write_scatter(
            full_gen,
            source_shard,
            data_path,
            key_fn=scatter_op.key_fn,
            num_output_shards=num_output_shards,
            sort_fn=scatter_op.sort_fn,
            combiner_fn=scatter_op.combiner_fn,
        )
        return TaskResult(shard=shard)

    def chunk_path_fn(idx: int) -> str:
        return f"{stage_dir}/shard-{shard_idx:04d}/chunk-{idx:04d}.pkl"

    return TaskResult(shard=_write_pickle_chunks(stage_gen, source_shard, chunk_path_fn))


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
    shard: Shard
    operations: list[PhysicalOp]
    stage_name: str = "output"
    aux_shards: dict[int, Shard] | None = None


class ZephyrWorkerError(RuntimeError):
    """Raised when a worker encounters a fatal (non-transient) error."""


# Application errors that should never be retried by the execute() retry loop.
# These are deterministic errors (bad plan, invalid config, programming bugs)
# that would fail identically on every attempt. Infrastructure errors (OSError,
# RuntimeError from dead actors, backend actor errors) are NOT listed here so they
# remain retryable.
_NON_RETRYABLE_ERRORS = (ZephyrWorkerError, ValueError, TypeError, KeyError, AttributeError, MemoryError)


# ---------------------------------------------------------------------------
# WorkerContext protocol — the public interface exposed to user task code
# ---------------------------------------------------------------------------


class WorkerContext(Protocol):
    def get_shared(self, name: str) -> Any: ...
    def increment_counter(self, name: str, value: int = 1) -> None: ...
    def get_counter_snapshot(self) -> CounterSnapshot: ...


_worker_ctx_var: ContextVar[WorkerContext | None] = ContextVar("zephyr_worker_ctx", default=None)


def zephyr_worker_ctx() -> WorkerContext:
    """Get the current worker's context. Only valid inside a worker task."""
    ctx = _worker_ctx_var.get()
    if ctx is None:
        raise RuntimeError("zephyr_worker_ctx() called outside of a worker task")
    return ctx


class StageRunner(Protocol):
    """Strategy a worker uses to execute a single shard. See ``zephyr.runners``."""

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, int]]: ...

    def live_counters(self) -> dict[str, int]: ...


def _default_stage_runner_factory_for(client: Client) -> Callable[[], StageRunner]:
    """Pick the default ``stage_runner_factory`` based on the client type.

    ``LocalClient`` is the dev/test backend — workers are threads in a
    single process, so per-shard subprocess isolation adds latency without
    delivering meaningful isolation. Distributed clients run each worker
    actor as its own VM where subprocess-per-shard gives real protection
    against native crashes and per-shard memory growth. Callers that want
    the other behavior pass ``stage_runner_factory=...`` explicitly.
    """
    from fray.local_backend import LocalClient

    if isinstance(client, LocalClient):
        from zephyr.runners import InlineRunner

        return InlineRunner

    from zephyr.runners import SubprocessRunner

    return SubprocessRunner


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
        from fray import current_actor

        # Task management state
        self._task_queue: deque[ShardTask] = deque()
        self._results: dict[int, TaskResult] = {}
        self._worker_states: dict[str, WorkerState] = {}
        self._last_seen: dict[str, float] = {}
        self._stage_name: str = ""
        # The index of the currently active stage. For joins and reshards, the index of the parent.
        self._current_stage_index: int = 0
        self._plan_stages: list = []  # PhysicalStage list, set in run_pipeline
        self._total_shards: int = 0
        self._completed_shards: int = 0
        self._retries: int = 0
        self._in_flight: dict[str, tuple[ShardTask, int]] = {}
        # _task_attempts: monotonic generation for stale-result rejection (bumps on every
        # requeue). _task_error_attempts: TASK-only counter, bounded by MAX_SHARD_FAILURES.
        self._task_attempts: dict[int, int] = {}
        self._task_error_attempts: dict[int, int] = {}
        self._fatal_error: str | None = None
        self._shard_errors: dict[int, list[str]] = {}
        self._chunk_prefix: str = ""
        self._execution_id: str = ""
        self._no_workers_timeout: float = 60.0
        # Per-worker in-flight counter snapshots and completed snapshots.
        # Each snapshot carries a monotonic generation so the coordinator
        # can discard stale or out-of-order heartbeats.
        self._worker_counters: dict[str, CounterSnapshot] = {}
        self._completed_counters: list[CounterSnapshot] = []

        # Worker management state (workers self-register via register_worker)
        self._worker_handles: dict[str, ActorHandle] = {}
        self._worker_group: Any = None  # ActorGroup, set via set_worker_group()
        self._coordinator_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._is_last_stage: bool = False
        self._initialized: bool = False
        self._pipeline_running: bool = False

        # Set at each _start_stage so _log_status can show average throughput since stage start.
        self._stage_monotonic_start: float | None = None

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
            if sys.is_finalizing():
                return
            try:
                self.check_heartbeats()
                self._check_worker_group()

                now = time.monotonic()
                if self._has_active_execution() and now - last_log_time > 5.0:
                    self._log_status()
                    self._report_task_stats()
                    last_log_time = now
            except Exception:
                if sys.is_finalizing():
                    return
                logger.exception("Coordinator loop crashed, aborting pipeline")
                self.abort("Coordinator loop crashed unexpectedly")
                return

            self._shutdown_event.wait(timeout=0.5)

    def _check_worker_group(self) -> None:
        """Abort the pipeline if the worker job has permanently terminated."""
        if self._worker_group is None or self._fatal_error is not None:
            return
        # After the last stage completes, workers exit cleanly via SHUTDOWN.
        # The worker job finishing at that point is expected, not a crash.
        with self._lock:
            if self._total_shards > 0 and self._completed_shards >= self._total_shards:
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

    def _report_task_stats(self) -> None:
        """Push task status text to the Iris coordinator if available."""
        iris_client = ctx.client if (ctx := get_iris_ctx()) is not None else None
        if iris_client is None:
            return

        job_info = get_job_info()
        if job_info is None:
            return

        with self._lock:
            current_stage_index = self._current_stage_index
            stage_name = self._stage_name
            plan_stages = self._plan_stages
            completed = self._completed_shards
            total_shards = self._total_shards
            in_flight = len(self._in_flight)
            queued = len(self._task_queue)
            stage_start = self._stage_monotonic_start

        totals = self.get_counters()
        items = totals.get(ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name), 0)
        bytes_processed = totals.get(ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name), 0)
        elapsed = time.monotonic() - (stage_start if stage_start is not None else float("inf"))
        item_rate = items / elapsed if elapsed > 0 else 0.0
        byte_rate = bytes_processed / elapsed if elapsed > 0 else 0.0

        lines = ["**Stages**\n"]
        for idx, stage in enumerate(plan_stages):
            stage_desc = _get_stage_description(stage)
            bullet = f"- **{stage_desc}**" if idx == current_stage_index else f"- {stage_desc}"
            lines.append(f"{bullet}")

        pct = int(100 * completed / total_shards) if total_shards > 0 else 0
        lines.append(
            f"\n**Shards** — {completed}/{total_shards} complete ({pct}%), {in_flight} in-flight, {queued} queued"
        )
        mib = bytes_processed / (1024 * 1024)
        mib_rate = byte_rate / (1024 * 1024)
        lines.append(f"\n**Throughput** — {items:,} items ({item_rate:.1f}/s), {mib:.1f} MiB ({mib_rate:.1f} MiB/s)")

        status_text_md = "\n".join(lines)[:MAX_STATUS_TEXT_LENGTH]
        try:
            iris_client.report_task_status_text(job_info.task_id, status_text_md)
        except Exception:
            logger.warning("Failed to report task status text to Iris controller", exc_info=True)

    def _log_status(self) -> None:
        with self._lock:
            states = list(self._worker_states.values())
            retried = {idx: att for idx, att in self._task_attempts.items() if att > 0}
        alive = sum(1 for s in states if s in {WorkerState.READY, WorkerState.BUSY})
        dead = sum(1 for s in states if s in {WorkerState.FAILED, WorkerState.DEAD})

        totals = self.get_counters()
        item_key = ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=self._stage_name)
        byte_key = ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=self._stage_name)
        base_msg = "[%s] [%s] %d/%d complete, %d in-flight, %d queued, %d/%d workers alive, %d dead"
        base_args = (
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

        # Map-only stages don't yield through ``_StageStatsGenerator`` and never
        # populate these counters. Drop the items/bytes_processed segment for
        # those stages.
        if item_key in totals or byte_key in totals:
            items = totals.get(item_key, 0)
            bytes_processed = totals.get(byte_key, 0)
            elapsed = time.monotonic() - (
                self._stage_monotonic_start if self._stage_monotonic_start is not None else float("inf")
            )
            item_rate = items / elapsed
            byte_rate = bytes_processed / elapsed
            logger.info(
                base_msg + "; items=%d (%.1f/s), bytes_processed=%.1fMiB (%.1fMiB/s)",
                *base_args,
                items,
                item_rate,
                bytes_processed / (1024 * 1024),
                byte_rate / (1024 * 1024),
            )
        else:
            logger.info(base_msg, *base_args)
        if retried:
            attempts_histogram = dict(sorted(Counter(retried.values()).items()))
            logger.warning("[%s] Shards retried (attempts: shard count): %s", self._execution_id, attempts_histogram)

    def _record_shard_failure(
        self,
        worker_id: str,
        kind: ShardFailureKind,
        error_info: str | None = None,
    ) -> bool:
        """Requeue the worker's in-flight shard; abort if a per-shard cap is hit.

        TASK errors are bounded by ``MAX_SHARD_FAILURES``. INFRA failures
        observed while the *same* shard was in flight are bounded by
        ``MAX_SHARD_INFRA_FAILURES`` so a payload that deterministically
        crashes its worker (native SIGSEGV, OOM) doesn't loop forever now
        that shard execution is in-process.

        Must be called with lock held. Returns True if the pipeline was aborted.
        """
        task_and_attempt = self._in_flight.pop(worker_id, None)

        # Zero counters but keep the generation watermark so late heartbeats
        # from the old task are rejected.
        existing = self._worker_counters.get(worker_id)
        if existing is not None:
            self._worker_counters[worker_id] = CounterSnapshot.empty(existing.generation)

        if task_and_attempt is None:
            return False

        task, _ = task_and_attempt
        shard_idx = task.shard_idx

        if error_info is not None:
            self._shard_errors.setdefault(shard_idx, []).append(error_info)

        # Bump generation regardless of kind so report_result rejects stale attempts.
        self._task_attempts[shard_idx] += 1

        if kind is ShardFailureKind.TASK:
            self._task_error_attempts[shard_idx] += 1
            error_attempts = self._task_error_attempts[shard_idx]
            if error_attempts >= MAX_SHARD_FAILURES:
                errors = self._shard_errors.get(shard_idx, [])
                error_detail = f"\nLast error:\n{errors[-1]}" if errors else ""
                logger.error(
                    "Shard %d has failed %d times (max %d), last failure on worker %s, aborting pipeline.",
                    shard_idx,
                    error_attempts,
                    MAX_SHARD_FAILURES,
                    worker_id,
                )
                self._fatal_error = (
                    f"Shard {shard_idx} failed {error_attempts} times "
                    f"(max {MAX_SHARD_FAILURES}), last failure on worker {worker_id}.{error_detail}"
                )
                return True

            logger.warning(
                "Shard %d failed on worker %s (task error %d/%d), re-queuing.",
                shard_idx,
                worker_id,
                error_attempts,
                MAX_SHARD_FAILURES,
            )
        else:
            self._task_infra_attempts[shard_idx] += 1
            infra_attempts = self._task_infra_attempts[shard_idx]
            if infra_attempts >= MAX_SHARD_INFRA_FAILURES:
                logger.error(
                    "Shard %d has been in flight during %d infra failures (max %d); "
                    "treating as a deterministic crasher (likely native SIGSEGV / OOM in shard "
                    "code) and aborting pipeline. Last failure on worker %s.",
                    shard_idx,
                    infra_attempts,
                    MAX_SHARD_INFRA_FAILURES,
                    worker_id,
                )
                self._fatal_error = (
                    f"Shard {shard_idx} crashed its worker {infra_attempts} times "
                    f"(max {MAX_SHARD_INFRA_FAILURES} infra failures while in flight); "
                    f"last failure on worker {worker_id}."
                )
                return True

            logger.warning(
                "Shard %d requeued from worker %s due to infra failure (preemption/heartbeat). "
                "Total generation: %d, task errors so far: %d/%d, infra-while-in-flight: %d/%d.",
                shard_idx,
                worker_id,
                self._task_attempts[shard_idx],
                self._task_error_attempts[shard_idx],
                MAX_SHARD_FAILURES,
                infra_attempts,
                MAX_SHARD_INFRA_FAILURES,
            )

        self._task_queue.append(task)
        self._retries += 1
        return False

    def _maybe_requeue_worker_task(self, worker_id: str) -> None:
        """Requeue the worker's in-flight task as an INFRA failure (preemption/heartbeat)."""
        self._record_shard_failure(worker_id, ShardFailureKind.INFRA)

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
                    # No more work to hand out — exit immediately.  If an
                    # in-flight worker crashes and its shard is requeued, Iris
                    # restarts the worker which re-registers and picks it up.
                    # _check_worker_group() detects permanent worker-job death
                    # as a failsafe so we never deadlock.
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

    def report_result(
        self,
        worker_id: str,
        shard_idx: int,
        attempt: int,
        result: TaskResult,
        counter_snapshot: CounterSnapshot,
    ) -> None:
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
            self._completed_counters.append(counter_snapshot)
            # Zero the in-flight counters but keep the generation watermark
            # so late heartbeats from this task are rejected.
            self._worker_counters[worker_id] = CounterSnapshot.empty(counter_snapshot.generation)

    def report_error(self, worker_id: str, shard_idx: int, error_info: str) -> None:
        """Worker reports a task failure. Re-queues up to MAX_SHARD_FAILURES."""
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            self._assert_in_flight_consistent(worker_id, shard_idx)
            aborted = self._record_shard_failure(worker_id, ShardFailureKind.TASK, error_info)
            self._worker_states[worker_id] = WorkerState.DEAD if aborted else WorkerState.READY

    def heartbeat(self, worker_id: str, counter_snapshot: CounterSnapshot | None = None) -> None:
        self._last_seen[worker_id] = time.monotonic()
        if counter_snapshot is not None:
            with self._lock:
                existing = self._worker_counters.get(worker_id)
                if existing is None or counter_snapshot.generation > existing.generation:
                    self._worker_counters[worker_id] = counter_snapshot

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

    def get_counters(self, worker_id: str | None = None) -> dict[str, int]:
        """Return counter values, optionally filtered to a single worker.

        Args:
            worker_id: If provided, return the latest snapshot for this worker
                only. If None, return totals derived from completed and
                in-flight snapshots.
        """
        with self._lock:
            if worker_id is not None:
                snap = self._worker_counters.get(worker_id)
                return dict(snap.counters) if snap is not None else {}

            totals: dict[str, int] = {}
            for snap in self._completed_counters:
                for name, value in snap.counters.items():
                    totals[name] = totals.get(name, 0) + value
            for snap in self._worker_counters.values():
                for name, value in snap.counters.items():
                    totals[name] = totals.get(name, 0) + value
            return totals

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

    def _start_stage(
        self,
        stage_name: str,
        current_stage_index: int,
        tasks: list[ShardTask],
        is_last_stage: bool = False,
    ) -> None:
        """Load a new stage's tasks into the queue."""
        with self._lock:
            self._task_queue = deque(tasks)
            self._results = {}
            self._stage_name = stage_name
            self._current_stage_index = current_stage_index
            self._total_shards = len(tasks)
            self._completed_shards = 0
            self._retries = 0
            self._in_flight = {}
            self._task_attempts = {task.shard_idx: 0 for task in tasks}
            self._task_error_attempts = {task.shard_idx: 0 for task in tasks}
            # Counts INFRA failures observed while this specific shard was in
            # flight on the dying worker — bounded by MAX_SHARD_INFRA_FAILURES
            # so a shard that deterministically crashes its worker (native
            # SIGSEGV, OOM) eventually aborts instead of retrying forever.
            self._task_infra_attempts = {task.shard_idx: 0 for task in tasks}
            self._shard_errors = {}
            self._fatal_error = None
            self._is_last_stage = is_last_stage
            # Only reset in-flight worker snapshots; completed snapshots
            # accumulate across stages for full pipeline visibility.
            self._worker_counters = {}
            self._stage_monotonic_start = time.monotonic()

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

            with self._lock:
                self._current_stage_index = 0
                self._plan_stages = list(plan.stages)

            for stage_idx, stage in enumerate(plan.stages):
                stage_label = f"stage{stage_idx}-{stage.stage_name(max_length=40)}"

                if stage.stage_type == StageType.RESHARD:
                    shards = _reshard_refs(shards, stage.output_shards or len(shards))
                    continue

                # Compute aux data for joins
                aux_per_shard = self._compute_join_aux(stage.operations, shards, stage_idx)

                # Build and submit tasks
                tasks = _compute_tasks_from_shards(shards, stage, stage_name=stage_label, aux_per_shard=aux_per_shard)
                logger.info("[%s] Starting stage %s with %d tasks", self._execution_id, stage_label, len(tasks))
                self._start_stage(stage_label, stage_idx, tasks, is_last_stage=(stage_idx == last_worker_stage_idx))

                # Wait for stage completion
                self._wait_for_stage()

                # Collect and regroup results for next stage
                result_refs = self._collect_results()
                stage_is_scatter = any(isinstance(op, Scatter) for op in stage.operations)
                shards = _regroup_result_refs(
                    result_refs,
                    len(shards),
                    output_shard_count=stage.output_shards,
                    is_scatter=stage_is_scatter,
                )

            # Flatten final results — each shard may involve I/O (unpickling from
            # remote storage), so parallelize across shards with a thread pool.
            def _materialize_shard(shard):
                return list(shard)

            with ThreadPoolExecutor(max_workers=min(32, len(shards))) as flatten_pool:
                materialized = flatten_pool.map(_materialize_shard, shards)

            flat_result = []
            for items in materialized:
                flat_result.extend(items)

            return flat_result
        finally:
            with self._lock:
                self._pipeline_running = False

    def _compute_join_aux(
        self,
        operations: list[PhysicalOp],
        shard_refs: list[Shard],
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
                right_tasks = _compute_tasks_from_shards(right_refs, right_stage, stage_name=join_stage_label)
                self._start_stage(join_stage_label, parent_stage_idx, right_tasks)
                self._wait_for_stage()
                raw = self._collect_results()
                right_is_scatter = any(isinstance(op, Scatter) for op in right_stage.operations)
                right_refs = _regroup_result_refs(
                    raw,
                    len(right_refs),
                    output_shard_count=right_stage.output_shards,
                    is_scatter=right_is_scatter,
                )

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

        counters = self.get_counters()
        if counters:
            logger.info("[coordinator.shutdown] Final counters: %s", counters)

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

    def check_heartbeats(self, timeout: float = 120.0) -> None:
        """Marks stale workers as FAILED, re-queues their in-flight tasks."""
        with self._lock:
            self._check_worker_heartbeats(timeout)


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

    def __init__(
        self,
        coordinator_handle: ActorHandle,
        stage_runner_factory: Callable[[], StageRunner] | None = None,
    ):
        from fray import current_actor

        # ZephyrContext normally pre-resolves this via _CoordinatorJobConfig;
        # the fallback here covers callers that construct a worker directly
        # (mostly internal tests). Default to InlineRunner since direct
        # construction is a dev/test path.
        if stage_runner_factory is None:
            from zephyr.runners import InlineRunner

            stage_runner_factory = InlineRunner

        self._coordinator = coordinator_handle
        self._shutdown_event = threading.Event()
        self._chunk_prefix: str = ""
        self._execution_id: str = ""
        self._counter_generation: int = 0
        self._last_reported_counters: dict[str, int] = {}
        # Each worker owns its runner instance; the heartbeat thread polls
        # ``live_counters()`` off it while a task is in flight.
        self._stage_runner: StageRunner = stage_runner_factory()

        # Capture shutdown_event from the actor context while the ContextVar
        # is still set (child threads in Python <3.12 don't inherit it).
        actor_ctx = current_actor()
        self._host_shutdown_event = actor_ctx.shutdown_event
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

    def _heartbeat_counter_snapshot(self) -> CounterSnapshot | None:
        """Snapshot the running task's counters, or None when idle/unchanged.

        Delegates to the configured ``StageRunner.live_counters``: the inline
        runner exposes its in-memory ctx; the subprocess runner reads the
        child's flush file. Skipping when nothing has changed keeps
        heartbeat traffic cheap on the coordinator side. ``report_result``
        is the source of truth for the final per-task values.
        """
        current = self._stage_runner.live_counters()
        if current == self._last_reported_counters:
            return None
        self._last_reported_counters = current
        self._counter_generation += 1
        return CounterSnapshot(counters=current, generation=self._counter_generation)

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
            logger.debug("[%s] Polling loop ended, signaling host shutdown", self._worker_id)
            if self._host_shutdown_event is not None:
                self._host_shutdown_event.set()

    def _heartbeat_loop(
        self, coordinator: ActorHandle, interval: float = 5.0, max_consecutive_failures: int = 5
    ) -> None:
        logger.debug("[%s] Heartbeat loop starting", self._worker_id)
        heartbeat_count = 0
        consecutive_failures = 0
        while not self._shutdown_event.is_set():
            try:
                # Block on result to avoid congesting the coordinator RPC pipe
                # with fire-and-forget heartbeats. Only send counter snapshot
                # when values have changed.
                snapshot = self._heartbeat_counter_snapshot()
                coordinator.heartbeat.remote(
                    self._worker_id,
                    snapshot,
                ).result()
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
        backoff = ExponentialBackoff(initial=0.1, maximum=5.0)

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

            logger.info(
                "[%s] Executing task for stage %s shard %d (attempt %d)",
                self._worker_id,
                task.stage_name,
                task.shard_idx,
                attempt,
            )
            try:
                t_0 = time.monotonic()
                result, task_counters = self._execute_shard(task, config)
                logger.info(
                    "[%s] Task for shard %d completed in %.2f seconds",
                    self._worker_id,
                    task.shard_idx,
                    time.monotonic() - t_0,
                )
                # Block until coordinator records the result. This ensures
                # report_result is fully processed before the next pull_task,
                # preventing _in_flight tracking races. The counter snapshot
                # is built directly from the subprocess result file so no
                # parent-side counter state needs to be kept in sync.
                self._counter_generation += 1
                coordinator.report_result.remote(
                    self._worker_id,
                    task.shard_idx,
                    attempt,
                    result,
                    CounterSnapshot(counters=dict(task_counters), generation=self._counter_generation),
                ).result()
                task_count += 1
            except Exception as e:
                logger.error("Worker %s error on shard %d: %s", self._worker_id, task.shard_idx, e)
                coordinator.report_error.remote(
                    self._worker_id,
                    task.shard_idx,
                    "".join(traceback.format_exc()),
                ).result()

    def _execute_shard(self, task: ShardTask, config: dict) -> tuple[TaskResult, dict[str, int]]:
        """Hand the task to the configured ``StageRunner``.

        The runner owns the worker-context lifetime and decides whether to
        execute inline or in a subprocess; this method just plumbs the
        per-execution config and surfaces the final ``(TaskResult, counters)``
        for ``report_result``.
        """
        chunk_prefix = config["chunk_prefix"]
        execution_id = config["execution_id"]
        self._chunk_prefix = chunk_prefix
        self._execution_id = execution_id

        logger.info(
            "[%s] [shard %d/%d] Starting stage=%s, %d ops",
            execution_id,
            task.shard_idx,
            task.total_shards,
            task.stage_name,
            len(task.operations),
        )

        result, counters = self._stage_runner.execute(task, chunk_prefix, execution_id)
        logger.info(
            "[shard %d] Complete: %d refs produced",
            task.shard_idx,
            len(result.shard.refs),
        )
        return result, counters

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
    is_scatter: bool = False,
) -> list[Shard]:
    """Regroup worker output refs by output shard index without loading data.

    Non-scatter: each worker's ListShard maps to its own index (identity).
    Scatter: passes the list of scatter data-file paths to every reducer.
    Each reducer reads the per-mapper ``.scatter_meta`` sidecars in parallel
    to build its own ``ScatterReader`` without coordinator-side consolidation.
    """
    if is_scatter:
        # Scatter routes records into exactly ``output_shard_count`` buckets via
        # ``hash(key) % output_shard_count``; spawning more reduce tasks than that
        # produces empty output files for shard indices that no record hashes to.
        # When output_shard_count is None (group_by auto-detect), inherit the
        # input shard count.
        num_output = output_shard_count if output_shard_count is not None else input_shard_count

        # Collect all scatter file paths from all workers. The coordinator
        # does NOT read the sidecars or write a consolidated manifest —
        # reducers do their own parallel sidecar reads.
        all_paths: list[str] = []
        for result in result_refs.values():
            all_paths.extend(result.shard)

        shared_refs = MemChunk(items=all_paths)
        return [ListShard(refs=[shared_refs]) for _ in range(num_output)]

    # Non-scatter: 1:1 mapping from input shard index to output. Resharding
    # to a different shard count belongs to ReshardOp, not here.
    num_output = max(max(result_refs.keys(), default=0) + 1, input_shard_count)
    return [result_refs[idx].shard if idx in result_refs else ListShard(refs=[]) for idx in range(num_output)]


# ---------------------------------------------------------------------------
# Coordinator-as-Job infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZephyrExecutionResult:
    """Result of running a Zephyr pipeline.

    This is also the wire format pickled by ``_run_coordinator_job`` into the
    result file, so callers of ``ZephyrContext.execute`` receive it as-is.

    Attributes:
        results: Flat list of items produced by the terminal stage of the
            pipeline (e.g. output file paths for write stages).
        counters: Aggregated counter values from the run, including built-in
            zephyr counters (e.g. ``zephyr/records_in``) and any user counters
            recorded via ``zephyr.counters.increment``.
    """

    results: list
    counters: dict[str, int]


@dataclass(frozen=True)
class _CoordinatorJobConfig:
    """Serializable config for the coordinator job entrypoint."""

    plan: PhysicalPlan
    execution_id: str
    chunk_storage_prefix: str
    no_workers_timeout: float
    max_workers: int
    worker_resources: ResourceConfig
    name: str
    pipeline_id: int
    # None → workers fall back to InlineRunner (default). The factory is
    # cloudpickled and re-invoked once per worker actor, so per-runner
    # mutable state is per-worker.
    stage_runner_factory: Callable[[], StageRunner] | None = None


def _run_coordinator_job(config_path: str, result_path: str) -> None:
    """Entrypoint for the coordinator job.

    Hosts the coordinator actor in-process via host_actor(), creates
    worker actors as child jobs, runs the pipeline, and writes results
    to disk. The coordinator monitors worker job health directly in its
    maintenance loop (no separate watchdog thread).
    """
    from fray.client import current_client
    from iris.cluster.client.job_info import get_job_info

    logger.info("Loading coordinator config from %s", config_path)
    with open_url(config_path, "rb") as f:
        config: _CoordinatorJobConfig = cloudpickle.loads(f.read())

    job_info = get_job_info()
    attempt_id = job_info.attempt_id if job_info else 0

    logger.info(
        "Coordinator job starting: name=%s, execution_id=%s, pipeline=%d, attempt=%d",
        config.name,
        config.execution_id,
        config.pipeline_id,
        attempt_id,
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
        # Worker name includes attempt ID so that if a stale coordinator
        # process from a previous attempt is still running, its shutdown
        # targets the old name and cannot kill this attempt's workers.
        worker_name = f"zephyr-{config.name}-p{config.pipeline_id}-workers-a{attempt_id}"
        logger.info("Starting %d workers (max=%d, shards=%d)", actual_workers, config.max_workers, num_shards)
        worker_group = client.create_actor_group(
            ZephyrWorker,
            coordinator,
            config.stage_runner_factory,
            name=worker_name,
            count=actual_workers,
            resources=config.worker_resources,
            actor_config=ActorConfig(max_task_retries=10),
        )
        worker_group.wait_ready(count=1, timeout=3600.0)

        # Let the coordinator poll worker job health in its maintenance loop
        coordinator.set_worker_group.remote(worker_group).result()

    try:
        results = coordinator.run_pipeline.submit(config.plan, config.execution_id).result()
        counters = coordinator.get_counters.remote().result(timeout=10.0) or {}
        payload = ZephyrExecutionResult(results=results, counters=counters)

        ensure_parent_dir(result_path)
        with open_url(result_path, "wb") as f:
            f.write(cloudpickle.dumps(payload))
    except Exception as e:
        # Persist the exception so the caller can recover the original type
        # (important for non-retryable error detection).
        with suppress(Exception):
            ensure_parent_dir(result_path)
            with open_url(result_path, "wb") as f:
                f.write(cloudpickle.dumps(e))
        raise
    finally:
        # Signal coordinator shutdown first so workers receive SHUTDOWN from
        # pull_task and self-terminate via shutdown_event → exit_actor()
        # before worker_group.shutdown() tears down any remaining actors.
        with suppress(Exception):
            coordinator.shutdown.remote().result(timeout=10.0)
        if worker_group is not None:
            with suppress(Exception):
                worker_group.shutdown()
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
        coordinator_resources: Resource config for the coordinator job. Defaults to 2 GB.
        chunk_storage_prefix: Storage prefix for intermediate chunks. If None, defaults
            to MARIN_PREFIX/tmp/zephyr or /tmp/zephyr.
        name: Descriptive name for this context, used in actor group names for debugging.
            Defaults to a random 8-character hex string.
        no_workers_timeout: Seconds to wait for at least one worker before failing a stage.
            Defaults to 600s.
        max_execution_retries: Maximum number of times to retry a pipeline execution after
            an infrastructure failure (e.g., coordinator VM preemption). Application errors
            (ZephyrWorkerError) are never retried. Defaults to 100.
        stage_runner_factory: Zero-arg callable returning a fresh ``StageRunner``
            for each worker. If left ``None``, the default depends on the
            client: ``LocalClient`` (dev/test) gets ``InlineRunner`` so shards
            run in-process for speed; distributed clients (Iris) get
            ``SubprocessRunner`` so each shard is isolated against native
            crashes and per-shard memory growth. Pass an explicit factory to
            override.
    """

    client: Client | None = None
    max_workers: int | None = None
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="1g"))
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=1, ram="2g", preemptible=True)
    )
    chunk_storage_prefix: str | None = None
    name: str = ""
    no_workers_timeout: float | None = None
    # NOTE: 100 is fairly aggressive but it fits the preemptible env better
    max_execution_retries: int = 100
    stage_runner_factory: Callable[[], StageRunner] | None = None

    # Shared data staged by put(), uploaded to disk at the start of execute()
    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    # Handle to the coordinator job (for termination on retry/shutdown)
    _coordinator_job: JobHandle | None = field(default=None, repr=False)
    # NOTE: execute calls increment this at the very beginning
    _pipeline_id: int = field(default=-1, repr=False)

    def __post_init__(self):
        if self.client is None:
            from fray.client import current_client

            self.client = current_client()

        if self.max_workers is None:
            from fray.local_backend import LocalClient

            if isinstance(self.client, LocalClient):
                self.max_workers = os.cpu_count() or 1
            else:
                # Default to 128 for distributed, but allow override
                env_val = os.environ.get("ZEPHYR_MAX_WORKERS")
                self.max_workers = int(env_val) if env_val else 128

        if self.no_workers_timeout is None:
            self.no_workers_timeout = 6 * 60 * 60  # 6 hours

        if self.chunk_storage_prefix is None:
            # TODO: consider increasing TTL for long-running pipelines (e.g. multi-day fuzzy dedup)
            self.chunk_storage_prefix = marin_temp_bucket(ttl_days=1, prefix="zephyr")

        if self.stage_runner_factory is None:
            self.stage_runner_factory = _default_stage_runner_factory_for(self.client)

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
        verbose: bool = False,
        dry_run: bool = False,
    ) -> ZephyrExecutionResult:
        """Execute a dataset pipeline.

        Submits a coordinator *job* that creates coordinator and worker
        actors as child jobs, runs the pipeline, and writes results to
        disk. If the coordinator job dies (e.g., VM preemption), the
        pipeline is retried up to ``max_execution_retries`` times.
        Application errors (``ZephyrWorkerError``) are never retried.

        Returns:
            A ``ZephyrExecutionResult`` containing the flat list of results
            produced by the terminal stage and the aggregated counters from
            the run. Callers that only care about the results should access
            ``.results``; counters are exposed for callers that want to
            persist or surface them.
        """
        plan = compute_plan(dataset)
        if verbose or dry_run:
            _print_plan(dataset.operations, plan)
        if dry_run:
            return ZephyrExecutionResult(results=[], counters={})

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

            config_path = f"{self.chunk_storage_prefix}/{execution_id}/job-config.pkl"
            result_path = f"{self.chunk_storage_prefix}/{execution_id}/results.pkl"

            try:
                self._upload_shared_data(execution_id)

                config = _CoordinatorJobConfig(
                    plan=plan,
                    execution_id=execution_id,
                    chunk_storage_prefix=self.chunk_storage_prefix,
                    no_workers_timeout=self.no_workers_timeout,
                    max_workers=self.max_workers,
                    worker_resources=self.resources,
                    name=self.name,
                    pipeline_id=self._pipeline_id,
                    stage_runner_factory=self.stage_runner_factory,
                )
                ensure_parent_dir(config_path)
                with open_url(config_path, "wb") as f:
                    f.write(cloudpickle.dumps(config))

                job_name = f"zephyr-{self.name}-p{self._pipeline_id}-a{attempt}"
                # The wrapper job just blocks on child actors; real
                # resources are requested by the coordinator/worker children.
                # Set the context var so the coordinator job inherits self.client
                # instead of auto-detecting (which may pick a different backend).
                from fray.client import set_current_client

                with set_current_client(self.client):
                    self._coordinator_job = self.client.submit(
                        JobRequest(
                            name=job_name,
                            entrypoint=Entrypoint.from_callable(
                                _run_coordinator_job,
                                args=(config_path, result_path),
                            ),
                            resources=self.coordinator_resources,
                        )
                    )

                backoff.reset()
                logger.info("Coordinator job submitted: %s (job_id=%s)", job_name, self._coordinator_job.job_id)

                self._coordinator_job.wait(timeout=None, raise_on_failure=True)

                # Read results written by the coordinator job.
                # This must succeed — the job completed successfully.
                payload = _read_coordinator_result(result_path)
                if isinstance(payload, Exception):
                    raise payload
                return payload

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
    """Reshard shard refs by output shard index without loading data.

    Only supported on ListShards (non-scatter data).
    """
    output_by_shard: dict[int, list[Iterable]] = defaultdict(list)
    output_idx = 0
    for shard in shards:
        if not isinstance(shard, ListShard):
            raise ValueError("Reshard is only supported on ListShard (non-scatter data)")
        for chunk in shard.refs:
            output_by_shard[output_idx].append(chunk)
            output_idx = (output_idx + 1) % num_shards
    return [ListShard(refs=output_by_shard.get(idx, [])) for idx in range(num_shards)]


def _build_source_shards(source_items: list[SourceItem]) -> list[Shard]:
    """Build shard data from source items.

    Each source item becomes a single-element chunk in its assigned shard.
    """
    items_by_shard: dict[int, list] = defaultdict(list)
    for item in source_items:
        items_by_shard[item.shard_idx].append(item.data)

    num_shards = max(items_by_shard.keys()) + 1 if items_by_shard else 0
    shards: list[Shard] = []
    for i in range(num_shards):
        shards.append(ListShard(refs=[MemChunk(items=items_by_shard.get(i, []))]))

    return shards


def _compute_tasks_from_shards(
    shard_refs: list[Shard],
    stage,
    stage_name: str,
    aux_per_shard: list[dict[int, Shard]] | None = None,
) -> list[ShardTask]:
    """Convert shard references into ShardTasks for the coordinator."""
    total = len(shard_refs)
    tasks = []

    for i, shard in enumerate(shard_refs):
        aux_shards = None
        if aux_per_shard and aux_per_shard[i]:
            aux_shards = aux_per_shard[i]

        tasks.append(
            ShardTask(
                shard_idx=i,
                total_shards=total,
                shard=shard,
                operations=stage.operations,
                stage_name=stage_name,
                aux_shards=aux_shards,
            )
        )

    return tasks


def _get_stage_description(stage: PhysicalStage) -> str:
    """Get a description of a stage, including optional hints."""
    name = stage.stage_name()
    hint_parts = []
    if stage.stage_type == StageType.RESHARD:
        hint_parts.append(f"reshard→{stage.output_shards}")
    for op in stage.operations:
        if isinstance(op, Join) and op.right_plan is not None:
            hint_parts.append(f"join({len(op.right_plan.source_items)} items)")
    hint_str = f" [{', '.join(hint_parts)}]" if hint_parts else ""
    return f"{name}{hint_str}"


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

        stage_desc = _get_stage_description(stage)
        logger.info(f"  {i}. {stage_desc}")

    logger.info("\n=== End Plan ===\n")
