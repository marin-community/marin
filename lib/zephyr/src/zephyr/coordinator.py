# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator actor for Zephyr pipelines.

``ZephyrCoordinator`` owns pipeline execution: it holds the task queues,
dispatches shards to workers that have capacity, tracks in-flight work and
failures, and aggregates counters. ``run_pipeline`` is safe to call
concurrently — each call gets its own ``_PipelineExecution`` and the worker
pool serves them all. A failing pipeline fails only its own execution.

Workers reach the coordinator through the pull protocol in ``PullStatus`` /
``PullTask``; see ``zephyr.worker``. The pool hosting both is ``zephyr.pool``.
"""

import enum
import logging
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import cloudpickle
from fray.actor import ActorHandle, current_actor
from rigging import telltale
from rigging.filesystem import StoragePath
from rigging.timing import ExponentialBackoff, RateLimiter, log_time

from zephyr.plan import (
    Join,
    PhysicalOp,
    PhysicalPlan,
    PhysicalStage,
    Scatter,
    Shard,
    SourceItem,
    StageType,
)
from zephyr.shuffle import ListShard, MemChunk
from zephyr.stage_io import (
    ShardTask,
    TaskResult,
    ZephyrTaskResources,
    ZephyrWorkerError,
    _ensure_picklable_exception,
    _stage_throughput,
)
from zephyr.stats import (
    StatsWriter,
    ZephyrWorkerStatStatus,
    _push_iris_task_status,
)
from zephyr.worker_context import CounterEntry, CounterSnapshot, merge_counter_entries
from zephyr.writers import ensure_parent_dir

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

# Typical status text for a 6-stage pipeline is ~300 chars.
MAX_STATUS_TEXT_LENGTH = 1000

ZEPHYR_PROGRESS_TIME_METRIC = "zephyr_progress_time_seconds"

# Max pipelines one coordinator runs concurrently. Each active
# run_pipeline occupies one actor dispatch slot for its whole duration, so this
# bound keeps slots free for the pull_task/report_*/heartbeat calls that feed
# workers. Excess submissions are rejected rather than queued.
MAX_CONCURRENT_PIPELINES = 16


class ShardFailureKind(enum.StrEnum):
    """TASK failures count toward MAX_SHARD_FAILURES; INFRA failures (preemption) do not."""

    TASK = enum.auto()
    INFRA = enum.auto()


def _generate_execution_id() -> str:
    """Generate unique ID for this execution to avoid conflicts."""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _execution_result_path(prefix: str, execution_id: str) -> str:
    """Where a coordinator persists one pipeline's result payload."""
    return f"{prefix}/{execution_id}/results.pkl"


def _cleanup_execution(prefix: str, execution_id: str) -> None:
    """Remove all chunk files for an execution."""
    exec_dir = StoragePath(f"{prefix}/{execution_id}")

    with log_time(f"Cleaning up execution directory {exec_dir}"):
        if exec_dir.exists():
            try:
                exec_dir.rmtree()
            except Exception as e:
                logger.warning(f"Failed to cleanup chunks at {exec_dir}: {e}")


class WorkerState(enum.StrEnum):
    ACTIVE = enum.auto()
    FAILED = enum.auto()
    DONE = enum.auto()


@dataclass
class _InFlightEntry:
    """Coordinator's record of one in-flight task."""

    task: ShardTask
    attempt: int
    worker_id: str


class PullStatus(enum.StrEnum):
    """Control signals returned by ``ZephyrCoordinator.pull_task``.

    - ``RUN_TASK``: a task is available; the ``PullTask`` payload carries it.
    - ``NO_WORK_BACKOFF``: nothing dispatchable right now; worker sleeps and retries.
    - ``SHUTDOWN``: coordinator is shutting down; worker exits.
    """

    RUN_TASK = enum.auto()
    NO_WORK_BACKOFF = enum.auto()
    SHUTDOWN = enum.auto()


@dataclass(frozen=True)
class PullTask:
    """Task payload in a ``pull_task`` response when status is ``RUN_TASK``."""

    task: ShardTask
    attempt: int
    config: dict[str, Any]


class CoordinatorUnreachable(RuntimeError):
    """Worker lost contact with the coordinator. Retryable at the iris task level."""


@dataclass
class JobStatus:
    stage: str
    completed: int
    total: int
    retries: int
    in_flight: int
    queue_depth: int
    done: bool
    fatal_error: str | None
    workers: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CoordinatorInfo:
    """Coordinator config a connecting driver needs to submit pipelines.

    Returned by ``ZephyrCoordinator.describe()`` so a driver stages shared data
    and cleans up under the coordinator's own ``chunk_prefix``.
    """

    chunk_prefix: str


@dataclass(frozen=True)
class ZephyrExecutionResult:
    """Result of running a Zephyr pipeline.

    This is also the wire format the coordinator pickles into the result file,
    so callers of ``ZephyrContext.execute`` receive it as-is.

    Attributes:
        results: Flat list of items produced by the terminal stage of the
            pipeline (e.g. output file paths for write stages).
        counters: Aggregated counter values from the run, including built-in
            zephyr counters (e.g. ``zephyr/records_in``) and any user counters
            recorded via ``zephyr.counters.pipeline``.
    """

    results: list
    counters: dict[str, int | float]


@dataclass
class _PipelineExecution:
    """Coordinator-side state for one pipeline execution.

    The coordinator drives multiple executions concurrently; everything
    scoped to a single pipeline lives here. Worker
    membership, heartbeats, and in-flight counter snapshots stay on the
    coordinator — the worker pool is shared across executions.
    """

    execution_id: str
    # Per-task costs for this pipeline, supplied by the submitting driver so
    # pipelines with different resource needs can share one worker pool.
    map_cost: ZephyrTaskResources
    reduce_cost: ZephyrTaskResources
    task_queue: deque[ShardTask] = field(default_factory=deque)
    results: dict[int, TaskResult] = field(default_factory=dict)
    stage_name: str = ""
    # Index of the currently active stage. For joins and reshards, the parent's index.
    current_stage_index: int = 0
    plan_stages: list = field(default_factory=list)
    total_shards: int = 0
    completed_shards: int = 0
    retries: int = 0
    # Keyed by shard_idx so a single worker can have multiple tasks in flight.
    in_flight: dict[int, _InFlightEntry] = field(default_factory=dict)
    # task_attempts: monotonic generation for stale-result rejection (bumps on
    # every requeue). task_error_attempts: TASK-only counter bounded by
    # max_shard_failures. task_infra_attempts: INFRA-while-in-flight counter
    # bounded by max_shard_infra_failures.
    task_attempts: dict[int, int] = field(default_factory=dict)
    task_error_attempts: dict[int, int] = field(default_factory=dict)
    task_infra_attempts: dict[int, int] = field(default_factory=dict)
    fatal_error: str | None = None
    shard_errors: dict[int, list[str]] = field(default_factory=dict)
    # Set when a stage may have completed (result, failure, or abort) so
    # ``_wait_for_stage`` wakes immediately instead of sleeping out its backoff.
    stage_done: threading.Event = field(default_factory=threading.Event)
    # True once the final task-producing stage has been loaded, so a drained
    # queue means no further task can be dispatched for this pipeline.
    is_last_stage: bool = False
    # Completed-task counter snapshots; accumulate across this execution's stages.
    completed_counters: list[CounterSnapshot] = field(default_factory=list)
    # Set at each _start_stage so status logs show throughput since stage start.
    stage_monotonic_start: float | None = None
    done: bool = False

    def start_stage(
        self,
        stage_name: str,
        current_stage_index: int,
        tasks: list[ShardTask],
        *,
        is_last_stage: bool,
    ) -> None:
        """Load one stage's tasks and reset the per-stage bookkeeping.

        Counters and plan_stages span the whole execution and survive; anything
        scoped to a single stage starts over. Callers hold the coordinator lock.
        """
        self.task_queue = deque(tasks)
        self.results = {}
        self.in_flight = {}
        self.stage_name = stage_name
        self.current_stage_index = current_stage_index
        self.total_shards = len(tasks)
        self.completed_shards = 0
        self.retries = 0
        self.task_attempts = {task.shard_idx: 0 for task in tasks}
        self.task_error_attempts = {task.shard_idx: 0 for task in tasks}
        # INFRA failures seen while this shard was in flight on the dying
        # worker — bounded by max_shard_infra_failures so a shard that
        # deterministically crashes its worker (native SIGSEGV, OOM) aborts
        # instead of retrying forever.
        self.task_infra_attempts = {task.shard_idx: 0 for task in tasks}
        self.shard_errors = {}
        self.fatal_error = None
        self.is_last_stage = is_last_stage
        self.stage_monotonic_start = time.monotonic()
        self.stage_done.clear()

    def finish(self) -> None:
        """Mark the execution done and release the state only a live run needs.

        Completed counters stay: the coordinator folds them into its totals
        before dropping the execution. Callers hold the coordinator lock.
        """
        self.done = True
        self.task_queue.clear()
        self.in_flight.clear()
        self.results = {}

    def merged_counters(self) -> dict[str, CounterEntry]:
        """Fold this execution's completed snapshots into one set of entries."""
        merged, conflicted = merge_counter_entries(
            (k, entry) for snap in self.completed_counters for k, entry in snap.counters.items()
        )
        return {k: e for k, e in merged.items() if k not in conflicted}


def _aggregate_counter_snapshots(
    snapshots: Iterable[CounterSnapshot],
    stage: str | None,
) -> dict[str, int | float]:
    """Merge counter snapshots into totals, honoring per-key aggregation hints.

    Folds with :func:`merge_counter_entries`, the same reducer the worker uses
    across its concurrent runners, so an AVERAGE counter is weighted by each
    entry's observation count. Counters whose snapshots disagree on an
    aggregation are dropped rather than raised on — stats collection never
    interrupts execution. When ``stage`` is given, only entries recorded under
    that stage label are included.
    """
    merged, conflicted = merge_counter_entries(
        (k, entry) for snap in snapshots for k, entry in snap.counters.items() if stage is None or entry.stage == stage
    )
    return {k: e.value for k, e in merged.items() if k not in conflicted}


class ZephyrCoordinator:
    """Central coordinator actor that owns and manages the worker pool.

    The coordinator runs a background loop for heartbeat checking and manages
    pipeline execution internally. Workers poll the coordinator for tasks
    until receiving a SHUTDOWN signal.

    ``run_pipeline`` is safe to call concurrently: each call gets its own
    ``_PipelineExecution`` and the shared worker pool serves all active
    executions, dispatching tasks wherever their cost fits a worker's free
    resources. A pipeline failure only fails its own execution; the
    coordinator keeps serving the rest. Workers are long-lived: they idle
    between stages and between pipelines, and exit only on ``shutdown()``.
    """

    def __init__(
        self,
        chunk_prefix: str,
        worker_resources: ZephyrTaskResources,
        no_workers_timeout: float = 60.0,
        heartbeat_timeout: float = 120.0,
        max_shard_failures: int = MAX_SHARD_FAILURES,
        max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES,
        drain_idle_workers: bool = False,
    ) -> None:
        # Pipeline executions keyed by execution_id, insertion-ordered. All
        # per-pipeline state lives in the _PipelineExecution values.
        self._executions: dict[str, _PipelineExecution] = {}
        self._worker_resources = worker_resources
        # Set by a pool that will not receive more pipelines. Such a pool can
        # hand SHUTDOWN to workers that go idle during the last stage's tail,
        # releasing cluster capacity while stragglers finish. A standing pool
        # must never do this — its next pipeline needs those workers.
        self._drain_idle_workers = drain_idle_workers
        # Set when the coordinator as a whole can no longer make progress
        # (worker job permanently dead, maintenance loop crash). Fails all
        # active executions and rejects new ones.
        self._pool_error: str | None = None
        # Rotates which execution pull_task scans first, for cross-pipeline fairness.
        self._pull_offset: int = 0
        # Cumulative counters from executions already dropped, folded into one
        # set so a long-lived pool's memory does not grow with pipeline count.
        self._retired_counters: dict[str, CounterEntry] = {}

        # Worker management state (workers self-register via register_worker)
        self._worker_states: dict[str, WorkerState] = {}
        self._last_seen: dict[str, float] = {}
        self._chunk_prefix = chunk_prefix
        self._no_workers_timeout = no_workers_timeout
        self._heartbeat_timeout = heartbeat_timeout
        self._max_shard_failures = max_shard_failures
        self._max_shard_infra_failures = max_shard_infra_failures
        # Per-worker in-flight counter snapshots. Each snapshot carries a
        # monotonic generation so the coordinator can discard stale or
        # out-of-order heartbeats.
        self._worker_counters: dict[str, CounterSnapshot] = {}
        self._worker_handles: dict[str, ActorHandle] = {}
        self._worker_group: Any = None  # ActorGroup, set via set_worker_group()
        self._coordinator_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        # Unix time of the newest stage start or shard completion across all
        # executions, published as ZEPHYR_PROGRESS_TIME_METRIC so a stalled
        # coordinator is visible. Per-pipeline progress lives on the executions.
        self._progress_time_seconds: float = 0.0

        # Lock for accessing coordinator state from background thread
        self._lock = threading.Lock()

        # Throttle Iris task-status pushes; the coordinator loop ticks more
        # frequently than the UI needs to refresh.
        self._task_stats_limiter = RateLimiter(interval_seconds=10.0)

        actor_ctx = current_actor()
        self._name = f"{actor_ctx.group_name}"

        self._stats_writer = StatsWriter.connect()

        logger.info("Coordinator initialized")

        # Start coordinator background loop (heartbeat checking only)
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop, daemon=True, name="zephyr-coordinator-loop"
        )
        self._coordinator_thread.start()

    def set_worker_group(self, worker_group: Any) -> None:
        """Set the worker ActorGroup so the coordinator can detect permanent worker death."""
        self._worker_group = worker_group

    def register_worker(self, worker_id: str, worker_handle: ActorHandle) -> None:
        """Called by workers when they come online to register with coordinator.

        Handles re-registration from reconstructed workers (e.g. after node
        preemption) by updating the stale handle and resetting worker state.

        Returns the current stage epoch.
        """
        with self._lock:
            if worker_id in self._worker_handles:
                logger.info("Worker %s re-registering (likely reconstructed), updating handle", worker_id)
                self._worker_handles[worker_id] = worker_handle
                self._worker_states[worker_id] = WorkerState.ACTIVE
                self._last_seen[worker_id] = time.monotonic()
                # NOTE: if there was a task assigned to the worker, there's a race condition between marking
                # the worker as unhealthy via heartbeat and re-registration. If we do not requeue we may silently
                # lose tasks.
                self._maybe_requeue_worker_tasks(worker_id)
            else:
                self._worker_handles[worker_id] = worker_handle
                self._worker_states[worker_id] = WorkerState.ACTIVE
                self._last_seen[worker_id] = time.monotonic()
                logger.info("Worker %s registered, total: %d", worker_id, len(self._worker_handles))

    def deregister_worker(self, worker_id: str) -> None:
        """Remove a sub-worker that has finished its stage pool."""
        with self._lock:
            self._worker_handles.pop(worker_id, None)
            self._worker_states.pop(worker_id, None)
            self._last_seen.pop(worker_id, None)

    def _coordinator_loop(self) -> None:
        """Background loop for heartbeat checking and worker job monitoring."""
        last_log_time = 0.0

        while not self._shutdown_event.is_set():
            if sys.is_finalizing():
                return
            try:
                self.check_heartbeats(self._heartbeat_timeout)
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
        """Abort all executions if the worker job has permanently terminated."""
        if self._worker_group is None or self._pool_error is not None:
            return
        if self._drain_idle_workers:
            # Workers release themselves once the last stage drains, so a
            # terminal worker job is only a crash while shards are outstanding.
            # With none outstanding it is the expected ending. Before the first
            # pipeline there is nothing to abort either, so this stays quiet
            # until a pipeline actually has work in flight.
            with self._lock:
                outstanding = any(r.completed_shards < r.total_shards for r in self._executions.values() if not r.done)
            if not outstanding:
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
        with self._lock:
            return any(
                not run.done and run.total_shards > 0 and run.completed_shards < run.total_shards
                for run in self._executions.values()
            )

    def _publish_telltale(self) -> None:
        """Publish pipeline counters as gauges on the coordinator's telltale page.

        The coordinator is the only process that both holds the aggregated
        counters and serves the routes: shards run in short-lived subprocesses
        under ``SubprocessRunner`` (the distributed default), whose registries
        nobody scrapes.
        """
        for name, value in self.get_counters().items():
            telltale.publish_gauge(name, value, f"zephyr counter {name}")
        with self._lock:
            progress_time_seconds = self._progress_time_seconds
        telltale.publish_gauge(
            ZEPHYR_PROGRESS_TIME_METRIC,
            progress_time_seconds,
            "Unix time of the current stage start or most recent shard completion",
        )

    def _build_status_md(self) -> tuple[str, str]:
        """Render pipeline progress as ``(detail, summary)`` markdown."""
        with self._lock:
            snapshot = [
                (
                    run.execution_id,
                    list(run.plan_stages),
                    run.current_stage_index,
                    run.completed_shards,
                    run.total_shards,
                    len(run.in_flight),
                    len(run.task_queue),
                )
                for run in self._executions.values()
                if not run.done
            ]

        detail_lines: list[str] = []
        summary_lines: list[str] = []
        for execution_id, plan_stages, stage_index, completed, total, in_flight, queued in snapshot:
            detail_lines.append(f"**{execution_id}**")
            for idx, stage in enumerate(plan_stages):
                stage_desc = _get_stage_description(stage)
                detail_lines.append(f"- **{stage_desc}**" if idx == stage_index else f"- {stage_desc}")
            pct = int(100 * completed / total) if total > 0 else 0
            detail_lines.append(
                f"**Shards** — {completed}/{total} complete ({pct}%), {in_flight} in-flight, {queued} queued\n"
            )
            current_desc = _get_stage_description(plan_stages[stage_index]) if plan_stages else ""
            summary_lines.append(
                f"**{current_desc}** ({stage_index + 1}/{len(plan_stages)}) — {completed}/{total} shards ({pct}%)"
            )

        detail_md = "\n".join(detail_lines)[:MAX_STATUS_TEXT_LENGTH] or "idle"
        summary_md = "  \n".join(summary_lines)[:MAX_STATUS_TEXT_LENGTH] or "idle"
        return detail_md, summary_md

    def _report_task_stats(self) -> None:
        """Publish pipeline progress to telltale, and to the Iris coordinator if available."""
        detail_md, summary_md = self._build_status_md()
        # Eager, unlike the Iris push below: the telltale page is process-local
        # and serves every run, including the ones outside an Iris task that the
        # push skips entirely.
        telltale.set_status(summary_md)
        self._publish_telltale()
        _push_iris_task_status(self._task_stats_limiter, lambda: (detail_md, summary_md))

    def _log_status(self) -> None:
        with self._lock:
            states = list(self._worker_states.values())
            lines = [
                (
                    run.execution_id,
                    run.stage_name,
                    run.completed_shards,
                    run.total_shards,
                    len(run.in_flight),
                    len(run.task_queue),
                    {idx: att for idx, att in run.task_attempts.items() if att > 0},
                    run.stage_monotonic_start,
                    list(run.completed_counters) + list(self._worker_counters.values()),
                )
                for run in self._executions.values()
                if not run.done
            ]
        alive = sum(1 for s in states if s == WorkerState.ACTIVE)
        dead = sum(1 for s in states if s in {WorkerState.FAILED, WorkerState.DONE})

        for execution_id, stage_name, completed, total, in_flight, queued, retried, stage_start, snaps in lines:
            base_msg = "[%s] [%s] %d/%d complete, %d in-flight, %d queued, %d/%d workers alive, %d dead"
            base_args = (
                execution_id,
                stage_name,
                completed,
                total,
                in_flight,
                queued,
                alive,
                len(self._worker_handles),
                dead,
            )

            # Map-only stages don't yield through ``_wrap_stage_stats`` and never
            # populate these counters. Drop the items/bytes_processed segment for
            # those stages. In-flight snapshots are stage-filtered, not
            # execution-filtered — two concurrent pipelines running an
            # identically-labelled stage share the live segment (log-only).
            elapsed = time.monotonic() - (stage_start or time.monotonic())
            throughput = _stage_throughput(_aggregate_counter_snapshots(snaps, stage_name), elapsed)
            if throughput is not None:
                logger.info(base_msg + "; %s", *base_args, throughput)
            else:
                logger.info(base_msg, *base_args)
            if retried:
                attempts_histogram = dict(sorted(Counter(retried.values()).items()))
                logger.warning("[%s] Shards retried (attempts: shard count): %s", execution_id, attempts_histogram)

    def _emit_stage_stat(self, run: _PipelineExecution, *, failed: bool = False) -> None:
        """Emit one ZephyrStageStat row to finelog at stage completion or failure."""
        with self._lock:
            stage_name = run.stage_name
            execution_id = run.execution_id
            total = run.total_shards
            stage_start = run.stage_monotonic_start
            elapsed = time.monotonic() - stage_start if stage_start else 0.0
            snaps = list(run.completed_counters)
        status = ZephyrWorkerStatStatus.FAILED if failed else ZephyrWorkerStatStatus.END
        stage_counters = _aggregate_counter_snapshots(snaps, stage_name)
        self._stats_writer.emit_stage_stat(stage_counters, stage_name, execution_id, elapsed, total, status)

    def _record_shard_failure(
        self,
        run: _PipelineExecution,
        shard_idx: int,
        worker_id: str,
        kind: ShardFailureKind,
        error_info: str | None = None,
    ) -> bool:
        """Requeue a specific in-flight shard; abort its execution if a per-shard cap is hit.

        TASK errors are bounded by ``MAX_SHARD_FAILURES``. INFRA failures
        observed while the *same* shard was in flight are bounded by
        ``MAX_SHARD_INFRA_FAILURES`` so a payload that deterministically
        crashes its worker (native SIGSEGV, OOM) doesn't loop forever now
        that shard execution is in-process.

        Must be called with lock held. Returns True if the execution was aborted.
        """
        entry = run.in_flight.pop(shard_idx, None)

        # Zero counters but keep the generation watermark so late heartbeats
        # from the old task are rejected.
        existing = self._worker_counters.get(worker_id)
        if existing is not None:
            self._worker_counters[worker_id] = CounterSnapshot.empty(existing.generation)

        if entry is None:
            return False

        task = entry.task

        if error_info is not None:
            run.shard_errors.setdefault(shard_idx, []).append(error_info)

        # Bump generation regardless of kind so report_result rejects stale attempts.
        run.task_attempts[shard_idx] += 1
        # Wake _wait_for_stage on every accounted failure (requeue or abort);
        # the waiter re-checks fatal_error / completed counts after waking.
        run.stage_done.set()

        if kind is ShardFailureKind.TASK:
            run.task_error_attempts[shard_idx] += 1
            error_attempts = run.task_error_attempts[shard_idx]
            if error_attempts >= self._max_shard_failures:
                errors = run.shard_errors.get(shard_idx, [])
                error_detail = f"\nLast error:\n{errors[-1]}" if errors else ""
                logger.error(
                    "[%s] Shard %d has failed %d times (max %d), last failure on worker %s, aborting pipeline.",
                    run.execution_id,
                    shard_idx,
                    error_attempts,
                    self._max_shard_failures,
                    worker_id,
                )
                run.fatal_error = (
                    f"Shard {shard_idx} failed {error_attempts} times "
                    f"(max {self._max_shard_failures}), last failure on worker {worker_id}.{error_detail}"
                )
                return True

            logger.warning(
                "[%s] Shard %d failed on worker %s (task error %d/%d), re-queuing.",
                run.execution_id,
                shard_idx,
                worker_id,
                error_attempts,
                self._max_shard_failures,
            )
        else:
            run.task_infra_attempts[shard_idx] += 1
            infra_attempts = run.task_infra_attempts[shard_idx]
            if infra_attempts >= self._max_shard_infra_failures:
                logger.error(
                    "[%s] Shard %d has been in flight during %d infra failures (max %d); "
                    "treating as a deterministic crasher (likely native SIGSEGV / OOM in shard "
                    "code) and aborting pipeline. Last failure on worker %s.",
                    run.execution_id,
                    shard_idx,
                    infra_attempts,
                    self._max_shard_infra_failures,
                    worker_id,
                )
                run.fatal_error = (
                    f"Shard {shard_idx} crashed its worker {infra_attempts} times "
                    f"(max {self._max_shard_infra_failures} infra failures while in flight); "
                    f"last failure on worker {worker_id}."
                )
                return True

            logger.warning(
                "[%s] Shard %d requeued from worker %s due to infra failure (preemption/heartbeat). "
                "Total generation: %d, task errors so far: %d/%d, infra-while-in-flight: %d/%d.",
                run.execution_id,
                shard_idx,
                worker_id,
                run.task_attempts[shard_idx],
                run.task_error_attempts[shard_idx],
                self._max_shard_failures,
                infra_attempts,
                self._max_shard_infra_failures,
            )

        run.task_queue.append(task)
        run.retries += 1
        return False

    def _maybe_requeue_worker_tasks(self, worker_id: str) -> None:
        """Requeue all in-flight tasks for a worker as INFRA failures (preemption/heartbeat)."""
        for run in self._executions.values():
            shards_to_requeue = [
                shard_idx for shard_idx, entry in list(run.in_flight.items()) if entry.worker_id == worker_id
            ]
            for shard_idx in shards_to_requeue:
                self._record_shard_failure(run, shard_idx, worker_id, ShardFailureKind.INFRA)

    def pull_task(
        self,
        worker_id: str,
        available: ZephyrTaskResources,
    ) -> tuple[PullStatus, PullTask | None]:
        """Called by workers to get next task.

        Workers provide their current available resources so the coordinator
        can gate dispatch on worker capacity against the next task's
        requirements. Active executions are scanned round-robin so concurrent
        pipelines share the pool fairly; the first execution whose head task
        fits the worker wins.

        Args:
            worker_id: Unique ID for this worker.
            available: CPU and memory currently available on the worker.

        Returns:
            ``(status, work)`` where ``work`` is a ``PullTask`` when
            ``status`` is ``RUN_TASK`` and ``None`` for all other statuses.
        """
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            self._worker_states[worker_id] = WorkerState.ACTIVE

            if self._shutdown_event.is_set():
                self._worker_states[worker_id] = WorkerState.DONE
                return PullStatus.SHUTDOWN, None

            dispatchable = [r for r in self._executions.values() if not r.done and r.fatal_error is None]
            for i in range(len(dispatchable)):
                run = dispatchable[(self._pull_offset + i) % len(dispatchable)]
                if not run.task_queue or not available.can_fit(run.task_queue[0].cost):
                    continue
                self._pull_offset = (self._pull_offset + i + 1) % len(dispatchable)
                task = run.task_queue.popleft()
                attempt = run.task_attempts[task.shard_idx]
                run.in_flight[task.shard_idx] = _InFlightEntry(task=task, attempt=attempt, worker_id=worker_id)
                config = {
                    "chunk_prefix": self._chunk_prefix,
                    "execution_id": run.execution_id,
                }
                return PullStatus.RUN_TASK, PullTask(task=task, attempt=attempt, config=config)

            if self._drain_idle_workers and self._worker_is_releasable_locked(worker_id):
                self._worker_states[worker_id] = WorkerState.DONE
                return PullStatus.SHUTDOWN, None

            # Otherwise the worker idles: more tasks can still arrive, from a
            # later stage of this pipeline or from the next pipeline.
            return PullStatus.NO_WORK_BACKOFF, None

    def _worker_is_releasable_locked(self, worker_id: str) -> bool:
        """True when no further task can be dispatched to this worker. Lock held.

        Every active execution must have loaded its last task-producing stage
        and drained its queue, and this worker must hold nothing in flight that
        could fail and be requeued onto it. A peer's requeued shard is picked up
        by a restarted worker, and ``_check_worker_group`` is the failsafe
        against the pool dying out entirely.
        """
        active = [r for r in self._executions.values() if not r.done]
        if not active:
            # No pipeline registered yet (the pool boots its workers before the
            # driver submits) or the pipeline already finished. Never release on
            # an empty registry: an `all()` over it is vacuously true, which
            # would retire the whole pool before its first task is queued.
            # Teardown is shutdown()'s job, not this predicate's.
            return False
        if not all(r.is_last_stage and not r.task_queue for r in active):
            return False
        return not any(e.worker_id == worker_id for r in active for e in r.in_flight.values())

    def _assert_in_flight_consistent(self, run: _PipelineExecution, worker_id: str, shard_idx: int) -> None:
        """Assert in_flight[shard_idx], if present, is owned by the reporting worker.

        Call only after verifying the report matches the current task attempt.
        Workers block on report_result/report_error before calling pull_task, so
        a current-attempt report should always match the in-flight owner when the
        entry is present. The entry may be absent if a heartbeat timeout already
        re-queued the task and the shard completed or moved on.
        """
        entry = run.in_flight.get(shard_idx)
        if entry is not None:
            assert entry.worker_id == worker_id, (
                f"in_flight mismatch for shard {shard_idx}: reported by {worker_id}, "
                f"but tracked as owned by {entry.worker_id}. "
                f"This indicates report_result/pull_task reordering — workers must block on report_result."
            )

    def report_result(
        self,
        worker_id: str,
        execution_id: str,
        shard_idx: int,
        attempt: int,
        result: TaskResult,
        counter_snapshot: CounterSnapshot,
    ) -> None:
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()

            run = self._executions.get(execution_id)
            if run is None:
                logger.warning(
                    "Ignoring result from worker %s for finished/unknown execution %s (shard %d)",
                    worker_id,
                    execution_id,
                    shard_idx,
                )
                # The task's runner is finished; drop its stale in-flight
                # snapshot so it stops polluting cross-pipeline totals until
                # the worker's next heartbeat lands.
                self._clear_worker_inflight_counters(worker_id, counter_snapshot.generation)
                return

            current_attempt = run.task_attempts.get(shard_idx, 0)
            if attempt != current_attempt:
                logger.warning(
                    f"Ignoring stale result from worker {worker_id} for shard {shard_idx} "
                    f"(attempt {attempt}, current {current_attempt})"
                )
                return

            if shard_idx in run.results:
                # Iris retries a transient actor RPC failure, so a result whose
                # reply was lost arrives twice with the same attempt. Counting
                # it again would let the stage finish while another shard is
                # still running, silently dropping that shard's output.
                logger.warning(
                    "Ignoring duplicate result from worker %s for shard %d (attempt %d)",
                    worker_id,
                    shard_idx,
                    attempt,
                )
                return

            self._assert_in_flight_consistent(run, worker_id, shard_idx)

            run.results[shard_idx] = result
            run.completed_shards += 1
            self._progress_time_seconds = time.time()
            run.in_flight.pop(shard_idx, None)
            run.completed_counters.append(counter_snapshot)
            # Zero the in-flight counters but keep the generation watermark
            # so late heartbeats from this task are rejected.
            self._clear_worker_inflight_counters(worker_id, counter_snapshot.generation)
            run.stage_done.set()

    def _clear_worker_inflight_counters(self, worker_id: str, generation: int) -> None:
        """Zero a worker's in-flight snapshot, keeping the generation watermark.

        Called when a task's runner is done, so late heartbeats from that task
        (strictly-lower generation) are rejected and the finished task's live
        counters stop being folded into cross-pipeline totals. Lock must be held.
        """
        existing = self._worker_counters.get(worker_id)
        watermark = max(generation, existing.generation) if existing is not None else generation
        self._worker_counters[worker_id] = CounterSnapshot.empty(watermark)

    def report_error(self, worker_id: str, execution_id: str, shard_idx: int, attempt: int, error_info: str) -> None:
        """Worker reports a task failure. Re-queues up to MAX_SHARD_FAILURES."""
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()

            run = self._executions.get(execution_id)
            if run is None:
                logger.warning(
                    "Ignoring error from worker %s for finished/unknown execution %s (shard %d)",
                    worker_id,
                    execution_id,
                    shard_idx,
                )
                # Drop the finished task's stale in-flight snapshot. No incoming
                # generation here, so just re-stamp the existing watermark.
                existing = self._worker_counters.get(worker_id)
                if existing is not None:
                    self._worker_counters[worker_id] = CounterSnapshot.empty(existing.generation)
                return

            current_attempt = run.task_attempts.get(shard_idx, 0)
            if attempt != current_attempt:
                logger.warning(
                    f"Ignoring stale error from worker {worker_id} for shard {shard_idx} "
                    f"(attempt {attempt}, current {current_attempt})"
                )
                return

            self._assert_in_flight_consistent(run, worker_id, shard_idx)
            self._record_shard_failure(run, shard_idx, worker_id, ShardFailureKind.TASK, error_info)

    def heartbeat(self, worker_id: str, counter_snapshot: CounterSnapshot | None = None) -> None:
        self._last_seen[worker_id] = time.monotonic()
        if counter_snapshot is not None:
            with self._lock:
                existing = self._worker_counters.get(worker_id)
                if existing is None or counter_snapshot.generation > existing.generation:
                    self._worker_counters[worker_id] = counter_snapshot

    def get_status(self) -> JobStatus:
        """Aggregate status across all active executions plus worker health."""
        with self._lock:
            active = [r for r in self._executions.values() if not r.done]
            return JobStatus(
                stage="; ".join(r.stage_name for r in active if r.stage_name),
                completed=sum(r.completed_shards for r in active),
                total=sum(r.total_shards for r in active),
                retries=sum(r.retries for r in active),
                in_flight=sum(len(r.in_flight) for r in active),
                queue_depth=sum(len(r.task_queue) for r in active),
                done=self._shutdown_event.is_set(),
                fatal_error=self._fatal_error_locked(),
                workers={
                    wid: {
                        "state": state.value,
                        "last_seen_ago": time.monotonic() - self._last_seen.get(wid, 0),
                    }
                    for wid, state in self._worker_states.items()
                },
            )

    def describe(self) -> CoordinatorInfo:
        """Coordinator config a connecting driver needs to submit pipelines."""
        return CoordinatorInfo(chunk_prefix=self._chunk_prefix)

    def is_shutdown(self) -> bool:
        """True once shutdown() ran; the pool's serve loop polls this to exit."""
        return self._shutdown_event.is_set()

    def has_outstanding_work(self) -> bool:
        """True while some active execution still has shards to finish.

        The pool job's serve loop uses this to tell a worker group that drained
        on purpose from one that died: only the latter is a failure.
        """
        with self._lock:
            return any(r.completed_shards < r.total_shards for r in self._executions.values() if not r.done)

    def is_ready(self) -> bool:
        """True once the pool's workers exist and this coordinator can serve.

        ``_ZephyrPool.start()`` waits on this rather than on the actor merely
        answering: the coordinator is hosted before its worker group is created,
        so a reachable coordinator does not yet mean a usable pool.
        """
        return self._worker_group is not None and not self._shutdown_event.is_set()

    def get_counters(self, worker_id: str | None = None, *, stage: str | None = None) -> dict[str, int | float]:
        """Return counter values, optionally filtered to a single worker or stage.

        Args:
            worker_id: If provided, return the latest snapshot for this worker
                only. If None, return totals derived from all registered
                executions' completed snapshots plus in-flight snapshots,
                applying per-key aggregation hints.
            stage: If provided, only include entries with ``entry.stage == stage``.
                If None (default), include all entries regardless of stage.

        Snapshots are folded with :func:`merge_counter_entries`, the same
        reducer the worker uses to combine its concurrent runners, so an
        AVERAGE counter is weighted by each entry's observation count rather
        than treating one shard's single sample as equal to another's
        thousand.

        If snapshots disagree on a counter's aggregation (only possible for
        user counters that reuse a name with different ``set_aggregation``
        modes), the counter is omitted from the result and a warning is
        logged — stats collection never raises into the execution path.

        A finished execution is folded into a single retained snapshot, so its
        totals survive the execution being dropped from the registry.
        """
        with self._lock:
            if worker_id is not None:
                snap = self._worker_counters.get(worker_id)
                if snap is None:
                    return {}
                return {k: e.value for k, e in snap.counters.items() if stage is None or e.stage == stage}

            all_snaps = (
                [CounterSnapshot(counters=dict(self._retired_counters), generation=0)]
                + [snap for run in self._executions.values() for snap in run.completed_counters]
                + list(self._worker_counters.values())
            )

        return _aggregate_counter_snapshots(all_snaps, stage)

    def _fatal_error_locked(self) -> str | None:
        if self._pool_error is not None:
            return self._pool_error
        return next((r.fatal_error for r in self._executions.values() if r.fatal_error), None)

    def get_fatal_error(self) -> str | None:
        """Pool-level error if any, else the first execution's fatal error."""
        with self._lock:
            return self._fatal_error_locked()

    def abort(self, reason: str) -> None:
        """Fail the coordinator pool: every active execution aborts immediately.

        Called when the coordinator as a whole cannot make progress — the
        worker job terminated permanently (e.g. all retries exhausted after
        OOM) or the maintenance loop crashed. New run_pipeline calls are
        rejected afterwards.
        """
        with self._lock:
            if self._pool_error is None:
                logger.error("Coordinator aborted: %s", reason)
                self._pool_error = reason
            for run in self._executions.values():
                if run.fatal_error is None:
                    run.fatal_error = reason
                run.stage_done.set()

    def _start_stage(
        self,
        run: _PipelineExecution,
        stage_name: str,
        current_stage_index: int,
        tasks: list[ShardTask],
        is_last_stage: bool = False,
    ) -> None:
        """Load a new stage's tasks into the execution's queue."""
        with self._lock:
            run.start_stage(stage_name, current_stage_index, tasks, is_last_stage=is_last_stage)
            self._progress_time_seconds = time.time()

    def _wait_for_stage(self, run: _PipelineExecution) -> None:
        """Block until the execution's current stage completes or errors."""
        backoff = ExponentialBackoff(initial=0.1, maximum=1.0)
        last_log_completed = -1
        start_time = time.monotonic()
        all_dead_since: float | None = None
        no_workers_timeout = self._no_workers_timeout

        while True:
            with self._lock:
                error = run.fatal_error or self._pool_error
                if error:
                    raise ZephyrWorkerError(error)

                completed = run.completed_shards
                total = run.total_shards

                if completed >= total:
                    return

                # Count alive workers (READY or BUSY), not just total registered.
                # Dead/failed workers stay in _worker_handles but can't make progress.
                alive_workers = sum(1 for s in self._worker_states.values() if s == WorkerState.ACTIVE)

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
                logger.info("[%s] [%s] %d/%d tasks completed", run.execution_id, run.stage_name, completed, total)
                last_log_completed = completed
                backoff.reset()

            # Wake promptly on completions / errors / aborts; the timeout still
            # bounds the sleep so the no-alive-workers timer fires regardless.
            if run.stage_done.wait(timeout=backoff.next_interval()):
                run.stage_done.clear()

    def _collect_results(self, run: _PipelineExecution) -> dict[int, TaskResult]:
        """Return results for the completed stage."""
        with self._lock:
            return dict(run.results)

    def run_pipeline(
        self,
        plan: PhysicalPlan,
        execution_id: str,
        map_cost: ZephyrTaskResources,
        reduce_cost: ZephyrTaskResources,
    ) -> None:
        """Run one pipeline, blocking until done. The result goes to storage.

        Safe to call concurrently: each call gets its own execution state and
        the shared worker pool serves all active executions. ``map_cost`` /
        ``reduce_cost`` are the per-task resource costs baked into this
        pipeline's tasks, supplied by the submitting driver.
        """
        for kind, cost in (("map", map_cost), ("reduce", reduce_cost)):
            if not self._worker_resources.can_fit(cost):
                raise ValueError(
                    f"{kind} task cost {cost} exceeds per-worker resources {self._worker_resources}; "
                    "this pipeline could never schedule. Lower the pipeline's "
                    "`map/reduce_task_resources`."
                )

        with self._lock:
            if self._shutdown_event.is_set():
                # Workers already got SHUTDOWN from pull_task, so an accepted
                # pipeline would block in _wait_for_stage until
                # no_workers_timeout (6h by default). A driver reaching here
                # holds an endpoint whose pool the owner already tore down.
                raise ZephyrWorkerError("Coordinator is shut down; cannot accept new pipelines")
            if self._pool_error is not None:
                raise ZephyrWorkerError(f"Coordinator pool failed: {self._pool_error}")
            if execution_id in self._executions:
                raise ValueError(f"Execution {execution_id} is already registered")
            active = sum(1 for r in self._executions.values() if not r.done)
            if active >= MAX_CONCURRENT_PIPELINES:
                raise RuntimeError(
                    f"Coordinator already runs {active} concurrent pipelines (max {MAX_CONCURRENT_PIPELINES})"
                )
            run = _PipelineExecution(execution_id=execution_id, map_cost=map_cost, reduce_cost=reduce_cost)
            self._executions[execution_id] = run

        # With several pipelines in flight this names whichever started most
        # recently. The label is telemetry only, so a coarse value beats none.
        telltale.set_global_labels(source="zephyr", run=execution_id)

        result_path = _execution_result_path(self._chunk_prefix, execution_id)
        try:
            shards = _build_source_shards(plan.source_items)
            if not shards:
                self._persist_result(result_path, ZephyrExecutionResult(results=[], counters={}))
                return None

            last_worker_stage_idx = max(
                (i for i, st in enumerate(plan.stages) if st.stage_type != StageType.RESHARD),
                default=-1,
            )

            with self._lock:
                run.plan_stages = list(plan.stages)

            for stage_idx, stage in enumerate(plan.stages):
                if stage.stage_type == StageType.RESHARD:
                    shards = _reshard_refs(shards, stage.output_shards or len(shards))
                    continue

                aux_per_shard = self._compute_join_aux(run, stage.operations, shards, stage_idx)
                shards = self._run_worker_stage(
                    run,
                    stage,
                    shards,
                    stage_label=f"stage{stage_idx}-{stage.stage_name(max_length=40)}",
                    stage_index_for_state=stage_idx,
                    aux_per_shard=aux_per_shard,
                    is_last_stage=(stage_idx == last_worker_stage_idx),
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

            with self._lock:
                counters = _aggregate_counter_snapshots(list(run.completed_counters), None)
            self._persist_result(result_path, ZephyrExecutionResult(results=flat_result, counters=counters))
            return None
        except Exception as e:
            # Persist the normalized exception so the driver can recover the
            # original type even when the actor transport cannot carry it.
            with suppress(Exception):
                self._persist_result(result_path, _ensure_picklable_exception(e))
            raise
        finally:
            with self._lock:
                run.finish()
                # Drop the finished execution so coordinator state does not grow
                # without bound. Late reports for it are logged and ignored.
                # Its counters are folded into one retained snapshot first, so
                # coordinator totals still cover pipelines that already ended.
                retained = run.merged_counters()
                if retained:
                    # Fold into the single cumulative snapshot rather than
                    # appending one per pipeline: a standing pool runs
                    # indefinitely, and a growing list would leak and make every
                    # counter read slower.
                    merged, conflicted = merge_counter_entries(
                        (k, e) for snap in (self._retired_counters, retained) for k, e in snap.items()
                    )
                    self._retired_counters = {k: e for k, e in merged.items() if k not in conflicted}
                self._executions.pop(execution_id, None)

    def _persist_result(self, result_path: str, payload: Any) -> None:
        """Write a pipeline's result payload to storage.

        The driver reads this file rather than the actor return value, so a
        result survives a transport that cannot carry it and an exception keeps
        its original type through ``_ensure_picklable_exception``. Nothing is
        returned: sending the payload back as well would serialize and ship a
        whole pipeline's results for the driver to discard.
        """
        ensure_parent_dir(result_path)
        StoragePath(result_path).write_bytes(cloudpickle.dumps(payload))

    def _run_worker_stage(
        self,
        run: _PipelineExecution,
        stage: PhysicalStage,
        shards: list[Shard],
        *,
        stage_label: str,
        stage_index_for_state: int,
        aux_per_shard: list[dict[int, Shard]] | None = None,
        is_last_stage: bool = False,
    ) -> list[Shard]:
        """Submit a worker stage, wait for completion, return regrouped output shards.

        ``stage_index_for_state`` is the index reported in coordinator state for
        UI/logging — for join right-sub-stages this is the *parent* stage index
        so progress reports stay attached to the user-visible stage.
        """
        cost = run.map_cost if stage.stage_type == StageType.MAP_WORKER else run.reduce_cost
        tasks = _compute_tasks_from_shards(
            shards,
            stage,
            stage_name=stage_label,
            aux_per_shard=aux_per_shard,
            cost=cost,
        )
        logger.info(
            "[%s] Starting stage %s (%s) with %d tasks", run.execution_id, stage_label, stage.stage_type, len(tasks)
        )
        self._start_stage(run, stage_label, stage_index_for_state, tasks, is_last_stage=is_last_stage)
        try:
            self._wait_for_stage(run)
        except Exception:
            self._emit_stage_stat(run, failed=True)
            raise
        self._emit_stage_stat(run)

        result_refs = self._collect_results(run)

        if any(isinstance(op, Scatter) for op in stage.operations):
            return _regroup_scatter_refs(result_refs, len(shards), stage.output_shards)
        return _regroup_map_refs(result_refs, len(shards))

    def _compute_join_aux(
        self,
        run: _PipelineExecution,
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

                right_refs = self._run_worker_stage(
                    run,
                    right_stage,
                    right_refs,
                    stage_label=f"join-right-{parent_stage_idx}-{i}-stage{stage_idx}",
                    stage_index_for_state=parent_stage_idx,
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

        # Fail any in-flight executions first so their run_pipeline callers
        # return promptly with a clean error, instead of blocking in
        # _wait_for_stage until no_workers_timeout once workers stop pulling
        # (their pull_task returns SHUTDOWN the moment _shutdown_event is set).
        with self._lock:
            for run in self._executions.values():
                if not run.done and run.fatal_error is None:
                    run.fatal_error = "Coordinator is shutting down"
                run.stage_done.set()

        self._shutdown_event.set()

        # Wait for coordinator thread to exit
        if self._coordinator_thread is not None:
            self._coordinator_thread.join(timeout=5.0)

        self._stats_writer.close()

        logger.info("Coordinator shutdown complete")

    def check_heartbeats(self, timeout: float = 120.0) -> None:
        """Marks stale workers as FAILED, re-queues their in-flight tasks."""
        with self._lock:
            now = time.monotonic()
            for worker_id, last in list(self._last_seen.items()):
                if now - last > timeout and self._worker_states.get(worker_id) not in {
                    WorkerState.FAILED,
                    WorkerState.DONE,
                }:
                    logger.warning(f"Zephyr worker {worker_id} failed to heartbeat within timeout ({now - last:.1f}s)")
                    self._worker_states[worker_id] = WorkerState.FAILED
                    self._maybe_requeue_worker_tasks(worker_id)


def _regroup_scatter_refs(
    result_refs: dict[int, TaskResult],
    input_shard_count: int,
    output_shard_count: int | None,
) -> list[Shard]:
    """Fan a scatter stage's outputs out to its reducers without loading data.

    Scatter routes records into exactly ``output_shard_count`` buckets via
    ``hash(key) % output_shard_count``; spawning more reduce tasks than that
    produces empty output files for shard indices that no record hashes to.
    When ``output_shard_count`` is None (group_by auto-detect), inherit the
    input shard count.

    Every reducer receives the full list of scatter data-file paths and reads
    the per-mapper ``.scatter_meta`` sidecars in parallel to build its own
    ``ScatterReader`` — the coordinator never consolidates a manifest.
    """
    num_output = output_shard_count if output_shard_count is not None else input_shard_count
    all_paths: list[str] = []
    for result in result_refs.values():
        all_paths.extend(result.shard)
    shared_refs = MemChunk(items=all_paths)
    return [ListShard(refs=[shared_refs]) for _ in range(num_output)]


def _regroup_map_refs(result_refs: dict[int, TaskResult], input_shard_count: int) -> list[Shard]:
    """Map a non-scatter stage's outputs 1:1 from input shard index to output.

    Each worker's ListShard keeps its own index. Resharding to a different
    shard count belongs to ReshardOp, not here.
    """
    num_output = max(max(result_refs.keys(), default=0) + 1, input_shard_count)
    return [result_refs[idx].shard if idx in result_refs else ListShard(refs=[]) for idx in range(num_output)]


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
    stage: PhysicalStage,
    stage_name: str,
    aux_per_shard: list[dict[int, Shard]] | None,
    cost: ZephyrTaskResources,
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
                cost=cost,
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
