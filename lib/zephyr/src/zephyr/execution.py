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

import enum
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import cloudpickle
import humanfriendly
from fray.actor import ActorFuture, ActorGroup, ActorHandle, current_actor
from fray.client import Client
from fray.current_client import current_client
from fray.local_backend import LocalClient
from fray.types import ActorConfig, ResourceConfig
from iris.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from rigging import telemetry
from rigging.filesystem import StoragePath, TransferBudgetExceeded, marin_temp_bucket
from rigging.timing import ExponentialBackoff, RateLimiter, log_time

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
from zephyr.runners import InlineRunner, SubprocessRunner
from zephyr.shuffle import ListShard, MemChunk
from zephyr.stage_io import (
    ShardTask,
    StageRunner,
    TaskResult,
    ZephyrTaskResources,
    ZephyrWorkerError,
    _ensure_picklable_exception,
    _shared_data_path,
    _stage_throughput,
)
from zephyr.stats import (
    StatsWriter,
    ZephyrWorkerStatStatus,
)
from zephyr.worker_context import Aggregation, CounterEntry, CounterSnapshot, merge_counter_entries
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

MAX_WORKERS_PER_JOB = 1_024
MAX_CONCURRENT_PIPELINES = 16

ZEPHYR_PROGRESS_TIME_METRIC = "progress_time_seconds"

_SNAPSHOT_ATTRIBUTES = telemetry.snapshot_attributes("gauge", telemetry.CURRENT_SNAPSHOT)


class ShardFailureKind(enum.StrEnum):
    """TASK failures count toward MAX_SHARD_FAILURES; INFRA failures (preemption) do not."""

    TASK = enum.auto()
    INFRA = enum.auto()


def _generate_execution_id() -> str:
    """Generate unique ID for this execution to avoid conflicts."""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _format_worker_status_md(active_tasks: int, stage: str) -> tuple[str, str]:
    """Return ``(detail_md, summary_md)`` for the worker's Iris task-status push.

    Returns ``("idle", "idle")`` when no task is in flight or no stage has run
    yet; otherwise produces a one-line summary plus a two-line detail block.
    """
    if active_tasks == 0 or not stage:
        return "idle", "idle"
    summary = f"**{stage}** — {active_tasks} task(s)"
    detail = "  \n".join([f"**Stage**: {stage}", f"**Active tasks**: {active_tasks}"])
    return detail, summary


def _push_iris_task_status(
    rate_limiter: RateLimiter,
    build_md: Callable[[], tuple[str, str]],
) -> None:
    """Push ``(detail, summary)`` markdown to the active Iris task's status, if any.

    No-op when not running inside an Iris task or when ``rate_limiter`` declines
    this tick. ``build_md`` is invoked lazily after the gating checks so the
    formatting work is skipped on the no-op path.
    """
    iris_client = ctx.client if (ctx := get_iris_ctx()) is not None else None
    if iris_client is None:
        return
    job_info = get_job_info()
    if job_info is None:
        return
    if not rate_limiter.should_run():
        return
    detail_md, summary_md = build_md()
    try:
        iris_client.report_task_status_text(job_info.task_id, job_info.attempt_id, detail_md, summary_md)
    except Exception:
        logger.warning("Failed to report task status text to Iris controller", exc_info=True)


def _execution_result_path(prefix: str, execution_id: str) -> str:
    """Return the result path for one execution."""
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
    - ``NO_WORK_BACKOFF``: no task is ready. The worker sleeps and retries.
    - ``SHUTDOWN``: pipeline finished or coordinator shutting down; worker exits.
    """

    RUN_TASK = enum.auto()
    NO_WORK_BACKOFF = enum.auto()
    SHUTDOWN = enum.auto()


@dataclass(frozen=True)
class PullTask:
    """Task payload in a ``pull_task`` response when status is ``RUN_TASK``."""

    task: ShardTask
    attempt: int
    chunk_prefix: str
    execution_id: str
    stage_generation: int


class CoordinatorUnreachable(RuntimeError):
    """Worker lost contact with the coordinator. Retryable at the iris task level."""


# Application errors that should never be retried by the execute() retry loop.
# These are deterministic errors (bad plan, invalid config, programming bugs)
# that would fail identically on every attempt. Infrastructure errors (OSError,
# RuntimeError from dead actors, backend actor errors) are NOT listed here so they
# remain retryable. TransferBudgetExceeded is deterministic: the cross-region
# budget is global and persists for the life of the process, so every retry hits
# the same wall while re-transferring data across regions.
_NON_RETRYABLE_ERRORS = (
    ZephyrWorkerError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    MemoryError,
    TransferBudgetExceeded,
)


def _default_stage_runner_factory_for(client: Client) -> Callable[[], StageRunner]:
    """Pick the default ``stage_runner_factory`` based on the client type.

    ``LocalClient`` is the dev/test backend — workers are threads in a
    single process, so per-shard subprocess isolation adds latency without
    delivering meaningful isolation. Distributed clients run each worker
    actor as its own VM where subprocess-per-shard gives real protection
    against native crashes and per-shard memory growth. Callers that want
    the other behavior pass ``stage_runner_factory=...`` explicitly.
    """
    if isinstance(client, LocalClient):
        return lambda: InlineRunner()

    return lambda: SubprocessRunner()


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
    fatal_error: str | None
    workers: dict[str, dict[str, Any]]


@dataclass
class _PipelineExecution:
    """Coordinator-side state for one pipeline execution.

    The coordinator drives multiple executions concurrently. All state
    scoped to a single pipeline lives here. Worker
    membership, heartbeats, and in-flight counter snapshots stay on the
    coordinator. The worker pool is shared across executions.
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
    # Bumped on every stage load. Attempt numbers restart per stage, so
    # (shard_idx, attempt) alone cannot tell a late report from stage N apart
    # from the same shard's first attempt in stage N+1. This value identifies it.
    stage_generation: int = 0
    # True once the final task-producing stage has been loaded, so a drained
    # queue means no further task can be dispatched for this pipeline.
    is_last_stage: bool = False
    # Folded completed-task counters, keyed by stage, name, and aggregation.
    completed_totals: dict[tuple[str | None, str, Aggregation], CounterEntry] = field(default_factory=dict)
    # Set at each _start_stage so status logs show throughput since stage start.
    stage_monotonic_start: float | None = None
    done: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    terminal_error: Exception | None = None
    storage_cleanup_safe: bool = False

    def start_stage(
        self,
        stage_name: str,
        current_stage_index: int,
        tasks: list[ShardTask],
        *,
        is_last_stage: bool,
    ) -> None:
        """Load one stage's tasks and reset the per-stage bookkeeping.

        Counters and plan_stages span the full execution. All state
        scoped to a single stage starts over. Callers hold the coordinator lock.
        """
        self.stage_generation += 1
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
        # worker. max_shard_infra_failures limits these failures so a shard that
        # deterministically crashes its worker (native SIGSEGV, OOM) aborts
        # instead of retrying forever.
        self.task_infra_attempts = {task.shard_idx: 0 for task in tasks}
        self.shard_errors = {}
        self.fatal_error = None
        self.is_last_stage = is_last_stage
        self.stage_monotonic_start = time.monotonic()
        self.stage_done.clear()

    def finish(self, *, storage_cleanup_safe: bool) -> None:
        """Mark the execution done and release the state only a live run needs.

        Completed counters stay: the coordinator folds them into its totals
        before dropping the execution. Callers hold the coordinator lock.
        """
        self.done = True
        self.storage_cleanup_safe = storage_cleanup_safe
        self.task_queue.clear()
        self.in_flight.clear()
        self.results = {}

    def fold_counters(self, snapshot: CounterSnapshot) -> None:
        """Fold one completed task snapshot into bounded execution state."""
        for name, entry in snapshot.counters.items():
            key = (entry.stage, name, entry.aggregation)
            accumulated = self.completed_totals.get(key)
            if accumulated is None:
                self.completed_totals[key] = CounterEntry(
                    entry.value,
                    entry.aggregation,
                    entry.stage,
                    entry.count,
                )
            else:
                accumulated.merge(entry)

    def merged_counters(self, stage: str | None = None) -> dict[str, CounterEntry]:
        """Return merged completed counters for this execution."""
        merged, conflicted = merge_counter_entries(
            (name, entry)
            for (entry_stage, name, _), entry in self.completed_totals.items()
            if stage is None or entry_stage == stage
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
    aggregation are dropped instead of raised. Stats collection never
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
    resources. A pipeline failure only fails its own execution. The
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
        # must never do this because its next pipeline needs those workers.
        self._drain_idle_workers = drain_idle_workers
        # Set when the coordinator as a whole can no longer make progress
        # (worker job permanently dead, maintenance loop crash). Fails all
        # active executions and rejects new ones.
        self._pool_error: str | None = None
        # Rotates which execution pull_task scans first, for cross-pipeline fairness.
        self._pull_offset: int = 0
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
        self._host_shutdown_event = actor_ctx.shutdown_event

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

    def _publish_telemetry(self) -> None:
        """Publish coordinator-owned counter snapshots."""
        with self._lock:
            progress_time_seconds = self._progress_time_seconds
            snapshots = [
                (run.execution_id, {name: entry.value for name, entry in run.merged_counters().items()})
                for run in self._executions.values()
            ]
        for execution_id, counters in snapshots:
            attributes = {**_SNAPSHOT_ATTRIBUTES, "run": execution_id}
            for name, value in counters.items():
                metric_name = re.sub(r"[^a-zA-Z0-9_]", "_", name.removeprefix("zephyr/"))
                telemetry.gauge(metric_name).set(value, attributes=attributes)
            telemetry.gauge(ZEPHYR_PROGRESS_TIME_METRIC, unit="s").set(
                progress_time_seconds,
                attributes=attributes,
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
                f"**Shards** - {completed}/{total} complete ({pct}%), {in_flight} in-flight, {queued} queued\n"
            )
            current_desc = _get_stage_description(plan_stages[stage_index]) if plan_stages else ""
            summary_lines.append(
                f"**{current_desc}** ({stage_index + 1}/{len(plan_stages)}) - {completed}/{total} shards ({pct}%)"
            )

        detail_md = "\n".join(detail_lines)[:MAX_STATUS_TEXT_LENGTH] or "idle"
        summary_md = "  \n".join(summary_lines)[:MAX_STATUS_TEXT_LENGTH] or "idle"
        return detail_md, summary_md

    def _report_task_stats(self) -> None:
        """Publish pipeline progress telemetry and Iris task status."""
        detail_md, summary_md = self._build_status_md()
        self._publish_telemetry()
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
                    [
                        CounterSnapshot(counters=run.merged_counters(), generation=0),
                        *self._worker_counters.values(),
                    ],
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

            # Map-only stages do not yield through ``_wrap_stage_stats`` and never
            # populate these counters. Drop the items/bytes_processed segment for
            # those stages. In-flight snapshots are stage-filtered, not
            # execution-filtered. Two concurrent pipelines that run an
            # identically-labelled stage share the live segment (log-only).
            elapsed = time.monotonic() - (stage_start or time.monotonic())
            throughput = _stage_throughput(_aggregate_counter_snapshots(snaps, stage_name), elapsed)
            if throughput is not None:
                logger.info(base_msg + ". %s", *base_args, throughput)
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
            stage_counters = {name: entry.value for name, entry in run.merged_counters(stage_name).items()}
        status = ZephyrWorkerStatStatus.FAILED if failed else ZephyrWorkerStatStatus.END
        self._stats_writer.emit_stage_stat(stage_counters, stage_name, execution_id, elapsed, total, status)

    def _record_shard_failure(
        self,
        run: _PipelineExecution,
        shard_idx: int,
        worker_id: str,
        kind: ShardFailureKind,
        error_info: str | None = None,
    ) -> bool:
        """Requeue an in-flight shard. Abort its execution at a per-shard limit.

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
                    "[%s] Shard %d has been in flight during %d infra failures (max %d). "
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

        if run.fatal_error is not None:
            # The execution is winding down and its storage is about to be
            # deleted. A requeued shard would have no valid output location
            # write to, and would keep the drain from ever finishing.
            logger.info("[%s] Not requeuing shard %d: execution already failed", run.execution_id, shard_idx)
            return True

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
        pipelines share the pool fairly. The first execution whose head task
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
                return PullStatus.RUN_TASK, PullTask(
                    task=task,
                    attempt=attempt,
                    chunk_prefix=self._chunk_prefix,
                    execution_id=run.execution_id,
                    stage_generation=run.stage_generation,
                )

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

    def _is_current_stage(self, run: _PipelineExecution, stage_generation: int, shard_idx: int) -> bool:
        """False for a report belonging to a stage this execution has moved past.

        A pool runs its coordinator and workers from one revision, so the stamp
        is always present. There is no unstamped path to accept.
        """
        if stage_generation == run.stage_generation:
            return True
        logger.warning(
            "Ignoring report for shard %d from stage generation %d (current %d)",
            shard_idx,
            stage_generation,
            run.stage_generation,
        )
        return False

    def report_result(
        self,
        worker_id: str,
        execution_id: str,
        shard_idx: int,
        attempt: int,
        result: TaskResult,
        counter_snapshot: CounterSnapshot,
        stage_generation: int,
    ) -> None:
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()

            run = self._executions.get(execution_id)
            if run is None or run.done:
                logger.warning(
                    "Ignoring result from worker %s for finished/unknown execution %s (shard %d)",
                    worker_id,
                    execution_id,
                    shard_idx,
                )
                # The task runner is complete. Drop its stale in-flight
                # snapshot so it stops polluting cross-pipeline totals until
                # the worker's next heartbeat lands.
                self._clear_worker_inflight_counters(worker_id, counter_snapshot.generation)
                return

            current_attempt = run.task_attempts.get(shard_idx, 0)
            if not self._is_current_stage(run, stage_generation, shard_idx):
                return

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
            run.fold_counters(counter_snapshot)
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

    def report_error(
        self,
        worker_id: str,
        execution_id: str,
        shard_idx: int,
        attempt: int,
        error_info: str,
        stage_generation: int,
    ) -> None:
        """Worker reports a task failure. Re-queues up to MAX_SHARD_FAILURES."""
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()

            run = self._executions.get(execution_id)
            if run is None or run.done:
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

            if not self._is_current_stage(run, stage_generation, shard_idx):
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

        """
        with self._lock:
            if worker_id is not None:
                snap = self._worker_counters.get(worker_id)
                if snap is None:
                    return {}
                return {k: e.value for k, e in snap.counters.items() if stage is None or e.stage == stage}

            all_snaps = [
                CounterSnapshot(counters=run.merged_counters(), generation=0) for run in self._executions.values()
            ]
            all_snaps.extend(self._worker_counters.values())

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

        Called when the coordinator as a whole cannot make progress. This can
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

        Calls for different execution IDs can run concurrently. A duplicate
        call joins the first call for its execution ID.
        """
        for task_kind, cost in (("map", map_cost), ("reduce", reduce_cost)):
            if not self._worker_resources.can_fit(cost):
                raise ValueError(f"{task_kind} task cost {cost} exceeds per-worker resources {self._worker_resources}")

        with self._lock:
            if self._shutdown_event.is_set():
                # Workers already got SHUTDOWN from pull_task, so an accepted
                # pipeline would block in _wait_for_stage until
                # no_workers_timeout (6h by default). A driver reaching here
                # holds an endpoint whose pool the owner already tore down.
                raise ZephyrWorkerError("Coordinator is shut down and cannot accept new pipelines")
            if self._pool_error is not None:
                raise ZephyrWorkerError(f"Coordinator pool failed: {self._pool_error}")
            existing = self._executions.get(execution_id)
            if existing is not None:
                run = existing
                owns_execution = False
            else:
                active = sum(1 for r in self._executions.values() if not r.done)
                if active >= MAX_CONCURRENT_PIPELINES:
                    raise RuntimeError(
                        f"Coordinator already runs {active} concurrent pipelines (max {MAX_CONCURRENT_PIPELINES})"
                    )
                run = _PipelineExecution(
                    execution_id=execution_id,
                    map_cost=map_cost,
                    reduce_cost=reduce_cost,
                )
                self._executions[execution_id] = run
                owns_execution = True

        if not owns_execution:
            run.finished.wait()
            if run.terminal_error is not None:
                raise run.terminal_error
            return None

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
                counters = {name: entry.value for name, entry in run.merged_counters().items()}
            self._persist_result(result_path, ZephyrExecutionResult(results=flat_result, counters=counters))
            return None
        except Exception as e:
            # Persist the normalized exception so the driver can recover the
            # original type even when the actor transport cannot carry it.
            terminal_error = _ensure_picklable_exception(e)
            assert isinstance(terminal_error, Exception)
            with suppress(Exception):
                self._persist_result(result_path, terminal_error)
            with self._lock:
                run.terminal_error = terminal_error
            if terminal_error is e:
                raise
            raise terminal_error from e
        finally:
            storage_cleanup_safe = self._drain_execution(run)
            with self._lock:
                run.finish(storage_cleanup_safe=storage_cleanup_safe)
                run.finished.set()

    def release_execution(self, execution_id: str) -> None:
        """Release terminal state and delete storage after all tasks drain."""
        with self._lock:
            run = self._executions.get(execution_id)
            if run is None:
                return
            if not run.done:
                raise RuntimeError(f"Execution {execution_id} is still active")
            self._executions.pop(execution_id, None)

        if run.storage_cleanup_safe:
            _cleanup_execution(self._chunk_prefix, execution_id)

    def _drain_execution(self, run: _PipelineExecution, timeout: float = 300.0) -> bool:
        """Stop dispatching for this execution and wait for its tasks to retire.

        ``release_execution`` deletes the execution's storage only when this
        drain completed. A task still running would lose its shared data or
        write chunk files after cleanup.

        Reports that arrive during the drain are recorded normally. ``pull_task``
        stops handing this execution out because its queue is cleared here.
        Shards are not requeued once the execution has failed.
        """
        with self._lock:
            run.task_queue.clear()
            if not run.in_flight:
                return True
            outstanding = len(run.in_flight)
        logger.info("[%s] Draining %d in-flight task(s) before teardown", run.execution_id, outstanding)

        # Deliberately not short-circuited by shutdown: a task mid-write is
        # just as dangerous to delete storage under during teardown.
        deadline = time.monotonic() + timeout
        backoff = ExponentialBackoff(initial=0.1, maximum=2.0)
        while time.monotonic() < deadline:
            with self._lock:
                if not run.in_flight:
                    return True
            time.sleep(backoff.next_interval())

        with self._lock:
            stragglers = sorted(run.in_flight)
        logger.warning(
            "[%s] %d task(s) still in flight after %.0fs; keeping this execution's storage: %s",
            run.execution_id,
            len(stragglers),
            timeout,
            stragglers,
        )
        return False

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
        if self._host_shutdown_event is not None:
            self._host_shutdown_event.set()

        # Wait for coordinator thread to exit
        if self._coordinator_thread is not None:
            self._coordinator_thread.join(timeout=5.0)

        self._stats_writer.close()

        logger.info("Coordinator shutdown complete")

    def stop_workers(self) -> None:
        """Tell workers to exit while the coordinator actor stays reachable."""
        self._shutdown_event.set()

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


# ---------------------------------------------------------------------------
# ZephyrWorker
# ---------------------------------------------------------------------------


class ZephyrWorker:
    """Long-lived worker actor with a single poll loop and per-task threads.

    The worker registers once with the coordinator, then loops: polls with its
    current available ``ZephyrTaskResources``, spins off a thread per dispatched task,
    and immediately polls again. The coordinator dispatches when the worker's
    available resources can fit the next task's cost; otherwise it returns
    NO_WORK_BACKOFF.
    """

    def __init__(
        self,
        coordinator_handle: ActorHandle,
        stage_runner_factory: Callable[[], StageRunner],
        total_resources: ZephyrTaskResources,
    ):
        self._coordinator = coordinator_handle
        self._stage_runner_factory = stage_runner_factory
        self._shutdown_event = threading.Event()
        self._counter_generation: int = 0
        self._last_reported_counters: dict[str, CounterEntry] = {}
        # Runners and sub-IDs for currently active slots — written by _stage_manager,
        # read (snapshotted) by heartbeat thread.
        self._active_runners: list[StageRunner] = []
        self._active_task_count: int = 0
        self._current_stage_name: str = ""

        # Resource pool: each accepted task deducts its cost and restores it on
        # completion. The coordinator gates dispatch on the available amount,
        # so the pool implicitly limits concurrency.
        self._available: ZephyrTaskResources = total_resources
        self._resources_lock = threading.Lock()
        # Set by task threads on completion so the poll loop wakes early.
        self._task_completed_event = threading.Event()

        # Throttle Iris status pushes; the heartbeat loop ticks faster than
        # the UI needs to refresh.
        self._iris_status_limiter = RateLimiter(interval_seconds=10.0)

        # Capture actor context while ContextVar is still set (child threads
        # in Python <3.12 don't inherit it).
        self._actor_ctx = current_actor()
        self._host_shutdown_event = self._actor_ctx.shutdown_event
        self._worker_id = f"{self._actor_ctx.group_name}-{self._actor_ctx.index}"
        self._actor_handle = self._actor_ctx.handle

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator_handle,),
            daemon=True,
            name=f"zephyr-hb-{self._worker_id}",
        )
        self._heartbeat_thread.start()

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"zephyr-poll-{self._worker_id}",
        )
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        """Single poll loop: requests tasks from the coordinator using current available resources.

        Registers once with the coordinator, then loops: polls with available CPU and
        memory, spins off a thread per dispatched task, and immediately polls again.
        At stage boundaries, the loop sleeps briefly and then polls again.
        """
        logger.info("[%s] Poll loop starting", self._worker_id)
        try:
            self._coordinator.register_worker.remote(self._worker_id, self._actor_handle).result(timeout=30.0)
        except Exception:
            logger.error("[%s] Failed to register with coordinator", self._worker_id, exc_info=True)
            self._shutdown_event.set()
            if self._host_shutdown_event is not None:
                self._host_shutdown_event.set()
            return

        backoff = ExponentialBackoff(initial=0.1, maximum=5.0)
        in_flight_threads: list[threading.Thread] = []
        future: ActorFuture | None = None
        future_start = 0.0
        warned = False

        while not self._shutdown_event.is_set():
            # Prune finished threads to avoid unbounded growth.
            in_flight_threads = [t for t in in_flight_threads if t.is_alive()]

            with self._resources_lock:
                avail = self._available

            # If no resources are available, skip the coordinator round-trip and
            # wait for a task to finish freeing capacity.
            if avail.cpu == 0 and avail.memory == 0:
                self._task_completed_event.wait(timeout=backoff.next_interval())
                self._task_completed_event.clear()
                continue

            # Short timeout keeps the thread responsive to shutdown without
            # killing it on slow coordinator deserialization.
            try:
                if future is None:
                    future = self._coordinator.pull_task.remote(self._worker_id, avail)
                    future_start = time.monotonic()
                    warned = False
                response = future.result(timeout=0.5)
            except TimeoutError:
                elapsed = time.monotonic() - future_start
                if elapsed > 30 and not warned:
                    logger.warning("[%s] Waiting for pull_task response (%.0fs)", self._worker_id, elapsed)
                    warned = True
                continue
            except Exception as e:
                logger.info("[%s] pull_task failed (coordinator may be dead): %s", self._worker_id, e)
                break

            future = None
            status, work = response

            if status == PullStatus.SHUTDOWN:
                logger.info("[%s] Received SHUTDOWN from coordinator", self._worker_id)
                break

            if status != PullStatus.RUN_TASK:
                # Sleep until resources become free or new tasks arrive.
                wait = backoff.next_interval()
                self._task_completed_event.wait(timeout=wait)
                self._task_completed_event.clear()
                continue

            backoff.reset()
            assert work is not None

            with self._resources_lock:
                self._available = self._available - work.task.cost

            runner = self._stage_runner_factory()
            with self._resources_lock:
                self._active_runners.append(runner)
            self._active_task_count += 1
            self._current_stage_name = work.task.stage_name

            t = threading.Thread(
                target=self._task_thread,
                args=(work, runner),
                daemon=True,
                name=f"zephyr-task-{self._worker_id}-s{work.task.shard_idx}",
            )
            in_flight_threads.append(t)
            t.start()

        # Drain in-flight tasks before deregistering.
        for t in in_flight_threads:
            t.join()

        logger.info("[%s] Poll loop exiting", self._worker_id)
        with suppress(Exception):
            self._coordinator.deregister_worker.remote(self._worker_id).result(timeout=10.0)

        self._shutdown_event.set()
        if self._host_shutdown_event is not None:
            self._host_shutdown_event.set()

    def _task_thread(
        self,
        work: PullTask,
        runner: StageRunner,
    ) -> None:
        """Execute one shard task, report the result, and restore task.cost to the pool."""
        task_start = time.monotonic()
        task = work.task
        try:
            result, task_counters = self._execute_shard(task, work.chunk_prefix, work.execution_id, runner)
            logger.info("[%s] Shard %d done in %.2fs", self._worker_id, task.shard_idx, time.monotonic() - task_start)
            # Block until coordinator records result — prevents _in_flight races.
            self._counter_generation += 1
            self._coordinator.report_result.remote(
                self._worker_id,
                work.execution_id,
                task.shard_idx,
                work.attempt,
                result,
                CounterSnapshot(counters=dict(task_counters), generation=self._counter_generation),
                work.stage_generation,
            ).result()
        except Exception:
            logger.error("Worker %s error on shard %d", self._worker_id, task.shard_idx, exc_info=True)
            self._coordinator.report_error.remote(
                self._worker_id,
                work.execution_id,
                task.shard_idx,
                work.attempt,
                "".join(traceback.format_exc()),
                work.stage_generation,
            ).result()
        finally:
            with self._resources_lock:
                self._available = self._available + task.cost
                if runner in self._active_runners:
                    self._active_runners.remove(runner)
            self._active_task_count = max(0, self._active_task_count - 1)
            self._task_completed_event.set()

    def _report_worker_iris_status(self) -> None:
        """Push worker status text to Iris for UI display. Called on each heartbeat."""

        def build_md() -> tuple[str, str]:
            stage = self._current_stage_name
            stage_values = {k: e.value for k, e in self._last_reported_counters.items() if e.stage == stage}
            throughput = _stage_throughput(stage_values, 1.0) if stage else None
            if throughput is not None:
                logger.info("[%s] [%s] throughput: %s", self._worker_id, stage, throughput)
            return _format_worker_status_md(self._active_task_count, stage)

        _push_iris_task_status(self._iris_status_limiter, build_md)

    def _heartbeat_counter_snapshot(self) -> CounterSnapshot | None:
        """Aggregate live counters from all active runners; return None if unchanged."""
        with self._resources_lock:
            runners = list(self._active_runners)
        current, _ = merge_counter_entries((name, entry) for r in runners for name, entry in r.live_counters().items())
        if current == self._last_reported_counters:
            return None
        self._last_reported_counters = current
        self._counter_generation += 1
        return CounterSnapshot(counters=current, generation=self._counter_generation)

    def _heartbeat_loop(
        self, coordinator: ActorHandle, interval: float = 5.0, max_consecutive_failures: int = 5
    ) -> None:
        logger.debug("[%s] Heartbeat loop starting", self._worker_id)
        heartbeat_count = 0
        consecutive_failures = 0
        while not self._shutdown_event.is_set():
            try:
                snapshot = self._heartbeat_counter_snapshot()
                coordinator.heartbeat.remote(self._worker_id, snapshot).result()
                heartbeat_count += 1
                consecutive_failures = 0
                if heartbeat_count % 10 == 1:
                    logger.debug("[%s] Sent heartbeat #%d", self._worker_id, heartbeat_count)
                self._report_worker_iris_status()
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
                        "[%s] %d consecutive heartbeat failures — coordinator unreachable, shutting down",
                        self._worker_id,
                        consecutive_failures,
                    )
                    self._actor_ctx.fail(
                        CoordinatorUnreachable(f"{consecutive_failures} consecutive heartbeat failures")
                    )
                    self._shutdown_event.set()
                    break
            self._shutdown_event.wait(timeout=interval)
        logger.debug("[%s] Heartbeat loop exiting after %d beats", self._worker_id, heartbeat_count)

    def _execute_shard(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
        stage_runner: StageRunner,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        logger.info(
            "[%s] [shard %d/%d] stage=%s, %d ops",
            execution_id,
            task.shard_idx,
            task.total_shards,
            task.stage_name,
            len(task.operations),
        )
        result, counters = stage_runner.execute(task, chunk_prefix, execution_id)
        logger.info("[shard %d] Complete: %d refs produced", task.shard_idx, len(result.shard.refs))
        return result, counters

    def __repr__(self) -> str:
        return f"ZephyrWorker(id={self._worker_id})"

    def shutdown(self) -> None:
        """Signal the worker to stop accepting new tasks."""
        self._shutdown_event.set()
        self._task_completed_event.set()
        for thread in (self._poll_thread, self._heartbeat_thread):
            if thread is not threading.current_thread():
                thread.join(timeout=10.0)
        if self._host_shutdown_event is not None:
            self._host_shutdown_event.set()


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


# ---------------------------------------------------------------------------
# Coordinator-as-Job infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZephyrExecutionResult:
    """Result of running a Zephyr pipeline.

    The coordinator writes this wire format to the execution result file.

    Attributes:
        results: Flat list of items produced by the terminal stage of the
            pipeline (e.g. output file paths for write stages).
        counters: Aggregated counter values from the run, including built-in
            zephyr counters (e.g. ``zephyr/records_in``) and any user counters
            recorded via ``zephyr.counters.pipeline``.
    """

    results: list
    counters: dict[str, int | float]


def _read_coordinator_result(result_path: str) -> Any:
    """Read the coordinator job's result file. Returns the deserialized object."""
    data = StoragePath(result_path).read_bytes()
    try:
        return cloudpickle.loads(data)
    except Exception as e:
        # The coordinator normalizes exceptions before persisting, so a revival
        # failure here means a genuinely corrupt or version-incompatible payload.
        # Surface a clear non-retryable error instead of letting an opaque
        # unpickle error crash the driver mid-recovery.
        raise ZephyrWorkerError(f"Could not deserialize coordinator result at {result_path}: {e!r}") from e


def _try_read_coordinator_result(result_path: str) -> Any:
    """Best-effort read of the result file. Returns None if unreadable.

    Used only in the retry error-recovery path where the coordinator job
    may have crashed before writing the file.
    """
    try:
        return _read_coordinator_result(result_path)
    except Exception:
        return None


def _tasks_per_worker(worker_resources: ResourceConfig, task_resources: ResourceConfig) -> int:
    """Return how many concurrent copies of *task_resources* fit in *worker_resources*.

    Packing uses cpu and ram only. Zephyr does not track disk in runtime
    admission (``ZephyrTaskResources`` is cpu+memory), so disk is ignored here
    even though it still applies to Iris worker sizing.
    """
    ratios = [worker_resources.cpu / task_resources.cpu]
    worker_ram = humanfriendly.parse_size(worker_resources.ram, binary=True)
    task_ram = humanfriendly.parse_size(task_resources.ram, binary=True)
    if task_ram > 0:
        ratios.append(worker_ram / task_ram)
    return max(1, math.floor(min(ratios)))


def _validate_task_resources(
    worker_resources: ResourceConfig,
    task_resources: ResourceConfig,
    task_name: str,
) -> None:
    """Make sure that one task fits the fixed worker pool."""
    fields = (
        "device",
        "preemptible",
        "regions",
        "zone",
        "target_cluster",
        "replicas",
        "device_alternatives",
        "image",
    )
    for field_name in fields:
        worker_value = getattr(worker_resources, field_name)
        task_value = getattr(task_resources, field_name)
        if task_value != worker_value:
            raise ValueError(
                f"Task resource field '{field_name}' must match the worker pool ({task_value!r} != {worker_value!r})"
            )

    worker_ram = humanfriendly.parse_size(worker_resources.ram, binary=True)
    worker_disk = humanfriendly.parse_size(worker_resources.disk, binary=True)
    task_ram = humanfriendly.parse_size(task_resources.ram, binary=True)
    task_disk = humanfriendly.parse_size(task_resources.disk, binary=True)
    if worker_resources.cpu < task_resources.cpu or worker_ram < task_ram or worker_disk < task_disk:
        raise ValueError(
            f"{task_name} task resources exceed one Zephyr worker: task={task_resources}, worker={worker_resources}"
        )


def _resolve_task_resources(
    worker_resources: ResourceConfig,
    map_task_resources: ResourceConfig | None,
    reduce_task_resources: ResourceConfig | None,
) -> tuple[ResourceConfig, ResourceConfig, int]:
    """Resolve task resources and calculate the task packing limit."""
    resolved_map = map_task_resources or worker_resources
    resolved_reduce = reduce_task_resources or resolved_map
    _validate_task_resources(worker_resources, resolved_map, "map")
    _validate_task_resources(worker_resources, resolved_reduce, "reduce")
    tasks_per_worker = min(
        _tasks_per_worker(worker_resources, resolved_map),
        _tasks_per_worker(worker_resources, resolved_reduce),
    )
    return resolved_map, resolved_reduce, tasks_per_worker


class _ContextState(enum.StrEnum):
    NEW = enum.auto()
    OWNER = enum.auto()
    BORROWED = enum.auto()
    CLOSED = enum.auto()


class _IdleWorkerPolicy(enum.StrEnum):
    DRAIN = enum.auto()
    RETAIN = enum.auto()


@dataclass
class ZephyrContext:
    """Execute Zephyr pipelines with a dedicated or shared worker pool.

    A plain execute() call creates a dedicated pool. A context manager starts
    one shared pool and keeps it active until context exit. Child jobs can
    receive the entered context through Fray serialization. Borrowed contexts
    keep the coordinator handle but do not own pool shutdown.
    """

    client: Client | None = None
    max_workers: int | None = None
    resources: ResourceConfig | None = None
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=0.1, ram="1g", preemptible=False)
    )
    chunk_storage_prefix: str | None = None
    name: str = ""
    no_workers_timeout: float | None = None
    max_execution_retries: int = 100
    stage_runner_factory: Callable[[], StageRunner] | None = None
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES

    _shared_data: ContextVar[dict[str, Any] | None] = field(init=False, repr=False)
    _state: _ContextState = field(init=False, default=_ContextState.NEW, repr=False)
    _coordinator_group: ActorGroup | None = field(init=False, default=None, repr=False)
    _coordinator_handle: ActorHandle | None = field(init=False, default=None, repr=False)
    _worker_group: ActorGroup | None = field(init=False, default=None, repr=False)
    _state_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = current_client()

        if env_val := os.environ.get("ZEPHYR_MAX_WORKERS"):
            if self.max_workers is None:
                try:
                    self.max_workers = int(env_val)
                except ValueError as e:
                    raise ValueError(f"Invalid ZEPHYR_MAX_WORKERS environment variable value: {env_val}") from e
            else:
                logger.info("Ignoring ZEPHYR_MAX_WORKERS because max_workers is set.")

        if self.resources is None:
            self.resources = ResourceConfig(cpu=1, ram="1g")

        if self.no_workers_timeout is None:
            self.no_workers_timeout = 6 * 60 * 60

        if self.chunk_storage_prefix is None:
            self.chunk_storage_prefix = marin_temp_bucket(ttl_days=1, prefix="zephyr")

        if self.stage_runner_factory is None:
            assert self.client is not None
            self.stage_runner_factory = _default_stage_runner_factory_for(self.client)

        self.name = f"{self.name}-{uuid.uuid4().hex[:8]}"
        self._shared_data = ContextVar(f"zephyr_shared_data_{self.name}", default=None)
        self._state_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize execution access without pool ownership."""
        state = dict(self.__dict__)
        state["_serialized_shared_data"] = dict(self._shared_data.get() or {})
        state.pop("_shared_data", None)
        state.pop("_state_lock", None)
        state["client"] = None
        state["_coordinator_group"] = None
        state["_worker_group"] = None
        if self._state is _ContextState.OWNER:
            state["_state"] = _ContextState.BORROWED
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        shared_data = state.pop("_serialized_shared_data", {})
        self.__dict__.update(state)
        if self._state is _ContextState.NEW:
            self.client = current_client()
        self._shared_data = ContextVar(f"zephyr_shared_data_{self.name}", default=None)
        self._shared_data.set(dict(shared_data))
        self._state_lock = threading.Lock()

    def put(self, name: str, obj: Any) -> None:
        """Add one value to the current logical shared-data view."""
        current = self._shared_data.get() or {}
        self._shared_data.set({**current, name: obj})

    def _upload_shared_data(self, execution_id: str) -> None:
        """Write the current logical shared-data view for one execution."""
        for name, obj in dict(self._shared_data.get() or {}).items():
            path = _shared_data_path(self.chunk_storage_prefix, execution_id, name)
            ensure_parent_dir(path)
            started = time.monotonic()
            data = cloudpickle.dumps(obj)
            StoragePath(path).write_bytes(data)
            logger.info(
                "Shared data '%s' written to %s (serialized %d bytes in %.2fs)",
                name,
                path,
                len(data),
                time.monotonic() - started,
            )

    def _worker_limit(self) -> int:
        assert self.client is not None
        if self.max_workers is not None:
            return self.max_workers
        if isinstance(self.client, LocalClient):
            return os.cpu_count() or 1
        return MAX_WORKERS_PER_JOB

    def _start_pool(
        self,
        worker_count: int,
        idle_policy: _IdleWorkerPolicy,
    ) -> tuple[ActorGroup, ActorHandle, ActorGroup]:
        """Start one coordinator job and its worker group."""
        assert self.client is not None
        assert self.resources is not None
        assert self.stage_runner_factory is not None
        assert self.chunk_storage_prefix is not None
        assert self.no_workers_timeout is not None

        pool_id = uuid.uuid4().hex[:8]
        coordinator_name = f"zephyr-{self.name}-coordinator-{pool_id}"
        coordinator_group = self.client.create_actor_group(
            ZephyrCoordinator,
            self.chunk_storage_prefix,
            ZephyrTaskResources.from_resource_config(self.resources),
            self.no_workers_timeout,
            self.heartbeat_timeout,
            self.max_shard_failures,
            self.max_shard_infra_failures,
            idle_policy is _IdleWorkerPolicy.DRAIN,
            name=coordinator_name,
            count=1,
            resources=self.coordinator_resources,
            actor_config=ActorConfig(max_concurrency=100),
        )
        coordinator: ActorHandle | None = None
        worker_group: ActorGroup | None = None
        try:
            coordinator = coordinator_group.wait_ready(count=1)[0]
            worker_name = f"zephyr-{self.name}-workers-{pool_id}"
            worker_group = self.client.create_actor_group(
                ZephyrWorker,
                coordinator,
                self.stage_runner_factory,
                ZephyrTaskResources.from_resource_config(self.resources),
                name=worker_name,
                count=worker_count,
                resources=self.resources,
                actor_config=ActorConfig(max_task_retries=10),
            )
            ready_wait = float(os.environ.get("ZEPHYR_WORKERS_READY_WAIT") or 12 * 60 * 60)
            worker_group.wait_ready(count=1, timeout=ready_wait)
            coordinator.set_worker_group.remote(worker_group).result()
            return coordinator_group, coordinator, worker_group
        except Exception:
            if coordinator is not None:
                with suppress(Exception):
                    coordinator.shutdown.remote().result(timeout=10.0)
            if worker_group is not None:
                with suppress(Exception):
                    worker_group.shutdown()
            with suppress(Exception):
                coordinator_group.shutdown()
            raise

    def _stop_pool(
        self,
        coordinator_group: ActorGroup,
        coordinator: ActorHandle,
        worker_group: ActorGroup,
    ) -> None:
        """Stop a coordinator and its workers."""
        with suppress(Exception):
            coordinator.stop_workers.remote().result(timeout=10.0)

        with suppress(Exception):
            if isinstance(self.client, LocalClient):
                for worker in worker_group.wait_ready():
                    worker.shutdown.remote().result(timeout=10.0)
                worker_group.shutdown()
            else:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if worker_group.is_done():
                        break
                    time.sleep(0.1)
                else:
                    logger.warning("Workers did not exit naturally, terminating")
                    worker_group.shutdown()

        with suppress(Exception):
            coordinator.shutdown.remote().result(timeout=10.0)

        with suppress(Exception):
            coordinator_group.shutdown()

    def start(self) -> "ZephyrContext":
        """Start a shared pool and retain idle workers."""
        with self._state_lock:
            if self._state is not _ContextState.NEW:
                raise RuntimeError(f"Cannot start ZephyrContext in state {self._state}")
            groups = self._start_pool(self._worker_limit(), _IdleWorkerPolicy.RETAIN)
            self._coordinator_group, self._coordinator_handle, self._worker_group = groups
            self._state = _ContextState.OWNER
        return self

    def _run_on_coordinator(
        self,
        coordinator: ActorHandle,
        plan: PhysicalPlan,
        execution_id: str,
        map_task_resources: ResourceConfig,
        reduce_task_resources: ResourceConfig,
    ) -> ZephyrExecutionResult:
        """Run one plan on an existing coordinator and read its stored result."""
        result_path = _execution_result_path(self.chunk_storage_prefix, execution_id)
        try:
            coordinator.run_pipeline.submit(
                plan,
                execution_id,
                ZephyrTaskResources.from_resource_config(map_task_resources),
                ZephyrTaskResources.from_resource_config(reduce_task_resources),
            ).result()
        except Exception:
            payload = _try_read_coordinator_result(result_path)
            if isinstance(payload, Exception):
                raise payload from None
            raise

        payload = _read_coordinator_result(result_path)
        if isinstance(payload, Exception):
            raise payload
        return payload

    def execute(
        self,
        dataset: Dataset,
        verbose: bool = False,
        dry_run: bool = False,
        *,
        map_task_resources: ResourceConfig | None = None,
        reduce_task_resources: ResourceConfig | None = None,
    ) -> ZephyrExecutionResult:
        """Execute one dataset on a dedicated or supplied shared pool."""
        plan = compute_plan(dataset)
        if verbose or dry_run:
            _print_plan(dataset.operations, plan)
        if dry_run:
            return ZephyrExecutionResult(results=[], counters={})
        if plan.num_shards <= 0:
            logger.warning("No shards in plan, returning empty results.")
            return ZephyrExecutionResult(results=[], counters={})

        assert self.resources is not None
        resolved_map, resolved_reduce, tasks_per_worker = _resolve_task_resources(
            self.resources,
            map_task_resources,
            reduce_task_resources,
        )

        with self._state_lock:
            if self._state is _ContextState.CLOSED:
                raise RuntimeError("Cannot execute with a closed ZephyrContext")
            state = self._state
            coordinator = self._coordinator_handle

        if state in {_ContextState.OWNER, _ContextState.BORROWED}:
            assert coordinator is not None
            execution_id = _generate_execution_id()
            logger.info("Starting shared Zephyr pipeline %s", execution_id)
            self._upload_shared_data(execution_id)
            try:
                return self._run_on_coordinator(
                    coordinator,
                    plan,
                    execution_id,
                    resolved_map,
                    resolved_reduce,
                )
            finally:
                with suppress(Exception):
                    coordinator.release_execution.remote(execution_id).result(timeout=10.0)

        assert state is _ContextState.NEW
        last_exception: Exception | None = None
        backoff = ExponentialBackoff(initial=2.0, maximum=60.0, factor=2.0, jitter=0.1)
        for attempt in range(self.max_execution_retries + 1):
            execution_id = _generate_execution_id()
            coordinator_group: ActorGroup | None = None
            worker_group: ActorGroup | None = None
            dedicated_coordinator: ActorHandle | None = None
            try:
                self._upload_shared_data(execution_id)
                needed_workers = math.ceil(plan.num_shards / tasks_per_worker)
                coordinator_group, dedicated_coordinator, worker_group = self._start_pool(
                    min(self._worker_limit(), needed_workers),
                    _IdleWorkerPolicy.DRAIN,
                )
                backoff.reset()
                return self._run_on_coordinator(
                    dedicated_coordinator,
                    plan,
                    execution_id,
                    resolved_map,
                    resolved_reduce,
                )
            except _NON_RETRYABLE_ERRORS:
                raise
            except Exception as error:
                payload = _try_read_coordinator_result(_execution_result_path(self.chunk_storage_prefix, execution_id))
                if isinstance(payload, _NON_RETRYABLE_ERRORS):
                    raise payload from None
                last_exception = error
                if attempt >= self.max_execution_retries:
                    raise
                delay = backoff.next_interval()
                logger.warning(
                    "Pipeline attempt %d failed (%d retries left), retrying in %.1fs: %s",
                    attempt,
                    self.max_execution_retries - attempt,
                    delay,
                    error,
                )
                time.sleep(delay)
            finally:
                if coordinator_group is not None and dedicated_coordinator is not None and worker_group is not None:
                    self._stop_pool(coordinator_group, dedicated_coordinator, worker_group)
                _cleanup_execution(self.chunk_storage_prefix, execution_id)

        raise last_exception  # type: ignore[misc]

    def shutdown(self) -> None:
        """Stop an owned shared pool."""
        with self._state_lock:
            if self._state is _ContextState.BORROWED:
                raise RuntimeError("A borrowed ZephyrContext cannot stop its shared pool")
            if self._state in {_ContextState.NEW, _ContextState.CLOSED}:
                return
            assert self._state is _ContextState.OWNER
            assert self._coordinator_group is not None
            assert self._coordinator_handle is not None
            assert self._worker_group is not None
            groups = (self._coordinator_group, self._coordinator_handle, self._worker_group)
            self._coordinator_group = None
            self._coordinator_handle = None
            self._worker_group = None
            self._state = _ContextState.CLOSED

        self._stop_pool(*groups)

    def __enter__(self) -> "ZephyrContext":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


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
