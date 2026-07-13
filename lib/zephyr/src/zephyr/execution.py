# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job-based execution engine for Zephyr pipelines.

The coordinator runs as a fray *job* that internally creates coordinator and
worker *actors* as child jobs. Workers pull tasks from the coordinator actor,
execute shard operations, and report results back. Because actors are children
of the coordinator job, Iris cascading termination automatically cleans them
up when the coordinator exits or is killed — preventing stale-coordinator
bugs where orphaned coordinators and workers consume resources indefinitely.

Two lifecycles share the same coordinator/worker classes:

* **Dedicated** (default): each ``ZephyrContext.execute()`` submits a fresh
  coordinator job that runs exactly one pipeline and tears everything down.
* **Shared**: a ``ZephyrPool`` submits a long-lived coordinator job with a
  fixed worker pool and publishes the coordinator's actor endpoint. Drivers
  connect with ``ZephyrContext(coordinator_endpoint=...)`` (or the
  ``ZEPHYR_COORDINATOR_ENDPOINT`` env var) and submit pipelines that run
  *concurrently*; the coordinator schedules tasks from all active pipelines
  onto whichever workers have free resources. A failing pipeline only fails
  its own execution — the coordinator and the worker pool keep serving the
  others.
"""

import enum
import logging
import math
import os
import sys
import threading
import time
import traceback
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import cloudpickle
import humanfriendly
from fray.actor import ActorFuture, ActorHandle, current_actor
from fray.client import Client, JobHandle
from fray.current_client import current_client, set_current_client
from fray.local_backend import LocalClient
from fray.types import ActorConfig, Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, create_environment
from fray.types import JobStatus as FrayJobStatus
from iris.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from rigging import telltale
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

MAX_WORKERS_PER_JOB = 1_024

ZEPHYR_PROGRESS_TIME_METRIC = "zephyr_progress_time_seconds"

# Max pipelines a shared coordinator runs concurrently. Each active
# run_pipeline occupies one actor dispatch slot for its whole duration, so this
# bound keeps slots free for the pull_task/report_*/heartbeat calls that feed
# workers. Excess submissions are rejected rather than queued.
MAX_CONCURRENT_PIPELINES = 16

# Worker task-retry budget for shared pools — effectively "replenish forever".
# Preemptions on contended clusters can be misclassified as task failures
# (marin#7121: dirty k8s preemptions land as FAILED), and a long-lived shared
# pool must survive them rather than die on a small failure budget.
SHARED_POOL_MAX_TASK_RETRIES = 1000

# How long ZephyrPool.start() waits for the shared coordinator job to
# publish its endpoint before giving up.
SHARED_CONTEXT_START_TIMEOUT = 600.0

# Shared coordinator serve-loop poll cadence (worker-job death / shutdown).
_SERVE_POLL_INTERVAL = 5.0

# When set, a ZephyrContext with no explicit ``coordinator_endpoint`` connects
# to this shared coordinator. Lets a pool owner propagate the endpoint to
# per-step driver jobs via the environment (e.g. Iris ``-e``) instead of
# threading it through every step's code.
ZEPHYR_COORDINATOR_ENDPOINT_ENV = "ZEPHYR_COORDINATOR_ENDPOINT"


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
    - ``NO_WORK_BACKOFF``: queue is empty mid-stage; worker sleeps and retries.
    - ``STAGE_COMPLETED``: non-last stage boundary; worker sleeps and re-polls.
    - ``SHUTDOWN``: pipeline finished or coordinator shutting down; worker exits.
    """

    RUN_TASK = enum.auto()
    NO_WORK_BACKOFF = enum.auto()
    STAGE_COMPLETED = enum.auto()
    SHUTDOWN = enum.auto()


@dataclass(frozen=True)
class PullTask:
    """Task payload in a ``pull_task`` response when status is ``RUN_TASK``."""

    task: ShardTask
    attempt: int
    config: dict[str, Any]


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


@dataclass(frozen=True)
class CoordinatorInfo:
    """Coordinator config a connecting driver needs to submit pipelines.

    Returned by ``ZephyrCoordinator.describe()`` so shared-context drivers can
    upload shared data / clean up under the coordinator's ``chunk_prefix``.
    """

    chunk_prefix: str
    worker_resources: ZephyrTaskResources
    shared: bool


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
            recorded via ``zephyr.counters.pipeline``.
    """

    results: list
    counters: dict[str, int | float]


@dataclass
class _PipelineExecution:
    """Coordinator-side state for one pipeline execution.

    The coordinator can drive multiple executions concurrently (shared
    context); everything scoped to a single pipeline lives here. Worker
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
    # When True, idle workers at a non-last stage boundary get STAGE_COMPLETED.
    stage_complete: bool = False
    # When True, idle workers on the last stage receive SHUTDOWN once all
    # tasks are in-flight, so they exit eagerly instead of polling until
    # coordinator.shutdown(). Never set on shared coordinators — their
    # workers are long-lived and idle between pipelines.
    is_last_stage: bool = False
    # Completed-task counter snapshots; accumulate across this execution's stages.
    completed_counters: list[CounterSnapshot] = field(default_factory=list)
    # Set at each _start_stage so status logs show throughput since stage start.
    stage_monotonic_start: float | None = None
    # True once run_pipeline returned or raised for this execution.
    done: bool = False


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
    coordinator keeps serving the rest. ``shared=True`` marks a long-lived
    coordinator whose workers idle between pipelines instead of exiting when
    a pipeline winds down.
    """

    def __init__(
        self,
        chunk_prefix: str,
        worker_resources: ZephyrTaskResources,
        no_workers_timeout: float = 60.0,
        heartbeat_timeout: float = 120.0,
        max_shard_failures: int = MAX_SHARD_FAILURES,
        max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES,
        shared: bool = False,
    ) -> None:
        # Pipeline executions keyed by execution_id, insertion-ordered. All
        # per-pipeline state lives in the _PipelineExecution values.
        self._executions: dict[str, _PipelineExecution] = {}
        # Per-worker resource pool size; run_pipeline validates task costs fit.
        self._worker_resources = worker_resources
        self._shared = shared
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

        self._stats_writer = StatsWriter.connect()

        logger.info("Coordinator initialized (shared=%s)", shared)

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

    def _mark_stage_complete(self, run: _PipelineExecution) -> None:
        with self._lock:
            run.stage_complete = True

    def _current_execution_locked(self) -> _PipelineExecution | None:
        """Most recent execution, preferring active ones. Lock must be held.

        Non-shared coordinators run exactly one pipeline; this recovers it
        for the legacy single-pipeline worker-lifecycle decisions.
        """
        runs = list(self._executions.values())
        for run in reversed(runs):
            if not run.done:
                return run
        return runs[-1] if runs else None

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
        if not self._shared:
            # After the last stage completes, workers exit cleanly via SHUTDOWN.
            # The worker job finishing at that point is expected, not a crash.
            # (Shared workers only exit at coordinator shutdown, so a finished
            # worker job is always fatal there.)
            with self._lock:
                run = self._current_execution_locked()
                if run is not None and run.total_shards > 0 and run.completed_shards >= run.total_shards:
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

            if self._shared:
                # Shared workers are long-lived: they idle between pipelines
                # and only exit on coordinator shutdown.
                return PullStatus.NO_WORK_BACKOFF, None

            run = self._current_execution_locked()
            if run is None or run.fatal_error is not None or run.task_queue:
                # Nothing submitted yet, an aborted pipeline (the coordinator
                # job is about to tear down), or a head task this worker can't
                # fit — sleep and re-poll.
                return PullStatus.NO_WORK_BACKOFF, None
            if run.done or run.is_last_stage:
                # Last stage has no more tasks to hand out — the pipeline is
                # winding down for this worker. Only exit if this worker has no
                # tasks in flight that could fail and be requeued. If a peer's
                # shard is requeued, Iris restarts that worker to pick it up.
                # _check_worker_group() detects permanent worker-job death as a
                # failsafe so we never deadlock.
                if not any(e.worker_id == worker_id for e in run.in_flight.values()):
                    self._worker_states[worker_id] = WorkerState.DONE
                    return PullStatus.SHUTDOWN, None
                return PullStatus.NO_WORK_BACKOFF, None
            if run.stage_complete:
                # Non-last stage boundary — worker sleeps briefly and polls
                # again for the next stage.
                return PullStatus.STAGE_COMPLETED, None
            return PullStatus.NO_WORK_BACKOFF, None

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
        return CoordinatorInfo(
            chunk_prefix=self._chunk_prefix,
            worker_resources=self._worker_resources,
            shared=self._shared,
        )

    def is_shutdown(self) -> bool:
        """True once shutdown() ran; the shared serve loop polls this to exit."""
        return self._shutdown_event.is_set()

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

        On shared coordinators finished executions are dropped, so totals
        cover only executions still registered.
        """
        with self._lock:
            if worker_id is not None:
                snap = self._worker_counters.get(worker_id)
                if snap is None:
                    return {}
                return {k: e.value for k, e in snap.counters.items() if stage is None or e.stage == stage}

            all_snaps = [snap for run in self._executions.values() for snap in run.completed_counters] + list(
                self._worker_counters.values()
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
            run.task_queue = deque(tasks)
            run.results = {}
            run.stage_name = stage_name
            run.current_stage_index = current_stage_index
            run.total_shards = len(tasks)
            run.completed_shards = 0
            run.retries = 0
            run.in_flight = {}
            run.task_attempts = {task.shard_idx: 0 for task in tasks}
            run.task_error_attempts = {task.shard_idx: 0 for task in tasks}
            # Counts INFRA failures observed while this specific shard was in
            # flight on the dying worker — bounded by MAX_SHARD_INFRA_FAILURES
            # so a shard that deterministically crashes its worker (native
            # SIGSEGV, OOM) eventually aborts instead of retrying forever.
            run.task_infra_attempts = {task.shard_idx: 0 for task in tasks}
            run.shard_errors = {}
            run.fatal_error = None
            # Shared workers never exit eagerly — they idle for the next pipeline.
            run.is_last_stage = is_last_stage and not self._shared
            run.stage_complete = False
            run.stage_monotonic_start = time.monotonic()
            self._progress_time_seconds = time.time()
            run.stage_done.clear()

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
    ) -> ZephyrExecutionResult:
        """Run one pipeline, blocking until done.

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

        if not self._shared:
            # A dedicated coordinator serves exactly one pipeline, so its
            # metrics can carry that run's identity. A shared coordinator
            # serves many at once — a global run label would name whichever
            # pipeline started last.
            telltale.set_global_labels(source="zephyr", run=execution_id)

        try:
            shards = _build_source_shards(plan.source_items)
            if not shards:
                return ZephyrExecutionResult(results=[], counters={})

            last_worker_stage_idx = max(
                (i for i, s in enumerate(plan.stages) if s.stage_type != StageType.RESHARD),
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
            return ZephyrExecutionResult(results=flat_result, counters=counters)
        finally:
            with self._lock:
                run.done = True
                run.task_queue.clear()
                run.in_flight.clear()
                run.results = {}
                if self._shared:
                    # Long-lived coordinator: drop finished executions so state
                    # doesn't grow without bound. Late reports for this
                    # execution are logged and ignored.
                    self._executions.pop(execution_id, None)

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

        self._mark_stage_complete(run)

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
        # Monotonic counter-snapshot generation, bumped from concurrent task
        # threads and the heartbeat thread; guard the read-modify-write so two
        # snapshots never collide on a generation (the coordinator drops
        # non-strictly-greater generations).
        self._counter_generation: int = 0
        self._counter_generation_lock = threading.Lock()
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

        threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator_handle,),
            daemon=True,
            name=f"zephyr-hb-{self._worker_id}",
        ).start()

        threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"zephyr-poll-{self._worker_id}",
        ).start()

    def _poll_loop(self) -> None:
        """Single poll loop: requests tasks from the coordinator using current available resources.

        Registers once with the coordinator, then loops: polls with available CPU and
        memory, spins off a thread per dispatched task, and immediately polls again.
        Stage transitions (STAGE_COMPLETED) are treated the same as NO_WORK_BACKOFF
        — the loop sleeps briefly then re-polls for the next stage's tasks.
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

            if future is None:
                future = self._coordinator.pull_task.remote(self._worker_id, avail)
                future_start = time.monotonic()
                warned = False

            # Short timeout keeps the thread responsive to shutdown without
            # killing it on slow coordinator deserialization.
            try:
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
                # NO_WORK_BACKOFF or STAGE_COMPLETED — sleep until resources free up
                # or new tasks arrive. Wake early if a task completes.
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
                args=(work.task, work.attempt, work.config, runner),
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
        task: ShardTask,
        attempt: int,
        config: dict,
        runner: StageRunner,
    ) -> None:
        """Execute one shard task, report the result, and restore task.cost to the pool."""
        task_start = time.monotonic()
        execution_id = config["execution_id"]
        try:
            result, task_counters = self._execute_shard(task, config, runner)
            logger.info("[%s] Shard %d done in %.2fs", self._worker_id, task.shard_idx, time.monotonic() - task_start)
            # Block until coordinator records result — prevents in_flight races.
            self._coordinator.report_result.remote(
                self._worker_id,
                execution_id,
                task.shard_idx,
                attempt,
                result,
                CounterSnapshot(counters=dict(task_counters), generation=self._next_counter_generation()),
            ).result()
        except Exception:
            logger.error("Worker %s error on shard %d", self._worker_id, task.shard_idx, exc_info=True)
            self._coordinator.report_error.remote(
                self._worker_id,
                execution_id,
                task.shard_idx,
                attempt,
                "".join(traceback.format_exc()),
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

    def _next_counter_generation(self) -> int:
        """Atomically bump and return the worker's counter-snapshot generation."""
        with self._counter_generation_lock:
            self._counter_generation += 1
            return self._counter_generation

    def _heartbeat_counter_snapshot(self) -> CounterSnapshot | None:
        """Aggregate live counters from all active runners; return None if unchanged."""
        with self._resources_lock:
            runners = list(self._active_runners)
        current, _ = merge_counter_entries((name, entry) for r in runners for name, entry in r.live_counters().items())
        if current == self._last_reported_counters:
            return None
        self._last_reported_counters = current
        return CounterSnapshot(counters=current, generation=self._next_counter_generation())

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
        self, task: ShardTask, config: dict, stage_runner: StageRunner
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        chunk_prefix = config["chunk_prefix"]
        execution_id = config["execution_id"]
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


def _stop_worker_group(client: Client, worker_group: Any) -> None:
    """Give workers a brief window to exit on their own, then terminate.

    Workers receive SHUTDOWN from pull_task after coordinator.shutdown() and
    self-terminate; waiting lets their Iris tasks record SUCCEEDED instead of
    KILLED (#5484). LocalActorGroup has no Iris task state to wait on — its
    synthetic job handles are marked succeeded at registration and is_done()
    is permanently False — so the graceful-exit wait would always exhaust its
    full budget without observing any change. Skip it for LocalClient.
    """
    with suppress(Exception):
        if isinstance(client, LocalClient):
            worker_group.shutdown()
            return
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if worker_group.is_done():
                return
            time.sleep(0.5)
        logger.warning("Workers did not exit naturally, terminating")
        worker_group.shutdown()


def _job_environment(
    pip_dependency_groups: list[str] | None,
    job_env_vars: dict[str, str] | None,
) -> EnvironmentConfig | None:
    """Environment for a coordinator + worker pool job: uv dependency extras + env vars.

    A shared pool's generic workers run every connecting pipeline's stage code,
    so the pool must be launched with the union of the stages' extras and any
    env vars they need. Returns None when neither is set, so the job inherits
    the parent environment exactly as before — existing callers are unaffected.
    """
    if not pip_dependency_groups and not job_env_vars:
        return None
    return create_environment(extras=pip_dependency_groups, env_vars=job_env_vars)


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
    # Cloudpickled and re-invoked per worker slot, so per-runner mutable
    # state is per-slot.
    stage_runner_factory: Callable[[], StageRunner]
    map_task_resources: ResourceConfig
    reduce_task_resources: ResourceConfig
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES


def _run_coordinator_job(config_path: str, result_path: str) -> None:
    """Entrypoint for the coordinator job.

    Hosts the coordinator actor in-process via host_actor(), creates
    worker actors as child jobs, runs the pipeline, and writes results
    to disk. The coordinator monitors worker job health directly in its
    maintenance loop (no separate watchdog thread).
    """
    logger.info("Loading coordinator config from %s", config_path)
    config: _CoordinatorJobConfig = cloudpickle.loads(StoragePath(config_path).read_bytes())

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

    # Compute per-task resource costs.
    total = ZephyrTaskResources.from_resource_config(config.worker_resources)
    map_cost = ZephyrTaskResources.from_resource_config(config.map_task_resources)
    reduce_cost = ZephyrTaskResources.from_resource_config(config.reduce_task_resources)

    # Host coordinator actor in this process (no child job needed)
    coord_name = f"zephyr-{config.name}-p{config.pipeline_id}-coord"
    hosted = client.host_actor(
        ZephyrCoordinator,
        config.chunk_storage_prefix,
        total,
        config.no_workers_timeout,
        config.heartbeat_timeout,
        config.max_shard_failures,
        config.max_shard_infra_failures,
        name=coord_name,
        actor_config=ActorConfig(max_concurrency=100),
    )
    coordinator = hosted.handle
    worker_group = None
    # host_actor starts a non-daemon uvicorn thread; the finally below must
    # run on every exit path or the process will stay alive after the main
    # body raises and the Iris task will be stuck RUNNING.
    try:
        # Create workers (child jobs)

        if config.max_workers > 0:
            # Worker name includes attempt ID so that if a stale coordinator
            # process from a previous attempt is still running, its shutdown
            # targets the old name and cannot kill this attempt's workers.
            worker_name = f"zephyr-{config.name}-p{config.pipeline_id}-workers-a{attempt_id}"
            logger.info("Starting %d workers (shards=%d)", config.max_workers, config.plan.num_shards)
            worker_group = client.create_actor_group(
                ZephyrWorker,
                coordinator,
                config.stage_runner_factory,
                total,
                name=worker_name,
                count=config.max_workers,
                resources=config.worker_resources,
                actor_config=ActorConfig(max_task_retries=10),
            )
            ready_wait_s = float(os.environ.get("ZEPHYR_WORKERS_READY_WAIT") or 12 * 60 * 60)
            worker_group.wait_ready(count=1, timeout=ready_wait_s)

            # Let the coordinator poll worker job health in its maintenance loop
            coordinator.set_worker_group.remote(worker_group).result()

        try:
            payload = coordinator.run_pipeline.submit(config.plan, config.execution_id, map_cost, reduce_cost).result()

            ensure_parent_dir(result_path)
            StoragePath(result_path).write_bytes(cloudpickle.dumps(payload))
        except Exception as e:
            # Persist the exception so the caller can recover the original type
            # (important for non-retryable error detection). Normalize first so a
            # subclass that cannot round-trip through pickle does not make the
            # caller's revival crash and mask the real failure.
            with suppress(Exception):
                ensure_parent_dir(result_path)
                StoragePath(result_path).write_bytes(cloudpickle.dumps(_ensure_picklable_exception(e)))
            raise
    finally:
        # Signal coordinator shutdown first so workers receive SHUTDOWN from
        # pull_task and self-terminate via shutdown_event → exit_actor(). Then
        # give the worker job a brief window to land in a terminal state on
        # its own so its Iris tasks record SUCCEEDED instead of KILLED
        # (#5484); fall back to forcibly terminating if they don't.
        with suppress(Exception):
            coordinator.shutdown.remote().result(timeout=10.0)
        if worker_group is not None:
            _stop_worker_group(client, worker_group)
        with suppress(Exception):
            hosted.shutdown()


@dataclass(frozen=True)
class _SharedContextConfig:
    """Serializable config for the shared (long-lived) coordinator job entrypoint."""

    name: str
    chunk_storage_prefix: str
    max_workers: int
    worker_resources: ResourceConfig
    # Cloudpickled and re-invoked per worker slot, so per-runner mutable
    # state is per-slot.
    stage_runner_factory: Callable[[], StageRunner]
    no_workers_timeout: float
    heartbeat_timeout: float
    max_shard_failures: int
    max_shard_infra_failures: int


def _run_shared_coordinator_job(config_path: str, endpoint_path: str) -> None:
    """Entrypoint for a shared, long-lived coordinator job.

    Hosts the coordinator actor, boots a fixed pool of ``max_workers``
    workers, publishes the coordinator's endpoint name to ``endpoint_path``,
    and serves until the creating context shuts it down. Drivers connect via
    ``ZephyrContext(coordinator_endpoint=...)`` and submit pipelines through
    ``run_pipeline``; the coordinator runs them concurrently. The job fails
    (visibly) if the worker pool terminates permanently.
    """
    logger.info("Loading shared coordinator config from %s", config_path)
    config: _SharedContextConfig = cloudpickle.loads(StoragePath(config_path).read_bytes())

    job_info = get_job_info()
    attempt_id = job_info.attempt_id if job_info else 0
    logger.info("Shared coordinator job starting: name=%s, attempt=%d", config.name, attempt_id)

    client = current_client()
    total = ZephyrTaskResources.from_resource_config(config.worker_resources)

    hosted = client.host_actor(
        ZephyrCoordinator,
        config.chunk_storage_prefix,
        total,
        config.no_workers_timeout,
        config.heartbeat_timeout,
        config.max_shard_failures,
        config.max_shard_infra_failures,
        shared=True,
        name=f"zephyr-{config.name}-coord",
        actor_config=ActorConfig(max_concurrency=100),
    )
    coordinator = hosted.handle
    worker_group = None
    try:
        # Worker name includes attempt ID so that if a stale coordinator
        # process from a previous attempt is still running, its shutdown
        # targets the old name and cannot kill this attempt's workers.
        worker_name = f"zephyr-{config.name}-workers-a{attempt_id}"
        logger.info("Starting %d shared workers", config.max_workers)
        worker_group = client.create_actor_group(
            ZephyrWorker,
            coordinator,
            config.stage_runner_factory,
            total,
            name=worker_name,
            count=config.max_workers,
            resources=config.worker_resources,
            actor_config=ActorConfig(max_task_retries=SHARED_POOL_MAX_TASK_RETRIES),
        )
        coordinator.set_worker_group.remote(worker_group).result()

        ensure_parent_dir(endpoint_path)
        StoragePath(endpoint_path).write_bytes(hosted.endpoint.encode())
        logger.info("Shared coordinator ready: endpoint=%s", hosted.endpoint)

        while True:
            if coordinator.is_shutdown.remote().result(timeout=30.0):
                logger.info("Coordinator shut down; exiting serve loop")
                return
            if worker_group.is_done():
                raise RuntimeError(
                    "Shared zephyr worker pool terminated permanently (all retries exhausted); "
                    "shutting the shared context down."
                )
            time.sleep(_SERVE_POLL_INTERVAL)
    finally:
        with suppress(Exception):
            coordinator.shutdown.remote().result(timeout=10.0)
        if worker_group is not None:
            _stop_worker_group(client, worker_group)
        with suppress(Exception):
            hosted.shutdown()


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


def _compute_min_tasks_per_worker(
    worker_resources: ResourceConfig,
    map_resources: ResourceConfig,
    reduce_resources: ResourceConfig,
) -> int:
    """Compute how many concurrent tasks fit on one worker given map/reduce task costs.

    Uses the tighter of the map and reduce packing densities so workers sized for
    both stage types can keep enough tasks in flight for either.
    """
    for field_name in ["device", "preemptible", "regions", "zone", "replicas", "image", "device_alternatives"]:
        map_val = getattr(map_resources, field_name)
        reduce_val = getattr(reduce_resources, field_name)
        if map_val != reduce_val:
            raise ValueError(
                f"Field '{field_name}' cannot differ between map_task_resources ({map_val}) "
                f"and reduce_task_resources ({reduce_val}). Set the same value on both."
            )

    return min(
        _tasks_per_worker(worker_resources, map_resources),
        _tasks_per_worker(worker_resources, reduce_resources),
    )


# 6 hours: long enough to wait out cluster contention for at least one worker.
_DEFAULT_NO_WORKERS_TIMEOUT = 6 * 60 * 60


def _default_max_workers(client: Client) -> int:
    """Default worker count: cpu_count for LocalClient, else ``ZEPHYR_MAX_WORKERS`` or 128."""
    if isinstance(client, LocalClient):
        return os.cpu_count() or 1
    env_val = os.environ.get("ZEPHYR_MAX_WORKERS")
    return int(env_val) if env_val else 128


@dataclass
class ZephyrPool:
    """A shared, long-lived Zephyr coordinator + worker pool.

    Start one pool and run many pipelines against it concurrently — from this
    process and from other Iris jobs. Open the pool as a context manager to get
    the coordinator endpoint; a driver runs pipelines on it by pointing a
    ``ZephyrContext`` at that endpoint::

        with ZephyrPool(name="ingest", max_workers=200, resources=ResourceConfig(cpu=2, ram="8g")) as endpoint:
            ZephyrContext(coordinator_endpoint=endpoint, resources=ResourceConfig(cpu=1, ram="2g")).execute(pipeline)

    Drivers connect either by passing ``coordinator_endpoint=endpoint`` or by
    exporting the endpoint as ``ZEPHYR_COORDINATOR_ENDPOINT`` on their jobs (a
    plain ``ZephyrContext()`` then picks it up). Tasks from all active pipelines
    are packed onto whichever workers have free capacity; a failing pipeline
    only fails its own ``execute()``, and preempted workers are replenished by
    Iris. The pool is torn down when the ``with`` block exits, or on
    ``shutdown()``.

    ``resources`` x ``max_workers`` sizes the worker pool. Connecting pipelines
    declare their own per-task cost via the driver's
    ``map/reduce_task_resources``, independent of the pool's worker size.

    Args:
        client: The fray client to use. If None, auto-detects via current_client().
        max_workers: Number of workers in the pool. If None, defaults to
            os.cpu_count() for LocalClient, or ``ZEPHYR_MAX_WORKERS`` / 128 for
            distributed clients.
        resources: Resource config per worker.
        coordinator_resources: Resource config for the (non-preemptible) coordinator job.
        chunk_storage_prefix: Storage prefix for intermediate chunks. Defaults to a
            temp bucket under MARIN_PREFIX.
        name: Descriptive name, used in the coordinator/worker job names.
        no_workers_timeout: Seconds a pipeline waits for at least one live worker
            before failing a stage.
        stage_runner_factory: Callable ``() -> StageRunner``. Defaults to
            ``InlineRunner`` for LocalClient and ``SubprocessRunner`` otherwise.
        heartbeat_timeout: Seconds without a worker heartbeat before the coordinator
            marks the worker FAILED and requeues its in-flight shard.
        max_shard_failures: Max explicit task-error retries per shard before a
            pipeline aborts (the pipeline, not the pool).
        max_shard_infra_failures: Max infra failures observed while the same shard is
            in flight before treating the shard payload as a deterministic crasher.
        pip_dependency_groups: Extra uv dependency groups the coordinator + worker
            pool job is launched with. Because the pool's generic workers run
            every connecting pipeline's stage functions, a shared pool must carry
            the union of the stages' extras (e.g. ``["datakit"]`` so embed/assign
            workers have luxical/faiss/sklearn/scipy).
        job_env_vars: Env vars set on the coordinator + worker pool job (e.g.
            ``{"JAX_PLATFORMS": "cpu"}`` so jax stages don't probe CUDA when the
            pool spills onto GPU nodes).
    """

    client: Client | None = None
    max_workers: int | None = None
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="1g"))
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=0.1, ram="1g", preemptible=False)
    )
    chunk_storage_prefix: str | None = None
    name: str = ""
    no_workers_timeout: float | None = None
    stage_runner_factory: Callable[[], StageRunner] | None = None
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES
    pip_dependency_groups: list[str] | None = None
    job_env_vars: dict[str, str] | None = None

    # Coordinator endpoint, published once the pool is started.
    endpoint: str | None = field(default=None, repr=False)
    _serve_job: JobHandle | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.client is None:
            self.client = current_client()
        if self.max_workers is None:
            self.max_workers = _default_max_workers(self.client)
        if self.no_workers_timeout is None:
            self.no_workers_timeout = _DEFAULT_NO_WORKERS_TIMEOUT
        if self.chunk_storage_prefix is None:
            self.chunk_storage_prefix = marin_temp_bucket(ttl_days=1, prefix="zephyr")
        if self.stage_runner_factory is None:
            self.stage_runner_factory = _default_stage_runner_factory_for(self.client)
        # Unique per pool so its coordinator job and endpoint file never collide.
        self.name = f"{self.name}-{uuid.uuid4().hex[:8]}"

    def start(self, timeout: float = SHARED_CONTEXT_START_TIMEOUT) -> str:
        """Submit the shared coordinator + worker pool; block until its endpoint is published.

        Returns the coordinator endpoint. Pass it to drivers as
        ``ZephyrContext(coordinator_endpoint=...)`` or export it as
        ``ZEPHYR_COORDINATOR_ENDPOINT``. Call ``shutdown()`` (or use the pool as
        a ``with`` block) to tear it down.
        """
        if self._serve_job is not None:
            raise RuntimeError("Pool is already started")

        config = _SharedContextConfig(
            name=self.name,
            chunk_storage_prefix=self.chunk_storage_prefix,
            max_workers=self.max_workers,
            worker_resources=self.resources,
            stage_runner_factory=self.stage_runner_factory,
            no_workers_timeout=self.no_workers_timeout,
            heartbeat_timeout=self.heartbeat_timeout,
            max_shard_failures=self.max_shard_failures,
            max_shard_infra_failures=self.max_shard_infra_failures,
        )
        # self.name is uuid-suffixed, so this directory is unique per pool.
        base = f"{self.chunk_storage_prefix}/{self.name}-shared"
        config_path = f"{base}/config.pkl"
        endpoint_path = f"{base}/endpoint.txt"
        ensure_parent_dir(config_path)
        StoragePath(config_path).write_bytes(cloudpickle.dumps(config))

        # Set the context var so the coordinator job inherits self.client
        # instead of auto-detecting (which may pick a different backend).
        with set_current_client(self.client):
            self._serve_job = self.client.submit(
                JobRequest(
                    name=f"zephyr-{self.name}-shared",
                    entrypoint=Entrypoint.from_callable(
                        _run_shared_coordinator_job,
                        args=(config_path, endpoint_path),
                    ),
                    resources=self.coordinator_resources,
                    environment=_job_environment(self.pip_dependency_groups, self.job_env_vars),
                )
            )
        logger.info("Shared coordinator job submitted: %s", self._serve_job.job_id)

        # If the job never publishes its endpoint (crash, or slow-scheduling
        # timeout), terminate it before re-raising so we never leak a running
        # coordinator + full worker pool — the stale-coordinator failure mode
        # this feature is meant to avoid.
        try:
            endpoint_file = StoragePath(endpoint_path)
            backoff = ExponentialBackoff(initial=0.2, maximum=5.0)
            deadline = time.monotonic() + timeout
            while not endpoint_file.exists():
                status = self._serve_job.status()
                if status in (FrayJobStatus.FAILED, FrayJobStatus.STOPPED, FrayJobStatus.SUCCEEDED):
                    raise RuntimeError(f"Shared coordinator job exited ({status}) before publishing its endpoint")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Shared coordinator did not publish an endpoint within {timeout}s")
                time.sleep(backoff.next_interval())
        except BaseException:
            with suppress(Exception):
                self._serve_job.terminate()
            self._serve_job = None
            raise

        self.endpoint = endpoint_file.read_bytes().decode()
        logger.info("Shared zephyr coordinator ready: %s", self.endpoint)
        return self.endpoint

    def shutdown(self) -> None:
        """Gracefully stop the coordinator (workers drain and exit) and terminate the pool job."""
        if self._serve_job is None:
            return
        if self.endpoint is not None:
            with suppress(Exception):
                self.client.get_actor(self.endpoint).shutdown.remote().result(timeout=30.0)
        with suppress(Exception):
            # Give the serve loop a moment to exit cleanly (workers land
            # SUCCEEDED); terminate() below is the backstop that cascades.
            self._serve_job.wait(timeout=60.0, raise_on_failure=False)
        with suppress(Exception):
            self._serve_job.terminate()
        self._serve_job = None
        self.endpoint = None

    def __enter__(self) -> str:
        """Start the pool and return its coordinator endpoint; ``__exit__`` tears it down."""
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


@dataclass
class ZephyrContext:
    """Execution context for Zephyr pipelines.

    By default each execute() call submits a coordinator *job* that internally
    creates coordinator and worker actors as child jobs. The coordinator job
    owns the full lifecycle: it boots workers, runs the pipeline, writes
    results to disk, and tears everything down. Iris cascading termination
    ensures that if the coordinator job dies, its children are cleaned up
    automatically.

    To run on a shared, long-lived worker pool instead of a fresh coordinator
    per pipeline, point the context at a ``ZephyrPool``'s endpoint — either by
    passing ``coordinator_endpoint=...`` or by setting the
    ``ZEPHYR_COORDINATOR_ENDPOINT`` env var (which a plain ``ZephyrContext()``
    picks up). In that mode execute() submits the pipeline to the shared
    coordinator, which runs it concurrently with other drivers' pipelines and
    packs tasks onto workers by free capacity, and there is no driver-side
    retry: the shared coordinator is non-preemptible and owns worker recovery.
    ``map/reduce_task_resources`` still describe this pipeline's per-task cost;
    the pool's worker size belongs to the ``ZephyrPool``.

    Args:
        coordinator_endpoint: Endpoint of a shared coordinator to connect to (as
            returned by ``ZephyrPool.start()`` / ``ZephyrPool`` as a context
            manager). When None, falls back to the ``ZEPHYR_COORDINATOR_ENDPOINT``
            env var; when that is also unset, the context is dedicated.
        client: The fray client to use. If None, auto-detects using current_client().
        max_workers: Upper bound on worker count. The actual count is
            min(max_workers, num_shards), computed at first execute(). If None,
            defaults to os.cpu_count() for LocalClient, or ``MAX_WORKERS_PER_JOB``
            (1024) for distributed clients.
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
        stage_runner_factory: Callable ``() -> StageRunner``.
            Defaults to ``InlineRunner`` for ``LocalClient`` and ``SubprocessRunner``
            for distributed clients.
        map_task_resources: ResourceConfig specifying resources required by a single map task.
            Defaults to ``resources``. Requires ``resources`` to be set explicitly.
        reduce_task_resources: ResourceConfig specifying resources required by a single reduce task.
            Defaults to ``map_task_resources``.
        heartbeat_timeout: Seconds without a worker heartbeat before the coordinator
            marks the worker FAILED and requeues its in-flight shard. Defaults to 120.
            Long-running stages (e.g. vLLM inference with cold XLA compile) may need
            to raise this; the JAX/XLA tracer can starve the worker's heartbeat thread
            during compile.
        max_shard_failures: Maximum explicit task-error retries per shard before the
            pipeline aborts. Defaults to ``MAX_SHARD_FAILURES``.
        max_shard_infra_failures: Maximum infra failures (preemption / heartbeat timeout)
            observed while the same shard was in flight before treating the shard payload
            as a deterministic crasher and aborting. Defaults to ``MAX_SHARD_INFRA_FAILURES``.
        pip_dependency_groups: Extra uv dependency groups the *dedicated* coordinator
            + worker job is launched with. Ignored when connected to a shared pool
            (the ``ZephyrPool`` owns the pool's environment).
        job_env_vars: Env vars set on the dedicated coordinator + worker job. Also
            ignored when connected to a shared pool.
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
    # NOTE: 100 is fairly aggressive but it fits the preemptible env better
    max_execution_retries: int = 100
    stage_runner_factory: Callable[[], StageRunner] | None = None
    map_task_resources: ResourceConfig | None = None
    reduce_task_resources: ResourceConfig | None = None
    heartbeat_timeout: float = 120.0
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES
    coordinator_endpoint: str | None = None
    pip_dependency_groups: list[str] | None = None
    job_env_vars: dict[str, str] | None = None

    # Shared data staged by put(), uploaded to disk at the start of execute()
    _shared_data: dict[str, Any] = field(default_factory=dict, repr=False)
    # Handle to the coordinator job (for termination on retry/shutdown)
    _coordinator_job: JobHandle | None = field(default=None, repr=False)
    # Cached describe() of the shared coordinator we are connected to
    _coordinator_info: CoordinatorInfo | None = field(default=None, repr=False)
    # NOTE: execute calls increment this at the very beginning
    _pipeline_id: int = field(default=-1, repr=False)
    min_tasks_per_worker: int = field(init=False, default=1, repr=False)

    def __post_init__(self):
        if self.client is None:
            self.client = current_client()

        # Fall back to the env var so a pool owner can point per-step driver
        # jobs at a shared coordinator without threading the endpoint through
        # code. An explicit `coordinator_endpoint` always wins.
        if self.coordinator_endpoint is None:
            self.coordinator_endpoint = os.environ.get(ZEPHYR_COORDINATOR_ENDPOINT_ENV) or None

        if env_val := os.environ.get("ZEPHYR_MAX_WORKERS"):
            if self.max_workers is None:
                try:
                    self.max_workers = int(env_val)
                except ValueError as e:
                    raise ValueError(f"Invalid ZEPHYR_MAX_WORKERS environment variable value: {env_val}") from e
            else:
                logger.info("Ignoring ZEPHYR_MAX_WORKERS environment variable in favor of max_workers variable.")

        if self.map_task_resources is not None and self.resources is None:
            raise ValueError("Setting map_task_resources without setting resources is an error.")
        if self.reduce_task_resources is not None and self.map_task_resources is None:
            raise ValueError("Setting reduce_task_resources without setting map_task_resources is an error.")

        if self.resources is None:
            self.resources = ResourceConfig(cpu=1, ram="1g")
        if self.map_task_resources is None:
            self.map_task_resources = self.resources
        if self.reduce_task_resources is None:
            self.reduce_task_resources = self.map_task_resources

        # Sizing checks
        resources_ram = humanfriendly.parse_size(self.resources.ram, binary=True)
        resources_disk = humanfriendly.parse_size(self.resources.disk, binary=True)
        for task_resources, name in [
            (self.map_task_resources, "map_task"),
            (self.reduce_task_resources, "reduce_task"),
        ]:
            task_ram = humanfriendly.parse_size(task_resources.ram, binary=True)
            task_disk = humanfriendly.parse_size(task_resources.disk, binary=True)
            if self.resources.cpu < task_resources.cpu or resources_ram < task_ram or resources_disk < task_disk:
                raise ValueError(
                    f"Overall resources ({self.resources}) must be larger than or equal to "
                    f"{name} resources ({task_resources}) on all dimensions (cpu, ram, disk)."
                )

        self.min_tasks_per_worker = _compute_min_tasks_per_worker(
            self.resources, self.map_task_resources, self.reduce_task_resources
        )

        if self.no_workers_timeout is None:
            self.no_workers_timeout = _DEFAULT_NO_WORKERS_TIMEOUT

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

    def _upload_shared_data(self, prefix: str, execution_id: str) -> None:
        """Serialize all staged shared data to disk under the execution directory."""
        for name, obj in self._shared_data.items():
            path = _shared_data_path(prefix, execution_id, name)
            ensure_parent_dir(path)
            t0 = time.monotonic()
            data = cloudpickle.dumps(obj)
            elapsed = time.monotonic() - t0
            StoragePath(path).write_bytes(data)
            logger.info(
                "Shared data '%s' written to %s (serialized %d bytes in %.2fs)",
                name,
                path,
                len(data),
                elapsed,
            )

    def _execute_shared(self, plan: PhysicalPlan) -> ZephyrExecutionResult:
        """Run one pipeline on the shared coordinator, blocking until done.

        There is no driver-side retry loop here: the shared coordinator job
        is non-preemptible and owns worker recovery, so an error surfacing
        means the pipeline (or the whole pool) genuinely failed.
        """
        coordinator = self.client.get_actor(self.coordinator_endpoint)
        if self._coordinator_info is None:
            self._coordinator_info = coordinator.describe.remote().result(timeout=60.0)
        prefix = self._coordinator_info.chunk_prefix

        assert self.map_task_resources is not None and self.reduce_task_resources is not None
        map_cost = ZephyrTaskResources.from_resource_config(self.map_task_resources)
        reduce_cost = ZephyrTaskResources.from_resource_config(self.reduce_task_resources)

        execution_id = _generate_execution_id()
        logger.info("Submitting zephyr pipeline %s to shared coordinator %s", execution_id, self.coordinator_endpoint)
        try:
            self._upload_shared_data(prefix, execution_id)
            return coordinator.run_pipeline.submit(plan, execution_id, map_cost, reduce_cost).result()
        finally:
            _cleanup_execution(prefix, execution_id)

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

        When connected to a shared coordinator (``coordinator_endpoint``),
        the pipeline is submitted there instead and runs concurrently with
        other drivers' pipelines on the shared worker pool.

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

        if plan.num_shards <= 0:
            logger.warning("No shards in plan, returning empty results.")
            return ZephyrExecutionResult(results=[], counters={})

        if self.coordinator_endpoint is not None:
            return self._execute_shared(plan)

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
                self._upload_shared_data(self.chunk_storage_prefix, execution_id)

                assert self.resources is not None

                limit = self.max_workers
                if limit is None and isinstance(self.client, LocalClient):
                    limit = os.cpu_count() or 1

                needed_workers = math.ceil(plan.num_shards / self.min_tasks_per_worker)
                actual_workers = min((limit or MAX_WORKERS_PER_JOB), needed_workers)

                config = _CoordinatorJobConfig(
                    plan=plan,
                    execution_id=execution_id,
                    chunk_storage_prefix=self.chunk_storage_prefix,
                    no_workers_timeout=self.no_workers_timeout,
                    max_workers=actual_workers,
                    worker_resources=self.resources,
                    name=self.name,
                    pipeline_id=self._pipeline_id,
                    map_task_resources=self.map_task_resources,
                    reduce_task_resources=self.reduce_task_resources,
                    stage_runner_factory=self.stage_runner_factory,
                    heartbeat_timeout=self.heartbeat_timeout,
                    max_shard_failures=self.max_shard_failures,
                    max_shard_infra_failures=self.max_shard_infra_failures,
                )
                ensure_parent_dir(config_path)
                StoragePath(config_path).write_bytes(cloudpickle.dumps(config))

                job_name = f"zephyr-{self.name}-p{self._pipeline_id}-a{attempt}"
                # The wrapper job just blocks on child actors; real
                # resources are requested by the coordinator/worker children.
                # Set the context var so the coordinator job inherits self.client
                # instead of auto-detecting (which may pick a different backend).
                with set_current_client(self.client):
                    self._coordinator_job = self.client.submit(
                        JobRequest(
                            name=job_name,
                            entrypoint=Entrypoint.from_callable(
                                _run_coordinator_job,
                                args=(config_path, result_path),
                            ),
                            resources=self.coordinator_resources,
                            environment=_job_environment(self.pip_dependency_groups, self.job_env_vars),
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
        """Terminate a lingering dedicated coordinator job, if any.

        The dedicated ``execute()`` path already tears its coordinator down in
        a ``finally``, so this is only needed as a belt-and-suspenders cleanup.
        A context connected to a shared pool owns no infrastructure — the
        ``ZephyrPool`` owns the pool — so shutdown() never affects the pool.
        """
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
