# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator actor and pull protocol for Zephyr pipelines."""

import enum
import logging
import re
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any

import cloudpickle
from fray.actor import ActorHandle, current_actor
from iris.cluster.client.job_info import get_job_info
from rigging import telemetry
from rigging.filesystem import StoragePath
from rigging.timing import ExponentialBackoff, RateLimiter, log_time
from starlette.types import ASGIApp

from zephyr.dashboard import (
    DEFAULT_COUNTER_LIMIT,
    DEFAULT_WORKER_LIMIT,
    MAX_COUNTER_LIMIT,
    MAX_WORKER_LIMIT,
    bounded_limit,
    counter_value,
    create_dashboard_application,
    pipeline_plan,
    source_node_id,
    stage_node_id,
)
from zephyr.plan import Join, PhysicalOp, PhysicalPlan, PhysicalStage, Scatter, Shard, SourceItem, StageType
from zephyr.rpc import dashboard_pb2
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
    ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY,
    ZEPHYR_WORKER_MEM_CURRENT_KEY,
    StatsWriter,
    ZephyrWorkerStatStatus,
    _push_iris_task_status,
)
from zephyr.worker_context import Aggregation, CounterEntry, CounterSnapshot, merge_counter_entries
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)

MAX_SHARD_FAILURES = 3
MAX_SHARD_INFRA_FAILURES = 20
MAX_STATUS_TEXT_LENGTH = 1000
MAX_CONCURRENT_PIPELINES = 16
ZEPHYR_PROGRESS_TIME_METRIC = "progress_time_seconds"

_SNAPSHOT_ATTRIBUTES = telemetry.snapshot_attributes("gauge", telemetry.CURRENT_SNAPSHOT)


def _now_ms() -> int:
    return int(time.time() * 1000)


class ShardFailureKind(enum.StrEnum):
    """TASK failures count toward MAX_SHARD_FAILURES; INFRA failures (preemption) do not."""

    TASK = enum.auto()
    INFRA = enum.auto()


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
    plan: PhysicalPlan | None = None
    pipeline_name: str = ""
    started_at_ms: int = 0
    finished_at_ms: int = 0
    task_queue: deque[ShardTask] = field(default_factory=deque)
    results: dict[int, TaskResult] = field(default_factory=dict)
    stage_name: str = ""
    # Joins and reshards use the parent stage index.
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
    progress_time_seconds: float = 0.0
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
        self.finished_at_ms = _now_ms()
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


@dataclass(frozen=True)
class _DashboardWorkerSnapshot:
    worker_id: str
    task_id: str
    state: str
    last_seen_age_seconds: float
    assignments: tuple[tuple[str, int], ...]
    counters: CounterSnapshot


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
        max_concurrent_pipelines: int = MAX_CONCURRENT_PIPELINES,
        expected_workers: int = 0,
    ) -> None:
        # Pipeline executions keyed by execution_id, insertion-ordered. All
        # per-pipeline state lives in the _PipelineExecution values.
        self._executions: dict[str, _PipelineExecution] = {}
        self._worker_resources = worker_resources
        self._expected_workers = expected_workers
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
        self._max_concurrent_pipelines = max_concurrent_pipelines
        # Per-worker in-flight counter snapshots. Each snapshot carries a
        # monotonic generation so the coordinator can discard stale or
        # out-of-order heartbeats.
        self._worker_counters: dict[tuple[str, str], CounterSnapshot] = {}
        self._worker_handles: dict[str, ActorHandle] = {}
        self._worker_task_ids: dict[str, str] = {}
        self._worker_group: Any = None  # ActorGroup, set via set_worker_group()
        self._coordinator_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        # Throttle Iris task-status pushes; the coordinator loop ticks more
        # frequently than the UI needs to refresh.
        self._task_stats_limiter = RateLimiter(interval_seconds=10.0)

        actor_ctx = current_actor()
        self._name = f"{actor_ctx.group_name}"
        self._host_shutdown_event = actor_ctx.shutdown_event
        job_info = get_job_info()
        self._coordinator_task_id = job_info.task_id.to_wire() if job_info is not None else ""

        self._stats_writer = StatsWriter.connect()
        self._web_application = create_dashboard_application(self)
        self._result_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="zephyr-result")

        logger.info("Coordinator initialized")

        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop, daemon=True, name="zephyr-coordinator-loop"
        )
        self._coordinator_thread.start()

    @property
    def web_application(self) -> ASGIApp:
        """Return the dashboard app that shares the actor endpoint."""
        return self._web_application

    def _dashboard_run_locked(self, execution_id: str) -> _PipelineExecution | None:
        if execution_id:
            return self._executions.get(execution_id)
        return next((run for run in reversed(self._executions.values()) if not run.done), None)

    def _dashboard_phase_locked(self, run: _PipelineExecution) -> int:
        if run.fatal_error is not None or run.terminal_error is not None:
            return dashboard_pb2.PIPELINE_PHASE_FAILED
        if run.done:
            return dashboard_pb2.PIPELINE_PHASE_SUCCEEDED
        if not self._worker_states:
            return dashboard_pb2.PIPELINE_PHASE_WAITING_FOR_WORKERS
        return dashboard_pb2.PIPELINE_PHASE_RUNNING

    def _dashboard_pipelines_response(self) -> dashboard_pb2.ListPipelinesResponse:
        with self._lock:
            active = [run for run in self._executions.values() if not run.done]
            summaries = [
                dashboard_pb2.PipelineSummary(
                    execution_id=run.execution_id,
                    pipeline_name=run.pipeline_name or run.execution_id,
                    phase=self._dashboard_phase_locked(run),
                    current_stage=run.stage_name,
                    completed_shards=run.completed_shards,
                    total_shards=run.total_shards,
                    started_at_ms=run.started_at_ms,
                    fatal_error=run.fatal_error or (str(run.terminal_error) if run.terminal_error is not None else ""),
                )
                for run in reversed(active)
            ]
        return dashboard_pb2.ListPipelinesResponse(pipelines=summaries)

    def _dashboard_plan_response(self, execution_id: str) -> dashboard_pb2.GetPlanResponse:
        with self._lock:
            run = self._dashboard_run_locked(execution_id)
            if run is None or run.plan is None:
                return dashboard_pb2.GetPlanResponse(execution_id=execution_id)
            plan = run.plan
            pipeline_name = run.pipeline_name
            selected_execution_id = run.execution_id
        return pipeline_plan(
            plan,
            pipeline_name=pipeline_name or selected_execution_id,
            pipeline_id=0,
            execution_id=selected_execution_id,
        )

    def _worker_counter_snapshot_locked(self, worker_id: str) -> CounterSnapshot:
        merged, conflicted = merge_counter_entries(
            (name, entry)
            for (snapshot_worker_id, _), snapshot in self._worker_counters.items()
            if snapshot_worker_id == worker_id
            for name, entry in snapshot.counters.items()
        )
        return CounterSnapshot(
            counters={name: entry for name, entry in merged.items() if name not in conflicted},
            generation=0,
        )

    def _dashboard_status_response(self, execution_id: str) -> dashboard_pb2.GetStatusResponse:
        with self._lock:
            run = self._dashboard_run_locked(execution_id)
            if run is None:
                return dashboard_pb2.GetStatusResponse(execution_id=execution_id)

            worker_states = Counter(state.value for state in self._worker_states.values())
            worker_snapshots = [
                snapshot
                for (worker_id, snapshot_execution_id), snapshot in self._worker_counters.items()
                if worker_id in self._worker_states and snapshot_execution_id == run.execution_id
            ]
            cpu_percent = sum(
                float(snapshot.counters.get(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, CounterEntry(0)).value)
                for snapshot in worker_snapshots
            )
            memory_bytes = sum(
                int(snapshot.counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY, CounterEntry(0)).value)
                for snapshot in worker_snapshots
            )
            active_workers = worker_states.get(WorkerState.ACTIVE.value, 0)
            cpu_capacity = active_workers * self._worker_resources.cpu
            memory_capacity = active_workers * self._worker_resources.memory
            safe_plan = (
                pipeline_plan(
                    run.plan,
                    pipeline_name=run.pipeline_name or run.execution_id,
                    pipeline_id=0,
                    execution_id=run.execution_id,
                )
                if run.plan is not None
                else dashboard_pb2.GetPlanResponse()
            )
            nodes_by_id = {node.node_id: node for node in safe_plan.nodes}
            node_statuses: list[dashboard_pb2.PlanNodeStatus] = []
            phase = self._dashboard_phase_locked(run)
            for node in safe_plan.nodes:
                if node.stage_type == "SOURCE":
                    state = dashboard_pb2.PLAN_NODE_STATE_SUCCEEDED
                else:
                    parent = nodes_by_id.get(node.parent_node_id)
                    stage_index = parent.stage_index if parent is not None else node.stage_index
                    if stage_index < run.current_stage_index or run.done:
                        state = dashboard_pb2.PLAN_NODE_STATE_SUCCEEDED
                    elif stage_index > run.current_stage_index or not run.stage_name:
                        state = dashboard_pb2.PLAN_NODE_STATE_PENDING
                    elif phase == dashboard_pb2.PIPELINE_PHASE_FAILED:
                        state = dashboard_pb2.PLAN_NODE_STATE_FAILED
                    elif run.total_shards > 0 and run.completed_shards >= run.total_shards:
                        state = dashboard_pb2.PLAN_NODE_STATE_SUCCEEDED
                    else:
                        state = dashboard_pb2.PLAN_NODE_STATE_RUNNING
                node_statuses.append(
                    dashboard_pb2.PlanNodeStatus(
                        node_id=node.node_id,
                        state=state,
                        started_at_ms=run.started_at_ms if state != dashboard_pb2.PLAN_NODE_STATE_PENDING else 0,
                        finished_at_ms=(
                            run.finished_at_ms
                            if state in {dashboard_pb2.PLAN_NODE_STATE_SUCCEEDED, dashboard_pb2.PLAN_NODE_STATE_FAILED}
                            else 0
                        ),
                    )
                )

            current_node_id = (
                stage_node_id("main", run.current_stage_index) if run.stage_name else source_node_id("main")
            )
            return dashboard_pb2.GetStatusResponse(
                execution_id=run.execution_id,
                phase=phase,
                current_node_id=current_node_id,
                current_stage=run.stage_name,
                current_stage_index=run.current_stage_index,
                total_stages=len(run.plan.stages) if run.plan is not None else 0,
                completed_shards=run.completed_shards,
                total_shards=run.total_shards,
                in_flight_shards=len(run.in_flight),
                queued_shards=len(run.task_queue),
                retries=run.retries,
                started_at_ms=run.started_at_ms,
                finished_at_ms=run.finished_at_ms,
                fatal_error=run.fatal_error or (str(run.terminal_error) if run.terminal_error is not None else ""),
                coordinator_task_id=self._coordinator_task_id,
                expected_workers=self._expected_workers,
                worker_states=[
                    dashboard_pb2.WorkerStateCount(state=state, count=count)
                    for state, count in sorted(worker_states.items())
                ],
                resources=dashboard_pb2.ResourceUsage(
                    cpu_cores=cpu_percent / 100,
                    cpu_capacity_cores=cpu_capacity,
                    cpu_utilization=(cpu_percent / 100 / cpu_capacity) if cpu_capacity else 0,
                    memory_bytes=memory_bytes,
                    memory_capacity_bytes=memory_capacity,
                    memory_utilization=(memory_bytes / memory_capacity) if memory_capacity else 0,
                ),
                node_statuses=node_statuses,
            )

    def _dashboard_metrics_response(self, execution_id: str, max_points: int) -> dashboard_pb2.GetMetricsResponse:
        with self._lock:
            run = self._dashboard_run_locked(execution_id)
            selected_execution_id = run.execution_id if run is not None else execution_id
        if not selected_execution_id:
            return dashboard_pb2.GetMetricsResponse(warning="No active pipeline is selected.")
        result = self._stats_writer.query_pipeline_metrics(selected_execution_id, max_points)
        return dashboard_pb2.GetMetricsResponse(
            points=[
                dashboard_pb2.MetricPoint(
                    timestamp_ms=point.timestamp_ms,
                    stage=point.stage,
                    item_rate=point.item_rate,
                    byte_rate=point.byte_rate,
                    cpu_cores=point.cpu_cores,
                    memory_bytes=point.memory_bytes,
                    active_shards=point.active_shards,
                )
                for point in result.points
            ],
            warning=result.warning,
        )

    def _dashboard_counter_entries(self, execution_id: str) -> list[tuple[str, CounterEntry]]:
        with self._lock:
            run = self._dashboard_run_locked(execution_id)
            if run is None:
                return []
            entries = [(name, replace(entry)) for (_, name, _), entry in run.completed_totals.items()]
            entries.extend(
                (name, replace(entry))
                for (worker_id, snapshot_execution_id), snapshot in self._worker_counters.items()
                if worker_id in self._worker_states and snapshot_execution_id == run.execution_id
                for name, entry in snapshot.counters.items()
            )

        by_stage: dict[str | None, list[tuple[str, CounterEntry]]] = defaultdict(list)
        for name, entry in entries:
            by_stage[entry.stage].append((name, entry))
        result: list[tuple[str, CounterEntry]] = []
        for stage, stage_entries in by_stage.items():
            merged, conflicted = merge_counter_entries(stage_entries)
            result.extend(
                (name, replace(entry, stage=stage)) for name, entry in merged.items() if name not in conflicted
            )
        return result

    def _dashboard_counters_response(
        self, request: dashboard_pb2.ListCountersRequest
    ) -> dashboard_pb2.ListCountersResponse:
        search = request.search.casefold()
        entries = [
            (name, entry)
            for name, entry in self._dashboard_counter_entries(request.execution_id)
            if (not request.stage or entry.stage == request.stage)
            and (not search or search in name.casefold() or search in (entry.stage or "").casefold())
        ]
        sort_keys: dict[str, Callable[[tuple[str, CounterEntry]], object]] = {
            "name": lambda item: item[0].casefold(),
            "stage": lambda item: (item[1].stage or "").casefold(),
            "value": lambda item: item[1].value,
            "aggregation": lambda item: item[1].aggregation.value,
            "observations": lambda item: item[1].count,
        }
        entries.sort(key=sort_keys.get(request.sort_field, sort_keys["name"]), reverse=request.sort_descending)
        total = len(entries)
        offset = max(request.offset, 0)
        limit = bounded_limit(request.limit, DEFAULT_COUNTER_LIMIT, MAX_COUNTER_LIMIT)
        return dashboard_pb2.ListCountersResponse(
            counters=[counter_value(name, entry) for name, entry in entries[offset : offset + limit]],
            total=total,
        )

    def _dashboard_workers_response(
        self, request: dashboard_pb2.ListWorkersRequest
    ) -> dashboard_pb2.ListWorkersResponse:
        now = time.monotonic()
        with self._lock:
            assignments_by_worker: dict[str, list[tuple[str, int]]] = defaultdict(list)
            for run in self._executions.values():
                if run.done:
                    continue
                for shard, entry in run.in_flight.items():
                    assignments_by_worker[entry.worker_id].append((run.execution_id, shard))
            snapshots = [
                _DashboardWorkerSnapshot(
                    worker_id=worker_id,
                    task_id=self._worker_task_ids.get(worker_id, ""),
                    state=state.value,
                    last_seen_age_seconds=max(0, now - self._last_seen.get(worker_id, now)),
                    assignments=tuple(sorted(assignments_by_worker[worker_id])),
                    counters=self._worker_counter_snapshot_locked(worker_id),
                )
                for worker_id, state in self._worker_states.items()
            ]

        search = request.search.casefold()
        snapshots = [
            snapshot
            for snapshot in snapshots
            if not search
            or search in snapshot.worker_id.casefold()
            or search in snapshot.task_id.casefold()
            or search in snapshot.state.casefold()
            or any(search in execution_id.casefold() for execution_id, _ in snapshot.assignments)
        ]
        sort_keys: dict[str, Callable[[_DashboardWorkerSnapshot], object]] = {
            "worker_id": lambda snapshot: snapshot.worker_id.casefold(),
            "state": lambda snapshot: snapshot.state,
            "last_seen": lambda snapshot: snapshot.last_seen_age_seconds,
            "cpu": (
                lambda snapshot: snapshot.counters.counters.get(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, CounterEntry(0)).value
            ),
            "memory": (
                lambda snapshot: snapshot.counters.counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY, CounterEntry(0)).value
            ),
            "active_shards": lambda snapshot: len(snapshot.assignments),
        }
        snapshots.sort(key=sort_keys.get(request.sort_field, sort_keys["worker_id"]), reverse=request.sort_descending)
        total = len(snapshots)
        offset = max(request.offset, 0)
        limit = bounded_limit(request.limit, DEFAULT_WORKER_LIMIT, MAX_WORKER_LIMIT)
        workers = [
            dashboard_pb2.WorkerStatus(
                worker_id=snapshot.worker_id,
                task_id=snapshot.task_id,
                state=snapshot.state,
                last_seen_age_seconds=snapshot.last_seen_age_seconds,
                assignments=[
                    dashboard_pb2.WorkerAssignment(execution_id=execution_id, shard=shard)
                    for execution_id, shard in snapshot.assignments
                ],
                cpu_percent=float(
                    snapshot.counters.counters.get(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, CounterEntry(0)).value
                ),
                memory_bytes=int(snapshot.counters.counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY, CounterEntry(0)).value),
            )
            for snapshot in snapshots[offset : offset + limit]
        ]
        return dashboard_pb2.ListWorkersResponse(workers=workers, total=total)

    def set_worker_group(self, worker_group: Any) -> None:
        """Set the worker ActorGroup so the coordinator can detect permanent worker death."""
        self._worker_group = worker_group

    def register_worker(self, worker_id: str, worker_handle: ActorHandle, task_id: str = "") -> None:
        """Called by workers when they come online to register with coordinator.

        Handles re-registration from reconstructed workers (e.g. after node
        preemption) by updating the stale handle and resetting worker state.

        Returns the current stage epoch.
        """
        with self._lock:
            if worker_id in self._worker_handles:
                logger.info("Worker %s re-registering (likely reconstructed), updating handle", worker_id)
                self._worker_handles[worker_id] = worker_handle
                self._worker_task_ids[worker_id] = task_id
                self._worker_states[worker_id] = WorkerState.ACTIVE
                self._last_seen[worker_id] = time.monotonic()
                # NOTE: if there was a task assigned to the worker, there's a race condition between marking
                # the worker as unhealthy via heartbeat and re-registration. If we do not requeue we may silently
                # lose tasks.
                self._maybe_requeue_worker_tasks(worker_id)
            else:
                self._worker_handles[worker_id] = worker_handle
                self._worker_task_ids[worker_id] = task_id
                self._worker_states[worker_id] = WorkerState.ACTIVE
                self._last_seen[worker_id] = time.monotonic()
                logger.info("Worker %s registered, total: %d", worker_id, len(self._worker_handles))

    def deregister_worker(self, worker_id: str) -> None:
        """Remove a sub-worker that has finished its stage pool."""
        with self._lock:
            self._worker_handles.pop(worker_id, None)
            self._worker_states.pop(worker_id, None)
            self._last_seen.pop(worker_id, None)
            self._worker_task_ids.pop(worker_id, None)
            for key in [key for key in self._worker_counters if key[0] == worker_id]:
                self._worker_counters.pop(key)

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
            snapshots = [
                (
                    run.execution_id,
                    run.progress_time_seconds,
                    {name: entry.value for name, entry in run.merged_counters().items()},
                )
                for run in self._executions.values()
                if not run.done
            ]
        for execution_id, progress_time_seconds, counters in snapshots:
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
        try:
            self._publish_telemetry()
        except Exception:
            logger.warning("Failed to publish coordinator telemetry", exc_info=True)
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
                        *(
                            snapshot
                            for (_, execution_id), snapshot in self._worker_counters.items()
                            if execution_id == run.execution_id
                        ),
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
            # those stages.
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
    ) -> None:
        """Requeue an in-flight shard. Abort its execution at a per-shard limit.

        TASK errors are bounded by ``MAX_SHARD_FAILURES``. INFRA failures
        observed while the *same* shard was in flight are bounded by
        ``MAX_SHARD_INFRA_FAILURES``. This limit stops a shard that repeatedly
        causes a native crash or OOM.

        Must be called with the lock held.
        """
        entry = run.in_flight.pop(shard_idx, None)

        # Zero counters but keep the generation watermark so late heartbeats
        # from the old task are rejected.
        counter_key = (worker_id, run.execution_id)
        existing = self._worker_counters.get(counter_key)
        if existing is not None:
            self._worker_counters[counter_key] = CounterSnapshot.empty(existing.generation)

        if entry is None:
            return

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
                return

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
                    f"(max {self._max_shard_infra_failures} infra failures while in flight). "
                    f"Last failure on worker {worker_id}."
                )
                return

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
            # A requeued shard has no valid output location to write to after a failure.
            logger.info("[%s] Not requeuing shard %d: execution already failed", run.execution_id, shard_idx)
            return

        run.task_queue.append(task)
        run.retries += 1

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
                # snapshot so it stops changing the execution totals until
                # the worker's next heartbeat lands.
                self._clear_worker_inflight_counters(worker_id, execution_id, counter_snapshot.generation)
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
            run.progress_time_seconds = time.time()
            run.in_flight.pop(shard_idx, None)
            run.fold_counters(counter_snapshot)
            # Zero the in-flight counters but keep the generation watermark
            # so late heartbeats from this task are rejected.
            self._clear_worker_inflight_counters(worker_id, execution_id, counter_snapshot.generation)
            run.stage_done.set()

    def _clear_worker_inflight_counters(self, worker_id: str, execution_id: str, generation: int) -> None:
        """Zero a worker's in-flight snapshot, keeping the generation watermark.

        Called when a task's runner is done, so late heartbeats from that task
        (strictly-lower generation) are rejected and the finished task's live
        counters stop being folded into execution totals. Lock must be held.
        """
        counter_key = (worker_id, execution_id)
        existing = self._worker_counters.get(counter_key)
        watermark = max(generation, existing.generation) if existing is not None else generation
        self._worker_counters[counter_key] = CounterSnapshot.empty(watermark)

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
                counter_key = (worker_id, execution_id)
                existing = self._worker_counters.get(counter_key)
                if existing is not None:
                    self._worker_counters[counter_key] = CounterSnapshot.empty(existing.generation)
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

    def heartbeat(self, worker_id: str, counter_snapshots: dict[str, CounterSnapshot] | None = None) -> None:
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            for execution_id, counter_snapshot in (counter_snapshots or {}).items():
                counter_key = (worker_id, execution_id)
                existing = self._worker_counters.get(counter_key)
                if existing is None or counter_snapshot.generation > existing.generation:
                    self._worker_counters[counter_key] = counter_snapshot

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
                worker_snapshots = [
                    snapshot
                    for (snapshot_worker_id, _), snapshot in self._worker_counters.items()
                    if snapshot_worker_id == worker_id
                ]
                return _aggregate_counter_snapshots(worker_snapshots, stage)

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

        This applies when the worker job stops permanently or the maintenance
        loop fails. New run_pipeline calls are rejected afterwards.
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
            run.progress_time_seconds = time.time()

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
        pipeline_name: str,
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
                if active >= self._max_concurrent_pipelines:
                    raise RuntimeError(
                        f"Coordinator already runs {active} concurrent pipelines "
                        f"(max {self._max_concurrent_pipelines})"
                    )
                run = _PipelineExecution(
                    execution_id=execution_id,
                    map_cost=map_cost,
                    reduce_cost=reduce_cost,
                    plan=plan,
                    pipeline_name=pipeline_name,
                    started_at_ms=_now_ms(),
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

            materialized = self._result_executor.map(list, shards)

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
            for key in [key for key in self._worker_counters if key[1] == execution_id]:
                self._worker_counters.pop(key)

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

        self._result_executor.shutdown(wait=True, cancel_futures=True)
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
