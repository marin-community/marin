# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dashboard queries over coordinator state."""

import time
from _thread import LockType
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sized
from dataclasses import dataclass
from typing import Protocol

from zephyr.dashboard import (
    SOURCE_STAGE_TYPE,
    CounterPage,
    CounterQuery,
    PipelineList,
    PipelineMetrics,
    PipelinePhase,
    PipelinePlan,
    PipelineStatus,
    PipelineSummary,
    PlanNodeState,
    PlanNodeStatus,
    ResourceUsage,
    WorkerAssignment,
    WorkerPage,
    WorkerQuery,
    WorkerStateCount,
    WorkerStatus,
    counter_value,
    pipeline_plan,
)
from zephyr.plan import PhysicalPlan
from zephyr.stage_io import ZephyrTaskResources
from zephyr.stats import ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, ZEPHYR_WORKER_MEM_CURRENT_KEY, StatsWriter
from zephyr.worker_context import Aggregation, CounterEntry, CounterSnapshot, merge_counter_entries


class InFlightTask(Protocol):
    worker_id: str


class DashboardExecution(Protocol):
    """Execution fields that dashboard queries read under the coordinator lock."""

    execution_id: str
    pipeline_name: str
    plan: PhysicalPlan | None
    dashboard_plan: PipelinePlan | None
    stage_name: str
    current_stage_index: int
    completed_shards: int
    total_shards: int
    retries: int
    started_at_ms: int
    finished_at_ms: int
    fatal_error: str | None
    terminal_error: Exception | None
    done: bool

    @property
    def in_flight(self) -> Mapping[int, InFlightTask]: ...

    @property
    def task_queue(self) -> Sized: ...

    @property
    def completed_totals(self) -> Mapping[tuple[str | None, str, Aggregation], CounterEntry]: ...


class DashboardCoordinator(Protocol):
    """Coordinator state and lock shared with dashboard queries."""

    _lock: LockType
    _worker_resources: ZephyrTaskResources
    _coordinator_task_id: str
    _expected_workers: int
    _stats_writer: StatsWriter

    @property
    def _executions(self) -> Mapping[str, DashboardExecution]: ...

    @property
    def _worker_states(self) -> Mapping[str, str]: ...

    @property
    def _worker_counters(self) -> Mapping[tuple[str, str], CounterSnapshot]: ...

    @property
    def _worker_task_ids(self) -> Mapping[str, str]: ...

    @property
    def _last_seen(self) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class _DashboardWorkerSnapshot:
    worker_id: str
    task_id: str
    state: str
    last_seen_age_seconds: float
    assignments: tuple[tuple[str, int], ...]
    counters: dict[str, CounterEntry]


class CoordinatorDashboard:
    """Read coordinator state and build dashboard responses."""

    def __init__(self, coordinator: DashboardCoordinator, active_worker_state: str) -> None:
        self._coordinator = coordinator
        self._active_worker_state = active_worker_state

    def _run_locked(self, execution_id: str) -> DashboardExecution | None:
        if execution_id:
            return self._coordinator._executions.get(execution_id)
        return next((run for run in reversed(tuple(self._coordinator._executions.values())) if not run.done), None)

    def _phase_locked(self, run: DashboardExecution) -> PipelinePhase:
        if run.fatal_error is not None or run.terminal_error is not None:
            return PipelinePhase.FAILED
        if run.done:
            return PipelinePhase.SUCCEEDED
        if self._active_worker_state not in self._coordinator._worker_states.values():
            return PipelinePhase.WAITING_FOR_WORKERS
        return PipelinePhase.RUNNING

    def pipelines(self) -> PipelineList:
        with self._coordinator._lock:
            active = [run for run in self._coordinator._executions.values() if not run.done]
            return PipelineList(
                pipelines=tuple(
                    PipelineSummary(
                        execution_id=run.execution_id,
                        pipeline_name=run.pipeline_name or run.execution_id,
                        current_stage=run.stage_name,
                    )
                    for run in reversed(active)
                )
            )

    def _plan_locked(self, run: DashboardExecution) -> PipelinePlan:
        if run.dashboard_plan is None and run.plan is not None:
            run.dashboard_plan = pipeline_plan(
                run.plan,
                pipeline_name=run.pipeline_name or run.execution_id,
                execution_id=run.execution_id,
            )
        return run.dashboard_plan or PipelinePlan(pipeline_name="", execution_id=run.execution_id)

    def plan(self, execution_id: str) -> PipelinePlan:
        with self._coordinator._lock:
            run = self._run_locked(execution_id)
            if run is None:
                return PipelinePlan(pipeline_name="", execution_id=execution_id)
            return self._plan_locked(run)

    def _worker_counter_entries_locked(self, worker_id: str) -> dict[str, CounterEntry]:
        merged, conflicted = merge_counter_entries(
            (name, entry)
            for (snapshot_worker_id, _), snapshot in self._coordinator._worker_counters.items()
            if snapshot_worker_id == worker_id
            for name, entry in snapshot.counters.items()
        )
        return {name: entry for name, entry in merged.items() if name not in conflicted}

    def status(self, execution_id: str) -> PipelineStatus:
        with self._coordinator._lock:
            run = self._run_locked(execution_id)
            if run is None:
                return PipelineStatus(execution_id=execution_id)

            worker_states = Counter(state for state in self._coordinator._worker_states.values())
            worker_snapshots = [
                snapshot
                for (worker_id, snapshot_execution_id), snapshot in self._coordinator._worker_counters.items()
                if worker_id in self._coordinator._worker_states and snapshot_execution_id == run.execution_id
            ]
            cpu_percent = sum(
                float(snapshot.counters.get(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, CounterEntry(0)).value)
                for snapshot in worker_snapshots
            )
            memory_bytes = sum(
                int(snapshot.counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY, CounterEntry(0)).value)
                for snapshot in worker_snapshots
            )
            active_workers = worker_states.get(self._active_worker_state, 0)
            cpu_capacity = active_workers * self._coordinator._worker_resources.cpu
            memory_capacity = active_workers * self._coordinator._worker_resources.memory
            safe_plan = self._plan_locked(run)
            nodes_by_id = {node.node_id: node for node in safe_plan.nodes}
            node_statuses: list[PlanNodeStatus] = []
            phase = self._phase_locked(run)
            for node in safe_plan.nodes:
                if node.stage_type == SOURCE_STAGE_TYPE:
                    state = PlanNodeState.SUCCEEDED
                else:
                    parent = nodes_by_id.get(node.parent_node_id)
                    stage_index = parent.stage_index if parent is not None else node.stage_index
                    if stage_index < run.current_stage_index:
                        state = PlanNodeState.SUCCEEDED
                    elif phase is PipelinePhase.FAILED and stage_index == run.current_stage_index:
                        state = PlanNodeState.FAILED
                    elif stage_index > run.current_stage_index or not run.stage_name:
                        state = PlanNodeState.PENDING
                    elif run.done or (run.total_shards > 0 and run.completed_shards >= run.total_shards):
                        state = PlanNodeState.SUCCEEDED
                    else:
                        state = PlanNodeState.RUNNING
                node_statuses.append(PlanNodeStatus(node_id=node.node_id, state=state))

            return PipelineStatus(
                execution_id=run.execution_id,
                phase=phase,
                current_stage=run.stage_name,
                completed_shards=run.completed_shards,
                total_shards=run.total_shards,
                in_flight_shards=len(run.in_flight),
                queued_shards=len(run.task_queue),
                retries=run.retries,
                started_at_ms=run.started_at_ms,
                finished_at_ms=run.finished_at_ms,
                fatal_error=run.fatal_error or (str(run.terminal_error) if run.terminal_error is not None else ""),
                coordinator_task_id=self._coordinator._coordinator_task_id,
                expected_workers=self._coordinator._expected_workers,
                worker_states=tuple(
                    WorkerStateCount(state=state, count=count) for state, count in sorted(worker_states.items())
                ),
                resources=ResourceUsage(
                    cpu_cores=cpu_percent / 100,
                    cpu_utilization=(cpu_percent / 100 / cpu_capacity) if cpu_capacity else 0,
                    memory_bytes=memory_bytes,
                    memory_utilization=(memory_bytes / memory_capacity) if memory_capacity else 0,
                ),
                node_statuses=tuple(node_statuses),
            )

    def metrics(self, execution_id: str, max_points: int) -> PipelineMetrics:
        with self._coordinator._lock:
            run = self._run_locked(execution_id)
            if run is None:
                return PipelineMetrics(warning="The selected pipeline is not active.")
            selected_execution_id = run.execution_id
        result = self._coordinator._stats_writer.query_pipeline_metrics(selected_execution_id, max_points)
        return PipelineMetrics(points=result.points, warning=result.warning)

    def _counter_entries(self, execution_id: str) -> list[tuple[str, CounterEntry]]:
        with self._coordinator._lock:
            run = self._run_locked(execution_id)
            if run is None:
                return []
            entries = [(name, entry) for (_, name, _), entry in run.completed_totals.items()]
            entries.extend(
                (name, entry)
                for (worker_id, snapshot_execution_id), snapshot in self._coordinator._worker_counters.items()
                if worker_id in self._coordinator._worker_states and snapshot_execution_id == run.execution_id
                for name, entry in snapshot.counters.items()
            )

        by_stage: dict[str | None, list[tuple[str, CounterEntry]]] = defaultdict(list)
        for name, entry in entries:
            by_stage[entry.stage].append((name, entry))
        result: list[tuple[str, CounterEntry]] = []
        for stage_entries in by_stage.values():
            merged, conflicted = merge_counter_entries(stage_entries)
            result.extend((name, entry) for name, entry in merged.items() if name not in conflicted)
        return result

    def counters(self, query: CounterQuery) -> CounterPage:
        search = query.search.casefold()
        available_entries = self._counter_entries(query.execution_id)
        stages = tuple(sorted({entry.stage for _, entry in available_entries if entry.stage}))
        entries = [
            (name, entry)
            for name, entry in available_entries
            if (not query.stage or entry.stage == query.stage)
            and (not search or search in name.casefold() or search in (entry.stage or "").casefold())
        ]
        sort_keys: dict[str, Callable[[tuple[str, CounterEntry]], object]] = {
            "name": lambda item: item[0].casefold(),
            "stage": lambda item: (item[1].stage or "").casefold(),
            "value": lambda item: item[1].value,
            "aggregation": lambda item: item[1].aggregation.value,
            "observations": lambda item: item[1].count,
        }
        entries.sort(key=sort_keys.get(query.sort_field, sort_keys["name"]), reverse=query.sort_descending)
        page = entries[query.offset : query.offset + query.limit]
        return CounterPage(
            counters=tuple(counter_value(name, entry) for name, entry in page),
            total=len(entries),
            stages=stages,
        )

    def workers(self, query: WorkerQuery) -> WorkerPage:
        now = time.monotonic()
        with self._coordinator._lock:
            assignments_by_worker: dict[str, list[tuple[str, int]]] = defaultdict(list)
            for run in self._coordinator._executions.values():
                if run.done:
                    continue
                for shard, entry in run.in_flight.items():
                    assignments_by_worker[entry.worker_id].append((run.execution_id, shard))
            snapshots = [
                _DashboardWorkerSnapshot(
                    worker_id=worker_id,
                    task_id=self._coordinator._worker_task_ids.get(worker_id, ""),
                    state=state,
                    last_seen_age_seconds=max(0, now - self._coordinator._last_seen.get(worker_id, now)),
                    assignments=tuple(sorted(assignments_by_worker[worker_id])),
                    counters=self._worker_counter_entries_locked(worker_id),
                )
                for worker_id, state in self._coordinator._worker_states.items()
            ]

        search = query.search.casefold()
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
            "cpu": lambda snapshot: snapshot.counters.get(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, CounterEntry(0)).value,
            "memory": lambda snapshot: snapshot.counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY, CounterEntry(0)).value,
            "active_shards": lambda snapshot: len(snapshot.assignments),
        }
        snapshots.sort(key=sort_keys.get(query.sort_field, sort_keys["worker_id"]), reverse=query.sort_descending)
        workers = tuple(
            WorkerStatus(
                worker_id=snapshot.worker_id,
                task_id=snapshot.task_id,
                state=snapshot.state,
                last_seen_age_seconds=snapshot.last_seen_age_seconds,
                assignments=tuple(
                    WorkerAssignment(execution_id=execution_id, shard=shard)
                    for execution_id, shard in snapshot.assignments
                ),
                cpu_percent=float(snapshot.counters.get(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, CounterEntry(0)).value),
                memory_bytes=int(snapshot.counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY, CounterEntry(0)).value),
            )
            for snapshot in snapshots[query.offset : query.offset + query.limit]
        )
        return WorkerPage(workers=workers, total=len(snapshots))
