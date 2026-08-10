# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed operational controller APIs backed by persistence."""

from dataclasses import dataclass

from finelog.client import Table
from rigging.timing import Duration, Timestamp

from iris.cluster.controller.budget import compute_user_spend
from iris.cluster.controller.persistence import operations as ops
from iris.cluster.controller.persistence import reads, writes
from iris.cluster.controller.persistence.checkpoint import CHECKPOINT_EPOCH_META_KEY
from iris.cluster.controller.persistence.database import ControllerDB, Tx
from iris.cluster.controller.persistence.json_codec import resource_spec_from_job_row, worker_metadata_from_row
from iris.cluster.controller.task_state import TaskDetailRow
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.federation.protocol import (
    FederationJobDelta,
    FederationSyncBatch,
    SyncedAttempt,
    SyncedEndpoint,
    SyncedJob,
    SyncedTask,
)
from iris.cluster.types import PendingTask
from iris.resources.execution import ResourceSpec
from iris.resources.names import (
    JobName,
    WorkerId,
)
from iris.resources.state import JobState
from iris.resources.worker import WorkerMetadata


@dataclass(frozen=True, slots=True)
class WorkerDetail:
    worker: reads.WorkerRecord
    attributes: dict[str, str | int | float]
    metadata: WorkerMetadata
    running_tasks: frozenset[JobName]


@dataclass(frozen=True, slots=True)
class UserStateCounts:
    user: str
    task_state_counts: dict[int, int]
    job_state_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class RawQueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class UserBudgetView:
    user_id: str
    budget_limit: int
    budget_spent: int
    max_band: int


@dataclass(frozen=True, slots=True)
class SchedulerStateInputs:
    budgets: tuple[reads.UserBudget, ...]
    user_spend: dict[str, int]
    pending_rows: tuple[PendingTask, ...]
    pending_requested_bands: dict[JobName, int]
    running_rows: tuple[reads.RunningTaskBandRecord, ...]


@dataclass(frozen=True, slots=True)
class BackendCounts:
    pending_by_backend: dict[str, int]
    running_by_backend: dict[str, int]
    workers_by_scale_group: dict[str, int]


@dataclass(frozen=True, slots=True)
class FederatedRoute:
    peer_id: str


class DatabaseOperations:
    """Database persistence operations."""

    def __init__(self, database: ControllerDB) -> None:
        self._database = database

    def attach_task_event_table(self, table: Table) -> None:
        self._database.attach_task_event_table(table)

    def probe_database(self) -> int | None:
        with self._database.read_snapshot() as transaction:
            return reads.probe_database(transaction, CHECKPOINT_EPOCH_META_KEY)

    def raw_query(self, sql: str) -> RawQueryResult:
        with self._database.read_snapshot() as transaction:
            result = reads.execute_raw_select(transaction, sql)
        return RawQueryResult(tuple(result.columns), tuple(tuple(row) for row in result.rows))


class FederationOperations:
    """Federation persistence operations."""

    def __init__(self, database: ControllerDB) -> None:
        self._database = database

    def received_requester(self, job_id: JobName) -> str | None:
        with self._database.read_snapshot() as snapshot:
            handoff = reads.received_handoff(snapshot, job_id)
        return handoff.requester_id if handoff is not None else None

    def federated_handle(self, root_job: JobName) -> FederatedRoute | None:
        with self._database.read_snapshot() as snapshot:
            handle = reads.federated_handle(snapshot, root_job)
        return FederatedRoute(handle.peer_id) if handle is not None else None

    def federation_batch(
        self,
        requester_id: str,
        cursor: str,
        *,
        backend_ids: tuple[str, ...],
    ) -> FederationSyncBatch:
        """Build one snapshot-consistent native federation delta batch."""
        cursor_sequence = int(cursor) if cursor else 0
        deltas: list[FederationJobDelta] = []
        with self._database.read_snapshot() as snapshot:
            minimum_sequence = reads.changelog_min_seq(snapshot)
            next_cursor = str(reads.changelog_max_seq(snapshot))
            endpoints = self._federation_endpoints(snapshot, requester_id, Timestamp.now())
            stale = (not cursor) or (minimum_sequence > 0 and cursor_sequence < minimum_sequence - 1)
            if stale:
                for job_id in reads.received_jobs_for_requester(snapshot, requester_id):
                    delta = self._federation_delta(
                        snapshot,
                        job_id,
                        all_tasks=True,
                        task_indexes=set(),
                        backend_ids=backend_ids,
                    )
                    if delta is not None:
                        deltas.append(delta)
                return FederationSyncBatch(tuple(deltas), next_cursor, True, endpoints)

            tombstoned: dict[JobName, bool] = {}
            all_tasks: dict[JobName, bool] = {}
            indexes: dict[JobName, set[int]] = {}
            order: list[JobName] = []
            for row in reads.changelog_rows_since(snapshot, requester_id, cursor_sequence):
                if row.job_id not in tombstoned:
                    tombstoned[row.job_id] = False
                    all_tasks[row.job_id] = False
                    indexes[row.job_id] = set()
                    order.append(row.job_id)
                if row.tombstone:
                    tombstoned[row.job_id] = True
                elif tombstoned[row.job_id]:
                    tombstoned[row.job_id] = False
                    all_tasks[row.job_id] = True
                elif row.task_index is None:
                    all_tasks[row.job_id] = True
                else:
                    indexes[row.job_id].add(row.task_index)

            for job_id in order:
                if tombstoned[job_id]:
                    deltas.append(FederationJobDelta(job_id, None, (), True))
                    continue
                delta = self._federation_delta(
                    snapshot,
                    job_id,
                    all_tasks=all_tasks[job_id],
                    task_indexes=indexes[job_id],
                    backend_ids=backend_ids,
                )
                if delta is not None:
                    deltas.append(delta)
        return FederationSyncBatch(tuple(deltas), next_cursor, False, endpoints)

    def _federation_delta(
        self,
        snapshot: Tx,
        job_id: JobName,
        *,
        all_tasks: bool,
        task_indexes: set[int],
        backend_ids: tuple[str, ...],
    ) -> FederationJobDelta | None:
        job = reads.get_job_detail(snapshot, job_id)
        if job is None:
            return None
        task_rows = [
            row
            for row in reads.task_details_for_job(snapshot, job_id)
            if all_tasks or row.task_id.task_index in task_indexes
        ]
        attempts_by_task = reads.all_attempts_for_tasks(snapshot, [row.task_id for row in task_rows])
        changed_tasks = tuple(
            self._federation_task(row, attempts_by_task.get(row.task_id, ()), backend_ids) for row in task_rows
        )
        return FederationJobDelta(
            job_id=job_id,
            summary=SyncedJob(
                job_id=job.job_id,
                state=job.state,
                error_message=job.error or "",
                exit_code=job.exit_code,
                submitted_at=job.submitted_at_ms,
                started_at=job.started_at_ms,
                finished_at=job.finished_at_ms,
                task_count=job.num_tasks,
                backend_id=_canonical_backend_id(str(job.backend_id or ""), backend_ids),
                resources=resource_spec_from_job_row(job),
            ),
            changed_tasks=changed_tasks,
            tombstone=False,
        )

    @staticmethod
    def _federation_task(
        row: TaskDetailRow,
        attempts: tuple[reads.AttemptRecord, ...],
        backend_ids: tuple[str, ...],
    ) -> SyncedTask:
        current = attempts[-1] if attempts else None
        worker_id = current.worker_id if current is not None else row.current_worker_id
        return SyncedTask(
            task_id=row.task_id,
            state=row.state,
            error_message=row.error or "",
            exit_code=row.exit_code,
            submitted_at=row.submitted_at_ms,
            started_at=current.started_at_ms if current is not None else None,
            finished_at=current.finished_at_ms if current is not None else None,
            current_attempt_id=row.current_attempt_id,
            worker_address=row.current_worker_address or "",
            worker_label=str(worker_id or row.peer_worker_label or ""),
            status_message=row.status_message or "",
            backend_id=_canonical_backend_id(str(row.backend_id or ""), backend_ids),
            attempts=tuple(
                SyncedAttempt(
                    attempt_id=attempt.attempt_id,
                    state=attempt.state,
                    exit_code=attempt.exit_code,
                    error_message=attempt.error or "",
                    attempt_uid=attempt.attempt_uid,
                    started_at=attempt.started_at_ms,
                    finished_at=attempt.finished_at_ms,
                )
                for attempt in attempts
            ),
        )

    @staticmethod
    def _federation_endpoints(snapshot: Tx, requester_id: str, now: Timestamp) -> tuple[SyncedEndpoint, ...]:
        endpoints = []
        for endpoint in reads.live_endpoints_for_requester(snapshot, requester_id, now):
            remaining = None
            if endpoint.lease_deadline is not None:
                remaining = Duration.from_ms(max(0, endpoint.lease_deadline.epoch_ms() - now.epoch_ms()))
            endpoints.append(
                SyncedEndpoint(
                    endpoint_id=endpoint.endpoint_id,
                    name=endpoint.name,
                    address=endpoint.address,
                    task_id=endpoint.task_id,
                    access=endpoint.access,
                    metadata=dict(endpoint.metadata),
                    lease_remaining=remaining,
                )
            )
        return tuple(endpoints)


class WorkerOperations:
    """Worker persistence operations."""

    def __init__(self, database: ControllerDB) -> None:
        self._database = database

    def register_worker(
        self,
        *,
        worker_id: WorkerId,
        address: str,
        metadata: WorkerMetadata,
        timestamp: Timestamp,
        health: WorkerHealthTracker,
        slice_id: str,
        scale_group: str,
    ) -> None:
        with self._database.transaction() as transaction:
            ops.worker.register(
                transaction,
                worker_id=worker_id,
                address=address,
                metadata=metadata,
                ts=timestamp,
                health=health,
                slice_id=slice_id,
                scale_group=scale_group,
            )

    def stale_workers_at_address(self, worker_id: WorkerId, address: str) -> list[WorkerId]:
        with self._database.read_snapshot() as snapshot:
            return reads.worker_ids_at_address(snapshot, address, exclude=worker_id)

    def worker(self, worker_id: WorkerId) -> reads.WorkerRecord | None:
        with self._database.read_snapshot() as snapshot:
            return reads.get_worker_detail(snapshot, worker_id)

    def worker_detail(self, worker_id: WorkerId) -> WorkerDetail | None:
        with self._database.read_snapshot() as snapshot:
            worker = reads.get_worker_detail(snapshot, worker_id)
            if worker is None:
                return None
            attributes = reads.worker_attributes_by_id(snapshot, [worker_id]).get(worker_id, {})
            running = reads.running_tasks_by_worker(snapshot, {worker_id}).get(worker_id, set())
        return WorkerDetail(
            worker=worker,
            attributes=attributes,
            metadata=worker_metadata_from_row(worker, attributes),
            running_tasks=frozenset(running),
        )

    def worker_roster(self) -> list[tuple[reads.WorkerRecord, dict[str, str | int | float]]]:
        with self._database.read_snapshot() as snapshot:
            workers = reads.all_worker_details(snapshot)
            if not workers:
                return []
            attributes = reads.worker_attributes_by_id(snapshot, [worker.worker_id for worker in workers])
        return [(worker, attributes.get(worker.worker_id, {})) for worker in workers]

    def recent_worker_attempts(
        self, worker_id: WorkerId, limit: int
    ) -> tuple[list[reads.AttemptRecord], dict[JobName, ResourceSpec]]:
        with self._database.read_snapshot() as snapshot:
            attempts = reads.recent_attempts_for_worker(snapshot, worker_id, limit=limit)
            job_ids = {row.task_id.parent for row in attempts if row.task_id.parent is not None}
            resources = reads.resource_specs_for_jobs(snapshot, job_ids)
        return list(attempts), resources


class TaskOperations:
    """Task persistence operations."""

    def __init__(self, database: ControllerDB) -> None:
        self._database = database

    def task_detail_with_attempts(
        self, task_id: JobName
    ) -> tuple[TaskDetailRow, tuple[reads.AttemptRecord, ...]] | None:
        with self._database.read_snapshot() as snapshot:
            task = reads.get_task_detail(snapshot, task_id)
            if task is None:
                return None
            attempts = reads.attempts_for_task(snapshot, task_id)
        return task, attempts


class UserOperations:
    """User persistence operations."""

    def __init__(self, database: ControllerDB) -> None:
        self._database = database

    def live_user_stats(self) -> list[UserStateCounts]:
        active_states = (JobState.PENDING, JobState.BUILDING, JobState.RUNNING)
        with self._database.read_snapshot() as snapshot:
            rows = reads.live_user_state_counts(snapshot, active_states)
        return [
            UserStateCounts(row.user_id, task_state_counts=row.task_states, job_state_counts=row.job_states)
            for row in rows
        ]

    def set_user_budget(self, user_id: str, budget_limit: int, max_band: int, timestamp: Timestamp) -> None:
        with self._database.transaction() as transaction:
            writes.set_user_budget(transaction, user_id, budget_limit, max_band, timestamp)

    def user_budget(self, user_id: str) -> UserBudgetView | None:
        with self._database.read_snapshot() as snapshot:
            budget = reads.get_user_budget(snapshot, user_id)
            if budget is None:
                return None
            spend = compute_user_spend(snapshot)
        return UserBudgetView(budget.user_id, budget.budget_limit, spend.get(user_id, 0), budget.max_band)

    def user_budgets(self) -> tuple[UserBudgetView, ...]:
        with self._database.read_snapshot() as snapshot:
            budgets = reads.list_user_budgets(snapshot)
            spend = compute_user_spend(snapshot)
        return tuple(
            UserBudgetView(budget.user_id, budget.budget_limit, spend.get(budget.user_id, 0), budget.max_band)
            for budget in budgets
        )


class SchedulingOperations:
    """Scheduling persistence operations."""

    def __init__(self, database: ControllerDB) -> None:
        self._database = database

    def scheduler_state_inputs(self) -> SchedulerStateInputs:
        with self._database.read_snapshot() as snapshot:
            budgets = tuple(reads.list_user_budgets(snapshot))
            user_spend = compute_user_spend(snapshot)
            pending_rows = tuple(reads.pending_tasks_with_jobs(snapshot))
            pending_bands = reads.get_priority_bands(snapshot, {row.job_id for row in pending_rows})
            running_rows = tuple(reads.running_task_band_rows(snapshot))
        return SchedulerStateInputs(budgets, user_spend, pending_rows, pending_bands, running_rows)

    def backend_counts(self, pending_state: int, running_state: int) -> BackendCounts:
        with self._database.read_snapshot() as snapshot:
            return BackendCounts(
                pending_by_backend=reads.task_counts_by_backend(snapshot, pending_state),
                running_by_backend=reads.task_counts_by_backend(snapshot, running_state),
                workers_by_scale_group=reads.worker_counts_by_scale_group(snapshot),
            )


@dataclass(frozen=True, slots=True)
class OperationalServices:
    """Named persistence-backed services consumed by the legacy RPC adapter."""

    database: DatabaseOperations
    federation: FederationOperations
    workers: WorkerOperations
    tasks: TaskOperations
    users: UserOperations
    scheduling: SchedulingOperations

    @classmethod
    def from_database(cls, database: ControllerDB) -> "OperationalServices":
        return cls(
            database=DatabaseOperations(database),
            federation=FederationOperations(database),
            workers=WorkerOperations(database),
            tasks=TaskOperations(database),
            users=UserOperations(database),
            scheduling=SchedulingOperations(database),
        )


def _canonical_backend_id(stored: str, backend_ids: tuple[str, ...]) -> str:
    if stored:
        return stored
    if len(backend_ids) == 1:
        return backend_ids[0]
    return ""
