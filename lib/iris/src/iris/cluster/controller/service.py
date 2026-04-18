# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller RPC service implementation handling job, task, and worker operations.

The controller expands jobs into tasks at submission time (a job with replicas=N
creates N tasks). Tasks are the unit of scheduling and execution. Job state is
aggregated from task states.
"""

import json
import logging
import secrets
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.constraints import Constraint, constraints_from_resources, merge_constraints
from iris.cluster.controller.auth import (
    DEFAULT_JWT_TTL_SECONDS,
    ControllerAuth,
    create_api_key,
    list_api_keys,
    revoke_api_key,
    revoke_login_keys_for_user,
)
from iris.rpc.auth import (
    AuthzAction,
    authorize,
    authorize_resource_owner,
    get_verified_identity,
    get_verified_user,
    require_identity,
)
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    API_KEYS,
    ATTEMPTS,
    ENDPOINTS,
    JOBS,
    TASKS,
    TERMINAL_JOB_STATES,
    TXN_ACTIONS,
    WORKERS,
    WORKER_RESOURCE_HISTORY,
    WORKER_TASK_HISTORY,
    ControllerDB,
    Endpoint,
    EndpointQuery,
    Job,
    Order,
    Task,
    TaskJobSummary,
    UserStats,
    Worker,
    _tasks_with_attempts,
    endpoint_query_predicate,
    running_tasks_by_worker,
    tasks_for_job_with_attempts,
)
from iris.cluster.controller.pending_diagnostics import PendingHint, build_job_pending_hints
from iris.cluster.controller.query import execute_raw_query
from iris.rpc import query_pb2
from iris.cluster.controller.scheduler import SchedulingContext
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.log_store import PROCESS_LOG_KEY, LogStore, task_log_key
from iris.cluster.process_status import get_process_status
from iris.cluster.runtime.profile import is_system_target, parse_profile_target, profile_local_process
from iris.cluster.types import JobName, TaskAttempt, WorkerId
from iris.rpc import cluster_pb2, vm_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.rpc.proto_utils import job_state_name, task_state_name
from iris.time_utils import Timestamp, Timer

logger = logging.getLogger(__name__)

DEFAULT_TRANSACTION_LIMIT = 50
DEFAULT_MAX_TOTAL_LINES = 100000

# Maximum bundle size in bytes (25 MB) - matches client-side limit
MAX_BUNDLE_SIZE_BYTES = 25 * 1024 * 1024

USER_TASK_STATES = (
    cluster_pb2.TASK_STATE_PENDING,
    cluster_pb2.TASK_STATE_ASSIGNED,
    cluster_pb2.TASK_STATE_BUILDING,
    cluster_pb2.TASK_STATE_RUNNING,
    cluster_pb2.TASK_STATE_SUCCEEDED,
    cluster_pb2.TASK_STATE_FAILED,
    cluster_pb2.TASK_STATE_KILLED,
    cluster_pb2.TASK_STATE_UNSCHEDULABLE,
    cluster_pb2.TASK_STATE_WORKER_FAILED,
)
USER_JOB_STATES = (
    cluster_pb2.JOB_STATE_PENDING,
    cluster_pb2.JOB_STATE_BUILDING,
    cluster_pb2.JOB_STATE_RUNNING,
    cluster_pb2.JOB_STATE_SUCCEEDED,
    cluster_pb2.JOB_STATE_FAILED,
    cluster_pb2.JOB_STATE_KILLED,
    cluster_pb2.JOB_STATE_WORKER_FAILED,
    cluster_pb2.JOB_STATE_UNSCHEDULABLE,
)


def task_to_proto(task: Task, worker_address: str = "") -> cluster_pb2.TaskStatus:
    """Convert a task row to a TaskStatus proto.

    Handles attempt conversion, timestamps, and resource_usage.
    The caller is responsible for resolving worker_address from worker_id if needed.
    """
    current_attempt = task.current_attempt

    attempts = []
    for attempt in task.attempts:
        proto_attempt = cluster_pb2.TaskAttempt(
            attempt_id=attempt.attempt_id,
            worker_id=str(attempt.worker_id) if attempt.worker_id else "",
            state=attempt.state,
            exit_code=attempt.exit_code or 0,
            error=attempt.error or "",
            is_worker_failure=attempt.is_worker_failure,
        )
        if attempt.started_at is not None:
            proto_attempt.started_at.CopyFrom(attempt.started_at.to_proto())
        if attempt.finished_at is not None:
            proto_attempt.finished_at.CopyFrom(attempt.finished_at.to_proto())
        attempts.append(proto_attempt)

    proto = cluster_pb2.TaskStatus(
        task_id=task.task_id.to_wire(),
        state=task.state,
        worker_id=str(task.active_worker_id) if task.active_worker_id else "",
        worker_address=worker_address or task.current_worker_address or "",
        exit_code=task.exit_code or 0,
        error=task.error or "",
        current_attempt_id=task.current_attempt_id,
        attempts=attempts,
    )
    if current_attempt and current_attempt.started_at:
        proto.started_at.CopyFrom(current_attempt.started_at.to_proto())
    if current_attempt and current_attempt.finished_at:
        proto.finished_at.CopyFrom(current_attempt.finished_at.to_proto())
    if task.resource_usage:
        proto.resource_usage.CopyFrom(task.resource_usage)
    # For pending tasks with prior terminal attempts, surface retry context.
    if task.state == cluster_pb2.TASK_STATE_PENDING and task.attempts and task.attempts[-1].is_terminal:
        last = task.attempts[-1]
        proto.pending_reason = (
            f"Retrying (attempt {len(task.attempts)}, " f"last: {cluster_pb2.TaskState.Name(last.state).lower()})"
        )
        proto.can_be_scheduled = True
    return proto


def worker_status_message(w: Worker) -> str:
    """Build a human-readable status message for unhealthy workers."""
    if w.healthy:
        return ""
    if w.consecutive_failures > 0:
        age = w.last_heartbeat.age_ms()
        return f"Heartbeat timeout ({w.consecutive_failures} failures, last seen {age // 1000}s ago)"
    return "Unhealthy (no failures recorded)"


_WORKER_TARGET_PREFIX = "/system/worker/"


def _parse_worker_target(target: str) -> str | None:
    """Extract worker_id from a /system/worker/<worker_id> target.

    Returns the worker_id string, or None if the target does not match.
    """
    if target.startswith(_WORKER_TARGET_PREFIX):
        worker_id = target[len(_WORKER_TARGET_PREFIX) :]
        if worker_id:
            return worker_id
    return None


def _task_state_key(state: int) -> str:
    """Return the lowercase RPC key for a task state enum."""
    return task_state_name(state).removeprefix("TASK_STATE_").lower()


def _job_state_key(state: int) -> str:
    """Return the lowercase RPC key for a job state enum."""
    return job_state_name(state).removeprefix("JOB_STATE_").lower()


def _active_job_count(job_state_counts: dict[int, int]) -> int:
    """Return the count of non-terminal jobs in a user aggregate."""
    return sum(count for state, count in job_state_counts.items() if state not in TERMINAL_JOB_STATES)


def _task_state_counts_for_summary(task_state_counts: dict[int, int]) -> dict[str, int]:
    """Convert enum-keyed task counts to the string-keyed RPC shape."""
    counts = {_task_state_key(state): 0 for state in USER_TASK_STATES}
    for state, count in task_state_counts.items():
        counts[_task_state_key(state)] = count
    return counts


def _job_state_counts_for_summary(job_state_counts: dict[int, int]) -> dict[str, int]:
    """Convert enum-keyed job counts to the string-keyed RPC shape."""
    counts = {_job_state_key(state): 0 for state in USER_JOB_STATES}
    for state, count in job_state_counts.items():
        counts[_job_state_key(state)] = count
    return counts


# =============================================================================
# DB query helpers — thin wrappers over snapshot() for common read patterns
# =============================================================================


def _read_job(db: ControllerDB, job_id: JobName) -> Job | None:
    with db.read_snapshot() as q:
        return q.one(JOBS, where=JOBS.c.job_id == job_id.to_wire())


def _read_task_with_attempts(db: ControllerDB, task_id: JobName) -> Task | None:
    task_wire = task_id.to_wire()
    with db.read_snapshot() as q:
        task = q.one(TASKS, where=TASKS.c.task_id == task_wire)
        if task is None:
            return None
        attempts = q.select(
            ATTEMPTS,
            where=ATTEMPTS.c.task_id == task_wire,
            order_by=(ATTEMPTS.c.attempt_id.asc(),),
        )
    return _tasks_with_attempts([task], attempts)[0]


def _read_worker(db: ControllerDB, worker_id: WorkerId) -> Worker | None:
    with db.read_snapshot() as q:
        return q.one(WORKERS, where=WORKERS.c.worker_id == str(worker_id))


@dataclass(frozen=True)
class _WorkerDetail:
    worker: Worker
    running_tasks: frozenset[JobName]
    resource_history: tuple[cluster_pb2.WorkerResourceSnapshot, ...]


def _read_worker_detail(
    db: ControllerDB, worker_id: WorkerId, *, resource_history_limit: int = 200
) -> _WorkerDetail | None:
    with db.read_snapshot() as q:
        worker = q.one(WORKERS, where=WORKERS.c.worker_id == str(worker_id))
        if worker is None:
            return None
        running_rows = q.raw(
            "SELECT t.task_id FROM tasks t "
            "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            "WHERE a.worker_id = ? AND t.state IN (?, ?, ?)",
            (str(worker_id), *ACTIVE_TASK_STATES),
            decoders={"task_id": JobName.from_wire},
        )
        resource_rows = q.select(
            WORKER_RESOURCE_HISTORY,
            columns=(WORKER_RESOURCE_HISTORY.c.snapshot_proto,),
            where=WORKER_RESOURCE_HISTORY.c.worker_id == str(worker_id),
            order_by=(WORKER_RESOURCE_HISTORY.c.id.desc(),),
            limit=max(resource_history_limit, 0),
        )
    resource_history = tuple(reversed([r.snapshot_proto for r in resource_rows]))
    return _WorkerDetail(
        worker=worker,
        running_tasks=frozenset(r.task_id for r in running_rows),
        resource_history=resource_history,
    )


def _child_jobs(db: ControllerDB, job_id: JobName) -> list[Job]:
    with db.read_snapshot() as q:
        return q.select(
            JOBS,
            where=JOBS.c.parent_job_id == job_id.to_wire(),
            order_by=(Order(JOBS.c.submitted_at_ms), Order(JOBS.c.job_id)),
        )


def _tasks_for_listing(db: ControllerDB, *, job_id: JobName | None = None) -> list[Task]:
    with db.read_snapshot() as q:
        tasks = q.select(
            TASKS,
            where=(TASKS.c.job_id == job_id.to_wire()) if job_id else None,
            order_by=((TASKS.c.job_id.asc(), TASKS.c.task_index.asc()) if job_id else (TASKS.c.task_id.asc(),)),
        )
        if not tasks:
            return []
        attempts = q.select(
            ATTEMPTS,
            where=ATTEMPTS.c.task_id.in_([t.task_id.to_wire() for t in tasks]),
            order_by=(ATTEMPTS.c.task_id.asc(), ATTEMPTS.c.attempt_id.asc()),
        )
    return _tasks_with_attempts(tasks, attempts)


def _worker_addresses_for_tasks(db: ControllerDB, tasks: list[Task]) -> dict[WorkerId, str]:
    """Fetch addresses only for workers referenced by the given tasks."""
    worker_ids = {t.worker_id for t in tasks if t.worker_id is not None}
    if not worker_ids:
        return {}
    placeholders = ",".join("?" for _ in worker_ids)
    with db.read_snapshot() as q:
        rows = q.raw(
            f"SELECT worker_id, address FROM workers WHERE worker_id IN ({placeholders})",
            tuple(str(wid) for wid in worker_ids),
        )
    return {WorkerId(str(row.worker_id)): row.address for row in rows}


# State display order for sorting (active states first)
_STATE_SORT_EXPR = (
    "CASE j.state"
    " WHEN 3 THEN 0"  # RUNNING
    " WHEN 2 THEN 1"  # BUILDING
    " WHEN 1 THEN 2"  # PENDING
    " WHEN 4 THEN 3"  # SUCCEEDED
    " WHEN 5 THEN 4"  # FAILED
    " WHEN 6 THEN 5"  # KILLED
    " WHEN 7 THEN 6"  # WORKER_FAILED
    " WHEN 8 THEN 7"  # UNSCHEDULABLE
    " ELSE 99 END"
)

_SORT_FIELD_TO_SQL: dict[int, str] = {
    cluster_pb2.Controller.JOB_SORT_FIELD_DATE: "j.submitted_at_ms",
    cluster_pb2.Controller.JOB_SORT_FIELD_NAME: "j.name",
    cluster_pb2.Controller.JOB_SORT_FIELD_STATE: _STATE_SORT_EXPR,
    cluster_pb2.Controller.JOB_SORT_FIELD_FAILURES: "agg_failures",
    cluster_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS: "agg_preemptions",
}


def _jobs_paginated(
    db: ControllerDB,
    states: tuple[int, ...],
    *,
    state_filter_int: int | None = None,
    name_filter: str = "",
    sort_field: int = 0,
    descending: bool = True,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[Job], int]:
    """Fetch a page of top-level jobs with SQL-level filtering, sorting, and pagination."""
    conditions = ["j.depth = 1"]
    params: list[object] = []

    if state_filter_int is not None:
        conditions.append("j.state = ?")
        params.append(state_filter_int)
    else:
        state_placeholders = ",".join("?" for _ in states)
        conditions.append(f"j.state IN ({state_placeholders})")
        params.extend(states)

    if name_filter:
        conditions.append("LOWER(j.name) LIKE ?")
        params.append(f"%{name_filter}%")

    where_clause = " AND ".join(conditions)
    direction = "DESC" if descending else "ASC"
    order_expr = _SORT_FIELD_TO_SQL.get(sort_field, "j.submitted_at_ms")

    count_sql = f"SELECT COUNT(*) FROM jobs j WHERE {where_clause}"

    # Only join tasks when sorting by failure/preemption aggregates.
    # The common case (sort by date, name, state) skips the expensive LEFT JOIN + GROUP BY.
    needs_task_agg = sort_field in (
        cluster_pb2.Controller.JOB_SORT_FIELD_FAILURES,
        cluster_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS,
    )

    if needs_task_agg:
        select_sql = f"""
            SELECT j.*,
                   COALESCE(SUM(t.failure_count), 0) AS agg_failures,
                   COALESCE(SUM(t.preemption_count), 0) AS agg_preemptions
            FROM jobs j
            LEFT JOIN tasks t ON j.job_id = t.job_id
            WHERE {where_clause}
            GROUP BY j.job_id
            ORDER BY {order_expr} {direction}
        """
    else:
        select_sql = f"""
            SELECT j.*
            FROM jobs j
            WHERE {where_clause}
            ORDER BY {order_expr} {direction}
        """

    select_params = list(params)
    if limit > 0:
        select_sql += " LIMIT ? OFFSET ?"
        select_params.extend([limit, offset])

    with db.read_snapshot() as q:
        total = q.execute_sql(count_sql, tuple(params)).fetchone()[0]
        rows = q.execute_sql(select_sql, tuple(select_params)).fetchall()

    jobs = [db.decode_job(row) for row in rows]
    return jobs, total


def _descendants_for_roots(db: ControllerDB, root_job_ids: list[str]) -> list[Job]:
    """Fetch all descendant jobs (depth > 1) for the given root job IDs in a single query."""
    if not root_job_ids:
        return []
    placeholders = ",".join("?" for _ in root_job_ids)
    sql = f"""
        SELECT j.*
        FROM jobs j
        WHERE j.root_job_id IN ({placeholders}) AND j.depth > 1
    """
    with db.read_snapshot() as q:
        rows = q.execute_sql(sql, tuple(root_job_ids)).fetchall()
    return [db.decode_job(row) for row in rows]


def _task_summaries_for_jobs(db: ControllerDB, job_ids: set[JobName] | None = None) -> dict[JobName, TaskJobSummary]:
    """Aggregate task counts per job using SQL GROUP BY instead of Python-side iteration."""
    if job_ids is not None:
        placeholders = ",".join("?" for _ in job_ids)
        where = f"WHERE t.job_id IN ({placeholders})"
        params: tuple[object, ...] = tuple(j.to_wire() for j in job_ids)
    else:
        where = ""
        params = ()

    sql = f"""
        SELECT t.job_id,
               t.state,
               COUNT(*) as cnt,
               SUM(t.failure_count) as total_failures,
               SUM(t.preemption_count) as total_preemptions
        FROM tasks t
        {where}
        GROUP BY t.job_id, t.state
    """
    completed_states = (cluster_pb2.TASK_STATE_SUCCEEDED, cluster_pb2.TASK_STATE_KILLED)
    with db.read_snapshot() as q:
        rows = q.raw(sql, params, decoders={"job_id": JobName.from_wire})

    summaries: dict[JobName, TaskJobSummary] = {}
    for row in rows:
        prev = summaries.get(row.job_id, TaskJobSummary(job_id=row.job_id))
        summaries[row.job_id] = TaskJobSummary(
            job_id=row.job_id,
            task_count=prev.task_count + row.cnt,
            completed_count=prev.completed_count + (row.cnt if row.state in completed_states else 0),
            failure_count=prev.failure_count + row.total_failures,
            preemption_count=prev.preemption_count + row.total_preemptions,
            task_state_counts={**prev.task_state_counts, row.state: row.cnt},
        )
    return summaries


def _worker_roster(db: ControllerDB) -> list[Worker]:
    with db.read_snapshot() as q:
        return q.select(WORKERS)


def _query_endpoints(db: ControllerDB, query: EndpointQuery = EndpointQuery()) -> list[Endpoint]:
    joins, where = endpoint_query_predicate(query)
    with db.read_snapshot() as q:
        return q.select(
            ENDPOINTS,
            where=where,
            joins=tuple(joins),
            limit=query.limit,
        )


def _descendant_jobs(db: ControllerDB, job_id: JobName) -> list[Job]:
    with db.read_snapshot() as q:
        return q.select(JOBS, where=JOBS.c.job_id.like(f"{job_id.to_wire()}/%"))


def _transaction_actions(db: ControllerDB, limit: int = 100) -> list:
    with db.read_snapshot() as q:
        actions = q.select(
            TXN_ACTIONS,
            order_by=(TXN_ACTIONS.c.created_at_ms.desc(),),
            limit=limit,
        )
    return list(reversed(actions))


def _live_user_stats(db: ControllerDB) -> list[UserStats]:
    """Aggregate job/task counts per user for active (non-terminal) jobs."""
    active_states = ",".join(
        str(s)
        for s in (
            cluster_pb2.JOB_STATE_PENDING,
            cluster_pb2.JOB_STATE_BUILDING,
            cluster_pb2.JOB_STATE_RUNNING,
        )
    )
    with db.read_snapshot() as q:
        job_rows = q.raw(
            f"SELECT j.user_id, j.state, COUNT(*) as cnt FROM jobs j "
            f"WHERE j.state IN ({active_states}) GROUP BY j.user_id, j.state"
        )
        task_rows = q.raw(
            f"SELECT j.user_id, t.state, COUNT(*) as cnt "
            f"FROM tasks t JOIN jobs j ON t.job_id = j.job_id "
            f"WHERE j.state IN ({active_states}) "
            f"GROUP BY j.user_id, t.state"
        )
    by_user: dict[str, UserStats] = {}
    for row in job_rows:
        stats = by_user.setdefault(row.user_id, UserStats(user=row.user_id))
        stats.job_state_counts[row.state] = row.cnt
    for row in task_rows:
        stats = by_user.setdefault(row.user_id, UserStats(user=row.user_id))
        stats.task_state_counts[row.state] = row.cnt
    return list(by_user.values())


def _tasks_for_worker(db: ControllerDB, worker_id: WorkerId, limit: int = 50) -> list[Task]:
    with db.read_snapshot() as q:
        history_rows = q.select(
            WORKER_TASK_HISTORY,
            columns=(WORKER_TASK_HISTORY.c.task_id,),
            where=WORKER_TASK_HISTORY.c.worker_id == str(worker_id),
            order_by=(WORKER_TASK_HISTORY.c.assigned_at_ms.desc(),),
            limit=limit,
        )
    task_ids = [r.task_id for r in history_rows]
    if not task_ids:
        return []
    task_wires = [tid.to_wire() for tid in task_ids]
    with db.read_snapshot() as q:
        tasks = q.select(
            TASKS,
            where=TASKS.c.task_id.in_(task_wires),
            order_by=(TASKS.c.task_id.asc(),),
        )
        attempts = q.select(
            ATTEMPTS,
            where=ATTEMPTS.c.task_id.in_(task_wires),
            order_by=(ATTEMPTS.c.task_id.asc(), ATTEMPTS.c.attempt_id.asc()),
        )
    task_map = {t.task_id: t for t in _tasks_with_attempts(tasks, attempts)}
    return [task for tid in task_ids if (task := task_map.get(tid)) is not None]


class AutoscalerProtocol(Protocol):
    """Protocol for autoscaler operations used by ControllerServiceImpl."""

    def get_status(self) -> vm_pb2.AutoscalerStatus:
        """Get autoscaler status."""
        ...

    def get_vm(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Get info for a specific VM."""
        ...

    def check_coscheduling_feasibility(
        self,
        replicas: int,
        constraints: list[cluster_pb2.Constraint],
    ) -> str | None:
        """Check if a coscheduled job can be scheduled. Returns error message or None."""
        ...

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get initialization log for a VM."""
        ...


class StubFactoryProtocol(Protocol):
    """Protocol for getting cached worker RPC stubs."""

    def get_stub(self, address: str) -> WorkerServiceClientSync: ...
    def evict(self, address: str) -> None: ...


class ControllerProtocol(Protocol):
    """Protocol for controller operations used by ControllerServiceImpl."""

    def wake(self) -> None: ...

    def kill_tasks_on_workers(self, task_ids: set[JobName]) -> None: ...

    def create_scheduling_context(self, workers: list[Worker]) -> SchedulingContext: ...

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None: ...

    def begin_checkpoint(self) -> tuple[str, Any]: ...

    @property
    def autoscaler(self) -> AutoscalerProtocol | None: ...

    stub_factory: StubFactoryProtocol


def _inject_resource_constraints(
    request: cluster_pb2.Controller.LaunchJobRequest,
) -> cluster_pb2.Controller.LaunchJobRequest:
    """Merge auto-generated device constraints into a job submission request.

    Constraints derived from ResourceSpecProto.device (device-type, device-variant)
    are merged with any explicit user constraints on the request.  For canonical
    keys the user's explicit constraints replace auto-generated ones, so e.g.
    a user-provided multi-variant IN constraint overrides the single-variant
    EQ constraint from the resource spec.
    """
    auto = constraints_from_resources(request.resources)
    if not auto:
        return request

    user = [Constraint.from_proto(c) for c in request.constraints]
    merged = merge_constraints(auto, user)

    new_request = cluster_pb2.Controller.LaunchJobRequest()
    new_request.CopyFrom(request)
    del new_request.constraints[:]
    for c in merged:
        new_request.constraints.append(c.to_proto())
    return new_request


class ControllerServiceImpl:
    """ControllerService RPC implementation.

    Args:
        transitions: State machine for DB mutations (submit, cancel, register, etc.)
        db: Query interface for direct DB reads
        controller: Controller runtime for scheduling and worker management
        bundle_store: Bundle store for zip storage.
        log_store: Log store for task and process logs.
    """

    def __init__(
        self,
        transitions: ControllerTransitions,
        db: ControllerDB,
        controller: ControllerProtocol,
        bundle_store: BundleStore,
        log_store: LogStore,
        auth: ControllerAuth | None = None,
    ):
        self._transitions = transitions
        self._db = db
        self._controller = controller
        self._bundle_store = bundle_store
        self._log_store = log_store
        self._timer = Timer()
        self._auth = auth or ControllerAuth()

    def bundle_zip(self, bundle_id: str) -> bytes:
        return self._bundle_store.get_zip(bundle_id)

    def _get_autoscaler_pending_hints(self) -> dict[str, PendingHint]:
        """Build autoscaler-based pending hints keyed by job id."""
        autoscaler = self._controller.autoscaler
        if autoscaler is None:
            return {}
        status = autoscaler.get_status()
        if not status.HasField("last_routing_decision"):
            return {}
        return build_job_pending_hints(status.last_routing_decision)

    def _authorize_job_owner(self, job_id: JobName) -> None:
        """Raise PERMISSION_DENIED if the authenticated user doesn't own this job.

        Skipped when no auth provider is configured (null-auth mode).
        """
        if not self._auth.provider:
            return
        authorize_resource_owner(job_id.user)

    def launch_job(
        self,
        request: cluster_pb2.Controller.LaunchJobRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.LaunchJobResponse:
        """Submit a new job to the controller.

        The job is expanded into tasks based on the replicas field
        (defaulting to 1). Each task has ID "/job/.../index".
        """
        if not request.name:
            raise ConnectError(Code.INVALID_ARGUMENT, "Job name is required")

        job_id = JobName.from_wire(request.name)

        # When an auth provider is configured, override the user segment with
        # the verified identity to prevent impersonation. Only override for
        # root-level submissions; child jobs inherit the parent's user.
        verified_user = get_verified_user()
        if self._auth.provider and verified_user is not None and job_id.is_root:
            job_id = JobName.root(verified_user, job_id.name)

        # For non-root jobs, verify the caller owns the parent hierarchy
        if self._auth.provider and verified_user is not None and not job_id.is_root:
            self._authorize_job_owner(job_id)

        # Reject submissions if the parent job has already terminated
        if job_id.parent:
            parent_job = _read_job(self._db, job_id.parent)
            if parent_job and parent_job.is_finished():
                raise ConnectError(
                    Code.FAILED_PRECONDITION,
                    f"Cannot submit job: parent job {job_id.parent} has terminated "
                    f"(state={cluster_pb2.JobState.Name(parent_job.state)})",
                )

        existing_job = _read_job(self._db, job_id)
        if existing_job:
            policy = request.existing_job_policy
            if policy == cluster_pb2.EXISTING_JOB_POLICY_ERROR:
                raise ConnectError(
                    Code.ALREADY_EXISTS,
                    f"Job {job_id} already exists (state={cluster_pb2.JobState.Name(existing_job.state)})",
                )
            elif policy == cluster_pb2.EXISTING_JOB_POLICY_KEEP:
                if not existing_job.is_finished():
                    return cluster_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())
                # Job finished, replace it (KEEP only preserves running jobs)
                self._transitions.remove_finished_job(job_id)
            elif policy == cluster_pb2.EXISTING_JOB_POLICY_RECREATE:
                if not existing_job.is_finished():
                    self._transitions.cancel_job(job_id, "Replaced by new submission")
                self._transitions.remove_finished_job(job_id)
            elif existing_job.is_finished():
                # Default/UNSPECIFIED: replace finished jobs
                logger.info(
                    "Replacing finished job %s (state=%s) with new submission",
                    job_id,
                    cluster_pb2.JobState.Name(existing_job.state),
                )
                self._transitions.remove_finished_job(job_id)
            else:
                raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists and is still running")

        # Handle bundle_blob: upload to bundle store, then replace blob
        # with the resulting GCS path (preserving all other fields).
        if request.bundle_blob:
            # Validate bundle size
            bundle_size = len(request.bundle_blob)
            if bundle_size > MAX_BUNDLE_SIZE_BYTES:
                bundle_size_mb = bundle_size / (1024 * 1024)
                max_size_mb = MAX_BUNDLE_SIZE_BYTES / (1024 * 1024)
                raise ConnectError(
                    Code.INVALID_ARGUMENT,
                    f"Bundle size {bundle_size_mb:.1f}MB exceeds maximum {max_size_mb:.0f}MB",
                )

            bundle_id = self._bundle_store.write_zip(request.bundle_blob)

            new_request = cluster_pb2.Controller.LaunchJobRequest()
            new_request.CopyFrom(request)
            new_request.ClearField("bundle_blob")
            new_request.bundle_id = bundle_id
            request = new_request

        # Auto-inject device constraints from the resource spec.
        # Explicit user constraints for canonical keys (device-type,
        # device-variant, etc.) replace auto-generated ones.
        request = _inject_resource_constraints(request)

        # Reject coscheduled jobs that can never be scheduled: if no scaling
        # group has num_vms matching the replica count, the job would sit in
        # the queue forever.
        if request.HasField("coscheduling"):
            autoscaler = self._controller.autoscaler
            if autoscaler is not None:
                error = autoscaler.check_coscheduling_feasibility(
                    replicas=request.replicas,
                    constraints=list(request.constraints),
                )
                if error:
                    raise ConnectError(
                        Code.FAILED_PRECONDITION,
                        f"Job is unschedulable: {error}",
                    )

        self._transitions.submit_job(job_id, request, Timestamp.now())
        self._controller.wake()

        num_tasks = len(tasks_for_job_with_attempts(self._db, job_id))
        logger.info(f"Job {job_id} submitted with {num_tasks} task(s)")
        return cluster_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())

    def get_job_status(
        self,
        request: cluster_pb2.Controller.GetJobStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetJobStatusResponse:
        """Get status of a specific job including all task statuses."""
        job = _read_job(self._db, JobName.from_wire(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        # Build task statuses with attempts, aggregate counts in single pass
        tasks = tasks_for_job_with_attempts(self._db, job.job_id)
        worker_addr_by_id = _worker_addresses_for_tasks(self._db, tasks)

        task_statuses = []
        total_failure_count = 0
        total_preemption_count = 0
        for task in tasks:
            total_failure_count += task.failure_count
            total_preemption_count += task.preemption_count

            task_statuses.append(task_to_proto(task, worker_address=worker_addr_by_id.get(task.worker_id, "")))

        # Get scheduling diagnostics for pending jobs from cache
        # (populated each scheduling cycle by the controller).
        pending_reason = ""
        if job.state == cluster_pb2.JOB_STATE_PENDING:
            sched_reason = self._controller.get_job_scheduling_diagnostics(job.job_id.to_wire())
            pending_reason = sched_reason or "Pending scheduler feedback"
            hint = self._get_autoscaler_pending_hints().get(job.job_id.to_wire())
            if hint is not None:
                scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
                pending_reason = f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

        # Build the JobStatus proto and set timestamps
        proto_job_status = cluster_pb2.JobStatus(
            job_id=job.job_id.to_wire(),
            state=job.state,
            error=job.error or "",
            exit_code=job.exit_code or 0,
            failure_count=total_failure_count,
            preemption_count=total_preemption_count,
            tasks=task_statuses,
            name=job.request.name if job.request else "",
            pending_reason=pending_reason,
        )
        if job.request:
            proto_job_status.resources.CopyFrom(job.request.resources)
        if job.started_at:
            proto_job_status.started_at.CopyFrom(job.started_at.to_proto())
        if job.finished_at:
            proto_job_status.finished_at.CopyFrom(job.finished_at.to_proto())
        if job.submitted_at:
            proto_job_status.submitted_at.CopyFrom(job.submitted_at.to_proto())

        return cluster_pb2.Controller.GetJobStatusResponse(
            job=proto_job_status,
            request=job.request,
        )

    def terminate_job(
        self,
        request: cluster_pb2.Controller.TerminateJobRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Terminate a running job and all its children.

        Cascade termination is performed depth-first: all children are
        terminated before the parent. All tasks within each job are killed.
        """
        job_id = JobName.from_wire(request.job_id)
        job = _read_job(self._db, job_id)
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

        self._authorize_job_owner(job_id)
        self._terminate_job_tree(job_id)
        return cluster_pb2.Empty()

    def _terminate_job_tree(self, job_id: JobName) -> None:
        """Recursively terminate a job and all its descendants (depth-first)."""
        job = _read_job(self._db, job_id)
        if not job:
            return

        # First, terminate all children recursively
        children = _child_jobs(self._db, job_id)
        for child in children:
            self._terminate_job_tree(child.job_id)

        if job.is_finished():
            return

        result = self._transitions.cancel_job(job_id, reason="Terminated by user")

        # Send kill RPCs to workers for any tasks that were killed
        if result.tasks_to_kill:
            self._controller.kill_tasks_on_workers(result.tasks_to_kill)

    def _job_to_proto(
        self,
        j: Job,
        task_summary: TaskJobSummary | None,
        autoscaler_pending_hints: dict[str, PendingHint],
    ) -> cluster_pb2.JobStatus:
        """Convert a Job + its task summary into a JobStatus proto."""
        job_name = j.request.name if j.request else ""
        task_state_counts = (
            {_task_state_key(state): count for state, count in task_summary.task_state_counts.items()}
            if task_summary
            else {}
        )

        pending_reason = j.error or ""
        if j.state == cluster_pb2.JOB_STATE_PENDING:
            sched_reason = self._controller.get_job_scheduling_diagnostics(j.job_id.to_wire())
            pending_reason = sched_reason or "Pending scheduler feedback"
            hint = autoscaler_pending_hints.get(j.job_id.to_wire())
            if hint is not None:
                scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
                pending_reason = f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

        proto_job = cluster_pb2.JobStatus(
            job_id=j.job_id.to_wire(),
            state=j.state,
            error=j.error or "",
            exit_code=j.exit_code or 0,
            failure_count=task_summary.failure_count if task_summary else 0,
            preemption_count=task_summary.preemption_count if task_summary else 0,
            name=job_name,
            resources=j.request.resources if j.request else cluster_pb2.ResourceSpecProto(),
            task_state_counts=task_state_counts,
            task_count=task_summary.task_count if task_summary else 0,
            completed_count=task_summary.completed_count if task_summary else 0,
            pending_reason=pending_reason,
        )
        if j.started_at:
            proto_job.started_at.CopyFrom(j.started_at.to_proto())
        if j.finished_at:
            proto_job.finished_at.CopyFrom(j.finished_at.to_proto())
        if j.submitted_at:
            proto_job.submitted_at.CopyFrom(j.submitted_at.to_proto())
        return proto_job

    def _jobs_to_protos(
        self,
        jobs: list[Job],
        task_summaries: dict[JobName, TaskJobSummary],
        autoscaler_pending_hints: dict[str, PendingHint],
    ) -> list[cluster_pb2.JobStatus]:
        return [self._job_to_proto(j, task_summaries.get(j.job_id), autoscaler_pending_hints) for j in jobs]

    def list_jobs(
        self,
        request: cluster_pb2.Controller.ListJobsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListJobsResponse:
        """List jobs with SQL-level filtering, sorting, and pagination."""
        name_filter = request.name_filter.lower() if request.name_filter else ""
        state_filter = request.state_filter.lower() if request.state_filter else ""
        autoscaler_pending_hints = self._get_autoscaler_pending_hints()

        sort_field = request.sort_field or cluster_pb2.Controller.JOB_SORT_FIELD_DATE
        sort_dir = request.sort_direction
        if sort_dir == cluster_pb2.Controller.SORT_DIRECTION_UNSPECIFIED:
            sort_dir = (
                cluster_pb2.Controller.SORT_DIRECTION_DESC
                if sort_field == cluster_pb2.Controller.JOB_SORT_FIELD_DATE
                else cluster_pb2.Controller.SORT_DIRECTION_ASC
            )
        reverse = sort_dir == cluster_pb2.Controller.SORT_DIRECTION_DESC

        offset = max(request.offset, 0)
        limit = min(request.limit, 500) if request.limit > 0 else 0

        state_filter_int: int | None = None
        if state_filter:
            for st in USER_JOB_STATES:
                if cluster_pb2.JobState.Name(st).replace("JOB_STATE_", "").lower() == state_filter:
                    state_filter_int = st
                    break
            if state_filter_int is None:
                return cluster_pb2.Controller.ListJobsResponse(jobs=[], total_count=0, has_more=False)

        jobs, total_count = _jobs_paginated(
            self._db,
            USER_JOB_STATES,
            state_filter_int=state_filter_int,
            name_filter=name_filter,
            sort_field=sort_field,
            descending=reverse,
            offset=offset,
            limit=limit,
        )
        # Also fetch descendants so the dashboard can build the job tree.
        descendants = _descendants_for_roots(self._db, [j.job_id.to_wire() for j in jobs])
        all_db_jobs = jobs + descendants
        task_summaries = _task_summaries_for_jobs(self._db, {j.job_id for j in all_db_jobs})
        all_jobs = self._jobs_to_protos(all_db_jobs, task_summaries, autoscaler_pending_hints)
        has_more = limit > 0 and offset + limit < total_count
        return cluster_pb2.Controller.ListJobsResponse(
            jobs=all_jobs,
            total_count=total_count,
            has_more=has_more,
        )

    # --- Task Management ---

    def get_task_status(
        self,
        request: cluster_pb2.Controller.GetTaskStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetTaskStatusResponse:
        """Get status of a specific task."""
        try:
            task_id = JobName.from_wire(request.task_id)
            task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        task = _read_task_with_attempts(self._db, task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {task_id} not found")
        # Look up worker address
        worker_address = ""
        if task.worker_id:
            worker = _read_worker(self._db, task.worker_id)
            if worker:
                worker_address = worker.address

        return cluster_pb2.Controller.GetTaskStatusResponse(
            task=task_to_proto(task, worker_address=worker_address),
        )

    def list_tasks(
        self,
        request: cluster_pb2.Controller.ListTasksRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListTasksResponse:
        """List all tasks, optionally filtered by job_id."""
        job_id = JobName.from_wire(request.job_id) if request.job_id else None
        tasks = _tasks_for_listing(self._db, job_id=job_id)
        worker_addr_by_id = _worker_addresses_for_tasks(self._db, tasks)

        task_statuses = []
        for task in tasks:
            proto_task_status = task_to_proto(task, worker_address=worker_addr_by_id.get(task.worker_id, ""))

            # Don't add scheduling diagnostics in list view - too expensive
            # Users should check job detail page for scheduling diagnostics
            if task.state == cluster_pb2.TASK_STATE_PENDING:
                proto_task_status.can_be_scheduled = task.can_be_scheduled()

            task_statuses.append(proto_task_status)

        return cluster_pb2.Controller.ListTasksResponse(tasks=task_statuses)

    # --- Worker Management ---

    def register(
        self,
        request: cluster_pb2.Controller.RegisterRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.RegisterResponse:
        """One-shot worker registration. Returns worker_id.

        Worker registers once, then waits for heartbeats from the controller.
        """
        if self._auth.provider is not None:
            authorize(AuthzAction.REGISTER_WORKER)

        if not request.worker_id:
            logger.error("Worker at %s registered without worker_id", request.address)
            return cluster_pb2.Controller.RegisterResponse(
                worker_id="",
                accepted=False,
            )
        worker_id = WorkerId(request.worker_id)

        self._transitions.register_or_refresh_worker(
            worker_id=worker_id,
            address=request.address,
            metadata=request.metadata,
            ts=Timestamp.now(),
        )
        self._controller.wake()

        logger.info("Worker registered: %s at %s", worker_id, request.address)
        return cluster_pb2.Controller.RegisterResponse(
            worker_id=str(worker_id),
            accepted=True,
        )

    def list_workers(
        self,
        request: cluster_pb2.Controller.ListWorkersRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListWorkersResponse:
        """List all workers with their running task counts."""
        workers = []
        worker_rows = _worker_roster(self._db)
        running_by_worker = running_tasks_by_worker(self._db, {worker.worker_id for worker in worker_rows})
        for worker in worker_rows:
            workers.append(
                cluster_pb2.Controller.WorkerHealthStatus(
                    worker_id=worker.worker_id,
                    healthy=worker.healthy,
                    consecutive_failures=worker.consecutive_failures,
                    last_heartbeat=worker.last_heartbeat.to_proto(),
                    running_job_ids=[task_id.to_wire() for task_id in running_by_worker.get(worker.worker_id, [])],
                    address=worker.address,
                    metadata=worker.metadata,
                    status_message=worker_status_message(worker),
                )
            )
        return cluster_pb2.Controller.ListWorkersResponse(workers=workers)

    # --- Endpoint Management ---

    def register_endpoint(
        self,
        request: cluster_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.RegisterEndpointResponse:
        """Register a service endpoint.

        The ``task_id`` field carries the calling task's wire-format task ID
        (e.g. ``/user/job/0``).  The endpoint is associated with the owning
        task so that retry cleanup removes stale endpoints from earlier
        attempts.

        Endpoints are registered regardless of job state, but only become
        visible to clients (via lookup/list) when the job is executing (not
        in a terminal state).
        """
        endpoint_id = request.endpoint_id or str(uuid.uuid4())

        task_id = JobName.from_wire(request.task_id)
        job_id, _task_index = task_id.require_task()

        job = _read_job(self._db, job_id)
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.task_id} not found")

        task = _read_task_with_attempts(self._db, task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")
        if request.attempt_id != task.current_attempt_id:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Stale attempt: task {request.task_id} attempt {request.attempt_id} "
                f"!= current {task.current_attempt_id}",
            )

        endpoint = Endpoint(
            endpoint_id=endpoint_id,
            name=request.name,
            address=request.address,
            job_id=job_id,
            metadata=dict(request.metadata),
            registered_at=Timestamp.now(),
        )

        self._transitions.add_endpoint(endpoint, task_id=task_id)

        return cluster_pb2.Controller.RegisterEndpointResponse(endpoint_id=endpoint_id)

    def unregister_endpoint(
        self,
        request: cluster_pb2.Controller.UnregisterEndpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        """Unregister a service endpoint. Idempotent."""
        self._transitions.remove_endpoint(request.endpoint_id)
        return cluster_pb2.Empty()

    def list_endpoints(
        self,
        request: cluster_pb2.Controller.ListEndpointsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListEndpointsResponse:
        """List endpoints by name prefix (or exact name when request.exact is set)."""
        endpoints = _query_endpoints(
            self._db,
            EndpointQuery(
                exact_name=request.prefix if request.exact else None,
                name_prefix=None if request.exact else request.prefix,
            ),
        )
        return cluster_pb2.Controller.ListEndpointsResponse(
            endpoints=[
                cluster_pb2.Controller.Endpoint(
                    endpoint_id=e.endpoint_id,
                    name=e.name,
                    address=e.address,
                    task_id=e.job_id.to_wire(),
                    metadata=e.metadata,
                )
                for e in endpoints
            ]
        )

    # --- Autoscaler ---

    def get_autoscaler_status(
        self,
        request: cluster_pb2.Controller.GetAutoscalerStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetAutoscalerStatusResponse:
        """Get current autoscaler status with worker info populated."""
        autoscaler = self._controller.autoscaler
        if not autoscaler:
            return cluster_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())

        status = autoscaler.get_status()

        # Build a map of worker_id -> (worker_id, healthy) for enriching VmInfo
        workers = _worker_roster(self._db)
        worker_id_to_info: dict[str, tuple[str, bool]] = {}
        for w in workers:
            worker_id_to_info[w.worker_id] = (w.worker_id, w.healthy)

        # Fetch running task counts per worker for dashboard display
        all_worker_ids = {WorkerId(w.worker_id) for w in workers}
        running_by_worker = running_tasks_by_worker(self._db, all_worker_ids) if all_worker_ids else {}

        # Enrich VmInfo objects with worker information by matching vm_id to worker_id
        for group in status.groups:
            for slice_info in group.slices:
                for vm in slice_info.vms:
                    worker_info = worker_id_to_info.get(vm.vm_id)
                    if worker_info:
                        vm.worker_id = worker_info[0]
                        vm.worker_healthy = worker_info[1]
                        wid = WorkerId(vm.worker_id)
                        vm.running_task_count = len(running_by_worker.get(wid, set()))

        return cluster_pb2.Controller.GetAutoscalerStatusResponse(status=status)

    # --- VM Logs ---

    # --- Task/Job Logs (batch fetching) ---

    def get_task_logs(
        self,
        request: cluster_pb2.Controller.GetTaskLogsRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.Controller.GetTaskLogsResponse:
        """Get logs for a task or all tasks in a job from the in-memory log store.

        Logs are forwarded from workers via heartbeat and accumulated in the
        controller's log store.  No remote storage I/O occurs.

        If request.id ends in a numeric index, treat as single task.
        Otherwise treat as job ID and fetch logs from all tasks.

        When attempt_id is specified (>= 0), fetches logs only from that specific attempt.
        """
        job_name = JobName.from_wire(request.id)
        max_lines = request.max_total_lines if request.max_total_lines > 0 else DEFAULT_MAX_TOTAL_LINES
        requested_attempt_id = request.attempt_id
        log_store = self._log_store

        # Collect child job statuses when requested (for streaming UI).
        child_job_statuses: list[cluster_pb2.JobStatus] = []
        if not job_name.is_task and request.include_children:
            jobs = _descendant_jobs(self._db, job_name)
            for job in jobs:
                child_status = cluster_pb2.JobStatus(
                    job_id=job.job_id.to_wire(),
                    state=job.state,
                    exit_code=job.exit_code or 0,
                    error=job.error or "",
                )
                if job.finished_at:
                    child_status.finished_at.CopyFrom(job.finished_at.to_proto())
                child_job_statuses.append(child_status)

        # Build the log key or prefix for the query.
        job_wire = job_name.to_wire()
        cursor = request.cursor
        substring_filter = request.substring

        if job_name.is_task and requested_attempt_id >= 0:
            # Exact key: single task + single attempt
            log_result = log_store.get_logs(
                task_log_key(TaskAttempt(task_id=job_name, attempt_id=requested_attempt_id)),
                since_ms=request.since_ms,
                cursor=cursor,
                substring_filter=substring_filter,
                max_lines=max_lines,
                tail=request.tail,
                min_level=request.min_level,
            )
            for entry in log_result.entries:
                entry.attempt_id = requested_attempt_id
        elif job_name.is_task:
            # All attempts of a single task: prefix "task_wire:"
            log_result = log_store.get_logs_by_prefix(
                job_wire + ":",
                cursor=cursor,
                since_ms=request.since_ms,
                substring_filter=substring_filter,
                max_lines=max_lines,
                tail=request.tail,
                min_level=request.min_level,
            )
        else:
            # All tasks in a job: prefix "job_wire/"
            # When include_children is False, use shallow=True to exclude
            # descendant job logs (only match direct task keys).
            log_result = log_store.get_logs_by_prefix(
                job_wire + "/",
                cursor=cursor,
                since_ms=request.since_ms,
                substring_filter=substring_filter,
                max_lines=max_lines,
                tail=request.tail,
                min_level=request.min_level,
                shallow=not request.include_children,
            )

        truncated = max_lines > 0 and len(log_result.entries) >= max_lines

        batch = cluster_pb2.Controller.TaskLogBatch(
            task_id=request.id,
            logs=log_result.entries,
        )

        return cluster_pb2.Controller.GetTaskLogsResponse(
            task_logs=[batch],
            truncated=truncated,
            child_job_statuses=child_job_statuses,
            cursor=log_result.cursor,
        )

    # --- Profiling ---

    def _resolve_worker_stub(self, worker_id_str: str) -> WorkerServiceClientSync:
        """Resolve a worker ID to a healthy stub, raising ConnectError on failure."""
        worker = self._transitions.get_worker(WorkerId(worker_id_str))
        if not worker:
            raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id_str} not found")
        if not worker.healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id_str} is unavailable")
        return self._controller.stub_factory.get_stub(worker.address)

    def profile_task(
        self,
        request: cluster_pb2.ProfileTaskRequest,
        ctx: RequestContext,
    ) -> cluster_pb2.ProfileTaskResponse:
        """Profile a running task or system process.

        Target routing:
        - /system/process: the controller process itself
        - /system/worker/<worker_id>: proxy to a specific worker (profiles the worker process)
        - /job/.../task/N: proxied to the task's worker
        """
        # Handle controller-local targets: profile the controller process itself
        if is_system_target(request.target):
            if not request.HasField("profile_type"):
                raise ConnectError(Code.INVALID_ARGUMENT, "profile_type is required")
            try:
                duration = request.duration_seconds or 10
                data = profile_local_process(duration, request.profile_type)
                return cluster_pb2.ProfileTaskResponse(profile_data=data)
            except Exception as e:
                return cluster_pb2.ProfileTaskResponse(error=str(e))

        # /system/worker/<worker_id>: proxy profile to the worker's own process
        worker_id = _parse_worker_target(request.target)
        if worker_id is not None:
            stub = self._resolve_worker_stub(worker_id)
            forwarded = cluster_pb2.ProfileTaskRequest(
                target="/system/process",
                duration_seconds=request.duration_seconds,
                profile_type=request.profile_type,
            )
            timeout_ms = (request.duration_seconds or 10) * 1000 + 30000
            resp = stub.profile_task(forwarded, timeout_ms=timeout_ms)
            return cluster_pb2.ProfileTaskResponse(
                profile_data=resp.profile_data,
                error=resp.error,
            )

        # Task target: parse optional :attempt_id, validate, proxy to worker
        try:
            target = parse_profile_target(request.target)
            target.task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        task = _read_task_with_attempts(self._db, target.task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.target} not found")

        task_worker_id = task.worker_id
        if not task_worker_id:
            raise ConnectError(Code.FAILED_PRECONDITION, f"Task {request.target} not assigned to a worker")

        worker = _read_worker(self._db, task_worker_id)
        if not worker or not worker.healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {task_worker_id} is unavailable")

        timeout_ms = (request.duration_seconds or 10) * 1000 + 30000
        stub = self._controller.stub_factory.get_stub(worker.address)
        resp = stub.profile_task(request, timeout_ms=timeout_ms)
        return cluster_pb2.ProfileTaskResponse(
            profile_data=resp.profile_data,
            error=resp.error,
        )

    # --- Transactions ---

    def get_transactions(
        self,
        request: cluster_pb2.Controller.GetTransactionsRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetTransactionsResponse:
        """Get recent controller actions for the dashboard action log."""
        limit = request.limit if request.limit > 0 else DEFAULT_TRANSACTION_LIMIT
        actions = []
        for action in _transaction_actions(self._db, limit=limit):
            details_str = json.dumps(action.details) if action.details else ""
            proto_action = cluster_pb2.Controller.TransactionAction(
                action=action.action,
                entity_id=action.entity_id,
                details=details_str,
            )
            proto_action.timestamp.CopyFrom(action.timestamp.to_proto())
            actions.append(proto_action)
        return cluster_pb2.Controller.GetTransactionsResponse(actions=actions)

    def list_users(
        self,
        request: cluster_pb2.Controller.ListUsersRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.ListUsersResponse:
        """Return live per-user aggregate counts for the dashboard."""
        del request, ctx
        users = sorted(
            _live_user_stats(self._db),
            key=lambda entry: (
                -_active_job_count(entry.job_state_counts),
                -(entry.task_state_counts.get(cluster_pb2.TASK_STATE_RUNNING, 0)),
                entry.user,
            ),
        )
        return cluster_pb2.Controller.ListUsersResponse(
            users=[
                cluster_pb2.Controller.UserSummary(
                    user=entry.user,
                    task_state_counts=_task_state_counts_for_summary(entry.task_state_counts),
                    job_state_counts=_job_state_counts_for_summary(entry.job_state_counts),
                )
                for entry in users
            ]
        )

    def fetch_logs(
        self,
        request: cluster_pb2.FetchLogsRequest,
        ctx: Any,
    ) -> cluster_pb2.FetchLogsResponse:
        """Fetch logs by source key with filtering and pagination.

        Source routing:
        - /system/process, /job/...: served from the controller's own LogStore
        - /system/worker/<worker_id>: proxied to the worker's FetchLogs(/system/process)
        """
        worker_id = _parse_worker_target(request.source)
        if worker_id is not None:
            stub = self._resolve_worker_stub(worker_id)
            forwarded = cluster_pb2.FetchLogsRequest(
                source="/system/process",
                since_ms=request.since_ms,
                cursor=request.cursor,
                substring=request.substring,
                max_lines=request.max_lines,
                tail=request.tail,
                min_level=request.min_level,
            )
            return stub.fetch_logs(forwarded, timeout_ms=10000)

        max_lines = request.max_lines if request.max_lines > 0 else 1000
        result = self._log_store.get_logs(
            request.source,
            since_ms=request.since_ms,
            cursor=request.cursor,
            substring_filter=request.substring,
            max_lines=max_lines,
            tail=request.tail,
            min_level=request.min_level,
        )
        return cluster_pb2.FetchLogsResponse(entries=result.entries, cursor=result.cursor)

    # --- Worker Detail ---

    def get_worker_status(
        self,
        request: cluster_pb2.Controller.GetWorkerStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.GetWorkerStatusResponse:
        """Return detail for a single worker, keyed by worker ID.

        Workers and VMs are independent: the worker detail page shows only
        worker state (health, tasks, logs). VM status lives on the Autoscaler
        tab.
        """
        if not request.id:
            raise ConnectError(Code.INVALID_ARGUMENT, "id is required")

        detail = _read_worker_detail(self._db, WorkerId(str(request.id)))
        if not detail:
            raise ConnectError(Code.NOT_FOUND, f"No worker found for '{request.id}'")

        worker = detail.worker
        worker_health = cluster_pb2.Controller.WorkerHealthStatus(
            worker_id=worker.worker_id,
            healthy=worker.healthy,
            consecutive_failures=worker.consecutive_failures,
            last_heartbeat=worker.last_heartbeat.to_proto(),
            running_job_ids=[tid.to_wire() for tid in detail.running_tasks],
            address=worker.address,
            metadata=worker.metadata,
            status_message=worker_status_message(worker),
        )

        # Fetch worker daemon logs via FetchLogs(/system/process) if worker is healthy
        worker_log_entries: list[cluster_pb2.FetchLogsResponse] = []
        if worker.healthy:
            try:
                stub = self._controller.stub_factory.get_stub(worker.address)
                fetch_resp = stub.fetch_logs(
                    cluster_pb2.FetchLogsRequest(
                        source=PROCESS_LOG_KEY,
                        max_lines=200,
                        tail=True,
                    ),
                    timeout_ms=10000,
                )
                worker_log_entries = list(fetch_resp.entries)
            except Exception:
                logger.debug("Failed to fetch worker logs for %s", request.id, exc_info=True)

        # Collect recent task history for this worker
        tasks = _tasks_for_worker(self._db, worker.worker_id, limit=50)
        recent_tasks = [task_to_proto(task) for task in tasks]

        resp = cluster_pb2.Controller.GetWorkerStatusResponse(
            worker_log_entries=worker_log_entries,
            recent_tasks=recent_tasks,
        )
        resp.worker.CopyFrom(worker_health)
        resource_history = detail.resource_history
        if resource_history:
            resp.current_resources.CopyFrom(resource_history[-1])
        for snapshot in resource_history:
            resp.resource_history.append(snapshot)
        return resp

    def begin_checkpoint(
        self,
        request: cluster_pb2.Controller.BeginCheckpointRequest,
        ctx: Any,
    ) -> cluster_pb2.Controller.BeginCheckpointResponse:
        path, result = self._controller.begin_checkpoint()
        resp = cluster_pb2.Controller.BeginCheckpointResponse(
            checkpoint_path=path,
            job_count=result.job_count,
            task_count=result.task_count,
            worker_count=result.worker_count,
        )
        resp.created_at.CopyFrom(result.created_at.to_proto())
        return resp

    def get_process_status(
        self,
        request: cluster_pb2.GetProcessStatusRequest,
        ctx: Any,
    ) -> cluster_pb2.GetProcessStatusResponse:
        """Return process info and recent logs.

        Target routing (same convention as ProfileTask/FetchLogs):
        - empty or /system/process: the controller process itself
        - /system/worker/<worker_id>: proxy to a specific worker
        """
        target = request.target
        if not target or target == "/system/process":
            return get_process_status(request, self._log_store, self._timer)

        # Parse /system/worker/<worker_id>
        worker_id = _parse_worker_target(target)
        if worker_id is None:
            raise ConnectError(Code.INVALID_ARGUMENT, f"Invalid target: {target}")

        worker = self._transitions.get_worker(WorkerId(worker_id))
        if not worker:
            raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id} not found")
        if not worker.healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")

        stub = self._controller.stub_factory.get_stub(worker.address)
        # Forward with target set to /system/process so the worker returns its own status
        forwarded = cluster_pb2.GetProcessStatusRequest(
            max_log_lines=request.max_log_lines,
            log_substring=request.log_substring,
            min_log_level=request.min_log_level,
            target="/system/process",
        )
        return stub.get_process_status(forwarded, timeout_ms=10000)

    # ── Auth RPCs ────────────────────────────────────────────────────────

    def get_auth_info(
        self,
        request: cluster_pb2.GetAuthInfoRequest,
        ctx: Any,
    ) -> cluster_pb2.GetAuthInfoResponse:
        return cluster_pb2.GetAuthInfoResponse(
            provider=self._auth.provider or "",
            gcp_project_id=self._auth.gcp_project_id or "",
        )

    def login(
        self,
        request: cluster_pb2.LoginRequest,
        ctx: Any,
    ) -> cluster_pb2.LoginResponse:
        if not self._auth.login_verifier:
            raise ConnectError(Code.UNIMPLEMENTED, "Login not available (no identity provider configured)")
        if not self._auth.jwt_manager:
            raise ConnectError(Code.INTERNAL, "JWT manager not configured")

        try:
            login_identity = self._auth.login_verifier.verify(request.identity_token)
        except ValueError as exc:
            logger.info("Login verification failed: %s", exc)
            raise ConnectError(Code.UNAUTHENTICATED, "Identity verification failed") from exc

        username = login_identity.user_id
        if username.startswith("system:"):
            raise ConnectError(Code.PERMISSION_DENIED, "Reserved username prefix")

        now = Timestamp.now()
        self._db.ensure_user(username, now)
        role = self._db.get_user_role(username)

        # Revoke old login keys and propagate to in-memory revocation set
        revoked_ids = revoke_login_keys_for_user(self._db, username, now)
        for jti in revoked_ids:
            self._auth.jwt_manager.revoke(jti)

        key_id = f"iris_k_{secrets.token_urlsafe(8)}"
        expires_at = Timestamp.from_ms(now.epoch_ms() + DEFAULT_JWT_TTL_SECONDS * 1000)
        create_api_key(
            self._db,
            key_id=key_id,
            key_hash=f"jwt:{key_id}",
            key_prefix="jwt",
            user_id=username,
            name=f"login-{now.epoch_ms()}",
            now=now,
            expires_at=expires_at,
        )

        jwt_token = self._auth.jwt_manager.create_token(username, role, key_id)
        logger.info(
            "Login: user=%s, role=%s, new_key=%s, revoked=%d old login keys", username, role, key_id, len(revoked_ids)
        )
        return cluster_pb2.LoginResponse(token=jwt_token, key_id=key_id, user_id=username)

    def create_api_key(
        self,
        request: cluster_pb2.CreateApiKeyRequest,
        ctx: Any,
    ) -> cluster_pb2.CreateApiKeyResponse:
        if not self._auth.jwt_manager:
            raise ConnectError(Code.INTERNAL, "JWT manager not configured")

        identity = require_identity()
        target_user = request.user_id or identity.user_id
        if target_user != identity.user_id:
            authorize(AuthzAction.MANAGE_OTHER_KEYS)

        now = Timestamp.now()
        self._db.ensure_user(target_user, now)
        role = self._db.get_user_role(target_user)

        key_id = f"iris_k_{secrets.token_urlsafe(8)}"
        ttl = request.ttl_ms // 1000 if request.ttl_ms > 0 else DEFAULT_JWT_TTL_SECONDS
        # Always persist the actual JWT expiry so the DB and token agree.
        expires_at = Timestamp.from_ms(now.epoch_ms() + ttl * 1000)

        create_api_key(
            self._db,
            key_id=key_id,
            key_hash=f"jwt:{key_id}",
            key_prefix="jwt",
            user_id=target_user,
            name=request.name or f"key-{now.epoch_ms()}",
            now=now,
            expires_at=expires_at,
        )

        jwt_token = self._auth.jwt_manager.create_token(target_user, role, key_id, ttl_seconds=ttl)
        # Use key_id prefix (not JWT prefix — all HS256 JWTs share the same header)
        return cluster_pb2.CreateApiKeyResponse(key_id=key_id, token=jwt_token, key_prefix=key_id[:8])

    def revoke_api_key(
        self,
        request: cluster_pb2.RevokeApiKeyRequest,
        ctx: Any,
    ) -> cluster_pb2.Empty:
        identity = require_identity()
        with self._db.read_snapshot() as q:
            key = q.one(API_KEYS, where=API_KEYS.c.key_id == request.key_id)
        if key is None:
            raise ConnectError(Code.NOT_FOUND, f"API key not found: {request.key_id}")
        if key.user_id != identity.user_id:
            authorize(AuthzAction.MANAGE_OTHER_KEYS)
        revoke_api_key(self._db, request.key_id, Timestamp.now())
        if self._auth.jwt_manager:
            self._auth.jwt_manager.revoke(request.key_id)
        return cluster_pb2.Empty()

    def list_api_keys(
        self,
        request: cluster_pb2.ListApiKeysRequest,
        ctx: Any,
    ) -> cluster_pb2.ListApiKeysResponse:
        identity = require_identity()
        target_user = request.user_id or identity.user_id
        if target_user != identity.user_id:
            authorize(AuthzAction.MANAGE_OTHER_KEYS)

        keys = list_api_keys(self._db, user_id=target_user if target_user else None)
        key_infos = []
        for k in keys:
            key_infos.append(
                cluster_pb2.ApiKeyInfo(
                    key_id=k.key_id,
                    key_prefix=k.key_prefix,
                    user_id=k.user_id,
                    name=k.name,
                    created_at_ms=k.created_at.epoch_ms(),
                    last_used_at_ms=k.last_used_at.epoch_ms() if k.last_used_at else 0,
                    expires_at_ms=k.expires_at.epoch_ms() if k.expires_at else 0,
                    revoked=k.revoked_at is not None,
                )
            )
        return cluster_pb2.ListApiKeysResponse(keys=key_infos)

    def get_current_user(
        self,
        request: cluster_pb2.GetCurrentUserRequest,
        ctx: Any,
    ) -> cluster_pb2.GetCurrentUserResponse:
        identity = get_verified_identity()
        if identity is None:
            return cluster_pb2.GetCurrentUserResponse(user_id="anonymous", role="")
        return cluster_pb2.GetCurrentUserResponse(
            user_id=identity.user_id,
            role=identity.role,
        )

    def execute_raw_query(
        self,
        request: query_pb2.RawQueryRequest,
        ctx: Any,
    ) -> query_pb2.RawQueryResponse:
        identity = require_identity()
        if identity.role != "admin":
            raise ConnectError(Code.PERMISSION_DENIED, "admin role required for raw queries")
        result = execute_raw_query(self._db, request.sql)
        return query_pb2.RawQueryResponse(
            columns=result.columns,
            rows=result.rows,
        )
