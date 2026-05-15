# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-side helpers: module-level functions taking ``tx: db.Tx`` as first argument.

All public functions return SA ``Row`` objects (or ``Sequence[Row]`` / dicts of
them).  Return shapes are documented per-function.

Areas covered (previously split across reads/<area>.py):
  budgets        — user budgets and roles
  dashboard      — job listing, task summaries, parent-child helpers
  jobs           — job/job_config lookups and CTEs
  reservations   — reservation_claims reads
  scheduler      — per-worker resource usage, running-tasks map
  task_attempts  — bulk attempt lookups
  tasks          — task detail and active-task projections
  workers        — worker detail, liveness helpers, schedulable workers
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rigging.timing import Timestamp
from sqlalchemy import Integer, bindparam, case, cast, func, literal_column, select, tuple_

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.codec import device_counts_from_json
from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import (
    USER_ROLE_DEFAULT,
    job_config_table,
    job_workdir_files_table,
    jobs_table,
    reservation_claims_table,
    task_attempts_table,
    tasks_table,
    user_budgets_table,
    users_table,
    workers_table,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2

# ---------------------------------------------------------------------------
# Query-result dataclasses (previously rows.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActiveTaskRow:
    """Task projection joined with ``jobs`` + ``job_config``.

    Shared by every cascade/scheduling query (``_kill_non_terminal_tasks``,
    ``_find_coscheduled_siblings``, ``cancel_job``, ``preempt_task``,
    ``cancel_tasks_for_timeout``, ``_remove_failed_worker``, poll paths).
    Callers that need resource info for RPC payloads use ``PendingDispatchRow``
    instead; ``ActiveTaskRow`` carries only the fields needed for state-machine
    and cascade logic.
    """

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    current_worker_id: WorkerId | None
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    is_reservation_holder: bool
    has_coscheduling: bool


@dataclass(frozen=True, slots=True)
class PendingDispatchRow:
    """Scheduling payload for a task being dispatched to a direct provider.

    Unlike :class:`ActiveTaskRow`, this row carries the full serialized
    runtime configuration (entrypoint / environment / ports / constraints
    / task_image / timeout) so the caller can assemble a
    ``RunTaskRequest``. Kept separate so other active-task queries don't
    pay for loading these JSON blobs. Used for both PENDING-promotion and
    ASSIGNED-redrive paths (see `reads.tasks.list_*_for_direct_provider`).
    """

    task_id: JobName
    job_id: JobName
    current_attempt_id: int
    num_tasks: int
    resources: "job_pb2.ResourceSpecProto"
    entrypoint_json: str
    environment_json: str
    bundle_id: str
    ports_json: list
    constraints_json: str | None
    task_image: str
    timeout_ms: int | None


@dataclass(frozen=True, slots=True)
class WorkerResourceUsage:
    """Aggregate resources currently held by unfinished worker-bound attempts.

    Computed by ``reads.resource_usage_by_worker``; the scheduler
    subtracts these from a worker's totals to derive available capacity.
    """

    cpu_millicores: int
    memory_bytes: int
    gpu_count: int
    tpu_count: int


@dataclass(frozen=True)
class TaskJobSummary:
    job_id: JobName
    task_count: int = 0
    completed_count: int = 0
    failure_count: int = 0
    preemption_count: int = 0
    task_state_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class UserBudget:
    user_id: str
    budget_limit: int
    max_band: int
    updated_at: Timestamp


@dataclass(frozen=True)
class ReservationClaim:
    """A claim binding a worker to a specific reservation entry.

    The controller assigns unclaimed workers to unsatisfied reservation entries
    each scheduling cycle. Once every entry for a job is claimed, the
    reservation gate opens and the job's tasks can be scheduled.
    """

    job_id: str
    entry_idx: int


# ---------------------------------------------------------------------------
# User budgets (previously reads/budgets.py)
# ---------------------------------------------------------------------------


def get_user_budget(tx: Tx, user_id: str) -> UserBudget | None:
    """Return :class:`UserBudget` for ``user_id``, or None."""
    row = tx.execute(
        select(
            user_budgets_table.c.user_id,
            user_budgets_table.c.budget_limit,
            user_budgets_table.c.max_band,
            user_budgets_table.c.updated_at_ms,
        ).where(user_budgets_table.c.user_id == bindparam("user_id")),
        {"user_id": user_id},
    ).first()
    if row is None:
        return None
    return UserBudget(
        user_id=str(row.user_id),
        budget_limit=int(row.budget_limit),
        max_band=int(row.max_band),
        updated_at=row.updated_at_ms,
    )


def list_user_budgets(tx: Tx) -> list[UserBudget]:
    """Return every :class:`UserBudget` row."""
    rows = tx.execute(
        select(
            user_budgets_table.c.user_id,
            user_budgets_table.c.budget_limit,
            user_budgets_table.c.max_band,
            user_budgets_table.c.updated_at_ms,
        )
    ).all()
    return [
        UserBudget(
            user_id=str(row.user_id),
            budget_limit=int(row.budget_limit),
            max_band=int(row.max_band),
            updated_at=row.updated_at_ms,
        )
        for row in rows
    ]


def get_all_user_budget_limits(tx: Tx) -> dict[str, int]:
    """Return ``{user_id: budget_limit}`` for every user with a budget row."""
    rows = tx.execute(
        select(
            user_budgets_table.c.user_id,
            user_budgets_table.c.budget_limit,
        )
    ).all()
    return {str(row.user_id): int(row.budget_limit) for row in rows}


def get_user_role(tx: Tx, user_id: str) -> str:
    """Return the user's role, or ``USER_ROLE_DEFAULT`` if not found."""
    row = tx.execute(
        select(users_table.c.role).where(users_table.c.user_id == bindparam("user_id")),
        {"user_id": user_id},
    ).first()
    return str(row.role) if row is not None else USER_ROLE_DEFAULT


# ---------------------------------------------------------------------------
# Dashboard composite reads (previously reads/dashboard.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sort-field whitelist
# ---------------------------------------------------------------------------

_STATE_SORT_ORDER: dict[int, int] = {
    job_pb2.JOB_STATE_RUNNING: 0,
    job_pb2.JOB_STATE_BUILDING: 1,
    job_pb2.JOB_STATE_PENDING: 2,
    job_pb2.JOB_STATE_SUCCEEDED: 3,
    job_pb2.JOB_STATE_FAILED: 4,
    job_pb2.JOB_STATE_KILLED: 5,
    job_pb2.JOB_STATE_WORKER_FAILED: 6,
    job_pb2.JOB_STATE_UNSCHEDULABLE: 7,
}

_AGG_FAILURES = func.coalesce(func.sum(tasks_table.c.failure_count), 0).label("agg_failures")
_AGG_PREEMPTIONS = func.coalesce(func.sum(tasks_table.c.preemption_count), 0).label("agg_preemptions")

_STATE_SORT_CASE = case(
    {state: order for state, order in _STATE_SORT_ORDER.items()},
    value=jobs_table.c.state,
    else_=99,
)

_SORT_FIELD_TO_COLUMN = {
    controller_pb2.Controller.JOB_SORT_FIELD_DATE: jobs_table.c.submitted_at_ms,
    controller_pb2.Controller.JOB_SORT_FIELD_NAME: jobs_table.c.name,
    controller_pb2.Controller.JOB_SORT_FIELD_STATE: _STATE_SORT_CASE,
    controller_pb2.Controller.JOB_SORT_FIELD_FAILURES: _AGG_FAILURES,
    controller_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS: _AGG_PREEMPTIONS,
}

_NEEDS_TASK_AGG: frozenset[int] = frozenset(
    {
        controller_pb2.Controller.JOB_SORT_FIELD_FAILURES,
        controller_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS,
    }
)

# ---------------------------------------------------------------------------
# Job listing projection (12-col subset of jobs + job_config)
# ---------------------------------------------------------------------------

_JOB_ROW_COLUMNS = (
    jobs_table.c.job_id,
    jobs_table.c.state,
    jobs_table.c.submitted_at_ms,
    jobs_table.c.started_at_ms,
    jobs_table.c.finished_at_ms,
    jobs_table.c.error,
    jobs_table.c.exit_code,
    jobs_table.c.name,
    jobs_table.c.depth,
    job_config_table.c.res_cpu_millicores,
    job_config_table.c.res_memory_bytes,
    job_config_table.c.res_disk_bytes,
    job_config_table.c.res_device_json,
)

# Task states considered "completed" for dashboard task-summary counts.
_COMPLETED_TASK_STATES = (job_pb2.TASK_STATE_SUCCEEDED, job_pb2.TASK_STATE_KILLED)


def _apply_job_filters(
    stmt,
    *,
    depth_filter: int | None,
    parent_filter: str | None,
    state_ids: tuple[int, ...],
    name_filter: str,
    job_id_prefix: str,
):
    """Apply the standard set of job WHERE predicates to ``stmt``.

    Works for both the main SELECT and the COUNT SELECT because neither
    requires knowledge of which columns are projected.
    """
    if depth_filter is not None:
        stmt = stmt.where(jobs_table.c.depth == depth_filter)
    if parent_filter is not None:
        stmt = stmt.where(jobs_table.c.parent_job_id == JobName.from_wire(parent_filter))
    stmt = stmt.where(jobs_table.c.state.in_(list(state_ids)))
    if name_filter:
        stmt = stmt.where(jobs_table.c.name.like(f"%{name_filter.lower()}%"))
    if job_id_prefix:
        escaped = job_id_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = stmt.where(jobs_table.c.job_id.like(f"{escaped}%", escape="\\"))
    return stmt


def list_jobs(
    tx: Tx,
    query: controller_pb2.Controller.JobQuery,
    state_ids: tuple[int, ...],
) -> tuple[list, int]:
    """Return ``(rows, total_count)`` for the given dashboard ``JobQuery``.

    ``state_ids`` is the pre-resolved state filter (always non-empty); the
    caller owns "unknown state -> empty page" handling so a bad filter never
    reaches SQL.
    """
    assert state_ids, "list_jobs requires at least one state id"

    scope = query.scope or controller_pb2.Controller.JOB_QUERY_SCOPE_ALL
    parent_filter = None
    depth_filter = None
    if scope == controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS:
        depth_filter = 1
    elif scope == controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN:
        if not query.parent_job_id:
            raise ValueError("query.parent_job_id is required for JOB_QUERY_SCOPE_CHILDREN")
        parent_filter = query.parent_job_id

    sort_field = query.sort_field or controller_pb2.Controller.JOB_SORT_FIELD_DATE
    sort_direction = query.sort_direction
    if sort_direction == controller_pb2.Controller.SORT_DIRECTION_UNSPECIFIED:
        sort_direction = (
            controller_pb2.Controller.SORT_DIRECTION_DESC
            if sort_field == controller_pb2.Controller.JOB_SORT_FIELD_DATE
            else controller_pb2.Controller.SORT_DIRECTION_ASC
        )
    descending = sort_direction == controller_pb2.Controller.SORT_DIRECTION_DESC
    order_column = _SORT_FIELD_TO_COLUMN.get(sort_field, jobs_table.c.submitted_at_ms)
    order_expr = order_column.desc() if descending else order_column.asc()

    needs_task_agg = sort_field in _NEEDS_TASK_AGG

    select_columns = _JOB_ROW_COLUMNS
    if needs_task_agg:
        select_columns = (*_JOB_ROW_COLUMNS, _AGG_FAILURES, _AGG_PREEMPTIONS)

    stmt = select(*select_columns).select_from(
        jobs_table.join(job_config_table, job_config_table.c.job_id == jobs_table.c.job_id)
    )
    if needs_task_agg:
        stmt = stmt.outerjoin(tasks_table, tasks_table.c.job_id == jobs_table.c.job_id)

    stmt = _apply_job_filters(
        stmt,
        depth_filter=depth_filter,
        parent_filter=parent_filter,
        state_ids=state_ids,
        name_filter=query.name_filter,
        job_id_prefix=query.job_id_prefix,
    )

    if needs_task_agg:
        stmt = stmt.group_by(jobs_table.c.job_id)

    stmt = stmt.order_by(order_expr)

    count_stmt = _apply_job_filters(
        select(func.count()).select_from(jobs_table),
        depth_filter=depth_filter,
        parent_filter=parent_filter,
        state_ids=state_ids,
        name_filter=query.name_filter,
        job_id_prefix=query.job_id_prefix,
    )

    offset = max(query.offset, 0)
    limit = max(query.limit, 0)
    if limit > 0:
        stmt = stmt.limit(limit).offset(offset)

    rows = tx.execute(stmt).all()
    total = int(tx.execute(count_stmt).scalar() or 0)
    return rows, total


def task_summaries_for_jobs(tx: Tx, job_ids: Iterable[JobName]) -> dict[JobName, TaskJobSummary]:
    """Return ``{job_id: TaskJobSummary}`` aggregating each job's tasks."""
    ids = list(job_ids)
    if not ids:
        return {}

    stmt = (
        select(
            tasks_table.c.job_id,
            tasks_table.c.state,
            func.count().label("cnt"),
            cast(func.coalesce(func.sum(tasks_table.c.failure_count), 0), Integer).label("total_failures"),
            cast(func.coalesce(func.sum(tasks_table.c.preemption_count), 0), Integer).label("total_preemptions"),
        )
        .where(tasks_table.c.job_id.in_(ids))
        .group_by(tasks_table.c.job_id, tasks_table.c.state)
    )

    rows = tx.execute(stmt).all()
    summaries: dict[JobName, TaskJobSummary] = {}
    for row in rows:
        jid = row.job_id
        prev = summaries.get(jid, TaskJobSummary(job_id=jid))
        cnt = int(row.cnt)
        state = int(row.state)
        summaries[jid] = TaskJobSummary(
            job_id=jid,
            task_count=prev.task_count + cnt,
            completed_count=prev.completed_count + (cnt if state in _COMPLETED_TASK_STATES else 0),
            failure_count=prev.failure_count + int(row.total_failures),
            preemption_count=prev.preemption_count + int(row.total_preemptions),
            task_state_counts={**prev.task_state_counts, state: cnt},
        )
    return summaries


def parent_ids_with_children(tx: Tx, job_ids: Iterable[JobName]) -> set[JobName]:
    """Return the subset of ``job_ids`` that currently have at least one direct child."""
    ids = list(job_ids)
    if not ids:
        return set()
    rows = tx.execute(
        select(jobs_table.c.parent_job_id)
        .where(jobs_table.c.parent_job_id.in_(bindparam("parent_ids", expanding=True)))
        .distinct(),
        {"parent_ids": ids},
    ).all()
    return {row.parent_job_id for row in rows if row.parent_job_id is not None}


# ---------------------------------------------------------------------------
# Job and job_config reads (previously reads/jobs.py)
# ---------------------------------------------------------------------------


def get_job_state(tx: Tx, job_id: JobName) -> int | None:
    """Return the ``state`` column for ``job_id``, or None if absent."""
    row = tx.execute(
        select(jobs_table.c.state).where(jobs_table.c.job_id == bindparam("job_id")),
        {"job_id": job_id},
    ).first()
    return int(row.state) if row is not None else None


def get_job_detail(tx: Tx, job_id: JobName):
    """Return SA Row for ``job_id`` (joined with job_config) or None."""
    return tx.execute(
        select(
            jobs_table.c.job_id,
            jobs_table.c.state,
            jobs_table.c.submitted_at_ms,
            jobs_table.c.root_submitted_at_ms,
            jobs_table.c.started_at_ms,
            jobs_table.c.finished_at_ms,
            jobs_table.c.scheduling_deadline_epoch_ms,
            jobs_table.c.error,
            jobs_table.c.exit_code,
            jobs_table.c.num_tasks,
            jobs_table.c.is_reservation_holder,
            jobs_table.c.has_reservation,
            jobs_table.c.name,
            jobs_table.c.depth,
            job_config_table.c.res_cpu_millicores,
            job_config_table.c.res_memory_bytes,
            job_config_table.c.res_disk_bytes,
            job_config_table.c.res_device_json,
            job_config_table.c.constraints_json,
            job_config_table.c.has_coscheduling,
            job_config_table.c.coscheduling_group_by,
            job_config_table.c.scheduling_timeout_ms,
            job_config_table.c.max_task_failures,
            job_config_table.c.entrypoint_json,
            job_config_table.c.environment_json,
            job_config_table.c.bundle_id,
            job_config_table.c.ports_json,
            job_config_table.c.max_retries_failure,
            job_config_table.c.max_retries_preemption,
            job_config_table.c.timeout_ms,
            job_config_table.c.preemption_policy,
            job_config_table.c.existing_job_policy,
            job_config_table.c.priority_band,
            job_config_table.c.task_image,
            job_config_table.c.submit_argv_json,
            job_config_table.c.reservation_json,
            job_config_table.c.fail_if_exists,
        )
        .select_from(jobs_table.join(job_config_table, jobs_table.c.job_id == job_config_table.c.job_id))
        .where(jobs_table.c.job_id == bindparam("job_id")),
        {"job_id": job_id},
    ).first()


def get_job_config(tx: Tx, job_id: JobName) -> dict | None:
    """Return the ``job_config`` row as ``{column_name: value}``, or None."""
    row = (
        tx.execute(
            select(job_config_table).where(job_config_table.c.job_id == bindparam("job_id")),
            {"job_id": job_id},
        )
        .mappings()
        .first()
    )
    return dict(row) if row is not None else None


def bulk_get_job_configs(tx: Tx, job_ids: Iterable[JobName]) -> dict[JobName, dict]:
    """Return ``{job_id: row_dict}`` for each ``job_config`` row found.

    Missing ids are absent from the result; callers treat absence as
    ``get_job_config -> None``.
    """
    ids = list(job_ids)
    if not ids:
        return {}
    rows = tx.execute(select(job_config_table).where(job_config_table.c.job_id.in_(ids))).mappings().all()
    return {row["job_id"]: dict(row) for row in rows}


def _priority_bands_stmt(ids: list[JobName]):
    """Build a recursive CTE that walks the parent_job_id chain for each id.

    Walks parent_job_id chain until a non-UNSPECIFIED priority_band is found.
    """
    j = jobs_table.alias("j")
    jc = job_config_table.alias("jc")
    base_q = (
        select(
            j.c.job_id.label("input_id"),
            j.c.job_id.label("current_id"),
            jc.c.priority_band.label("current_band"),
            j.c.parent_job_id.label("parent_id"),
        )
        .select_from(j.join(jc, jc.c.job_id == j.c.job_id))
        .where(j.c.job_id.in_(ids))
    )
    chain = base_q.cte("chain", recursive=True)
    j2 = jobs_table.alias("j2")
    jc2 = job_config_table.alias("jc2")
    recursive_q = (
        select(
            chain.c.input_id,
            j2.c.job_id.label("current_id"),
            jc2.c.priority_band.label("current_band"),
            j2.c.parent_job_id.label("parent_id"),
        )
        .select_from(chain.join(j2, j2.c.job_id == chain.c.parent_id).join(jc2, jc2.c.job_id == j2.c.job_id))
        .where(chain.c.current_band == 0)
    )
    full_chain = chain.union_all(recursive_q)
    return select(full_chain.c.input_id, full_chain.c.current_band).where(full_chain.c.current_band != 0)


def get_priority_bands(tx: Tx, job_ids: Iterable[JobName]) -> dict[JobName, int]:
    """Return ``{job_id: resolved priority_band}`` for the given jobs.

    Walks the parent_job_id chain for jobs with UNSPECIFIED (0) band until a
    non-zero band is found. Jobs whose entire ancestor chain is UNSPECIFIED
    fall back to ``PRIORITY_BAND_INTERACTIVE``.
    """
    ids = list(job_ids)
    if not ids:
        return {}
    rows = tx.execute(_priority_bands_stmt(ids)).all()
    resolved: dict[JobName, int] = {}
    for row in rows:
        resolved[row.input_id] = int(row.current_band)
    for jid in ids:
        resolved.setdefault(jid, int(job_pb2.PRIORITY_BAND_INTERACTIVE))
    return resolved


def get_workdir_files(tx: Tx, job_id: JobName) -> dict[str, bytes]:
    """Return ``{filename: data}`` for all workdir files attached to ``job_id``."""
    rows = tx.execute(
        select(job_workdir_files_table.c.filename, job_workdir_files_table.c.data).where(
            job_workdir_files_table.c.job_id == bindparam("job_id")
        ),
        {"job_id": job_id},
    ).all()
    return {str(row.filename): bytes(row.data) for row in rows}


def _has_unfinished_worker_attempts_stmt(job_id: JobName):
    base = select(jobs_table.c.job_id).where(jobs_table.c.job_id == job_id).cte("subtree", recursive=True)
    j = jobs_table.alias("j")
    recursive_q = select(j.c.job_id).join(base, j.c.parent_job_id == base.c.job_id)
    subtree = base.union_all(recursive_q)
    t = tasks_table.alias("t")
    ta = task_attempts_table.alias("ta")
    return (
        select(literal_column("1"))
        .select_from(t.join(ta, ta.c.task_id == t.c.task_id))
        .where(
            t.c.job_id.in_(select(subtree.c.job_id)),
            ta.c.worker_id.is_not(None),
            ta.c.finished_at_ms.is_(None),
        )
        .limit(1)
    )


def has_unfinished_worker_attempts(tx: Tx, job_id: JobName) -> bool:
    """Return True if any task under ``job_id`` (subtree) has a worker-bound unfinished attempt."""
    row = tx.execute(_has_unfinished_worker_attempts_stmt(job_id)).first()
    return row is not None


# ---------------------------------------------------------------------------
# Reservation reads (previously reads/reservations.py)
# ---------------------------------------------------------------------------


def list_claims(tx: Tx) -> dict[WorkerId, ReservationClaim]:
    """Return ``{WorkerId: ReservationClaim}`` for every reservation claim."""
    rows = tx.execute(
        select(
            reservation_claims_table.c.worker_id,
            reservation_claims_table.c.job_id,
            reservation_claims_table.c.entry_idx,
        )
    ).all()
    return {
        row.worker_id: ReservationClaim(
            job_id=str(row.job_id),
            entry_idx=int(row.entry_idx),
        )
        for row in rows
    }


# ---------------------------------------------------------------------------
# Scheduler-tick read helpers (previously reads/scheduler.py)
# ---------------------------------------------------------------------------


def resource_usage_by_worker(tx: Tx) -> dict[WorkerId, WorkerResourceUsage]:
    """Aggregate resources held by unfinished worker-bound attempts.

    Reservation-holder job rows are excluded (filtered in Python, not SQL).
    Two-step approach: the inline ``JOIN jobs ON is_reservation_holder = 0``
    is intentionally avoided because it drives SQLite from the ``jobs`` table
    (full scan ~24k rows on production) and pushes the read from ~3 ms to
    ~380 ms. The small set of reservation-holder job ids is fetched once and
    filtered in Python.
    """
    holder_rows = tx.execute(
        select(jobs_table.c.job_id).where(jobs_table.c.is_reservation_holder == True)  # noqa: E712
    ).all()
    holder_jobs: set[JobName] = {row.job_id for row in holder_rows}
    rows = tx.execute(
        select(
            task_attempts_table.c.worker_id,
            tasks_table.c.job_id,
            job_config_table.c.res_cpu_millicores,
            job_config_table.c.res_memory_bytes,
            job_config_table.c.res_device_json,
        )
        .select_from(
            task_attempts_table.join(tasks_table, tasks_table.c.task_id == task_attempts_table.c.task_id).join(
                job_config_table, job_config_table.c.job_id == tasks_table.c.job_id
            )
        )
        .where(
            task_attempts_table.c.worker_id.is_not(None),
            task_attempts_table.c.finished_at_ms.is_(None),
        )
    ).all()

    cpu: dict[WorkerId, int] = {}
    mem: dict[WorkerId, int] = {}
    gpu: dict[WorkerId, int] = {}
    tpu: dict[WorkerId, int] = {}
    for row in rows:
        if row.job_id in holder_jobs:
            continue
        wid: WorkerId = row.worker_id
        cpu[wid] = cpu.get(wid, 0) + int(row.res_cpu_millicores)
        mem[wid] = mem.get(wid, 0) + int(row.res_memory_bytes)
        counts = device_counts_from_json(row.res_device_json)
        gpu[wid] = gpu.get(wid, 0) + counts.gpu
        tpu[wid] = tpu.get(wid, 0) + counts.tpu
    return {
        wid: WorkerResourceUsage(
            cpu_millicores=cpu.get(wid, 0),
            memory_bytes=mem.get(wid, 0),
            gpu_count=gpu.get(wid, 0),
            tpu_count=tpu.get(wid, 0),
        )
        for wid in cpu.keys() | mem.keys() | gpu.keys() | tpu.keys()
    }


_SCHEDULER_ACTIVE_TASK_STATES = (
    int(job_pb2.TASK_STATE_ASSIGNED),
    int(job_pb2.TASK_STATE_BUILDING),
    int(job_pb2.TASK_STATE_RUNNING),
)


def running_tasks_by_worker(tx: Tx, worker_ids: set[WorkerId]) -> dict[WorkerId, set[JobName]]:
    """Return the set of currently-running task IDs for each worker."""
    if not worker_ids:
        return {}
    rows = tx.execute(
        select(tasks_table.c.current_worker_id.label("worker_id"), tasks_table.c.task_id).where(
            tasks_table.c.current_worker_id.in_(list(worker_ids)),
            tasks_table.c.state.in_(list(_SCHEDULER_ACTIVE_TASK_STATES)),
        ),
    ).all()
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        running[row.worker_id].add(row.task_id)
    return running


# ---------------------------------------------------------------------------
# Task-attempt reads (previously reads/task_attempts.py)
# ---------------------------------------------------------------------------

ATTEMPT_COLS = (
    task_attempts_table.c.task_id,
    task_attempts_table.c.attempt_id,
    task_attempts_table.c.worker_id,
    task_attempts_table.c.state,
    task_attempts_table.c.created_at_ms,
    task_attempts_table.c.started_at_ms,
    task_attempts_table.c.finished_at_ms,
    task_attempts_table.c.exit_code,
    task_attempts_table.c.error,
)

_BULK_GET_CHUNK_SIZE = 450


def bulk_get_attempts(
    tx: Tx,
    keys: Sequence[tuple[JobName, int]],
) -> dict[tuple[JobName, int], object]:
    """Return ``{(task_id, attempt_id): Row}`` for the requested keys.

    Drives lookups through the ``task_attempts`` PK. Missing keys are silently
    absent. Chunks at 450 keys per statement to keep the bound parameter list
    under SQLite's 999-parameter limit (2 binds per pair).
    """
    if not keys:
        return {}
    unique: list[tuple[JobName, int]] = list({k: None for k in keys}.keys())
    result: dict[tuple[JobName, int], object] = {}
    pair_cols = tuple_(task_attempts_table.c.task_id, task_attempts_table.c.attempt_id)
    for chunk_start in range(0, len(unique), _BULK_GET_CHUNK_SIZE):
        chunk = unique[chunk_start : chunk_start + _BULK_GET_CHUNK_SIZE]
        stmt = select(*ATTEMPT_COLS).where(pair_cols.in_(chunk))
        rows = tx.execute(stmt).all()
        for row in rows:
            result[(row.task_id, row.attempt_id)] = row
    return result


# ---------------------------------------------------------------------------
# Task reads (previously reads/tasks.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskScope:
    """Scope predicate for active-task queries.

    Exactly one field must be set. The store validates at the call boundary.
    ``null_worker=True`` matches rows where ``current_worker_id IS NULL``
    (direct-provider-promoted tasks).
    """

    job_id: JobName | None = None
    job_subtree: Sequence[JobName] | None = None
    worker_id: WorkerId | None = None
    worker_ids: Sequence[WorkerId] | None = None
    task_ids: Sequence[JobName] | None = None
    null_worker: bool = False


class TaskDetailRow(Protocol):
    """Shape of the SA Row returned by ``get_task_detail`` and values in ``bulk_get_task_detail``.

    Columns match ``TASK_DETAIL_COLS``.  Consumers in ``transitions.py`` use
    this Protocol as the value type of the task map.
    """

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at_ms: object  # Timestamp from TimestampMsType; typed as object to avoid a circular dep
    priority_band: int
    error: str | None
    exit_code: int | None
    started_at_ms: object | None
    finished_at_ms: object | None
    current_worker_id: str | None
    current_worker_address: str | None
    container_id: str | None


TASK_DETAIL_COLS = (
    tasks_table.c.task_id,
    tasks_table.c.job_id,
    tasks_table.c.state,
    tasks_table.c.current_attempt_id,
    tasks_table.c.failure_count,
    tasks_table.c.preemption_count,
    tasks_table.c.max_retries_failure,
    tasks_table.c.max_retries_preemption,
    tasks_table.c.submitted_at_ms,
    tasks_table.c.priority_band,
    tasks_table.c.error,
    tasks_table.c.exit_code,
    tasks_table.c.started_at_ms,
    tasks_table.c.finished_at_ms,
    tasks_table.c.current_worker_id,
    tasks_table.c.current_worker_address,
    tasks_table.c.container_id,
)


def get_task_detail(tx: Tx, task_id: JobName) -> TaskDetailRow | None:
    """Return SA Row for ``task_id`` or None."""
    return tx.execute(  # type: ignore[return-value]
        select(*TASK_DETAIL_COLS).where(tasks_table.c.task_id == bindparam("task_id")),
        {"task_id": task_id},
    ).first()


def bulk_get_task_detail(tx: Tx, task_ids: Iterable[JobName]) -> dict[JobName, TaskDetailRow]:
    """Return ``{task_id: TaskDetailRow}`` for all ``task_ids`` that exist. Missing keys are silently absent."""
    ids = list(task_ids)
    if not ids:
        return {}
    rows = tx.execute(
        select(*TASK_DETAIL_COLS).where(tasks_table.c.task_id.in_(bindparam("task_ids", expanding=True))),
        {"task_ids": ids},
    ).all()
    return {row.task_id: row for row in rows}  # type: ignore[return-value]


_ACTIVE_TASK_COLS = (
    tasks_table.c.task_id,
    tasks_table.c.job_id,
    tasks_table.c.state,
    tasks_table.c.current_attempt_id,
    tasks_table.c.current_worker_id,
    tasks_table.c.failure_count,
    tasks_table.c.preemption_count,
    tasks_table.c.max_retries_failure,
    tasks_table.c.max_retries_preemption,
    jobs_table.c.is_reservation_holder,
    job_config_table.c.has_coscheduling,
)

_ACTIVE_TASK_FROM = tasks_table.join(jobs_table, jobs_table.c.job_id == tasks_table.c.job_id).join(
    job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
)


def _row_to_active_task(row) -> ActiveTaskRow:
    return ActiveTaskRow(
        task_id=row.task_id,
        job_id=row.job_id,
        state=int(row.state),
        current_attempt_id=int(row.current_attempt_id),
        current_worker_id=row.current_worker_id,
        failure_count=int(row.failure_count),
        preemption_count=int(row.preemption_count),
        max_retries_failure=int(row.max_retries_failure),
        max_retries_preemption=int(row.max_retries_preemption),
        is_reservation_holder=bool(row.is_reservation_holder),
        has_coscheduling=bool(row.has_coscheduling),
    )


def list_active_tasks(
    tx: Tx,
    scope: TaskScope,
    *,
    states: Iterable[int],
    exclude_task_id: JobName | None = None,
    exclude_reservation_holders: bool = False,
    order_by_task_id: bool = False,
    limit: int | None = None,
) -> list[ActiveTaskRow]:
    """Return :class:`ActiveTaskRow` rows matching ``scope`` and ``states``.

    Exactly one scope field must be set. State filter is applied as an IN
    predicate.
    """
    scope_set = sum(
        1 for x in (scope.job_id, scope.job_subtree, scope.worker_id, scope.worker_ids, scope.task_ids) if x is not None
    ) + (1 if scope.null_worker else 0)
    if scope_set != 1:
        raise ValueError(
            "TaskScope must set exactly one of: job_id, job_subtree, worker_id, worker_ids, task_ids, null_worker"
        )

    states_tuple = tuple(states)
    if not states_tuple:
        return []

    stmt = select(*_ACTIVE_TASK_COLS).select_from(_ACTIVE_TASK_FROM)

    if scope.job_id is not None:
        stmt = stmt.where(tasks_table.c.job_id == scope.job_id)
    elif scope.job_subtree is not None:
        if not scope.job_subtree:
            return []
        stmt = stmt.where(tasks_table.c.job_id.in_(list(scope.job_subtree)))
    elif scope.worker_id is not None:
        stmt = stmt.where(tasks_table.c.current_worker_id == scope.worker_id)
    elif scope.worker_ids is not None:
        if not scope.worker_ids:
            return []
        stmt = stmt.where(tasks_table.c.current_worker_id.in_(list(scope.worker_ids)))
    elif scope.task_ids is not None:
        if not scope.task_ids:
            return []
        stmt = stmt.where(tasks_table.c.task_id.in_(list(scope.task_ids)))
    else:  # null_worker
        stmt = stmt.where(tasks_table.c.current_worker_id.is_(None))

    if exclude_task_id is not None:
        stmt = stmt.where(tasks_table.c.task_id != exclude_task_id)
    if exclude_reservation_holders:
        stmt = stmt.where(jobs_table.c.is_reservation_holder == False)  # noqa: E712

    stmt = stmt.where(tasks_table.c.state.in_(states_tuple))
    if order_by_task_id:
        stmt = stmt.order_by(tasks_table.c.task_id.asc())
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = tx.execute(stmt).all()
    return [_row_to_active_task(row) for row in rows]


# ---------------------------------------------------------------------------
# Worker reads (previously reads/workers.py)
# ---------------------------------------------------------------------------


class WorkerLivenessSource(Protocol):
    """Read-only view over the in-memory worker liveness tracker."""

    def all(self) -> dict[WorkerId, "_LivenessEntry"]: ...


class _LivenessEntry(Protocol):
    healthy: bool
    active: bool
    last_heartbeat_ms: int


class WorkerAttrsSource(Protocol):
    """Read-only view over the worker_attributes cache."""

    def all(self) -> dict[WorkerId, dict[str, AttributeValue]]: ...


WORKER_DETAIL_COLS = (
    workers_table.c.worker_id,
    workers_table.c.address,
    workers_table.c.total_cpu_millicores,
    workers_table.c.total_memory_bytes,
    workers_table.c.total_gpu_count,
    workers_table.c.total_tpu_count,
    workers_table.c.device_type,
    workers_table.c.device_variant,
    workers_table.c.md_hostname,
    workers_table.c.md_ip_address,
    workers_table.c.md_cpu_count,
    workers_table.c.md_memory_bytes,
    workers_table.c.md_disk_bytes,
    workers_table.c.md_tpu_name,
    workers_table.c.md_tpu_worker_hostnames,
    workers_table.c.md_tpu_worker_id,
    workers_table.c.md_tpu_chips_per_host_bounds,
    workers_table.c.md_gpu_count,
    workers_table.c.md_gpu_name,
    workers_table.c.md_gpu_memory_mb,
    workers_table.c.md_gce_instance_name,
    workers_table.c.md_gce_zone,
    workers_table.c.md_git_hash,
    workers_table.c.md_device_json,
)


def get_worker_detail(tx: Tx, worker_id: WorkerId):
    """Return SA Row for ``worker_id`` or None."""
    return tx.execute(
        select(*WORKER_DETAIL_COLS).where(workers_table.c.worker_id == bindparam("worker_id")),
        {"worker_id": worker_id},
    ).first()


def list_active_healthy_workers(tx: Tx, health: WorkerLivenessSource) -> dict[WorkerId, str]:
    """Return ``{worker_id: address}`` for all active+healthy workers.

    Fetches the full roster from SQL and filters by the in-memory health
    tracker in Python. The expanding ``IN (...)`` form pays SA Core overhead
    proportional to the IN list; since almost every persisted worker is
    healthy, fetching the whole roster and filtering by dict lookup is
    cheaper than the IN expansion.
    """
    liveness = health.all()
    live_ids = {wid for wid, ent in liveness.items() if ent.healthy and ent.active}
    if not live_ids:
        return {}
    rows = tx.execute(select(workers_table.c.worker_id, workers_table.c.address)).all()
    return {row.worker_id: str(row.address) for row in rows if row.worker_id in live_ids}


def filter_existing_workers(tx: Tx, worker_ids: Iterable[WorkerId]) -> set[str]:
    """Return the subset of ``worker_ids`` (as strings) that have a ``workers`` row."""
    ids = list(worker_ids)
    if not ids:
        return set()
    rows = tx.execute(
        select(workers_table.c.worker_id).where(workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))),
        {"worker_ids": ids},
    ).all()
    return {str(row.worker_id) for row in rows}


@dataclass(frozen=True, slots=True)
class SchedulableWorker:
    """Worker shape consumed by the scheduler.

    Field names mirror the :class:`scheduler.WorkerSnapshot` protocol so
    instances flow into ``Scheduler.create_scheduling_context`` without
    an adapter.
    """

    worker_id: WorkerId
    address: str
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    attributes: dict[str, AttributeValue]


def healthy_active_workers_with_attributes(
    tx: Tx,
    health: WorkerLivenessSource,
    attrs: WorkerAttrsSource,
) -> list[SchedulableWorker]:
    """Return healthy + active workers with their attributes hydrated.

    Reads the full worker roster and post-filters with the in-memory health
    tracker. See :func:`list_active_healthy_workers` for why we skip the
    SQL-side ``IN (...)`` filter.
    """
    liveness = health.all()
    healthy_active = {wid for wid, ent in liveness.items() if ent.healthy and ent.active}
    if not healthy_active:
        return []
    rows = tx.execute(
        select(
            workers_table.c.worker_id,
            workers_table.c.address,
            workers_table.c.total_cpu_millicores,
            workers_table.c.total_memory_bytes,
            workers_table.c.total_gpu_count,
            workers_table.c.total_tpu_count,
            workers_table.c.device_type,
            workers_table.c.device_variant,
        )
    ).all()
    attrs_by_worker = attrs.all()
    return [
        SchedulableWorker(
            worker_id=row.worker_id,
            address=str(row.address),
            total_cpu_millicores=int(row.total_cpu_millicores),
            total_memory_bytes=int(row.total_memory_bytes),
            total_gpu_count=int(row.total_gpu_count),
            total_tpu_count=int(row.total_tpu_count),
            device_type=str(row.device_type),
            device_variant=str(row.device_variant),
            attributes=attrs_by_worker.get(row.worker_id, {}),
        )
        for row in rows
        if row.worker_id in healthy_active
    ]
