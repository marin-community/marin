# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-side helpers: module-level functions taking ``tx: db.Tx`` as first argument.

Return shapes vary and are documented per-function: some return raw SA ``Row``
objects (or ``Sequence[Row]`` / dicts of them), others return typed dataclasses
(``ControlSnapshot``, ``RowCounts``, ``PendingTask``, ``ReconcileRow``, …) or
plain sets/dicts of ids.

Areas covered:
  budgets         — user budgets
  dashboard       — job listing, task summaries, parent-child helpers
  jobs            — job/job_config lookups and CTEs
  scheduling      — pending tasks, running tasks, per-user spend
  task_attempts   — bulk attempt lookups
  tasks           — task detail and active-task projections
  workers         — worker detail, liveness helpers, schedulable workers
  control-cycle   — the per-tick ControlSnapshot built via load_control_snapshot
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from rigging.timing import Timestamp
from sqlalchemy import Row, bindparam, case, exists, func, literal_column, select, tuple_

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.attempt_counts import (
    AttemptCounts,
    failure_count_expr,
    preemption_count_expr,
)
from iris.cluster.controller.codec import (
    device_counts_from_json,
    resource_spec_from_scalars,
)
from iris.cluster.controller.db import Tx
from iris.cluster.controller.reconcile.policy import NON_TERMINAL_TASK_STATES
from iris.cluster.controller.reconcile.worker import ReconcileRow
from iris.cluster.controller.schema import (
    endpoints_table,
    federated_jobs_table,
    federated_tasks_table,
    federation_changelog_table,
    federation_sync_state_table,
    hint_rare_state,
    job_config_table,
    job_workdir_files_table,
    jobs_table,
    local_tasks,
    slices_table,
    task_attempts_table,
    tasks_table,
    user_budgets_table,
    workers_table,
)
from iris.cluster.controller.task_state import (
    ACTIVE_TASK_STATES,
    ActiveTaskRow,
    RunningTaskEntry,
    TaskDetailRow,
    task_row_can_be_scheduled,
)
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.federation.store import FederationDirection, HandoffState
from iris.cluster.types import (
    LOCAL_CLUSTER,
    TERMINAL_JOB_STATES,
    AttemptUid,
    EndpointAccess,
    JobName,
    PendingTask,
    WorkerId,
    WorkerUsability,
)
from iris.rpc import controller_pb2, job_pb2

# ---------------------------------------------------------------------------
# Query-result dataclasses (previously rows.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PendingDispatchRow:
    """Scheduling payload for a task being dispatched to a direct provider.

    Unlike :class:`ActiveTaskRow`, this row carries the full serialized
    runtime configuration (entrypoint / environment / ports / constraints
    / task_image / timeout) so the caller can assemble a
    ``RunTaskRequest``. Kept separate so other active-task queries don't
    pay for loading these JSON blobs. Used for both PENDING-promotion and
    ASSIGNED-redrive paths (see ``dispatch.drain_for_dispatch``).
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
    # Coscheduling + priority drive Kueue gang admission on the direct path.
    has_coscheduling: bool
    coscheduling_group_by: str  # "" when not coscheduled
    # Effective band from tasks.priority_band (normalized to INTERACTIVE at
    # submit, overwritten with any over-budget demotion at assign time) — NOT
    # the immutable requested band in job_config. The Kueue WorkloadPriorityClass
    # must mirror the band Iris actually enforces, and tasks.priority_band is
    # never UNSPECIFIED(0), so the provider's plain .get() resolves correctly.
    priority_band: int  # job_pb2.PriorityBand, effective
    # Requested container security profile (job_config). UNSPECIFIED(0) resolves
    # to DEFAULT when the backend applies it.
    container_profile: int  # job_pb2.ContainerProfile


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

# Job-level failure/preemption totals for the sort, derived from task_attempts
# (the FROM in ``list_jobs`` outer-joins tasks → task_attempts when a count sort
# is requested, so these aggregate a job's attempts across all its tasks).
_AGG_FAILURES = failure_count_expr().label("agg_failures")
_AGG_PREEMPTIONS = preemption_count_expr().label("agg_preemptions")

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
    jobs_table.c.num_tasks,
    jobs_table.c.name,
    jobs_table.c.depth,
    job_config_table.c.res_cpu_millicores,
    job_config_table.c.res_memory_bytes,
    job_config_table.c.res_disk_bytes,
    job_config_table.c.res_device_json,
    jobs_table.c.backend_id,
    jobs_table.c.cluster,
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
    backend_id_filter: str = "",
    cluster_filter: str = "",
):
    """Apply the standard set of job WHERE predicates to ``stmt``.

    Works for both the main SELECT and the COUNT SELECT because neither
    requires knowledge of which columns are projected.
    """
    if depth_filter is not None:
        stmt = stmt.where(jobs_table.c.depth == depth_filter)
    if parent_filter is not None:
        stmt = stmt.where(jobs_table.c.parent_job_id == JobName.from_wire(parent_filter))
    stmt = stmt.where(jobs_table.c.state.in_(bindparam("job_state_ids", expanding=True)))
    if name_filter:
        stmt = stmt.where(jobs_table.c.name.like(f"%{name_filter.lower()}%"))
    if job_id_prefix:
        escaped = job_id_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = stmt.where(jobs_table.c.job_id.like(f"{escaped}%", escape="\\"))
    if backend_id_filter:
        stmt = stmt.where(jobs_table.c.backend_id == backend_id_filter)
    if cluster_filter:
        stmt = stmt.where(jobs_table.c.cluster == cluster_filter)
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
        stmt = stmt.outerjoin(tasks_table, tasks_table.c.job_id == jobs_table.c.job_id).outerjoin(
            task_attempts_table, task_attempts_table.c.task_id == tasks_table.c.task_id
        )

    stmt = _apply_job_filters(
        stmt,
        depth_filter=depth_filter,
        parent_filter=parent_filter,
        state_ids=state_ids,
        name_filter=query.name_filter,
        job_id_prefix=query.job_id_prefix,
        backend_id_filter=query.backend_id,
        cluster_filter=query.cluster,
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
        backend_id_filter=query.backend_id,
        cluster_filter=query.cluster,
    )

    offset = max(query.offset, 0)
    limit = max(query.limit, 0)
    if limit > 0:
        stmt = stmt.limit(limit).offset(offset)

    params = {"job_state_ids": list(state_ids)}
    rows = list(tx.execute(stmt, params).all())
    total = int(tx.execute(count_stmt, params).scalar() or 0)
    return rows, total


# Task-count / completed / state-histogram come from the ``tasks`` table; the
# failure/preemption totals are supplied separately, derived from attempts (via
# the AttemptCountsProjection cache) and merged in ``task_summaries_for_jobs``.
_TASK_SUMMARIES_FOR_JOBS_STMT = (
    select(
        tasks_table.c.job_id,
        tasks_table.c.state,
        func.count().label("cnt"),
    )
    .where(tasks_table.c.job_id.in_(bindparam("job_ids", expanding=True)))
    .group_by(tasks_table.c.job_id, tasks_table.c.state)
)


def task_summaries_for_jobs(
    tx: Tx,
    job_ids: Iterable[JobName],
    *,
    attempt_counts: Mapping[JobName, AttemptCounts] | None = None,
) -> dict[JobName, TaskJobSummary]:
    """Return ``{job_id: TaskJobSummary}`` aggregating each job's tasks.

    ``attempt_counts`` carries the per-job failure/preemption totals derived from
    ``task_attempts`` (typically from the cache); a job absent from it (or a
    ``None`` map) contributes zero for those two counters.
    """
    ids = list(job_ids)
    if not ids:
        return {}
    counts = attempt_counts or {}

    rows = tx.execute(_TASK_SUMMARIES_FOR_JOBS_STMT, {"job_ids": ids}).all()
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
            task_state_counts={**prev.task_state_counts, state: cnt},
        )
    return {
        jid: replace(
            summary,
            failure_count=counts.get(jid, AttemptCounts()).failure_count,
            preemption_count=counts.get(jid, AttemptCounts()).preemption_count,
        )
        for jid, summary in summaries.items()
    }


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


def find_prunable_job(tx: Tx, terminal_states: Iterable[int], before_ts: Timestamp) -> JobName | None:
    """Return one terminal *local* job finished before ``before_ts``, or None.

    Federated jobs (``cluster != 'local'``) are excluded: the parent mirrors the
    peer, so a peer-issued tombstone is the only path that deletes their rows.
    """
    row = tx.execute(
        select(jobs_table.c.job_id)
        .where(
            jobs_table.c.state.in_(bindparam("terminal_states", expanding=True)),
            jobs_table.c.finished_at_ms.is_not(None),
            jobs_table.c.finished_at_ms < bindparam("before_ts"),
            jobs_table.c.cluster == LOCAL_CLUSTER,
        )
        .limit(1),
        {"terminal_states": list(terminal_states), "before_ts": before_ts},
    ).first()
    return row.job_id if row is not None else None


def find_prunable_slice(tx: Tx, before_ms: int) -> str | None:
    """Return one orphaned slice older than ``before_ms``, or None.

    A ``slices`` row mirrors a set of live worker VMs. Once no ``workers`` row
    references the slice (``workers.slice_id``), the slice has no VMs behind it
    and the row is garbage — independent of whether its scale group still exists
    in config. ``workers.slice_id`` (written at registration) is the authoritative
    backing test, not ``slices.worker_ids``, which can go stale/empty while live
    workers still point at the slice.

    The ``created_at_ms`` floor protects a freshly-created slice whose VMs are
    still booting and have not registered their workers yet.
    """
    row = tx.execute(
        select(slices_table.c.slice_id)
        .where(
            slices_table.c.created_at_ms < bindparam("before_ms"),
            ~exists().where(workers_table.c.slice_id == slices_table.c.slice_id),
        )
        .limit(1),
        {"before_ms": before_ms},
    ).first()
    return row.slice_id if row is not None else None


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
            jobs_table.c.name,
            jobs_table.c.depth,
            jobs_table.c.parent_job_id,
            jobs_table.c.backend_id,
            jobs_table.c.cluster,
            jobs_table.c.submitting_user,
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
            job_config_table.c.container_profile,
            job_config_table.c.submit_argv_json,
            job_config_table.c.fail_if_exists,
        )
        .select_from(jobs_table.join(job_config_table, jobs_table.c.job_id == job_config_table.c.job_id))
        .where(jobs_table.c.job_id == bindparam("job_id")),
        {"job_id": job_id},
    ).first()


def bulk_get_job_configs(tx: Tx, job_ids: Iterable[JobName]) -> dict[JobName, dict]:
    """Return ``{job_id: config_dict}`` for all ``job_ids`` that have a config row.

    Missing keys are silently absent. Uses a single IN-list query.
    """
    ids = list(job_ids)
    if not ids:
        return {}
    rows = (
        tx.execute(
            select(job_config_table).where(job_config_table.c.job_id.in_(bindparam("job_ids", expanding=True))),
            {"job_ids": ids},
        )
        .mappings()
        .all()
    )
    return {row["job_id"]: dict(row) for row in rows}


def _build_priority_bands_stmt():
    """Build the recursive-CTE statement once with an expanding bindparam.

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
        .where(j.c.job_id.in_(bindparam("job_ids", expanding=True)))
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


_PRIORITY_BANDS_STMT = _build_priority_bands_stmt()


def get_priority_bands(tx: Tx, job_ids: Iterable[JobName]) -> dict[JobName, int]:
    """Return ``{job_id: resolved priority_band}`` for the given jobs.

    Walks the parent_job_id chain for jobs with UNSPECIFIED (0) band until a
    non-zero band is found. Jobs whose entire ancestor chain is UNSPECIFIED
    fall back to ``PRIORITY_BAND_INTERACTIVE``.
    """
    ids = list(job_ids)
    if not ids:
        return {}
    rows = tx.execute(_PRIORITY_BANDS_STMT, {"job_ids": ids}).all()
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
# Scheduler-tick read helpers (previously reads/scheduler.py)
# ---------------------------------------------------------------------------


def resource_usage_by_worker(tx: Tx) -> dict[WorkerId, WorkerResourceUsage]:
    """Aggregate resources held by unfinished worker-bound attempts, keyed by worker."""
    rows = tx.execute(
        select(
            task_attempts_table.c.worker_id,
            local_tasks.c.job_id,
            job_config_table.c.res_cpu_millicores,
            job_config_table.c.res_memory_bytes,
            job_config_table.c.res_device_json,
        )
        .select_from(
            task_attempts_table.join(local_tasks, local_tasks.c.task_id == task_attempts_table.c.task_id).join(
                job_config_table, job_config_table.c.job_id == local_tasks.c.job_id
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


_RUNNING_TASKS_BY_WORKER_STMT = select(tasks_table.c.current_worker_id.label("worker_id"), tasks_table.c.task_id).where(
    tasks_table.c.current_worker_id.in_(bindparam("worker_ids", expanding=True)),
    tasks_table.c.state.in_(bindparam("states", expanding=True)),
)


_BUILDING_COUNTS_STATES = (job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_ASSIGNED)


_BUILDING_COUNTS_STMT = (
    select(
        local_tasks.c.current_worker_id.label("worker_id"),
        func.count().label("cnt"),
    )
    .where(
        local_tasks.c.current_worker_id.in_(bindparam("worker_ids", expanding=True)),
        local_tasks.c.state.in_(bindparam("states", expanding=True)),
    )
    .group_by(local_tasks.c.current_worker_id)
)


def building_counts(tx: Tx, worker_ids: Sequence[WorkerId]) -> dict[WorkerId, int]:
    """Count BUILDING+ASSIGNED tasks per worker."""
    if not worker_ids:
        return {}
    rows = tx.execute(
        _BUILDING_COUNTS_STMT,
        {"worker_ids": list(worker_ids), "states": list(_BUILDING_COUNTS_STATES)},
    ).all()
    return {row.worker_id: int(row.cnt) for row in rows}


def running_tasks_by_worker(tx: Tx, worker_ids: set[WorkerId]) -> dict[WorkerId, set[JobName]]:
    """Return the set of currently-running task IDs for each worker."""
    if not worker_ids:
        return {}
    rows = tx.execute(
        _RUNNING_TASKS_BY_WORKER_STMT,
        {"worker_ids": list(worker_ids), "states": list(_SCHEDULER_ACTIVE_TASK_STATES)},
    ).all()
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        running[row.worker_id].add(row.task_id)
    return running


# ---------------------------------------------------------------------------
# Scheduling-policy reads (pending tasks, running tasks, budgets)
# ---------------------------------------------------------------------------

# Task columns needed to evaluate scheduling priority and retry budget. Shared by
# the scheduler's pending-task query (pending_tasks_with_jobs, which joins
# additional job/job_config columns) and the dashboard scheduler-summary path
# (service.GetSchedulerSummary), so the two stay aligned as priority columns evolve.
# Sourced from local_tasks: both are control-plane views that never act on
# peer-owned rows, so both build their FROM/WHERE on local_tasks.
PENDING_TASK_COLS = (
    local_tasks.c.task_id,
    local_tasks.c.job_id,
    local_tasks.c.backend_id,
    local_tasks.c.state,
    local_tasks.c.current_attempt_id,
    local_tasks.c.max_retries_failure,
    local_tasks.c.max_retries_preemption,
    local_tasks.c.submitted_at_ms,
    local_tasks.c.priority_band,
    local_tasks.c.priority_neg_depth,
    local_tasks.c.priority_root_submitted_ms,
    local_tasks.c.priority_insertion,
)


def _row_to_pending_task(row: Row) -> PendingTask:
    return PendingTask(
        task_id=row.task_id,
        job_id=row.job_id,
        backend_id=str(row.backend_id),
        state=int(row.state),
        current_attempt_id=int(row.current_attempt_id),
        max_retries_failure=int(row.max_retries_failure),
        max_retries_preemption=int(row.max_retries_preemption),
        submitted_at_ms=row.submitted_at_ms,
        priority_band=int(row.priority_band),
        priority_neg_depth=int(row.priority_neg_depth),
        priority_root_submitted_ms=int(row.priority_root_submitted_ms),
        priority_insertion=int(row.priority_insertion),
        job_state=int(row.job_state),
        scheduling_deadline_epoch_ms=row.scheduling_deadline_epoch_ms,
        scheduling_timeout_ms=row.scheduling_timeout_ms,
        has_coscheduling=bool(row.has_coscheduling),
        coscheduling_group_by=row.coscheduling_group_by,
        constraints_json=row.constraints_json,
        res_cpu_millicores=int(row.res_cpu_millicores),
        res_memory_bytes=int(row.res_memory_bytes),
        res_disk_bytes=int(row.res_disk_bytes),
        res_device_json=row.res_device_json,
    )


_PENDING_TASKS_STMT = (
    select(
        *PENDING_TASK_COLS,
        # job columns (label job_state to avoid clash with tasks.state)
        jobs_table.c.state.label("job_state"),
        jobs_table.c.scheduling_deadline_epoch_ms,
        # job_config columns
        job_config_table.c.scheduling_timeout_ms,
        job_config_table.c.has_coscheduling,
        job_config_table.c.coscheduling_group_by,
        job_config_table.c.constraints_json,
        job_config_table.c.res_cpu_millicores,
        job_config_table.c.res_memory_bytes,
        job_config_table.c.res_disk_bytes,
        job_config_table.c.res_device_json,
    )
    .select_from(
        local_tasks.join(jobs_table, jobs_table.c.job_id == local_tasks.c.job_id).join(
            job_config_table, job_config_table.c.job_id == local_tasks.c.job_id
        )
    )
    .where(local_tasks.c.state == bindparam("state"))
    .order_by(
        local_tasks.c.priority_neg_depth.asc(),
        local_tasks.c.priority_root_submitted_ms.asc(),
        local_tasks.c.submitted_at_ms.asc(),
        local_tasks.c.priority_insertion.asc(),
    )
)


def pending_tasks_with_jobs(tx: Tx) -> list[PendingTask]:
    """Return scheduling inputs for PENDING tasks, joining task + job + job_config in one query.

    Rows that cannot currently be scheduled (per :func:`task_row_can_be_scheduled`)
    are filtered out, so callers receive only actionable pending work.
    """
    rows = tx.execute(_PENDING_TASKS_STMT, {"state": job_pb2.TASK_STATE_PENDING}).all()
    pending_tasks = [_row_to_pending_task(row) for row in rows]
    return [task for task in pending_tasks if task_row_can_be_scheduled(task)]


_RUNNING_TASK_BAND_STMT = (
    select(
        local_tasks.c.task_id,
        local_tasks.c.priority_band,
        local_tasks.c.current_worker_id.label("worker_id"),
        job_config_table.c.res_cpu_millicores,
        job_config_table.c.res_memory_bytes,
        job_config_table.c.res_disk_bytes,
        job_config_table.c.res_device_json,
        job_config_table.c.has_coscheduling,
    )
    .select_from(local_tasks.join(job_config_table, local_tasks.c.job_id == job_config_table.c.job_id))
    .where(
        local_tasks.c.state == bindparam("state"),
        local_tasks.c.current_worker_id.is_not(None),
    )
)


def running_task_band_rows(tx: Tx) -> Sequence[Row]:
    """Return RUNNING worker-bound tasks with band, worker, and resource columns.

    The band is ``tasks.priority_band`` (stamped at assignment time), not the
    immutable requested band in ``job_config``.
    """
    return tx.execute(_RUNNING_TASK_BAND_STMT, {"state": job_pb2.TASK_STATE_RUNNING}).all()


_USER_SPEND_STMT = (
    select(
        local_tasks.c.job_id,
        job_config_table.c.res_cpu_millicores,
        job_config_table.c.res_memory_bytes,
        job_config_table.c.res_device_json,
        func.count().label("task_count"),
    )
    .select_from(local_tasks.join(job_config_table, job_config_table.c.job_id == local_tasks.c.job_id))
    .where(hint_rare_state(local_tasks.c.state.in_(bindparam("states", expanding=True))))
    .where(job_config_table.c.priority_band != job_pb2.PRIORITY_BAND_BATCH)
    .group_by(local_tasks.c.job_id)
)


def user_spend_rows(tx: Tx) -> Sequence[Row]:
    """Return per-job resource rows for active, non-BATCH tasks (budget spend basis).

    Each row carries ``(job_id, res_cpu_millicores, res_memory_bytes,
    res_device_json, task_count)``. ``job_config.priority_band`` (the user's
    requested band) drives the BATCH exclusion, not the stamped
    ``tasks.priority_band``, so scheduler-downgraded jobs still count.
    """
    return tx.execute(_USER_SPEND_STMT, {"states": list(ACTIVE_TASK_STATES)}).all()


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
    task_attempts_table.c.attempt_uid,
)

_BULK_GET_CHUNK_SIZE = 450

_BULK_GET_ATTEMPTS_STMT = select(*ATTEMPT_COLS).where(
    tuple_(task_attempts_table.c.task_id, task_attempts_table.c.attempt_id).in_(bindparam("keys", expanding=True))
)


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
    for chunk_start in range(0, len(unique), _BULK_GET_CHUNK_SIZE):
        chunk = unique[chunk_start : chunk_start + _BULK_GET_CHUNK_SIZE]
        rows = tx.execute(_BULK_GET_ATTEMPTS_STMT, {"keys": chunk}).all()
        for row in rows:
            result[(row.task_id, row.attempt_id)] = row
    return result


def attempt_counts_for_tasks(tx: Tx, task_ids: Sequence[JobName]) -> dict[JobName, AttemptCounts]:
    """Return ``{task_id: AttemptCounts}`` derived from each task's attempt rows.

    Tasks with no attempts are absent from the map (callers default to zero).
    """
    ids = list(task_ids)
    if not ids:
        return {}
    rows = tx.execute(
        select(
            task_attempts_table.c.task_id,
            failure_count_expr().label("failure_count"),
            preemption_count_expr().label("preemption_count"),
        )
        .where(task_attempts_table.c.task_id.in_(bindparam("task_ids", expanding=True)))
        .group_by(task_attempts_table.c.task_id),
        {"task_ids": ids},
    ).all()
    return {
        row.task_id: AttemptCounts(failure_count=int(row.failure_count), preemption_count=int(row.preemption_count))
        for row in rows
    }


def attempt_counts_for_jobs(tx: Tx, job_ids: Sequence[JobName]) -> dict[JobName, AttemptCounts]:
    """Return ``{job_id: AttemptCounts}`` summing every task's derived counts per job.

    Jobs with no attempt rows are absent from the map (callers default to zero).
    """
    ids = list(job_ids)
    if not ids:
        return {}
    rows = tx.execute(
        select(
            tasks_table.c.job_id,
            failure_count_expr().label("failure_count"),
            preemption_count_expr().label("preemption_count"),
        )
        .select_from(tasks_table.join(task_attempts_table, task_attempts_table.c.task_id == tasks_table.c.task_id))
        .where(tasks_table.c.job_id.in_(bindparam("job_ids", expanding=True)))
        .group_by(tasks_table.c.job_id),
        {"job_ids": ids},
    ).all()
    return {
        row.job_id: AttemptCounts(failure_count=int(row.failure_count), preemption_count=int(row.preemption_count))
        for row in rows
    }


def all_attempts_for_tasks(tx: Tx, task_ids: Sequence[JobName]) -> dict[JobName, tuple[object, ...]]:
    """Return ``{task_id: (attempt_row, ...)}`` with every attempt per task, ascending by attempt id.

    Returns the complete attempt history per task, with no per-task cap.
    """
    if not task_ids:
        return {}
    rows = tx.execute(
        select(*ATTEMPT_COLS)
        .where(task_attempts_table.c.task_id.in_(bindparam("task_ids", expanding=True)))
        .order_by(task_attempts_table.c.task_id.asc(), task_attempts_table.c.attempt_id.asc()),
        {"task_ids": list(task_ids)},
    ).all()
    grouped: dict[JobName, list[object]] = {}
    for row in rows:
        grouped.setdefault(row.task_id, []).append(row)
    return {task_id: tuple(attempts) for task_id, attempts in grouped.items()}


# Resolution joins ``local_tasks`` so it only ever resolves attempts of locally
# owned tasks. A federated task's mirrored attempts (task ``cluster`` set to a peer)
# are excluded, keeping this worker-routing / reconcile reader off the fold.
_RESOLVE_ATTEMPT_UIDS_STMT = (
    select(
        task_attempts_table.c.attempt_uid,
        task_attempts_table.c.task_id,
        task_attempts_table.c.attempt_id,
    )
    .select_from(task_attempts_table.join(local_tasks, local_tasks.c.task_id == task_attempts_table.c.task_id))
    .where(task_attempts_table.c.attempt_uid.in_(bindparam("uids", expanding=True)))
)


def resolve_attempt_uids(
    tx: Tx,
    uids: Sequence[AttemptUid],
) -> dict[AttemptUid, tuple[JobName, int]]:
    """Return ``{attempt_uid: (task_id, attempt_id)}`` for locally owned tasks' UIDs.

    Drives the worker-routing path: an ``AttemptObservation`` carrying an
    ``attempt_uid`` is resolved to its composite key through the
    ``idx_task_attempts_uid`` unique index. Restricted to ``local_tasks`` so a
    federated task's mirrored attempt never resolves. Missing UIDs are silently
    absent.
    """
    if not uids:
        return {}
    unique = list(dict.fromkeys(uids))
    result: dict[AttemptUid, tuple[JobName, int]] = {}
    for chunk_start in range(0, len(unique), _BULK_GET_CHUNK_SIZE):
        chunk = unique[chunk_start : chunk_start + _BULK_GET_CHUNK_SIZE]
        rows = tx.execute(_RESOLVE_ATTEMPT_UIDS_STMT, {"uids": chunk}).all()
        for row in rows:
            result[AttemptUid(row.attempt_uid)] = (row.task_id, row.attempt_id)
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


TASK_DETAIL_COLS = (
    tasks_table.c.task_id,
    tasks_table.c.job_id,
    tasks_table.c.state,
    tasks_table.c.current_attempt_id,
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
    tasks_table.c.backend_id,
    tasks_table.c.cluster,
)


def task_detail_query():
    """Select ``TASK_DETAIL_COLS`` plus the ``federated_tasks.peer_worker_label`` sidecar.

    The outer join yields ``peer_worker_label`` for a federated task (its display
    worker identity, since it has no local worker row) and NULL for a local task.
    """
    return select(*TASK_DETAIL_COLS, federated_tasks_table.c.peer_worker_label).select_from(
        tasks_table.outerjoin(federated_tasks_table, federated_tasks_table.c.task_id == tasks_table.c.task_id)
    )


def _task_detail_from_row(row, counts: AttemptCounts) -> TaskDetailRow:
    """Assemble a :class:`TaskDetailRow` from a ``task_detail_query`` row plus its
    derived attempt counts."""
    return TaskDetailRow(
        task_id=row.task_id,
        job_id=row.job_id,
        state=int(row.state),
        current_attempt_id=int(row.current_attempt_id),
        failure_count=counts.failure_count,
        preemption_count=counts.preemption_count,
        max_retries_failure=int(row.max_retries_failure),
        max_retries_preemption=int(row.max_retries_preemption),
        submitted_at_ms=row.submitted_at_ms,
        priority_band=int(row.priority_band),
        error=row.error,
        exit_code=row.exit_code,
        started_at_ms=row.started_at_ms,
        finished_at_ms=row.finished_at_ms,
        current_worker_id=row.current_worker_id,
        current_worker_address=row.current_worker_address,
        container_id=row.container_id,
        backend_id=str(row.backend_id or ""),
        cluster=str(row.cluster),
        peer_worker_label=row.peer_worker_label,
    )


def get_task_detail(tx: Tx, task_id: JobName) -> TaskDetailRow | None:
    """Return the :class:`TaskDetailRow` for ``task_id`` or None.

    Failure/preemption counts are derived from the task's attempt rows.
    """
    row = tx.execute(
        task_detail_query().where(tasks_table.c.task_id == bindparam("task_id")),
        {"task_id": task_id},
    ).first()
    if row is None:
        return None
    counts = attempt_counts_for_tasks(tx, [task_id]).get(task_id, AttemptCounts())
    return _task_detail_from_row(row, counts)


def bulk_get_task_detail(tx: Tx, task_ids: Iterable[JobName]) -> dict[JobName, TaskDetailRow]:
    """Return ``{task_id: TaskDetailRow}`` for all ``task_ids`` that exist. Missing keys are silently absent.

    Failure/preemption counts are derived from the tasks' attempt rows in one
    aggregate query.
    """
    ids = list(task_ids)
    if not ids:
        return {}
    rows = tx.execute(
        task_detail_query().where(tasks_table.c.task_id.in_(bindparam("task_ids", expanding=True))),
        {"task_ids": ids},
    ).all()
    counts = attempt_counts_for_tasks(tx, [row.task_id for row in rows])
    return {row.task_id: _task_detail_from_row(row, counts.get(row.task_id, AttemptCounts())) for row in rows}


_ACTIVE_TASK_COLS = (
    local_tasks.c.task_id,
    local_tasks.c.job_id,
    local_tasks.c.state,
    local_tasks.c.current_attempt_id,
    local_tasks.c.current_worker_id,
    local_tasks.c.max_retries_failure,
    local_tasks.c.max_retries_preemption,
    job_config_table.c.has_coscheduling,
)

_ACTIVE_TASK_FROM = local_tasks.join(job_config_table, job_config_table.c.job_id == local_tasks.c.job_id)


def _row_to_active_task(row, counts: AttemptCounts) -> ActiveTaskRow:
    return ActiveTaskRow(
        task_id=row.task_id,
        job_id=row.job_id,
        state=int(row.state),
        current_attempt_id=int(row.current_attempt_id),
        current_worker_id=row.current_worker_id,
        preemption_count=counts.preemption_count,
        max_retries_failure=int(row.max_retries_failure),
        max_retries_preemption=int(row.max_retries_preemption),
        has_coscheduling=bool(row.has_coscheduling),
    )


def list_active_tasks(
    tx: Tx,
    scope: TaskScope,
    *,
    states: Iterable[int],
    exclude_task_id: JobName | None = None,
    order_by_task_id: bool = False,
    limit: int | None = None,
    backend_id: str | None = None,
) -> list[ActiveTaskRow]:
    """Return :class:`ActiveTaskRow` rows matching ``scope`` and ``states``.

    Exactly one scope field must be set. State filter is applied as an IN
    predicate. ``backend_id`` narrows to one backend's tasks (omit for all).
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

    params: dict[str, object] = {}
    if scope.job_id is not None:
        stmt = stmt.where(local_tasks.c.job_id == scope.job_id)
    elif scope.job_subtree is not None:
        if not scope.job_subtree:
            return []
        stmt = stmt.where(local_tasks.c.job_id.in_(bindparam("scope_job_ids", expanding=True)))
        params["scope_job_ids"] = list(scope.job_subtree)
    elif scope.worker_id is not None:
        stmt = stmt.where(local_tasks.c.current_worker_id == scope.worker_id)
    elif scope.worker_ids is not None:
        if not scope.worker_ids:
            return []
        stmt = stmt.where(local_tasks.c.current_worker_id.in_(bindparam("scope_worker_ids", expanding=True)))
        params["scope_worker_ids"] = list(scope.worker_ids)
    elif scope.task_ids is not None:
        if not scope.task_ids:
            return []
        stmt = stmt.where(local_tasks.c.task_id.in_(bindparam("scope_task_ids", expanding=True)))
        params["scope_task_ids"] = list(scope.task_ids)
    else:  # null_worker
        stmt = stmt.where(local_tasks.c.current_worker_id.is_(None))

    if exclude_task_id is not None:
        stmt = stmt.where(local_tasks.c.task_id != exclude_task_id)

    if backend_id is not None:
        stmt = stmt.where(local_tasks.c.backend_id == backend_id)

    stmt = stmt.where(local_tasks.c.state.in_(bindparam("active_states", expanding=True)))
    params["active_states"] = list(states_tuple)
    if order_by_task_id:
        stmt = stmt.order_by(local_tasks.c.task_id.asc())
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = tx.execute(stmt, params).all()
    counts = attempt_counts_for_tasks(tx, [row.task_id for row in rows])
    return [_row_to_active_task(row, counts.get(row.task_id, AttemptCounts())) for row in rows]


def list_active_tasks_for_jobs(
    tx: Tx,
    job_ids: Iterable[JobName],
    *,
    states: Iterable[int],
) -> dict[JobName, tuple[ActiveTaskRow, ...]]:
    """Return ``{job_id: (ActiveTaskRow, ...)}`` for all ``job_ids`` in one query.

    Jobs with no matching tasks map to ``()``. Uses a single IN-list query
    instead of one query per job. State filter is applied as an IN predicate,
    identical to ``list_active_tasks``.
    """
    ids = list(job_ids)
    states_tuple = tuple(states)
    result: dict[JobName, tuple[ActiveTaskRow, ...]] = {jid: () for jid in ids}
    if not ids or not states_tuple:
        return result

    stmt = (
        select(*_ACTIVE_TASK_COLS)
        .select_from(_ACTIVE_TASK_FROM)
        .where(
            local_tasks.c.job_id.in_(bindparam("bulk_job_ids", expanding=True)),
            local_tasks.c.state.in_(bindparam("active_states", expanding=True)),
        )
    )
    rows = tx.execute(stmt, {"bulk_job_ids": ids, "active_states": list(states_tuple)}).all()
    counts = attempt_counts_for_tasks(tx, [row.task_id for row in rows])
    lists: dict[JobName, list[ActiveTaskRow]] = {}
    for row in rows:
        lists.setdefault(row.job_id, []).append(_row_to_active_task(row, counts.get(row.task_id, AttemptCounts())))
    for jid, task_rows in lists.items():
        result[jid] = tuple(task_rows)
    return result


def count_active_tasks_for_user(tx: Tx, user_id: str) -> int:
    """Return the number of non-terminal tasks across all jobs owned by ``user_id``."""
    return int(
        tx.execute(
            select(func.count())
            .select_from(local_tasks.join(jobs_table, jobs_table.c.job_id == local_tasks.c.job_id))
            .where(jobs_table.c.user_id == bindparam("user_id"))
            .where(local_tasks.c.state.in_(bindparam("states", expanding=True))),
            {"user_id": user_id, "states": list(NON_TERMINAL_TASK_STATES)},
        ).scalar()
        or 0
    )


# ---------------------------------------------------------------------------
# Worker reads (previously reads/workers.py)
# ---------------------------------------------------------------------------


class WorkerLivenessSource(Protocol):
    """Read-only view over the in-memory worker liveness tracker."""

    # Mapping (covariant value) rather than dict (invariant) so a concrete
    # tracker returning dict[WorkerId, WorkerLiveness] satisfies the protocol.
    def all(self) -> Mapping[WorkerId, "_LivenessEntry"]: ...


class _LivenessEntry(Protocol):
    @property
    def usability(self) -> WorkerUsability: ...


class WorkerAttrsSource(Protocol):
    """Read-only view over the worker_attributes cache."""

    def all(self) -> Mapping[WorkerId, dict[str, AttributeValue]]: ...


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
    workers_table.c.md_device_json,
    workers_table.c.md_provenance_json,
    workers_table.c.scale_group,
)


def get_worker_detail(tx: Tx, worker_id: WorkerId):
    """Return SA Row for ``worker_id`` or None."""
    return tx.execute(
        select(*WORKER_DETAIL_COLS).where(workers_table.c.worker_id == bindparam("worker_id")),
        {"worker_id": worker_id},
    ).first()


def _healthy_active_worker_ids(health: WorkerLivenessSource) -> set[WorkerId]:
    """Reconcile-target worker ids: every non-``DEAD`` worker (``HEALTHY | DEGRADED``).

    The reconcile pass must keep probing a worker that is mid-failure (so its
    liveness can recover or cross the teardown threshold), so it targets degraded
    workers too — only :data:`WorkerUsability.DEAD` drops out. Scheduling placement
    uses the stricter :func:`_schedulable_worker_ids`.

    Callers post-filter the ``workers`` table in Python against this set rather than
    pushing a SQL ``IN`` — cheaper, since almost every persisted worker is healthy.
    """
    return {wid for wid, ent in health.all().items() if ent.usability is not WorkerUsability.DEAD}


def _schedulable_worker_ids(health: WorkerLivenessSource) -> set[WorkerId]:
    """Scheduling-placement worker ids: only :data:`WorkerUsability.HEALTHY` workers.

    Excludes degraded (mid-failure) workers so new tasks stop landing on an
    unreachable worker immediately, rather than for the whole detection window
    before teardown. Reconcile keeps probing them (:func:`_healthy_active_worker_ids`);
    only placement is gated.
    """
    return {wid for wid, ent in health.all().items() if ent.usability is WorkerUsability.HEALTHY}


def list_active_healthy_workers(tx: Tx, health: WorkerLivenessSource) -> dict[WorkerId, str]:
    """Return ``{worker_id: address}`` for all active+healthy workers."""
    live_ids = _healthy_active_worker_ids(health)
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


def bulk_get_worker_addresses(tx: Tx, worker_ids: Iterable[WorkerId]) -> dict[WorkerId, str]:
    """Return ``{worker_id: address}`` for all ``worker_ids`` that have a ``workers`` row.

    Missing keys are silently absent. Uses a single IN-list query.
    """
    ids = list(worker_ids)
    if not ids:
        return {}
    rows = tx.execute(
        select(workers_table.c.worker_id, workers_table.c.address).where(
            workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))
        ),
        {"worker_ids": ids},
    ).all()
    return {row.worker_id: str(row.address) for row in rows}


def worker_ids_at_address(tx: Tx, address: str, *, exclude: WorkerId) -> list[WorkerId]:
    """Return worker ids whose row holds ``address``, excluding ``exclude``.

    Detects a recycled internal IP: when GCP reuses a deleted VM's IP for a new
    one, two rows end up sharing one ``address``. Passing the new registrant as
    ``exclude`` yields the stale prior owners.
    """
    rows = tx.execute(
        select(workers_table.c.worker_id).where(
            workers_table.c.address == address,
            workers_table.c.worker_id != exclude,
        )
    ).all()
    return [WorkerId(str(row.worker_id)) for row in rows]


@dataclass(frozen=True, slots=True)
class SchedulableWorker:
    """Worker shape consumed by the scheduler.

    Field names mirror the :class:`scheduler.WorkerSnapshot` protocol so
    instances flow through ``worker_snapshot_from_row`` into
    ``SchedulingContext`` without an adapter.
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
    """Return schedulable workers (healthy, active, not failing) with attributes.

    Reads the full worker roster and post-filters with the in-memory health
    tracker via :func:`_schedulable_worker_ids` — a worker mid-failure is still
    reconciled but is not a placement target. See :func:`_healthy_active_worker_ids`
    for why we skip the SQL-side ``IN (...)`` filter.
    """
    healthy_active = _schedulable_worker_ids(health)
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


# ---------------------------------------------------------------------------
# Direct-provider dispatch helpers (used by dispatch.py)
# ---------------------------------------------------------------------------

# Columns selected for every pending-dispatch / redrive query.  Covers all
# fields required to build a RunTaskRequest without a second DB round-trip.
PENDING_DISPATCH_COLS = (
    local_tasks.c.task_id,
    local_tasks.c.job_id,
    local_tasks.c.current_attempt_id,
    jobs_table.c.num_tasks,
    job_config_table.c.res_cpu_millicores,
    job_config_table.c.res_memory_bytes,
    job_config_table.c.res_disk_bytes,
    job_config_table.c.res_device_json,
    job_config_table.c.entrypoint_json,
    job_config_table.c.environment_json,
    job_config_table.c.bundle_id,
    job_config_table.c.ports_json,
    job_config_table.c.constraints_json,
    job_config_table.c.task_image,
    job_config_table.c.timeout_ms,
    job_config_table.c.has_coscheduling,
    job_config_table.c.coscheduling_group_by,
    # Effective band (tasks), not the immutable requested band (job_config):
    # see PendingDispatchRow.priority_band.
    local_tasks.c.priority_band,
    job_config_table.c.container_profile,
)


def pending_dispatch_row(r) -> PendingDispatchRow:
    """Decode a raw SA result row into a :class:`PendingDispatchRow`."""
    _tms = r.timeout_ms
    return PendingDispatchRow(
        task_id=r.task_id,
        job_id=r.job_id,
        current_attempt_id=int(r.current_attempt_id),
        num_tasks=int(r.num_tasks),
        resources=resource_spec_from_scalars(
            int(r.res_cpu_millicores),
            int(r.res_memory_bytes),
            int(r.res_disk_bytes),
            r.res_device_json,
        ),
        entrypoint_json=str(r.entrypoint_json),
        environment_json=str(r.environment_json),
        bundle_id=str(r.bundle_id),
        ports_json=r.ports_json,
        constraints_json=r.constraints_json,
        task_image=str(r.task_image),
        timeout_ms=int(_tms) if _tms is not None else None,
        has_coscheduling=bool(r.has_coscheduling),
        coscheduling_group_by=str(r.coscheduling_group_by),
        priority_band=int(r.priority_band),
        container_profile=int(r.container_profile),
    )


# ---------------------------------------------------------------------------
# Control-cycle snapshot (the reconcile control tick reads through here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowCounts:
    """Aggregate registry sizes, used for checkpoint metadata."""

    jobs: int
    tasks: int
    workers: int


def row_counts(tx: Tx) -> RowCounts:
    """Return total row counts for the jobs, tasks, and workers tables."""
    return RowCounts(
        jobs=int(tx.execute(select(func.count()).select_from(jobs_table)).scalar() or 0),
        tasks=int(tx.execute(select(func.count()).select_from(tasks_table)).scalar() or 0),
        workers=int(tx.execute(select(func.count()).select_from(workers_table)).scalar() or 0),
    )


def worker_scale_groups(tx: Tx) -> dict[WorkerId, str]:
    """Return ``{worker_id: scale_group}`` for every persisted worker.

    The controller maps each worker's scale group to its owning backend to
    partition the per-tick snapshot. Workers with no scale group map to ``""``.
    """
    rows = tx.execute(select(workers_table.c.worker_id, workers_table.c.scale_group)).all()
    return {WorkerId(str(row.worker_id)): str(row.scale_group or "") for row in rows}


def owned_worker_ids(tx: Tx, owns_scale_group: Callable[[str], bool]) -> set[WorkerId]:
    """The workers whose scale group ``owns_scale_group`` claims, in the read ``tx``."""
    return {wid for wid, scale_group in worker_scale_groups(tx).items() if owns_scale_group(scale_group)}


_EXECUTING_TASK_STATES = (int(job_pb2.TASK_STATE_BUILDING), int(job_pb2.TASK_STATE_RUNNING))

_EXECUTION_TIMEOUT_STMT = (
    select(
        local_tasks.c.task_id,
        task_attempts_table.c.started_at_ms,
        job_config_table.c.timeout_ms,
    )
    .select_from(
        local_tasks.join(job_config_table, job_config_table.c.job_id == local_tasks.c.job_id).join(
            task_attempts_table,
            (task_attempts_table.c.task_id == local_tasks.c.task_id)
            & (task_attempts_table.c.attempt_id == local_tasks.c.current_attempt_id),
        )
    )
    .where(
        hint_rare_state(local_tasks.c.state.in_(bindparam("executing_states", expanding=True))),
        job_config_table.c.timeout_ms.is_not(None),
        job_config_table.c.timeout_ms > 0,
        task_attempts_table.c.started_at_ms.is_not(None),
    )
)


def scan_execution_timeout_rows(tx: Tx) -> Sequence[Row]:
    """Return ``(task_id, started_at_ms, timeout_ms)`` for executing tasks that declare a timeout.

    Whether a task has actually exceeded its deadline is left to the caller,
    which holds the tick clock; this only returns the candidates.
    """
    return tx.execute(_EXECUTION_TIMEOUT_STMT, {"executing_states": list(_EXECUTING_TASK_STATES)}).all()


_RECONCILE_ROWS_STMT = (
    select(
        task_attempts_table.c.worker_id,
        local_tasks.c.task_id,
        task_attempts_table.c.attempt_id,
        local_tasks.c.state.label("task_state"),
        task_attempts_table.c.state.label("attempt_state"),
        local_tasks.c.job_id,
        task_attempts_table.c.attempt_uid,
    )
    .select_from(
        task_attempts_table.join(
            local_tasks,
            (local_tasks.c.task_id == task_attempts_table.c.task_id)
            & (local_tasks.c.current_attempt_id == task_attempts_table.c.attempt_id),
        )
    )
    .where(
        task_attempts_table.c.worker_id.is_not(None),
        task_attempts_table.c.finished_at_ms.is_(None),
    )
)


def load_reconcile_rows(tx: Tx, worker_ids: Iterable[WorkerId]) -> list[ReconcileRow]:
    """Return the live ``(task, attempt, worker)`` tuples driving one reconcile tick.

    Workers not in ``worker_ids`` are filtered in Python so the partial index
    ``idx_task_attempts_live_workerbound`` stays active rather than falling back
    to a scan on a long IN list. Task state is deliberately NOT filtered: active
    rows (ASSIGNED/BUILDING/RUNNING) drive normal reconciliation; rows whose task
    has already moved to a terminal state but whose attempt is still worker-bound
    (worker_id set, finished_at_ms NULL) are stranded attempts whose terminal
    Reconcile observation was lost. Including them gives the worker a second
    chance to report -- with the real terminal status or via the MISSING
    synthesis in ``handle_reconcile`` -- so the reconcile path can stamp
    finished_at_ms. Without this, a single lost RPC strands the attempt forever.
    """
    target_ids = set(worker_ids)
    if not target_ids:
        return []
    rows = tx.execute(_RECONCILE_ROWS_STMT).all()
    return [
        ReconcileRow(
            worker_id=row.worker_id,
            task_id=row.task_id,
            attempt_id=int(row.attempt_id),
            task_state=int(row.task_state),
            attempt_state=int(row.attempt_state),
            job_id=row.job_id,
            attempt_uid=AttemptUid(str(row.attempt_uid)),
        )
        for row in rows
        if row.worker_id in target_ids
    ]


@dataclass(frozen=True, slots=True)
class ControlSnapshot:
    """The DB-less per-tick input the controller hands to a :class:`TaskBackend`.

    One snapshot type feeds all three uniform backend methods; each control loop
    populates the section its phase needs and leaves the rest empty (the
    ``scan_timeouts`` flag is the pattern). The backend reads its section and
    never touches the database.

    * ``worker_addresses`` — ``{worker_id: address}`` for active + healthy workers.
    * ``reconcile_rows`` — live ``(task, attempt, worker)`` tuples across those
      workers (see :func:`load_reconcile_rows`).
    * ``timeout_rows`` — executing tasks past their declared deadline; empty
      unless the caller requested the timeout sweep this tick.
    * ``job_specs`` — per-job ``RunTaskRequest`` templates for ASSIGNED reconcile
      rows, so a worker-daemon backend can build its per-worker reconcile plans.
    * ``tasks_to_run`` / ``running_tasks`` — the dispatch drain for a cluster
      backend that owns placement (built only when that backend reconciles).

    Worker liveness is never persisted and never read off the snapshot: the
    controller owns its in-memory :class:`WorkerHealthTracker` directly and folds
    backend-observed health events into it. The tracker is passed to
    :func:`load_control_snapshot` only to select the live worker set.
    """

    worker_addresses: dict[WorkerId, str]
    reconcile_rows: list[ReconcileRow]
    timeout_rows: Sequence[Row]
    job_specs: dict[JobName, job_pb2.RunTaskRequest] = field(default_factory=dict)
    tasks_to_run: list[job_pb2.RunTaskRequest] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)


def load_control_snapshot(
    tx: Tx,
    health: WorkerHealthTracker,
    *,
    scan_timeouts: bool,
) -> ControlSnapshot:
    """Build the per-cycle :class:`ControlSnapshot` in one read transaction.

    ``health`` selects the live worker set (see :func:`list_active_healthy_workers`);
    it is not stored on the snapshot. ``scan_timeouts`` includes the
    execution-timeout rows in the same snapshot.
    """
    worker_addresses = list_active_healthy_workers(tx, health)
    reconcile_rows = load_reconcile_rows(tx, worker_addresses.keys()) if worker_addresses else []
    timeout_rows = scan_execution_timeout_rows(tx) if scan_timeouts else []
    return ControlSnapshot(
        worker_addresses=worker_addresses,
        reconcile_rows=reconcile_rows,
        timeout_rows=timeout_rows,
    )


# ---------------------------------------------------------------------------
# Federation (parent side: handles; peer side: the change-log sync page)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FederatedHandle:
    """A parent-side SENT federated job handle (``federated_jobs`` ⋈ ``jobs``)."""

    job_id: JobName  # this cluster's local (root) job id; the peer runs the same id
    peer_id: str
    owner_principal: str
    handoff_state: int
    cancel_intent_version: int


_SENT_HANDLE_COLUMNS = (
    federated_jobs_table.c.job_id,
    federated_jobs_table.c.peer_id,
    federated_jobs_table.c.owner_principal,
    federated_jobs_table.c.handoff_state,
    federated_jobs_table.c.cancel_intent_version,
)


def _sent_handle(row) -> FederatedHandle:
    return FederatedHandle(
        job_id=row.job_id,
        peer_id=row.peer_id,
        owner_principal=row.owner_principal,
        handoff_state=int(row.handoff_state),
        cancel_intent_version=int(row.cancel_intent_version),
    )


def federated_handle(tx: Tx, job_id: JobName) -> FederatedHandle | None:
    """The SENT federated handle for ``job_id``, or ``None`` if it is not one.

    Restricted to SENT rows: a RECEIVED row (this cluster is the peer) runs as an
    ordinary local job, so it is not a handle a parent-side cancel routes through.
    """
    row = tx.execute(
        select(*_SENT_HANDLE_COLUMNS).where(
            federated_jobs_table.c.job_id == job_id,
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
        )
    ).first()
    return _sent_handle(row) if row is not None else None


def _uncancelled_sent_handles(tx: Tx, handoff_state: HandoffState) -> list[FederatedHandle]:
    """SENT handles in ``handoff_state`` whose cancel intent is unset.

    The shared read behind the delivery and promotion queues: a cancelled handle
    (``cancel_intent_version > 0``) is excluded so neither the retry loop nor the
    tick's promotion ever acts on a job the user already asked to cancel.
    """
    rows = tx.execute(
        select(*_SENT_HANDLE_COLUMNS).where(
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
            federated_jobs_table.c.handoff_state == bindparam("handoff_state"),
            federated_jobs_table.c.cancel_intent_version == 0,
        ),
        {"handoff_state": int(handoff_state)},
    ).all()
    return [_sent_handle(r) for r in rows]


def pending_handoff_handles(tx: Tx) -> list[FederatedHandle]:
    """SENT handles still awaiting delivery to the peer: ``PENDING_HANDOFF``, uncancelled.

    Read each sync pass by the retry loop, which (re-)delivers them to the peer.
    """
    return _uncancelled_sent_handles(tx, HandoffState.PENDING_HANDOFF)


def queued_handoff_handles(tx: Tx) -> list[FederatedHandle]:
    """SENT handles queued on the parent awaiting a peer with free capacity: uncancelled ``QUEUED_HANDOFF``.

    Read each control tick by the federation pass, which promotes any it can place.
    """
    return _uncancelled_sent_handles(tx, HandoffState.QUEUED_HANDOFF)


def expired_queued_handoffs(tx: Tx, now_ms: int) -> list[JobName]:
    """Queued federated jobs whose scheduling deadline has passed, to fail this tick.

    A queued handoff owns no task rows, so the task-level scheduling-timeout scan never
    sees it; without this a job with a ``scheduling_timeout`` would wait in the queue
    past its deadline. Returns the nonterminal ``QUEUED_HANDOFF`` jobs whose stored
    ``scheduling_deadline_epoch_ms`` is set and already elapsed; the tick marks them
    ``UNSCHEDULABLE``.
    """
    rows = tx.execute(
        select(federated_jobs_table.c.job_id)
        .select_from(federated_jobs_table.join(jobs_table, federated_jobs_table.c.job_id == jobs_table.c.job_id))
        .where(
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
            federated_jobs_table.c.handoff_state == int(HandoffState.QUEUED_HANDOFF),
            jobs_table.c.state.notin_(list(TERMINAL_JOB_STATES)),
            jobs_table.c.scheduling_deadline_epoch_ms.isnot(None),
            jobs_table.c.scheduling_deadline_epoch_ms < now_ms,
        )
    ).all()
    return [r.job_id for r in rows]


def pending_cancel_handles(tx: Tx) -> list[FederatedHandle]:
    """SENT handles with a cancel intent set whose local mirrored job is not terminal.

    These are the routed ``TerminateJob`` targets the sync loop re-drives until the
    peer acks or sync mirrors the job terminal/pruned.
    """
    rows = tx.execute(
        select(*_SENT_HANDLE_COLUMNS)
        .select_from(federated_jobs_table.join(jobs_table, jobs_table.c.job_id == federated_jobs_table.c.job_id))
        .where(
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
            federated_jobs_table.c.cancel_intent_version > 0,
            jobs_table.c.state.notin_(bindparam("terminal_states", expanding=True)),
        ),
        {"terminal_states": list(TERMINAL_JOB_STATES)},
    ).all()
    return [_sent_handle(r) for r in rows]


def federated_sent_job(tx: Tx, peer_id: str, job_id: JobName) -> JobName | None:
    """``job_id`` iff a SENT ``federated_jobs`` handle for ``(peer_id, job_id)`` exists.

    Job ids are cluster-invariant — the peer runs and reports the same id the parent
    handed it — so the peer's reported id IS the local id. Confirms the parent
    actually handed ``job_id`` to ``peer_id`` before mirroring the peer's report; a
    peer reporting an id it was never handed is ignored.
    """
    row = tx.execute(
        select(federated_jobs_table.c.job_id).where(
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
            federated_jobs_table.c.peer_id == peer_id,
            federated_jobs_table.c.job_id == job_id,
        )
    ).first()
    return row.job_id if row is not None else None


def has_received_job_from_peer(tx: Tx, peer_id: str, job_id: JobName) -> bool:
    """Whether ``job_id`` is a RECEIVED handle this cluster took from ``peer_id``.

    Authorizes a federated /proxy: a peer's federation bearer may reach an endpoint
    only on a job that peer actually handed here. Received ownership is recorded on
    the root, so pass the endpoint job's root id.
    """
    row = tx.execute(
        select(federated_jobs_table.c.job_id).where(
            federated_jobs_table.c.direction == int(FederationDirection.RECEIVED),
            federated_jobs_table.c.peer_id == peer_id,
            federated_jobs_table.c.job_id == job_id,
        )
    ).first()
    return row is not None


def parent_mirror_seed(tx: Tx, parent_job_id: JobName):
    """The ``submitting_user`` and ``root_submitted_at_ms`` of an existing parent row.

    Seeds a mirrored child job — one born on a peer under a received root and
    reported back over sync — from its already-present parent: the whole federated
    subtree shares the root's submitter and root submit time. Returns the SA Row
    (``.submitting_user``, ``.root_submitted_at_ms``) or ``None`` when the parent is
    absent (a delta arrived out of order).
    """
    return tx.execute(
        select(jobs_table.c.submitting_user, jobs_table.c.root_submitted_at_ms).where(
            jobs_table.c.job_id == bindparam("job_id")
        ),
        {"job_id": parent_job_id},
    ).first()


def handoff_states(tx: Tx, job_ids: Sequence[JobName]) -> dict[JobName, int]:
    """``{job_id: handoff_state}`` for the SENT handles among ``job_ids``.

    The list path's batched counterpart to :func:`federated_handle`: one
    ``IN (...)`` read for a whole page instead of a per-job handle load. Job ids
    without a SENT handle (local jobs, peer-side RECEIVED rows) are simply absent
    from the result.
    """
    if not job_ids:
        return {}
    rows = tx.execute(
        select(federated_jobs_table.c.job_id, federated_jobs_table.c.handoff_state).where(
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
            federated_jobs_table.c.job_id.in_(bindparam("job_ids", expanding=True)),
        ),
        {"job_ids": list(job_ids)},
    ).all()
    return {r.job_id: int(r.handoff_state) for r in rows}


def received_requester(tx: Tx, job_id: JobName) -> str | None:
    """The requester ``peer_id`` of a RECEIVED ``federated_jobs`` row for ``job_id``, else ``None``.

    Drives peer-side handoff admission: a re-drive from the same requester is an
    idempotent replay; any other existing row is a genuine collision.
    """
    row = tx.execute(
        select(federated_jobs_table.c.peer_id).where(
            federated_jobs_table.c.job_id == job_id,
            federated_jobs_table.c.direction == int(FederationDirection.RECEIVED),
        )
    ).first()
    return row.peer_id if row is not None else None


def federated_handles_for_peer(tx: Tx, peer_id: str) -> set[JobName]:
    """The set of *handed-off* SENT job ids delegated to ``peer_id``.

    Restricted to ``HANDED_OFF`` handles: a still-``PENDING_HANDOFF`` handle is not
    on the peer yet (the re-drive owns it), so its absence from the peer's active
    set is expected — a full resync must not reap it.
    """
    rows = tx.execute(
        select(federated_jobs_table.c.job_id).where(
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
            federated_jobs_table.c.peer_id == peer_id,
            federated_jobs_table.c.handoff_state == bindparam("handed_off"),
        ),
        {"handed_off": int(HandoffState.HANDED_OFF)},
    ).all()
    return {r.job_id for r in rows}


def active_federated_job_count(tx: Tx, peer_id: str) -> int:
    """Count of non-terminal SENT federated handles delegated to ``peer_id``."""
    count = tx.execute(
        select(func.count())
        .select_from(federated_jobs_table.join(jobs_table, jobs_table.c.job_id == federated_jobs_table.c.job_id))
        .where(
            federated_jobs_table.c.direction == int(FederationDirection.SENT),
            federated_jobs_table.c.peer_id == bindparam("peer_id"),
            jobs_table.c.state.notin_(bindparam("terminal_states", expanding=True)),
        ),
        {"peer_id": peer_id, "terminal_states": list(TERMINAL_JOB_STATES)},
    ).scalar()
    return int(count or 0)


def read_sync_cursor(tx: Tx, peer_id: str) -> str:
    """The persisted delta-sync cursor for ``peer_id`` ("" on first contact)."""
    row = tx.execute(
        select(federation_sync_state_table.c.cursor).where(federation_sync_state_table.c.peer_id == peer_id)
    ).first()
    return row.cursor if row is not None else ""


# --- peer side: row-shaped changelog reads (the sync page is assembled in the
#     service, which needs the current job/task state to build each delta) ---


@dataclass(frozen=True)
class ChangelogRow:
    """One raw ``federation_changelog`` row for a requester."""

    job_id: JobName  # the cluster-invariant job id (same on peer and requester)
    task_index: int | None  # None = a job-level change ("all tasks")
    tombstone: bool
    seq: int


def changelog_max_seq(tx: Tx) -> int:
    """The highest changelog ``seq`` written, or 0 when the changelog is empty."""
    return int(tx.execute(select(func.coalesce(func.max(federation_changelog_table.c.seq), 0))).scalar() or 0)


def changelog_min_seq(tx: Tx) -> int:
    """The lowest changelog ``seq`` retained, or 0 when the changelog is empty."""
    return int(tx.execute(select(func.coalesce(func.min(federation_changelog_table.c.seq), 0))).scalar() or 0)


def received_jobs_for_requester(tx: Tx, requester_id: str) -> list[JobName]:
    """Every still-present job this peer received from ``requester_id`` (the full set
    a stale/first-contact requester is resynced with)."""
    rows = tx.execute(
        select(federated_jobs_table.c.job_id)
        .select_from(federated_jobs_table.join(jobs_table, jobs_table.c.job_id == federated_jobs_table.c.job_id))
        .where(
            federated_jobs_table.c.direction == int(FederationDirection.RECEIVED),
            federated_jobs_table.c.peer_id == requester_id,
        )
    ).all()
    return [r.job_id for r in rows]


@dataclass(frozen=True)
class ReceivedEndpointRow:
    """One live endpoint on a RECEIVED job, for the federation endpoint snapshot."""

    endpoint_id: str
    name: str
    address: str
    task_id: JobName
    access: int
    metadata: dict
    lease_deadline: Timestamp | None


def live_endpoints_for_requester(tx: Tx, requester_id: str, now: Timestamp) -> list[ReceivedEndpointRow]:
    """Every live endpoint across the jobs this peer received from ``requester_id``.

    Scoped through the received *root*: only the handed-off root gets a RECEIVED
    ``federated_jobs`` row, but a job it spawns runs locally on this peer under the
    same root, so an endpoint registered by such a child task is matched via its
    job's ``root_job_id`` rather than a direct ``federated_jobs`` handle. Expired
    leases are excluded (parity with the endpoint registry's own reads), so the
    parent mirror set-replaced from this matches what the child would serve.
    """
    rows = tx.execute(
        select(
            endpoints_table.c.endpoint_id,
            endpoints_table.c.name,
            endpoints_table.c.address,
            endpoints_table.c.task_id,
            endpoints_table.c.access,
            endpoints_table.c.metadata_json,
            endpoints_table.c.lease_deadline_ms,
        )
        .select_from(
            endpoints_table.join(jobs_table, jobs_table.c.job_id == endpoints_table.c.job_id).join(
                federated_jobs_table, federated_jobs_table.c.job_id == jobs_table.c.root_job_id
            )
        )
        .where(
            federated_jobs_table.c.direction == int(FederationDirection.RECEIVED),
            federated_jobs_table.c.peer_id == requester_id,
        )
    ).all()
    result: list[ReceivedEndpointRow] = []
    for r in rows:
        deadline = r.lease_deadline_ms
        if deadline is not None and deadline <= now:
            continue
        result.append(
            ReceivedEndpointRow(
                endpoint_id=r.endpoint_id,
                name=r.name,
                address=r.address,
                task_id=r.task_id,
                access=EndpointAccess.ENDPOINT_ACCESS_PRIVATE if r.access is None else int(r.access),
                metadata=r.metadata_json,
                lease_deadline=deadline,
            )
        )
    return result


def changelog_rows_since(tx: Tx, requester_id: str, cursor_seq: int) -> list[ChangelogRow]:
    """Every changelog row for ``requester_id`` past ``cursor_seq``, in ``seq`` order.

    Attribution is join-free: each row carries its ``requester_id``, so a tombstone
    is reported even after its job (and any RECEIVED handle) is deleted.
    """
    rows = tx.execute(
        select(
            federation_changelog_table.c.job_id,
            federation_changelog_table.c.task_index,
            federation_changelog_table.c.tombstone,
            federation_changelog_table.c.seq,
        )
        .where(
            federation_changelog_table.c.requester_id == requester_id,
            federation_changelog_table.c.seq > bindparam("cursor_seq"),
        )
        .order_by(federation_changelog_table.c.seq),
        {"cursor_seq": cursor_seq},
    ).all()
    return [
        ChangelogRow(
            job_id=r.job_id,
            task_index=int(r.task_index) if r.task_index is not None else None,
            tombstone=bool(r.tombstone),
            seq=int(r.seq),
        )
        for r in rows
    ]
