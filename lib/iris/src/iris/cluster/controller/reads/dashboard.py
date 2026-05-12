# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dashboard composite reads (SA Core port).

The dashboard composes a small handful of heavy queries: a paged, sortable
job listing (``list_jobs``); per-job task aggregates
(``task_summaries_for_jobs``); and a parent-with-children lookup
(``parent_ids_with_children``). Today these live as ad-hoc SQL on
:mod:`iris.cluster.controller.service` lines 656-803. Stage 10 of the SA
Core migration ports them alongside the legacy implementations; parity
tests in ``tests/cluster/controller/test_reads_dashboard.py`` assert the
two paths return equal results against the same DB state.

Implementation notes:

* :func:`list_jobs` uses SA Core ``select(...)`` chaining (``.where``,
  ``.order_by``, ``.limit``, ``.offset``) rather than ``text(...)`` because
  the query shape is dynamic. The sort-field whitelist
  (:data:`_SORT_FIELD_TO_COLUMN`) maps protobuf sort-field ints to SA
  column references; the input integer is never spliced into SQL — only
  the resulting column expression is. This preserves the existing
  injection-safety contract from the legacy ``_SORT_FIELD_TO_SQL`` map.
* The dashboard ``JOB_ROW_PROJECTION`` is a 12-column subset of jobs +
  job_config; :func:`_row_to_job_row` builds the :class:`JobRow`
  dataclass by hand using the same decoders as the legacy projection.
* :func:`task_summaries_for_jobs` aggregates per-job task counts via
  ``func.count``/``func.sum`` + ``group_by``. The Python-side reduction
  mirrors the legacy loop in :func:`service._task_summaries_for_jobs`
  byte-for-byte.
"""

from collections.abc import Iterable

from sqlalchemy import Integer, bindparam, case, cast, func, select, text

from iris.cluster.controller.db import TaskJobSummary
from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema import JobRow, _nullable, decode_timestamp_ms
from iris.cluster.controller.schema_v2 import (
    job_config_table,
    jobs_table,
    tasks_table,
)
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2

# ---------------------------------------------------------------------------
# Sort-field whitelist
#
# Maps proto JobSortField enum values to SA column expressions. The
# enum value is used as a dict key only; the value is the safe SA Core
# expression that flows into ORDER BY. Unknown enum values fall through
# to the default (submitted_at_ms).
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


# Aggregate columns referenced by the failure/preemption sort fields. Defined
# at module scope so the same SA Label objects flow into both .order_by(...)
# and the GROUP BY clause without re-creating expressions.
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
# JobRow projection (12-col subset of jobs + job_config)
# ---------------------------------------------------------------------------

_JOB_ROW_COLUMNS = (
    jobs_table.c.job_id.label("job_id"),
    jobs_table.c.state.label("state"),
    jobs_table.c.submitted_at_ms.label("submitted_at_ms"),
    jobs_table.c.started_at_ms.label("started_at_ms"),
    jobs_table.c.finished_at_ms.label("finished_at_ms"),
    jobs_table.c.error.label("error"),
    jobs_table.c.exit_code.label("exit_code"),
    jobs_table.c.name.label("name"),
    jobs_table.c.depth.label("depth"),
    job_config_table.c.res_cpu_millicores.label("res_cpu_millicores"),
    job_config_table.c.res_memory_bytes.label("res_memory_bytes"),
    job_config_table.c.res_disk_bytes.label("res_disk_bytes"),
    job_config_table.c.res_device_json.label("res_device_json"),
)


_NULL_TS = _nullable(decode_timestamp_ms)
_NULL_INT = _nullable(int)
_NULL_STR = _nullable(str)


def _row_to_job_row(row) -> JobRow:
    # SA Core's TypeDecorators have already produced ``Timestamp`` / ``JobName``
    # objects for the typed columns; the local decoders here mirror the legacy
    # ``JOB_ROW_PROJECTION``: pass through the typed values, fall back to the
    # nullable-int/nullable-str helpers for the remaining columns.
    submitted_at = row.submitted_at_ms
    started_at = row.started_at_ms
    finished_at = row.finished_at_ms
    return JobRow(
        job_id=row.job_id,
        state=int(row.state),
        submitted_at=submitted_at,
        started_at=started_at,
        finished_at=finished_at,
        error=_NULL_STR(row.error),
        exit_code=_NULL_INT(row.exit_code),
        name=str(row.name),
        depth=int(row.depth),
        res_cpu_millicores=int(row.res_cpu_millicores),
        res_memory_bytes=int(row.res_memory_bytes),
        res_disk_bytes=int(row.res_disk_bytes),
        res_device_json=_NULL_STR(row.res_device_json),
    )


# ---------------------------------------------------------------------------
# list_jobs (paged, sortable dashboard listing)
# ---------------------------------------------------------------------------


def list_jobs(
    tx: Tx,
    query: controller_pb2.Controller.JobQuery,
    state_ids: tuple[int, ...],
) -> tuple[list[JobRow], int]:
    """Return ``(rows, total_count)`` for the given dashboard ``JobQuery``.

    Mirrors :func:`iris.cluster.controller.service._query_jobs` byte-for-byte.
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

    if depth_filter is not None:
        stmt = stmt.where(jobs_table.c.depth == depth_filter)
    if parent_filter is not None:
        stmt = stmt.where(jobs_table.c.parent_job_id == JobName.from_wire(parent_filter))

    stmt = stmt.where(jobs_table.c.state.in_(list(state_ids)))

    if query.name_filter:
        # Legacy lowercases the needle before LIKE; the indexed column is not
        # lowercased so this match remains case-sensitive against the stored
        # value, exactly as today.
        stmt = stmt.where(jobs_table.c.name.like(f"%{query.name_filter.lower()}%"))

    if needs_task_agg:
        stmt = stmt.group_by(jobs_table.c.job_id)

    stmt = stmt.order_by(order_expr)

    # COUNT(*) filters on j.* only — the FK to job_config means the joined
    # form cannot diverge; the join would just add a B-tree probe per row.
    count_stmt = select(func.count()).select_from(jobs_table)
    if depth_filter is not None:
        count_stmt = count_stmt.where(jobs_table.c.depth == depth_filter)
    if parent_filter is not None:
        count_stmt = count_stmt.where(jobs_table.c.parent_job_id == JobName.from_wire(parent_filter))
    count_stmt = count_stmt.where(jobs_table.c.state.in_(list(state_ids)))
    if query.name_filter:
        count_stmt = count_stmt.where(jobs_table.c.name.like(f"%{query.name_filter.lower()}%"))

    offset = max(query.offset, 0)
    limit = max(query.limit, 0)
    if limit > 0:
        stmt = stmt.limit(limit).offset(offset)

    rows = tx.execute(stmt).all()
    total = int(tx.execute(count_stmt).scalar() or 0)
    return [_row_to_job_row(row) for row in rows], total


# ---------------------------------------------------------------------------
# Per-job task summaries (aggregate)
# ---------------------------------------------------------------------------

_COMPLETED_TASK_STATES = (job_pb2.TASK_STATE_SUCCEEDED, job_pb2.TASK_STATE_KILLED)


def task_summaries_for_jobs(tx: Tx, job_ids: Iterable[JobName]) -> dict[JobName, TaskJobSummary]:
    """Return ``{job_id: TaskJobSummary}`` aggregating each job's tasks.

    Mirrors :func:`iris.cluster.controller.service._task_summaries_for_jobs`.
    """
    ids = list(job_ids)
    if not ids:
        return {}

    stmt = (
        select(
            tasks_table.c.job_id.label("job_id"),
            tasks_table.c.state.label("state"),
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


# ---------------------------------------------------------------------------
# Parents that currently have at least one direct child
# ---------------------------------------------------------------------------

_PARENT_IDS_WITH_CHILDREN_SQL = text(
    "SELECT DISTINCT j.parent_job_id FROM jobs j WHERE j.parent_job_id IN :pids"
).bindparams(bindparam("pids", expanding=True))


def parent_ids_with_children(tx: Tx, job_ids: Iterable[JobName]) -> set[JobName]:
    """Return the subset of ``job_ids`` that currently have at least one direct child."""
    ids = [j.to_wire() for j in job_ids]
    if not ids:
        return set()
    rows = tx.execute(_PARENT_IDS_WITH_CHILDREN_SQL, {"pids": ids}).all()
    return {JobName.from_wire(str(row.parent_job_id)) for row in rows if row.parent_job_id}
