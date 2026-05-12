# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task read helpers (SA Core expression language).

All queries use ``select(table.c.col, ...)`` rather than ``text("SELECT
...")``. TypeDecorators on the schema_v2 columns decode values on read so
callers receive ``JobName``, ``Timestamp``, and ``WorkerId`` directly.

Return shapes:

* ``get_detail`` — SA ``Row`` or ``None``
* ``bulk_get_detail`` — ``dict[JobName, Row]``
* ``get_job_id`` — ``JobName | None``
* ``get_current_attempt_id`` — ``int | None``
* ``get_priority_band_for_job`` — ``int | None``
* ``state_counts_for_job`` — ``dict[int, int]``
* ``first_error_for_job`` — ``str | None``
* ``list_active`` — ``list[ActiveTaskRow]``
* ``get_with_resources`` — ``ActiveTaskRow | None``
* ``list_pending_for_direct_provider`` — ``list[PendingDispatchRow]``
* ``list_assigned_null_worker_for_direct_provider`` — ``list[PendingDispatchRow]``
"""

from collections.abc import Iterable

from sqlalchemy import bindparam, func, select, tuple_

from iris.cluster.controller.codec import resource_spec_from_scalars
from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.rows import (
    ActiveTaskRow,
    PendingDispatchRow,
    TaskScope,
)
from iris.cluster.controller.schema_v2 import job_config_table, jobs_table, tasks_table
from iris.cluster.types import JobName
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Task detail projection columns — shared by get_detail and bulk_get_detail
# ---------------------------------------------------------------------------

_TASK_DETAIL_COLS = (
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

GET_TASK_DETAIL_QUERY = select(*_TASK_DETAIL_COLS).where(tasks_table.c.task_id == bindparam("task_id"))

BULK_TASK_DETAIL_QUERY = select(*_TASK_DETAIL_COLS).where(
    tasks_table.c.task_id.in_(bindparam("task_ids", expanding=True))
)


def get_detail(tx: Tx, task_id: JobName):
    """Return SA Row for ``task_id`` or None.

    Row fields (TypeDecorator-decoded): task_id (JobName), job_id (JobName),
    submitted_at_ms (Timestamp), started_at_ms (Timestamp|None),
    finished_at_ms (Timestamp|None), current_worker_id (WorkerId|None).
    Remaining fields are plain int/str/None.
    """
    return tx.execute(GET_TASK_DETAIL_QUERY, {"task_id": task_id}).first()


def bulk_get_detail(tx: Tx, task_ids: Iterable[JobName]) -> dict[JobName, object]:
    """Return ``{task_id: Row}`` for all ``task_ids`` that exist.

    Missing keys are silently absent. SA's expanding bindparam handles the
    SQLite parameter cap transparently; no manual chunking needed.
    """
    ids = list(task_ids)
    if not ids:
        return {}
    rows = tx.execute(BULK_TASK_DETAIL_QUERY, {"task_ids": ids}).all()
    return {row.task_id: row for row in rows}


# ---------------------------------------------------------------------------
# Simple scalar lookups
# ---------------------------------------------------------------------------

GET_JOB_ID_QUERY = select(tasks_table.c.job_id).where(tasks_table.c.task_id == bindparam("task_id"))

GET_CURRENT_ATTEMPT_QUERY = select(tasks_table.c.current_attempt_id).where(tasks_table.c.task_id == bindparam("task_id"))

GET_PRIORITY_BAND_FOR_JOB_QUERY = (
    select(tasks_table.c.priority_band).where(tasks_table.c.job_id == bindparam("job_id")).limit(1)
)

STATE_COUNTS_FOR_JOB_QUERY = (
    select(tasks_table.c.state, func.count().label("c"))
    .where(tasks_table.c.job_id == bindparam("job_id"))
    .group_by(tasks_table.c.state)
)

FIRST_ERROR_FOR_JOB_QUERY = (
    select(tasks_table.c.error)
    .where(
        tasks_table.c.job_id == bindparam("job_id"),
        tasks_table.c.error.is_not(None),
    )
    .order_by(tasks_table.c.task_index)
    .limit(1)
)


def get_job_id(tx: Tx, task_id: JobName) -> JobName | None:
    """Return the owning ``job_id`` for ``task_id``, or None."""
    row = tx.execute(GET_JOB_ID_QUERY, {"task_id": task_id}).first()
    return row.job_id if row is not None else None


def get_current_attempt_id(tx: Tx, task_id: JobName) -> int | None:
    """Return ``tasks.current_attempt_id`` for ``task_id``, or None."""
    row = tx.execute(GET_CURRENT_ATTEMPT_QUERY, {"task_id": task_id}).first()
    return int(row.current_attempt_id) if row is not None else None


def get_priority_band_for_job(tx: Tx, job_id: JobName) -> int | None:
    """Return one task's ``priority_band`` for ``job_id`` (all tasks share it), or None."""
    row = tx.execute(GET_PRIORITY_BAND_FOR_JOB_QUERY, {"job_id": job_id}).first()
    return int(row.priority_band) if row is not None else None


def state_counts_for_job(tx: Tx, job_id: JobName) -> dict[int, int]:
    """Return ``{state: count}`` for every task state of ``job_id``."""
    rows = tx.execute(STATE_COUNTS_FOR_JOB_QUERY, {"job_id": job_id}).all()
    return {int(row.state): int(row.c) for row in rows}


def first_error_for_job(tx: Tx, job_id: JobName) -> str | None:
    """Return the first non-null ``error`` (ordered by task_index) for ``job_id``."""
    row = tx.execute(FIRST_ERROR_FOR_JOB_QUERY, {"job_id": job_id}).first()
    return str(row.error) if row is not None else None


# ---------------------------------------------------------------------------
# Active / dispatch projections
# ---------------------------------------------------------------------------

# Columns for the active-task join projection (tasks + jobs + job_config).
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
    job_config_table.c.res_cpu_millicores,
    job_config_table.c.res_memory_bytes,
    job_config_table.c.res_disk_bytes,
    job_config_table.c.res_device_json,
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
        resources=resource_spec_from_scalars(
            int(row.res_cpu_millicores),
            int(row.res_memory_bytes),
            int(row.res_disk_bytes),
            row.res_device_json,
        ),
    )


def list_active(
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
    predicate. Returns list[ActiveTaskRow].
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


GET_WITH_RESOURCES_QUERY = (
    select(*_ACTIVE_TASK_COLS).select_from(_ACTIVE_TASK_FROM).where(tasks_table.c.task_id == bindparam("task_id"))
)


def get_with_resources(tx: Tx, task_id: JobName) -> ActiveTaskRow | None:
    """Fetch a single task with its job_config resource projection.

    No state filter; callers (``preempt_task``) check ``state`` themselves.
    Returns ActiveTaskRow or None.
    """
    row = tx.execute(GET_WITH_RESOURCES_QUERY, {"task_id": task_id}).first()
    return _row_to_active_task(row) if row is not None else None


# ---------------------------------------------------------------------------
# Dispatch projection (direct-provider paths)
# ---------------------------------------------------------------------------

_DISPATCH_COLS = (
    tasks_table.c.task_id,
    tasks_table.c.job_id,
    tasks_table.c.current_attempt_id,
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
)

_DISPATCH_FROM = tasks_table.join(jobs_table, jobs_table.c.job_id == tasks_table.c.job_id).join(
    job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
)

LIST_PENDING_DISPATCH_QUERY = (
    select(*_DISPATCH_COLS)
    .select_from(_DISPATCH_FROM)
    .where(
        tasks_table.c.state == bindparam("state"),
        jobs_table.c.is_reservation_holder == False,  # noqa: E712
    )
    .limit(bindparam("limit"))
)

LIST_ASSIGNED_NULL_WORKER_DISPATCH_QUERY = (
    select(*_DISPATCH_COLS)
    .select_from(_DISPATCH_FROM)
    .where(
        tasks_table.c.state == bindparam("state"),
        tasks_table.c.current_worker_id.is_(None),
        jobs_table.c.is_reservation_holder == False,  # noqa: E712
    )
)


def _row_to_dispatch(row) -> PendingDispatchRow:
    timeout_ms = row.timeout_ms
    return PendingDispatchRow(
        task_id=row.task_id,
        job_id=row.job_id,
        current_attempt_id=int(row.current_attempt_id),
        num_tasks=int(row.num_tasks),
        resources=resource_spec_from_scalars(
            int(row.res_cpu_millicores),
            int(row.res_memory_bytes),
            int(row.res_disk_bytes),
            row.res_device_json,
        ),
        entrypoint_json=str(row.entrypoint_json),
        environment_json=str(row.environment_json),
        bundle_id=str(row.bundle_id),
        ports_json=str(row.ports_json),
        constraints_json=row.constraints_json,
        task_image=str(row.task_image),
        timeout_ms=int(timeout_ms) if timeout_ms is not None else None,
    )


def list_pending_for_direct_provider(tx: Tx, limit: int) -> list[PendingDispatchRow]:
    """Return pending non-holder tasks eligible for direct-provider dispatch.

    Returns list[PendingDispatchRow].
    """
    if limit <= 0:
        return []
    rows = tx.execute(
        LIST_PENDING_DISPATCH_QUERY,
        {"state": int(job_pb2.TASK_STATE_PENDING), "limit": limit},
    ).all()
    return [_row_to_dispatch(row) for row in rows]


def list_assigned_null_worker_for_direct_provider(tx: Tx) -> list[PendingDispatchRow]:
    """Return ASSIGNED+null-worker rows with full runtime payload for redrive.

    Returns list[PendingDispatchRow].
    """
    rows = tx.execute(
        LIST_ASSIGNED_NULL_WORKER_DISPATCH_QUERY,
        {"state": int(job_pb2.TASK_STATE_ASSIGNED)},
    ).all()
    return [_row_to_dispatch(row) for row in rows]


# ---------------------------------------------------------------------------
# Bulk attempt lookup (kept here for proximity with task reads)
# ---------------------------------------------------------------------------

_BULK_GET_CHUNK_SIZE = 450


def bulk_get_detail_chunked(
    tx: Tx,
    task_ids: Iterable[JobName],
) -> dict[JobName, object]:
    """Return ``{task_id: Row}`` for all ``task_ids`` that exist, chunked at 450.

    Useful when the caller holds more than SQLite's parameter limit in ids.
    For small lists ``bulk_get_detail`` is sufficient.
    """
    ids = list(task_ids)
    if not ids:
        return {}
    result: dict[JobName, object] = {}
    pair_col = tuple_(tasks_table.c.task_id)
    for chunk_start in range(0, len(ids), _BULK_GET_CHUNK_SIZE):
        chunk = ids[chunk_start : chunk_start + _BULK_GET_CHUNK_SIZE]
        stmt = select(*_TASK_DETAIL_COLS).where(pair_col.in_([(tid,) for tid in chunk]))
        for row in tx.execute(stmt).all():
            result[row.task_id] = row
    return result
