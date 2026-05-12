# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task read helpers (SA Core port).

Named ``text(...)`` SQL constants and small helpers for the read paths
on :class:`iris.cluster.controller.stores.TaskStore`. Stage 9 of the SA
Core migration introduces this module alongside the legacy methods;
parity tests in ``tests/cluster/controller/test_reads_tasks.py`` assert
the two paths return equal results.

Notes:

* The full task-detail projection mirrors ``TASK_DETAIL_PROJECTION``
  exactly. The decoder is built by hand here rather than reusing
  ``TASK_DETAIL_PROJECTION.decode`` because SA's ``Row`` exposes column
  values via ``__getattr__``, which is faster than the dict-style
  ``row[col]`` path the legacy projection uses.
* ``list_active`` accepts a :class:`TaskScope` (same shape as the
  legacy method) and composes a SA Core ``select(...)`` against the
  schema_v2 tables. Dynamic predicates compose cleanly via
  ``select.where(...)``; ``text()`` is reserved for the static hot-path
  queries that benefit from compile caching.
* ``bulk_get_detail`` uses an expanding bindparam (SA's
  ``IN (VALUES …)`` machinery) which handles SQLite's 999-parameter
  cap transparently — no manual chunking needed.
"""

from collections.abc import Iterable

from sqlalchemy import bindparam, select, text

from iris.cluster.controller.codec import resource_spec_from_scalars
from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema import (
    TaskDetailRow,
    _decode_bool_int,
    _nullable,
    decode_timestamp_ms,
)
from iris.cluster.controller.schema_v2 import job_config_table, jobs_table, tasks_table
from iris.cluster.controller.stores import (
    ActiveTaskRow,
    PendingDispatchRow,
    TaskScope,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Task detail
# ---------------------------------------------------------------------------

_TASK_DETAIL_SQL = text(
    "SELECT "
    "t.task_id AS task_id, "
    "t.job_id AS job_id, "
    "t.state AS state, "
    "t.current_attempt_id AS current_attempt_id, "
    "t.failure_count AS failure_count, "
    "t.preemption_count AS preemption_count, "
    "t.max_retries_failure AS max_retries_failure, "
    "t.max_retries_preemption AS max_retries_preemption, "
    "t.submitted_at_ms AS submitted_at_ms, "
    "t.priority_band AS priority_band, "
    "t.error AS error, "
    "t.exit_code AS exit_code, "
    "t.started_at_ms AS started_at_ms, "
    "t.finished_at_ms AS finished_at_ms, "
    "t.current_worker_id AS current_worker_id, "
    "t.current_worker_address AS current_worker_address, "
    "t.container_id AS container_id "
    "FROM tasks t WHERE t.task_id = :tid"
)

_BULK_TASK_DETAIL_SQL = text(
    "SELECT "
    "t.task_id AS task_id, "
    "t.job_id AS job_id, "
    "t.state AS state, "
    "t.current_attempt_id AS current_attempt_id, "
    "t.failure_count AS failure_count, "
    "t.preemption_count AS preemption_count, "
    "t.max_retries_failure AS max_retries_failure, "
    "t.max_retries_preemption AS max_retries_preemption, "
    "t.submitted_at_ms AS submitted_at_ms, "
    "t.priority_band AS priority_band, "
    "t.error AS error, "
    "t.exit_code AS exit_code, "
    "t.started_at_ms AS started_at_ms, "
    "t.finished_at_ms AS finished_at_ms, "
    "t.current_worker_id AS current_worker_id, "
    "t.current_worker_address AS current_worker_address, "
    "t.container_id AS container_id "
    "FROM tasks t WHERE t.task_id IN :tids"
).bindparams(bindparam("tids", expanding=True))


_NULL_TS = _nullable(decode_timestamp_ms)
_NULL_INT = _nullable(int)
_NULL_STR = _nullable(str)


def _row_to_task_detail(row) -> TaskDetailRow:
    current_worker_id = row.current_worker_id
    return TaskDetailRow(
        task_id=JobName.from_wire(str(row.task_id)),
        job_id=JobName.from_wire(str(row.job_id)),
        state=int(row.state),
        current_attempt_id=int(row.current_attempt_id),
        failure_count=int(row.failure_count),
        preemption_count=int(row.preemption_count),
        max_retries_failure=int(row.max_retries_failure),
        max_retries_preemption=int(row.max_retries_preemption),
        submitted_at=decode_timestamp_ms(row.submitted_at_ms),
        priority_band=int(row.priority_band),
        error=_NULL_STR(row.error),
        exit_code=_NULL_INT(row.exit_code),
        started_at=_NULL_TS(row.started_at_ms),
        finished_at=_NULL_TS(row.finished_at_ms),
        current_worker_id=WorkerId(str(current_worker_id)) if current_worker_id is not None else None,
        current_worker_address=_NULL_STR(row.current_worker_address),
        container_id=_NULL_STR(row.container_id),
    )


def get_detail(tx: Tx, task_id: JobName) -> TaskDetailRow | None:
    """Return the full :class:`TaskDetailRow` for ``task_id``, or None."""
    row = tx.execute(_TASK_DETAIL_SQL, {"tid": task_id.to_wire()}).first()
    if row is None:
        return None
    return _row_to_task_detail(row)


def bulk_get_detail(tx: Tx, task_ids: Iterable[JobName]) -> dict[JobName, TaskDetailRow]:
    """Return ``{task_id: TaskDetailRow}`` for all ``task_ids`` that exist.

    SA's expanding bindparam handles the SQLite parameter cap by emitting
    ``IN (VALUES …)`` chunks under the hood, so no manual chunking is
    needed at this layer.
    """
    wires = [tid.to_wire() for tid in task_ids]
    if not wires:
        return {}
    rows = tx.execute(_BULK_TASK_DETAIL_SQL, {"tids": wires}).all()
    return {JobName.from_wire(str(row.task_id)): _row_to_task_detail(row) for row in rows}


# ---------------------------------------------------------------------------
# Simple lookups
# ---------------------------------------------------------------------------

_GET_JOB_ID_SQL = text("SELECT job_id FROM tasks WHERE task_id = :tid")
_GET_CURRENT_ATTEMPT_SQL = text("SELECT current_attempt_id FROM tasks WHERE task_id = :tid")
_GET_PRIORITY_BAND_FOR_JOB_SQL = text("SELECT priority_band FROM tasks WHERE job_id = :jid LIMIT 1")
_STATE_COUNTS_FOR_JOB_SQL = text("SELECT state AS state, COUNT(*) AS c FROM tasks WHERE job_id = :jid GROUP BY state")
_FIRST_ERROR_FOR_JOB_SQL = text(
    "SELECT error FROM tasks WHERE job_id = :jid AND error IS NOT NULL ORDER BY task_index LIMIT 1"
)


def get_job_id(tx: Tx, task_id: JobName) -> JobName | None:
    """Return the owning ``job_id`` for ``task_id``, or None."""
    row = tx.execute(_GET_JOB_ID_SQL, {"tid": task_id.to_wire()}).first()
    return JobName.from_wire(str(row.job_id)) if row is not None else None


def get_current_attempt_id(tx: Tx, task_id: JobName) -> int | None:
    """Return ``tasks.current_attempt_id`` for ``task_id``, or None."""
    row = tx.execute(_GET_CURRENT_ATTEMPT_SQL, {"tid": task_id.to_wire()}).first()
    return int(row.current_attempt_id) if row is not None else None


def get_priority_band_for_job(tx: Tx, job_id: JobName) -> int | None:
    """Return one task's ``priority_band`` for ``job_id`` (all tasks share it)."""
    row = tx.execute(_GET_PRIORITY_BAND_FOR_JOB_SQL, {"jid": job_id.to_wire()}).first()
    return int(row.priority_band) if row is not None else None


def state_counts_for_job(tx: Tx, job_id: JobName) -> dict[int, int]:
    """Return ``{state: count}`` for every task of ``job_id``."""
    rows = tx.execute(_STATE_COUNTS_FOR_JOB_SQL, {"jid": job_id.to_wire()}).all()
    return {int(row.state): int(row.c) for row in rows}


def first_error_for_job(tx: Tx, job_id: JobName) -> str | None:
    """Return the first non-null ``error`` (ordered by task_index) for ``job_id``."""
    row = tx.execute(_FIRST_ERROR_FOR_JOB_SQL, {"jid": job_id.to_wire()}).first()
    return str(row.error) if row is not None else None


# ---------------------------------------------------------------------------
# Active / dispatch projections
# ---------------------------------------------------------------------------

# Reproduces ``stores._ACTIVE_TASK_PROJECTION``. Aliased so the join uses
# the same column qualifiers as the legacy query.
_ACTIVE_TASK_COLUMNS = (
    tasks_table.c.task_id.label("task_id"),
    tasks_table.c.job_id.label("job_id"),
    tasks_table.c.state.label("state"),
    tasks_table.c.current_attempt_id.label("current_attempt_id"),
    tasks_table.c.current_worker_id.label("current_worker_id"),
    tasks_table.c.failure_count.label("failure_count"),
    tasks_table.c.preemption_count.label("preemption_count"),
    tasks_table.c.max_retries_failure.label("max_retries_failure"),
    tasks_table.c.max_retries_preemption.label("max_retries_preemption"),
    jobs_table.c.is_reservation_holder.label("is_reservation_holder"),
    job_config_table.c.has_coscheduling.label("has_coscheduling"),
    job_config_table.c.res_cpu_millicores.label("res_cpu_millicores"),
    job_config_table.c.res_memory_bytes.label("res_memory_bytes"),
    job_config_table.c.res_disk_bytes.label("res_disk_bytes"),
    job_config_table.c.res_device_json.label("res_device_json"),
)


def _row_to_active_task(row) -> ActiveTaskRow:
    current_worker_id = row.current_worker_id
    return ActiveTaskRow(
        task_id=JobName.from_wire(str(row.task_id)),
        job_id=JobName.from_wire(str(row.job_id)),
        state=int(row.state),
        current_attempt_id=int(row.current_attempt_id),
        current_worker_id=WorkerId(str(current_worker_id)) if current_worker_id is not None else None,
        failure_count=int(row.failure_count),
        preemption_count=int(row.preemption_count),
        max_retries_failure=int(row.max_retries_failure),
        max_retries_preemption=int(row.max_retries_preemption),
        is_reservation_holder=_decode_bool_int(row.is_reservation_holder),
        has_coscheduling=_decode_bool_int(row.has_coscheduling),
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

    Semantics match :meth:`stores.TaskStore.list_active` byte-for-byte;
    see that docstring for scope and state-filter rules.
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

    stmt = select(*_ACTIVE_TASK_COLUMNS).select_from(
        tasks_table.join(jobs_table, jobs_table.c.job_id == tasks_table.c.job_id).join(
            job_config_table, job_config_table.c.job_id == jobs_table.c.job_id
        )
    )

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
        stmt = stmt.where(jobs_table.c.is_reservation_holder == 0)

    stmt = stmt.where(tasks_table.c.state.in_(states_tuple))
    if order_by_task_id:
        stmt = stmt.order_by(tasks_table.c.task_id.asc())
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = tx.execute(stmt).all()
    return [_row_to_active_task(row) for row in rows]


_GET_WITH_RESOURCES_SQL = text(
    "SELECT t.task_id AS task_id, t.job_id AS job_id, t.state AS state, "
    "t.current_attempt_id AS current_attempt_id, t.current_worker_id AS current_worker_id, "
    "t.failure_count AS failure_count, t.preemption_count AS preemption_count, "
    "t.max_retries_failure AS max_retries_failure, t.max_retries_preemption AS max_retries_preemption, "
    "j.is_reservation_holder AS is_reservation_holder, "
    "jc.has_coscheduling AS has_coscheduling, "
    "jc.res_cpu_millicores AS res_cpu_millicores, jc.res_memory_bytes AS res_memory_bytes, "
    "jc.res_disk_bytes AS res_disk_bytes, jc.res_device_json AS res_device_json "
    "FROM tasks t JOIN jobs j ON j.job_id = t.job_id "
    "JOIN job_config jc ON jc.job_id = j.job_id "
    "WHERE t.task_id = :tid"
)


def get_with_resources(tx: Tx, task_id: JobName) -> ActiveTaskRow | None:
    """Fetch a single task with its job_config resource projection.

    No state filter; callers (``preempt_task``) check ``state`` themselves.
    """
    row = tx.execute(_GET_WITH_RESOURCES_SQL, {"tid": task_id.to_wire()}).first()
    return _row_to_active_task(row) if row is not None else None


# ---------------------------------------------------------------------------
# Dispatch projection (direct-provider paths)
# ---------------------------------------------------------------------------

_DISPATCH_COLS = (
    "t.task_id AS task_id, t.job_id AS job_id, t.current_attempt_id AS current_attempt_id, "
    "j.num_tasks AS num_tasks, "
    "jc.res_cpu_millicores AS res_cpu_millicores, jc.res_memory_bytes AS res_memory_bytes, "
    "jc.res_disk_bytes AS res_disk_bytes, jc.res_device_json AS res_device_json, "
    "jc.entrypoint_json AS entrypoint_json, jc.environment_json AS environment_json, "
    "jc.bundle_id AS bundle_id, jc.ports_json AS ports_json, "
    "jc.constraints_json AS constraints_json, jc.task_image AS task_image, "
    "jc.timeout_ms AS timeout_ms"
)


_LIST_PENDING_DISPATCH_SQL = text(
    f"SELECT {_DISPATCH_COLS} FROM tasks t "
    "JOIN jobs j ON j.job_id = t.job_id "
    "JOIN job_config jc ON jc.job_id = j.job_id "
    "WHERE t.state = :state AND j.is_reservation_holder = 0 LIMIT :limit"
)


_LIST_ASSIGNED_NULL_WORKER_DISPATCH_SQL = text(
    f"SELECT {_DISPATCH_COLS} FROM tasks t "
    "JOIN jobs j ON j.job_id = t.job_id "
    "JOIN job_config jc ON jc.job_id = j.job_id "
    "WHERE t.state = :state AND t.current_worker_id IS NULL AND j.is_reservation_holder = 0"
)


def _row_to_dispatch(row) -> PendingDispatchRow:
    timeout_ms = row.timeout_ms
    return PendingDispatchRow(
        task_id=JobName.from_wire(str(row.task_id)),
        job_id=JobName.from_wire(str(row.job_id)),
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
    """Return pending non-holder tasks eligible for direct-provider dispatch."""
    if limit <= 0:
        return []
    rows = tx.execute(
        _LIST_PENDING_DISPATCH_SQL,
        {"state": int(job_pb2.TASK_STATE_PENDING), "limit": limit},
    ).all()
    return [_row_to_dispatch(row) for row in rows]


def list_assigned_null_worker_for_direct_provider(tx: Tx) -> list[PendingDispatchRow]:
    """Return ASSIGNED+null-worker rows with full runtime payload, for redrive."""
    rows = tx.execute(
        _LIST_ASSIGNED_NULL_WORKER_DISPATCH_SQL,
        {"state": int(job_pb2.TASK_STATE_ASSIGNED)},
    ).all()
    return [_row_to_dispatch(row) for row in rows]
