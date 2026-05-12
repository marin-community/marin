# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job and job_config read helpers (SA Core port).

Named ``text(...)`` SQL constants plus small dataclass-constructing
helpers for every read that today lives on
:class:`iris.cluster.controller.stores.JobStore`. Stage 8 of the
SQLAlchemy Core migration introduces this module alongside the legacy
``JobStore`` reads; parity tests in
``tests/cluster/controller/test_reads_jobs.py`` assert the two paths
return identical results against the same DB state. The legacy methods
remain unchanged in this stage — call-site switchover happens in stage
13 once every read path has an SA Core equivalent.

Implementation notes:

* Simple lookups use ``text("SELECT ... WHERE ... = :jid")`` with
  bindparams. This matches the idiom established in
  ``reads/scheduler.py`` (Stage 5). ``text()`` avoids ~370 µs/call of
  ``select(...)`` compilation overhead even with the SA statement
  cache enabled.
* Recursive CTEs (``get_priority_bands``, ``list_descendants``,
  ``list_subtree``, ``has_unfinished_worker_attempts``) use ``text()``
  directly. The CTE SQL is short and matches the legacy queries
  verbatim, which is more readable than the SA Core
  ``select().cte(recursive=True)`` API for a one-shot port.
* ``get_detail`` builds a :class:`JobDetailRow` by hand using the same
  decoder helpers (``decode_timestamp_ms``, ``_decode_bool_int``,
  ``_nullable``) as the legacy ``JOB_DETAIL_PROJECTION``. The two
  paths must produce equal dataclasses for parity tests to pass.
"""

from collections.abc import Iterable

from sqlalchemy import bindparam, text

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema import (
    JobDetailRow,
    _decode_bool_int,
    _nullable,
    decode_timestamp_ms,
)
from iris.cluster.controller.stores import JobRecomputeBasis
from iris.cluster.types import TERMINAL_JOB_STATES, JobName
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Simple lookups
# ---------------------------------------------------------------------------

_GET_STATE_SQL = text("SELECT state FROM jobs WHERE job_id = :jid")
_GET_ROOT_SUBMITTED_AT_SQL = text("SELECT root_submitted_at_ms FROM jobs WHERE job_id = :jid")
_GET_PREEMPTION_INFO_SQL = text(
    "SELECT jc.preemption_policy AS preemption_policy, j.num_tasks AS num_tasks "
    "FROM jobs j JOIN job_config jc ON jc.job_id = j.job_id WHERE j.job_id = :jid"
)
_GET_RECOMPUTE_BASIS_SQL = text(
    "SELECT j.state AS state, j.started_at_ms AS started_at_ms, jc.max_task_failures AS max_task_failures "
    "FROM jobs j JOIN job_config jc ON jc.job_id = j.job_id WHERE j.job_id = :jid"
)


def get_state(tx: Tx, job_id: JobName) -> int | None:
    """Return the ``state`` column for ``job_id``, or None if absent."""
    row = tx.execute(_GET_STATE_SQL, {"jid": job_id.to_wire()}).first()
    return int(row.state) if row is not None else None


def get_root_submitted_at_ms(tx: Tx, job_id: JobName) -> int | None:
    """Return ``root_submitted_at_ms`` for ``job_id``, or None if absent."""
    row = tx.execute(_GET_ROOT_SUBMITTED_AT_SQL, {"jid": job_id.to_wire()}).first()
    return int(row.root_submitted_at_ms) if row is not None else None


def get_preemption_info(tx: Tx, job_id: JobName) -> tuple[int, int] | None:
    """Return ``(preemption_policy, num_tasks)`` or None if the job is gone."""
    row = tx.execute(_GET_PREEMPTION_INFO_SQL, {"jid": job_id.to_wire()}).first()
    if row is None:
        return None
    return int(row.preemption_policy), int(row.num_tasks)


def get_recompute_basis(tx: Tx, job_id: JobName) -> JobRecomputeBasis | None:
    """Return the inputs to ``_recompute_job_state`` for ``job_id``."""
    row = tx.execute(_GET_RECOMPUTE_BASIS_SQL, {"jid": job_id.to_wire()}).first()
    if row is None:
        return None
    return JobRecomputeBasis(
        state=int(row.state),
        started_at_ms=int(row.started_at_ms) if row.started_at_ms is not None else None,
        max_task_failures=int(row.max_task_failures),
    )


# ---------------------------------------------------------------------------
# Full job detail / job_config
# ---------------------------------------------------------------------------

# Column list and aliases match ``schema.JOB_DETAIL_PROJECTION`` exactly:
# duplicated columns (name, has_reservation) come from ``jobs`` (alias ``j``)
# because ``_job_columns`` resolves JOBS before JOB_CONFIG in the lookup.
_JOB_DETAIL_SQL = text(
    "SELECT "
    "j.job_id AS job_id, "
    "j.state AS state, "
    "j.submitted_at_ms AS submitted_at_ms, "
    "j.root_submitted_at_ms AS root_submitted_at_ms, "
    "j.started_at_ms AS started_at_ms, "
    "j.finished_at_ms AS finished_at_ms, "
    "j.scheduling_deadline_epoch_ms AS scheduling_deadline_epoch_ms, "
    "j.error AS error, "
    "j.exit_code AS exit_code, "
    "j.num_tasks AS num_tasks, "
    "j.is_reservation_holder AS is_reservation_holder, "
    "j.has_reservation AS has_reservation, "
    "j.name AS name, "
    "j.depth AS depth, "
    "jc.res_cpu_millicores AS res_cpu_millicores, "
    "jc.res_memory_bytes AS res_memory_bytes, "
    "jc.res_disk_bytes AS res_disk_bytes, "
    "jc.res_device_json AS res_device_json, "
    "jc.constraints_json AS constraints_json, "
    "jc.has_coscheduling AS has_coscheduling, "
    "jc.coscheduling_group_by AS coscheduling_group_by, "
    "jc.scheduling_timeout_ms AS scheduling_timeout_ms, "
    "jc.max_task_failures AS max_task_failures, "
    "jc.entrypoint_json AS entrypoint_json, "
    "jc.environment_json AS environment_json, "
    "jc.bundle_id AS bundle_id, "
    "jc.ports_json AS ports_json, "
    "jc.max_retries_failure AS max_retries_failure, "
    "jc.max_retries_preemption AS max_retries_preemption, "
    "jc.timeout_ms AS timeout_ms, "
    "jc.preemption_policy AS preemption_policy, "
    "jc.existing_job_policy AS existing_job_policy, "
    "jc.priority_band AS priority_band, "
    "jc.task_image AS task_image, "
    "jc.submit_argv_json AS submit_argv_json, "
    "jc.reservation_json AS reservation_json, "
    "jc.fail_if_exists AS fail_if_exists "
    "FROM jobs j JOIN job_config jc ON jc.job_id = j.job_id "
    "WHERE j.job_id = :jid"
)

_NULL_TIMESTAMP_MS = _nullable(decode_timestamp_ms)
_NULL_INT = _nullable(int)
_NULL_STR = _nullable(str)


def get_detail(tx: Tx, job_id: JobName) -> JobDetailRow | None:
    """Return the full :class:`JobDetailRow` for ``job_id`` (37-column projection)."""
    row = tx.execute(_JOB_DETAIL_SQL, {"jid": job_id.to_wire()}).first()
    if row is None:
        return None
    return JobDetailRow(
        job_id=JobName.from_wire(str(row.job_id)),
        state=int(row.state),
        submitted_at=decode_timestamp_ms(row.submitted_at_ms),
        root_submitted_at=decode_timestamp_ms(row.root_submitted_at_ms),
        started_at=_NULL_TIMESTAMP_MS(row.started_at_ms),
        finished_at=_NULL_TIMESTAMP_MS(row.finished_at_ms),
        scheduling_deadline_epoch_ms=_NULL_INT(row.scheduling_deadline_epoch_ms),
        error=_NULL_STR(row.error),
        exit_code=_NULL_INT(row.exit_code),
        num_tasks=int(row.num_tasks),
        is_reservation_holder=_decode_bool_int(row.is_reservation_holder),
        has_reservation=_decode_bool_int(row.has_reservation),
        name=str(row.name),
        depth=int(row.depth),
        res_cpu_millicores=int(row.res_cpu_millicores),
        res_memory_bytes=int(row.res_memory_bytes),
        res_disk_bytes=int(row.res_disk_bytes),
        res_device_json=_NULL_STR(row.res_device_json),
        constraints_json=_NULL_STR(row.constraints_json),
        has_coscheduling=_decode_bool_int(row.has_coscheduling),
        coscheduling_group_by=str(row.coscheduling_group_by),
        scheduling_timeout_ms=_NULL_INT(row.scheduling_timeout_ms),
        max_task_failures=int(row.max_task_failures),
        entrypoint_json=str(row.entrypoint_json),
        environment_json=str(row.environment_json),
        bundle_id=str(row.bundle_id),
        ports_json=str(row.ports_json),
        max_retries_failure=int(row.max_retries_failure),
        max_retries_preemption=int(row.max_retries_preemption),
        timeout_ms=_NULL_INT(row.timeout_ms),
        preemption_policy=int(row.preemption_policy),
        existing_job_policy=int(row.existing_job_policy),
        priority_band=int(row.priority_band),
        task_image=str(row.task_image),
        submit_argv_json=str(row.submit_argv_json),
        reservation_json=_NULL_STR(row.reservation_json),
        fail_if_exists=_decode_bool_int(row.fail_if_exists),
    )


_GET_CONFIG_SQL = text("SELECT * FROM job_config WHERE job_id = :jid")


def get_config(tx: Tx, job_id: JobName) -> dict | None:
    """Return the raw ``job_config`` row as ``{column_name: value}``, or None.

    Mirrors :meth:`stores.JobStore.get_config` — callers currently
    access fields by string key (e.g. ``jc["res_cpu_millicores"]``).
    """
    row = tx.execute(_GET_CONFIG_SQL, {"jid": job_id.to_wire()}).mappings().first()
    return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Recursive CTEs
# ---------------------------------------------------------------------------

# Walks parent_job_id chain until a non-UNSPECIFIED priority_band is found.
# Inputs whose entire ancestor chain is UNSPECIFIED are absent from the
# result; the caller substitutes INTERACTIVE for those.
_PRIORITY_BANDS_SQL = text(
    "WITH RECURSIVE chain(input_id, current_id, current_band, parent_id) AS ("
    "  SELECT j.job_id, j.job_id, jc.priority_band, j.parent_job_id "
    "  FROM jobs j JOIN job_config jc ON jc.job_id = j.job_id "
    "  WHERE j.job_id IN :wires "
    "  UNION ALL "
    "  SELECT chain.input_id, j.job_id, jc.priority_band, j.parent_job_id "
    "  FROM chain "
    "  JOIN jobs j ON j.job_id = chain.parent_id "
    "  JOIN job_config jc ON jc.job_id = j.job_id "
    "  WHERE chain.current_band = 0"
    ") "
    "SELECT input_id, current_band FROM chain WHERE current_band != 0"
).bindparams(bindparam("wires", expanding=True))


def get_priority_bands(tx: Tx, job_ids: Iterable[JobName]) -> dict[JobName, int]:
    """Return ``{job_id: resolved priority_band}`` for the given jobs.

    See :meth:`stores.JobStore.get_priority_bands` for the resolution
    rule. ``UNSPECIFIED`` (0) inputs whose entire ancestor chain is
    also UNSPECIFIED fall back to ``PRIORITY_BAND_INTERACTIVE``.
    """
    ids = list(job_ids)
    if not ids:
        return {}
    wires = [jid.to_wire() for jid in ids]
    rows = tx.execute(_PRIORITY_BANDS_SQL, {"wires": wires}).all()
    resolved: dict[JobName, int] = {}
    for row in rows:
        resolved[JobName.from_wire(str(row.input_id))] = int(row.current_band)
    for jid in ids:
        resolved.setdefault(jid, int(job_pb2.PRIORITY_BAND_INTERACTIVE))
    return resolved


_LIST_DESCENDANTS_SQL = text(
    "WITH RECURSIVE subtree(job_id) AS ("
    "  SELECT job_id FROM jobs WHERE parent_job_id = :parent "
    "  UNION ALL "
    "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
    ") SELECT job_id FROM subtree"
)

_LIST_DESCENDANTS_EXCLUDE_HOLDERS_SQL = text(
    "WITH RECURSIVE subtree(job_id) AS ("
    "  SELECT job_id FROM jobs WHERE parent_job_id = :parent AND is_reservation_holder = 0 "
    "  UNION ALL "
    "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id "
    "   WHERE j.is_reservation_holder = 0"
    ") SELECT job_id FROM subtree"
)


def list_descendants(
    tx: Tx,
    parent_id: JobName,
    *,
    exclude_reservation_holders: bool = False,
) -> list[JobName]:
    """Return all transitive descendants of ``parent_id`` (not ``parent_id`` itself).

    See :meth:`stores.JobStore.list_descendants` for the semantics of
    ``exclude_reservation_holders``.
    """
    stmt = _LIST_DESCENDANTS_EXCLUDE_HOLDERS_SQL if exclude_reservation_holders else _LIST_DESCENDANTS_SQL
    rows = tx.execute(stmt, {"parent": parent_id.to_wire()}).all()
    return [JobName.from_wire(str(row.job_id)) for row in rows]


_LIST_SUBTREE_SQL = text(
    "WITH RECURSIVE subtree(job_id) AS ("
    "  SELECT job_id FROM jobs WHERE job_id = :root "
    "  UNION ALL "
    "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
    ") SELECT job_id FROM subtree"
)


def list_subtree(tx: Tx, root_id: JobName) -> list[JobName]:
    """Return ``root_id`` and all its transitive descendants."""
    rows = tx.execute(_LIST_SUBTREE_SQL, {"root": root_id.to_wire()}).all()
    return [JobName.from_wire(str(row.job_id)) for row in rows]


# ---------------------------------------------------------------------------
# Misc reads
# ---------------------------------------------------------------------------

_FIND_PRUNABLE_SQL = text(
    "SELECT job_id FROM jobs WHERE state IN :terminal_states "
    "AND finished_at_ms IS NOT NULL AND finished_at_ms < :before_ms LIMIT 1"
).bindparams(bindparam("terminal_states", expanding=True))


def find_prunable(tx: Tx, before_ms: int) -> JobName | None:
    """Return one terminal job whose ``finished_at_ms < before_ms``, or None."""
    row = tx.execute(
        _FIND_PRUNABLE_SQL,
        {"terminal_states": list(TERMINAL_JOB_STATES), "before_ms": before_ms},
    ).first()
    return JobName.from_wire(str(row.job_id)) if row is not None else None


_GET_WORKDIR_FILES_SQL = text("SELECT filename, data FROM job_workdir_files WHERE job_id = :jid")


def get_workdir_files(tx: Tx, job_id: JobName) -> dict[str, bytes]:
    """Return ``{filename: data}`` for all workdir files attached to ``job_id``."""
    rows = tx.execute(_GET_WORKDIR_FILES_SQL, {"jid": job_id.to_wire()}).all()
    return {str(row.filename): bytes(row.data) for row in rows}


_HAS_UNFINISHED_WORKER_ATTEMPTS_SQL = text(
    "WITH RECURSIVE subtree(job_id) AS ("
    "  SELECT job_id FROM jobs WHERE job_id = :jid "
    "  UNION ALL "
    "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
    ") "
    "SELECT 1 FROM tasks t "
    "JOIN task_attempts ta ON ta.task_id = t.task_id "
    "WHERE t.job_id IN subtree "
    "  AND ta.worker_id IS NOT NULL "
    "  AND ta.finished_at_ms IS NULL "
    "LIMIT 1"
)


def has_unfinished_worker_attempts(tx: Tx, job_id: JobName) -> bool:
    """True if any task under ``job_id`` (subtree) still has a worker-bound attempt.

    See :meth:`stores.JobStore.has_unfinished_worker_attempts` for the
    reason this gate exists.
    """
    row = tx.execute(_HAS_UNFINISHED_WORKER_ATTEMPTS_SQL, {"jid": job_id.to_wire()}).first()
    return row is not None
