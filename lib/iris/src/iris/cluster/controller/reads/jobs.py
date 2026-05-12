# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job and job_config read helpers (SA Core expression language).

Every query uses ``select(table.c.col, ...)`` rather than ``text("SELECT
...")``. TypeDecorators on the schema_v2 columns decode values on read so
callers receive ``JobName``, ``Timestamp``, and ``bool`` directly without
manual conversion.

Return shapes:

* ``get_state`` — ``int | None``
* ``get_root_submitted_at_ms`` — ``int | None`` (epoch-ms)
* ``get_preemption_info`` — ``tuple[int, int] | None``
* ``get_recompute_basis`` — ``JobRecomputeBasis | None``
* ``get_detail`` — SA ``Row`` with TypeDecorator-decoded fields, or ``None``
* ``get_config`` — ``dict[str, Any] | None`` (raw mapping from job_config)
* ``get_priority_bands`` — ``dict[JobName, int]``
* ``list_descendants`` / ``list_subtree`` — ``list[JobName]``
* ``find_prunable`` — ``JobName | None``
* ``get_workdir_files`` — ``dict[str, bytes]``
* ``has_unfinished_worker_attempts`` — ``bool``

Recursive CTEs (``get_priority_bands``, ``list_descendants``,
``list_subtree``, ``has_unfinished_worker_attempts``) use ``text()``
because the self-referential SQL is cleaner than the SA Core
``select().cte(recursive=True)`` spelling for these specific shapes.
"""

from collections.abc import Iterable

from rigging.timing import Timestamp
from sqlalchemy import bindparam, select, text

from iris.cluster.controller.db import Tx
from iris.cluster.controller.rows import JobRecomputeBasis
from iris.cluster.controller.schema import (
    job_config_table,
    job_workdir_files_table,
    jobs_table,
)
from iris.cluster.types import TERMINAL_JOB_STATES, JobName
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Simple scalar lookups
# ---------------------------------------------------------------------------

GET_STATE_QUERY = select(jobs_table.c.state).where(jobs_table.c.job_id == bindparam("job_id"))

GET_ROOT_SUBMITTED_AT_QUERY = select(jobs_table.c.root_submitted_at_ms).where(jobs_table.c.job_id == bindparam("job_id"))

GET_PREEMPTION_INFO_QUERY = (
    select(job_config_table.c.preemption_policy, jobs_table.c.num_tasks)
    .select_from(jobs_table.join(job_config_table, jobs_table.c.job_id == job_config_table.c.job_id))
    .where(jobs_table.c.job_id == bindparam("job_id"))
)

GET_RECOMPUTE_BASIS_QUERY = (
    select(jobs_table.c.state, jobs_table.c.started_at_ms, job_config_table.c.max_task_failures)
    .select_from(jobs_table.join(job_config_table, jobs_table.c.job_id == job_config_table.c.job_id))
    .where(jobs_table.c.job_id == bindparam("job_id"))
)


def get_state(tx: Tx, job_id: JobName) -> int | None:
    """Return the ``state`` column for ``job_id``, or None if absent."""
    row = tx.execute(GET_STATE_QUERY, {"job_id": job_id}).first()
    return int(row.state) if row is not None else None


def get_root_submitted_at_ms(tx: Tx, job_id: JobName) -> int | None:
    """Return ``root_submitted_at_ms`` (epoch-ms int) for ``job_id``, or None if absent."""
    row = tx.execute(GET_ROOT_SUBMITTED_AT_QUERY, {"job_id": job_id}).first()
    # TimestampMsType decodes to Timestamp; convert back to int for legacy callers.
    return row.root_submitted_at_ms.epoch_ms() if row is not None else None


def get_preemption_info(tx: Tx, job_id: JobName) -> tuple[int, int] | None:
    """Return ``(preemption_policy, num_tasks)`` or None if the job is absent."""
    row = tx.execute(GET_PREEMPTION_INFO_QUERY, {"job_id": job_id}).first()
    if row is None:
        return None
    return int(row.preemption_policy), int(row.num_tasks)


def get_recompute_basis(tx: Tx, job_id: JobName) -> JobRecomputeBasis | None:
    """Return the inputs to ``_recompute_job_state`` for ``job_id``.

    Returns JobRecomputeBasis with ``started_at_ms`` as an epoch-ms int or None.
    """
    row = tx.execute(GET_RECOMPUTE_BASIS_QUERY, {"job_id": job_id}).first()
    if row is None:
        return None
    started_at = row.started_at_ms
    return JobRecomputeBasis(
        state=int(row.state),
        started_at_ms=started_at.epoch_ms() if started_at is not None else None,
        max_task_failures=int(row.max_task_failures),
    )


# ---------------------------------------------------------------------------
# Full job detail / job_config
# ---------------------------------------------------------------------------

# 37-column join of jobs + job_config. TypeDecorators decode:
#   job_id -> JobName, *_at_ms -> Timestamp, is_reservation_holder -> bool, etc.
JOB_DETAIL_QUERY = (
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
    .where(jobs_table.c.job_id == bindparam("job_id"))
)


def get_detail(tx: Tx, job_id: JobName):
    """Return SA Row for ``job_id`` or None.

    Row fields (TypeDecorator-decoded): job_id (JobName), submitted_at_ms
    (Timestamp), root_submitted_at_ms (Timestamp), started_at_ms (Timestamp|None),
    finished_at_ms (Timestamp|None), is_reservation_holder (bool),
    has_reservation (bool), has_coscheduling (bool), fail_if_exists (bool).
    Remaining fields are plain int/str/None as stored.
    """
    return tx.execute(JOB_DETAIL_QUERY, {"job_id": job_id}).first()


GET_CONFIG_QUERY = select(job_config_table).where(job_config_table.c.job_id == bindparam("job_id"))


def get_config(tx: Tx, job_id: JobName) -> dict | None:
    """Return the ``job_config`` row as ``{column_name: value}``, or None.

    Values are TypeDecorator-decoded (job_id -> JobName, has_reservation -> bool, etc.).
    Callers access fields by string key, e.g. ``jc["res_cpu_millicores"]``.
    """
    row = tx.execute(GET_CONFIG_QUERY, {"job_id": job_id}).mappings().first()
    return dict(row) if row is not None else None


# ---------------------------------------------------------------------------
# Recursive CTEs — text() is cleaner than SA Core CTE API for these shapes.
# ---------------------------------------------------------------------------

# Walks parent_job_id chain until a non-UNSPECIFIED priority_band is found.
# Inputs whose entire ancestor chain is UNSPECIFIED are absent from the result;
# the caller substitutes INTERACTIVE for those.
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

    Walks the parent_job_id chain for UNSPECIFIED (0) jobs until a non-zero
    band is found. Jobs whose entire ancestor chain is UNSPECIFIED fall back
    to ``PRIORITY_BAND_INTERACTIVE``.
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

    When ``exclude_reservation_holders=True``, reservation-holder nodes and
    their subtrees are pruned. Returns list[JobName].
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
    """Return ``root_id`` and all its transitive descendants. Returns list[JobName]."""
    rows = tx.execute(_LIST_SUBTREE_SQL, {"root": root_id.to_wire()}).all()
    return [JobName.from_wire(str(row.job_id)) for row in rows]


# ---------------------------------------------------------------------------
# Misc reads
# ---------------------------------------------------------------------------

FIND_PRUNABLE_QUERY = (
    select(jobs_table.c.job_id)
    .where(
        jobs_table.c.state.in_(bindparam("terminal_states", expanding=True)),
        jobs_table.c.finished_at_ms.is_not(None),
        jobs_table.c.finished_at_ms < bindparam("before_ts"),
    )
    .limit(1)
)


def find_prunable(tx: Tx, before_ms: int) -> JobName | None:
    """Return one terminal job whose ``finished_at_ms < before_ms``, or None.

    ``before_ms`` is an epoch-millisecond integer; it is converted to a
    ``Timestamp`` so the TimestampMsType bind processor can call ``.epoch_ms()``.
    Returns JobName or None.
    """
    row = tx.execute(
        FIND_PRUNABLE_QUERY,
        {"terminal_states": list(TERMINAL_JOB_STATES), "before_ts": Timestamp.from_ms(before_ms)},
    ).first()
    return row.job_id if row is not None else None


GET_WORKDIR_FILES_QUERY = select(job_workdir_files_table.c.filename, job_workdir_files_table.c.data).where(
    job_workdir_files_table.c.job_id == bindparam("job_id")
)


def get_workdir_files(tx: Tx, job_id: JobName) -> dict[str, bytes]:
    """Return ``{filename: data}`` for all workdir files attached to ``job_id``."""
    rows = tx.execute(GET_WORKDIR_FILES_QUERY, {"job_id": job_id}).all()
    return {str(row.filename): bytes(row.data) for row in rows}


# Recursive CTE: true if any task in the subtree rooted at job_id has a
# worker-bound unfinished attempt. Kept as text() for the same reason as the
# other recursive CTEs above.
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
    """Return True if any task under ``job_id`` (subtree) has a worker-bound unfinished attempt."""
    row = tx.execute(_HAS_UNFINISHED_WORKER_ATTEMPTS_SQL, {"jid": job_id.to_wire()}).first()
    return row is not None
