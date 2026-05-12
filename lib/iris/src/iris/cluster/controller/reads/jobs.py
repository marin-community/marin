# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job and job_config read helpers.

Return shapes:

* ``get_state`` — ``int | None``
* ``get_detail`` — SA ``Row`` or ``None``
* ``get_config`` — ``dict[str, Any] | None`` (raw mapping from job_config)
* ``get_priority_bands`` — ``dict[JobName, int]``
* ``get_workdir_files`` — ``dict[str, bytes]``
* ``has_unfinished_worker_attempts`` — ``bool``

Recursive CTEs (``get_priority_bands``, ``has_unfinished_worker_attempts``)
use ``text()`` because the self-referential SQL is cleaner than the
``select().cte(recursive=True)`` spelling for these specific shapes.
"""

from collections.abc import Iterable

from sqlalchemy import bindparam, select, text

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import (
    job_config_table,
    job_workdir_files_table,
    jobs_table,
)
from iris.cluster.types import JobName
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Simple scalar lookups
# ---------------------------------------------------------------------------


def get_state(tx: Tx, job_id: JobName) -> int | None:
    """Return the ``state`` column for ``job_id``, or None if absent."""
    row = tx.execute(
        select(jobs_table.c.state).where(jobs_table.c.job_id == bindparam("job_id")),
        {"job_id": job_id},
    ).first()
    return int(row.state) if row is not None else None


# ---------------------------------------------------------------------------
# Full job detail / job_config
# ---------------------------------------------------------------------------


def get_detail(tx: Tx, job_id: JobName):
    """Return SA Row for ``job_id`` or None."""
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


def get_config(tx: Tx, job_id: JobName) -> dict | None:
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


# ---------------------------------------------------------------------------
# Recursive CTEs
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

    Walks the parent_job_id chain for jobs with UNSPECIFIED (0) band until a
    non-zero band is found. Jobs whose entire ancestor chain is UNSPECIFIED
    fall back to ``PRIORITY_BAND_INTERACTIVE``.
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


def get_workdir_files(tx: Tx, job_id: JobName) -> dict[str, bytes]:
    """Return ``{filename: data}`` for all workdir files attached to ``job_id``."""
    rows = tx.execute(
        select(job_workdir_files_table.c.filename, job_workdir_files_table.c.data).where(
            job_workdir_files_table.c.job_id == bindparam("job_id")
        ),
        {"job_id": job_id},
    ).all()
    return {str(row.filename): bytes(row.data) for row in rows}


# Recursive CTE: true if any task in the subtree rooted at job_id has a
# worker-bound unfinished attempt.
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
