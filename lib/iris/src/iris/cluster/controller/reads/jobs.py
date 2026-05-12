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
"""

from collections.abc import Iterable

from sqlalchemy import bindparam, literal_column, select

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import (
    job_config_table,
    job_workdir_files_table,
    jobs_table,
    task_attempts_table,
    tasks_table,
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


def _priority_bands_stmt(ids: list[JobName]):
    """Build a recursive CTE that walks the parent_job_id chain for each id.

    Walks parent_job_id chain until a non-UNSPECIFIED priority_band is found.
    Inputs whose entire ancestor chain is UNSPECIFIED are absent from the result;
    the caller substitutes INTERACTIVE for those.

    The CTE tracks four columns per row:
      input_id     — original job ID we are resolving
      current_id   — current ancestor being examined
      current_band — priority_band at that ancestor
      parent_id    — next ancestor to examine (None if root)
    """
    j = jobs_table.alias("j")
    jc = job_config_table.alias("jc")
    # Base: one row per input job ID
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
    # Recursive: step to next ancestor when current_band is still UNSPECIFIED (0)
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
    """Build a recursive CTE that returns 1 if the subtree under ``job_id`` has unfinished attempts."""
    # Recursive CTE: true if any task in the subtree rooted at job_id has a
    # worker-bound unfinished attempt.
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
