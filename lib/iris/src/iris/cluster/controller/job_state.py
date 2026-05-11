# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure derivation of ``JobState`` from task-state counts.

``JobState`` is a deterministic function of the per-task-state counts on a
job plus a small ``basis`` (``max_task_failures`` from ``job_config`` and
``started_at_ms`` from ``jobs``). Historically the result was recomputed
inline at eight call sites in ``transitions.py``; this module collapses the
formula into a single pure function and a SQL view (``jobs_with_state``)
that mirrors it, so dashboards and the controller agree on what a job's
state means.

The formula stays the controller's single source of truth: ``JobStore``
calls ``compute_job_state`` to read a derived value without writing, and
``ControllerTransitions._recompute_job_state`` calls the same function
before persisting the result back onto ``jobs.state`` for fast columnar
queries / indexes.
"""

from __future__ import annotations

from iris.cluster.types import TERMINAL_TASK_STATES
from iris.rpc import job_pb2


def compute_job_state(
    counts: dict[int, int],
    *,
    max_task_failures: int,
    started_at_ms: int | None,
) -> int:
    """Derive ``JobState`` from task-state counts and job metadata.

    ``counts`` maps ``TaskState`` integer to the number of tasks of the
    owning job currently in that state. ``max_task_failures`` is the
    failure-tolerance budget from ``job_config``; ``started_at_ms`` is the
    job's first-RUNNING timestamp, used so retries that put tasks back into
    PENDING still keep the job in RUNNING.

    Returns one of the ``JOB_STATE_*`` constants from ``job_pb2``. The
    returned value is purely a function of the inputs — no DB access, no
    side effects.

    Caller is responsible for terminal-stickiness: under the derived model
    a job goes terminal when its last running task does, and the cascade
    that fires from that transition kills the cohort; this function may
    therefore be called freely without first-checking the stored state.
    """
    total = sum(counts.values())
    if total > 0 and counts.get(job_pb2.TASK_STATE_SUCCEEDED, 0) == total:
        return job_pb2.JOB_STATE_SUCCEEDED
    if counts.get(job_pb2.TASK_STATE_FAILED, 0) > max_task_failures:
        return job_pb2.JOB_STATE_FAILED
    if counts.get(job_pb2.TASK_STATE_UNSCHEDULABLE, 0) > 0:
        return job_pb2.JOB_STATE_UNSCHEDULABLE
    if counts.get(job_pb2.TASK_STATE_KILLED, 0) > 0:
        return job_pb2.JOB_STATE_KILLED
    if (
        total > 0
        and (counts.get(job_pb2.TASK_STATE_WORKER_FAILED, 0) + counts.get(job_pb2.TASK_STATE_PREEMPTED, 0)) > 0
        and all(s in TERMINAL_TASK_STATES for s in counts)
    ):
        return job_pb2.JOB_STATE_WORKER_FAILED
    if (
        counts.get(job_pb2.TASK_STATE_ASSIGNED, 0) > 0
        or counts.get(job_pb2.TASK_STATE_BUILDING, 0) > 0
        or counts.get(job_pb2.TASK_STATE_RUNNING, 0) > 0
    ):
        return job_pb2.JOB_STATE_RUNNING
    if started_at_ms is not None:
        # Retries put tasks back into PENDING; keep the job running once it has started.
        return job_pb2.JOB_STATE_RUNNING
    if total > 0:
        return job_pb2.JOB_STATE_PENDING
    # No tasks (yet): mirror the legacy behaviour of leaving the stored state
    # untouched. Callers in transitions.py compare against the prior state
    # before writing; the view path returns the column directly when total=0
    # via COALESCE, so the choice here is only observable in the (parity)
    # tests, where the deterministic answer is PENDING.
    return job_pb2.JOB_STATE_PENDING


# ---------------------------------------------------------------------------
# SQL view: ``jobs_with_state``
#
# A read-only view that exposes every column of ``jobs`` and replaces the
# stored ``state`` column with a derived value computed from the per-job
# ``tasks`` aggregate, using the same formula as ``compute_job_state``.
#
# The view is *not yet* the authoritative source — the column ``jobs.state``
# is still maintained by ``ControllerTransitions._recompute_job_state``. The
# view exists so callers that want a derived answer (CLI / dashboard parity
# checks, future Phase E ``read_state`` consumers) can avoid the staleness
# window between a task transition and the recompute write. A direct
# ``jobs.state`` read remains the fast path for indexed filters.
#
# Performance: the view aggregates ``tasks`` per job using the existing
# ``idx_tasks_job_state`` index. For point lookups (``WHERE job_id = ?``)
# this is a single index range scan; for unfiltered scans it materializes
# one ``GROUP BY job_id`` per query, which is acceptable for CLI / dashboard
# usage but should not be put on the per-tick scheduling hot path.
# ---------------------------------------------------------------------------


# Aggregated per-job task counts. Materialized inline as a subquery so SQLite
# can push job_id predicates into the aggregation when the view is filtered.
_JOB_TASK_COUNTS_SUBQUERY = """
SELECT
    job_id,
    SUM(CASE WHEN state = {pending}        THEN 1 ELSE 0 END) AS cnt_pending,
    SUM(CASE WHEN state = {building}       THEN 1 ELSE 0 END) AS cnt_building,
    SUM(CASE WHEN state = {running}        THEN 1 ELSE 0 END) AS cnt_running,
    SUM(CASE WHEN state = {succeeded}      THEN 1 ELSE 0 END) AS cnt_succeeded,
    SUM(CASE WHEN state = {failed}         THEN 1 ELSE 0 END) AS cnt_failed,
    SUM(CASE WHEN state = {killed}         THEN 1 ELSE 0 END) AS cnt_killed,
    SUM(CASE WHEN state = {worker_failed}  THEN 1 ELSE 0 END) AS cnt_worker_failed,
    SUM(CASE WHEN state = {unschedulable}  THEN 1 ELSE 0 END) AS cnt_unschedulable,
    SUM(CASE WHEN state = {assigned}       THEN 1 ELSE 0 END) AS cnt_assigned,
    SUM(CASE WHEN state = {preempted}      THEN 1 ELSE 0 END) AS cnt_preempted,
    COUNT(*) AS cnt_total,
    SUM(CASE WHEN state NOT IN ({terminal_csv}) THEN 1 ELSE 0 END) AS cnt_non_terminal
FROM tasks
GROUP BY job_id
""".format(
    pending=job_pb2.TASK_STATE_PENDING,
    building=job_pb2.TASK_STATE_BUILDING,
    running=job_pb2.TASK_STATE_RUNNING,
    succeeded=job_pb2.TASK_STATE_SUCCEEDED,
    failed=job_pb2.TASK_STATE_FAILED,
    killed=job_pb2.TASK_STATE_KILLED,
    worker_failed=job_pb2.TASK_STATE_WORKER_FAILED,
    unschedulable=job_pb2.TASK_STATE_UNSCHEDULABLE,
    assigned=job_pb2.TASK_STATE_ASSIGNED,
    preempted=job_pb2.TASK_STATE_PREEMPTED,
    terminal_csv=",".join(str(s) for s in sorted(TERMINAL_TASK_STATES)),
)


# Translation of ``compute_job_state`` into SQL. The CASE arms are evaluated
# in the same order as the Python function; the column projection below
# selects every ``jobs`` column except ``state`` and replaces it with the
# derived value.
_DERIVED_STATE_CASE = f"""
CASE
    WHEN COALESCE(c.cnt_total, 0) > 0
         AND c.cnt_succeeded = c.cnt_total
        THEN {job_pb2.JOB_STATE_SUCCEEDED}
    WHEN COALESCE(c.cnt_failed, 0) > jc.max_task_failures
        THEN {job_pb2.JOB_STATE_FAILED}
    WHEN COALESCE(c.cnt_unschedulable, 0) > 0
        THEN {job_pb2.JOB_STATE_UNSCHEDULABLE}
    WHEN COALESCE(c.cnt_killed, 0) > 0
        THEN {job_pb2.JOB_STATE_KILLED}
    WHEN COALESCE(c.cnt_total, 0) > 0
         AND (COALESCE(c.cnt_worker_failed, 0) + COALESCE(c.cnt_preempted, 0)) > 0
         AND COALESCE(c.cnt_non_terminal, 0) = 0
        THEN {job_pb2.JOB_STATE_WORKER_FAILED}
    WHEN COALESCE(c.cnt_assigned, 0) > 0
         OR COALESCE(c.cnt_building, 0) > 0
         OR COALESCE(c.cnt_running, 0) > 0
        THEN {job_pb2.JOB_STATE_RUNNING}
    WHEN j.started_at_ms IS NOT NULL
        THEN {job_pb2.JOB_STATE_RUNNING}
    WHEN COALESCE(c.cnt_total, 0) > 0
        THEN {job_pb2.JOB_STATE_PENDING}
    ELSE {job_pb2.JOB_STATE_PENDING}
END
"""


# Columns from ``jobs`` excluding ``state`` — kept in the same physical order
# as the table definition in ``schema.py`` so callers that select ``*`` from
# the view see a stable column layout.
_JOBS_COLUMNS_WITHOUT_STATE = (
    "job_id",
    "user_id",
    "parent_job_id",
    "root_job_id",
    "depth",
    "submitted_at_ms",
    "root_submitted_at_ms",
    "started_at_ms",
    "finished_at_ms",
    "scheduling_deadline_epoch_ms",
    "error",
    "exit_code",
    "num_tasks",
    "is_reservation_holder",
    "name",
    "has_reservation",
)


JOBS_WITH_STATE_VIEW_SQL = (
    "CREATE VIEW IF NOT EXISTS jobs_with_state AS\n"
    "SELECT\n    "
    + ",\n    ".join(f"j.{col}" for col in _JOBS_COLUMNS_WITHOUT_STATE)
    + f",\n    {_DERIVED_STATE_CASE.strip()} AS state\n"
    "FROM jobs j\n"
    "JOIN job_config jc ON jc.job_id = j.job_id\n"
    f"LEFT JOIN ({_JOB_TASK_COUNTS_SUBQUERY.strip()}) c ON c.job_id = j.job_id"
)


DROP_JOBS_WITH_STATE_VIEW_SQL = "DROP VIEW IF EXISTS jobs_with_state"
