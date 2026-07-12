# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the denormalized ``tasks.failure_count`` / ``tasks.preemption_count``.

Both counters are now derived from the task's ``task_attempts`` rows (see
``iris.cluster.controller.attempt_counts``) and served from an in-memory cache,
so the columns are a removed second source of truth. Their two covering indexes
(``idx_tasks_job_failures``, ``idx_tasks_job_state_counts``) go with them —
``idx_tasks_job_state`` already covers ``(job_id, state)``. A new
``idx_task_attempts_task_state`` on ``(task_id, state, started_at_ms)`` makes the
per-task / per-job derivation aggregates index-only.

Indexes are dropped before the columns because SQLite refuses to drop a column
referenced by an index. Idempotent: each step is guarded / uses IF [NOT] EXISTS,
so a crash mid-run is safe to retry.
"""


def _has_column(raw_conn, table: str, column: str) -> bool:
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == column for row in raw_conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(raw_conn) -> None:
    raw_conn.execute("DROP INDEX IF EXISTS idx_tasks_job_failures")
    raw_conn.execute("DROP INDEX IF EXISTS idx_tasks_job_state_counts")
    if _has_column(raw_conn, "tasks", "failure_count"):
        raw_conn.execute("ALTER TABLE tasks DROP COLUMN failure_count")
    if _has_column(raw_conn, "tasks", "preemption_count"):
        raw_conn.execute("ALTER TABLE tasks DROP COLUMN preemption_count")
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_attempts_task_state " "ON task_attempts (task_id, state, started_at_ms)"
    )
