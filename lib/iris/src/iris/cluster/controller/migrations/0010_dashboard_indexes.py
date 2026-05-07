# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # Speed up _descendants_for_roots which queries WHERE root_job_id IN (...) AND depth > 1.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_root_depth ON jobs(root_job_id, depth)")

    # Speed up _building_counts and running_tasks_by_worker: both JOIN tasks to task_attempts
    # on (task_id, current_attempt_id) with a state filter. Leading with state lets SQLite
    # narrow to the few BUILDING/ASSIGNED/RUNNING tasks before hitting the join.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_state_attempt " "ON tasks(state, task_id, current_attempt_id, job_id)"
    )

    # Speed up _jobs_paginated (date sort): the existing idx_jobs_depth_state is
    # (depth, state, submitted_at_ms) but an IN clause on state breaks the sort.
    # Leading with (depth, submitted_at_ms) lets SQLite use the index for both
    # the depth=1 filter and the ORDER BY, then post-filter by state.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_depth_submitted " "ON jobs(depth, submitted_at_ms DESC)")

    # Speed up _jobs_paginated (sort by failures/preemptions): the LEFT JOIN tasks
    # aggregates failure_count per job. This covering index lets the join scan
    # (job_id, failure_count, preemption_count) without touching the tasks table.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_job_failures " "ON tasks(job_id, failure_count, preemption_count)"
    )

    # Historical: this also created idx_worker_resource_history_ts on
    # worker_resource_history. The table is dropped in 0040 and no longer
    # appears in MAIN_TABLES, so a fresh DB never has the table here. The
    # index drop is handled by 0040 for legacy DBs.
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='worker_resource_history'"
    ).fetchone()
    if has_table:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_worker_resource_history_ts "
            "ON worker_resource_history(worker_id, timestamp_ms DESC)"
        )
