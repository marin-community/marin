# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # Cover _live_user_stats GROUP BY user_id, state
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_state ON jobs(user_id, state)")

    # Cover _building_counts: start from the small set of BUILDING/ASSIGNED
    # tasks instead of scanning all attempts per worker.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state)")

    # Index for endpoint lookups by job_id.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_endpoints_job_id ON endpoints(job_id)")

    # Purge stale endpoints for terminal jobs. These accumulated due to a bug
    # where endpoint cleanup was gated on worker_id being non-NULL, so tasks
    # that went terminal without a worker assignment leaked their endpoints.
    # Terminal states: SUCCEEDED=4, FAILED=5, KILLED=6, WORKER_FAILED=7, UNSCHEDULABLE=8
    conn.execute(
        "DELETE FROM endpoints WHERE job_id IN (" "  SELECT job_id FROM jobs WHERE state IN (4, 5, 6, 7, 8)" ")"
    )
