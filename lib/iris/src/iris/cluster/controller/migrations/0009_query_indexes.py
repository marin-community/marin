# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # Cover _live_user_stats GROUP BY user_id, state
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_state ON jobs(user_id, state)")

    # Cover _building_counts: start from the small set of BUILDING/ASSIGNED
    # tasks instead of scanning all attempts per worker.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state)")
