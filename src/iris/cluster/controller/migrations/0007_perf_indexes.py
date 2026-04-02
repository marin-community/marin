# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
CREATE INDEX IF NOT EXISTS idx_task_attempts_worker_task
    ON task_attempts(worker_id, task_id, attempt_id);

CREATE INDEX IF NOT EXISTS idx_jobs_state
    ON jobs(state, submitted_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_jobs_depth_state
    ON jobs(depth, state, submitted_at_ms DESC);
"""
    )
