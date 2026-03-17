# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # Speed up _descendants_for_roots which queries WHERE root_job_id IN (...) AND depth > 1.
    # Without this, the query falls back to scanning idx_jobs_depth_state.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_root_depth ON jobs(root_job_id, depth)")
