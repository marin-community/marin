# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # idx_task_attempts_worker (worker_id) is fully superseded by
    # idx_task_attempts_worker_task (worker_id, task_id, attempt_id) from migration 0007.
    # The narrower index wastes space and never gets chosen by the planner.
    conn.execute("DROP INDEX IF EXISTS idx_task_attempts_worker")
