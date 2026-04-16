# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # Purge endpoints whose task is already in a terminal state.
    # These accumulated because register_endpoint did not check task state
    # before inserting, so endpoints registered after a task went terminal
    # were never cleaned up.
    # Terminal states: SUCCEEDED=4, FAILED=5, KILLED=6, WORKER_FAILED=7, UNSCHEDULABLE=8
    conn.execute(
        "DELETE FROM endpoints WHERE task_id IN (" "  SELECT task_id FROM tasks WHERE state IN (4, 5, 6, 7, 8)" ")"
    )
