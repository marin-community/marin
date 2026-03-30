# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def migrate(conn: sqlite3.Connection) -> None:
    if not _has_column(conn, "tasks", "current_worker_id"):
        conn.execute("ALTER TABLE tasks ADD COLUMN current_worker_id TEXT REFERENCES workers(worker_id)")
    if not _has_column(conn, "tasks", "current_worker_address"):
        conn.execute("ALTER TABLE tasks ADD COLUMN current_worker_address TEXT")

    conn.execute(
        """
        UPDATE tasks
        SET current_worker_id = (
            SELECT a.worker_id
            FROM task_attempts a
            WHERE a.task_id = tasks.task_id
              AND a.attempt_id = tasks.current_attempt_id
        )
        WHERE current_attempt_id >= 0
          AND current_worker_id IS NULL
        """
    )
