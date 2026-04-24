# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_stats_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
            items_processed INTEGER NOT NULL DEFAULT 0,
            bytes_processed INTEGER NOT NULL DEFAULT 0,
            timestamp_ms INTEGER NOT NULL
        )
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_stats_history_task" " ON task_stats_history(task_id, id DESC)")

    existing = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "status_message" not in existing:
        conn.execute("ALTER TABLE tasks ADD COLUMN status_message TEXT NOT NULL DEFAULT ''")
