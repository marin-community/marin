# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
CREATE TABLE IF NOT EXISTS task_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    profile_data BLOB NOT NULL,
    captured_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_profiles_task ON task_profiles(task_id, id DESC);

CREATE TRIGGER IF NOT EXISTS trg_task_profiles_cap
AFTER INSERT ON task_profiles
BEGIN
  DELETE FROM task_profiles
   WHERE task_id = NEW.task_id
     AND id NOT IN (
       SELECT id FROM task_profiles
        WHERE task_id = NEW.task_id
        ORDER BY id DESC
        LIMIT 10
     );
END;
"""
    )
