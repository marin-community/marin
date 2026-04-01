# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # ADD COLUMN is not idempotent in SQLite (no IF NOT EXISTS), so check first.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(task_profiles)").fetchall()}
    if "profile_kind" not in cols:
        conn.execute("ALTER TABLE task_profiles ADD COLUMN profile_kind TEXT NOT NULL DEFAULT 'cpu'")

    conn.executescript(
        """
-- Recreate the index to include profile_kind for efficient per-kind queries.
DROP INDEX IF EXISTS idx_task_profiles_task;
CREATE INDEX IF NOT EXISTS idx_task_profiles_task_kind ON task_profiles(task_id, profile_kind, id DESC);

-- Replace the cap trigger: keep 10 per (task_id, profile_kind) instead of per task_id.
DROP TRIGGER IF EXISTS trg_task_profiles_cap;
CREATE TRIGGER IF NOT EXISTS trg_task_profiles_cap
AFTER INSERT ON task_profiles
BEGIN
  DELETE FROM task_profiles
   WHERE task_id = NEW.task_id
     AND profile_kind = NEW.profile_kind
     AND id NOT IN (
       SELECT id FROM task_profiles
        WHERE task_id = NEW.task_id
          AND profile_kind = NEW.profile_kind
        ORDER BY id DESC
        LIMIT 10
     );
END;
"""
    )
