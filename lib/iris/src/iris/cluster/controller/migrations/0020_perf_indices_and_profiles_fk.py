# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Performance indices, task_profiles FK cascade, and incremental auto_vacuum.

1. Partial index on tasks.current_worker_id (skip NULLs).
2. Recreate task_profiles with ON DELETE CASCADE FK to tasks(task_id),
   dropping orphan rows that reference deleted tasks.
3. Set PRAGMA auto_vacuum = INCREMENTAL (takes effect on next VACUUM).
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # --- 1. Partial index on current_worker_id ---
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tasks_current_worker
        ON tasks(current_worker_id)
        WHERE current_worker_id IS NOT NULL
        """
    )

    # --- 2. Recreate task_profiles with FK cascade ---
    conn.execute(
        """
        CREATE TABLE task_profiles_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
            profile_data BLOB NOT NULL,
            captured_at_ms INTEGER NOT NULL,
            profile_kind TEXT NOT NULL DEFAULT 'cpu'
        )
        """
    )
    # Only copy rows whose task_id still exists in tasks — skip orphans.
    conn.execute(
        """
        INSERT INTO task_profiles_new (id, task_id, profile_data, captured_at_ms, profile_kind)
        SELECT p.id, p.task_id, p.profile_data, p.captured_at_ms, p.profile_kind
        FROM task_profiles p
        WHERE EXISTS (SELECT 1 FROM tasks t WHERE t.task_id = p.task_id)
        """
    )
    conn.execute("DROP TABLE task_profiles")
    conn.execute("ALTER TABLE task_profiles_new RENAME TO task_profiles")

    conn.execute(
        """
        CREATE INDEX idx_task_profiles_task_kind
        ON task_profiles(task_id, profile_kind, id DESC)
        """
    )
    conn.execute(
        """
        CREATE TRIGGER trg_task_profiles_cap
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
        END
        """
    )

    # --- 3. Incremental auto_vacuum ---
    conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
