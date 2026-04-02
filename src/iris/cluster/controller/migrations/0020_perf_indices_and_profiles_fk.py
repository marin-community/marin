# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Performance indices and task_profiles cap trigger.

1. Partial index on tasks.current_worker_id (skip NULLs).
2. Composite index on task_profiles for efficient per-task lookups.
3. Trigger to cap profiles per (task_id, profile_kind) at 10 rows.

Orphan task_profiles (referencing deleted tasks) are cleaned up incrementally
by prune_old_data rather than in a blocking migration.
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

    # --- 2. Index for task_profiles lookups ---
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_task_profiles_task_kind
        ON task_profiles(task_id, profile_kind, id DESC)
        """
    )

    # --- 3. Cap trigger ---
    conn.execute(
        """
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
        END
        """
    )
