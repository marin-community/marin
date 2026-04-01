# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ON DELETE SET NULL to foreign keys referencing workers(worker_id).

SQLite doesn't support ALTER CONSTRAINT, so we recreate the affected tables.
- task_attempts.worker_id: SET NULL on worker delete
- tasks.current_worker_id: SET NULL on worker delete
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # --- task_attempts: recreate with ON DELETE SET NULL on worker_id ---
    conn.execute(
        """
        CREATE TABLE task_attempts_new (
            task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
            attempt_id INTEGER NOT NULL,
            worker_id TEXT REFERENCES workers(worker_id) ON DELETE SET NULL,
            state INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            started_at_ms INTEGER,
            finished_at_ms INTEGER,
            exit_code INTEGER,
            error TEXT,
            PRIMARY KEY (task_id, attempt_id)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO task_attempts_new
        SELECT task_id, attempt_id, worker_id, state, created_at_ms,
               started_at_ms, finished_at_ms, exit_code, error
        FROM task_attempts
        """
    )
    conn.execute("DROP TABLE task_attempts")
    conn.execute("ALTER TABLE task_attempts_new RENAME TO task_attempts")

    # Recreate the composite index from 0007 (the single-column one was dropped in 0015)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_attempts_worker_task " "ON task_attempts(worker_id, task_id, attempt_id)"
    )

    # Recreate the trigger from 0001_init (dropped when the table was rebuilt)
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_task_attempt_active_worker
        BEFORE INSERT ON task_attempts
        FOR EACH ROW
        WHEN NEW.worker_id IS NOT NULL
        BEGIN
          SELECT
            CASE
              WHEN NOT EXISTS(
                SELECT 1 FROM workers w
                WHERE w.worker_id = NEW.worker_id
                  AND w.active = 1
                  AND w.healthy = 1
              )
              THEN RAISE(ABORT, 'task attempt worker must be active and healthy')
            END;
        END;
        """
    )
