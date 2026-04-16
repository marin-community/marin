# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add ON DELETE CASCADE FK on ``worker_task_history.task_id -> tasks(task_id)``.

Prior to this migration ``worker_task_history`` had no FK on ``task_id``, so
``remove_finished_job`` (which cascades from ``jobs`` -> ``tasks`` ->
``task_attempts``) left ``worker_task_history`` rows orphaned. Resubmitting a
job with the same ``job_id`` produced fresh ``tasks`` rows that silently
re-attached to the stale history — the fingerprint that was mis-diagnosed as
the reservation-holder reset branch misfiring during the 2026-04-16 outage.

SQLite can't add a FK via ALTER TABLE, so follow the standard
create-new/copy/drop/rename dance, dropping orphan rows in the copy (that's
the whole point; otherwise the FK would be violated the moment we turn it on)
and recreating every index that existed on the old table.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE worker_task_history_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
            task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
            assigned_at_ms INTEGER NOT NULL
        )
        """
    )
    # Deliberately drop orphan rows (task_id no longer in tasks). Preserving
    # them would violate the new FK constraint the instant it's enabled.
    conn.execute(
        """
        INSERT INTO worker_task_history_new (id, worker_id, task_id, assigned_at_ms)
        SELECT id, worker_id, task_id, assigned_at_ms
        FROM worker_task_history
        WHERE task_id IN (SELECT task_id FROM tasks)
        """
    )
    conn.execute("DROP TABLE worker_task_history")
    conn.execute("ALTER TABLE worker_task_history_new RENAME TO worker_task_history")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_worker_task_history_worker "
        "ON worker_task_history(worker_id, assigned_at_ms DESC)"
    )
