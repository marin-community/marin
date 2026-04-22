# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile ``tasks.current_attempt_id`` with ``task_attempts`` rows.

Prior to the fix in ``transitions.py`` (``fail_worker`` reservation-holder
branch), a worker failure on a reservation-holder task reset
``current_attempt_id`` to ``-1`` while leaving the task's previous attempt
rows (``attempt_id = 0..N-1``) intact. The next scheduling cycle computed
``attempt_id = current_attempt_id + 1 = 0`` and tried to insert a
``task_attempts`` row that already existed, raising
``sqlite3.IntegrityError: UNIQUE constraint failed`` and killing the
scheduling thread.

This migration heals any task where ``current_attempt_id`` is behind the
largest ``attempt_id`` present in ``task_attempts``. After the update the
invariant "next assignment's attempt_id (``current_attempt_id + 1``) is
never already taken" holds again, so the scheduler can recover on restart.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE tasks
        SET current_attempt_id = (
            SELECT MAX(attempt_id)
            FROM task_attempts
            WHERE task_attempts.task_id = tasks.task_id
        )
        WHERE EXISTS (
            SELECT 1 FROM task_attempts
            WHERE task_attempts.task_id = tasks.task_id
              AND task_attempts.attempt_id > tasks.current_attempt_id
        )
        """
    )
