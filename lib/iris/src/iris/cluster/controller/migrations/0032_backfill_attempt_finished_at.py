# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill ``task_attempts.finished_at_ms`` on terminal rows that were left NULL.

Prior to the fix in ``transitions.py``, a terminal attempt (e.g. FAILED) whose
parent task was rolled back to PENDING for retry had its ``finished_at_ms``
dropped to NULL alongside the task. This left orphaned dead attempts in the DB
with no completion timestamp, which in turn confused UI rendering that treats
``finished_at_ms IS NULL`` as "not done yet".

Best-effort backfill: use the ``created_at_ms`` of the next attempt on the same
task as the completion time, since the controller created the next attempt in
the same transaction that wrote the (dropped) terminal_ms. If there is no
next attempt, fall back to ``started_at_ms`` and then ``created_at_ms``.

Terminal states (see ``iris.cluster.types.TERMINAL_TASK_STATES``):
  4 SUCCEEDED, 5 FAILED, 6 KILLED, 7 WORKER_FAILED, 8 UNSCHEDULABLE, 10 PREEMPTED
"""

import sqlite3

_TERMINAL_STATES = (4, 5, 6, 7, 8, 10)


def migrate(conn: sqlite3.Connection) -> None:
    placeholders = ",".join("?" for _ in _TERMINAL_STATES)
    conn.execute(
        f"""
        WITH next_attempt AS (
            SELECT
                task_id,
                attempt_id,
                LEAD(created_at_ms) OVER (
                    PARTITION BY task_id ORDER BY attempt_id
                ) AS next_created_at_ms
            FROM task_attempts
        )
        UPDATE task_attempts
        SET finished_at_ms = COALESCE(
            (SELECT n.next_created_at_ms
             FROM next_attempt n
             WHERE n.task_id = task_attempts.task_id
               AND n.attempt_id = task_attempts.attempt_id),
            task_attempts.started_at_ms,
            task_attempts.created_at_ms
        )
        WHERE finished_at_ms IS NULL
          AND state IN ({placeholders})
        """,
        _TERMINAL_STATES,
    )
