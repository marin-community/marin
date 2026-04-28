# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finalize orphan task_attempts rows left active by the cancel_job bug.

Pre-fix versions of ``ControllerTransitions.cancel_job`` updated tasks /
jobs / workers but never finalized the corresponding ``task_attempts``
row. The current attempt stayed at state RUNNING/BUILDING/ASSIGNED with
``finished_at_ms = NULL`` long after the task itself went terminal,
which made the dashboard report killed tasks as still occupying their
old worker — surfacing as bogus "two TPU jobs on one worker"
co-scheduling violations even when committed_* accounting was correct.

This migration heals two classes of stale rows so the post-restart
dashboard matches reality:

1. **Task is terminal but attempt is still active.** Mark the attempt
   PREEMPTED and stamp ``finished_at_ms``.
2. **Attempt is superseded** (``attempt_id != tasks.current_attempt_id``)
   but still active. Same fix; the controller already routes new heartbeats
   for the live attempt and ignores stale attempt RPCs via the
   ``attempt.state in TERMINAL_TASK_STATES`` short-circuit in
   ``apply_state_updates``.

Idempotent — the WHERE clause filters by current state so rerunning is
a no-op once applied.
"""

import sqlite3

# Mirror of iris.cluster.controller.db.ACTIVE_TASK_STATES /
# TERMINAL_TASK_STATES. Inlined to keep migrations free of imports from the
# evolving controller package, since old migrations must keep working as
# the codebase changes.
_ACTIVE = (2, 3, 9)  # BUILDING, RUNNING, ASSIGNED
_TERMINAL_TASK = (4, 5, 6, 7, 8)  # SUCCEEDED, FAILED, KILLED, WORKER_FAILED, UNSCHEDULABLE
_TASK_STATE_PREEMPTED = 10
_RECONCILE_REASON = "Reconciled: orphan attempt left active by cancel_job"


def migrate(conn: sqlite3.Connection) -> None:
    active_placeholders = ",".join("?" * len(_ACTIVE))
    terminal_placeholders = ",".join("?" * len(_TERMINAL_TASK))
    now_ms = int(conn.execute("SELECT CAST(strftime('%s','now') AS INTEGER) * 1000").fetchone()[0])
    conn.execute(
        f"""
        UPDATE task_attempts
        SET state = ?,
            finished_at_ms = COALESCE(finished_at_ms, ?),
            error = COALESCE(error, ?)
        WHERE state IN ({active_placeholders})
          AND task_id IN (
            SELECT ta.task_id FROM task_attempts ta
            JOIN tasks t ON t.task_id = ta.task_id
            WHERE ta.state IN ({active_placeholders})
              AND (
                t.state IN ({terminal_placeholders})
                OR ta.attempt_id != t.current_attempt_id
              )
          )
          AND (
            task_id IN (SELECT task_id FROM tasks WHERE state IN ({terminal_placeholders}))
            OR attempt_id != (
              SELECT current_attempt_id FROM tasks WHERE tasks.task_id = task_attempts.task_id
            )
          )
        """,
        (
            _TASK_STATE_PREEMPTED,
            now_ms,
            _RECONCILE_REASON,
            *_ACTIVE,
            *_ACTIVE,
            *_TERMINAL_TASK,
            *_TERMINAL_TASK,
        ),
    )
