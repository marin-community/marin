# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable SQL mutation helpers for controller transactions.

Each function takes a TransactionCursor (or compatible execute interface) as its
first argument so transaction boundaries remain explicit in the caller.
"""

from iris.cluster.controller.db import TransactionCursor


def delete_task_endpoints(cur: TransactionCursor, task_id: str) -> None:
    """Remove all registered endpoints for a task."""
    cur.execute("DELETE FROM endpoints WHERE task_id = ?", (task_id,))


def enqueue_run_dispatch(
    cur: TransactionCursor,
    worker_id: str,
    payload_proto: bytes,
    now_ms: int,
) -> None:
    """Queue a 'run' dispatch entry for delivery on the next heartbeat."""
    cur.execute(
        "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
        "VALUES (?, 'run', ?, NULL, ?)",
        (worker_id, payload_proto, now_ms),
    )


def enqueue_kill_dispatch(
    cur: TransactionCursor,
    worker_id: str | None,
    task_id: str,
    now_ms: int,
) -> None:
    """Queue a 'kill' dispatch entry for delivery on the next heartbeat."""
    cur.execute(
        "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
        "VALUES (?, 'kill', NULL, ?, ?)",
        (worker_id, task_id, now_ms),
    )


def insert_task_attempt(
    cur: TransactionCursor,
    task_id: str,
    attempt_id: int,
    worker_id: str | None,
    state: int,
    now_ms: int,
) -> None:
    """Record a new task attempt row."""
    cur.execute(
        "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) " "VALUES (?, ?, ?, ?, ?)",
        (task_id, attempt_id, worker_id, state, now_ms),
    )
