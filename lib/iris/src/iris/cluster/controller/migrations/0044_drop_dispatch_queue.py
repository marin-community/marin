# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the ``dispatch_queue`` table.

Worker-bound dispatch never used this table once the polling reconcile
loop landed (assignments are derived from ``tasks.state = ASSIGNED`` and
``task_attempts.worker_id``). The K8s direct provider stopped using it
as well: pod creation is driven by the same ``tasks``/``task_attempts``
snapshot, and pod kills are derived from a desired-vs-actual pod diff
inside the provider's sync loop.

If a controller is upgraded with rows still in flight, log a warning
before dropping; in steady state the table is empty. SQLite supports
``DROP TABLE`` unconditionally.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def migrate(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "dispatch_queue"):
        return
    row = conn.execute("SELECT COUNT(*) FROM dispatch_queue").fetchone()
    pending = int(row[0]) if row is not None else 0
    if pending:
        logger.warning(
            "dispatch_queue still had %d rows at migration time; dropping anyway. "
            "These were buffered K8s kill entries; the new sync loop derives kills "
            "from a pod-diff against the active task set, so the residual rows are "
            "safe to discard.",
            pending,
        )
    conn.execute("DROP TABLE dispatch_queue")
