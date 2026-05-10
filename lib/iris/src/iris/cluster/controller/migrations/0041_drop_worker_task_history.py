# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the dead `worker_task_history` SQLite table.

Per-task / per-worker stats now live in finelog's `iris.worker` and
`iris.task` Tables. The controller-side `worker_task_history` table was
appended on every task assign and pruned on a 60s cadence, but no
production code path read from it; only a regression test and a benchmark
script ever ran SELECT against it.

This migration removes the table and its indices. Migration 0033
(adding ON DELETE CASCADE on `task_id`) is now superseded but left in
place for replay determinism.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_worker_task_history_worker")
    conn.execute("DROP INDEX IF EXISTS idx_worker_task_history_task")
    conn.execute("DROP TABLE IF EXISTS worker_task_history")
