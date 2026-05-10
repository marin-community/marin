# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the cached committed-resource columns on ``workers``.

Worker resource usage is now derived from unfinished worker-bound
``task_attempts`` (``worker_id IS NOT NULL AND finished_at_ms IS NULL``)
via ``TaskAttemptStore.resource_usage_by_worker``. The ``committed_*``
counters were a redundant cache that drifted out of sync with reality on
producer/heartbeat path splits — see #5470 — so they are removed.

SQLite supports ``ALTER TABLE ... DROP COLUMN`` since 3.35.
"""

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


_COLUMNS_TO_DROP = (
    "committed_cpu_millicores",
    "committed_mem_bytes",
    "committed_gpu",
    "committed_tpu",
)


def migrate(conn: sqlite3.Connection) -> None:
    for col in _COLUMNS_TO_DROP:
        if _has_column(conn, "workers", col):
            conn.execute(f"ALTER TABLE workers DROP COLUMN {col}")
