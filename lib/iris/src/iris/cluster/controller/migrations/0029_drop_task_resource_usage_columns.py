# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop denormalized resource_usage columns from tasks table.

These columns cached the latest task_resource_history row on every heartbeat.
The detail view now reads directly from task_resource_history instead.
"""

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


_COLUMNS_TO_DROP = (
    "resource_usage_memory_mb",
    "resource_usage_disk_mb",
    "resource_usage_cpu_millicores",
    "resource_usage_memory_peak_mb",
    "resource_usage_process_count",
)


def migrate(conn: sqlite3.Connection) -> None:
    for col in _COLUMNS_TO_DROP:
        if _has_column(conn, "tasks", col):
            conn.execute(f"ALTER TABLE tasks DROP COLUMN {col}")
