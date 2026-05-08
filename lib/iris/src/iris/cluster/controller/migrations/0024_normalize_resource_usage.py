# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def migrate(conn: sqlite3.Connection) -> None:
    columns_to_add = (
        ("resource_usage_memory_mb", "INTEGER"),
        ("resource_usage_disk_mb", "INTEGER"),
        ("resource_usage_cpu_millicores", "INTEGER"),
        ("resource_usage_memory_peak_mb", "INTEGER"),
        ("resource_usage_process_count", "INTEGER"),
    )
    for column, ddl in columns_to_add:
        if not _has_column(conn, "tasks", column):
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} {ddl}")

    # Backfill from existing BLOB column (only needed for upgrades, not fresh DBs).
    if _has_column(conn, "tasks", "resource_usage_proto"):
        from iris.rpc import job_pb2

        rows = conn.execute(
            "SELECT task_id, resource_usage_proto FROM tasks WHERE resource_usage_proto IS NOT NULL"
        ).fetchall()
        for task_id, blob in rows:
            usage = job_pb2.ResourceUsage()
            usage.ParseFromString(blob)
            conn.execute(
                "UPDATE tasks SET resource_usage_memory_mb = ?, resource_usage_disk_mb = ?, "
                "resource_usage_cpu_millicores = ?, resource_usage_memory_peak_mb = ?, "
                "resource_usage_process_count = ? WHERE task_id = ?",
                (
                    usage.memory_mb or None,
                    usage.disk_mb or None,
                    usage.cpu_millicores or None,
                    usage.memory_peak_mb or None,
                    usage.process_count or None,
                    task_id,
                ),
            )
        conn.execute("ALTER TABLE tasks DROP COLUMN resource_usage_proto")
