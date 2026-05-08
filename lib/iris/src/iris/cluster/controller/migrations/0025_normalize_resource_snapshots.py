# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


SNAPSHOT_COLUMNS = (
    ("snapshot_host_cpu_percent", "INTEGER"),
    ("snapshot_memory_used_bytes", "INTEGER"),
    ("snapshot_memory_total_bytes", "INTEGER"),
    ("snapshot_disk_used_bytes", "INTEGER"),
    ("snapshot_disk_total_bytes", "INTEGER"),
    ("snapshot_running_task_count", "INTEGER"),
    ("snapshot_total_process_count", "INTEGER"),
    ("snapshot_net_recv_bps", "INTEGER"),
    ("snapshot_net_sent_bps", "INTEGER"),
)


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def migrate(conn: sqlite3.Connection) -> None:
    # Add scalar columns to both tables (when they still exist — 0040 drops
    # both worker_resource_history and the snapshot_* columns on workers, so
    # on a fresh DB this migration is effectively idempotent groundwork).
    for table in ("workers", "worker_resource_history"):
        if not _has_table(conn, table):
            continue
        for column, ddl in SNAPSHOT_COLUMNS:
            if not _has_column(conn, table, column):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    # Backfill only needed for upgrades, not fresh DBs.
    if not _has_column(conn, "workers", "resource_snapshot_proto"):
        return

    from iris.rpc import job_pb2

    rows = conn.execute(
        "SELECT worker_id, resource_snapshot_proto FROM workers WHERE resource_snapshot_proto IS NOT NULL"
    ).fetchall()
    for worker_id, blob in rows:
        snap = job_pb2.WorkerResourceSnapshot()
        snap.ParseFromString(blob)
        conn.execute(
            "UPDATE workers SET "
            "snapshot_host_cpu_percent = ?, snapshot_memory_used_bytes = ?, "
            "snapshot_memory_total_bytes = ?, snapshot_disk_used_bytes = ?, "
            "snapshot_disk_total_bytes = ?, snapshot_running_task_count = ?, "
            "snapshot_total_process_count = ?, snapshot_net_recv_bps = ?, "
            "snapshot_net_sent_bps = ? WHERE worker_id = ?",
            (
                snap.host_cpu_percent or None,
                snap.memory_used_bytes or None,
                snap.memory_total_bytes or None,
                snap.disk_used_bytes or None,
                snap.disk_total_bytes or None,
                snap.running_task_count or None,
                snap.total_process_count or None,
                snap.net_recv_bps or None,
                snap.net_sent_bps or None,
                worker_id,
            ),
        )

    # Backfill worker_resource_history.snapshot_proto
    rows = conn.execute(
        "SELECT id, snapshot_proto FROM worker_resource_history WHERE snapshot_proto IS NOT NULL"
    ).fetchall()
    for row_id, blob in rows:
        snap = job_pb2.WorkerResourceSnapshot()
        snap.ParseFromString(blob)
        conn.execute(
            "UPDATE worker_resource_history SET "
            "snapshot_host_cpu_percent = ?, snapshot_memory_used_bytes = ?, "
            "snapshot_memory_total_bytes = ?, snapshot_disk_used_bytes = ?, "
            "snapshot_disk_total_bytes = ?, snapshot_running_task_count = ?, "
            "snapshot_total_process_count = ?, snapshot_net_recv_bps = ?, "
            "snapshot_net_sent_bps = ? WHERE id = ?",
            (
                snap.host_cpu_percent or None,
                snap.memory_used_bytes or None,
                snap.memory_total_bytes or None,
                snap.disk_used_bytes or None,
                snap.disk_total_bytes or None,
                snap.running_task_count or None,
                snap.total_process_count or None,
                snap.net_recv_bps or None,
                snap.net_sent_bps or None,
                row_id,
            ),
        )

    # Drop old BLOB columns.
    if _has_column(conn, "workers", "resource_snapshot_proto"):
        conn.execute("ALTER TABLE workers DROP COLUMN resource_snapshot_proto")
    if _has_column(conn, "worker_resource_history", "snapshot_proto"):
        conn.execute("ALTER TABLE worker_resource_history DROP COLUMN snapshot_proto")
