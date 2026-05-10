# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop the resource-history tables and the cached ``snapshot_*`` columns.

Per-tick host utilization and per-attempt task resource samples now live
in the ``iris.worker`` / ``iris.task`` stats namespaces (controller-stats
migration). The controller DB only tracks decisions, so the time-series
tables and their cached "latest" copy on ``workers`` go away here.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_worker_resource_history_worker")
    conn.execute("DROP INDEX IF EXISTS idx_worker_resource_history_ts")
    conn.execute("DROP TABLE IF EXISTS worker_resource_history")
    conn.execute("DROP INDEX IF EXISTS idx_task_resource_history_task_attempt")
    conn.execute("DROP TABLE IF EXISTS task_resource_history")

    existing = {row[1] for row in conn.execute("PRAGMA table_info(workers)").fetchall()}
    for col in (
        "snapshot_host_cpu_percent",
        "snapshot_memory_used_bytes",
        "snapshot_memory_total_bytes",
        "snapshot_disk_used_bytes",
        "snapshot_disk_total_bytes",
        "snapshot_running_task_count",
        "snapshot_total_process_count",
        "snapshot_net_recv_bps",
        "snapshot_net_sent_bps",
    ):
        if col in existing:
            conn.execute(f"ALTER TABLE workers DROP COLUMN {col}")
