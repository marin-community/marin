# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def migrate(conn: sqlite3.Connection) -> None:
    # On fresh DBs, job_config already has these columns; skip adding to jobs.
    if _table_exists(conn, "job_config"):
        return

    if not _has_column(conn, "jobs", "has_reservation"):
        conn.execute("ALTER TABLE jobs ADD COLUMN has_reservation INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_has_reservation "
        "ON jobs(has_reservation, state) WHERE has_reservation = 1"
    )

    # Backfill: scan all rows once and mark only those with reservations.
    # Skip if any rows already have has_reservation=1 (migration already ran).
    already_backfilled = conn.execute("SELECT 1 FROM jobs WHERE has_reservation = 1 LIMIT 1").fetchone()
    if already_backfilled:
        return

    if not _has_column(conn, "jobs", "request_proto"):
        return

    from iris.cluster.controller.transitions import _has_reservation_flag
    from iris.rpc import controller_pb2

    rows = conn.execute("SELECT job_id, request_proto FROM jobs WHERE request_proto IS NOT NULL").fetchall()
    for row in rows:
        proto = controller_pb2.Controller.LaunchJobRequest()
        proto.ParseFromString(row[1])
        if _has_reservation_flag(proto):
            conn.execute("UPDATE jobs SET has_reservation = 1 WHERE job_id = ?", (row[0],))
