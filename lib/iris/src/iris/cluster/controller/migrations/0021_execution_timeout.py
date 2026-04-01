# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def migrate(conn: sqlite3.Connection) -> None:
    if not _has_column(conn, "jobs", "execution_timeout_ms"):
        conn.execute("ALTER TABLE jobs ADD COLUMN execution_timeout_ms INTEGER")

    # Backfill from request_proto for existing jobs.
    from iris.rpc import cluster_pb2

    rows = conn.execute(
        "SELECT job_id, request_proto FROM jobs WHERE request_proto IS NOT NULL AND execution_timeout_ms IS NULL"
    ).fetchall()
    for job_id, request_blob in rows:
        request = cluster_pb2.Controller.LaunchJobRequest()
        request.ParseFromString(request_blob)
        timeout_ms = (
            int(request.timeout.milliseconds)
            if request.HasField("timeout") and request.timeout.milliseconds > 0
            else None
        )
        if timeout_ms is not None:
            conn.execute(
                "UPDATE jobs SET execution_timeout_ms = ? WHERE job_id = ?",
                (timeout_ms, job_id),
            )
