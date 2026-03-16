# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE jobs ADD COLUMN name TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_name ON jobs(name)")

    # Backfill name from request_proto in batches to avoid holding the write lock too long.
    from iris.rpc import cluster_pb2

    while True:
        rows = conn.execute("SELECT job_id, request_proto FROM jobs WHERE name = '' LIMIT 1000").fetchall()
        if not rows:
            break
        for row in rows:
            proto = cluster_pb2.Controller.LaunchJobRequest()
            proto.ParseFromString(row[1])
            conn.execute("UPDATE jobs SET name = ? WHERE job_id = ?", (proto.name, row[0]))
