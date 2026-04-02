# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Allow NULL worker_id in dispatch_queue for direct provider kill entries.

    The original schema had worker_id TEXT NOT NULL with a FK to workers.
    Direct provider inserts kill entries with worker_id=NULL (no worker daemon).
    """
    # Check if dispatch_queue already allows NULL worker_id (migration already applied).
    info = {row[1]: row for row in conn.execute("PRAGMA table_info(dispatch_queue)").fetchall()}
    if "worker_id" not in info or info["worker_id"][3] == 0:
        return

    conn.executescript(
        """
        CREATE TABLE dispatch_queue_v11 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT REFERENCES workers(worker_id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('run', 'kill')),
            payload_proto BLOB,
            task_id TEXT,
            created_at_ms INTEGER NOT NULL
        );
        INSERT INTO dispatch_queue_v11 SELECT * FROM dispatch_queue;
        DROP TABLE dispatch_queue;
        ALTER TABLE dispatch_queue_v11 RENAME TO dispatch_queue;
        CREATE INDEX IF NOT EXISTS idx_dispatch_worker ON dispatch_queue(worker_id, id);
    """
    )
