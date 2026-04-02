# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Add slice_id and scale_group to workers, drop tracked_workers."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(workers)").fetchall()}
    if "slice_id" not in columns:
        conn.execute("ALTER TABLE workers ADD COLUMN slice_id TEXT NOT NULL DEFAULT ''")
    if "scale_group" not in columns:
        conn.execute("ALTER TABLE workers ADD COLUMN scale_group TEXT NOT NULL DEFAULT ''")

    # Backfill from tracked_workers if it still exists (upgrade path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "tracked_workers" in tables:
        conn.execute(
            """
            UPDATE workers SET
                slice_id = COALESCE(
                    (SELECT tw.slice_id FROM tracked_workers tw WHERE tw.worker_id = workers.worker_id), ''),
                scale_group = COALESCE(
                    (SELECT tw.scale_group FROM tracked_workers tw WHERE tw.worker_id = workers.worker_id), '')
            WHERE EXISTS (SELECT 1 FROM tracked_workers tw WHERE tw.worker_id = workers.worker_id)
            """
        )
        conn.execute("DROP TABLE tracked_workers")
