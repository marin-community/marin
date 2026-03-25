# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Add consecutive_task_crashes column to workers table.

    Tracks per-worker consecutive task failures with infrastructure-fatal
    exit codes (e.g. SIGSEGV/139). When the count reaches a threshold,
    the worker is quarantined and its backing VM is deleted.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(workers)").fetchall()}
    if "consecutive_task_crashes" not in columns:
        conn.execute("ALTER TABLE workers ADD COLUMN consecutive_task_crashes INTEGER NOT NULL DEFAULT 0")
