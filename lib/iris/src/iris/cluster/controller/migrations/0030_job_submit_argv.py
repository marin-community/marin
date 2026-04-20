# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add submit_argv_json column to job_config for CLI invocation bookkeeping."""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    # ADD COLUMN is not idempotent in SQLite (no IF NOT EXISTS), so check first.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(job_config)").fetchall()}
    if "submit_argv_json" not in cols:
        conn.execute("ALTER TABLE job_config ADD COLUMN submit_argv_json TEXT NOT NULL DEFAULT '[]'")
