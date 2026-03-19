# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Add container_name column to tasks table.

    Stores the Docker container name (e.g. iris-user-job-0-a1-3f8a1b2c)
    reported by the worker during heartbeat. Only populated while a task
    is actively running; NULL otherwise.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "container_name" not in columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN container_name TEXT")
