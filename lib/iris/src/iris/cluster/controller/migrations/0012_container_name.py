# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    """Add container_id column to tasks table.

    Stores the platform container identifier (Docker container ID, K8s pod name)
    reported by the worker during heartbeat. Only populated while a task
    is actively running; NULL otherwise.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "container_id" not in columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN container_id TEXT")
