# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0051_task_attempt_outputs``."""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0051_task_attempt_outputs.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0051", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0051_creates_idempotent_attempt_output_table_with_cascade() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "CREATE TABLE task_attempts (task_id VARCHAR NOT NULL, attempt_id INTEGER NOT NULL, "
        "PRIMARY KEY (task_id, attempt_id))"
    )
    migration = _load_migration()

    migration.migrate(conn)
    migration.migrate(conn)
    conn.execute("INSERT INTO task_attempts VALUES ('/user/job/0', 0)")
    conn.execute("INSERT INTO task_attempt_outputs VALUES ('/user/job/0', 0, '{\"state\":\"uploaded\"}')")
    conn.execute("DELETE FROM task_attempts WHERE task_id='/user/job/0' AND attempt_id=0")

    assert conn.execute("SELECT COUNT(*) FROM task_attempt_outputs").fetchone()[0] == 0
    conn.close()
