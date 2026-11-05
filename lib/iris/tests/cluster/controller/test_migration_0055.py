# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0055_task_attempt_region``."""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0055_task_attempt_region.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0055", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0055_adds_a_null_region_to_existing_attempts_and_can_rerun() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE task_attempts (task_id TEXT, attempt_id INTEGER)")
    conn.execute("INSERT INTO task_attempts VALUES ('/a/0', 0)")
    migration = _load_migration()

    migration.migrate(conn)
    migration.migrate(conn)

    assert conn.execute("SELECT task_id, attempt_id, region FROM task_attempts").fetchall() == [("/a/0", 0, None)]
    conn.close()
