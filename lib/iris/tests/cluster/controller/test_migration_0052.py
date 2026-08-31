# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0052_attempt_runtime_release``."""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0052_attempt_runtime_release.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0052", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0052_preserves_finished_attempts_as_released() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE task_attempts (attempt_uid VARCHAR NOT NULL, finished_at_ms INTEGER, " "PRIMARY KEY (attempt_uid))"
    )
    conn.executemany(
        "INSERT INTO task_attempts VALUES (?, ?)",
        [("finished", 1234), ("running", None)],
    )
    migration = _load_migration()

    migration.migrate(conn)
    migration.migrate(conn)

    rows = conn.execute("SELECT attempt_uid, runtime_released_at_ms FROM task_attempts ORDER BY attempt_uid").fetchall()
    assert rows == [("finished", 1234), ("running", None)]
    conn.close()
