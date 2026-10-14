# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0053_user_accelerator_cap``."""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0053_user_accelerator_cap.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0053", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0053_leaves_existing_budgets_uncapped_and_seedable() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE user_budgets (user_id TEXT PRIMARY KEY, budget_limit INTEGER NOT NULL DEFAULT 0, "
        "max_band INTEGER NOT NULL DEFAULT 2, updated_at_ms INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO user_budgets VALUES ('belay', 75000, 2, 1000)")
    migration = _load_migration()

    migration.migrate(conn)
    migration.migrate(conn)

    rows = conn.execute("SELECT user_id, budget_limit, max_accelerators, cap_set_at_ms FROM user_budgets").fetchall()
    assert rows == [("belay", 75000, 0, None)]
    conn.close()
