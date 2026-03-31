# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for budget migration and DB accessors."""

from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB, UserBudget
from rigging.timing import Timestamp


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(db_dir=tmp_path)


def _create_user(db: ControllerDB, user_id: str, created_at_ms: int = 1000) -> None:
    db.execute(
        "INSERT OR IGNORE INTO users (user_id, created_at_ms) VALUES (?, ?)",
        (user_id, created_at_ms),
    )


def test_migration_creates_user_budgets_table(db: ControllerDB) -> None:
    """The 0013 migration creates the user_budgets table."""
    row = db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name='user_budgets'")
    assert row is not None


def test_migration_adds_priority_band_column(db: ControllerDB) -> None:
    """The 0013 migration adds priority_band column to tasks."""
    columns = {row[1] for row in db.fetchall("PRAGMA table_info(tasks)")}
    assert "priority_band" in columns


def test_migration_seeds_budgets_for_existing_users(tmp_path: Path) -> None:
    """Users created before the migration get seeded budget rows."""
    # Create a DB, add a user, then verify the migration seeded a budget row.
    # ControllerDB.__init__ applies all migrations including 0013, but users
    # table is empty at that point. We test the INSERT OR IGNORE path by
    # adding a user and re-running the seed SQL.
    db = ControllerDB(db_dir=tmp_path)
    _create_user(db, "alice", created_at_ms=5000)
    # Re-run the seed statement (idempotent)
    db.execute(
        "INSERT OR IGNORE INTO user_budgets(user_id, budget_limit, max_band, updated_at_ms) "
        "SELECT user_id, 0, 2, created_at_ms FROM users"
    )
    budget = db.get_user_budget("alice")
    assert budget is not None
    assert budget.budget_limit == 0
    assert budget.max_band == 2
    db.close()


def test_pending_index_includes_priority_band(db: ControllerDB) -> None:
    """The rebuilt idx_tasks_pending includes priority_band."""
    rows = db.fetchall("PRAGMA index_info(idx_tasks_pending)")
    col_names = [row["name"] for row in rows]
    assert "priority_band" in col_names
    # priority_band should come right after state
    assert col_names.index("priority_band") == 1


def test_set_and_get_user_budget(db: ControllerDB) -> None:
    """Round-trip set_user_budget / get_user_budget."""
    _create_user(db, "bob")
    now = Timestamp.from_ms(2000)
    db.set_user_budget("bob", budget_limit=5000, max_band=1, now=now)

    budget = db.get_user_budget("bob")
    assert budget is not None
    assert isinstance(budget, UserBudget)
    assert budget.user_id == "bob"
    assert budget.budget_limit == 5000
    assert budget.max_band == 1
    assert budget.updated_at == now


def test_set_user_budget_upsert(db: ControllerDB) -> None:
    """set_user_budget updates an existing row on conflict."""
    _create_user(db, "carol")
    db.set_user_budget("carol", budget_limit=100, max_band=2, now=Timestamp.from_ms(1000))
    db.set_user_budget("carol", budget_limit=999, max_band=3, now=Timestamp.from_ms(2000))

    budget = db.get_user_budget("carol")
    assert budget is not None
    assert budget.budget_limit == 999
    assert budget.max_band == 3
    assert budget.updated_at == Timestamp.from_ms(2000)


def test_get_user_budget_returns_none_for_unknown(db: ControllerDB) -> None:
    assert db.get_user_budget("nonexistent") is None


def test_list_user_budgets(db: ControllerDB) -> None:
    _create_user(db, "u1")
    _create_user(db, "u2")
    _create_user(db, "u3")
    now = Timestamp.from_ms(3000)
    db.set_user_budget("u1", budget_limit=10, max_band=1, now=now)
    db.set_user_budget("u2", budget_limit=20, max_band=2, now=now)
    db.set_user_budget("u3", budget_limit=30, max_band=3, now=now)

    budgets = db.list_user_budgets()
    assert len(budgets) == 3
    assert all(isinstance(b, UserBudget) for b in budgets)
    by_user = {b.user_id: b for b in budgets}
    assert by_user["u1"].budget_limit == 10
    assert by_user["u2"].budget_limit == 20
    assert by_user["u3"].budget_limit == 30


def test_list_user_budgets_empty(db: ControllerDB) -> None:
    assert db.list_user_budgets() == []
