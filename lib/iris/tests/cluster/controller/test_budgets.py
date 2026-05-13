# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for budget migration and DB accessors."""

from pathlib import Path

import pytest
from iris.cluster.controller import reads, writes
from iris.cluster.controller.budget import reconcile_user_budget_tiers
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.reads import UserBudget
from iris.rpc import config_pb2, job_pb2
from rigging.timing import Timestamp
from sqlalchemy import text


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(db_dir=tmp_path)


def _create_user(db: ControllerDB, user_id: str, created_at_ms: int = 1000) -> None:
    with db.transaction() as tx:
        writes.ensure_user(tx, user_id, Timestamp.from_ms(created_at_ms))


def _set_user_budget(db: ControllerDB, user_id: str, budget_limit: int, max_band: int, now: Timestamp) -> None:
    with db.transaction() as tx:
        writes.set_user_budget(tx, user_id, budget_limit, max_band, now)


def _get_user_budget(db: ControllerDB, user_id: str) -> UserBudget | None:
    with db.read_snapshot() as snap:
        return reads.get_user_budget(snap, user_id)


def _list_user_budgets(db: ControllerDB) -> list[UserBudget]:
    with db.read_snapshot() as snap:
        return reads.list_user_budgets(snap)


def test_migration_creates_user_budgets_table(db: ControllerDB) -> None:
    """The 0013 migration creates the user_budgets table."""
    with db.read_snapshot() as q:
        row = q.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='user_budgets'")).first()
    assert row is not None


def test_migration_adds_priority_band_column(db: ControllerDB) -> None:
    """The 0013 migration adds priority_band column to tasks."""
    with db.read_snapshot() as q:
        columns = {row[1] for row in q.execute(text("PRAGMA table_info(tasks)")).all()}
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
    with db.transaction() as tx:
        tx.execute(
            text(
                "INSERT OR IGNORE INTO user_budgets(user_id, budget_limit, max_band, updated_at_ms) "
                "SELECT user_id, 0, 2, created_at_ms FROM users"
            )
        )
    budget = _get_user_budget(db, "alice")
    assert budget is not None
    assert budget.budget_limit == 0
    assert budget.max_band == 2
    db.close()


def test_pending_index_includes_priority_band(db: ControllerDB) -> None:
    """The rebuilt idx_tasks_pending includes priority_band."""
    with db.read_snapshot() as q:
        rows = q.execute(text("PRAGMA index_info(idx_tasks_pending)")).all()
    col_names = [row[2] for row in rows]
    assert "priority_band" in col_names
    # priority_band should come right after state
    assert col_names.index("priority_band") == 1


def test_set_and_get_user_budget(db: ControllerDB) -> None:
    """Round-trip set_user_budget / get_user_budget."""
    _create_user(db, "bob")
    now = Timestamp.from_ms(2000)
    _set_user_budget(db, "bob", budget_limit=5000, max_band=1, now=now)

    budget = _get_user_budget(db, "bob")
    assert budget is not None
    assert isinstance(budget, UserBudget)
    assert budget.user_id == "bob"
    assert budget.budget_limit == 5000
    assert budget.max_band == 1
    assert budget.updated_at == now


def test_set_user_budget_upsert(db: ControllerDB) -> None:
    """set_user_budget updates an existing row on conflict."""
    _create_user(db, "carol")
    _set_user_budget(db, "carol", budget_limit=100, max_band=2, now=Timestamp.from_ms(1000))
    _set_user_budget(db, "carol", budget_limit=999, max_band=3, now=Timestamp.from_ms(2000))

    budget = _get_user_budget(db, "carol")
    assert budget is not None
    assert budget.budget_limit == 999
    assert budget.max_band == 3
    assert budget.updated_at == Timestamp.from_ms(2000)


def test_get_user_budget_returns_none_for_unknown(db: ControllerDB) -> None:
    assert _get_user_budget(db, "nonexistent") is None


def test_list_user_budgets(db: ControllerDB) -> None:
    _create_user(db, "u1")
    _create_user(db, "u2")
    _create_user(db, "u3")
    now = Timestamp.from_ms(3000)
    _set_user_budget(db, "u1", budget_limit=10, max_band=1, now=now)
    _set_user_budget(db, "u2", budget_limit=20, max_band=2, now=now)
    _set_user_budget(db, "u3", budget_limit=30, max_band=3, now=now)

    budgets = _list_user_budgets(db)
    assert len(budgets) == 3
    assert all(isinstance(b, UserBudget) for b in budgets)
    by_user = {b.user_id: b for b in budgets}
    assert by_user["u1"].budget_limit == 10
    assert by_user["u2"].budget_limit == 20
    assert by_user["u3"].budget_limit == 30


def test_list_user_budgets_empty(db: ControllerDB) -> None:
    assert _list_user_budgets(db) == []


# --- reconcile_user_budget_tiers ------------------------------------------------


def _tier(user_ids: list[str], budget_limit: int, max_band: int) -> config_pb2.UserBudgetTier:
    return config_pb2.UserBudgetTier(user_ids=user_ids, budget_limit=budget_limit, max_band=max_band)


def test_reconcile_creates_rows_for_fresh_users(db: ControllerDB) -> None:
    """On a fresh DB, reconcile_user_budget_tiers upserts the intended rows.

    Without this, listed users would have no row and fall back to
    UserBudgetDefaults when the scheduler or launch-job guard looks them up.
    """
    tiers = [
        _tier(["alice", "bob"], 75000, job_pb2.PRIORITY_BAND_PRODUCTION),
        _tier(["carol"], 75000, job_pb2.PRIORITY_BAND_INTERACTIVE),
    ]
    count = reconcile_user_budget_tiers(db, tiers, Timestamp.from_ms(1000))
    assert count == 3

    alice = _get_user_budget(db, "alice")
    assert alice is not None
    assert alice.budget_limit == 75000
    assert alice.max_band == job_pb2.PRIORITY_BAND_PRODUCTION

    carol = _get_user_budget(db, "carol")
    assert carol is not None
    assert carol.max_band == job_pb2.PRIORITY_BAND_INTERACTIVE


def test_reconcile_upserts_existing_rows(db: ControllerDB) -> None:
    """Running reconcile twice updates rows to the latest config values."""
    first = [_tier(["dave"], 10_000, job_pb2.PRIORITY_BAND_BATCH)]
    reconcile_user_budget_tiers(db, first, Timestamp.from_ms(1000))

    # Promote dave: new budget + higher band.
    second = [_tier(["dave"], 75_000, job_pb2.PRIORITY_BAND_PRODUCTION)]
    reconcile_user_budget_tiers(db, second, Timestamp.from_ms(2000))

    dave = _get_user_budget(db, "dave")
    assert dave is not None
    assert dave.budget_limit == 75_000
    assert dave.max_band == job_pb2.PRIORITY_BAND_PRODUCTION


def test_reconcile_later_tier_overrides_earlier(db: ControllerDB) -> None:
    """If a user_id appears in multiple tiers, later tiers win."""
    tiers = [
        _tier(["eve"], 75000, job_pb2.PRIORITY_BAND_INTERACTIVE),
        _tier(["eve"], 75000, job_pb2.PRIORITY_BAND_PRODUCTION),
    ]
    reconcile_user_budget_tiers(db, tiers, Timestamp.from_ms(1000))

    eve = _get_user_budget(db, "eve")
    assert eve is not None
    assert eve.max_band == job_pb2.PRIORITY_BAND_PRODUCTION


def test_reconcile_no_tiers_is_noop(db: ControllerDB) -> None:
    """Empty config leaves the DB untouched."""
    assert reconcile_user_budget_tiers(db, [], Timestamp.from_ms(1000)) == 0
    assert _list_user_budgets(db) == []


def test_reconcile_rejects_unspecified_band(db: ControllerDB) -> None:
    """A config missing max_band surfaces as a ValueError, not a silent BATCH."""
    tiers = [_tier(["frank"], 75000, job_pb2.PRIORITY_BAND_UNSPECIFIED)]
    with pytest.raises(ValueError, match="max_band must be one of"):
        reconcile_user_budget_tiers(db, tiers, Timestamp.from_ms(1000))


def test_reconcile_rejects_empty_user_id(db: ControllerDB) -> None:
    tiers = [_tier(["grace", ""], 75000, job_pb2.PRIORITY_BAND_INTERACTIVE)]
    with pytest.raises(ValueError, match="empty entry"):
        reconcile_user_budget_tiers(db, tiers, Timestamp.from_ms(1000))


def test_reconcile_creates_user_row_for_fk(db: ControllerDB) -> None:
    """user_budgets has a FK on users; reconcile must ensure_user first."""
    tiers = [_tier(["henry"], 75000, job_pb2.PRIORITY_BAND_INTERACTIVE)]
    reconcile_user_budget_tiers(db, tiers, Timestamp.from_ms(1000))

    with db.read_snapshot() as q:
        row = q.execute(text("SELECT user_id FROM users WHERE user_id = 'henry'")).first()
    assert row is not None
