# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``reads/budgets.py`` (Stage 10 of the SA Core migration).

Each test exercises a legacy budget/user read against its SA Core port
in :mod:`iris.cluster.controller.reads.budgets` and asserts the two
paths return equal results against the same DB state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from iris.cluster.controller import db_v2
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.reads import budgets as reads_budgets
from rigging.timing import Timestamp


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(db_dir=tmp_path)


def _create_user(db: ControllerDB, user_id: str, role: str | None = None) -> None:
    if role is None:
        db.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at_ms) VALUES (?, ?)",
            (user_id, 1000),
        )
    else:
        db.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at_ms, role) VALUES (?, ?, ?)",
            (user_id, 1000, role),
        )


def test_get_user_budget_parity(db: ControllerDB) -> None:
    _create_user(db, "alice")
    db.set_user_budget("alice", budget_limit=42, max_band=3, now=Timestamp.from_ms(5000))

    legacy = db.get_user_budget("alice")
    with db_v2.read_snapshot(db.sa_read_engine) as sa_tx:
        sa = reads_budgets.get_user_budget(sa_tx, "alice")
    assert legacy == sa
    assert legacy is not None
    assert legacy.budget_limit == 42


def test_get_user_budget_missing_parity(db: ControllerDB) -> None:
    legacy = db.get_user_budget("nobody")
    with db_v2.read_snapshot(db.sa_read_engine) as sa_tx:
        sa = reads_budgets.get_user_budget(sa_tx, "nobody")
    assert legacy == sa is None


def test_list_user_budgets_parity(db: ControllerDB) -> None:
    for uid, lim in [("u1", 10), ("u2", 20), ("u3", 30)]:
        _create_user(db, uid)
        db.set_user_budget(uid, budget_limit=lim, max_band=1, now=Timestamp.from_ms(1000))

    legacy = sorted(db.list_user_budgets(), key=lambda b: b.user_id)
    with db_v2.read_snapshot(db.sa_read_engine) as sa_tx:
        sa = sorted(reads_budgets.list_user_budgets(sa_tx), key=lambda b: b.user_id)
    assert legacy == sa
    assert len(legacy) == 3


def test_get_all_user_budget_limits_parity(db: ControllerDB) -> None:
    for uid, lim in [("u1", 100), ("u2", 200)]:
        _create_user(db, uid)
        db.set_user_budget(uid, budget_limit=lim, max_band=2, now=Timestamp.from_ms(1000))

    legacy = db.get_all_user_budget_limits()
    with db_v2.read_snapshot(db.sa_read_engine) as sa_tx:
        sa = reads_budgets.get_all_user_budget_limits(sa_tx)
    assert legacy == sa
    assert legacy == {"u1": 100, "u2": 200}


def test_get_user_role_parity(db: ControllerDB) -> None:
    _create_user(db, "admin_user", role="admin")
    _create_user(db, "plain_user", role="user")

    legacy_admin = db.get_user_role("admin_user")
    legacy_plain = db.get_user_role("plain_user")
    legacy_missing = db.get_user_role("nobody")

    with db_v2.read_snapshot(db.sa_read_engine) as sa_tx:
        sa_admin = reads_budgets.get_user_role(sa_tx, "admin_user")
        sa_plain = reads_budgets.get_user_role(sa_tx, "plain_user")
        sa_missing = reads_budgets.get_user_role(sa_tx, "nobody")

    assert (legacy_admin, legacy_plain, legacy_missing) == ("admin", "user", "user")
    assert (sa_admin, sa_plain, sa_missing) == (legacy_admin, legacy_plain, legacy_missing)
