# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``writes/reservations.py`` (Stage 11 of the SA Core migration)."""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from iris.cluster.controller import db_v2
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.writes import reservations as writes_reservations
from iris.cluster.types import WorkerId


@contextmanager
def _fresh_db():
    tmp = Path(tempfile.mkdtemp(prefix="iris_writes_res_"))
    db = ControllerDB(db_dir=tmp)
    try:
        yield db, ControllerStore(db)
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def db_pair():
    with _fresh_db() as legacy, _fresh_db() as sa:
        yield legacy, sa


def _dump(db: ControllerDB, table: str, *, order_by: str | None = None) -> list[dict]:
    sql = f"SELECT * FROM {table}"
    if order_by is not None:
        sql += f" ORDER BY {order_by}"
    with db.read_snapshot() as q:
        return [dict(row) for row in q.fetchall(sql)]


# --- replace_claims ---------------------------------------------------------


def test_replace_claims_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    claims = {
        WorkerId("w1"): ("/u/j1", 0),
        WorkerId("w2"): ("/u/j2", 1),
    }
    with legacy_store.transaction() as cur:
        legacy_store.reservations.replace_claims(cur, claims)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_reservations.replace_claims(tx, claims)

    assert _dump(legacy_db, "reservation_claims", order_by="worker_id") == _dump(
        sa_db, "reservation_claims", order_by="worker_id"
    )


def test_replace_claims_overwrites_prior(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    first = {WorkerId("w1"): ("/u/j1", 0), WorkerId("w2"): ("/u/j2", 1)}
    second = {WorkerId("w3"): ("/u/j3", 0)}
    with legacy_store.transaction() as cur:
        legacy_store.reservations.replace_claims(cur, first)
        legacy_store.reservations.replace_claims(cur, second)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_reservations.replace_claims(tx, first)
        writes_reservations.replace_claims(tx, second)

    assert _dump(legacy_db, "reservation_claims", order_by="worker_id") == _dump(
        sa_db, "reservation_claims", order_by="worker_id"
    )
    assert _dump(legacy_db, "reservation_claims") == [{"worker_id": "w3", "job_id": "/u/j3", "entry_idx": 0}]


def test_replace_claims_empty_clears_table(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    seed = {WorkerId("w1"): ("/u/j1", 0)}
    with legacy_store.transaction() as cur:
        legacy_store.reservations.replace_claims(cur, seed)
        legacy_store.reservations.replace_claims(cur, {})
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_reservations.replace_claims(tx, seed)
        writes_reservations.replace_claims(tx, {})

    assert _dump(legacy_db, "reservation_claims") == [] == _dump(sa_db, "reservation_claims")


# --- next_submission_ms ----------------------------------------------------


def test_next_submission_ms_first_call_parity(db_pair):
    """First call inserts the row and returns the requested ts."""
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    with legacy_store.transaction() as cur:
        legacy_first = legacy_store.reservations.next_submission_ms(cur, 1000)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        sa_first = writes_reservations.next_submission_ms(tx, 1000)

    assert legacy_first == sa_first == 1000
    assert _dump(legacy_db, "meta", order_by="key") == _dump(sa_db, "meta", order_by="key")


def test_next_submission_ms_monotone_parity(db_pair):
    """Re-calls return ``max(submitted, last + 1)``."""
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    with legacy_store.transaction() as cur:
        legacy_first = legacy_store.reservations.next_submission_ms(cur, 1000)
        # Re-submit with an earlier ts; result must bump to 1001.
        legacy_second = legacy_store.reservations.next_submission_ms(cur, 999)
        legacy_third = legacy_store.reservations.next_submission_ms(cur, 5000)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        sa_first = writes_reservations.next_submission_ms(tx, 1000)
        sa_second = writes_reservations.next_submission_ms(tx, 999)
        sa_third = writes_reservations.next_submission_ms(tx, 5000)

    assert (legacy_first, legacy_second, legacy_third) == (sa_first, sa_second, sa_third)
    assert legacy_first == 1000
    assert legacy_second == 1001
    assert legacy_third == 5000
    assert _dump(legacy_db, "meta", order_by="key") == _dump(sa_db, "meta", order_by="key")
