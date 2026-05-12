# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``reads/reservations.py`` (Stage 10 of the SA Core migration).

Each test exercises a legacy reservation read against its SA Core port
in :mod:`iris.cluster.controller.reads.reservations` and asserts the two
paths return equal results against the same DB state.
"""

from __future__ import annotations

from iris.cluster.controller import db_v2
from iris.cluster.controller.controller import _read_reservation_claims
from iris.cluster.controller.reads import reservations as reads_reservations
from iris.cluster.types import JobName, WorkerId


def _seed_claim(state, worker_id: str, job_id: str, entry_idx: int) -> None:
    state._db.execute(
        "INSERT INTO reservation_claims(worker_id, job_id, entry_idx) VALUES (?, ?, ?)",
        (worker_id, job_id, entry_idx),
    )


def _ensure_worker_row(state, worker_id: str) -> None:
    # ``reservation_claims.worker_id`` has no FK in the legacy schema, but
    # several tests in this suite also touch ``workers``. Adding minimal rows
    # keeps the snapshot consistent and matches the production code path
    # where every claim is for a worker that was previously upserted.
    state._db.execute(
        "INSERT OR IGNORE INTO workers(worker_id, address) VALUES (?, ?)",
        (worker_id, f"{worker_id}:80"),
    )


def test_list_claims_parity(state):
    _ensure_worker_row(state, "w1")
    _ensure_worker_row(state, "w2")
    jid = JobName.root("test-user", "j1").to_wire()
    _seed_claim(state, "w1", jid, 0)
    _seed_claim(state, "w2", jid, 1)

    legacy = _read_reservation_claims(state._db)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_reservations.list_claims(sa_tx)
    assert legacy == sa
    assert set(legacy.keys()) == {WorkerId("w1"), WorkerId("w2")}


def test_list_claims_empty_parity(state):
    legacy = _read_reservation_claims(state._db)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_reservations.list_claims(sa_tx)
    assert legacy == sa == {}


def test_get_claim_for_worker_parity(state):
    _ensure_worker_row(state, "w1")
    jid = JobName.root("test-user", "j1").to_wire()
    _seed_claim(state, "w1", jid, 7)

    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_reservations.get_claim_for_worker(sa_tx, WorkerId("w1"))
        missing = reads_reservations.get_claim_for_worker(sa_tx, WorkerId("missing"))
    assert sa == (jid, 7)
    assert missing is None


def test_list_claims_for_job_parity(state):
    _ensure_worker_row(state, "w1")
    _ensure_worker_row(state, "w2")
    _ensure_worker_row(state, "w3")
    j1 = JobName.root("test-user", "j1").to_wire()
    j2 = JobName.root("test-user", "j2").to_wire()
    _seed_claim(state, "w1", j1, 0)
    _seed_claim(state, "w2", j1, 1)
    _seed_claim(state, "w3", j2, 0)

    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        j1_claims = reads_reservations.list_claims_for_job(sa_tx, j1)
        j2_claims = reads_reservations.list_claims_for_job(sa_tx, j2)
        absent = reads_reservations.list_claims_for_job(sa_tx, "/nobody")

    assert {wid for wid, _ in j1_claims} == {WorkerId("w1"), WorkerId("w2")}
    assert j2_claims == [(WorkerId("w3"), 0)]
    assert absent == []


def test_count_claims_for_job_parity(state):
    _ensure_worker_row(state, "w1")
    _ensure_worker_row(state, "w2")
    j1 = JobName.root("test-user", "j1").to_wire()
    _seed_claim(state, "w1", j1, 0)
    _seed_claim(state, "w2", j1, 1)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_reservations.count_claims_for_job(sa_tx, j1) == 2
        assert reads_reservations.count_claims_for_job(sa_tx, "/nobody") == 0


def test_get_last_submission_ms_unset_parity(state):
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_reservations.get_last_submission_ms(sa_tx) == 0


def test_get_last_submission_ms_after_write_parity(state):
    # Drive the legacy upsert path through ``ReservationStore.next_submission_ms``.
    with state._store.transaction() as cur:
        state._store.reservations.next_submission_ms(cur, submitted_ms=12345)

    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_reservations.get_last_submission_ms(sa_tx) == 12345
