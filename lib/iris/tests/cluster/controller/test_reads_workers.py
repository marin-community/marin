# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``reads/workers.py`` (Stage 10 of the SA Core migration).

Each test exercises a legacy ``WorkerStore`` read against its SA Core
port in :mod:`iris.cluster.controller.reads.workers` and asserts the two
paths return equal results against the same DB state.
"""

from __future__ import annotations

from iris.cluster.controller import db_v2
from iris.cluster.controller.db import healthy_active_workers_with_attributes as legacy_haw
from iris.cluster.controller.reads import workers as reads_workers
from iris.cluster.types import WorkerId

from .conftest import make_worker_metadata, register_worker


def test_address_parity(state):
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.address(legacy_tx, wid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.address(sa_tx, wid)
    assert legacy == sa == "host:8080"

    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_workers.address(sa_tx, WorkerId("missing")) is None


def test_get_detail_parity(state):
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata(cpu=4))
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.get_detail(legacy_tx, wid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.get_detail(sa_tx, wid)
    assert legacy == sa
    assert legacy is not None

    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_workers.get_detail(sa_tx, WorkerId("missing")) is None


def test_active_healthy_address_parity(state):
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.active_healthy_address(legacy_tx, wid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.active_healthy_address(sa_tx, wid, state._store.health)
    assert legacy == sa == "host:8080"

    # Flip to unhealthy and confirm both paths return None.
    state._store.workers.set_health_for_test(wid, healthy=False)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.active_healthy_address(legacy_tx, wid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.active_healthy_address(sa_tx, wid, state._store.health)
    assert legacy == sa is None


def test_list_active_healthy_parity(state):
    w1 = register_worker(state, "w1", "host1:80", make_worker_metadata())
    w2 = register_worker(state, "w2", "host2:80", make_worker_metadata())
    register_worker(state, "w3", "host3:80", make_worker_metadata(), healthy=False)

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.list_active_healthy(legacy_tx)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.list_active_healthy(sa_tx, state._store.health)
    assert legacy == sa
    assert set(legacy.keys()) == {w1, w2}


def test_list_active_by_ids_parity(state):
    w1 = register_worker(state, "w1", "host:80", make_worker_metadata())
    w2 = register_worker(state, "w2", "host:81", make_worker_metadata())
    ids = [str(w1), str(w2), "missing"]
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.list_active_by_ids(legacy_tx, ids)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.list_active_by_ids(sa_tx, ids, state._store.health)
    assert legacy == sa
    assert {w.worker_id for w in legacy} == {w1, w2}


def test_list_active_by_ids_empty_parity(state):
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.list_active_by_ids(legacy_tx, [])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.list_active_by_ids(sa_tx, [], state._store.health)
    assert legacy == sa == []


def test_filter_existing_parity(state):
    w1 = register_worker(state, "w1", "host:80", make_worker_metadata())
    candidates = [w1, WorkerId("missing")]
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.filter_existing(legacy_tx, candidates)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.filter_existing(sa_tx, candidates)
    assert legacy == sa == {str(w1)}


def test_filter_existing_empty_parity(state):
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.workers.filter_existing(legacy_tx, [])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.filter_existing(sa_tx, [])
    assert legacy == sa == set()


def test_healthy_active_workers_with_attributes_parity(state):
    register_worker(state, "w1", "host:80", make_worker_metadata())
    register_worker(state, "w2", "host:81", make_worker_metadata(gpu_count=1, gpu_name="A100"))
    register_worker(state, "w3", "host:82", make_worker_metadata(), healthy=False)

    legacy = legacy_haw(state._db, state._store.health, state._store.worker_attrs)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_workers.healthy_active_workers_with_attributes(sa_tx, state._store.health, state._store.worker_attrs)

    legacy_by_id = {w.worker_id: w for w in legacy}
    sa_by_id = {w.worker_id: w for w in sa}
    assert legacy_by_id == sa_by_id
    assert set(legacy_by_id.keys()) == {WorkerId("w1"), WorkerId("w2")}
