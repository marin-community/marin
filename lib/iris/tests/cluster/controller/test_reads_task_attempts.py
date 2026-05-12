# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``reads/task_attempts.py`` (Stage 9 of the SA Core migration).

Each test exercises a legacy ``TaskAttemptStore`` read against its SA
Core port in :mod:`iris.cluster.controller.reads.task_attempts` and
asserts the two paths return equal results.
"""

from __future__ import annotations

from iris.cluster.controller import db_v2
from iris.cluster.controller.reads import task_attempts as reads_task_attempts
from iris.cluster.types import JobName

from .conftest import (
    dispatch_task,
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)


def _seed_attempt(state):
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())
    tasks = submit_job(state, "j", make_job_request("j"))
    dispatch_task(state, tasks[0], wid)
    return wid, tasks[0].task_id


def test_get_attempt_parity(state):
    _wid, task_id = _seed_attempt(state)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.get(legacy_tx, task_id, 0)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_task_attempts.get(sa_tx, task_id, 0)
    assert legacy == sa
    assert legacy is not None


def test_get_attempt_missing_parity(state):
    missing = JobName.from_string("/test-user/no-such/0")
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.get(legacy_tx, missing, 0)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_task_attempts.get(sa_tx, missing, 0)
    assert legacy == sa is None


def test_get_state_parity(state):
    _wid, task_id = _seed_attempt(state)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.get_state(legacy_tx, task_id, 0)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_task_attempts.get_state(sa_tx, task_id, 0)
    assert legacy == sa
    assert legacy is not None


def test_get_worker_id_parity(state):
    wid, task_id = _seed_attempt(state)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.get_worker_id(legacy_tx, task_id, 0)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_task_attempts.get_worker_id(sa_tx, task_id, 0)
    assert legacy == sa
    assert legacy == wid


def test_list_for_task_parity(state):
    """No legacy ``list_for_task`` method exists — assemble equivalent via per-attempt ``get``."""
    _wid, task_id = _seed_attempt(state)
    # Force a second attempt row so the list has more than one element.
    state._db.execute(
        "INSERT INTO task_attempts (task_id, attempt_id, state, created_at_ms) VALUES (?, ?, ?, ?)",
        (task_id.to_wire(), 1, 1, 1234),
    )
    with state._db.read_snapshot() as legacy_tx:
        a0 = state._store.attempts.get(legacy_tx, task_id, 0)
        a1 = state._store.attempts.get(legacy_tx, task_id, 1)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_task_attempts.list_for_task(sa_tx, task_id)
    assert sa == [a0, a1]


def test_bulk_get_for_updates_empty(state):
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_task_attempts.bulk_get_for_updates(sa_tx, []) == {}
    with state._db.read_snapshot() as legacy_tx:
        assert state._store.attempts.bulk_get_for_updates(legacy_tx, []) == {}


def test_bulk_get_for_updates_parity(state):
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())
    tasks = submit_job(state, "j", make_job_request("j", replicas=3))
    for t in tasks:
        dispatch_task(state, t, wid)

    keys = [(t.task_id, 0) for t in tasks]
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.bulk_get_for_updates(legacy_tx, keys)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_task_attempts.bulk_get_for_updates(sa_tx, keys)
    assert legacy == sa
    assert len(legacy) == 3


def test_bulk_get_for_updates_deduplicates(state):
    """Duplicate (task_id, attempt_id) entries must not produce duplicate rows."""
    _wid, task_id = _seed_attempt(state)
    keys = [(task_id, 0), (task_id, 0), (task_id, 0)]
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.bulk_get_for_updates(legacy_tx, keys)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_task_attempts.bulk_get_for_updates(sa_tx, keys)
    assert legacy == sa
    assert len(sa) == 1
