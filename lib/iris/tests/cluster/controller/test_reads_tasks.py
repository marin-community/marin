# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``reads/tasks.py`` (Stage 9 of the SA Core migration).

Each test exercises a legacy ``TaskStore`` read against its SA Core port
in :mod:`iris.cluster.controller.reads.tasks` and asserts the two paths
return equal results against the same DB state.
"""

from __future__ import annotations

from iris.cluster.controller import db_v2
from iris.cluster.controller.db import ACTIVE_TASK_STATES
from iris.cluster.controller.reads import tasks as reads_tasks
from iris.cluster.controller.stores import TaskScope
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from .conftest import (
    dispatch_task,
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)


def _submit_job(state, name: str, *, cpu: int = 1, replicas: int = 1):
    return submit_job(state, name, make_job_request(name, cpu=cpu, replicas=replicas))


def test_get_detail_parity(state):
    tasks = _submit_job(state, "j")
    tid = tasks[0].task_id
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.get_detail(legacy_tx, tid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.get_detail(sa_tx, tid)
    assert legacy == sa
    assert legacy is not None

    missing = JobName.from_string("/test-user/no-such/0")
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_tasks.get_detail(sa_tx, missing) is None


def test_bulk_get_detail_parity(state):
    a = _submit_job(state, "a", replicas=3)
    b = _submit_job(state, "b", replicas=2)
    ids = [t.task_id for t in a + b]
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.bulk_get_detail(legacy_tx, ids)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.bulk_get_detail(sa_tx, ids)
    assert legacy == sa
    assert set(legacy.keys()) == set(ids)


def test_bulk_get_detail_empty_parity(state):
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.bulk_get_detail(legacy_tx, [])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.bulk_get_detail(sa_tx, [])
    assert legacy == sa == {}


def test_get_job_id_parity(state):
    tasks = _submit_job(state, "j")
    tid = tasks[0].task_id
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.get_job_id(legacy_tx, tid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.get_job_id(sa_tx, tid)
    assert legacy == sa
    assert legacy == JobName.from_string("/test-user/j")


def test_get_current_attempt_id_parity(state):
    tasks = _submit_job(state, "j")
    tid = tasks[0].task_id
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.get_current_attempt_id(legacy_tx, tid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.get_current_attempt_id(sa_tx, tid)
    assert legacy == sa


def test_get_priority_band_for_job_parity(state):
    _submit_job(state, "j")
    jid = JobName.from_string("/test-user/j")
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.get_priority_band_for_job(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.get_priority_band_for_job(sa_tx, jid)
    assert legacy == sa
    assert legacy is not None


def test_state_counts_for_job_parity(state):
    _submit_job(state, "j", replicas=3)
    jid = JobName.from_string("/test-user/j")
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.state_counts_for_job(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.state_counts_for_job(sa_tx, jid)
    assert legacy == sa
    assert sum(legacy.values()) == 3


def test_first_error_for_job_parity(state):
    _submit_job(state, "j", replicas=2)
    jid = JobName.from_string("/test-user/j")

    # No errors yet — both paths return None.
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.first_error_for_job(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.first_error_for_job(sa_tx, jid)
    assert legacy == sa is None

    # Force-set an error on one row.
    state._db.execute(
        "UPDATE tasks SET error = ? WHERE task_id = ?",
        ("boom", JobName.from_string("/test-user/j/0").to_wire()),
    )
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.first_error_for_job(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.first_error_for_job(sa_tx, jid)
    assert legacy == sa == "boom"


def test_list_active_by_job_parity(state):
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())
    tasks = _submit_job(state, "j", replicas=2)
    dispatch_task(state, tasks[0], wid)

    jid = JobName.from_string("/test-user/j")
    scope = TaskScope(job_id=jid)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_active(legacy_tx, scope, states=ACTIVE_TASK_STATES)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_active(sa_tx, scope, states=ACTIVE_TASK_STATES)
    assert legacy == sa
    assert len(legacy) == 1  # only the dispatched task is ACTIVE


def test_list_active_by_worker_parity(state):
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())
    tasks = _submit_job(state, "j")
    dispatch_task(state, tasks[0], wid)

    scope = TaskScope(worker_id=wid)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_active(legacy_tx, scope, states=ACTIVE_TASK_STATES)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_active(sa_tx, scope, states=ACTIVE_TASK_STATES)
    assert legacy == sa
    assert len(legacy) == 1


def test_list_active_null_worker_parity(state):
    _submit_job(state, "j")  # PENDING with null current_worker_id
    scope = TaskScope(null_worker=True)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_active(
            legacy_tx,
            scope,
            states=[int(job_pb2.TASK_STATE_PENDING)],
        )
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_active(sa_tx, scope, states=[int(job_pb2.TASK_STATE_PENDING)])
    assert legacy == sa
    assert len(legacy) == 1


def test_list_active_task_ids_parity(state):
    tasks = _submit_job(state, "j", replicas=3)
    target_ids = [t.task_id for t in tasks[:2]]
    scope = TaskScope(task_ids=target_ids)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_active(
            legacy_tx,
            scope,
            states=[int(job_pb2.TASK_STATE_PENDING)],
        )
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_active(sa_tx, scope, states=[int(job_pb2.TASK_STATE_PENDING)])
    assert legacy == sa
    assert {t.task_id for t in legacy} == set(target_ids)


def test_list_active_empty_states_short_circuit(state):
    _submit_job(state, "j")
    scope = TaskScope(job_id=JobName.from_string("/test-user/j"))
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_active(legacy_tx, scope, states=[])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_active(sa_tx, scope, states=[])
    assert legacy == sa == []


def test_list_active_exclude_task_id_parity(state):
    tasks = _submit_job(state, "j", replicas=2)
    scope = TaskScope(job_id=JobName.from_string("/test-user/j"))
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_active(
            legacy_tx,
            scope,
            states=[int(job_pb2.TASK_STATE_PENDING)],
            exclude_task_id=tasks[0].task_id,
        )
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_active(
            sa_tx,
            scope,
            states=[int(job_pb2.TASK_STATE_PENDING)],
            exclude_task_id=tasks[0].task_id,
        )
    assert legacy == sa
    assert {t.task_id for t in legacy} == {tasks[1].task_id}


def test_list_active_order_by_task_id_with_limit_parity(state):
    tasks = _submit_job(state, "j", replicas=4)
    scope = TaskScope(job_id=JobName.from_string("/test-user/j"))
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_active(
            legacy_tx,
            scope,
            states=[int(job_pb2.TASK_STATE_PENDING)],
            order_by_task_id=True,
            limit=2,
        )
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_active(
            sa_tx,
            scope,
            states=[int(job_pb2.TASK_STATE_PENDING)],
            order_by_task_id=True,
            limit=2,
        )
    assert legacy == sa
    assert len(legacy) == 2
    _ = tasks


def test_get_with_resources_parity(state):
    tasks = _submit_job(state, "j", cpu=2)
    tid = tasks[0].task_id
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.get_with_resources(legacy_tx, tid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.get_with_resources(sa_tx, tid)
    assert legacy == sa
    assert legacy is not None

    missing = JobName.from_string("/test-user/no-such/0")
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_tasks.get_with_resources(sa_tx, missing) is None


def test_list_pending_for_direct_provider_parity(state):
    _submit_job(state, "j", replicas=3)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_pending_for_direct_provider(legacy_tx, limit=10)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_pending_for_direct_provider(sa_tx, limit=10)
    assert legacy == sa
    assert len(legacy) == 3


def test_list_pending_for_direct_provider_zero_limit(state):
    _submit_job(state, "j")
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_pending_for_direct_provider(legacy_tx, limit=0)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_pending_for_direct_provider(sa_tx, limit=0)
    assert legacy == sa == []


def test_list_assigned_null_worker_for_direct_provider_parity(state):
    tasks = _submit_job(state, "j")
    # Force ASSIGNED + null worker.
    state._db.execute(
        "UPDATE tasks SET state = ?, current_worker_id = NULL WHERE task_id = ?",
        (int(job_pb2.TASK_STATE_ASSIGNED), tasks[0].task_id.to_wire()),
    )
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.tasks.list_assigned_null_worker_for_direct_provider(legacy_tx)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_tasks.list_assigned_null_worker_for_direct_provider(sa_tx)
    assert legacy == sa
    assert len(legacy) == 1
