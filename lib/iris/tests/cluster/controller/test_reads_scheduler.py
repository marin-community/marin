# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``reads/scheduler.py`` (Stage 9 of the SA Core migration).

Covers the heavy scheduler-tick read paths and the two shared helpers
that used to live in :mod:`db`:

* ``resource_usage_by_worker`` (legacy
  :meth:`TaskAttemptStore.resource_usage_by_worker`)
* ``reconcile_rows_for_workers`` (legacy
  :meth:`TaskAttemptStore.reconcile_rows_for_workers`)
* ``running_tasks_by_worker`` (legacy :func:`db.running_tasks_by_worker`)
* ``timed_out_executing_tasks`` (legacy :func:`db.timed_out_executing_tasks`)

Each test sets up DB state via the legacy transition / submit paths and
asserts the legacy and SA-Core paths return equal results.
"""

from __future__ import annotations

from iris.cluster.controller import db as legacy_db
from iris.cluster.controller import db_v2
from iris.cluster.controller.reads import scheduler as reads_scheduler
from iris.cluster.types import JobName
from rigging.timing import Timestamp

from .conftest import (
    dispatch_task,
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)


def _seed_one_running_task(state):
    """Register a worker, submit a single-task job, and dispatch it."""
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata(cpu=8, memory_bytes=16 * 1024**3))
    tasks = submit_job(state, "j", make_job_request("j", cpu=2, memory_bytes=2 * 1024**3))
    dispatch_task(state, tasks[0], wid)
    return wid, tasks[0].task_id


def test_resource_usage_by_worker_empty_parity(state):
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.resource_usage_by_worker(legacy_tx)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.resource_usage_by_worker(sa_tx)
    assert legacy == sa == {}


def test_resource_usage_by_worker_populated_parity(state):
    wid, _task_id = _seed_one_running_task(state)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.resource_usage_by_worker(legacy_tx)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.resource_usage_by_worker(sa_tx)

    assert legacy == sa
    assert wid in legacy
    assert legacy[wid].cpu_millicores == 2000
    assert legacy[wid].memory_bytes == 2 * 1024**3


def test_resource_usage_by_worker_ignores_reservation_holders(state):
    """Reservation-holder rows must be skipped in both paths."""
    _seed_one_running_task(state)
    # Mark the owning job a reservation holder.
    state._db.execute(
        "UPDATE jobs SET is_reservation_holder = 1 WHERE job_id = ?",
        (JobName.from_string("/test-user/j").to_wire(),),
    )
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.resource_usage_by_worker(legacy_tx)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.resource_usage_by_worker(sa_tx)
    assert legacy == sa == {}


def test_reconcile_rows_for_workers_empty_parity(state):
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.reconcile_rows_for_workers(legacy_tx, [])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.reconcile_rows_for_workers(sa_tx, [])
    assert legacy == sa == []


def test_reconcile_rows_for_workers_populated_parity(state):
    wid, task_id = _seed_one_running_task(state)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.reconcile_rows_for_workers(legacy_tx, [wid])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.reconcile_rows_for_workers(sa_tx, [wid])
    assert legacy == sa
    assert len(legacy) == 1
    assert legacy[0].worker_id == wid
    assert legacy[0].task_id == task_id


def test_reconcile_rows_filters_unknown_workers(state):
    from iris.cluster.types import WorkerId

    _seed_one_running_task(state)
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.attempts.reconcile_rows_for_workers(legacy_tx, [WorkerId("not-a-real-worker")])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.reconcile_rows_for_workers(sa_tx, [WorkerId("not-a-real-worker")])
    assert legacy == sa == []


def test_running_tasks_by_worker_empty_parity(state):
    legacy = legacy_db.running_tasks_by_worker(state._db, set())
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.running_tasks_by_worker(sa_tx, set())
    assert legacy == sa == {}


def test_running_tasks_by_worker_populated_parity(state):
    wid, task_id = _seed_one_running_task(state)
    legacy = legacy_db.running_tasks_by_worker(state._db, {wid})
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.running_tasks_by_worker(sa_tx, {wid})
    assert legacy == sa
    assert legacy == {wid: {task_id}}


def test_timed_out_executing_tasks_no_timeouts(state):
    _seed_one_running_task(state)
    now = Timestamp.from_ms(10**12)
    legacy = legacy_db.timed_out_executing_tasks(state._db, now)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.timed_out_executing_tasks(sa_tx, now)
    # Default request has no timeout_ms set; neither path returns anything.
    assert [t.task_id for t in legacy] == [t.task_id for t in sa] == []


def test_timed_out_executing_tasks_with_timeout(state):
    _, task_id = _seed_one_running_task(state)
    # Inject a timeout on the job_config row + a started_at on the attempt.
    state._db.execute(
        "UPDATE job_config SET timeout_ms = ? WHERE job_id = ?",
        (1_000, JobName.from_string("/test-user/j").to_wire()),
    )
    # The dispatch helper sets the task's attempt to RUNNING; ensure
    # ``started_at_ms`` is non-null so the timeout pass considers it.
    state._db.execute(
        "UPDATE task_attempts SET started_at_ms = ? WHERE task_id = ?",
        (1_000, task_id.to_wire()),
    )
    now = Timestamp.from_ms(10_000)
    legacy = legacy_db.timed_out_executing_tasks(state._db, now)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_scheduler.timed_out_executing_tasks(sa_tx, now)
    assert {t.task_id for t in legacy} == {t.task_id for t in sa} == {task_id}
