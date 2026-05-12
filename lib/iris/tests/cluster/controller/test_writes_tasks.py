# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``writes/tasks.py`` (Stage 11 of the SA Core migration)."""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from iris.cluster.controller import db_v2
from iris.cluster.controller.db import ACTIVE_TASK_STATES, ControllerDB
from iris.cluster.controller.stores import (
    ControllerStore,
    JobInsertParams,
    TaskInsertParams,
    TaskStateUpdateParams,
)
from iris.cluster.controller.writes import jobs as writes_jobs
from iris.cluster.controller.writes import tasks as writes_tasks
from iris.cluster.types import TERMINAL_TASK_STATES, JobName, WorkerId
from iris.rpc import job_pb2


@contextmanager
def _fresh_db():
    tmp = Path(tempfile.mkdtemp(prefix="iris_writes_tasks_"))
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


def _make_job_params(name: str = "/test-user/j") -> JobInsertParams:
    return JobInsertParams(
        job_id=JobName.from_string(name),
        user_id="test-user",
        parent_job_id=None,
        root_job_id=name,
        depth=0,
        state=job_pb2.JOB_STATE_PENDING,
        submitted_at_ms=1000,
        root_submitted_at_ms=1000,
        started_at_ms=None,
        finished_at_ms=None,
        scheduling_deadline_epoch_ms=None,
        error=None,
        exit_code=None,
        num_tasks=1,
        is_reservation_holder=False,
        name="j",
        has_reservation=False,
    )


def _make_task_params(task_name: str, job_name: str = "/test-user/j", *, task_index: int = 0) -> TaskInsertParams:
    return TaskInsertParams(
        task_id=JobName.from_string(task_name),
        job_id=JobName.from_string(job_name),
        task_index=task_index,
        state=job_pb2.TASK_STATE_PENDING,
        submitted_at_ms=1000,
        max_retries_failure=0,
        max_retries_preemption=0,
        priority_neg_depth=0,
        priority_root_submitted_ms=1000,
        priority_insertion=0,
        priority_band=0,
    )


def _seed_job_and_user(store: ControllerStore, db: ControllerDB | None = None) -> None:
    """Insert user + job row via legacy path on whichever DB owns ``store``.

    ``db`` is unused; ``store`` already references its own DB.
    """
    del db
    with store.transaction() as cur:
        store.jobs.ensure_user(cur, "test-user", 1)
        store.jobs.insert(cur, _make_job_params())


def _seed_job_and_user_sa(db: ControllerDB) -> None:
    with db_v2.write_transaction(db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.ensure_user(tx, "test-user", 1)
        writes_jobs.insert_job(tx, _make_job_params())


def _seed_worker(db: ControllerDB, worker_id: str = "w1") -> None:
    """Insert a bare ``workers`` row so FKs on ``tasks.current_worker_id`` resolve."""
    with db.transaction() as cur:
        cur.execute(
            "INSERT INTO workers(worker_id, address) VALUES (?, ?)",
            (worker_id, f"{worker_id}:8080"),
        )


def _dump(db: ControllerDB, table: str, *, order_by: str | None = None) -> list[dict]:
    sql = f"SELECT * FROM {table}"
    if order_by is not None:
        sql += f" ORDER BY {order_by}"
    with db.read_snapshot() as q:
        return [dict(row) for row in q.fetchall(sql)]


# --- insert_task ------------------------------------------------------------


def test_insert_task_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_job_and_user(legacy_store)
    _seed_job_and_user_sa(sa_db)

    params = _make_task_params("/test-user/j/0")
    with legacy_store.transaction() as cur:
        legacy_store.tasks.insert(cur, params)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_tasks.insert_task(tx, params)

    assert _dump(legacy_db, "tasks", order_by="task_id") == _dump(sa_db, "tasks", order_by="task_id")


# --- mark_assigned / assign_task -------------------------------------------


@pytest.mark.parametrize("worker_id", ["w1", None])
@pytest.mark.parametrize("priority_band", [None, 5])
def test_mark_assigned_parity(db_pair, worker_id, priority_band):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_job_and_user(legacy_store)
    _seed_job_and_user_sa(sa_db)
    if worker_id is not None:
        _seed_worker(legacy_db, worker_id)
        _seed_worker(sa_db, worker_id)

    task_params = _make_task_params("/test-user/j/0")
    wid = WorkerId(worker_id) if worker_id else None
    with legacy_store.transaction() as cur:
        legacy_store.tasks.insert(cur, task_params)
        legacy_store.tasks.mark_assigned(
            cur,
            task_params.task_id,
            attempt_id=0,
            worker_id=wid,
            worker_address="addr:1",
            now_ms=2000,
            priority_band=priority_band,
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_tasks.insert_task(tx, task_params)
        writes_tasks.mark_assigned(
            tx,
            task_params.task_id,
            attempt_id=0,
            worker_id=wid,
            worker_address="addr:1",
            now_ms=2000,
            priority_band=priority_band,
        )

    assert _dump(legacy_db, "tasks") == _dump(sa_db, "tasks")


def test_assign_task_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_job_and_user(legacy_store)
    _seed_job_and_user_sa(sa_db)
    _seed_worker(legacy_db)
    _seed_worker(sa_db)

    task_params = _make_task_params("/test-user/j/0")
    wid = WorkerId("w1")
    with legacy_store.transaction() as cur:
        legacy_store.tasks.insert(cur, task_params)
        legacy_store.tasks.assign(
            cur,
            legacy_store.attempts,
            task_params.task_id,
            wid,
            "addr:1",
            attempt_id=0,
            now_ms=2000,
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_tasks.insert_task(tx, task_params)
        writes_tasks.assign_task(tx, task_params.task_id, wid, "addr:1", attempt_id=0, now_ms=2000)

    assert _dump(legacy_db, "tasks") == _dump(sa_db, "tasks")
    assert _dump(legacy_db, "task_attempts", order_by="task_id, attempt_id") == _dump(
        sa_db, "task_attempts", order_by="task_id, attempt_id"
    )


# --- apply_state_update -----------------------------------------------------


@pytest.mark.parametrize(
    "new_state",
    [job_pb2.TASK_STATE_RUNNING, job_pb2.TASK_STATE_SUCCEEDED, job_pb2.TASK_STATE_FAILED],
)
def test_apply_state_update_parity(db_pair, new_state):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_job_and_user(legacy_store)
    _seed_job_and_user_sa(sa_db)
    _seed_worker(legacy_db)
    _seed_worker(sa_db)

    task_params = _make_task_params("/test-user/j/0")
    state_params = TaskStateUpdateParams(
        task_id=task_params.task_id,
        state=new_state,
        error="some-error" if new_state in TERMINAL_TASK_STATES else None,
        exit_code=1 if new_state == job_pb2.TASK_STATE_FAILED else None,
        started_at_ms=1500,
        finished_at_ms=2000 if new_state in TERMINAL_TASK_STATES else None,
        failure_count=1,
        preemption_count=0,
    )
    active = set(ACTIVE_TASK_STATES)
    with legacy_store.transaction() as cur:
        legacy_store.tasks.insert(cur, task_params)
        legacy_store.tasks.mark_assigned(
            cur,
            task_params.task_id,
            attempt_id=0,
            worker_id=WorkerId("w1"),
            worker_address="a",
            now_ms=100,
        )
        legacy_store.tasks.apply_state_update(cur, state_params, active)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_tasks.insert_task(tx, task_params)
        writes_tasks.mark_assigned(
            tx,
            task_params.task_id,
            attempt_id=0,
            worker_id=WorkerId("w1"),
            worker_address="a",
            now_ms=100,
        )
        writes_tasks.apply_state_update(tx, state_params, active)

    assert _dump(legacy_db, "tasks") == _dump(sa_db, "tasks")


# --- mark_terminal ----------------------------------------------------------


@pytest.mark.parametrize("with_counts", [False, True])
@pytest.mark.parametrize("finished_at_ms", [None, 2000])
def test_mark_terminal_parity(db_pair, with_counts, finished_at_ms):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_job_and_user(legacy_store)
    _seed_job_and_user_sa(sa_db)
    _seed_worker(legacy_db)
    _seed_worker(sa_db)

    task_params = _make_task_params("/test-user/j/0")
    active = set(ACTIVE_TASK_STATES)
    kwargs: dict = {"active_states": active}
    if with_counts:
        kwargs["failure_count"] = 2
        kwargs["preemption_count"] = 1
    with legacy_store.transaction() as cur:
        legacy_store.tasks.insert(cur, task_params)
        legacy_store.tasks.mark_assigned(
            cur,
            task_params.task_id,
            attempt_id=0,
            worker_id=WorkerId("w1"),
            worker_address="a",
            now_ms=100,
        )
        legacy_store.tasks.mark_terminal(
            cur,
            task_params.task_id,
            job_pb2.TASK_STATE_FAILED,
            "err",
            finished_at_ms,
            **kwargs,
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_tasks.insert_task(tx, task_params)
        writes_tasks.mark_assigned(
            tx,
            task_params.task_id,
            attempt_id=0,
            worker_id=WorkerId("w1"),
            worker_address="a",
            now_ms=100,
        )
        writes_tasks.mark_terminal(
            tx,
            task_params.task_id,
            job_pb2.TASK_STATE_FAILED,
            "err",
            finished_at_ms,
            **kwargs,
        )

    assert _dump(legacy_db, "tasks") == _dump(sa_db, "tasks")


# --- bulk_kill_non_terminal ------------------------------------------------


def test_bulk_kill_non_terminal_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_job_and_user(legacy_store)
    _seed_job_and_user_sa(sa_db)

    t0 = _make_task_params("/test-user/j/0", task_index=0)
    t1 = _make_task_params("/test-user/j/1", task_index=1)
    with legacy_store.transaction() as cur:
        legacy_store.tasks.insert(cur, t0)
        legacy_store.tasks.insert(cur, t1)
        legacy_store.tasks.bulk_kill_non_terminal(
            cur,
            [t0.job_id],
            reason="cancelled",
            finished_at_ms=3000,
            terminal_states=set(TERMINAL_TASK_STATES),
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_tasks.insert_task(tx, t0)
        writes_tasks.insert_task(tx, t1)
        writes_tasks.bulk_kill_non_terminal(
            tx,
            [t0.job_id],
            reason="cancelled",
            finished_at_ms=3000,
            terminal_states=set(TERMINAL_TASK_STATES),
        )

    assert _dump(legacy_db, "tasks", order_by="task_id") == _dump(sa_db, "tasks", order_by="task_id")


def test_bulk_kill_non_terminal_empty_is_noop(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_job_and_user(legacy_store)
    _seed_job_and_user_sa(sa_db)
    with legacy_store.transaction() as cur:
        legacy_store.tasks.bulk_kill_non_terminal(
            cur,
            [],
            reason="x",
            finished_at_ms=1,
            terminal_states=set(TERMINAL_TASK_STATES),
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_tasks.bulk_kill_non_terminal(
            tx,
            [],
            reason="x",
            finished_at_ms=1,
            terminal_states=set(TERMINAL_TASK_STATES),
        )
    assert _dump(legacy_db, "tasks") == [] == _dump(sa_db, "tasks")


# --- update_container_id ---------------------------------------------------


def test_update_container_id_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_job_and_user(legacy_store)
    _seed_job_and_user_sa(sa_db)

    task_params = _make_task_params("/test-user/j/0")
    with legacy_store.transaction() as cur:
        legacy_store.tasks.insert(cur, task_params)
        legacy_store.tasks.update_container_id(cur, task_params.task_id, "container-abc")
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_tasks.insert_task(tx, task_params)
        writes_tasks.update_container_id(tx, task_params.task_id, "container-abc")

    assert _dump(legacy_db, "tasks") == _dump(sa_db, "tasks")
