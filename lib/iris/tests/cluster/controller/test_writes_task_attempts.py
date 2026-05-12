# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``writes/task_attempts.py`` (Stage 11 of the SA Core migration)."""

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
    TaskAttemptInsertParams,
    TaskAttemptUpdateParams,
    TaskInsertParams,
)
from iris.cluster.controller.writes import jobs as writes_jobs
from iris.cluster.controller.writes import task_attempts as writes_attempts
from iris.cluster.controller.writes import tasks as writes_tasks
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2


@contextmanager
def _fresh_db():
    tmp = Path(tempfile.mkdtemp(prefix="iris_writes_att_"))
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


def _job_params() -> JobInsertParams:
    return JobInsertParams(
        job_id=JobName.from_string("/test-user/j"),
        user_id="test-user",
        parent_job_id=None,
        root_job_id="/test-user/j",
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


def _task_params(name: str = "/test-user/j/0", *, task_index: int = 0) -> TaskInsertParams:
    return TaskInsertParams(
        task_id=JobName.from_string(name),
        job_id=JobName.from_string("/test-user/j"),
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


def _seed_legacy(store: ControllerStore, *, task_names: list[str] | None = None) -> None:
    with store.transaction() as cur:
        store.jobs.ensure_user(cur, "test-user", 1)
        store.jobs.insert(cur, _job_params())
        for i, name in enumerate(task_names or ["/test-user/j/0"]):
            store.tasks.insert(cur, _task_params(name, task_index=i))


def _seed_sa(db: ControllerDB, *, task_names: list[str] | None = None) -> None:
    with db_v2.write_transaction(db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.ensure_user(tx, "test-user", 1)
        writes_jobs.insert_job(tx, _job_params())
        for i, name in enumerate(task_names or ["/test-user/j/0"]):
            writes_tasks.insert_task(tx, _task_params(name, task_index=i))


def _seed_worker(db: ControllerDB, worker_id: str = "w1") -> None:
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


# --- insert_attempt ---------------------------------------------------------


@pytest.mark.parametrize("worker_id", ["w1", None])
def test_insert_attempt_parity(db_pair, worker_id):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_legacy(legacy_store)
    _seed_sa(sa_db)
    if worker_id is not None:
        _seed_worker(legacy_db, worker_id)
        _seed_worker(sa_db, worker_id)
    wid = WorkerId(worker_id) if worker_id else None

    params = TaskAttemptInsertParams(
        task_id=JobName.from_string("/test-user/j/0"),
        attempt_id=0,
        worker_id=wid,
        state=job_pb2.TASK_STATE_ASSIGNED,
        created_at_ms=2000,
    )
    with legacy_store.transaction() as cur:
        legacy_store.attempts.insert(cur, params)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_attempts.insert_attempt(tx, params)

    assert _dump(legacy_db, "task_attempts", order_by="task_id, attempt_id") == _dump(
        sa_db, "task_attempts", order_by="task_id, attempt_id"
    )


# --- mark_finished ----------------------------------------------------------


def test_mark_finished_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_legacy(legacy_store)
    _seed_sa(sa_db)
    _seed_worker(legacy_db)
    _seed_worker(sa_db)

    params = TaskAttemptInsertParams(
        task_id=JobName.from_string("/test-user/j/0"),
        attempt_id=0,
        worker_id=WorkerId("w1"),
        state=job_pb2.TASK_STATE_RUNNING,
        created_at_ms=2000,
    )
    with legacy_store.transaction() as cur:
        legacy_store.attempts.insert(cur, params)
        legacy_store.attempts.mark_finished(
            cur,
            params.task_id,
            params.attempt_id,
            job_pb2.TASK_STATE_SUCCEEDED,
            3000,
            error=None,
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_attempts.insert_attempt(tx, params)
        writes_attempts.mark_finished(
            tx,
            params.task_id,
            params.attempt_id,
            job_pb2.TASK_STATE_SUCCEEDED,
            3000,
            error=None,
        )

    assert _dump(legacy_db, "task_attempts") == _dump(sa_db, "task_attempts")


# --- apply_attempt_state ----------------------------------------------------


def test_apply_attempt_state_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_legacy(legacy_store)
    _seed_sa(sa_db)
    _seed_worker(legacy_db)
    _seed_worker(sa_db)

    params = TaskAttemptInsertParams(
        task_id=JobName.from_string("/test-user/j/0"),
        attempt_id=0,
        worker_id=WorkerId("w1"),
        state=job_pb2.TASK_STATE_RUNNING,
        created_at_ms=2000,
    )
    with legacy_store.transaction() as cur:
        legacy_store.attempts.insert(cur, params)
        legacy_store.attempts.apply_attempt_state(
            cur,
            params.task_id,
            params.attempt_id,
            job_pb2.TASK_STATE_KILLED,
            error="cancel",
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_attempts.insert_attempt(tx, params)
        writes_attempts.apply_attempt_state(
            tx,
            params.task_id,
            params.attempt_id,
            job_pb2.TASK_STATE_KILLED,
            error="cancel",
        )

    assert _dump(legacy_db, "task_attempts") == _dump(sa_db, "task_attempts")


# --- apply_update -----------------------------------------------------------


def test_apply_update_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_legacy(legacy_store)
    _seed_sa(sa_db)
    _seed_worker(legacy_db)
    _seed_worker(sa_db)

    base = TaskAttemptInsertParams(
        task_id=JobName.from_string("/test-user/j/0"),
        attempt_id=0,
        worker_id=WorkerId("w1"),
        state=job_pb2.TASK_STATE_ASSIGNED,
        created_at_ms=2000,
    )
    update = TaskAttemptUpdateParams(
        task_id=base.task_id,
        attempt_id=base.attempt_id,
        state=job_pb2.TASK_STATE_RUNNING,
        started_at_ms=2500,
        finished_at_ms=None,
        exit_code=None,
        error=None,
    )
    with legacy_store.transaction() as cur:
        legacy_store.attempts.insert(cur, base)
        legacy_store.attempts.apply_update(cur, update)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_attempts.insert_attempt(tx, base)
        writes_attempts.apply_update(tx, update)

    assert _dump(legacy_db, "task_attempts") == _dump(sa_db, "task_attempts")


# --- bulk_apply_attempt_state ----------------------------------------------


def test_bulk_apply_attempt_state_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_legacy(legacy_store, task_names=["/test-user/j/0", "/test-user/j/1"])
    _seed_sa(sa_db, task_names=["/test-user/j/0", "/test-user/j/1"])
    _seed_worker(legacy_db)
    _seed_worker(sa_db)

    a0 = TaskAttemptInsertParams(
        task_id=JobName.from_string("/test-user/j/0"),
        attempt_id=0,
        worker_id=WorkerId("w1"),
        state=job_pb2.TASK_STATE_RUNNING,
        created_at_ms=2000,
    )
    a1 = TaskAttemptInsertParams(
        task_id=JobName.from_string("/test-user/j/1"),
        attempt_id=0,
        worker_id=WorkerId("w1"),
        state=job_pb2.TASK_STATE_RUNNING,
        created_at_ms=2000,
    )
    job_id = JobName.from_string("/test-user/j")
    active = set(ACTIVE_TASK_STATES)
    with legacy_store.transaction() as cur:
        legacy_store.attempts.insert(cur, a0)
        legacy_store.attempts.insert(cur, a1)
        legacy_store.attempts.bulk_apply_attempt_state(
            cur,
            [job_id],
            job_pb2.TASK_STATE_KILLED,
            "cancel",
            active,
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_attempts.insert_attempt(tx, a0)
        writes_attempts.insert_attempt(tx, a1)
        writes_attempts.bulk_apply_attempt_state(
            tx,
            [job_id],
            job_pb2.TASK_STATE_KILLED,
            "cancel",
            active,
        )

    assert _dump(legacy_db, "task_attempts", order_by="task_id, attempt_id") == _dump(
        sa_db, "task_attempts", order_by="task_id, attempt_id"
    )


def test_bulk_apply_attempt_state_empty_is_noop(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_legacy(legacy_store)
    _seed_sa(sa_db)

    with legacy_store.transaction() as cur:
        legacy_store.attempts.bulk_apply_attempt_state(
            cur,
            [],
            job_pb2.TASK_STATE_KILLED,
            "x",
            set(ACTIVE_TASK_STATES),
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_attempts.bulk_apply_attempt_state(
            tx,
            [],
            job_pb2.TASK_STATE_KILLED,
            "x",
            set(ACTIVE_TASK_STATES),
        )

    assert _dump(legacy_db, "task_attempts") == [] == _dump(sa_db, "task_attempts")
