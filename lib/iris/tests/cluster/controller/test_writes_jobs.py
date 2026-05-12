# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``writes/jobs.py`` (Stage 11 of the SA Core migration).

Each test seeds two isolated ``ControllerDB`` instances with identical
state, applies a write through the legacy path on one and through the
SA Core path on the other, then asserts the resulting DB rows match.
This proves the SA write helpers can replace the legacy write methods
call-by-call without behavioral drift. The actual call-site switchover
lands in a later stage.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from iris.cluster.controller import db_v2
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.stores import (
    ControllerStore,
    JobConfigInsertParams,
    JobInsertParams,
)
from iris.cluster.controller.writes import jobs as writes_jobs
from iris.cluster.types import JobName
from iris.rpc import job_pb2


@contextmanager
def _fresh_db():
    """Yield a (db, store) pair on a fresh temp dir, cleaning up on exit."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_writes_test_"))
    db = ControllerDB(db_dir=tmp)
    try:
        yield db, ControllerStore(db)
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def db_pair():
    """Two isolated DB+store pairs for legacy vs SA parity checks."""
    with _fresh_db() as legacy, _fresh_db() as sa:
        yield legacy, sa


def _make_job_params(name: str = "/test-user/j", *, state: int = job_pb2.JOB_STATE_PENDING) -> JobInsertParams:
    return JobInsertParams(
        job_id=JobName.from_string(name),
        user_id="test-user",
        parent_job_id=None,
        root_job_id=name,
        depth=0,
        state=state,
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


def _make_config_params(name: str = "/test-user/j") -> JobConfigInsertParams:
    return JobConfigInsertParams(
        job_id=JobName.from_string(name),
        name="j",
        has_reservation=False,
        res_cpu_millicores=1000,
        res_memory_bytes=1024**3,
        res_disk_bytes=0,
        res_device_json=None,
        constraints_json="[]",
        has_coscheduling=False,
        coscheduling_group_by="",
        scheduling_timeout_ms=None,
        max_task_failures=0,
        entrypoint_json="{}",
        environment_json="{}",
        bundle_id="",
        ports_json="[]",
        max_retries_failure=0,
        max_retries_preemption=0,
        timeout_ms=None,
        preemption_policy=0,
        existing_job_policy=0,
        priority_band=0,
        task_image="",
    )


def _seed_user(store: ControllerStore, user_id: str = "test-user", now_ms: int = 999) -> None:
    with store.transaction() as cur:
        store.jobs.ensure_user(cur, user_id, now_ms)


def _seed_user_sa(db: ControllerDB, user_id: str = "test-user", now_ms: int = 999) -> None:
    with db_v2.write_transaction(db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.ensure_user(tx, user_id, now_ms)


def _dump_table(db: ControllerDB, table: str, *, order_by: str | None = None) -> list[dict]:
    """Read a whole table as a list of dicts via the legacy snapshot."""
    sql = f"SELECT * FROM {table}"
    if order_by is not None:
        sql += f" ORDER BY {order_by}"
    with db.read_snapshot() as q:
        return [dict(row) for row in q.fetchall(sql)]


# --- ensure_user ------------------------------------------------------------


def test_ensure_user_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    with legacy_store.transaction() as cur:
        legacy_store.jobs.ensure_user(cur, "user-a", 1234)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.ensure_user(tx, "user-a", 1234)

    assert _dump_table(legacy_db, "users", order_by="user_id") == _dump_table(sa_db, "users", order_by="user_id")


def test_ensure_user_is_idempotent(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    with legacy_store.transaction() as cur:
        legacy_store.jobs.ensure_user(cur, "user-a", 1)
        legacy_store.jobs.ensure_user(cur, "user-a", 2)  # second call no-ops
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.ensure_user(tx, "user-a", 1)
        writes_jobs.ensure_user(tx, "user-a", 2)

    legacy_rows = _dump_table(legacy_db, "users", order_by="user_id")
    sa_rows = _dump_table(sa_db, "users", order_by="user_id")
    assert legacy_rows == sa_rows
    assert len(legacy_rows) == 1
    assert legacy_rows[0]["created_at_ms"] == 1


# --- insert_job + insert_job_config -----------------------------------------


def test_insert_job_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    _seed_user(legacy_store)
    _seed_user_sa(sa_db)

    params = _make_job_params()
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)

    assert _dump_table(legacy_db, "jobs", order_by="job_id") == _dump_table(sa_db, "jobs", order_by="job_id")


def test_insert_job_with_holder_flag(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)

    params = JobInsertParams(
        job_id=JobName.from_string("/test-user/holder"),
        user_id="test-user",
        parent_job_id=None,
        root_job_id="/test-user/holder",
        depth=0,
        state=job_pb2.JOB_STATE_PENDING,
        submitted_at_ms=10,
        root_submitted_at_ms=10,
        started_at_ms=None,
        finished_at_ms=None,
        scheduling_deadline_epoch_ms=None,
        error=None,
        exit_code=None,
        num_tasks=0,
        is_reservation_holder=True,
        name="holder",
        has_reservation=True,
    )
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)

    legacy_rows = _dump_table(legacy_db, "jobs", order_by="job_id")
    sa_rows = _dump_table(sa_db, "jobs", order_by="job_id")
    assert legacy_rows == sa_rows
    assert legacy_rows[0]["is_reservation_holder"] == 1
    assert legacy_rows[0]["has_reservation"] == 1


def test_insert_job_config_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)

    job_params = _make_job_params()
    cfg_params = _make_config_params()
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, job_params)
        legacy_store.jobs.insert_config(cur, cfg_params)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, job_params)
        writes_jobs.insert_job_config(tx, cfg_params)

    assert _dump_table(legacy_db, "job_config", order_by="job_id") == _dump_table(sa_db, "job_config", order_by="job_id")


# --- delete_job -------------------------------------------------------------


def test_delete_job_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    job_params = _make_job_params()
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, job_params)
        legacy_store.jobs.delete(cur, job_params.job_id)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, job_params)
        writes_jobs.delete_job(tx, job_params.job_id)

    assert _dump_table(legacy_db, "jobs") == [] == _dump_table(sa_db, "jobs")


# --- insert_workdir_files ---------------------------------------------------


def test_insert_workdir_files_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    params = _make_job_params()
    files = {"main.py": b"print('hi')", "data.bin": b"\x00\x01\x02"}
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
        legacy_store.jobs.insert_workdir_files(cur, params.job_id, files)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)
        writes_jobs.insert_workdir_files(tx, params.job_id, files)

    assert _dump_table(legacy_db, "job_workdir_files", order_by="filename") == _dump_table(
        sa_db, "job_workdir_files", order_by="filename"
    )


def test_insert_workdir_files_empty_is_noop(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    params = _make_job_params()
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
        legacy_store.jobs.insert_workdir_files(cur, params.job_id, {})
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)
        writes_jobs.insert_workdir_files(tx, params.job_id, {})

    assert _dump_table(legacy_db, "job_workdir_files") == [] == _dump_table(sa_db, "job_workdir_files")


# --- update_state_if_not_terminal ------------------------------------------


def test_update_state_if_not_terminal_active_row(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    params = _make_job_params()
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
        legacy_store.jobs.update_state_if_not_terminal(
            cur, params.job_id, job_pb2.JOB_STATE_RUNNING, error=None, finished_at_ms=None
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)
        writes_jobs.update_state_if_not_terminal(
            tx, params.job_id, job_pb2.JOB_STATE_RUNNING, error=None, finished_at_ms=None
        )

    assert _dump_table(legacy_db, "jobs") == _dump_table(sa_db, "jobs")


def test_update_state_if_not_terminal_skips_terminal(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    params = _make_job_params(state=job_pb2.JOB_STATE_FAILED)
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
        legacy_store.jobs.update_state_if_not_terminal(
            cur, params.job_id, job_pb2.JOB_STATE_RUNNING, error="x", finished_at_ms=42
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)
        writes_jobs.update_state_if_not_terminal(
            tx, params.job_id, job_pb2.JOB_STATE_RUNNING, error="x", finished_at_ms=42
        )

    legacy_rows = _dump_table(legacy_db, "jobs")
    sa_rows = _dump_table(sa_db, "jobs")
    assert legacy_rows == sa_rows
    # Row stayed FAILED — guard worked.
    assert legacy_rows[0]["state"] == job_pb2.JOB_STATE_FAILED


# --- bulk_update_state ------------------------------------------------------


def test_bulk_update_state_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    ja = _make_job_params("/test-user/a")
    jb = _make_job_params("/test-user/b", state=job_pb2.JOB_STATE_SUCCEEDED)
    for store_or_db, fn in (
        (legacy_store, lambda cur: legacy_store.jobs.insert(cur, ja)),
        (legacy_store, lambda cur: legacy_store.jobs.insert(cur, jb)),
    ):
        with store_or_db.transaction() as cur:
            fn(cur)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, ja)
        writes_jobs.insert_job(tx, jb)

    ids = [ja.job_id, jb.job_id]
    with legacy_store.transaction() as cur:
        legacy_store.jobs.bulk_update_state(
            cur,
            ids,
            new_state=job_pb2.JOB_STATE_KILLED,
            error="bulk",
            finished_at_ms=99,
            guard_states=(job_pb2.JOB_STATE_SUCCEEDED,),
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.bulk_update_state(
            tx,
            ids,
            new_state=job_pb2.JOB_STATE_KILLED,
            error="bulk",
            finished_at_ms=99,
            guard_states=(job_pb2.JOB_STATE_SUCCEEDED,),
        )

    assert _dump_table(legacy_db, "jobs", order_by="job_id") == _dump_table(sa_db, "jobs", order_by="job_id")


def test_bulk_update_state_empty_is_noop(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    with legacy_store.transaction() as cur:
        legacy_store.jobs.bulk_update_state(
            cur, [], new_state=job_pb2.JOB_STATE_FAILED, error=None, finished_at_ms=None, guard_states=()
        )
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.bulk_update_state(
            tx, [], new_state=job_pb2.JOB_STATE_FAILED, error=None, finished_at_ms=None, guard_states=()
        )

    assert _dump_table(legacy_db, "jobs") == [] == _dump_table(sa_db, "jobs")


# --- mark_running_if_pending -----------------------------------------------


def test_mark_running_if_pending_from_pending(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    params = _make_job_params()
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
        legacy_store.jobs.mark_running_if_pending(cur, params.job_id, now_ms=500)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)
        writes_jobs.mark_running_if_pending(tx, params.job_id, now_ms=500)

    assert _dump_table(legacy_db, "jobs") == _dump_table(sa_db, "jobs")


def test_mark_running_if_pending_already_running(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    params = _make_job_params(state=job_pb2.JOB_STATE_RUNNING)
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
        legacy_store.jobs.mark_running_if_pending(cur, params.job_id, now_ms=500)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)
        writes_jobs.mark_running_if_pending(tx, params.job_id, now_ms=500)

    assert _dump_table(legacy_db, "jobs") == _dump_table(sa_db, "jobs")


# --- apply_recomputed_state ------------------------------------------------


@pytest.mark.parametrize(
    "new_state,error",
    [
        (job_pb2.JOB_STATE_RUNNING, None),
        (job_pb2.JOB_STATE_SUCCEEDED, None),
        (job_pb2.JOB_STATE_FAILED, "boom"),
        (job_pb2.JOB_STATE_KILLED, "killed"),
        (job_pb2.JOB_STATE_UNSCHEDULABLE, "no slot"),
        (job_pb2.JOB_STATE_WORKER_FAILED, "worker"),
    ],
)
def test_apply_recomputed_state_parity(db_pair, new_state, error):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    _seed_user(legacy_store)
    _seed_user_sa(sa_db)
    params = _make_job_params()
    with legacy_store.transaction() as cur:
        legacy_store.jobs.insert(cur, params)
        legacy_store.jobs.apply_recomputed_state(cur, params.job_id, new_state, now_ms=777, error=error)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_jobs.insert_job(tx, params)
        writes_jobs.apply_recomputed_state(tx, params.job_id, new_state, now_ms=777, error=error)

    assert _dump_table(legacy_db, "jobs") == _dump_table(sa_db, "jobs")


# --- reserve_priority_insertion_base ---------------------------------------


def test_reserve_priority_insertion_base_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa

    with legacy_store.transaction() as cur:
        legacy_first = legacy_store.jobs.reserve_priority_insertion_base(cur)
        legacy_second = legacy_store.jobs.reserve_priority_insertion_base(cur)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        sa_first = writes_jobs.reserve_priority_insertion_base(tx)
        sa_second = writes_jobs.reserve_priority_insertion_base(tx)

    assert legacy_first == sa_first == 1
    assert legacy_second == sa_second == 2
    assert _dump_table(legacy_db, "meta", order_by="key") == _dump_table(sa_db, "meta", order_by="key")
