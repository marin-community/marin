# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``writes/workers.py`` (Stage 11 of the SA Core migration)."""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from iris.cluster.controller import db_v2
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.stores import ControllerStore, WorkerUpsertParams
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.controller.writes import workers as writes_workers
from iris.cluster.types import WorkerId


@contextmanager
def _fresh_db():
    tmp = Path(tempfile.mkdtemp(prefix="iris_writes_w_"))
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


def _worker_params(worker_id: str = "w1") -> WorkerUpsertParams:
    return WorkerUpsertParams(
        worker_id=WorkerId(worker_id),
        address=f"{worker_id}:8080",
        total_cpu_millicores=16000,
        total_memory_bytes=32 * 1024**3,
        total_gpu_count=0,
        total_tpu_count=0,
        device_type="cpu",
        device_variant="cpu",
        slice_id="",
        scale_group="",
        md_hostname="host",
        md_ip_address="127.0.0.1",
        md_cpu_count=16,
        md_memory_bytes=32 * 1024**3,
        md_disk_bytes=100 * 1024**3,
        md_tpu_name="",
        md_tpu_worker_hostnames="",
        md_tpu_worker_id="",
        md_tpu_chips_per_host_bounds="",
        md_gpu_count=0,
        md_gpu_name="",
        md_gpu_memory_mb=0,
        md_gce_instance_name="",
        md_gce_zone="",
        md_git_hash="abc",
        md_device_json="{}",
    )


def _dump(db: ControllerDB, table: str, *, order_by: str | None = None) -> list[dict]:
    sql = f"SELECT * FROM {table}"
    if order_by is not None:
        sql += f" ORDER BY {order_by}"
    with db.read_snapshot() as q:
        return [dict(row) for row in q.fetchall(sql)]


# --- upsert_worker ----------------------------------------------------------


def test_upsert_worker_insert_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    params = _worker_params()

    with legacy_store.transaction() as cur:
        legacy_store.workers.upsert(cur, params, now_ms=1000)
    sa_health = WorkerHealthTracker()
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_workers.upsert_worker(tx, params, now_ms=1000, health=sa_health)

    assert _dump(legacy_db, "workers", order_by="worker_id") == _dump(sa_db, "workers", order_by="worker_id")
    # Post-commit hook fires for both paths.
    assert sa_health.liveness(WorkerId("w1")).last_heartbeat_ms == 1000


def test_upsert_worker_update_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    params = _worker_params()

    with legacy_store.transaction() as cur:
        legacy_store.workers.upsert(cur, params, now_ms=1000)
    sa_health = WorkerHealthTracker()
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_workers.upsert_worker(tx, params, now_ms=1000, health=sa_health)

    # Now update with a different address; the ON CONFLICT path triggers.
    from dataclasses import replace

    refreshed = replace(params, address="new-addr:9090", total_cpu_millicores=32000)
    with legacy_store.transaction() as cur:
        legacy_store.workers.upsert(cur, refreshed, now_ms=2000)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_workers.upsert_worker(tx, refreshed, now_ms=2000, health=sa_health)

    legacy_rows = _dump(legacy_db, "workers")
    sa_rows = _dump(sa_db, "workers")
    assert legacy_rows == sa_rows
    assert legacy_rows[0]["address"] == "new-addr:9090"
    assert legacy_rows[0]["total_cpu_millicores"] == 32000


# --- remove_worker ----------------------------------------------------------


def test_remove_worker_parity(db_pair):
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    params = _worker_params()

    with legacy_store.transaction() as cur:
        legacy_store.workers.upsert(cur, params, now_ms=1000)
        legacy_store.workers.remove(cur, params.worker_id)
    sa_health = WorkerHealthTracker()
    sa_attrs = WorkerAttrsProjection(sa_db)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_workers.upsert_worker(tx, params, now_ms=1000, health=sa_health)
        writes_workers.remove_worker(tx, params.worker_id, health=sa_health, worker_attrs=sa_attrs)

    assert _dump(legacy_db, "workers") == [] == _dump(sa_db, "workers")
    # The post-commit ``forget`` hook clears the tracker entry; a default
    # (untracked) ``WorkerLiveness`` is returned for the forgotten id.
    from iris.cluster.controller.worker_health import WorkerLiveness

    assert sa_health.liveness(WorkerId("w1")) == WorkerLiveness()


def test_remove_worker_nulls_attempts_and_tasks(db_pair):
    """Verify the pre-cascade UPDATEs leave parity (worker_id → NULL on attempts/tasks)."""
    legacy, sa = db_pair
    legacy_db, legacy_store = legacy
    sa_db, _ = sa
    params = _worker_params()

    # Seed identical state on both DBs: user, job, task, attempt holding worker_id.
    setup_sql = [
        ("INSERT INTO users(user_id, created_at_ms) VALUES (?, ?)", ("u", 1)),
        (
            "INSERT INTO jobs(job_id, user_id, root_job_id, depth, state, "
            "submitted_at_ms, root_submitted_at_ms, num_tasks, is_reservation_holder, name, has_reservation) "
            "VALUES (?, ?, ?, 0, 0, 1, 1, 1, 0, '', 0)",
            ("/u/j", "u", "/u/j"),
        ),
        (
            "INSERT INTO tasks(task_id, job_id, task_index, state, submitted_at_ms, "
            "max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
            "current_attempt_id, priority_neg_depth, priority_root_submitted_ms, priority_insertion, "
            "priority_band, current_worker_id) VALUES (?, ?, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ?)",
            ("/u/j/0", "/u/j", "w1"),
        ),
        (
            "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) " "VALUES (?, 0, ?, 0, 1)",
            ("/u/j/0", "w1"),
        ),
    ]

    for db, store in (legacy, sa):
        with db.transaction() as cur:
            store.workers.upsert(cur, params, now_ms=1000)
            for sql, args in setup_sql:
                cur.execute(sql, args)

    # Now do the remove via legacy on legacy DB; via SA on SA DB.
    with legacy_store.transaction() as cur:
        legacy_store.workers.remove(cur, params.worker_id)
    sa_health = WorkerHealthTracker()
    sa_health.register(params.worker_id, now_ms=1000)
    sa_attrs = WorkerAttrsProjection(sa_db)
    with db_v2.write_transaction(sa_db.sa_write_engine, threading.RLock()) as tx:
        writes_workers.remove_worker(tx, params.worker_id, health=sa_health, worker_attrs=sa_attrs)

    # Workers gone on both.
    assert _dump(legacy_db, "workers") == [] == _dump(sa_db, "workers")
    # ``tasks.current_worker_id`` set to NULL on both.
    assert _dump(legacy_db, "tasks") == _dump(sa_db, "tasks")
    # ``task_attempts.worker_id`` set to NULL on both.
    assert _dump(legacy_db, "task_attempts") == _dump(sa_db, "task_attempts")
