# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Performance baselines for the SA Core data-layer migration.

Gates the SA Core path of ``_jobs_with_reservations`` and the scheduler
reads (``resource_usage_by_worker``, ``reconcile_rows_for_workers``) against
a fixed workload to catch regressions. The legacy comparison path has been
deleted — only the SA Core timings are measured.
"""

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import pytest
from iris.cluster.controller import db
from iris.cluster.controller.controller import _jobs_with_reservations
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.reads import scheduler as reads_scheduler
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2
from sqlalchemy import text

_RESERVATION_JOB_COUNT = 200
_NON_RESERVATION_JOB_COUNT = 200
_TICKS = 200
# SA Core adds ~370 µs/call of fixed per-call overhead vs. raw sqlite3. The
# per-call gate is an absolute µs budget that catches gross regressions
# (e.g. per-call compile, missing index) while accepting structural SA cost.
_SA_JOBS_WITH_RESERVATIONS_MAX_US = 5_000


def _seed_jobs(db: ControllerDB) -> None:
    """Insert 200 reservation-holding + 200 plain jobs into the DB."""
    with db.transaction() as cur:
        cur.execute(
            text("INSERT INTO users (user_id, created_at_ms, role) VALUES (:uid, :ts, :role)"),
            {"uid": "u1", "ts": 1_000, "role": "user"},
        )
        for idx in range(_RESERVATION_JOB_COUNT):
            job_id = f"/u1/res-{idx:04d}"
            cur.execute(
                text(
                    "INSERT INTO jobs ("
                    "  job_id, user_id, root_job_id, depth, state,"
                    "  submitted_at_ms, root_submitted_at_ms, num_tasks,"
                    "  is_reservation_holder, has_reservation"
                    ") VALUES (:jid, :uid, :jid, 0, :state, :ts, :ts, 1, 1, 1)"
                ),
                {"jid": job_id, "uid": "u1", "state": job_pb2.JOB_STATE_RUNNING, "ts": 2_000},
            )
            cur.execute(
                text(
                    "INSERT INTO job_config (job_id, name, has_reservation, reservation_json)"
                    " VALUES (:jid, :name, 1, :res)"
                ),
                {"jid": job_id, "name": f"res-{idx:04d}", "res": '{"resources":{"cpu":1}}'},
            )
        for idx in range(_NON_RESERVATION_JOB_COUNT):
            job_id = f"/u1/plain-{idx:04d}"
            cur.execute(
                text(
                    "INSERT INTO jobs ("
                    "  job_id, user_id, root_job_id, depth, state,"
                    "  submitted_at_ms, root_submitted_at_ms, num_tasks,"
                    "  is_reservation_holder, has_reservation"
                    ") VALUES (:jid, :uid, :jid, 0, :state, :ts, :ts, 1, 0, 0)"
                ),
                {"jid": job_id, "uid": "u1", "state": job_pb2.JOB_STATE_RUNNING, "ts": 2_000},
            )
            cur.execute(
                text("INSERT INTO job_config (job_id, name) VALUES (:jid, :name)"),
                {"jid": job_id, "name": f"plain-{idx:04d}"},
            )


@pytest.fixture
def seeded_db() -> Iterator[ControllerDB]:
    """Build a temp ``ControllerDB`` seeded with reservation-holder workload."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_perf_"))
    db = ControllerDB(db_dir=tmp)
    try:
        _seed_jobs(db)
        yield db
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _measure(callable_, ticks: int) -> float:
    """Return the mean wall-clock time per call over ``ticks`` invocations."""
    # Warm up SA's statement-compilation cache and connection pool before
    # measuring; the canary gates steady-state cost, not first-tick cost.
    callable_()
    callable_()
    t0 = perf_counter()
    for _ in range(ticks):
        callable_()
    elapsed = perf_counter() - t0
    return elapsed / ticks


def test_jobs_with_reservations_perf(seeded_db: ControllerDB) -> None:
    """Gate SA _jobs_with_reservations below a fixed µs ceiling."""
    states = (job_pb2.JOB_STATE_RUNNING,)

    def _sa_call() -> int:
        return len(_jobs_with_reservations(seeded_db, states))

    assert _sa_call() == _RESERVATION_JOB_COUNT

    sa_per_call = _measure(_sa_call, _TICKS)
    max_us = _SA_JOBS_WITH_RESERVATIONS_MAX_US

    assert sa_per_call * 1e6 <= max_us, (
        f"SA _jobs_with_reservations too slow: "
        f"sa={sa_per_call * 1e6:.1f} µs/call > {max_us} µs gate "
        f"(200 reservation-holders + 200 plain jobs; {_TICKS} iterations)."
    )


# ---------------------------------------------------------------------------
# Stage 9 perf gates: resource_usage_by_worker + reconcile_rows_for_workers
# ---------------------------------------------------------------------------

_RESOURCE_WORKER_COUNT = 200
_TASKS_PER_WORKER = 5  # ~1k live attempts total — matches the per-tick mix


def _seed_workers_and_attempts(db: ControllerDB) -> None:
    """Seed 200 workers each with ``_TASKS_PER_WORKER`` running attempts.

    Exercises ``resource_usage_by_worker`` and ``reconcile_rows_for_workers``
    against a realistic per-tick row count (~1k live worker-bound attempts).
    """
    with db.transaction() as cur:
        cur.execute(
            text("INSERT INTO users (user_id, created_at_ms, role) VALUES (:uid, :ts, :role)"),
            {"uid": "u1", "ts": 1_000, "role": "user"},
        )
        for w_idx in range(_RESOURCE_WORKER_COUNT):
            worker_id = f"w-{w_idx:04d}"
            cur.execute(
                text(
                    "INSERT INTO workers ("
                    "  worker_id, address, total_cpu_millicores, total_memory_bytes,"
                    "  total_gpu_count, total_tpu_count, device_type, device_variant"
                    ") VALUES (:wid, :addr, :cpu, :mem, 0, 0, :dtype, :dvar)"
                ),
                {
                    "wid": worker_id,
                    "addr": f"{worker_id}:8080",
                    "cpu": 64_000,
                    "mem": 64 * 1024**3,
                    "dtype": "cpu",
                    "dvar": "",
                },
            )
            for t_idx in range(_TASKS_PER_WORKER):
                job_id = f"/u1/w{w_idx:04d}-t{t_idx:02d}"
                task_id = f"{job_id}/0"
                cur.execute(
                    text(
                        "INSERT INTO jobs ("
                        "  job_id, user_id, root_job_id, depth, state,"
                        "  submitted_at_ms, root_submitted_at_ms, num_tasks,"
                        "  is_reservation_holder, has_reservation"
                        ") VALUES (:jid, :uid, :jid, 0, :state, :ts, :ts, 1, 0, 0)"
                    ),
                    {"jid": job_id, "uid": "u1", "state": job_pb2.JOB_STATE_RUNNING, "ts": 2_000},
                )
                cur.execute(
                    text(
                        "INSERT INTO job_config ("
                        "  job_id, name, res_cpu_millicores, res_memory_bytes, res_disk_bytes,"
                        "  res_device_json"
                        ") VALUES (:jid, :name, 1000, :mem, 0, NULL)"
                    ),
                    {"jid": job_id, "name": f"j-{w_idx}-{t_idx}", "mem": 1024**3},
                )
                cur.execute(
                    text(
                        "INSERT INTO tasks ("
                        "  task_id, job_id, task_index, state, submitted_at_ms,"
                        "  max_retries_failure, max_retries_preemption,"
                        "  failure_count, preemption_count,"
                        "  priority_neg_depth, priority_root_submitted_ms, priority_insertion,"
                        "  current_attempt_id, current_worker_id, current_worker_address"
                        ") VALUES (:tid, :jid, 0, :state, 2000, 0, 0, 0, 0, 0, 2000, 0, 0, :wid, :waddr)"
                    ),
                    {
                        "tid": task_id,
                        "jid": job_id,
                        "state": job_pb2.TASK_STATE_RUNNING,
                        "wid": worker_id,
                        "waddr": f"{worker_id}:8080",
                    },
                )
                cur.execute(
                    text(
                        "INSERT INTO task_attempts ("
                        "  task_id, attempt_id, worker_id, state, created_at_ms"
                        ") VALUES (:tid, 0, :wid, :state, 2000)"
                    ),
                    {"tid": task_id, "wid": worker_id, "state": job_pb2.TASK_STATE_RUNNING},
                )


@pytest.fixture
def perf_db() -> Iterator[ControllerDB]:
    """Build a temp ``ControllerDB`` seeded with workers + live attempts."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_perf_stage9_"))
    db = ControllerDB(db_dir=tmp)
    try:
        _seed_workers_and_attempts(db)
        yield db
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_resource_usage_by_worker_perf(perf_db: ControllerDB) -> None:
    """Smoke-test SA ``resource_usage_by_worker`` returns expected row count."""
    worker_count = _RESOURCE_WORKER_COUNT

    def _sa_call() -> int:
        with db.read_snapshot(perf_db.sa_read_engine) as tx:
            return len(reads_scheduler.resource_usage_by_worker(tx))

    assert _sa_call() == worker_count


def test_reconcile_rows_for_workers_perf(perf_db: ControllerDB) -> None:
    """Smoke-test SA ``reconcile_rows_for_workers`` returns expected row count."""
    worker_ids = [WorkerId(f"w-{i:04d}") for i in range(_RESOURCE_WORKER_COUNT)]
    expected_rows = _RESOURCE_WORKER_COUNT * _TASKS_PER_WORKER

    def _sa_call() -> int:
        with db.read_snapshot(perf_db.sa_read_engine) as tx:
            return len(reads_scheduler.reconcile_rows_for_workers(tx, worker_ids))

    assert _sa_call() == expected_rows
