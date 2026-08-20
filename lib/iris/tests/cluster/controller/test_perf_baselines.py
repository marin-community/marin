# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cardinality coverage for controller reads used by large scheduling ticks."""

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from iris.cluster.controller.persistence import reads
from iris.cluster.controller.persistence.database import ControllerDB
from iris.rpc import job_pb2
from sqlalchemy import text

_RESOURCE_WORKER_COUNT = 200
_TASKS_PER_WORKER = 5  # ~1k live attempts total — matches the per-tick mix


def _seed_workers_and_attempts(db: ControllerDB) -> None:
    """Seed 200 workers each with ``_TASKS_PER_WORKER`` running attempts.

    Exercises ``resource_usage_by_worker`` and ``reconcile_rows_for_workers``
    against a realistic per-tick row count (~1k live worker-bound attempts).
    """
    with db.transaction() as cur:
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
                        "  submitted_at_ms, root_submitted_at_ms, num_tasks"
                        ") VALUES (:jid, :uid, :jid, 0, :state, :ts, :ts, 1)"
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
                        "  priority_neg_depth, priority_root_submitted_ms, priority_insertion,"
                        "  current_attempt_id, current_worker_id, current_worker_address"
                        ") VALUES (:tid, :jid, 0, :state, 2000, 0, 0, 0, 2000, 0, 0, :wid, :waddr)"
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
                        "  task_id, attempt_id, worker_id, state, created_at_ms, attempt_uid"
                        ") VALUES (:tid, 0, :wid, :state, 2000, :uid)"
                    ),
                    {
                        "tid": task_id,
                        "wid": worker_id,
                        "state": job_pb2.TASK_STATE_RUNNING,
                        "uid": f"{w_idx:08x}{t_idx:08x}",
                    },
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


def test_resource_usage_by_worker_returns_every_seeded_worker(perf_db: ControllerDB) -> None:
    with perf_db.read_snapshot() as tx:
        usage = reads.resource_usage_by_worker(tx)

    assert len(usage) == _RESOURCE_WORKER_COUNT
