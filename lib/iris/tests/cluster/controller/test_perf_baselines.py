# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Performance baselines for the SA Core data-layer migration.

Gates the SA Core scheduler reads (``resource_usage_by_worker`` and the inline
reconcile-rows query) against a fixed workload to catch regressions. The legacy
comparison path has been deleted — only the SA Core timings are measured.
"""

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import pytest
from iris.cluster.controller import ops, reads
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import task_attempts_table, tasks_table
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from sqlalchemy import select, text
from tests.cluster.controller._test_support import ControllerTestState

_TICKS = 200


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


def test_resource_usage_by_worker_perf(perf_db: ControllerDB) -> None:
    """Smoke-test SA ``resource_usage_by_worker`` returns expected row count."""
    worker_count = _RESOURCE_WORKER_COUNT

    def _sa_call() -> int:
        with perf_db.read_snapshot() as tx:
            return len(reads.resource_usage_by_worker(tx))

    assert _sa_call() == worker_count


def test_reconcile_rows_for_workers_perf(perf_db: ControllerDB) -> None:
    """Smoke-test the inline reconcile-rows query returns expected row count."""
    worker_ids = [WorkerId(f"w-{i:04d}") for i in range(_RESOURCE_WORKER_COUNT)]
    expected_rows = _RESOURCE_WORKER_COUNT * _TASKS_PER_WORKER

    def _sa_call() -> int:
        target_ids = set(worker_ids)
        with perf_db.read_snapshot() as tx:
            # Worker filter applied in Python to keep the partial index
            # ``idx_task_attempts_live_workerbound`` in play (a long IN list
            # on worker_id degrades to a scan).
            raw_rows = tx.execute(
                select(
                    task_attempts_table.c.worker_id,
                    tasks_table.c.task_id,
                    task_attempts_table.c.attempt_id,
                    tasks_table.c.state.label("task_state"),
                    task_attempts_table.c.state.label("attempt_state"),
                    tasks_table.c.job_id,
                )
                .select_from(
                    task_attempts_table.join(
                        tasks_table,
                        (tasks_table.c.task_id == task_attempts_table.c.task_id)
                        & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
                    )
                )
                .where(
                    task_attempts_table.c.worker_id.is_not(None),
                    task_attempts_table.c.finished_at_ms.is_(None),
                    tasks_table.c.state.in_(
                        [
                            int(job_pb2.TASK_STATE_ASSIGNED),
                            int(job_pb2.TASK_STATE_BUILDING),
                            int(job_pb2.TASK_STATE_RUNNING),
                        ]
                    ),
                ),
            ).all()
            return sum(1 for row in raw_rows if row.worker_id in target_ids)

    assert _sa_call() == expected_rows


# ---------------------------------------------------------------------------
# Tier 3 perf gates: bulk_insert_tasks + batch worker_attributes INSERT
# ---------------------------------------------------------------------------

_REPLICA_COUNT = 32
_ATTR_COUNT = 32
# Thresholds are ~10x the observed median on the development machine.
# Measured medians: submit=8.6ms, register=1.8ms.
# Tightened: reduce these once stable CI baselines are established.
_SUBMIT_32_REPLICAS_MAX_MS = 200
_REGISTER_32_ATTRS_MAX_MS = 50


@pytest.fixture
def submit_perf_db() -> Iterator[ControllerTestState]:
    """Fresh ControllerTestState for bulk-insert submit perf gate."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_perf_submit_"))
    controller_db = ControllerDB(db_dir=tmp)
    try:
        yield ControllerTestState(controller_db)
    finally:
        controller_db.close()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def register_perf_db() -> Iterator[ControllerTestState]:
    """Fresh ControllerTestState for bulk worker-attribute insert perf gate."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_perf_register_"))
    controller_db = ControllerDB(db_dir=tmp)
    try:
        yield ControllerTestState(controller_db)
    finally:
        controller_db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_submit_job_with_n_replicas_perf(submit_perf_db: ControllerTestState) -> None:
    """Gate bulk_insert_tasks submit path for 32 replicas below a ms ceiling."""
    state = submit_perf_db
    job_id_counter = [0]

    def _submit() -> None:
        job_id_counter[0] += 1
        name = f"/test-user/perf-job-{job_id_counter[0]:06d}"
        entrypoint = job_pb2.RuntimeEntrypoint()
        entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
        req = controller_pb2.Controller.LaunchJobRequest(
            name=name,
            entrypoint=entrypoint,
            resources=job_pb2.ResourceSpecProto(cpu_millicores=100, memory_bytes=256 * 1024**2),
            environment=job_pb2.EnvironmentConfig(),
            replicas=_REPLICA_COUNT,
        )
        jid = JobName.from_wire(name)
        with state._db.transaction() as cur:
            ops.job.submit(cur, job_id=jid, request=req, ts=Timestamp.now())

    per_call_s = _measure(_submit, _TICKS)
    max_ms = _SUBMIT_32_REPLICAS_MAX_MS
    assert per_call_s * 1e3 <= max_ms, (
        f"submit_job with {_REPLICA_COUNT} replicas too slow: "
        f"{per_call_s * 1e3:.1f} ms/call > {max_ms} ms gate ({_TICKS} iterations)."
    )


def test_register_worker_with_n_attributes_perf(register_perf_db: ControllerTestState) -> None:
    """Gate batch worker-attribute INSERT for 32 attributes below a ms ceiling."""
    state = register_perf_db
    worker_counter = [0]

    def _register() -> None:
        worker_counter[0] += 1
        wid = WorkerId(f"worker-{worker_counter[0]:06d}")
        attrs = {f"attr-key-{i}": job_pb2.AttributeValue(string_value=f"val-{i}") for i in range(_ATTR_COUNT)}
        metadata = job_pb2.WorkerMetadata(
            hostname="perf-worker",
            ip_address="10.0.0.1",
            cpu_count=64,
            memory_bytes=128 * 1024**3,
            disk_bytes=500 * 1024**3,
            device=job_pb2.DeviceConfig(cpu=job_pb2.CpuDevice(variant="cpu")),
            attributes=attrs,
        )
        with state._db.transaction() as cur:
            ops.worker.register(
                cur,
                worker_id=wid,
                address=f"{wid}:8080",
                metadata=metadata,
                ts=Timestamp.now(),
                health=state._health,
            )

    per_call_s = _measure(_register, _TICKS)
    max_ms = _REGISTER_32_ATTRS_MAX_MS
    assert per_call_s * 1e3 <= max_ms, (
        f"register_worker with {_ATTR_COUNT} attributes too slow: "
        f"{per_call_s * 1e3:.1f} ms/call > {max_ms} ms gate ({_TICKS} iterations)."
    )
