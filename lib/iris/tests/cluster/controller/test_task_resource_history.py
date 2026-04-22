# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task resource history: insert on heartbeat, logarithmic prune, and service layer."""

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
    TASK_RESOURCE_HISTORY_RETENTION,
    TASK_RESOURCE_HISTORY_TERMINAL_TTL,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2, controller_pb2
from rigging.timing import Timestamp


@pytest.fixture
def state(tmp_path):
    db = ControllerDB(db_dir=tmp_path)
    s = ControllerTransitions(db=db)
    yield s
    db.close()


WORKER_META = job_pb2.WorkerMetadata(
    hostname="test-host",
    ip_address="10.0.0.1",
    cpu_count=8,
    memory_bytes=16 * 1024**3,
    disk_bytes=100 * 1024**3,
)


def _setup_running_task(state: ControllerTransitions) -> tuple[JobName, JobName]:
    """Create a registered worker, submitted job, assigned and running task.
    Returns (job_id, task_id).
    """
    wid = WorkerId("w1")
    state.register_or_refresh_worker(worker_id=wid, address="host:8080", metadata=WORKER_META, ts=Timestamp.now())

    job_id = JobName.from_wire("/user/test-job")
    state.submit_job(
        job_id,
        controller_pb2.Controller.LaunchJobRequest(name="/user/test-job", replicas=1),
        Timestamp.now(),
    )
    task_id = job_id.task(0)
    state.queue_assignments([Assignment(task_id=task_id, worker_id=wid)])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=wid,
            worker_resource_snapshot=None,
            updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
        )
    )
    return job_id, task_id


def _count_history_rows(state: ControllerTransitions, task_id: JobName, attempt_id: int = 0) -> int:
    with state._db.read_snapshot() as q:
        rows = q.raw(
            "SELECT COUNT(*) as cnt FROM task_resource_history WHERE task_id = ? AND attempt_id = ?",
            (task_id.to_wire(), attempt_id),
        )
    return rows[0].cnt


def _send_resource_heartbeat(state: ControllerTransitions, task_id: JobName, cpu: int = 1000, mem: int = 512):
    """Send a steady-state heartbeat with resource usage."""
    usage = job_pb2.ResourceUsage(cpu_millicores=cpu, memory_mb=mem, disk_mb=10)
    state.apply_heartbeats_batch(
        [
            HeartbeatApplyRequest(
                worker_id=WorkerId("w1"),
                worker_resource_snapshot=job_pb2.WorkerResourceSnapshot(),
                updates=[
                    TaskUpdate(
                        task_id=task_id,
                        attempt_id=0,
                        new_state=job_pb2.TASK_STATE_RUNNING,
                        resource_usage=usage,
                    ),
                ],
            )
        ]
    )


def test_heartbeat_inserts_history(state):
    """Each heartbeat with resource_usage inserts a row with native columns."""
    _, task_id = _setup_running_task(state)

    assert _count_history_rows(state, task_id) == 0

    _send_resource_heartbeat(state, task_id, cpu=1000, mem=512)
    assert _count_history_rows(state, task_id) == 1

    # Verify native column values.
    with state._db.read_snapshot() as q:
        rows = q.raw(
            "SELECT cpu_millicores, memory_mb, disk_mb FROM task_resource_history "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id.to_wire(),),
        )
    assert rows[0].cpu_millicores == 1000
    assert rows[0].memory_mb == 512
    assert rows[0].disk_mb == 10

    _send_resource_heartbeat(state, task_id, cpu=2000, mem=1024)
    assert _count_history_rows(state, task_id) == 2


def test_prune_logarithmic_downsampling(state):
    """Prune triggers at 2*N rows and thins the older half."""
    _, task_id = _setup_running_task(state)

    # Insert exactly 2*N + 1 rows to trigger compaction.
    threshold = TASK_RESOURCE_HISTORY_RETENTION * 2
    for i in range(threshold + 1):
        _send_resource_heartbeat(state, task_id, cpu=1000 + i, mem=512 + i)

    assert _count_history_rows(state, task_id) == threshold + 1

    deleted = state.prune_task_resource_history()
    assert deleted > 0

    remaining = _count_history_rows(state, task_id)
    # Older half (~N+1 rows) is thinned by ~50%, plus newest N kept intact.
    # So remaining should be roughly 1.5*N.
    assert remaining < threshold
    assert remaining > TASK_RESOURCE_HISTORY_RETENTION


def test_prune_preserves_newest_rows(state):
    """The newest N rows are never touched by pruning."""
    _, task_id = _setup_running_task(state)

    threshold = TASK_RESOURCE_HISTORY_RETENTION * 2
    for i in range(threshold + 10):
        _send_resource_heartbeat(state, task_id, cpu=i, mem=i)

    # Read the newest N row IDs before pruning.
    with state._db.read_snapshot() as q:
        newest_before = [
            r.id
            for r in q.raw(
                "SELECT id FROM task_resource_history WHERE task_id = ? AND attempt_id = 0 " "ORDER BY id DESC LIMIT ?",
                (task_id.to_wire(), TASK_RESOURCE_HISTORY_RETENTION),
            )
        ]

    state.prune_task_resource_history()

    # All of the newest N rows should still exist.
    with state._db.read_snapshot() as q:
        surviving = {
            r.id
            for r in q.raw(
                "SELECT id FROM task_resource_history WHERE task_id = ? AND attempt_id = 0",
                (task_id.to_wire(),),
            )
        }
    assert set(newest_before).issubset(surviving)


def test_prune_noop_below_threshold(state):
    """No rows are deleted when count is at or below 2*N."""
    _, task_id = _setup_running_task(state)

    for i in range(TASK_RESOURCE_HISTORY_RETENTION):
        _send_resource_heartbeat(state, task_id, cpu=i, mem=i)

    deleted = state.prune_task_resource_history()
    assert deleted == 0
    assert _count_history_rows(state, task_id) == TASK_RESOURCE_HISTORY_RETENTION


def _force_terminal(state: ControllerTransitions, task_id: JobName, finished_age_ms: int) -> None:
    """Mark a task as SUCCEEDED with finished_at_ms set to `finished_age_ms`
    in the past. Bypasses the state machine — we only need the row shape."""
    now_ms = Timestamp.now().epoch_ms()
    state._db.execute(
        "UPDATE tasks SET state = ?, finished_at_ms = ? WHERE task_id = ?",
        (job_pb2.TASK_STATE_SUCCEEDED, now_ms - finished_age_ms, task_id.to_wire()),
    )


def test_prune_evicts_terminal_task_history_past_ttl(state):
    """Tasks terminal for longer than the TTL have all history removed."""
    _, task_id = _setup_running_task(state)
    _send_resource_heartbeat(state, task_id, cpu=1, mem=1)
    _send_resource_heartbeat(state, task_id, cpu=2, mem=2)
    assert _count_history_rows(state, task_id) == 2

    _force_terminal(state, task_id, finished_age_ms=TASK_RESOURCE_HISTORY_TERMINAL_TTL.to_ms() * 2)

    deleted = state.prune_task_resource_history()
    assert deleted == 2
    assert _count_history_rows(state, task_id) == 0


def test_prune_keeps_terminal_task_history_within_ttl(state):
    """Recently-terminal tasks (within TTL) keep their history."""
    _, task_id = _setup_running_task(state)
    _send_resource_heartbeat(state, task_id, cpu=1, mem=1)
    _send_resource_heartbeat(state, task_id, cpu=2, mem=2)

    # Terminal at half the TTL — must survive.
    _force_terminal(state, task_id, finished_age_ms=TASK_RESOURCE_HISTORY_TERMINAL_TTL.to_ms() // 2)

    deleted = state.prune_task_resource_history()
    assert deleted == 0
    assert _count_history_rows(state, task_id) == 2


def test_prune_keeps_running_task_history_regardless_of_finished_at(state):
    """TTL eviction gates on terminal state, not just finished_at_ms — a RUNNING
    task with a stale finished_at_ms (shouldn't happen, but guard against it)
    must not be evicted."""
    _, task_id = _setup_running_task(state)
    _send_resource_heartbeat(state, task_id, cpu=1, mem=1)

    # Stale finished_at_ms but state still RUNNING.
    now_ms = Timestamp.now().epoch_ms()
    stale_ms = now_ms - TASK_RESOURCE_HISTORY_TERMINAL_TTL.to_ms() * 2
    state._db.execute(
        "UPDATE tasks SET finished_at_ms = ? WHERE task_id = ?",
        (stale_ms, task_id.to_wire()),
    )

    deleted = state.prune_task_resource_history()
    assert deleted == 0
    assert _count_history_rows(state, task_id) == 1


def test_cascade_delete_on_job_removal(state):
    """Task resource history rows are deleted when the parent job is removed."""
    job_id, task_id = _setup_running_task(state)

    _send_resource_heartbeat(state, task_id, cpu=1000, mem=512)
    assert _count_history_rows(state, task_id) == 1

    # Cancel the job (makes it terminal), then remove it.
    state.cancel_job(job_id, reason="test cleanup")
    state.remove_finished_job(job_id)

    assert _count_history_rows(state, task_id) == 0
