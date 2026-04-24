# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task stats history: insert on record_task_stats, logarithmic prune, and TTL eviction."""

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
    TASK_STATS_HISTORY_RETENTION,
    TASK_STATS_HISTORY_TERMINAL_TTL,
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


def _count_stats_rows(state: ControllerTransitions, task_id: JobName) -> int:
    with state._db.read_snapshot() as q:
        rows = q.raw(
            "SELECT COUNT(*) as cnt FROM task_stats_history WHERE task_id = ?",
            (task_id.to_wire(),),
        )
    return rows[0].cnt


def _record_stats(
    state: ControllerTransitions,
    task_id: JobName,
    items: int = 100,
    bytes_: int = 1024,
    status: str = "ok",
) -> None:
    state.record_task_stats(task_id, items_processed=items, bytes_processed=bytes_, status=status)


def test_record_task_stats_inserts_row(state):
    """Each call to record_task_stats inserts a row and updates status_message."""
    _, task_id = _setup_running_task(state)

    assert _count_stats_rows(state, task_id) == 0

    _record_stats(state, task_id, items=10, bytes_=500, status="processing")
    assert _count_stats_rows(state, task_id) == 1

    # Verify column values.
    with state._db.read_snapshot() as q:
        rows = q.raw(
            "SELECT items_processed, bytes_processed FROM task_stats_history "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id.to_wire(),),
        )
    assert rows[0].items_processed == 10
    assert rows[0].bytes_processed == 500

    _record_stats(state, task_id, items=20, bytes_=1000)
    assert _count_stats_rows(state, task_id) == 2


def test_record_task_stats_updates_status_message(state):
    """record_task_stats updates the tasks.status_message column."""
    _, task_id = _setup_running_task(state)

    _record_stats(state, task_id, status="step 1 done")

    with state._db.read_snapshot() as q:
        rows = q.raw("SELECT status_message FROM tasks WHERE task_id = ?", (task_id.to_wire(),))
    assert rows[0].status_message == "step 1 done"

    _record_stats(state, task_id, status="step 2 done")
    with state._db.read_snapshot() as q:
        rows = q.raw("SELECT status_message FROM tasks WHERE task_id = ?", (task_id.to_wire(),))
    assert rows[0].status_message == "step 2 done"


def test_prune_logarithmic_downsampling(state):
    """Prune triggers at 2*N rows and thins the older half."""
    _, task_id = _setup_running_task(state)

    threshold = TASK_STATS_HISTORY_RETENTION * 2
    for i in range(threshold + 1):
        _record_stats(state, task_id, items=i, bytes_=i)

    assert _count_stats_rows(state, task_id) == threshold + 1

    deleted = state.prune_task_stats_history()
    assert deleted > 0

    remaining = _count_stats_rows(state, task_id)
    assert remaining < threshold
    assert remaining > TASK_STATS_HISTORY_RETENTION


def test_prune_preserves_newest_rows(state):
    """The newest N rows are never touched by pruning."""
    _, task_id = _setup_running_task(state)

    threshold = TASK_STATS_HISTORY_RETENTION * 2
    for i in range(threshold + 10):
        _record_stats(state, task_id, items=i, bytes_=i)

    with state._db.read_snapshot() as q:
        newest_before = [
            r.id
            for r in q.raw(
                "SELECT id FROM task_stats_history WHERE task_id = ? ORDER BY id DESC LIMIT ?",
                (task_id.to_wire(), TASK_STATS_HISTORY_RETENTION),
            )
        ]

    state.prune_task_stats_history()

    with state._db.read_snapshot() as q:
        surviving = {
            r.id
            for r in q.raw(
                "SELECT id FROM task_stats_history WHERE task_id = ?",
                (task_id.to_wire(),),
            )
        }
    assert set(newest_before).issubset(surviving)


def test_prune_noop_below_threshold(state):
    """No rows are deleted when count is at or below 2*N."""
    _, task_id = _setup_running_task(state)

    for i in range(TASK_STATS_HISTORY_RETENTION):
        _record_stats(state, task_id, items=i, bytes_=i)

    deleted = state.prune_task_stats_history()
    assert deleted == 0
    assert _count_stats_rows(state, task_id) == TASK_STATS_HISTORY_RETENTION


def _force_terminal(state: ControllerTransitions, task_id: JobName, finished_age_ms: int) -> None:
    """Mark a task as SUCCEEDED with finished_at_ms set to `finished_age_ms`
    in the past. Bypasses the state machine — we only need the row shape."""
    now_ms = Timestamp.now().epoch_ms()
    state._db.execute(
        "UPDATE tasks SET state = ?, finished_at_ms = ? WHERE task_id = ?",
        (job_pb2.TASK_STATE_SUCCEEDED, now_ms - finished_age_ms, task_id.to_wire()),
    )


def test_prune_evicts_terminal_task_history_past_ttl(state):
    """Tasks terminal for longer than the TTL have all stats history removed."""
    _, task_id = _setup_running_task(state)
    _record_stats(state, task_id, items=1, bytes_=1)
    _record_stats(state, task_id, items=2, bytes_=2)
    assert _count_stats_rows(state, task_id) == 2

    _force_terminal(state, task_id, finished_age_ms=TASK_STATS_HISTORY_TERMINAL_TTL.to_ms() * 2)

    deleted = state.prune_task_stats_history()
    assert deleted == 2
    assert _count_stats_rows(state, task_id) == 0


def test_prune_keeps_terminal_task_history_within_ttl(state):
    """Recently-terminal tasks (within TTL) keep their stats history."""
    _, task_id = _setup_running_task(state)
    _record_stats(state, task_id, items=1, bytes_=1)
    _record_stats(state, task_id, items=2, bytes_=2)

    _force_terminal(state, task_id, finished_age_ms=TASK_STATS_HISTORY_TERMINAL_TTL.to_ms() // 2)

    deleted = state.prune_task_stats_history()
    assert deleted == 0
    assert _count_stats_rows(state, task_id) == 2


def test_prune_keeps_running_task_history_regardless_of_finished_at(state):
    """TTL eviction gates on terminal state, not just finished_at_ms — a RUNNING
    task with a stale finished_at_ms must not be evicted."""
    _, task_id = _setup_running_task(state)
    _record_stats(state, task_id, items=1, bytes_=1)

    now_ms = Timestamp.now().epoch_ms()
    stale_ms = now_ms - TASK_STATS_HISTORY_TERMINAL_TTL.to_ms() * 2
    state._db.execute(
        "UPDATE tasks SET finished_at_ms = ? WHERE task_id = ?",
        (stale_ms, task_id.to_wire()),
    )

    deleted = state.prune_task_stats_history()
    assert deleted == 0
    assert _count_stats_rows(state, task_id) == 1


def test_cascade_delete_on_job_removal(state):
    """Task stats history rows are deleted when the parent job is removed."""
    job_id, task_id = _setup_running_task(state)

    _record_stats(state, task_id, items=100, bytes_=1024)
    assert _count_stats_rows(state, task_id) == 1

    state.cancel_job(job_id, reason="test cleanup")
    state.remove_finished_job(job_id)

    assert _count_stats_rows(state, task_id) == 0
