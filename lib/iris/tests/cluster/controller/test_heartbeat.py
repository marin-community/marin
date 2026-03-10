# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker heartbeat timeout handling, health checks, and unified heartbeat path."""

import time

import pytest
from iris.cluster.controller.db import TASKS, WORKERS
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatAction,
    HeartbeatApplyRequest,
    HeartbeatSnapshot,
    RunningTaskEntry,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Duration, Timestamp


@pytest.fixture
def state():
    return ControllerTransitions()


@pytest.fixture
def worker_metadata():
    return cluster_pb2.WorkerMetadata(
        hostname="test-host",
        ip_address="192.168.1.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
    )


def _register_worker(state, worker_id, worker_metadata, address="host:8080"):
    state.register_or_refresh_worker(
        worker_id=WorkerId(worker_id),
        address=address,
        metadata=worker_metadata,
        ts=Timestamp.now(),
    )


def _make_snapshot(worker_id, address="host:8080", running_tasks=None):
    return HeartbeatSnapshot(
        worker_id=WorkerId(worker_id),
        worker_address=address,
        vm_address="192.168.1.1",
        running_tasks=running_tasks or [],
        tasks_to_run=[],
        tasks_to_kill=[],
    )


def test_worker_heartbeat_expired_check(state, worker_metadata):
    """Test heartbeat expiration checks against worker row last_heartbeat."""
    state.register_or_refresh_worker(
        worker_id=WorkerId("worker1"),
        address="host:8080",
        metadata=worker_metadata,
        ts=Timestamp.now(),
    )

    with state._db.snapshot() as q:
        workers = q.select(WORKERS)
    worker = workers[0]

    # Short timeout should not expire immediately
    short_timeout = Duration.from_seconds(10.0)
    assert worker.last_heartbeat.age_ms() < short_timeout.to_ms()

    # Very short timeout might expire after a brief sleep
    very_short_timeout = Duration.from_ms(1)
    time.sleep(0.01)  # 10ms
    assert worker.last_heartbeat.age_ms() > very_short_timeout.to_ms()


def test_complete_heartbeat_success(state, worker_metadata):
    """Successful heartbeat with healthy worker returns OK."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    response = cluster_pb2.HeartbeatResponse(worker_healthy=True)
    result = state.complete_heartbeat(snapshot, response)

    assert result.action == HeartbeatAction.OK

    with state._db.snapshot() as q:
        worker = q.one(WORKERS, where=WORKERS.c.worker_id == "worker1")
    assert worker is not None
    assert worker.healthy


def test_fail_heartbeat_below_threshold(state, worker_metadata):
    """RPC failure below threshold returns TRANSIENT_FAILURE, worker stays alive."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    action = state.fail_heartbeat(snapshot, "connection refused")
    assert action == HeartbeatAction.TRANSIENT_FAILURE

    with state._db.snapshot() as q:
        worker = q.one(WORKERS, where=WORKERS.c.worker_id == "worker1")
    assert worker is not None
    assert worker.consecutive_failures == 1


def test_fail_heartbeat_at_threshold(worker_metadata):
    """RPC failures at threshold return WORKER_FAILED and prune the worker."""
    state = ControllerTransitions(heartbeat_failure_threshold=3)
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    for _i in range(2):
        action = state.fail_heartbeat(snapshot, "timeout")
        assert action == HeartbeatAction.TRANSIENT_FAILURE

    action = state.fail_heartbeat(snapshot, "timeout")
    assert action == HeartbeatAction.WORKER_FAILED

    with state._db.snapshot() as q:
        assert not q.exists(WORKERS, where=WORKERS.c.worker_id == "worker1")


def test_complete_heartbeat_unhealthy_worker(state, worker_metadata):
    """Worker reporting unhealthy immediately returns WORKER_FAILED."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    response = cluster_pb2.HeartbeatResponse(
        worker_healthy=False,
        health_error="disk free space 2.1% below threshold 5%",
    )
    result = state.complete_heartbeat(snapshot, response)

    assert result.action == HeartbeatAction.WORKER_FAILED
    with state._db.snapshot() as q:
        assert not q.exists(WORKERS, where=WORKERS.c.worker_id == "worker1")


def test_unhealthy_worker_cascades_to_tasks(state, worker_metadata):
    """An unhealthy worker's running tasks are marked WORKER_FAILED."""
    _register_worker(state, "worker1", worker_metadata)

    job_id = JobName.from_wire("/user/test-job")
    state.submit_job(
        job_id,
        cluster_pb2.Controller.LaunchJobRequest(
            name="/user/test-job",
            replicas=1,
        ),
        Timestamp.now(),
    )
    task_id = job_id.task(0)
    state.queue_assignments([Assignment(task_id=task_id, worker_id=WorkerId("worker1"))])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=WorkerId("worker1"),
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=0,
                    new_state=cluster_pb2.TASK_STATE_RUNNING,
                )
            ],
        )
    )

    snapshot = _make_snapshot("worker1", running_tasks=[RunningTaskEntry(task_id, 0)])
    response = cluster_pb2.HeartbeatResponse(
        worker_healthy=False,
        health_error="tempfile write failed",
        tasks=[
            cluster_pb2.Controller.WorkerTaskStatus(
                task_id=task_id.to_wire(),
                attempt_id=0,
                state=cluster_pb2.TASK_STATE_RUNNING,
            )
        ],
    )
    result = state.complete_heartbeat(snapshot, response)
    assert result.action == HeartbeatAction.WORKER_FAILED

    with state._db.snapshot() as q:
        task = q.one(TASKS, where=TASKS.c.task_id == task_id.to_wire())
    assert task is not None
    assert task.state == cluster_pb2.TASK_STATE_WORKER_FAILED
