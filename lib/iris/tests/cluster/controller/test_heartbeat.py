# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker heartbeat timeout handling, health checks, and unified heartbeat path."""

import time

import pytest
from iris.cluster.controller.events import (
    JobSubmittedEvent,
    TaskAssignedEvent,
    TaskStateChangedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.state import (
    ControllerState,
    HeartbeatAction,
    HeartbeatSnapshot,
    RunningTaskEntry,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Duration, Timestamp


@pytest.fixture
def state():
    return ControllerState()


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
    """Helper: register a worker with the state."""
    state.handle_event(
        WorkerRegisteredEvent(
            worker_id=WorkerId(worker_id),
            address=address,
            metadata=worker_metadata,
            timestamp=Timestamp.now(),
        )
    )


def _make_snapshot(worker_id, address="host:8080", running_tasks=None):
    """Helper: create a HeartbeatSnapshot."""
    return HeartbeatSnapshot(
        worker_id=WorkerId(worker_id),
        worker_address=address,
        vm_address="192.168.1.1",
        running_tasks=running_tasks or [],
        tasks_to_run=[],
        tasks_to_kill=[],
    )


def test_worker_heartbeat_expired_check(state, worker_metadata):
    """Test that ControllerWorker.is_heartbeat_expired detects expired heartbeats."""
    _register_worker(state, "worker1", worker_metadata)

    workers = list(state.list_all_workers())
    worker = workers[0]

    # Short timeout should not expire immediately
    short_timeout = Duration.from_seconds(10.0)
    assert not worker.is_heartbeat_expired(short_timeout)

    # Very short timeout might expire after a brief sleep
    very_short_timeout = Duration.from_ms(1)
    time.sleep(0.01)  # 10ms
    assert worker.is_heartbeat_expired(very_short_timeout)


def test_report_heartbeat_success(state, worker_metadata):
    """Successful heartbeat with healthy worker returns OK."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    response = cluster_pb2.HeartbeatResponse(worker_healthy=True)
    action = state.report_heartbeat(snapshot, response)

    assert action == HeartbeatAction.OK
    worker = state.get_worker(WorkerId("worker1"))
    assert worker is not None
    assert worker.healthy


def test_report_heartbeat_rpc_failure_below_threshold(state, worker_metadata):
    """RPC failure below threshold returns TRANSIENT_FAILURE, worker stays alive."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    action = state.report_heartbeat(snapshot, None, error="connection refused")
    assert action == HeartbeatAction.TRANSIENT_FAILURE

    worker = state.get_worker(WorkerId("worker1"))
    assert worker is not None
    assert worker.consecutive_failures == 1


def test_report_heartbeat_rpc_failure_at_threshold(state, worker_metadata):
    """RPC failures at threshold return WORKER_FAILED and prune the worker."""
    state_with_threshold = ControllerState(heartbeat_failure_threshold=3)
    _register_worker(state_with_threshold, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    # First two failures are transient
    for _i in range(2):
        action = state_with_threshold.report_heartbeat(snapshot, None, error="timeout")
        assert action == HeartbeatAction.TRANSIENT_FAILURE

    # Third failure exceeds the threshold
    action = state_with_threshold.report_heartbeat(snapshot, None, error="timeout")
    assert action == HeartbeatAction.WORKER_FAILED

    # Worker should be pruned from state
    worker = state_with_threshold.get_worker(WorkerId("worker1"))
    assert worker is None


def test_report_heartbeat_unhealthy_worker(state, worker_metadata):
    """Worker reporting unhealthy immediately returns WORKER_FAILED."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    response = cluster_pb2.HeartbeatResponse(
        worker_healthy=False,
        health_error="disk free space 2.1% below threshold 5%",
    )
    action = state.report_heartbeat(snapshot, response)

    assert action == HeartbeatAction.WORKER_FAILED
    # Worker should be pruned from state
    worker = state.get_worker(WorkerId("worker1"))
    assert worker is None


def test_unhealthy_worker_cascades_to_tasks(state, worker_metadata):
    """An unhealthy worker's running tasks are marked WORKER_FAILED."""
    _register_worker(state, "worker1", worker_metadata)

    # Submit and dispatch a job/task to the worker
    job_id = JobName.from_wire("/user/test-job")
    state.handle_event(
        JobSubmittedEvent(
            job_id=job_id,
            request=cluster_pb2.Controller.LaunchJobRequest(
                name="/user/test-job",
                replicas=1,
            ),
            timestamp=Timestamp.now(),
        )
    )
    task_id = job_id.task(0)
    state.handle_event(TaskAssignedEvent(task_id=task_id, worker_id=WorkerId("worker1")))
    state.handle_event(
        TaskStateChangedEvent(
            task_id=task_id,
            new_state=cluster_pb2.TASK_STATE_RUNNING,
            attempt_id=0,
        )
    )

    # Now report unhealthy heartbeat
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
    action = state.report_heartbeat(snapshot, response)
    assert action == HeartbeatAction.WORKER_FAILED

    # Task should be marked WORKER_FAILED
    task = state.get_task(task_id)
    assert task is not None
    assert task.state == cluster_pb2.TASK_STATE_WORKER_FAILED
