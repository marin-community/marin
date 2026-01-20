# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for controller state management.

These tests exercise end-to-end observable behavior through the event-driven API (handle_event).
They focus on:
- Full workflows (submit job -> dispatch tasks -> complete/fail)
- Important edge cases (retry exhaustion, worker failure cascades, failure domains)
- Final state verification rather than intermediate steps
"""

import threading

import pytest

from iris.cluster.controller.events import Event, EventType
from iris.cluster.controller.state import (
    ControllerEndpoint,
    ControllerJob,
    ControllerState,
    ControllerTask,
    ControllerWorker,
)
from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2

# =============================================================================
# Test Helpers
# =============================================================================


def dispatch_task(state: ControllerState, task: ControllerTask, worker_id: WorkerId) -> None:
    """Dispatch a task to a worker: assign + mark running."""
    state.handle_event(
        Event(
            EventType.TASK_ASSIGNED,
            task_id=task.task_id,
            worker_id=worker_id,
        )
    )
    state.handle_event(
        Event(
            EventType.TASK_STATE_CHANGED,
            task_id=task.task_id,
            new_state=cluster_pb2.TASK_STATE_RUNNING,
        )
    )


def transition_task(
    state: ControllerState,
    task_id: TaskId,
    new_state: int,
    *,
    error: str | None = None,
    exit_code: int | None = None,
) -> None:
    """Transition a task to a new state via handle_event."""
    state.handle_event(
        Event(
            EventType.TASK_STATE_CHANGED,
            task_id=task_id,
            new_state=new_state,
            error=error,
            exit_code=exit_code,
        )
    )


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


@pytest.fixture
def worker_metadata():
    """Create WorkerMetadata for testing."""

    def _make(
        cpu: int = 10,
        memory_bytes: int = 10 * 1024**3,
        disk_bytes: int = 10 * 1024**3,
    ) -> cluster_pb2.WorkerMetadata:
        device = cluster_pb2.DeviceConfig()
        device.cpu.CopyFrom(cluster_pb2.CpuDevice(variant="cpu"))

        return cluster_pb2.WorkerMetadata(
            hostname="test-worker",
            ip_address="127.0.0.1",
            cpu_count=cpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            device=device,
        )

    return _make


def _add_job(state, job):
    """Add a job with auto-expanded tasks."""
    return state.add_job(job)


# =============================================================================
# Job/Task Lifecycle Integration Tests
# =============================================================================


def test_job_lifecycle_success(job_request, worker_metadata):
    """E2E: Submit job -> dispatch task -> succeed -> verify final state."""
    state = ControllerState()

    # Setup: register worker
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    # Submit job via event
    req = job_request("test-job")
    req.resources.replicas = 2
    state.handle_event(
        Event(
            EventType.JOB_SUBMITTED,
            job_id=JobId("j1"),
            request=req,
            timestamp_ms=1000,
        )
    )

    job = state.get_job(JobId("j1"))
    tasks = state.get_job_tasks(JobId("j1"))

    assert job is not None
    assert len(tasks) == 2
    assert job.state == cluster_pb2.JOB_STATE_PENDING

    # Dispatch and succeed all tasks
    for task in tasks:
        dispatch_task(state, task, WorkerId("w1"))
        transition_task(state, task.task_id, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Verify final state
    assert job.state == cluster_pb2.JOB_STATE_SUCCEEDED
    for task in tasks:
        assert task.state == cluster_pb2.TASK_STATE_SUCCEEDED
        assert task.task_id not in worker.running_tasks
    assert len(state.peek_pending_tasks()) == 0


def test_job_lifecycle_failure_exhausted_retries(job_request, worker_metadata):
    """E2E: Task failure with no retries -> job fails."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    job = ControllerJob(job_id=JobId("j1"), request=job_request("job1"))
    tasks = _add_job(state, job)
    task = tasks[0]

    # Dispatch and fail (default max_retries_failure=0)
    dispatch_task(state, task, WorkerId("w1"))
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_FAILED, error="Task failed")

    # Verify final state
    assert task.state == cluster_pb2.TASK_STATE_FAILED
    assert task.is_finished()
    assert job.state == cluster_pb2.JOB_STATE_FAILED
    assert task.task_id not in worker.running_tasks


def test_task_failure_with_retry_requeues(job_request, worker_metadata):
    """E2E: Task failure with retries -> task requeued, job stays running."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    req = job_request("job1")
    req.max_task_failures = 1
    job = ControllerJob(job_id=JobId("j1"), request=req)
    tasks = _add_job(state, job)
    task = tasks[0]
    task.max_retries_failure = 1

    # First attempt fails
    dispatch_task(state, task, WorkerId("w1"))
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_FAILED)

    # Verify: task requeued, job still running
    assert task.state == cluster_pb2.TASK_STATE_FAILED
    assert task.can_be_scheduled()
    assert job.state == cluster_pb2.JOB_STATE_RUNNING
    pending = state.peek_pending_tasks()
    assert len(pending) == 1
    assert pending[0].task_id == task.task_id


def test_job_cancellation_kills_all_tasks(job_request, worker_metadata):
    """E2E: Job cancellation -> all tasks killed."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    req = job_request("test-job")
    req.resources.replicas = 3
    job = ControllerJob(job_id=JobId("j1"), request=req)
    tasks = state.add_job(job)

    # Dispatch 2 tasks, leave 1 pending
    dispatch_task(state, tasks[0], WorkerId("w1"))
    dispatch_task(state, tasks[1], WorkerId("w1"))

    # Cancel job
    state.handle_event(
        Event(
            EventType.JOB_CANCELLED,
            job_id=JobId("j1"),
            reason="User cancelled",
        )
    )

    # Verify all tasks killed
    assert job.state == cluster_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert task.state == cluster_pb2.TASK_STATE_KILLED


# =============================================================================
# Worker Failure Cascade Tests
# =============================================================================


def test_worker_failure_cascades_to_running_tasks(job_request, worker_metadata):
    """E2E: Worker failure -> running tasks transition to WORKER_FAILED and requeue."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    job = ControllerJob(job_id=JobId("j1"), request=job_request("job1"))
    tasks = state.add_job(job)
    task = tasks[0]
    task.max_retries_preemption = 1

    dispatch_task(state, task, WorkerId("w1"))

    # Worker fails
    state.handle_event(
        Event(
            EventType.WORKER_FAILED,
            worker_id=WorkerId("w1"),
            error="Connection lost",
        )
    )

    # Verify: worker unhealthy, task WORKER_FAILED and requeued
    assert worker.healthy is False
    assert task.state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert task.can_be_scheduled()
    pending = state.peek_pending_tasks()
    assert len(pending) == 1


# =============================================================================
# Failure Domain Tests (max_task_failures)
# =============================================================================


def test_failure_domain_kills_remaining_tasks(worker_metadata):
    """E2E: One task fails beyond retries -> remaining tasks killed, job fails."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="multi-task-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        max_task_failures=0,
    )
    job = ControllerJob(job_id=JobId("j1"), request=req)
    tasks = state.add_job(job)

    # Dispatch 2 tasks, leave 1 pending
    dispatch_task(state, tasks[0], WorkerId("w1"))
    dispatch_task(state, tasks[1], WorkerId("w1"))

    # Task-0 fails
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="Task failed")

    # Verify final state
    assert job.state == cluster_pb2.JOB_STATE_FAILED
    assert tasks[0].state == cluster_pb2.TASK_STATE_FAILED
    assert tasks[1].state == cluster_pb2.TASK_STATE_KILLED
    assert tasks[2].state == cluster_pb2.TASK_STATE_KILLED


def test_max_task_failures_tolerance(worker_metadata):
    """E2E: Job tolerates max_task_failures, then fails on next failure."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tolerant-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        max_task_failures=1,
    )
    job = ControllerJob(job_id=JobId("j1"), request=req)
    tasks = state.add_job(job)

    for task in tasks:
        dispatch_task(state, task, WorkerId("w1"))

    # First failure - job should keep running
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="First")
    assert job.state == cluster_pb2.JOB_STATE_RUNNING

    # Second task succeeds
    transition_task(state, tasks[1].task_id, cluster_pb2.TASK_STATE_SUCCEEDED)
    assert job.state == cluster_pb2.JOB_STATE_RUNNING

    # Third task fails - exceeds threshold, job fails
    transition_task(state, tasks[2].task_id, cluster_pb2.TASK_STATE_FAILED, error="Second")
    assert job.state == cluster_pb2.JOB_STATE_FAILED


def test_preemption_does_not_count_toward_max_task_failures(worker_metadata):
    """E2E: Worker failures (preemptions) don't count toward max_task_failures."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="preemption-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=2),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        max_task_failures=0,
    )
    job = ControllerJob(job_id=JobId("j1"), request=req)
    tasks = state.add_job(job)
    tasks[0].max_retries_preemption = 1

    dispatch_task(state, tasks[0], WorkerId("w1"))
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker died")

    # Preemption doesn't count toward failure threshold
    assert tasks[0].state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert tasks[0].can_be_scheduled()
    assert job.state == cluster_pb2.JOB_STATE_RUNNING


# =============================================================================
# Endpoint Cleanup Tests
# =============================================================================


def test_terminal_states_clean_up_endpoints(job_request, worker_metadata):
    """E2E: Task reaching terminal state removes associated endpoints."""
    state = ControllerState()

    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host:8080",
        metadata=worker_metadata(),
    )
    state.add_worker(worker)

    job = ControllerJob(job_id=JobId("j1"), request=job_request("job1"))
    tasks = _add_job(state, job)
    task = tasks[0]

    dispatch_task(state, task, WorkerId("w1"))

    ep = ControllerEndpoint(
        endpoint_id="ep1",
        name="j1/actor",
        address="a:1",
        job_id=JobId("j1"),
    )
    state.add_endpoint(ep, task.task_id)

    # Verify endpoint visible while running
    assert len(state.lookup_endpoints("j1/actor")) == 1

    # Task succeeds
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Endpoint removed
    assert state.lookup_endpoints("j1/actor") == []


def test_endpoint_visibility_by_job_state():
    """E2E: Endpoints only visible for RUNNING jobs."""
    state = ControllerState()

    job = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_PENDING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Not visible while pending
    assert len(state.lookup_endpoints("ns-1/actor")) == 0

    # Transition to running
    job.state = cluster_pb2.JOB_STATE_RUNNING
    assert len(state.lookup_endpoints("ns-1/actor")) == 1

    # Not visible after completion
    job.state = cluster_pb2.JOB_STATE_SUCCEEDED
    assert len(state.lookup_endpoints("ns-1/actor")) == 0


def test_namespace_isolation():
    """E2E: Endpoints are isolated by namespace prefix."""
    state = ControllerState()

    job1 = ControllerJob(
        job_id=JobId("ns-1"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    job2 = ControllerJob(
        job_id=JobId("ns-2"),
        request=cluster_pb2.Controller.LaunchJobRequest(name="test2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job1)
    state.add_job(job2)

    state.add_endpoint(
        ControllerEndpoint(
            endpoint_id="ep-1",
            name="ns-1/actor",
            address="10.0.0.1:8080",
            job_id=JobId("ns-1"),
        )
    )
    state.add_endpoint(
        ControllerEndpoint(
            endpoint_id="ep-2",
            name="ns-2/actor",
            address="10.0.0.2:8080",
            job_id=JobId("ns-2"),
        )
    )

    # Each namespace only sees its own endpoint
    results_ns1 = state.lookup_endpoints("ns-1/actor")
    assert len(results_ns1) == 1
    assert results_ns1[0].address == "10.0.0.1:8080"

    results_ns2 = state.lookup_endpoints("ns-2/actor")
    assert len(results_ns2) == 1
    assert results_ns2[0].address == "10.0.0.2:8080"


# =============================================================================
# Queue and Worker State Tests
# =============================================================================


def test_task_queue_fifo_order(job_request):
    """Tasks are returned in FIFO order."""
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=job_request("job1"), submitted_at_ms=100)
    job2 = ControllerJob(job_id=JobId("j2"), request=job_request("job2"), submitted_at_ms=200)
    _add_job(state, job1)
    _add_job(state, job2)

    pending = state.peek_pending_tasks()
    assert len(pending) == 2
    assert pending[0].job_id == JobId("j1")
    assert pending[1].job_id == JobId("j2")


def test_gang_job_tracking(job_request):
    """Gang jobs are tracked correctly."""
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=job_request("job1"), gang_id="gang1")
    job2 = ControllerJob(job_id=JobId("j2"), request=job_request("job2"), gang_id="gang1")
    job3 = ControllerJob(job_id=JobId("j3"), request=job_request("job3"), gang_id="gang2")

    _add_job(state, job1)
    _add_job(state, job2)
    _add_job(state, job3)

    gang1_jobs = state.get_gang_jobs("gang1")
    assert len(gang1_jobs) == 2
    assert {j.job_id for j in gang1_jobs} == {"j1", "j2"}

    gang2_jobs = state.get_gang_jobs("gang2")
    assert len(gang2_jobs) == 1

    assert state.get_gang_jobs("nonexistent") == []


def test_hierarchical_job_tracking(job_request):
    """Parent-child job relationships are tracked correctly."""
    state = ControllerState()

    parent = ControllerJob(job_id=JobId("parent"), request=job_request("parent"))
    child1 = ControllerJob(
        job_id=JobId("child1"),
        request=job_request("child1"),
        parent_job_id=JobId("parent"),
    )
    child2 = ControllerJob(
        job_id=JobId("child2"),
        request=job_request("child2"),
        parent_job_id=JobId("parent"),
    )
    grandchild = ControllerJob(
        job_id=JobId("grandchild"),
        request=job_request("grandchild"),
        parent_job_id=JobId("child1"),
    )

    _add_job(state, parent)
    _add_job(state, child1)
    _add_job(state, child2)
    _add_job(state, grandchild)

    # get_children only returns direct children
    children = state.get_children(JobId("parent"))
    assert len(children) == 2
    assert {c.job_id for c in children} == {"child1", "child2"}

    # No children for leaf nodes
    assert state.get_children(JobId("grandchild")) == []


def test_thread_safety(job_request):
    """Concurrent access doesn't corrupt state."""
    state = ControllerState()
    num_threads = 10
    jobs_per_thread = 50
    barrier = threading.Barrier(num_threads)
    errors = []

    def add_jobs(thread_id: int):
        try:
            barrier.wait()
            for i in range(jobs_per_thread):
                job_id = f"t{thread_id}_j{i}"
                job = ControllerJob(job_id=JobId(job_id), request=job_request(f"job-{job_id}"))
                _add_job(state, job)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_jobs, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    expected_count = num_threads * jobs_per_thread
    pending = state.peek_pending_tasks()
    assert len(pending) == expected_count
