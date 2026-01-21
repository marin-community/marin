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

from iris.cluster.controller.events import (
    JobCancelledEvent,
    JobSubmittedEvent,
    TaskAssignedEvent,
    TaskStateChangedEvent,
    WorkerFailedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.scheduler import Scheduler
from iris.cluster.controller.state import (
    MAX_REPLICAS_PER_JOB,
    ControllerEndpoint,
    ControllerJob,
    ControllerState,
    ControllerTask,
)
from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms

# =============================================================================
# Test Helpers
# =============================================================================


def dispatch_task(state: ControllerState, task: ControllerTask, worker_id: WorkerId) -> None:
    """Dispatch a task to a worker: assign + mark running."""
    state.handle_event(
        TaskAssignedEvent(
            task_id=task.task_id,
            worker_id=worker_id,
        )
    )
    state.handle_event(
        TaskStateChangedEvent(
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
        TaskStateChangedEvent(
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


def register_worker(
    state: ControllerState,
    worker_id: str,
    address: str,
    metadata: cluster_pb2.WorkerMetadata,
) -> WorkerId:
    """Register a worker via event."""
    wid = WorkerId(worker_id)
    state.handle_event(
        WorkerRegisteredEvent(
            worker_id=wid,
            address=address,
            metadata=metadata,
            timestamp_ms=now_ms(),
        )
    )
    return wid


def submit_job(
    state: ControllerState,
    job_id: str,
    request: cluster_pb2.Controller.LaunchJobRequest,
) -> list[ControllerTask]:
    """Submit a job via event and return tasks."""
    jid = JobId(job_id)
    state.handle_event(
        JobSubmittedEvent(
            job_id=jid,
            request=request,
            timestamp_ms=now_ms(),
        )
    )
    return state.get_job_tasks(jid)


# =============================================================================
# Job/Task Lifecycle Integration Tests
# =============================================================================


def test_job_lifecycle_success(job_request, worker_metadata):
    """E2E: Submit job -> dispatch task -> succeed -> verify final state."""
    state = ControllerState()

    # Setup: register worker
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())
    worker = state.get_worker(worker_id)

    # Submit job via event
    req = job_request("test-job")
    req.resources.replicas = 2
    tasks = submit_job(state, "j1", req)

    job = state.get_job(JobId("j1"))

    assert job is not None
    assert len(tasks) == 2
    assert job.state == cluster_pb2.JOB_STATE_PENDING

    # Dispatch and succeed all tasks
    for task in tasks:
        dispatch_task(state, task, worker_id)
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

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())
    worker = state.get_worker(worker_id)

    req = job_request("job1")
    tasks = submit_job(state, "j1", req)
    task = tasks[0]
    job = state.get_job(JobId("j1"))

    # Dispatch and fail (default max_retries_failure=0)
    dispatch_task(state, task, worker_id)
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_FAILED, error="Task failed")

    # Verify final state
    assert task.state == cluster_pb2.TASK_STATE_FAILED
    assert task.is_finished()
    assert job.state == cluster_pb2.JOB_STATE_FAILED
    assert task.task_id not in worker.running_tasks


def test_task_failure_with_retry_requeues(job_request, worker_metadata):
    """E2E: Task failure with retries -> task requeued, job stays running."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_task_failures = 1
    req.max_retries_failure = 1
    tasks = submit_job(state, "j1", req)
    task = tasks[0]
    job = state.get_job(JobId("j1"))

    # First attempt fails
    dispatch_task(state, task, worker_id)
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

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("test-job")
    req.resources.replicas = 3
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobId("j1"))

    # Dispatch 2 tasks, leave 1 pending
    dispatch_task(state, tasks[0], worker_id)
    dispatch_task(state, tasks[1], worker_id)

    # Cancel job
    state.handle_event(
        JobCancelledEvent(
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

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())
    worker = state.get_worker(worker_id)

    req = job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)

    # Worker fails
    state.handle_event(
        WorkerFailedEvent(
            worker_id=worker_id,
            error="Connection lost",
        )
    )

    # Verify: worker unhealthy, task WORKER_FAILED and requeued
    assert worker.healthy is False
    assert task.state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert task.can_be_scheduled()
    pending = state.peek_pending_tasks()
    assert len(pending) == 1


def test_dispatch_failure_marks_worker_failed_and_requeues_task(job_request, worker_metadata):
    """E2E: Dispatch RPC failure (task in PENDING) -> worker failed event cascades to task."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())
    worker = state.get_worker(worker_id)

    req = job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Task gets assigned (creates attempt, puts in PENDING state)
    state.handle_event(
        TaskAssignedEvent(
            task_id=task.task_id,
            worker_id=worker_id,
        )
    )
    assert task.state == cluster_pb2.TASK_STATE_PENDING
    assert task.current_attempt_id == 0

    # Dispatch RPC fails -> WORKER_FAILED event
    state.handle_event(
        WorkerFailedEvent(
            worker_id=worker_id,
            error="Dispatch RPC failed: Connection refused",
        )
    )

    # Verify cascade:
    # 1. Worker marked unhealthy
    assert worker.healthy is False

    # 2. Task marked as WORKER_FAILED (retriable)
    assert task.state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert task.preemption_count == 1
    assert task.can_be_scheduled()

    # 3. Task should be requeued for retry
    pending = state.peek_pending_tasks()
    assert len(pending) == 1
    assert pending[0].task_id == task.task_id

    # 4. Worker no longer has task assigned
    assert task.task_id not in worker.running_tasks


# =============================================================================
# Failure Domain Tests (max_task_failures)
# =============================================================================


def test_failure_domain_kills_remaining_tasks(worker_metadata):
    """E2E: One task fails beyond retries -> remaining tasks killed, job fails."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="multi-task-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        max_task_failures=0,
    )
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobId("j1"))

    # Dispatch 2 tasks, leave 1 pending
    dispatch_task(state, tasks[0], worker_id)
    dispatch_task(state, tasks[1], worker_id)

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

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tolerant-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        max_task_failures=1,
    )
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobId("j1"))

    for task in tasks:
        dispatch_task(state, task, worker_id)

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

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="preemption-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=2),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        max_task_failures=0,
        max_retries_preemption=1,
    )
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobId("j1"))

    dispatch_task(state, tasks[0], worker_id)
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

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)

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


def test_endpoint_visibility_by_job_state(job_request, worker_metadata):
    """E2E: Endpoints only visible for RUNNING jobs."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("test")
    tasks = submit_job(state, "ns-1", req)
    job = state.get_job(JobId("ns-1"))
    task = tasks[0]

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        job_id=JobId("ns-1"),
    )
    state.add_endpoint(ep)

    # Not visible while pending
    assert len(state.lookup_endpoints("ns-1/actor")) == 0

    # Transition to running by dispatching task
    dispatch_task(state, task, worker_id)
    assert job.state == cluster_pb2.JOB_STATE_RUNNING
    assert len(state.lookup_endpoints("ns-1/actor")) == 1

    # Not visible after completion
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_SUCCEEDED)
    assert job.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert len(state.lookup_endpoints("ns-1/actor")) == 0


def test_namespace_isolation(job_request, worker_metadata):
    """E2E: Endpoints are isolated by namespace prefix."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req1 = job_request("test1")
    req2 = job_request("test2")

    tasks1 = submit_job(state, "ns-1", req1)
    tasks2 = submit_job(state, "ns-2", req2)

    # Dispatch tasks to transition jobs to RUNNING state
    dispatch_task(state, tasks1[0], worker_id)
    dispatch_task(state, tasks2[0], worker_id)

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

    req1 = job_request("job1")
    req2 = job_request("job2")
    submit_job(state, "j1", req1)
    submit_job(state, "j2", req2)

    pending = state.peek_pending_tasks()
    assert len(pending) == 2
    assert pending[0].job_id == JobId("j1")
    assert pending[1].job_id == JobId("j2")


def test_gang_job_tracking(job_request):
    """Gang jobs are tracked correctly.

    Note: gang_id is not in the proto, so this test uses add_job() directly.
    """
    state = ControllerState()
    job1 = ControllerJob(job_id=JobId("j1"), request=job_request("job1"), gang_id="gang1")
    job2 = ControllerJob(job_id=JobId("j2"), request=job_request("job2"), gang_id="gang1")
    job3 = ControllerJob(job_id=JobId("j3"), request=job_request("job3"), gang_id="gang2")

    state.add_job(job1)
    state.add_job(job2)
    state.add_job(job3)

    gang1_jobs = state.get_gang_jobs("gang1")
    assert len(gang1_jobs) == 2
    assert {j.job_id for j in gang1_jobs} == {"j1", "j2"}

    gang2_jobs = state.get_gang_jobs("gang2")
    assert len(gang2_jobs) == 1

    assert state.get_gang_jobs("nonexistent") == []


def test_hierarchical_job_tracking(job_request):
    """Parent-child job relationships are tracked correctly."""
    state = ControllerState()

    parent_req = job_request("parent")
    submit_job(state, "parent", parent_req)

    child1_req = job_request("child1")
    child1_req.parent_job_id = "parent"
    submit_job(state, "child1", child1_req)

    child2_req = job_request("child2")
    child2_req.parent_job_id = "parent"
    submit_job(state, "child2", child2_req)

    grandchild_req = job_request("grandchild")
    grandchild_req.parent_job_id = "child1"
    submit_job(state, "grandchild", grandchild_req)

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
                req = job_request(f"job-{job_id}")
                submit_job(state, job_id, req)
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


# =============================================================================
# Validation Tests
# =============================================================================


def test_excessive_replicas_fails_job(job_request):
    """E2E: Job with replicas exceeding MAX_REPLICAS_PER_JOB fails immediately."""
    state = ControllerState()

    req = job_request("too-many-replicas")
    req.resources.replicas = MAX_REPLICAS_PER_JOB + 1

    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobId("j1"))

    assert job is not None
    assert job.state == cluster_pb2.JOB_STATE_FAILED
    assert f"exceeds max {MAX_REPLICAS_PER_JOB}" in job.error
    assert len(tasks) == 0
    assert len(state.peek_pending_tasks()) == 0


# =============================================================================
# Worker Resource Commitment Tests
# =============================================================================


@pytest.fixture
def make_job_request():
    """Create a LaunchJobRequest with configurable resources."""

    def _make(
        name: str = "test-job",
        cpu: int = 1,
        memory_bytes: int = 1024**3,
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpecProto(cpu=cpu, memory_bytes=memory_bytes),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


def test_worker_cannot_accept_task_when_resources_committed(make_job_request, worker_metadata):
    """E2E: A worker with committed resources cannot accept tasks that exceed remaining capacity.

    This exercises the full flow: task assignment commits resources, and the scheduler
    respects committed resources when evaluating capacity for subsequent tasks.
    """
    state = ControllerState()

    # Worker with 4 CPUs
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata(cpu=4))

    # First job uses 3 CPUs
    tasks1 = submit_job(state, "j1", make_job_request(cpu=3))
    dispatch_task(state, tasks1[0], worker_id)

    # Second job needs 2 CPUs - should not fit (only 1 CPU remaining)
    submit_job(state, "j2", make_job_request(cpu=2))

    # Scheduler should not assign the second task to this worker
    pending = state.peek_pending_tasks()
    assert len(pending) == 1  # j2's task is still pending

    workers = state.get_available_workers()
    scheduler = Scheduler(state)
    result = scheduler.find_assignments(pending, workers)

    # The task cannot be scheduled - no worker has sufficient capacity
    assert len(result.assignments) == 0
    assert pending[0].job_id == JobId("j2")


def test_worker_can_accept_new_task_after_previous_completes(make_job_request, worker_metadata):
    """E2E: After a task completes, its resources are freed and new tasks can be scheduled.

    This verifies that task completion releases committed resources back to the worker.
    """
    state = ControllerState()

    # Worker with 4 CPUs
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata(cpu=4))

    # First job uses 3 CPUs
    tasks1 = submit_job(state, "j1", make_job_request(cpu=3))
    dispatch_task(state, tasks1[0], worker_id)

    # Second job needs 3 CPUs - cannot fit while first is running
    submit_job(state, "j2", make_job_request(cpu=3))

    scheduler = Scheduler(state)

    # Verify second task cannot be scheduled yet
    pending = state.peek_pending_tasks()
    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert len(result.assignments) == 0

    # Complete the first task
    transition_task(state, tasks1[0].task_id, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Now the second task can be scheduled
    pending = state.peek_pending_tasks()
    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert len(result.assignments) == 1
    assert result.assignments[0][0].job_id == JobId("j2")


def test_multiple_small_tasks_fill_worker_capacity(make_job_request, worker_metadata):
    """E2E: Multiple small tasks can fill a worker's capacity, blocking further assignments.

    This verifies that the scheduler correctly tracks cumulative resource usage across
    multiple running tasks.
    """
    state = ControllerState()

    # Worker with 4 CPUs
    register_worker(state, "w1", "host:8080", worker_metadata(cpu=4))

    # Submit 3 jobs, each using 2 CPUs
    for i in range(3):
        submit_job(state, f"j{i}", make_job_request(cpu=2))

    scheduler = Scheduler(state)

    # First scheduling cycle: 2 tasks should fit (4 CPUs / 2 CPUs each = 2 tasks)
    pending = state.peek_pending_tasks()
    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert len(result.assignments) == 2

    # Apply the assignments to state
    for task, worker in result.assignments:
        dispatch_task(state, task, worker.worker_id)

    # Third task should still be pending
    pending = state.peek_pending_tasks()
    assert len(pending) == 1
    assert pending[0].job_id == JobId("j2")

    # Scheduler should not assign the third task (no capacity)
    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert len(result.assignments) == 0


# =============================================================================
# Worker Attributes Tests
# =============================================================================


def test_worker_registers_with_attributes(worker_metadata):
    """Worker attributes are extracted from metadata and stored on registration."""
    state = ControllerState()

    metadata = worker_metadata()
    metadata.attributes["tpu-name"].string_value = "my-tpu"
    metadata.attributes["tpu-worker-id"].int_value = 0

    worker_id = register_worker(state, "w1", "host:8080", metadata)

    worker = state.get_worker(worker_id)
    assert worker is not None
    assert "tpu-name" in worker.attributes
    assert worker.attributes["tpu-name"].value == "my-tpu"
    assert "tpu-worker-id" in worker.attributes
    assert worker.attributes["tpu-worker-id"].value == 0


def test_worker_attributes_with_multiple_types(worker_metadata):
    """Worker attributes support string, int, and float values."""
    state = ControllerState()

    metadata = worker_metadata()
    metadata.attributes["string-attr"].string_value = "hello"
    metadata.attributes["int-attr"].int_value = 42
    metadata.attributes["float-attr"].float_value = 3.14

    worker_id = register_worker(state, "w1", "host:8080", metadata)

    worker = state.get_worker(worker_id)
    assert worker.attributes["string-attr"].value == "hello"
    assert worker.attributes["int-attr"].value == 42
    assert worker.attributes["float-attr"].value == 3.14


def test_worker_attributes_updated_on_reregistration(worker_metadata):
    """Worker attributes are updated when worker re-registers."""
    state = ControllerState()

    # Initial registration with one attribute
    metadata1 = worker_metadata()
    metadata1.attributes["tpu-name"].string_value = "old-tpu"
    worker_id = register_worker(state, "w1", "host:8080", metadata1)

    worker = state.get_worker(worker_id)
    assert worker.attributes["tpu-name"].value == "old-tpu"

    # Re-registration with updated attribute
    metadata2 = worker_metadata()
    metadata2.attributes["tpu-name"].string_value = "new-tpu"
    metadata2.attributes["tpu-worker-id"].int_value = 5
    register_worker(state, "w1", "host:8080", metadata2)

    worker = state.get_worker(worker_id)
    assert worker.attributes["tpu-name"].value == "new-tpu"
    assert worker.attributes["tpu-worker-id"].value == 5


def test_worker_without_attributes():
    """Worker without attributes has empty attributes dict."""
    state = ControllerState()

    device = cluster_pb2.DeviceConfig()
    device.cpu.CopyFrom(cluster_pb2.CpuDevice(variant="cpu"))

    metadata = cluster_pb2.WorkerMetadata(
        hostname="test-worker",
        ip_address="127.0.0.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
        device=device,
    )

    worker_id = register_worker(state, "w1", "host:8080", metadata)
    worker = state.get_worker(worker_id)
    assert worker.attributes == {}
