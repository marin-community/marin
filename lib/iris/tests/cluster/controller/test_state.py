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
from iris.cluster.controller.controller import compute_demand_entries
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
    ControllerState,
    ControllerTask,
)
from iris.cluster.types import JobId, TaskId, WorkerId, DeviceType
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
            attempt_id=task.current_attempt_id,
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
    task = state.get_task(task_id)
    assert task is not None
    state.handle_event(
        TaskStateChangedEvent(
            task_id=task_id,
            new_state=new_state,
            attempt_id=task.current_attempt_id,
            error=error,
            exit_code=exit_code,
        )
    )


def _make_test_entrypoint() -> cluster_pb2.Entrypoint:
    """Create a minimal Entrypoint proto for testing."""
    entrypoint = cluster_pb2.Entrypoint()
    entrypoint.command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
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
    req.replicas = 2
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

    # Verify: task requeued (back to PENDING), job still running
    assert task.state == cluster_pb2.TASK_STATE_PENDING
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
    req.replicas = 3
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

    # Verify: worker unhealthy, task requeued (back to PENDING)
    assert worker.healthy is False
    assert task.state == cluster_pb2.TASK_STATE_PENDING
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

    # Task gets assigned (creates attempt, puts in ASSIGNED state)
    state.handle_event(
        TaskAssignedEvent(
            task_id=task.task_id,
            worker_id=worker_id,
        )
    )
    assert task.state == cluster_pb2.TASK_STATE_ASSIGNED
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

    # 2. Task requeued (back to PENDING for retry)
    assert task.state == cluster_pb2.TASK_STATE_PENDING
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
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=0,
        replicas=3,
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
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=3,
        environment=cluster_pb2.EnvironmentConfig(),
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
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=2,
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=0,
        max_retries_preemption=1,
    )
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobId("j1"))

    dispatch_task(state, tasks[0], worker_id)
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker died")

    # Preemption doesn't count toward failure threshold; task requeued to PENDING
    assert tasks[0].state == cluster_pb2.TASK_STATE_PENDING
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
    """Endpoints are visible for all non-terminal job states (PENDING, RUNNING, BUILDING)
    and hidden once the job reaches a terminal state."""
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

    # Visible while pending (task may be executing before job state updates)
    assert len(state.lookup_endpoints("ns-1/actor")) == 1

    # Still visible after transition to running
    dispatch_task(state, task, worker_id)
    assert job.state == cluster_pb2.JOB_STATE_RUNNING
    assert len(state.lookup_endpoints("ns-1/actor")) == 1

    # Not visible after completion (terminal state)
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
    req.replicas = MAX_REPLICAS_PER_JOB + 1

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
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu=cpu, memory_bytes=memory_bytes),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )

    return _make


def test_worker_cannot_accept_task_when_resources_committed(make_job_request, worker_metadata):
    """E2E: A worker with committed resources cannot accept tasks that exceed remaining capacity."""
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
    multiple running tasks. With round-robin scheduling, each worker gets at most one
    task per cycle, so we run multiple cycles to fill capacity.
    """
    state = ControllerState()

    # Worker with 4 CPUs
    register_worker(state, "w1", "host:8080", worker_metadata(cpu=4))

    # Submit 3 jobs, each using 2 CPUs
    for i in range(3):
        submit_job(state, f"j{i}", make_job_request(cpu=2))

    scheduler = Scheduler(state)

    # First scheduling cycle: 1 task assigned (round-robin: 1 per worker per cycle)
    pending = state.peek_pending_tasks()
    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert len(result.assignments) == 1
    for task, worker in result.assignments:
        dispatch_task(state, task, worker.worker_id)

    # Second scheduling cycle: 1 more task assigned (worker still has 2 CPUs)
    pending = state.peek_pending_tasks()
    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert len(result.assignments) == 1
    for task, worker in result.assignments:
        dispatch_task(state, task, worker.worker_id)

    # Third task should still be pending
    pending = state.peek_pending_tasks()
    assert len(pending) == 1
    assert pending[0].job_id == JobId("j2")

    # Scheduler should not assign the third task (no capacity - 4 CPUs used)
    result = scheduler.find_assignments(pending, state.get_available_workers())
    assert len(result.assignments) == 0


# =============================================================================
# Coscheduled Failure Cascade Tests
# =============================================================================


def test_coscheduled_task_failure_kills_siblings(worker_metadata):
    """When one coscheduled task fails terminally, all running siblings are killed."""
    state = ControllerState()

    # Register 4 workers (one per task)
    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    # Create coscheduled job with 4 tasks
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    tasks = submit_job(state, "j1", req)

    job = state.get_job(JobId("j1"))
    assert job.is_coscheduled

    # Dispatch all tasks
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0 (terminal failure with no retries)
    txn = state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks[0].task_id,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            attempt_id=tasks[0].current_attempt_id,
            error="OOM",
        )
    )

    # Task-0 should be FAILED, all other tasks should be WORKER_FAILED
    assert tasks[0].state == cluster_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        assert task.state == cluster_pb2.TASK_STATE_WORKER_FAILED
        assert task.task_id in txn.tasks_to_kill


def test_coscheduled_task_worker_failure_kills_siblings(worker_metadata):
    """WORKER_FAILED also triggers sibling kill when retries exhausted."""
    state = ControllerState()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    # Use max_retries_preemption=1 (not 0 because 0 gets defaulted to 100)
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
        max_retries_preemption=1,  # Allow one retry, so second failure is terminal
    )
    req.coscheduling.group_by = "tpu-name"
    tasks = submit_job(state, "j1", req)

    # Dispatch all tasks
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # First WORKER_FAILED is retriable (retries remaining)
    state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks[0].task_id,
            new_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
            attempt_id=tasks[0].current_attempt_id,
            error="Worker crashed (first)",
        )
    )

    # Task-0 is retriable, siblings still running
    assert tasks[0].preemption_count == 1
    assert tasks[0].can_be_scheduled()
    for task in tasks[1:]:
        assert task.state == cluster_pb2.TASK_STATE_RUNNING

    # Re-dispatch task-0
    dispatch_task(state, tasks[0], WorkerId("w0"))

    # Second WORKER_FAILED exhausts retries - now terminal
    txn = state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks[0].task_id,
            new_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
            attempt_id=tasks[0].current_attempt_id,
            error="Worker crashed (second)",
        )
    )

    assert tasks[0].state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert tasks[0].is_finished()
    for task in tasks[1:]:
        assert task.state == cluster_pb2.TASK_STATE_WORKER_FAILED
        assert task.task_id in txn.tasks_to_kill


def test_coscheduled_task_success_does_not_affect_siblings(worker_metadata):
    """Task success does NOT kill siblings."""
    state = ControllerState()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    tasks = submit_job(state, "j1", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Task-0 succeeds
    txn = state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks[0].task_id,
            new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
            attempt_id=tasks[0].current_attempt_id,
        )
    )

    # Task-0 succeeded, siblings still running
    assert tasks[0].state == cluster_pb2.TASK_STATE_SUCCEEDED
    for task in tasks[1:]:
        assert task.state == cluster_pb2.TASK_STATE_RUNNING
    assert len(txn.tasks_to_kill) == 0


def test_non_coscheduled_task_failure_does_not_kill_siblings(worker_metadata):
    """Regular jobs don't cascade failures to siblings."""
    state = ControllerState()

    for i in range(4):
        register_worker(state, f"w{i}", f"addr{i}:8080", worker_metadata())

    # Regular job (no coscheduling)
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=3,  # Allow failures without killing the job
    )
    tasks = submit_job(state, "j1", req)

    job = state.get_job(JobId("j1"))
    assert not job.is_coscheduled

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0
    txn = state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks[0].task_id,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            attempt_id=tasks[0].current_attempt_id,
            error="OOM",
        )
    )

    # Task-0 failed, but siblings are still running (no cascade)
    assert tasks[0].state == cluster_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        assert task.state == cluster_pb2.TASK_STATE_RUNNING

    # No tasks marked to kill from coscheduling cascade
    assert len(txn.tasks_to_kill) == 0


def test_coscheduled_retriable_failure_does_not_kill_siblings(worker_metadata):
    """When a coscheduled task fails but has retries remaining, siblings are NOT killed."""
    state = ControllerState()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
        max_retries_failure=1,  # Allow one retry
        max_task_failures=4,  # Don't fail job on task failure
    )
    req.coscheduling.group_by = "tpu-name"
    tasks = submit_job(state, "j1", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0 (first failure, has retry remaining)
    txn = state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks[0].task_id,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            attempt_id=tasks[0].current_attempt_id,
            error="OOM",
        )
    )

    # Task-0 failed but is retriable, requeued to PENDING
    assert tasks[0].state == cluster_pb2.TASK_STATE_PENDING
    assert tasks[0].can_be_scheduled()  # Can retry
    assert not tasks[0].is_finished()  # Not terminal

    # Siblings should still be running (no cascade for retriable failures)
    for task in tasks[1:]:
        assert task.state == cluster_pb2.TASK_STATE_RUNNING

    # No tasks marked for kill
    assert len(txn.tasks_to_kill) == 0


# =============================================================================
# compute_demand_entries Tests
# =============================================================================


# =============================================================================
# Stale Attempt Tracking Tests
# =============================================================================


def test_stale_attempt_ignored(job_request, worker_metadata):
    """Stale attempt report does not change task state."""
    state = ControllerState()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 2
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # First attempt: dispatch, then fail via worker failure (retriable)
    dispatch_task(state, task, worker_id)
    old_attempt_id = task.current_attempt_id
    assert old_attempt_id == 0

    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker died")

    # Second attempt
    dispatch_task(state, task, worker_id)
    assert task.current_attempt_id == 1
    assert task.state == cluster_pb2.TASK_STATE_RUNNING

    # Stale report from old attempt should be ignored
    state.handle_event(
        TaskStateChangedEvent(
            task_id=task.task_id,
            new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
            attempt_id=old_attempt_id,
        )
    )

    # Task should still be RUNNING on the new attempt
    assert task.state == cluster_pb2.TASK_STATE_RUNNING
    assert task.current_attempt_id == 1


def test_stale_attempt_error_log_for_non_terminal(caplog, job_request, worker_metadata):
    """Stale attempt report logs ERROR when the old attempt is not terminal."""
    import logging

    state = ControllerState()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 2
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # First attempt
    dispatch_task(state, task, worker_id)

    # Manually create a second attempt without properly terminating the first.
    # This simulates a scenario where the controller created a new attempt
    # but the old one is still non-terminal (a precondition violation).
    task.create_attempt(worker_id)
    assert task.current_attempt_id == 1
    # The old attempt (0) is still in RUNNING state (non-terminal)
    assert not task.attempts[0].is_terminal()

    with caplog.at_level(logging.ERROR, logger="iris.cluster.controller.state"):
        state.handle_event(
            TaskStateChangedEvent(
                task_id=task.task_id,
                new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
                attempt_id=0,
            )
        )

    assert any("Stale attempt precondition violation" in r.message for r in caplog.records)


# =============================================================================
# compute_demand_entries Tests
# =============================================================================


def test_compute_demand_entries_counts_coscheduled_job_once():
    """Coscheduled job with 4 tasks should count as 1 slice demand, not 4."""
    state = ControllerState()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    assert len(demand) == 1
    assert demand[0].device_type == DeviceType.TPU
    assert demand[0].device_variant == "v5litepod-16"
    assert demand[0].count == 1  # Only 1 slice needed for coscheduled job


def test_compute_demand_entries_counts_non_coscheduled_tasks_individually():
    """Non-coscheduled job with 4 tasks should count as 4 slices demand."""
    state = ControllerState()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    # No coscheduling set
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    assert len(demand) == 1
    assert demand[0].device_type == DeviceType.TPU
    assert demand[0].device_variant == "v5litepod-16"
    assert demand[0].count == 4  # 4 separate slices for non-coscheduled job


def test_compute_demand_entries_mixed_coscheduled_and_regular():
    """Mix of coscheduled and regular jobs should count correctly."""
    state = ControllerState()

    # Coscheduled job with 4 tasks -> 1 slice
    coscheduled_req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    coscheduled_req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", coscheduled_req)

    # Regular job with 2 tasks -> 2 slices
    regular_req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "j2", regular_req)

    demand = compute_demand_entries(state)
    assert len(demand) == 1
    assert demand[0].device_type == DeviceType.TPU
    assert demand[0].device_variant == "v5litepod-16"
    assert demand[0].count == 3  # 1 (coscheduled) + 2 (regular) = 3 slices


def test_compute_demand_entries_separates_by_preemptible_constraint():
    """Jobs with different preemptible constraints produce separate demand entries."""
    state = ControllerState()

    # Job requiring preemptible workers
    preemptible_req = cluster_pb2.Controller.LaunchJobRequest(
        name="preemptible-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key="preemptible",
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="true"),
            )
        ],
    )
    submit_job(state, "j1", preemptible_req)

    # Job requiring non-preemptible workers
    on_demand_req = cluster_pb2.Controller.LaunchJobRequest(
        name="on-demand-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key="preemptible",
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="false"),
            )
        ],
    )
    submit_job(state, "j2", on_demand_req)

    demand = compute_demand_entries(state)
    assert len(demand) == 2

    by_preemptible = {d.preemptible: d for d in demand}
    assert by_preemptible[True].count == 1
    assert by_preemptible[True].device_type == DeviceType.TPU
    assert by_preemptible[False].count == 1
    assert by_preemptible[False].device_type == DeviceType.TPU


def test_compute_demand_entries_no_preemptible_constraint_gives_none():
    """Job without preemptible constraint produces demand with preemptible=None."""
    state = ControllerState()

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="unconstrained-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    assert len(demand) == 1
    assert demand[0].preemptible is None
