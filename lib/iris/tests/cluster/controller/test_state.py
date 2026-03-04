# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller state management.

These tests exercise end-to-end observable behavior through the event-driven API (handle_event).
They focus on:
- Full workflows (submit job -> dispatch tasks -> complete/fail)
- Important edge cases (retry exhaustion, worker failure cascades, failure domains)
- Final state verification rather than intermediate steps
"""

import threading

import pytest
from iris.cluster.controller.autoscaler import DemandEntry
from iris.cluster.controller.controller import compute_demand_entries
from iris.cluster.controller.events import (
    JobCancelledEvent,
    JobSubmittedEvent,
    TaskAssignedEvent,
    TaskStateChangedEvent,
    WorkerFailedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.scheduler import JobRequirements, Scheduler
from iris.cluster.controller.state import (
    MAX_REPLICAS_PER_JOB,
    ControllerEndpoint,
    ControllerState,
    ControllerTask,
)
from iris.cluster.types import PREEMPTIBLE_ATTRIBUTE_KEY, REGION_ATTRIBUTE_KEY, DeviceType, JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.rpc.time_conversions import duration_to_proto
from rigging.time_utils import Duration, Timestamp

# =============================================================================
# Test Helpers
# =============================================================================


def _build_scheduling_context(scheduler: Scheduler, state: ControllerState):
    """Build a SchedulingContext from current state, mirroring what the controller does."""
    pending = state.peek_pending_tasks()
    workers = state.get_available_workers()
    task_ids = [t.task_id for t in pending]
    jobs: dict[JobName, JobRequirements] = {}
    for t in pending:
        job_id = t.task_id.parent
        if job_id and job_id not in jobs:
            job = state.get_job(job_id)
            if job:
                jobs[job_id] = JobRequirements(
                    resources=job.request.resources,
                    constraints=list(job.request.constraints),
                    is_coscheduled=job.is_coscheduled,
                    coscheduling_group_by=job.coscheduling_group_by,
                )
    return scheduler.create_scheduling_context(
        workers,
        building_counts=state.snapshot_building_counts(),
        pending_tasks=task_ids,
        jobs=jobs,
    )


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
    task_id: JobName,
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


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    """Create a minimal RuntimeEntrypoint proto for testing."""
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        job_name = JobName.root("test-user", name)
        return cluster_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
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
            timestamp=Timestamp.now(),
        )
    )
    return wid


def submit_job(
    state: ControllerState,
    job_id: str,
    request: cluster_pb2.Controller.LaunchJobRequest,
    timestamp_ms: int | None = None,
) -> list[ControllerTask]:
    """Submit a job via event and return tasks."""
    jid = JobName.from_string(job_id) if job_id.startswith("/") else JobName.root("test-user", job_id)
    request.name = jid.to_wire()
    state.handle_event(
        JobSubmittedEvent(
            job_id=jid,
            request=request,
            timestamp=Timestamp.from_ms(timestamp_ms) if timestamp_ms is not None else Timestamp.now(),
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

    job = state.get_job(JobName.root("test-user", "j1"))

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
    job = state.get_job(JobName.root("test-user", "j1"))

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
    job = state.get_job(JobName.root("test-user", "j1"))

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


def test_unschedulable_task_finalizes_job_with_timeout_error(job_request, worker_metadata):
    """E2E: Task UNSCHEDULABLE propagates timeout-style error to final job state."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.scheduling_timeout.CopyFrom(duration_to_proto(Duration.from_seconds(300)))
    tasks = submit_job(state, "j1", req)
    task = tasks[0]
    job = state.get_job(JobName.root("test-user", "j1"))

    dispatch_task(state, task, worker_id)
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_UNSCHEDULABLE)

    assert task.state == cluster_pb2.TASK_STATE_UNSCHEDULABLE
    assert task.error == "Scheduling timeout exceeded"
    assert job.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE
    assert job.error == "Scheduling timeout exceeded"


def test_job_cancellation_kills_all_tasks(job_request, worker_metadata):
    """E2E: Job cancellation -> all tasks killed."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("test-job")
    req.replicas = 3
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobName.root("test-user", "j1"))

    # Dispatch 2 tasks, leave 1 pending
    dispatch_task(state, tasks[0], worker_id)
    dispatch_task(state, tasks[1], worker_id)

    # Cancel job
    state.handle_event(
        JobCancelledEvent(
            job_id=JobName.root("test-user", "j1"),
            reason="User cancelled",
        )
    )

    # Verify all tasks killed
    assert job.state == cluster_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert task.state == cluster_pb2.TASK_STATE_KILLED


def test_cancelled_job_tasks_excluded_from_demand(job_request, worker_metadata):
    """Regression test for issue #2777: Killed tasks with no attempts should not appear in demand entries."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("test-job")
    req.replicas = 3
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobName.root("test-user", "j1"))

    # Dispatch 1 task, leave 2 pending (these will have no attempts when killed)
    dispatch_task(state, tasks[0], worker_id)

    # Cancel job - pending tasks will be killed with no attempts
    state.handle_event(
        JobCancelledEvent(
            job_id=JobName.root("test-user", "j1"),
            reason="User cancelled",
        )
    )

    # Verify all tasks are killed
    assert job.state == cluster_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert task.state == cluster_pb2.TASK_STATE_KILLED

    # Verify killed tasks are not schedulable (bug fix)
    for task in tasks:
        assert not task.can_be_scheduled(), f"Killed task {task.task_id} should not be schedulable"

    # Verify no pending tasks appear in queue
    pending = state.peek_pending_tasks()
    assert len(pending) == 0, "Killed tasks should not appear in pending queue"

    # Verify no demand entries generated for killed job
    demand_entries = compute_demand_entries(state)
    assert len(demand_entries) == 0, "Killed job should not generate demand entries"


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


def test_failed_worker_is_pruned_from_state(job_request, worker_metadata):
    """E2E: Worker failure removes worker from state, preventing dead worker accumulation."""
    state = ControllerState()

    w1 = register_worker(state, "w1", "host1:8080", worker_metadata())
    w2 = register_worker(state, "w2", "host2:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "j1", req)
    dispatch_task(state, tasks[0], w1)

    # Worker w1 fails
    state.handle_event(WorkerFailedEvent(worker_id=w1, error="Connection lost"))

    # w1 is gone from state entirely
    assert state.get_worker(w1) is None
    # w2 is still present
    assert state.get_worker(w2) is not None

    # list_all_workers only returns w2
    all_workers = state.list_all_workers()
    assert len(all_workers) == 1
    assert all_workers[0].worker_id == w2

    # Task was requeued despite worker removal
    assert tasks[0].state == cluster_pb2.TASK_STATE_PENDING
    assert tasks[0].can_be_scheduled()

    # A re-registering worker creates a fresh entry
    w1_again = register_worker(state, "w1", "host1:8080", worker_metadata())
    assert state.get_worker(w1_again) is not None
    assert state.get_worker(w1_again).healthy is True
    assert len(state.list_all_workers()) == 2


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


def test_task_assigned_to_missing_worker_is_ignored(job_request, worker_metadata):
    """Stale assignments to pruned workers are skipped without crashing."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())
    tasks = submit_job(state, "j1", job_request("job1"))
    task = tasks[0]

    # Worker disappears between scheduling and assignment commit.
    state.remove_worker(worker_id)
    state.handle_event(
        TaskAssignedEvent(
            task_id=task.task_id,
            worker_id=worker_id,
        )
    )

    # Task remains schedulable and no attempt/resources are committed.
    assert task.state == cluster_pb2.TASK_STATE_PENDING
    assert task.current_attempt_id == -1
    assert task.can_be_scheduled()
    assert task.task_id in {t.task_id for t in state.peek_pending_tasks()}


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
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=0,
        replicas=3,
    )
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobName.root("test-user", "j1"))

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
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=3,
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=1,
    )
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobName.root("test-user", "j1"))

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
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=2,
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=0,
        max_retries_preemption=1,
    )
    tasks = submit_job(state, "j1", req)
    job = state.get_job(JobName.root("test-user", "j1"))

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
        job_id=JobName.root("test-user", "j1"),
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
    job = state.get_job(JobName.root("test-user", "ns-1"))
    task = tasks[0]

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        job_id=JobName.root("test-user", "ns-1"),
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
            job_id=JobName.root("test-user", "ns-1"),
        )
    )
    state.add_endpoint(
        ControllerEndpoint(
            endpoint_id="ep-2",
            name="ns-2/actor",
            address="10.0.0.2:8080",
            job_id=JobName.root("test-user", "ns-2"),
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
    assert pending[0].job_id == JobName.root("test-user", "j1")
    assert pending[1].job_id == JobName.root("test-user", "j2")


def test_hierarchical_job_tracking(job_request):
    """Parent-child job relationships are tracked correctly."""
    state = ControllerState()

    parent_req = job_request("parent")
    submit_job(state, "parent", parent_req)

    child1_req = job_request("child1")
    submit_job(state, "/test-user/parent/child1", child1_req)

    child2_req = job_request("child2")
    submit_job(state, "/test-user/parent/child2", child2_req)

    grandchild_req = job_request("grandchild")
    submit_job(state, "/test-user/parent/child1/grandchild", grandchild_req)

    # get_children only returns direct children
    children = state.get_children(JobName.root("test-user", "parent"))
    assert len(children) == 2
    assert {c.job_id for c in children} == {
        JobName.from_string("/test-user/parent/child1"),
        JobName.from_string("/test-user/parent/child2"),
    }

    # No children for leaf nodes
    assert state.get_children(JobName.from_string("/test-user/parent/child1/grandchild")) == []


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
    job = state.get_job(JobName.root("test-user", "j1"))

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
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=cpu * 1000, memory_bytes=memory_bytes),
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

    scheduler = Scheduler()
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # The task cannot be scheduled - no worker has sufficient capacity
    assert len(result.assignments) == 0
    assert pending[0].job_id == JobName.root("test-user", "j2")


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

    scheduler = Scheduler()

    # Verify second task cannot be scheduled yet
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 0

    # Complete the first task
    transition_task(state, tasks1[0].task_id, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Now the second task can be scheduled
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    assert result.assignments[0][0].parent == JobName.root("test-user", "j2")


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

    scheduler = Scheduler()

    # First scheduling cycle: 1 task assigned (round-robin: 1 per worker per cycle)
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    for task_id, worker_id in result.assignments:
        task = state.get_task(task_id)
        dispatch_task(state, task, worker_id)

    # Second scheduling cycle: 1 more task assigned (worker still has 2 CPUs)
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    for task_id, worker_id in result.assignments:
        task = state.get_task(task_id)
        dispatch_task(state, task, worker_id)

    # Third task should still be pending
    pending = state.peek_pending_tasks()
    assert len(pending) == 1
    assert pending[0].job_id == JobName.root("test-user", "j2")

    # Scheduler should not assign the third task (no capacity - 4 CPUs used)
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
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
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    tasks = submit_job(state, "j1", req)

    job = state.get_job(JobName.root("test-user", "j1"))
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


def test_coscheduled_cascade_releases_worker_resources(worker_metadata):
    """Coscheduled sibling cascade must free committed resources on surviving workers.

    Regression test: previously, _cascade_coscheduled_failure marked siblings
    terminal but never called _cleanup_task_resources, leaking committed_cpu_millicores/mem
    on workers and permanently blocking future scheduling.
    """
    state = ControllerState()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="leak-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = "tpu-name"
    tasks = submit_job(state, "j-leak", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Verify resources are committed before failure
    for i in range(4):
        w = state.get_worker(WorkerId(f"w{i}"))
        assert w.committed_cpu_millicores == 2000
        assert len(w.running_tasks) == 1

    # Fail task-0 terminally → cascade kills siblings on w1, w2, w3
    state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks[0].task_id,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            attempt_id=tasks[0].current_attempt_id,
            error="OOM",
        )
    )

    # All surviving workers (w1..w3) must have resources fully released
    for i in range(1, 4):
        w = state.get_worker(WorkerId(f"w{i}"))
        assert w.committed_cpu_millicores == 0, f"w{i} has leaked committed_cpu_millicores={w.committed_cpu_millicores}"
        assert w.committed_mem == 0, f"w{i} has leaked committed_mem={w.committed_mem}"
        assert len(w.running_tasks) == 0, f"w{i} has phantom running_tasks={w.running_tasks}"

    # w0 should also be clean (task-0 was the trigger, cleaned up by _on_task_state_changed)
    w0 = state.get_worker(WorkerId("w0"))
    assert w0.committed_cpu_millicores == 0
    assert len(w0.running_tasks) == 0


def test_coscheduled_task_worker_failure_kills_siblings(worker_metadata):
    """WORKER_FAILED also triggers sibling kill when retries exhausted."""
    state = ControllerState()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    # Use max_retries_preemption=1 so second worker failure is terminal.
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
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
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
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
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=3,  # Allow failures without killing the job
    )
    tasks = submit_job(state, "j1", req)

    job = state.get_job(JobName.root("test-user", "j1"))
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
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
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
# log_directory Tests
# =============================================================================


def test_log_directory_persisted_on_first_running_heartbeat(job_request, worker_metadata):
    """log_directory is stored from running_tasks even when task state has not changed."""
    state = ControllerState()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    tasks = submit_job(state, "j1", job_request("job1"))
    task = tasks[0]
    dispatch_task(state, task, worker_id)

    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None

    # Worker reports task as RUNNING (same state the controller already has) with a log_directory.
    # This simulates the common steady-state heartbeat where the state hasn't changed but
    # log_directory has not yet been recorded by the controller.
    response = cluster_pb2.HeartbeatResponse(
        tasks=[
            cluster_pb2.Controller.WorkerTaskStatus(
                task_id=task.task_id.to_wire(),
                attempt_id=task.current_attempt_id,
                state=cluster_pb2.TASK_STATE_RUNNING,
                log_directory="s3://bucket/logs/task/0",
            )
        ]
    )
    state.complete_heartbeat(snapshot, response)

    assert task.attempts[task.current_attempt_id].log_directory == "s3://bucket/logs/task/0"


def test_log_directory_persisted_on_completed_task(job_request, worker_metadata):
    """log_directory is stored from completed_tasks report."""
    state = ControllerState()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    tasks = submit_job(state, "j1", job_request("job1"))
    task = tasks[0]
    dispatch_task(state, task, worker_id)

    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None

    response = cluster_pb2.HeartbeatResponse(
        tasks=[
            cluster_pb2.Controller.WorkerTaskStatus(
                task_id=task.task_id.to_wire(),
                attempt_id=task.current_attempt_id,
                state=cluster_pb2.TASK_STATE_SUCCEEDED,
                log_directory="s3://bucket/logs/task/0",
            )
        ]
    )
    state.complete_heartbeat(snapshot, response)

    assert task.attempts[task.current_attempt_id].log_directory == "s3://bucket/logs/task/0"


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
            cpu_millicores=1000,
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
    assert demand[0].task_ids == ["/test-user/j1/0", "/test-user/j1/1", "/test-user/j1/2", "/test-user/j1/3"]
    assert demand[0].coschedule_group_id == "/test-user/j1"


def test_compute_demand_entries_counts_non_coscheduled_tasks_individually():
    """Non-coscheduled job with 4 tasks should count as 4 slices demand."""
    state = ControllerState()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    # No coscheduling set
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    assert len(demand) == 4
    for entry in demand:
        assert entry.device_type == DeviceType.TPU
        assert entry.device_variant == "v5litepod-16"
        assert entry.coschedule_group_id is None
        assert len(entry.task_ids) == 1


def test_compute_demand_entries_mixed_coscheduled_and_regular():
    """Mix of coscheduled and regular jobs should count correctly."""
    state = ControllerState()

    # Coscheduled job with 4 tasks -> 1 slice
    coscheduled_req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
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
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "j2", regular_req)

    demand = compute_demand_entries(state)
    assert len(demand) == 3
    coscheduled = [entry for entry in demand if entry.coschedule_group_id == "/test-user/j1"]
    regular = [entry for entry in demand if entry.coschedule_group_id is None]
    assert len(coscheduled) == 1
    assert len(regular) == 2
    assert coscheduled[0].task_ids == ["/test-user/j1/0", "/test-user/j1/1", "/test-user/j1/2", "/test-user/j1/3"]
    for entry in regular:
        assert entry.device_type == DeviceType.TPU
        assert entry.device_variant == "v5litepod-16"


def test_compute_demand_entries_separates_by_preemptible_constraint():
    """Jobs with different preemptible constraints produce separate demand entries."""
    state = ControllerState()

    # Job requiring preemptible workers
    preemptible_req = cluster_pb2.Controller.LaunchJobRequest(
        name="preemptible-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key=PREEMPTIBLE_ATTRIBUTE_KEY,
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
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key=PREEMPTIBLE_ATTRIBUTE_KEY,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="false"),
            )
        ],
    )
    submit_job(state, "j2", on_demand_req)

    demand = compute_demand_entries(state)
    assert len(demand) == 2

    by_preemptible = {d.preemptible: d for d in demand}
    assert by_preemptible[True].device_type == DeviceType.TPU
    assert by_preemptible[True].task_ids == ["/test-user/j1/0"]
    assert by_preemptible[False].device_type == DeviceType.TPU
    assert by_preemptible[False].task_ids == ["/test-user/j2/0"]


def test_compute_demand_entries_no_preemptible_constraint_gives_none():
    """Job without preemptible constraint produces demand with preemptible=None."""
    state = ControllerState()

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="unconstrained-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
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


def test_compute_demand_entries_extracts_required_region():
    state = ControllerState()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regional-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key=REGION_ATTRIBUTE_KEY,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="us-west4"),
            )
        ],
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    assert len(demand) == 1
    assert demand[0].required_regions == frozenset({"us-west4"})
    assert demand[0].invalid_reason is None


def test_compute_demand_entries_marks_invalid_on_conflicting_region_constraints():
    state = ControllerState()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="invalid-regional-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key=REGION_ATTRIBUTE_KEY,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="us-west4"),
            ),
            cluster_pb2.Constraint(
                key=REGION_ATTRIBUTE_KEY,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="eu-west4"),
            ),
        ],
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    assert len(demand) == 1
    assert demand[0].invalid_reason is not None


# =============================================================================
# Reservation Demand Deduplication Tests
# =============================================================================


def _make_reservation_job_request(
    *,
    task_device: cluster_pb2.DeviceConfig,
    reservation_devices: list[cluster_pb2.DeviceConfig],
    replicas: int = 1,
) -> cluster_pb2.Controller.LaunchJobRequest:
    """Build a LaunchJobRequest with a reservation and task resources."""
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="reservation-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=task_device,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=replicas,
    )
    for dev in reservation_devices:
        req.reservation.entries.append(
            cluster_pb2.ReservationEntry(
                resources=cluster_pb2.ResourceSpecProto(
                    cpu_millicores=1000,
                    memory_bytes=1024**3,
                    device=dev,
                ),
            )
        )
    return req


def _h100_device() -> cluster_pb2.DeviceConfig:
    return cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="H100", count=8))


def _a100_device() -> cluster_pb2.DeviceConfig:
    return cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="A100", count=8))


def _is_synthetic_demand(state: ControllerState, demand_entry: DemandEntry) -> bool:
    """Check if a demand entry comes from a holder job task."""
    for tid in demand_entry.task_ids:
        task = state.get_task(JobName.from_string(tid))
        if task:
            job = state.get_job(task.job_id)
            if job and job.is_reservation_holder:
                return True
    return False


def test_demand_reservation_all_tasks_generate_demand():
    """2 H100 reservation + 2 H100 tasks = 4 total demand (no budget dedup).

    All tasks generate demand through a unified path. Holder tasks and real
    tasks are independent demand sources — preemption during scheduling
    (not demand) handles the dedup.
    """
    state = ControllerState()
    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 2


def test_demand_reservation_excess_tasks():
    """2 H100 reservation + 5 H100 tasks = 2 synthetic + 5 real task demand."""
    state = ControllerState()
    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=5,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 5


def test_demand_reservation_holder_uses_entry_resources():
    """Holder tasks use the reservation entry's resource spec, not the parent's.

    Each reservation entry carries its own resources and constraints. The
    holder job uses the entry's resources so the autoscaler provisions the
    correct device type even when the parent job differs.
    """
    state = ControllerState()
    # Job tasks request A100, but reservation entries specify H100.
    # Holder job should use the entry's H100 resource spec.
    req = _make_reservation_job_request(
        task_device=_a100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 2
    # Holder demand uses entry's H100 device, not parent's A100
    for d in synthetic_demand:
        assert d.device_variant == "H100"


def test_demand_reservation_mixed_jobs():
    """Reservation job + regular job: demand is independent per job."""
    state = ControllerState()

    # h100-job: 3 H100 tasks + 3 reservation entries
    h100_req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device(), _h100_device()],
        replicas=3,
    )
    submit_job(state, "h100-job", h100_req)

    a100_req = cluster_pb2.Controller.LaunchJobRequest(
        name="a100-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_a100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "a100-job", a100_req)

    demand = compute_demand_entries(state)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    # 3 synthetic holder tasks from h100-job's reservation
    assert len(synthetic_demand) == 3

    # h100-job: 3 real tasks + a100-job: 2 tasks = 5 real demand
    assert len(real_demand) == 5
    a100_demand = [d for d in real_demand if d.device_variant == "A100"]
    assert len(a100_demand) == 2


def test_demand_no_reservation_passes_all_tasks():
    """Job without reservation emits all task demand entries (no synthetic tasks)."""
    state = ControllerState()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state)
    assert len(demand) == 3
    for d in demand:
        assert not _is_synthetic_demand(state, d)


def test_demand_reservation_independent_per_job():
    """Each job's demand is independent — no cross-job interference."""
    state = ControllerState()

    # Job A: 2 H100 reservation, 2 H100 tasks
    job_a_req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "job-a", job_a_req)

    # Job B: no reservation, 2 H100 tasks (must all pass through)
    job_b_req = cluster_pb2.Controller.LaunchJobRequest(
        name="job-b",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "job-b", job_b_req)

    demand = compute_demand_entries(state)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    # Job A's 2 synthetic holder tasks
    assert len(synthetic_demand) == 2
    # Job A's 2 real tasks + Job B's 2 tasks = 4 real demand
    assert len(real_demand) == 4


# =============================================================================
# Depth-First Scheduling Priority Tests
# =============================================================================


def test_peek_pending_tasks_deeper_job_before_shallow(job_request):
    """Depth-first priority: deeper jobs come before shallow ones in queue order."""
    state = ControllerState()

    # Submit root job and child job (both with 1 CPU)
    submit_job(state, "root", job_request("root"), timestamp_ms=1000)
    submit_job(state, "/test-user/root/child", job_request("child"), timestamp_ms=2000)

    pending = state.peek_pending_tasks()
    assert len(pending) == 2
    # Child (depth 2) should come first
    assert pending[0].job_id == JobName.from_string("/test-user/root/child")
    assert pending[1].job_id == JobName.root("test-user", "root")


def test_peek_pending_tasks_older_root_tree_preferred(job_request):
    """At same depth, older root tree is preferred."""
    state = ControllerState()

    # Submit two root jobs at different timestamps
    req_a = job_request("user-a-job")
    submit_job(state, "user-a-job", req_a, timestamp_ms=1000)

    req_b = job_request("user-b-job")
    submit_job(state, "user-b-job", req_b, timestamp_ms=2000)

    pending = state.peek_pending_tasks()
    assert len(pending) == 2
    # user-a-job submitted first, should come first
    assert pending[0].job_id == JobName.root("test-user", "user-a-job")
    assert pending[1].job_id == JobName.root("test-user", "user-b-job")


def test_peek_pending_tasks_child_of_older_tree_beats_newer_root(job_request):
    """Child of older tree beats root of newer tree."""
    state = ControllerState()

    # Submit old tree
    submit_job(state, "old-tree", job_request("old-tree"), timestamp_ms=1000)

    # Submit new tree
    submit_job(state, "new-tree", job_request("new-tree"), timestamp_ms=2000)

    # Submit child of old tree (depth 2) after new tree
    submit_job(state, "/test-user/old-tree/child", job_request("child"), timestamp_ms=3000)

    pending = state.peek_pending_tasks()
    assert len(pending) == 3

    # Expected order: child (depth 2), old-tree (depth 1, older), new-tree (depth 1, newer)
    assert pending[0].job_id == JobName.from_string("/test-user/old-tree/child")
    assert pending[1].job_id == JobName.root("test-user", "old-tree")
    assert pending[2].job_id == JobName.root("test-user", "new-tree")


def test_peek_pending_tasks_fifo_within_same_depth_and_tree(job_request):
    """FIFO within same depth and tree."""
    state = ControllerState()

    # Submit parent first
    submit_job(state, "tree", job_request("tree"), timestamp_ms=1000)

    # Submit two children at different times
    submit_job(state, "/test-user/tree/child-a", job_request("child-a"), timestamp_ms=2000)
    submit_job(state, "/test-user/tree/child-b", job_request("child-b"), timestamp_ms=3000)

    pending = state.peek_pending_tasks()
    assert len(pending) == 3

    # Both children at depth 2, same root tree — child-a submitted first
    child_tasks = [t for t in pending if t.job_id.parent == JobName.root("test-user", "tree")]
    assert len(child_tasks) == 2
    assert child_tasks[0].job_id == JobName.from_string("/test-user/tree/child-a")
    assert child_tasks[1].job_id == JobName.from_string("/test-user/tree/child-b")


def test_child_job_inherits_root_submitted_at(job_request):
    """Child job inherits root_submitted_at from parent."""
    state = ControllerState()

    # Submit parent at known time
    parent_req = job_request("parent")
    submit_job(state, "parent", parent_req, timestamp_ms=1000)
    parent_job = state.get_job(JobName.root("test-user", "parent"))
    parent_submitted = parent_job.submitted_at

    # Submit child later
    child_req = job_request("child")
    submit_job(state, "/test-user/parent/child", child_req, timestamp_ms=2000)
    child_job = state.get_job(JobName.from_string("/test-user/parent/child"))

    # Child's root_submitted_at should equal parent's
    assert child_job.root_submitted_at == parent_submitted
    assert child_job.root_submitted_at == parent_job.root_submitted_at


def test_requeued_task_maintains_priority_position(job_request, worker_metadata):
    """Requeued task maintains its priority position (deeper job still prioritized)."""
    state = ControllerState()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Submit a deep job and a shallow job
    submit_job(state, "/test-user/tree/deep", job_request("deep"), timestamp_ms=1000)
    submit_job(state, "shallow", job_request("shallow"), timestamp_ms=2000)

    # Initially: deep job comes first
    pending = state.peek_pending_tasks()
    assert len(pending) == 2
    assert pending[0].job_id == JobName.from_string("/test-user/tree/deep")
    assert pending[1].job_id == JobName.root("test-user", "shallow")

    # Dispatch and fail the deep job's task (with retries enabled)
    deep_req = job_request("deep")
    deep_req.max_retries_failure = 1
    deep_tasks = submit_job(state, "/test-user/tree/deep-retry", deep_req, timestamp_ms=3000)
    submit_job(state, "shallow-2", job_request("shallow-2"), timestamp_ms=4000)

    dispatch_task(state, deep_tasks[0], worker_id)
    transition_task(state, deep_tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="Retriable failure")

    # Verify task was requeued
    assert deep_tasks[0].state == cluster_pb2.TASK_STATE_PENDING
    assert deep_tasks[0].can_be_scheduled()

    # Check queue order — requeued deep job should still come before shallow
    pending = state.peek_pending_tasks()
    deep_pending = [t for t in pending if t.job_id == JobName.from_string("/test-user/tree/deep-retry")]
    shallow_pending = [t for t in pending if t.job_id == JobName.root("test-user", "shallow-2")]

    assert len(deep_pending) == 1
    assert len(shallow_pending) == 1

    # Find indices
    deep_idx = pending.index(deep_pending[0])
    shallow_idx = pending.index(shallow_pending[0])
    assert deep_idx < shallow_idx, "Requeued deep task should still come before shallow task"


# =============================================================================
# Heartbeat Dispatch Transition Tests
# =============================================================================


def test_fail_heartbeat_clears_dispatch_when_worker_fails(job_request, worker_metadata):
    """Dispatch buffer is cleared when worker hits failure threshold.

    When consecutive heartbeat failures hit the threshold:
    1. Worker is marked unhealthy
    2. Running tasks transition to WORKER_FAILED
    3. Pending dispatch buffer is cleared (not orphaned)
    """
    from iris.cluster.controller.state import (
        HEARTBEAT_FAILURE_THRESHOLD,
        PendingDispatch,
    )

    state = ControllerState()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Submit and dispatch a task
    tasks = submit_job(state, "j1", job_request("job1"))
    dispatch_task(state, tasks[0], worker_id)

    # Buffer a dispatch for the worker
    fake_request = cluster_pb2.Worker.RunTaskRequest(task_id="/test-user/fake/0")
    state.buffer_dispatch(worker_id, fake_request)

    # Verify dispatch is buffered
    assert worker_id in state._pending_dispatch
    assert len(state._pending_dispatch[worker_id].tasks_to_run) == 1

    # Create a snapshot (simulating begin_heartbeat)
    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None
    assert len(snapshot.tasks_to_run) == 1

    # Verify buffer is now drained
    assert worker_id not in state._pending_dispatch

    # Simulate repeated failures up to threshold
    worker = state.get_worker(worker_id)
    worker.consecutive_failures = HEARTBEAT_FAILURE_THRESHOLD - 1

    # This fail_heartbeat should trigger worker failure
    state.fail_heartbeat(snapshot, "Connection refused")

    # Verify worker is now unhealthy
    assert not worker.healthy

    # Verify dispatch buffer was NOT repopulated (would be orphaned)
    # The fix clears the dispatch buffer when worker fails
    assert (
        worker_id not in state._pending_dispatch
        or not state._pending_dispatch.get(worker_id, PendingDispatch()).tasks_to_run
    )


def test_fail_heartbeat_requeues_dispatch_when_worker_healthy(job_request, worker_metadata):
    """Dispatch buffer is repopulated when worker remains healthy after failure.

    When heartbeat fails but worker is still below failure threshold,
    the dispatches should be requeued for the next heartbeat attempt.
    """

    state = ControllerState()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Submit and dispatch a task
    tasks = submit_job(state, "j1", job_request("job1"))
    dispatch_task(state, tasks[0], worker_id)

    # Buffer a dispatch
    fake_request = cluster_pb2.Worker.RunTaskRequest(task_id="/test-user/fake/0")
    state.buffer_dispatch(worker_id, fake_request)

    # Take snapshot
    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None

    # Fail heartbeat (worker stays healthy - below threshold)
    state.fail_heartbeat(snapshot, "Timeout")

    # Verify worker is still healthy
    worker = state.get_worker(worker_id)
    assert worker.healthy
    assert worker.consecutive_failures == 1

    # Verify dispatch was requeued
    assert worker_id in state._pending_dispatch
    assert len(state._pending_dispatch[worker_id].tasks_to_run) == 1


def test_complete_heartbeat_processes_task_states(job_request, worker_metadata):
    """complete_heartbeat properly processes task state changes from response."""
    state = ControllerState()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Submit and dispatch a task
    tasks = submit_job(state, "j1", job_request("job1"))
    dispatch_task(state, tasks[0], worker_id)

    # Take snapshot
    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None

    # Create a mock response with completed task
    response = cluster_pb2.HeartbeatResponse(
        tasks=[
            cluster_pb2.Controller.WorkerTaskStatus(
                task_id=tasks[0].task_id.to_wire(),
                state=cluster_pb2.TASK_STATE_SUCCEEDED,
                exit_code=0,
                attempt_id=0,
            )
        ]
    )

    # Complete heartbeat
    state.complete_heartbeat(snapshot, response)

    # Verify task is now succeeded
    task = state.get_task(tasks[0].task_id)
    assert task.state == cluster_pb2.TASK_STATE_SUCCEEDED

    # Verify job is succeeded
    job = state.get_job(tasks[0].job_id)
    assert job.state == cluster_pb2.JOB_STATE_SUCCEEDED


# =============================================================================
# Demand Dry-Run Scheduling Tests
#
# These tests verify that compute_demand_entries runs a dry-run scheduling pass
# to absorb tasks into existing worker capacity, and only emits demand for
# truly unschedulable tasks (not building-limited ones).
# =============================================================================


def _gpu_worker_metadata(
    *,
    cpu: int = 128,
    memory_gb: int = 256,
    variant: str = "H100",
    gpu_count: int = 8,
) -> cluster_pb2.WorkerMetadata:
    """Create worker metadata for a GPU worker."""
    return cluster_pb2.WorkerMetadata(
        hostname="gpu-worker",
        ip_address="10.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_gb * 1024**3,
        disk_bytes=100 * 1024**3,
        device=cluster_pb2.DeviceConfig(
            gpu=cluster_pb2.GpuDevice(variant=variant, count=gpu_count),
        ),
    )


def _tpu_worker_metadata(
    *,
    cpu: int = 128,
    memory_gb: int = 256,
    variant: str = "v5litepod-16",
    chip_count: int = 8,
) -> cluster_pb2.WorkerMetadata:
    """Create worker metadata for a TPU worker."""
    return cluster_pb2.WorkerMetadata(
        hostname="tpu-worker",
        ip_address="10.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_gb * 1024**3,
        disk_bytes=100 * 1024**3,
        device=cluster_pb2.DeviceConfig(
            tpu=cluster_pb2.TpuDevice(variant=variant, chip_count=chip_count),
        ),
    )


def _cpu_worker_metadata(
    *,
    cpu: int = 128,
    memory_gb: int = 256,
) -> cluster_pb2.WorkerMetadata:
    return cluster_pb2.WorkerMetadata(
        hostname="cpu-worker",
        ip_address="10.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_gb * 1024**3,
        disk_bytes=100 * 1024**3,
        device=cluster_pb2.DeviceConfig(
            cpu=cluster_pb2.CpuDevice(variant="cpu"),
        ),
    )


def test_demand_excludes_building_limited_tasks():
    """Worker has resources but is at building limit -> no demand emitted."""
    state = ControllerState()
    scheduler = Scheduler(max_building_tasks_per_worker=2)

    # Register a CPU worker with plenty of capacity
    wid = register_worker(state, "w1", "10.0.0.1:8080", _cpu_worker_metadata(cpu=128, memory_gb=256))

    # Submit a job with 1 pending CPU task
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="cpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    # Fill the worker with 2 building tasks (at the building limit).
    # These use minimal resources so the worker still has plenty of capacity.
    build_req = cluster_pb2.Controller.LaunchJobRequest(
        name="build-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=100,
            memory_bytes=1024**2,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    build_tasks = submit_job(state, "build-job", build_req)
    for bt in build_tasks:
        dispatch_task(state, bt, wid)
        transition_task(state, bt.task_id, cluster_pb2.TASK_STATE_BUILDING)

    # Now w1 has 2 building tasks (at limit), but has plenty of CPU/memory.
    # The pending task from j1 should be building-limited, not truly unschedulable.
    workers = state.get_available_workers()
    demand = compute_demand_entries(state, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 0, "Building-limited task should not generate demand"


def test_demand_includes_truly_unschedulable_tasks():
    """No worker with matching device type -> demand IS emitted."""
    state = ControllerState()
    scheduler = Scheduler()

    # Register a CPU-only worker
    register_worker(state, "w1", "10.0.0.1:8080", _cpu_worker_metadata())

    # Submit a job requiring H100 GPUs
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    workers = state.get_available_workers()
    demand = compute_demand_entries(state, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 1, "Task with no matching device should generate demand"


def test_demand_includes_resource_exhausted_tasks():
    """Worker has right device but insufficient CPU -> demand IS emitted."""
    state = ControllerState()
    scheduler = Scheduler()

    # Register a GPU worker with only 1 CPU core
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata(cpu=1))

    # Submit a job requiring 4 CPU cores
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=4000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    workers = state.get_available_workers()
    demand = compute_demand_entries(state, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 1, "Task exceeding worker CPU should generate demand"


def test_demand_holders_absorbed_by_dry_run():
    """Holder tasks participate in the dry-run and are absorbed when workers exist.

    Unlike the old design where holders always generated demand, they now
    participate in the dry-run like normal tasks and are absorbed when matching
    workers have available capacity.
    """
    state = ControllerState()
    scheduler = Scheduler()

    # Register a large GPU worker with capacity for 1 task
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata(cpu=2, memory_gb=4))

    # Submit a job with reservation (2 entries) and 2 tasks.
    # Worker can fit 1 task — so 1 task absorbed, 3 remain as demand.
    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    workers = state.get_available_workers()
    demand = compute_demand_entries(state, scheduler, workers)
    # Worker fits 1 task (holder or real). 3 remaining generate demand.
    assert len(demand) == 3


def test_demand_absorbs_capacity_before_emitting():
    """2 workers fit 1 task each, 3 pending tasks -> only 1 demand entry."""
    state = ControllerState()
    scheduler = Scheduler()

    # Register 2 GPU workers, each with enough capacity for 1 task
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata(cpu=2, memory_gb=4))
    register_worker(state, "w2", "10.0.0.2:8080", _gpu_worker_metadata(cpu=2, memory_gb=4))

    # Submit 3 tasks each needing 2 CPU cores (each worker fits exactly 1)
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=2000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    workers = state.get_available_workers()
    demand = compute_demand_entries(state, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 1, "Only 1 of 3 tasks should generate demand (2 absorbed)"


def test_demand_no_workers_falls_back_to_all_pending():
    """When no workers provided, all pending tasks generate demand (backward compat)."""
    state = ControllerState()

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    # No scheduler, no workers -> all tasks become demand
    demand = compute_demand_entries(state)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 3


def test_demand_building_limited_with_multiple_workers():
    """All matching workers at building limit -> no demand, even with multiple workers."""
    state = ControllerState()
    scheduler = Scheduler(max_building_tasks_per_worker=1)

    # Register 2 CPU workers
    wid1 = register_worker(state, "w1", "10.0.0.1:8080", _cpu_worker_metadata())
    wid2 = register_worker(state, "w2", "10.0.0.2:8080", _cpu_worker_metadata())

    # Fill both workers with 1 building task each (at limit since max=1).
    # Use minimal resources so workers retain plenty of capacity.
    for i, wid in enumerate([wid1, wid2]):
        build_req = cluster_pb2.Controller.LaunchJobRequest(
            name=f"build-{i}",
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(
                cpu_millicores=100,
                memory_bytes=1024**2,
            ),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )
        build_tasks = submit_job(state, f"build-{i}", build_req)
        dispatch_task(state, build_tasks[0], wid)
        transition_task(state, build_tasks[0].task_id, cluster_pb2.TASK_STATE_BUILDING)

    # Submit a new task
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="pending-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "pending-job", req)

    workers = state.get_available_workers()
    demand = compute_demand_entries(state, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 0, "All workers at building limit -> no demand"


def test_demand_mixed_building_limited_and_unschedulable():
    """Some tasks building-limited, some truly unschedulable -> only unschedulable emit demand."""
    state = ControllerState()
    scheduler = Scheduler(max_building_tasks_per_worker=1)

    # Register 1 GPU worker at building limit.
    # Use a minimal CPU task to fill the building slot so GPU capacity stays intact.
    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
    build_req = cluster_pb2.Controller.LaunchJobRequest(
        name="build-0",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=100,
            memory_bytes=1024**2,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    build_tasks = submit_job(state, "build-0", build_req)
    dispatch_task(state, build_tasks[0], wid)
    transition_task(state, build_tasks[0].task_id, cluster_pb2.TASK_STATE_BUILDING)

    # Task 1: H100 job (building-limited, worker has resources but at limit)
    h100_req = cluster_pb2.Controller.LaunchJobRequest(
        name="h100-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "h100-job", h100_req)

    # Task 2: A100 job (truly unschedulable, no A100 workers exist)
    a100_req = cluster_pb2.Controller.LaunchJobRequest(
        name="a100-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_a100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "a100-job", a100_req)

    workers = state.get_available_workers()
    demand = compute_demand_entries(state, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(task_demand) == 1
    assert "a100-job" in task_demand[0].task_ids[0], "Only A100 task should emit demand"


# =============================================================================
# Holder Task Zero-Resource Tests
# =============================================================================


def test_holder_tasks_consume_zero_resources():
    """Holder tasks consume zero resources when assigned to workers."""
    state = ControllerState()

    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = state.get_job_tasks(holder_job_id)
    assert len(holder_tasks) == 1

    worker_before = state.get_worker(wid)
    gpus_before = worker_before.available_gpus

    # Assign holder task
    state.handle_event(TaskAssignedEvent(task_id=holder_tasks[0].task_id, worker_id=wid))

    # Worker's available GPUs should NOT decrease (zero resources)
    worker_after = state.get_worker(wid)
    assert worker_after.available_gpus == gpus_before

    # But the task should be tracked in running_tasks
    assert holder_tasks[0].task_id in worker_after.running_tasks


def test_holder_task_cleanup_releases_no_resources():
    """When a holder task finishes, it doesn't release resources it never committed."""
    state = ControllerState()

    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = state.get_job_tasks(holder_job_id)

    # Assign holder task
    state.handle_event(TaskAssignedEvent(task_id=holder_tasks[0].task_id, worker_id=wid))

    worker_before = state.get_worker(wid)
    gpus_before = worker_before.available_gpus

    # Kill the holder task via parent job cancellation
    parent_job_id = JobName.root("test-user", "j1")
    state.handle_event(JobCancelledEvent(job_id=parent_job_id, reason="test"))

    # Worker GPUs should be unchanged (nothing to release)
    worker_after = state.get_worker(wid)
    assert worker_after.available_gpus == gpus_before
    assert holder_tasks[0].task_id not in worker_after.running_tasks


def test_holder_tasks_excluded_from_building_counts():
    """Holder tasks in ASSIGNED state should not consume building slots.

    Without this exclusion, a worker holding only a reservation task would be
    permanently "at building limit" and the real reserved task could never be
    assigned to that otherwise idle worker.
    """
    state = ControllerState()

    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = state.get_job_tasks(holder_job_id)
    assert len(holder_tasks) == 1

    # Assign holder task — it goes to ASSIGNED state
    state.handle_event(TaskAssignedEvent(task_id=holder_tasks[0].task_id, worker_id=wid))
    assert holder_tasks[0].state == cluster_pb2.TASK_STATE_ASSIGNED

    # Building counts should NOT include the holder task
    building_counts = state.snapshot_building_counts()
    assert building_counts.get(wid, 0) == 0


def test_snapshot_round_trip_preserves_reservation_holder():
    """Snapshot save/restore round-trip preserves is_reservation_holder flag."""
    from iris.cluster.controller.snapshot import create_snapshot, restore_snapshot

    state = ControllerState()

    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_job = state.get_job(holder_job_id)
    assert holder_job is not None
    assert holder_job.is_reservation_holder is True

    # Save and restore
    snap_result = create_snapshot(state, reservation_claims={})
    restored_state = ControllerState()
    restore_snapshot(snap_result.proto, restored_state)

    restored_holder = restored_state.get_job(holder_job_id)
    assert restored_holder is not None
    assert restored_holder.is_reservation_holder is True

    # Parent should not be a holder
    parent_job_id = JobName.root("test-user", "j1")
    restored_parent = restored_state.get_job(parent_job_id)
    assert restored_parent is not None
    assert restored_parent.is_reservation_holder is False
