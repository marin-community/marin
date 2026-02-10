# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller RPC service implementation.

These tests verify the RPC contract (input -> output) of the ControllerServiceImpl.
State changes are verified via RPC calls rather than internal state inspection.
"""

from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.controller.events import TaskAssignedEvent, TaskStateChangedEvent
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import ControllerState, ControllerTask
from iris.cluster.types import JobName, WorkerId
from iris.logging import BufferedLogRecord, LogRingBuffer
from iris.rpc import cluster_pb2


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    """Create a minimal RuntimeEntrypoint proto for testing."""
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


# =============================================================================
# Test Helpers - Wrap handle_event() for common test patterns
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


def transition_task(state: ControllerState, task: ControllerTask, new_state: int, *, error: str | None = None) -> None:
    """Transition a task to a new state via handle_event."""
    state.handle_event(
        TaskStateChangedEvent(
            task_id=task.task_id,
            new_state=new_state,
            attempt_id=task.current_attempt_id,
            error=error,
        )
    )


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(
        name: str = "test-job",
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 0,  # Default to 0 for tests (no implicit retries)
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        job_name = JobName.from_string(name) if name.startswith("/") else JobName.root(name)
        return cluster_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            replicas=replicas,
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


@pytest.fixture
def state():
    """Create a fresh ControllerState for each test."""
    return ControllerState()


class MockSchedulerWake:
    """Mock object that tracks scheduler protocol calls."""

    def __init__(self):
        self.wake = Mock()
        self.kill_tasks_on_workers = Mock()


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler with wake() method."""
    return MockSchedulerWake()


@pytest.fixture
def service(state, mock_scheduler):
    """Create a ControllerServiceImpl for testing."""
    return ControllerServiceImpl(state, mock_scheduler, bundle_prefix="file:///tmp/iris-test-bundles")


# =============================================================================
# Job Launch Tests
# =============================================================================


def test_launch_job_returns_job_id(service, job_request):
    """Verify launch_job returns a job_id and job can be queried via RPC."""
    request = job_request("test-job")

    response = service.launch_job(request, None)

    assert response.job_id == JobName.root("test-job").to_wire()

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-job").to_wire()), None
    )
    assert status_response.job.job_id == JobName.root("test-job").to_wire()
    assert status_response.job.state == cluster_pb2.JOB_STATE_PENDING


def test_launch_job_rejects_duplicate_name(service, job_request):
    """Verify launch_job rejects duplicate job names for running jobs."""
    request = job_request("duplicate-job")

    response = service.launch_job(request, None)
    assert response.job_id == JobName.root("duplicate-job").to_wire()

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "still running" in exc_info.value.message


def test_launch_job_replaces_finished_job_by_default(service, state, job_request):
    """Verify launch_job replaces finished jobs by default."""
    request = job_request("replaceable-job")

    # Submit initial job
    response = service.launch_job(request, None)
    assert response.job_id == JobName.root("replaceable-job").to_wire()

    # Mark the job as failed
    job = state.get_job(JobName.root("replaceable-job"))
    assert job is not None
    tasks = state.get_job_tasks(job.job_id)
    assert len(tasks) == 1
    task = tasks[0]
    state.handle_event(
        TaskStateChangedEvent(
            task_id=task.task_id,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            attempt_id=-1,
            error="Test failure",
        )
    )

    # Verify job is now failed
    job = state.get_job(JobName.root("replaceable-job"))
    assert job.state == cluster_pb2.JOB_STATE_FAILED

    # Submit again - should succeed (replaces the finished job)
    response = service.launch_job(request, None)
    assert response.job_id == JobName.root("replaceable-job").to_wire()

    # Verify the new job is pending
    job = state.get_job(JobName.root("replaceable-job"))
    assert job.state == cluster_pb2.JOB_STATE_PENDING


def test_launch_job_fail_if_exists_prevents_replacement(service, state, job_request):
    """Verify fail_if_exists=true prevents replacing finished jobs."""
    request = job_request("no-replace-job")

    # Submit initial job
    response = service.launch_job(request, None)
    assert response.job_id == JobName.root("no-replace-job").to_wire()

    # Mark the job as succeeded
    job = state.get_job(JobName.root("no-replace-job"))
    tasks = state.get_job_tasks(job.job_id)
    task = tasks[0]
    state.handle_event(
        TaskStateChangedEvent(
            task_id=task.task_id,
            new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
            attempt_id=-1,
        )
    )

    # Verify job is now succeeded
    job = state.get_job(JobName.root("no-replace-job"))
    assert job.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Submit again with fail_if_exists=true - should fail
    request_no_replace = job_request("no-replace-job")
    request_no_replace.fail_if_exists = True

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request_no_replace, None)

    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "SUCCEEDED" in exc_info.value.message


def test_launch_job_rejects_empty_name(service, state):
    """Verify launch_job rejects empty job names."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
    )

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.INVALID_ARGUMENT
    assert "name is required" in exc_info.value.message.lower()


# =============================================================================
# Job Status Tests
# =============================================================================


def test_get_job_status_returns_status(service, job_request):
    """Verify get_job_status returns correct status for launched job."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-job").to_wire())
    response = service.get_job_status(request, None)

    assert response.job.job_id == JobName.root("test-job").to_wire()
    assert response.job.state == cluster_pb2.JOB_STATE_PENDING


def test_get_job_status_not_found(service):
    """Verify get_job_status raises ConnectError for unknown job."""
    request = cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("nonexistent").to_wire())

    with pytest.raises(ConnectError) as exc_info:
        service.get_job_status(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


# =============================================================================
# Job Termination Tests
# =============================================================================


def test_terminate_job_marks_as_killed(service, job_request):
    """Verify terminate_job sets job state to KILLED via get_job_status."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-job").to_wire())
    response = service.terminate_job(request, None)

    assert isinstance(response, cluster_pb2.Empty)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-job").to_wire()), None
    )
    assert status_response.job.state == cluster_pb2.JOB_STATE_KILLED
    assert status_response.job.finished_at.epoch_ms > 0


def test_terminate_job_not_found(service):
    """Verify terminate_job raises ConnectError for unknown job."""
    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("nonexistent").to_wire())

    with pytest.raises(ConnectError) as exc_info:
        service.terminate_job(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


def test_terminate_pending_job(service, job_request):
    """Verify terminate_job works on pending jobs (not just running)."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-job").to_wire())
    service.terminate_job(request, None)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-job").to_wire()), None
    )
    assert status_response.job.state == cluster_pb2.JOB_STATE_KILLED
    assert status_response.job.finished_at.epoch_ms > 0


def test_terminate_job_cascades_to_children(service, job_request):
    """Verify terminate_job terminates all children when parent is terminated."""
    service.launch_job(job_request("parent"), None)
    service.launch_job(job_request("/parent/child1"), None)
    service.launch_job(job_request("/parent/child2"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("parent").to_wire())
    service.terminate_job(request, None)

    # Verify all jobs are killed via get_job_status RPC
    for job_name in [
        JobName.root("parent"),
        JobName.from_string("/parent/child1"),
        JobName.from_string("/parent/child2"),
    ]:
        status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=job_name.to_wire()), None)
        assert status.job.state == cluster_pb2.JOB_STATE_KILLED, f"Job {job_name} should be KILLED"


def test_terminate_job_only_affects_descendants(service, job_request):
    """Verify terminate_job does not affect sibling jobs."""
    service.launch_job(job_request("parent"), None)
    service.launch_job(job_request("/parent/child1"), None)
    service.launch_job(job_request("/parent/child2"), None)

    # Terminate only child1
    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.from_string("/parent/child1").to_wire())
    service.terminate_job(request, None)

    # Verify states via get_job_status RPC
    child1_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/parent/child1").to_wire()), None
    )
    assert child1_status.job.state == cluster_pb2.JOB_STATE_KILLED

    child2_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/parent/child2").to_wire()), None
    )
    assert child2_status.job.state == cluster_pb2.JOB_STATE_PENDING

    parent_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("parent").to_wire()), None
    )
    assert parent_status.job.state == cluster_pb2.JOB_STATE_PENDING


def test_terminate_job_skips_already_finished_children(service, state, job_request):
    """Verify terminate_job skips children already in terminal state."""
    from iris.cluster.controller.state import ControllerJob
    from iris.time_utils import Timestamp

    # Launch parent via RPC
    service.launch_job(job_request("parent"), None)

    # Create a child that's already succeeded (need to set up via state since
    # we can't naturally get a job to SUCCEEDED without worker interaction)
    child_succeeded = ControllerJob(
        job_id=JobName.from_string("/parent/child-succeeded"),
        request=job_request("/parent/child-succeeded"),
        state=cluster_pb2.JOB_STATE_SUCCEEDED,
        finished_at=Timestamp.from_ms(12345),
    )
    state.add_job(child_succeeded)

    # Launch running child via RPC
    service.launch_job(job_request("/parent/child-running"), None)

    # Terminate parent
    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("parent").to_wire())
    service.terminate_job(request, None)

    # Verify states via get_job_status RPC
    succeeded_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/parent/child-succeeded").to_wire()),
        None,
    )
    assert succeeded_status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED

    running_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/parent/child-running").to_wire()),
        None,
    )
    assert running_status.job.state == cluster_pb2.JOB_STATE_KILLED

    parent_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("parent").to_wire()),
        None,
    )
    assert parent_status.job.state == cluster_pb2.JOB_STATE_KILLED


def test_parent_job_failure_cascades_to_children(service, state, job_request):
    """Verify when a parent job fails, all children are automatically cancelled."""
    # Launch parent and children via RPC
    service.launch_job(job_request("parent"), None)
    service.launch_job(job_request("/parent/child1"), None)
    service.launch_job(job_request("/parent/child2"), None)

    # Get parent task and mark it as failed
    parent_job = state.get_job(JobName.root("parent"))
    parent_tasks = state.get_job_tasks(parent_job.job_id)
    parent_task = parent_tasks[0]

    state.handle_event(
        TaskStateChangedEvent(
            task_id=parent_task.task_id,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            attempt_id=-1,
            error="Parent task failed",
        )
    )

    # Verify all jobs are now in terminal states via get_job_status RPC
    parent_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("parent").to_wire()), None
    )
    assert parent_status.job.state == cluster_pb2.JOB_STATE_FAILED

    child1_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/parent/child1").to_wire()), None
    )
    assert child1_status.job.state == cluster_pb2.JOB_STATE_KILLED, "Child 1 should be killed when parent fails"

    child2_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/parent/child2").to_wire()), None
    )
    assert child2_status.job.state == cluster_pb2.JOB_STATE_KILLED, "Child 2 should be killed when parent fails"


def test_launch_job_rejects_child_of_failed_parent(service, state, job_request):
    """Verify launch_job rejects submissions to a failed parent's namespace."""
    # Launch and fail parent
    service.launch_job(job_request("failed-parent"), None)
    parent_job = state.get_job(JobName.root("failed-parent"))
    parent_tasks = state.get_job_tasks(parent_job.job_id)
    parent_task = parent_tasks[0]

    state.handle_event(
        TaskStateChangedEvent(
            task_id=parent_task.task_id,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            attempt_id=-1,
            error="Parent task failed",
        )
    )

    # Try to submit a child job - should fail
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(job_request("/failed-parent/new-child"), None)

    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert "terminated" in exc_info.value.message.lower() or "failed" in exc_info.value.message.lower()


# =============================================================================
# Job List Tests
# =============================================================================


def test_list_jobs_returns_all_jobs(service, job_request):
    """Verify list_jobs returns all jobs launched via RPC."""
    service.launch_job(job_request("job-1"), None)
    service.launch_job(job_request("job-2"), None)
    service.launch_job(job_request("job-3"), None)

    # Terminate one to get different state
    service.terminate_job(cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("job-3").to_wire()), None)

    request = cluster_pb2.Controller.ListJobsRequest()
    response = service.list_jobs(request, None)

    assert len(response.jobs) == 3
    job_ids = {j.job_id for j in response.jobs}
    assert job_ids == {
        JobName.root("job-1").to_wire(),
        JobName.root("job-2").to_wire(),
        JobName.root("job-3").to_wire(),
    }

    states_by_id = {j.job_id: j.state for j in response.jobs}
    assert states_by_id[JobName.root("job-1").to_wire()] == cluster_pb2.JOB_STATE_PENDING
    assert states_by_id[JobName.root("job-2").to_wire()] == cluster_pb2.JOB_STATE_PENDING
    assert states_by_id[JobName.root("job-3").to_wire()] == cluster_pb2.JOB_STATE_KILLED


# =============================================================================
# Worker Tests
# =============================================================================


def test_list_workers_returns_all(service, worker_metadata):
    """Verify list_workers returns all registered workers."""
    for i in range(3):
        request = cluster_pb2.Controller.RegisterRequest(
            address=f"host{i}:8080",
            metadata=worker_metadata(),
        )
        service.register(request, None)

    request = cluster_pb2.Controller.ListWorkersRequest()
    response = service.list_workers(request, None)

    assert len(response.workers) == 3

    # All workers should be healthy after registration
    for w in response.workers:
        assert w.healthy is True


def test_get_process_logs():
    """Test GetProcessLogs RPC retrieves logs from the buffer."""

    state = ControllerState()
    mock_scheduler = MockSchedulerWake()
    log_buffer = LogRingBuffer(maxlen=100)

    # Add some test log records
    log_buffer.append(BufferedLogRecord(timestamp=1000.0, level="INFO", logger_name="iris.test", message="Test log 1"))
    log_buffer.append(
        BufferedLogRecord(timestamp=1001.0, level="DEBUG", logger_name="iris.cluster.vm", message="Autoscaler log")
    )
    log_buffer.append(BufferedLogRecord(timestamp=1002.0, level="ERROR", logger_name="iris.test", message="Test log 2"))

    service = ControllerServiceImpl(
        state, mock_scheduler, bundle_prefix="file:///tmp/test-bundles", log_buffer=log_buffer
    )

    # Test: Get all logs
    response = service.get_process_logs(cluster_pb2.Controller.GetProcessLogsRequest(prefix="", limit=0), None)
    assert len(response.records) == 3
    assert response.records[0].message == "Test log 1"
    assert response.records[1].logger_name == "iris.cluster.vm"
    assert response.records[2].level == "ERROR"

    # Test: Filter by prefix
    response = service.get_process_logs(
        cluster_pb2.Controller.GetProcessLogsRequest(prefix="iris.cluster.vm", limit=0), None
    )
    assert len(response.records) == 1
    assert response.records[0].message == "Autoscaler log"

    # Test: Limit results
    response = service.get_process_logs(cluster_pb2.Controller.GetProcessLogsRequest(prefix="", limit=2), None)
    assert len(response.records) == 2
    assert response.records[0].message == "Autoscaler log"
    assert response.records[1].message == "Test log 2"


def test_get_process_logs_no_buffer():
    """Test GetProcessLogs returns empty when buffer is None."""
    state = ControllerState()
    mock_scheduler = MockSchedulerWake()
    service = ControllerServiceImpl(state, mock_scheduler, bundle_prefix="file:///tmp/test-bundles", log_buffer=None)

    response = service.get_process_logs(cluster_pb2.Controller.GetProcessLogsRequest(prefix="", limit=0), None)
    assert len(response.records) == 0
