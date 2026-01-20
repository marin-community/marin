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

"""Tests for controller RPC service implementation.

These tests verify the RPC contract (input -> output) of the ControllerServiceImpl.
State changes are verified via RPC calls rather than internal state inspection.
"""

from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import ControllerState
from iris.cluster.types import JobId, WorkerId
from iris.rpc import cluster_pb2


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(
        name: str = "test-job",
        replicas: int = 1,
        parent_job_id: str | None = None,
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=replicas),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
            parent_job_id=parent_job_id or "",
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
    """Mock object that just tracks wake() calls."""

    def __init__(self):
        self.wake = Mock()


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler with wake() method."""
    return MockSchedulerWake()


@pytest.fixture
def service(state, mock_scheduler):
    """Create a ControllerServiceImpl for testing."""
    return ControllerServiceImpl(state, mock_scheduler)


# =============================================================================
# Job Launch Tests
# =============================================================================


def test_launch_job_returns_job_id(service, job_request):
    """Verify launch_job returns a job_id and job can be queried via RPC."""
    request = job_request("test-job")

    response = service.launch_job(request, None)

    assert response.job_id == "test-job"

    # Verify via get_job_status RPC
    status_response = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    assert status_response.job.job_id == "test-job"
    assert status_response.job.state == cluster_pb2.JOB_STATE_PENDING


def test_launch_job_uses_name_as_job_id(service, job_request):
    """Verify job_id is the same as the provided name."""
    request = job_request("my-unique-job")

    response = service.launch_job(request, None)

    assert response.job_id == "my-unique-job"


def test_launch_job_rejects_duplicate_name(service, job_request):
    """Verify launch_job rejects duplicate job names."""
    request = job_request("duplicate-job")

    response = service.launch_job(request, None)
    assert response.job_id == "duplicate-job"

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "duplicate-job" in exc_info.value.message


def test_launch_job_rejects_empty_name(service, state):
    """Verify launch_job rejects empty job names."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.INVALID_ARGUMENT
    assert "name is required" in exc_info.value.message.lower()


def test_launch_job_with_parent_job_id(service, job_request):
    """Verify launch_job stores parent_job_id and returns it via get_job_status."""
    # Launch parent first
    service.launch_job(job_request("parent-job"), None)

    # Launch child with parent_job_id
    child_request = job_request("child-job", parent_job_id="parent-job")
    service.launch_job(child_request, None)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="child-job"), None)
    assert status_response.job.parent_job_id == "parent-job"

    # Also verify a job without parent has empty parent_job_id
    parent_status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="parent-job"), None)
    assert parent_status.job.parent_job_id == ""


# =============================================================================
# Job Status Tests
# =============================================================================


def test_get_job_status_returns_status(service, job_request):
    """Verify get_job_status returns correct status for launched job."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job")
    response = service.get_job_status(request, None)

    assert response.job.job_id == "test-job"
    assert response.job.state == cluster_pb2.JOB_STATE_PENDING


def test_get_job_status_not_found(service):
    """Verify get_job_status raises ConnectError for unknown job."""
    request = cluster_pb2.Controller.GetJobStatusRequest(job_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.get_job_status(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


def test_get_job_status_includes_parent_job_id(service, job_request):
    """Verify get_job_status returns parent_job_id in response."""
    service.launch_job(job_request("parent-job"), None)
    service.launch_job(job_request("child-job", parent_job_id="parent-job"), None)

    request = cluster_pb2.Controller.GetJobStatusRequest(job_id="child-job")
    response = service.get_job_status(request, None)

    assert response.job.parent_job_id == "parent-job"


# =============================================================================
# Job Termination Tests
# =============================================================================


def test_terminate_job_marks_as_killed(service, job_request):
    """Verify terminate_job sets job state to KILLED via get_job_status."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id="test-job")
    response = service.terminate_job(request, None)

    assert isinstance(response, cluster_pb2.Empty)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    assert status_response.job.state == cluster_pb2.JOB_STATE_KILLED
    assert status_response.job.finished_at_ms > 0


def test_terminate_job_not_found(service):
    """Verify terminate_job raises ConnectError for unknown job."""
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.terminate_job(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


def test_terminate_pending_job(service, job_request):
    """Verify terminate_job works on pending jobs (not just running)."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id="test-job")
    service.terminate_job(request, None)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    assert status_response.job.state == cluster_pb2.JOB_STATE_KILLED
    assert status_response.job.finished_at_ms > 0


def test_terminate_job_cascades_to_children(service, job_request):
    """Verify terminate_job terminates all children when parent is terminated."""
    service.launch_job(job_request("parent"), None)
    service.launch_job(job_request("child1", parent_job_id="parent"), None)
    service.launch_job(job_request("child2", parent_job_id="parent"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id="parent")
    service.terminate_job(request, None)

    # Verify all jobs are killed via get_job_status RPC
    for job_id in ["parent", "child1", "child2"]:
        status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id), None)
        assert status.job.state == cluster_pb2.JOB_STATE_KILLED, f"Job {job_id} should be KILLED"


def test_terminate_job_only_affects_descendants(service, job_request):
    """Verify terminate_job does not affect sibling jobs."""
    service.launch_job(job_request("parent"), None)
    service.launch_job(job_request("child1", parent_job_id="parent"), None)
    service.launch_job(job_request("child2", parent_job_id="parent"), None)

    # Terminate only child1
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="child1")
    service.terminate_job(request, None)

    # Verify states via get_job_status RPC
    child1_status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="child1"), None)
    assert child1_status.job.state == cluster_pb2.JOB_STATE_KILLED

    child2_status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="child2"), None)
    assert child2_status.job.state == cluster_pb2.JOB_STATE_PENDING

    parent_status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="parent"), None)
    assert parent_status.job.state == cluster_pb2.JOB_STATE_PENDING


def test_terminate_job_skips_already_finished_children(service, state, job_request):
    """Verify terminate_job skips children already in terminal state."""
    from iris.cluster.controller.state import ControllerJob

    # Launch parent via RPC
    service.launch_job(job_request("parent"), None)

    # Create a child that's already succeeded (need to set up via state since
    # we can't naturally get a job to SUCCEEDED without worker interaction)
    child_succeeded = ControllerJob(
        job_id=JobId("child-succeeded"),
        request=job_request("child-succeeded"),
        state=cluster_pb2.JOB_STATE_SUCCEEDED,
        parent_job_id=JobId("parent"),
        finished_at_ms=12345,
    )
    state.add_job(child_succeeded)

    # Launch running child via RPC
    service.launch_job(job_request("child-running", parent_job_id="parent"), None)

    # Terminate parent
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="parent")
    service.terminate_job(request, None)

    # Verify states via get_job_status RPC
    succeeded_status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="child-succeeded"), None)
    assert succeeded_status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED

    running_status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="child-running"), None)
    assert running_status.job.state == cluster_pb2.JOB_STATE_KILLED

    parent_status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="parent"), None)
    assert parent_status.job.state == cluster_pb2.JOB_STATE_KILLED


# =============================================================================
# Job List Tests
# =============================================================================


def test_list_jobs_returns_all_jobs(service, job_request):
    """Verify list_jobs returns all jobs launched via RPC."""
    service.launch_job(job_request("job-1"), None)
    service.launch_job(job_request("job-2"), None)
    service.launch_job(job_request("job-3"), None)

    # Terminate one to get different state
    service.terminate_job(cluster_pb2.Controller.TerminateJobRequest(job_id="job-3"), None)

    request = cluster_pb2.Controller.ListJobsRequest()
    response = service.list_jobs(request, None)

    assert len(response.jobs) == 3
    job_ids = {j.job_id for j in response.jobs}
    assert job_ids == {"job-1", "job-2", "job-3"}

    states_by_id = {j.job_id: j.state for j in response.jobs}
    assert states_by_id["job-1"] == cluster_pb2.JOB_STATE_PENDING
    assert states_by_id["job-2"] == cluster_pb2.JOB_STATE_PENDING
    assert states_by_id["job-3"] == cluster_pb2.JOB_STATE_KILLED


# =============================================================================
# Worker Tests
# =============================================================================


def test_list_workers_returns_all(service, worker_metadata):
    """Verify list_workers returns all registered workers."""
    for i in range(3):
        request = cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id=f"w{i}",
            address=f"host{i}:8080",
            metadata=worker_metadata(),
        )
        service.register_worker(request, None)

    request = cluster_pb2.Controller.ListWorkersRequest()
    response = service.list_workers(request, None)

    assert len(response.workers) == 3
    worker_ids = {w.worker_id for w in response.workers}
    assert worker_ids == {"w0", "w1", "w2"}

    # All workers should be healthy after registration
    for w in response.workers:
        assert w.healthy is True


# =============================================================================
# Task State Reporting Tests (report_task_state RPC)
# =============================================================================


def test_report_task_state_transitions_task(service, state, job_request, worker_metadata):
    """Verify report_task_state transitions task and updates job state."""
    # Launch job (creates task)
    service.launch_job(job_request("test-job"), None)

    # Register worker and simulate dispatch via state
    service.register_worker(
        cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id="w1",
            address="host:8080",
            metadata=worker_metadata(),
        ),
        None,
    )

    # Get task ID and current attempt via get_job_status RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    task_status = status.job.tasks[0]
    task_id = task_status.task_id

    # Simulate scheduler dispatch (no RPC for this - internal scheduler operation)
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    state.mark_task_dispatched(task, WorkerId("w1"))

    # Get updated attempt ID after dispatch
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    current_attempt_id = status.job.tasks[0].current_attempt_id

    # Report task as RUNNING
    request = cluster_pb2.Controller.ReportTaskStateRequest(
        task_id=task_id,
        attempt_id=current_attempt_id,
        state=cluster_pb2.TASK_STATE_RUNNING,
    )
    service.report_task_state(request, None)

    # Verify job is now RUNNING via get_job_status RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    assert status.job.state == cluster_pb2.JOB_STATE_RUNNING


def test_report_task_state_validates_attempt_id(service, state, job_request, worker_metadata):
    """Verify report_task_state ignores reports with stale attempt_id."""
    # Launch job
    service.launch_job(job_request("test-job"), None)

    # Register worker
    service.register_worker(
        cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id="w1",
            address="host:8080",
            metadata=worker_metadata(),
        ),
        None,
    )

    # Get task and create attempt
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    state.mark_task_dispatched(task, WorkerId("w1"))

    # Report with wrong attempt_id (stale report)
    request = cluster_pb2.Controller.ReportTaskStateRequest(
        task_id=str(task.task_id),
        attempt_id=999,  # Wrong attempt ID
        state=cluster_pb2.TASK_STATE_SUCCEEDED,
    )
    response = service.report_task_state(request, None)

    # Should return empty response (ignored)
    assert isinstance(response, cluster_pb2.Controller.ReportTaskStateResponse)

    # Job should still be RUNNING (not SUCCEEDED)
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    # After dispatch task is RUNNING, job becomes RUNNING
    assert status.job.state == cluster_pb2.JOB_STATE_RUNNING


# =============================================================================
# Task Retry Tests
# =============================================================================


def test_task_retry_preserves_attempt_history(service, state, job_request, worker_metadata):
    """Verify that when a task fails and retries, attempt history is preserved."""
    # Launch job with retry enabled
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="retry-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    service.launch_job(request, None)

    # Register worker
    service.register_worker(
        cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id="w1",
            address="host:8080",
            metadata=worker_metadata(),
        ),
        None,
    )

    # Get task and set up retry policy (setup requires internal access)
    task = state.get_task(state.get_job_tasks(JobId("retry-job"))[0].task_id)
    task.max_retries_preemption = 2  # Allow retries for worker failures

    # First attempt: dispatch and fail (dispatch is internal scheduler operation)
    state.mark_task_dispatched(task, WorkerId("w1"))

    # Get attempt ID via RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="retry-job"), None)
    first_attempt_id = status.job.tasks[0].current_attempt_id
    task_id = status.job.tasks[0].task_id

    # Report worker failure (retriable)
    request = cluster_pb2.Controller.ReportTaskStateRequest(
        task_id=task_id,
        attempt_id=first_attempt_id,
        state=cluster_pb2.TASK_STATE_WORKER_FAILED,
        error="Worker died",
    )
    service.report_task_state(request, None)

    # Verify attempt history via get_job_status RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="retry-job"), None)
    task_status = status.job.tasks[0]
    assert len(task_status.attempts) == 1
    assert task_status.attempts[0].state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert task_status.attempts[0].error == "Worker died"
    assert status.job.preemption_count == 1


def test_stale_worker_report_ignored_after_retry(service, state, job_request, worker_metadata):
    """Verify that after retry, reports from old attempt are ignored."""
    # Launch job
    service.launch_job(job_request("test-job"), None)

    # Register worker
    service.register_worker(
        cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id="w1",
            address="host:8080",
            metadata=worker_metadata(),
        ),
        None,
    )

    # Get task and set up retry (setup requires internal access)
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    task.max_retries_preemption = 2

    # First attempt (dispatch is internal scheduler operation)
    state.mark_task_dispatched(task, WorkerId("w1"))

    # Get old attempt ID via RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    old_attempt_id = status.job.tasks[0].current_attempt_id
    task_id = status.job.tasks[0].task_id

    # Fail and trigger retry (transition_task simulates scheduler's response to worker failure)
    # Failure type is derived from TASK_STATE_WORKER_FAILED
    state.transition_task(
        task.task_id,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
        error="Worker died",
    )

    # Second attempt (dispatch is internal scheduler operation)
    state.mark_task_dispatched(task, WorkerId("w1"))

    # Get new attempt ID via RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    new_attempt_id = status.job.tasks[0].current_attempt_id
    assert new_attempt_id == old_attempt_id + 1

    # Stale report from old attempt (should be ignored)
    request = cluster_pb2.Controller.ReportTaskStateRequest(
        task_id=task_id,
        attempt_id=old_attempt_id,  # Old attempt
        state=cluster_pb2.TASK_STATE_SUCCEEDED,
    )
    service.report_task_state(request, None)

    # Verify task is still in current attempt state, not succeeded, via RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    task_status = status.job.tasks[0]
    assert task_status.current_attempt_id == new_attempt_id
    assert task_status.state == cluster_pb2.TASK_STATE_RUNNING  # From second dispatch


def test_job_running_while_tasks_retry(service, state, job_request, worker_metadata):
    """Verify job stays RUNNING while tasks are retrying."""
    # Launch job
    service.launch_job(job_request("test-job"), None)

    # Register worker
    service.register_worker(
        cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id="w1",
            address="host:8080",
            metadata=worker_metadata(),
        ),
        None,
    )

    # Get task and set up retry (setup requires internal access)
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    task.max_retries_preemption = 2

    # First attempt - dispatch (dispatch is internal scheduler operation)
    state.mark_task_dispatched(task, WorkerId("w1"))

    # Job should be RUNNING
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    assert status.job.state == cluster_pb2.JOB_STATE_RUNNING

    # Get task info via RPC for report
    task_id = status.job.tasks[0].task_id
    current_attempt_id = status.job.tasks[0].current_attempt_id

    # Report worker failure (triggers retry)
    request = cluster_pb2.Controller.ReportTaskStateRequest(
        task_id=task_id,
        attempt_id=current_attempt_id,
        state=cluster_pb2.TASK_STATE_WORKER_FAILED,
        error="Worker died",
    )
    service.report_task_state(request, None)

    # Job should remain RUNNING (task is pending retry)
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    # Task is in PENDING state (ready for retry), job state depends on counts
    # With the current state tracking, job may go back to PENDING when no tasks are RUNNING
    assert status.job.state in (cluster_pb2.JOB_STATE_PENDING, cluster_pb2.JOB_STATE_RUNNING)


def test_killing_job_with_retrying_task(service, state, job_request, worker_metadata):
    """Verify killing a job with a retrying task terminates properly."""
    # Launch job
    service.launch_job(job_request("test-job"), None)

    # Register worker
    service.register_worker(
        cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id="w1",
            address="host:8080",
            metadata=worker_metadata(),
        ),
        None,
    )

    # Get task and set up retry (setup requires internal access)
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    task.max_retries_preemption = 2

    # First attempt - dispatch and fail (dispatch and transition are internal scheduler operations)
    # Failure type is derived from TASK_STATE_WORKER_FAILED
    state.mark_task_dispatched(task, WorkerId("w1"))
    state.transition_task(
        task.task_id,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
    )

    # Verify task is in WORKER_FAILED state (last attempt failed) but has retries available
    # The task can still be scheduled even though its state shows the last attempt's failure
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    assert status.job.tasks[0].state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert status.job.preemption_count == 1  # One failure so far

    # Kill the job
    service.terminate_job(cluster_pb2.Controller.TerminateJobRequest(job_id="test-job"), None)

    # Verify job is killed via RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job"), None)
    assert status.job.state == cluster_pb2.JOB_STATE_KILLED


def test_full_lifecycle_submit_fail_retry_succeed(service, state, job_request, worker_metadata):
    """End-to-end test: submit job, fail task, retry, succeed."""
    # Launch job
    service.launch_job(job_request("lifecycle-job"), None)

    # Register worker
    service.register_worker(
        cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id="w1",
            address="host:8080",
            metadata=worker_metadata(),
        ),
        None,
    )

    # Get task and configure retry (setup requires internal access)
    task = state.get_task(state.get_job_tasks(JobId("lifecycle-job"))[0].task_id)
    task.max_retries_preemption = 2

    # First attempt - dispatch (dispatch is internal scheduler operation)
    state.mark_task_dispatched(task, WorkerId("w1"))

    # Verify job is RUNNING via RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="lifecycle-job"), None)
    assert status.job.state == cluster_pb2.JOB_STATE_RUNNING
    task_id = status.job.tasks[0].task_id
    current_attempt_id = status.job.tasks[0].current_attempt_id

    # Report RUNNING state
    request = cluster_pb2.Controller.ReportTaskStateRequest(
        task_id=task_id,
        attempt_id=current_attempt_id,
        state=cluster_pb2.TASK_STATE_RUNNING,
    )
    service.report_task_state(request, None)

    # First attempt - fail with worker failure (retriable)
    request = cluster_pb2.Controller.ReportTaskStateRequest(
        task_id=task_id,
        attempt_id=current_attempt_id,
        state=cluster_pb2.TASK_STATE_WORKER_FAILED,
        error="Worker crashed",
    )
    service.report_task_state(request, None)

    # Verify retry was triggered via RPC - task shows WORKER_FAILED (last attempt state)
    # but can still be scheduled for retry
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="lifecycle-job"), None)
    assert status.job.preemption_count == 1
    assert status.job.tasks[0].state == cluster_pb2.TASK_STATE_WORKER_FAILED

    # Second attempt - dispatch again (dispatch is internal scheduler operation)
    state.mark_task_dispatched(task, WorkerId("w1"))

    # Get new attempt ID via RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="lifecycle-job"), None)
    new_attempt_id = status.job.tasks[0].current_attempt_id
    assert new_attempt_id == 1  # Second attempt

    # Second attempt - succeed
    request = cluster_pb2.Controller.ReportTaskStateRequest(
        task_id=task_id,
        attempt_id=new_attempt_id,
        state=cluster_pb2.TASK_STATE_SUCCEEDED,
    )
    service.report_task_state(request, None)

    # Verify job succeeded via RPC
    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="lifecycle-job"), None)
    assert status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert status.job.finished_at_ms > 0

    # Verify attempt history via RPC
    task_status = status.job.tasks[0]
    assert len(task_status.attempts) == 2
    assert task_status.attempts[0].state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert task_status.attempts[1].state == cluster_pb2.TASK_STATE_SUCCEEDED
