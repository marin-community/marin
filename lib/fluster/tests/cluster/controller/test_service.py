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

"""Tests for controller RPC service implementation."""

from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from fluster import cluster_pb2
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerJob, ControllerState
from fluster.cluster.types import JobId


@pytest.fixture
def make_job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.LaunchJobRequest:
        return cluster_pb2.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


@pytest.fixture
def state():
    """Create a fresh ControllerState for each test."""
    return ControllerState()


@pytest.fixture
def scheduler(state):
    """Create a mock scheduler."""
    # Use a mock dispatch function that always succeeds
    mock_dispatch = Mock(return_value=True)
    return Scheduler(state, mock_dispatch)


@pytest.fixture
def service(state, scheduler):
    """Create a ControllerServiceImpl for testing."""
    return ControllerServiceImpl(state, scheduler)


def test_launch_job_returns_job_id(service, make_job_request):
    """Verify launch_job returns a job_id and adds job to state."""
    request = make_job_request("test-job")

    response = service.launch_job(request, None)

    # Should return a job_id
    assert response.job_id
    assert len(response.job_id) > 0

    # Job should be in state
    job = service._state.get_job(JobId(response.job_id))
    assert job is not None
    assert job.state == cluster_pb2.JOB_STATE_PENDING
    assert job.request.name == "test-job"


def test_launch_job_wakes_scheduler(service, scheduler, make_job_request):
    """Verify launch_job wakes the scheduler."""
    # Spy on scheduler.wake()
    scheduler.wake = Mock()

    request = make_job_request("test-job")
    service.launch_job(request, None)

    # Should have called wake() once
    scheduler.wake.assert_called_once()


def test_get_job_status_returns_status(service, state, make_job_request):
    """Verify get_job_status returns status for existing job."""
    # Add a job directly to state
    job = ControllerJob(
        job_id=JobId("test-job-id"),
        request=make_job_request("test-job"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        submitted_at_ms=12345,
        started_at_ms=12350,
    )
    job.worker_id = "worker-1"
    state.add_job(job)

    # Get status
    request = cluster_pb2.GetJobStatusRequest(job_id="test-job-id")
    response = service.get_job_status(request, None)

    # Verify response
    assert response.job.job_id == "test-job-id"
    assert response.job.state == cluster_pb2.JOB_STATE_RUNNING
    assert response.job.started_at_ms == 12350
    assert response.job.worker_id == "worker-1"


def test_get_job_status_not_found(service):
    """Verify get_job_status raises ConnectError for unknown job."""
    request = cluster_pb2.GetJobStatusRequest(job_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.get_job_status(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


def test_terminate_job_marks_as_killed(service, state, make_job_request):
    """Verify terminate_job sets job state to KILLED."""
    # Add a running job
    job = ControllerJob(
        job_id=JobId("test-job-id"),
        request=make_job_request("test-job"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        submitted_at_ms=12345,
        started_at_ms=12350,
    )
    state.add_job(job)

    # Terminate it
    request = cluster_pb2.TerminateJobRequest(job_id="test-job-id")
    response = service.terminate_job(request, None)

    # Should return empty response
    assert isinstance(response, cluster_pb2.Empty)

    # Job should be marked KILLED
    assert job.state == cluster_pb2.JOB_STATE_KILLED
    assert job.finished_at_ms is not None
    assert job.finished_at_ms > job.started_at_ms


def test_terminate_job_not_found(service):
    """Verify terminate_job raises ConnectError for unknown job."""
    request = cluster_pb2.TerminateJobRequest(job_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.terminate_job(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


def test_list_jobs_returns_all_jobs(service, state, make_job_request):
    """Verify list_jobs returns all jobs in state."""
    # Add multiple jobs with different states
    job1 = ControllerJob(
        job_id=JobId("job-1"),
        request=make_job_request("job1"),
        state=cluster_pb2.JOB_STATE_PENDING,
    )
    job2 = ControllerJob(
        job_id=JobId("job-2"),
        request=make_job_request("job2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    job3 = ControllerJob(
        job_id=JobId("job-3"),
        request=make_job_request("job3"),
        state=cluster_pb2.JOB_STATE_SUCCEEDED,
    )
    state.add_job(job1)
    state.add_job(job2)
    state.add_job(job3)

    # List jobs
    request = cluster_pb2.ListJobsRequest()
    response = service.list_jobs(request, None)

    # Should return all jobs
    assert len(response.jobs) == 3
    job_ids = {j.job_id for j in response.jobs}
    assert job_ids == {"job-1", "job-2", "job-3"}

    # Verify states are correct
    states_by_id = {j.job_id: j.state for j in response.jobs}
    assert states_by_id["job-1"] == cluster_pb2.JOB_STATE_PENDING
    assert states_by_id["job-2"] == cluster_pb2.JOB_STATE_RUNNING
    assert states_by_id["job-3"] == cluster_pb2.JOB_STATE_SUCCEEDED


def test_list_jobs_empty(service):
    """Verify list_jobs returns empty list when no jobs exist."""
    request = cluster_pb2.ListJobsRequest()
    response = service.list_jobs(request, None)

    assert len(response.jobs) == 0


def test_get_job_status_includes_all_fields(service, state, make_job_request):
    """Verify get_job_status includes all JobStatus fields."""
    # Add a completed job with all fields populated
    job = ControllerJob(
        job_id=JobId("test-job-id"),
        request=make_job_request("test-job"),
        state=cluster_pb2.JOB_STATE_FAILED,
        submitted_at_ms=12345,
        started_at_ms=12350,
        finished_at_ms=12400,
    )
    job.worker_id = "worker-1"
    job.error = "Something went wrong"
    job.exit_code = 42
    state.add_job(job)

    # Get status
    request = cluster_pb2.GetJobStatusRequest(job_id="test-job-id")
    response = service.get_job_status(request, None)

    # Verify all fields
    assert response.job.job_id == "test-job-id"
    assert response.job.state == cluster_pb2.JOB_STATE_FAILED
    assert response.job.started_at_ms == 12350
    assert response.job.finished_at_ms == 12400
    assert response.job.worker_id == "worker-1"
    assert response.job.error == "Something went wrong"
    assert response.job.exit_code == 42


def test_launch_job_generates_unique_ids(service, make_job_request):
    """Verify each launch_job call generates a unique job_id."""
    request = make_job_request("test-job")

    # Launch multiple jobs
    response1 = service.launch_job(request, None)
    response2 = service.launch_job(request, None)
    response3 = service.launch_job(request, None)

    # All IDs should be unique
    job_ids = {response1.job_id, response2.job_id, response3.job_id}
    assert len(job_ids) == 3


def test_terminate_pending_job(service, state, make_job_request):
    """Verify terminate_job works on pending jobs (not just running)."""
    # Add a pending job
    job = ControllerJob(
        job_id=JobId("test-job-id"),
        request=make_job_request("test-job"),
        state=cluster_pb2.JOB_STATE_PENDING,
        submitted_at_ms=12345,
    )
    state.add_job(job)

    # Terminate it
    request = cluster_pb2.TerminateJobRequest(job_id="test-job-id")
    service.terminate_job(request, None)

    # Job should be marked KILLED even though it was never running
    assert job.state == cluster_pb2.JOB_STATE_KILLED
    assert job.finished_at_ms is not None
