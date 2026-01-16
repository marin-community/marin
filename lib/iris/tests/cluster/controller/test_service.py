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

from iris.rpc import cluster_pb2
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import ControllerJob, ControllerState
from iris.cluster.types import JobId, WorkerId, create_resource_spec


@pytest.fixture
def make_job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=create_resource_spec(cpu=1, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


@pytest.fixture
def make_resource_spec_fixture():
    """Create a minimal ResourceSpec for testing."""

    def _make() -> cluster_pb2.ResourceSpec:
        return create_resource_spec(cpu=1, memory="1g", disk="10g")

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


def test_launch_job_wakes_scheduler(service, mock_scheduler, make_job_request):
    """Verify launch_job wakes the scheduler."""
    request = make_job_request("test-job")
    service.launch_job(request, None)

    # Should have called wake() once
    mock_scheduler.wake.assert_called_once()


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
    request = cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job-id")
    response = service.get_job_status(request, None)

    # Verify response
    assert response.job.job_id == "test-job-id"
    assert response.job.state == cluster_pb2.JOB_STATE_RUNNING
    assert response.job.started_at_ms == 12350
    assert response.job.worker_id == "worker-1"


def test_get_job_status_not_found(service):
    """Verify get_job_status raises ConnectError for unknown job."""
    request = cluster_pb2.Controller.GetJobStatusRequest(job_id="nonexistent")

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
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="test-job-id")
    response = service.terminate_job(request, None)

    # Should return empty response
    assert isinstance(response, cluster_pb2.Empty)

    # Job should be marked KILLED
    assert job.state == cluster_pb2.JOB_STATE_KILLED
    assert job.finished_at_ms is not None
    assert job.finished_at_ms > job.started_at_ms


def test_terminate_job_not_found(service):
    """Verify terminate_job raises ConnectError for unknown job."""
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="nonexistent")

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
    request = cluster_pb2.Controller.ListJobsRequest()
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
    request = cluster_pb2.Controller.GetJobStatusRequest(job_id="test-job-id")
    response = service.get_job_status(request, None)

    # Verify all fields
    assert response.job.job_id == "test-job-id"
    assert response.job.state == cluster_pb2.JOB_STATE_FAILED
    assert response.job.started_at_ms == 12350
    assert response.job.finished_at_ms == 12400
    assert response.job.worker_id == "worker-1"
    assert response.job.error == "Something went wrong"
    assert response.job.exit_code == 42


def test_launch_job_uses_name_as_job_id(service, make_job_request):
    """Verify job_id is the same as the provided name."""
    request = make_job_request("my-unique-job")

    response = service.launch_job(request, None)

    # job_id should be the name
    assert response.job_id == "my-unique-job"


def test_launch_job_rejects_duplicate_name(service, make_job_request):
    """Verify launch_job rejects duplicate job names."""
    request = make_job_request("duplicate-job")

    # First launch should succeed
    response = service.launch_job(request, None)
    assert response.job_id == "duplicate-job"

    # Second launch with same name should fail
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "duplicate-job" in exc_info.value.message


def test_launch_job_rejects_empty_name(service, state):
    """Verify launch_job rejects empty job names."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="",  # Empty name
        serialized_entrypoint=b"test",
        resources=create_resource_spec(cpu=1, memory="1g"),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.INVALID_ARGUMENT
    assert "name is required" in exc_info.value.message.lower()


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
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="test-job-id")
    service.terminate_job(request, None)

    # Job should be marked KILLED even though it was never running
    assert job.state == cluster_pb2.JOB_STATE_KILLED
    assert job.finished_at_ms is not None


def test_register_worker(service, state, make_resource_spec_fixture):
    """Verify register_worker adds worker to state."""
    request = cluster_pb2.Controller.RegisterWorkerRequest(
        worker_id="w1",
        address="host1:8080",
        resources=make_resource_spec_fixture(),
    )

    response = service.register_worker(request, None)

    assert response.accepted is True
    worker = state.get_worker(WorkerId("w1"))
    assert worker is not None
    assert worker.address == "host1:8080"
    assert worker.healthy is True


def test_register_worker_logs_action(service, state, make_resource_spec_fixture):
    """Verify register_worker logs an action."""
    request = cluster_pb2.Controller.RegisterWorkerRequest(
        worker_id="w1",
        address="host1:8080",
        resources=make_resource_spec_fixture(),
    )

    service.register_worker(request, None)

    actions = state.get_recent_actions()
    assert len(actions) == 1
    assert actions[0].action == "worker_registered"
    assert actions[0].worker_id == "w1"


def test_list_workers_returns_all(service, state, make_resource_spec_fixture):
    """Verify list_workers returns all workers."""
    from iris.cluster.controller.state import ControllerWorker

    # Add multiple workers
    for i in range(3):
        worker = ControllerWorker(
            worker_id=WorkerId(f"w{i}"),
            address=f"host{i}:8080",
            resources=make_resource_spec_fixture(),
            healthy=(i != 1),  # w1 is unhealthy
        )
        state.add_worker(worker)

    request = cluster_pb2.Controller.ListWorkersRequest()
    response = service.list_workers(request, None)

    assert len(response.workers) == 3
    worker_ids = {w.worker_id for w in response.workers}
    assert worker_ids == {"w0", "w1", "w2"}

    # Check healthy status
    workers_by_id = {w.worker_id: w for w in response.workers}
    assert workers_by_id["w0"].healthy is True
    assert workers_by_id["w1"].healthy is False
    assert workers_by_id["w2"].healthy is True


def test_launch_job_logs_action(service, state, make_job_request):
    """Verify launch_job logs an action."""
    request = make_job_request("test-job")
    response = service.launch_job(request, None)

    actions = state.get_recent_actions()
    assert len(actions) == 1
    assert actions[0].action == "job_submitted"
    assert actions[0].job_id == response.job_id
    assert actions[0].details == "test-job"


def test_terminate_job_logs_action(service, state, make_job_request):
    """Verify terminate_job logs an action."""
    # Add a running job
    job = ControllerJob(
        job_id=JobId("test-job-id"),
        request=make_job_request("test-job"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        submitted_at_ms=12345,
    )
    state.add_job(job)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id="test-job-id")
    service.terminate_job(request, None)

    actions = state.get_recent_actions()
    assert len(actions) == 1
    assert actions[0].action == "job_killed"
    assert actions[0].job_id == "test-job-id"


# =============================================================================
# Hierarchical Job Tests
# =============================================================================


def test_launch_job_with_parent_job_id(service, state):
    """Verify launch_job stores parent_job_id from request."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="child-job",
        serialized_entrypoint=b"test",
        resources=create_resource_spec(cpu=1, memory="1g"),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        parent_job_id="parent-123",
    )

    response = service.launch_job(request, None)

    job = state.get_job(JobId(response.job_id))
    assert job is not None
    assert job.parent_job_id == "parent-123"


def test_launch_job_without_parent_job_id(service, state):
    """Verify launch_job sets parent_job_id to None when not provided."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="root-job",
        serialized_entrypoint=b"test",
        resources=create_resource_spec(cpu=1, memory="1g"),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )

    response = service.launch_job(request, None)

    job = state.get_job(JobId(response.job_id))
    assert job is not None
    assert job.parent_job_id is None


def test_get_job_status_includes_parent_job_id(service, state, make_job_request):
    """Verify get_job_status returns parent_job_id in response."""
    # Add a job with a parent
    job = ControllerJob(
        job_id=JobId("child-job"),
        request=make_job_request("child"),
        parent_job_id=JobId("parent-job"),
    )
    state.add_job(job)

    request = cluster_pb2.Controller.GetJobStatusRequest(job_id="child-job")
    response = service.get_job_status(request, None)

    assert response.job.parent_job_id == "parent-job"


def test_terminate_job_cascades_to_children(service, state, make_job_request):
    """Verify terminate_job terminates all children when parent is terminated."""
    # Create parent and child jobs
    parent = ControllerJob(
        job_id=JobId("parent"),
        request=make_job_request("parent"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    child1 = ControllerJob(
        job_id=JobId("child1"),
        request=make_job_request("child1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        parent_job_id=JobId("parent"),
    )
    child2 = ControllerJob(
        job_id=JobId("child2"),
        request=make_job_request("child2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        parent_job_id=JobId("parent"),
    )
    state.add_job(parent)
    state.add_job(child1)
    state.add_job(child2)

    # Terminate parent
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="parent")
    service.terminate_job(request, None)

    # All jobs should be killed
    assert parent.state == cluster_pb2.JOB_STATE_KILLED
    assert child1.state == cluster_pb2.JOB_STATE_KILLED
    assert child2.state == cluster_pb2.JOB_STATE_KILLED


def test_terminate_job_cascades_depth_first(service, state, make_job_request):
    """Verify terminate_job terminates children before parent (depth-first)."""
    # Create a 3-level hierarchy: grandparent -> parent -> child
    grandparent = ControllerJob(
        job_id=JobId("grandparent"),
        request=make_job_request("grandparent"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    parent = ControllerJob(
        job_id=JobId("parent"),
        request=make_job_request("parent"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        parent_job_id=JobId("grandparent"),
    )
    child = ControllerJob(
        job_id=JobId("child"),
        request=make_job_request("child"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        parent_job_id=JobId("parent"),
    )
    state.add_job(grandparent)
    state.add_job(parent)
    state.add_job(child)

    # Terminate grandparent
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="grandparent")
    service.terminate_job(request, None)

    # All should be killed
    assert grandparent.state == cluster_pb2.JOB_STATE_KILLED
    assert parent.state == cluster_pb2.JOB_STATE_KILLED
    assert child.state == cluster_pb2.JOB_STATE_KILLED

    # Verify termination order via action log (children killed before parents)
    actions = state.get_recent_actions()
    killed_order = [a.job_id for a in actions if a.action == "job_killed"]
    # Child should be killed first, then parent, then grandparent
    assert killed_order == ["child", "parent", "grandparent"]


def test_terminate_job_only_affects_descendants(service, state, make_job_request):
    """Verify terminate_job does not affect sibling jobs."""
    # Create two sibling jobs under the same parent
    parent = ControllerJob(
        job_id=JobId("parent"),
        request=make_job_request("parent"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    child1 = ControllerJob(
        job_id=JobId("child1"),
        request=make_job_request("child1"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        parent_job_id=JobId("parent"),
    )
    child2 = ControllerJob(
        job_id=JobId("child2"),
        request=make_job_request("child2"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        parent_job_id=JobId("parent"),
    )
    state.add_job(parent)
    state.add_job(child1)
    state.add_job(child2)

    # Terminate only child1
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="child1")
    service.terminate_job(request, None)

    # Only child1 should be killed
    assert child1.state == cluster_pb2.JOB_STATE_KILLED
    assert child2.state == cluster_pb2.JOB_STATE_RUNNING
    assert parent.state == cluster_pb2.JOB_STATE_RUNNING


def test_terminate_job_skips_already_finished_children(service, state, make_job_request):
    """Verify terminate_job skips children already in terminal state."""
    parent = ControllerJob(
        job_id=JobId("parent"),
        request=make_job_request("parent"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    # Child already succeeded
    child_succeeded = ControllerJob(
        job_id=JobId("child-succeeded"),
        request=make_job_request("child-succeeded"),
        state=cluster_pb2.JOB_STATE_SUCCEEDED,
        parent_job_id=JobId("parent"),
        finished_at_ms=12345,
    )
    # Child still running
    child_running = ControllerJob(
        job_id=JobId("child-running"),
        request=make_job_request("child-running"),
        state=cluster_pb2.JOB_STATE_RUNNING,
        parent_job_id=JobId("parent"),
    )
    state.add_job(parent)
    state.add_job(child_succeeded)
    state.add_job(child_running)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id="parent")
    service.terminate_job(request, None)

    # Succeeded child should remain in SUCCEEDED state
    assert child_succeeded.state == cluster_pb2.JOB_STATE_SUCCEEDED
    # Running child should be killed
    assert child_running.state == cluster_pb2.JOB_STATE_KILLED
    # Parent should be killed
    assert parent.state == cluster_pb2.JOB_STATE_KILLED
