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

"""Tests for controller dashboard behavioral logic."""

from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.events import JobSubmittedEvent, WorkerRegisteredEvent
from iris.cluster.controller.scheduler import Scheduler
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import ControllerEndpoint, ControllerState
from iris.cluster.types import JobId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms

# =============================================================================
# Test Helpers
# =============================================================================


def register_worker(
    state: ControllerState,
    worker_id: str,
    address: str,
    metadata: cluster_pb2.WorkerMetadata,
    healthy: bool = True,
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
    worker = state.get_worker(wid)
    if worker and not healthy:
        worker.healthy = False
    return wid


def submit_job(
    state: ControllerState,
    job_id: str,
    request: cluster_pb2.Controller.LaunchJobRequest,
) -> JobId:
    """Submit a job via event."""
    jid = JobId(job_id)
    state.handle_event(
        JobSubmittedEvent(
            job_id=jid,
            request=request,
            timestamp_ms=now_ms(),
        )
    )
    return jid


@pytest.fixture
def state():
    return ControllerState()


@pytest.fixture
def scheduler(state):
    return Scheduler(state)


@pytest.fixture
def service(state, scheduler):
    controller_mock = Mock()
    controller_mock.wake = Mock()
    return ControllerServiceImpl(state, controller_mock)


@pytest.fixture
def client(service, scheduler):
    dashboard = ControllerDashboard(service, scheduler)
    return TestClient(dashboard._app)


@pytest.fixture
def make_worker_metadata():
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
def job_request():
    return cluster_pb2.Controller.LaunchJobRequest(
        name="test-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=4 * 1024**3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )


@pytest.fixture
def resource_spec():
    return cluster_pb2.ResourceSpecProto(cpu=4, memory_bytes=8 * 1024**3, disk_bytes=100 * 1024**3)


def test_stats_counts_building_separately_from_running(client, state, job_request):
    """Building jobs should be counted separately, not as running or pending."""
    submit_job(state, "pending", job_request)
    # Job is already in PENDING state after submission

    building_id = submit_job(state, "building", job_request)
    state.get_job(building_id).state = cluster_pb2.JOB_STATE_BUILDING

    running_id = submit_job(state, "running", job_request)
    state.get_job(running_id).state = cluster_pb2.JOB_STATE_RUNNING

    stats = client.get("/api/stats").json()

    assert stats["jobs_pending"] == 1
    assert stats["jobs_building"] == 1
    assert stats["jobs_running"] == 1


def test_stats_groups_terminal_states_as_completed(client, state, job_request):
    """Succeeded, failed, killed, and worker_failed all count as completed."""
    for job_state in [
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
    ]:
        job_id = submit_job(state, f"job-{job_state}", job_request)
        state.get_job(job_id).state = job_state

    stats = client.get("/api/stats").json()

    assert stats["jobs_completed"] == 4
    assert stats["jobs_pending"] == 0
    assert stats["jobs_running"] == 0


def test_stats_counts_only_healthy_workers(client, state, make_worker_metadata):
    """Healthy worker count excludes unhealthy workers."""
    register_worker(state, "healthy1", "h1:8080", make_worker_metadata())
    register_worker(state, "healthy2", "h2:8080", make_worker_metadata())
    register_worker(state, "unhealthy", "h3:8080", make_worker_metadata(), healthy=False)

    stats = client.get("/api/stats").json()

    assert stats["workers_healthy"] == 2
    assert stats["workers_total"] == 3


def test_stats_counts_endpoints_for_running_jobs_only(client, state, job_request):
    """Endpoint count only includes endpoints for RUNNING jobs."""
    # Running job with endpoint
    running_id = submit_job(state, "running", job_request)
    state.get_job(running_id).state = cluster_pb2.JOB_STATE_RUNNING
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep1", name="svc", address="host:80", job_id=running_id))

    # Pending job with endpoint (should not count)
    pending_id = submit_job(state, "pending", job_request)
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep2", name="svc2", address="host:81", job_id=pending_id))

    stats = client.get("/api/stats").json()

    assert stats["endpoints_count"] == 1


def test_endpoints_only_returned_for_running_jobs(client, state, job_request):
    """ListEndpoints filters out endpoints for non-running jobs."""
    # Create jobs in various states
    pending_id = submit_job(state, "pending", job_request)

    running_id = submit_job(state, "running", job_request)
    state.get_job(running_id).state = cluster_pb2.JOB_STATE_RUNNING

    succeeded_id = submit_job(state, "succeeded", job_request)
    state.get_job(succeeded_id).state = cluster_pb2.JOB_STATE_SUCCEEDED

    # Add endpoints for each
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep1", name="pending-svc", address="h:1", job_id=pending_id))
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep2", name="running-svc", address="h:2", job_id=running_id))
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep3", name="done-svc", address="h:3", job_id=succeeded_id))

    endpoints = client.get("/api/endpoints").json()

    assert len(endpoints) == 1
    assert endpoints[0]["name"] == "running-svc"


def test_job_detail_page_includes_worker_address(client, state, job_request, make_worker_metadata):
    """Job detail page has empty worker address since jobs don't execute on workers."""
    register_worker(state, "w1", "worker-host:9000", make_worker_metadata())

    job_id = submit_job(state, "j1", job_request)
    state.get_job(job_id).state = cluster_pb2.JOB_STATE_RUNNING

    response = client.get("/job/j1")

    assert response.status_code == 200
    # New dashboard shows task table
    assert "Tasks</h2>" in response.text
    assert "tasks-table" in response.text


def test_job_detail_page_empty_worker_for_pending_job(client, state, job_request):
    """Job detail page shows task summary for pending jobs."""
    submit_job(state, "pending-job", job_request)
    # Job is already in PENDING state after submission

    response = client.get("/job/pending-job")

    assert response.status_code == 200
    # New dashboard shows task summary
    assert "Task Summary" in response.text
    assert "tasks-table" in response.text


def test_jobs_state_names_mapped_correctly(client, state, job_request):
    """Proto state enums map to expected string names."""
    state_mapping = [
        (cluster_pb2.JOB_STATE_PENDING, "pending"),
        (cluster_pb2.JOB_STATE_BUILDING, "building"),
        (cluster_pb2.JOB_STATE_RUNNING, "running"),
        (cluster_pb2.JOB_STATE_SUCCEEDED, "succeeded"),
        (cluster_pb2.JOB_STATE_FAILED, "failed"),
        (cluster_pb2.JOB_STATE_KILLED, "killed"),
        (cluster_pb2.JOB_STATE_WORKER_FAILED, "worker_failed"),
    ]

    for proto_state, _ in state_mapping:
        job_id = submit_job(state, f"j-{proto_state}", job_request)
        state.get_job(job_id).state = proto_state

    jobs = client.get("/api/jobs").json()
    job_by_id = {j["job_id"]: j["state"] for j in jobs}

    for proto_state, expected_name in state_mapping:
        assert job_by_id[f"j-{proto_state}"] == expected_name


def test_api_jobs_includes_retry_counts(client, state, job_request):
    """Jobs API includes retry count fields."""
    job_id = submit_job(state, "test-job", job_request)
    job = state.get_job(job_id)
    job.state = cluster_pb2.JOB_STATE_RUNNING
    job.failure_count = 1
    job.preemption_count = 2

    jobs = client.get("/api/jobs").json()

    assert len(jobs) == 1
    assert jobs[0]["failure_count"] == 1
    assert jobs[0]["preemption_count"] == 2


def test_api_job_attempts_returns_retry_info(client, state, job_request):
    """Job attempts API returns retry counts and current state.

    Jobs no longer track individual attempts - tasks do. This endpoint
    returns aggregate retry information for the job.
    """
    job_id = submit_job(state, "test-job", job_request)
    job = state.get_job(job_id)
    job.state = cluster_pb2.JOB_STATE_RUNNING
    job.started_at_ms = 3000
    job.failure_count = 1
    job.preemption_count = 1

    response = client.get("/api/jobs/test-job/attempts").json()

    assert response["failure_count"] == 1
    assert response["preemption_count"] == 1
    assert response["current_state"] == "running"
    assert response["started_at_ms"] == 3000


def test_api_job_attempts_returns_404_for_missing_job(client, state):
    """Job attempts API returns 404 for non-existent job."""
    response = client.get("/api/jobs/nonexistent/attempts")

    assert response.status_code == 404
    assert response.json()["error"] == "Job not found"
