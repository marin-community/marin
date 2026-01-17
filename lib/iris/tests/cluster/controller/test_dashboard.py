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

from iris.rpc import cluster_pb2
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import (
    ControllerEndpoint,
    ControllerJob,
    ControllerState,
    ControllerWorker,
)
from iris.cluster.types import JobId, WorkerId


@pytest.fixture
def state():
    return ControllerState()


@pytest.fixture
def service(state):
    scheduler = Mock()
    scheduler.wake = Mock()
    return ControllerServiceImpl(state, scheduler)


@pytest.fixture
def client(service):
    dashboard = ControllerDashboard(service)
    return TestClient(dashboard._app)


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
    state.add_job(ControllerJob(job_id=JobId("pending"), request=job_request, state=cluster_pb2.JOB_STATE_PENDING))
    state.add_job(ControllerJob(job_id=JobId("building"), request=job_request, state=cluster_pb2.JOB_STATE_BUILDING))
    state.add_job(ControllerJob(job_id=JobId("running"), request=job_request, state=cluster_pb2.JOB_STATE_RUNNING))

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
        state.add_job(ControllerJob(job_id=JobId(f"job-{job_state}"), request=job_request, state=job_state))

    stats = client.get("/api/stats").json()

    assert stats["jobs_completed"] == 4
    assert stats["jobs_pending"] == 0
    assert stats["jobs_running"] == 0


def test_stats_counts_only_healthy_workers(client, state, resource_spec):
    """Healthy worker count excludes unhealthy workers."""
    state.add_worker(
        ControllerWorker(worker_id=WorkerId("healthy1"), address="h1:8080", resources=resource_spec, healthy=True)
    )
    state.add_worker(
        ControllerWorker(worker_id=WorkerId("healthy2"), address="h2:8080", resources=resource_spec, healthy=True)
    )
    state.add_worker(
        ControllerWorker(worker_id=WorkerId("unhealthy"), address="h3:8080", resources=resource_spec, healthy=False)
    )

    stats = client.get("/api/stats").json()

    assert stats["workers_healthy"] == 2
    assert stats["workers_total"] == 3


def test_stats_counts_endpoints_for_running_jobs_only(client, state, job_request):
    """Endpoint count only includes endpoints for RUNNING jobs."""
    # Running job with endpoint
    state.add_job(ControllerJob(job_id=JobId("running"), request=job_request, state=cluster_pb2.JOB_STATE_RUNNING))
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep1", name="svc", address="host:80", job_id=JobId("running")))

    # Pending job with endpoint (should not count)
    state.add_job(ControllerJob(job_id=JobId("pending"), request=job_request, state=cluster_pb2.JOB_STATE_PENDING))
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep2", name="svc2", address="host:81", job_id=JobId("pending")))

    stats = client.get("/api/stats").json()

    assert stats["endpoints_count"] == 1


def test_endpoints_only_returned_for_running_jobs(client, state, job_request):
    """ListEndpoints filters out endpoints for non-running jobs."""
    # Create jobs in various states
    state.add_job(ControllerJob(job_id=JobId("pending"), request=job_request, state=cluster_pb2.JOB_STATE_PENDING))
    state.add_job(ControllerJob(job_id=JobId("running"), request=job_request, state=cluster_pb2.JOB_STATE_RUNNING))
    state.add_job(ControllerJob(job_id=JobId("succeeded"), request=job_request, state=cluster_pb2.JOB_STATE_SUCCEEDED))

    # Add endpoints for each
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep1", name="pending-svc", address="h:1", job_id=JobId("pending")))
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep2", name="running-svc", address="h:2", job_id=JobId("running")))
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep3", name="done-svc", address="h:3", job_id=JobId("succeeded")))

    endpoints = client.get("/api/endpoints").json()

    assert len(endpoints) == 1
    assert endpoints[0]["name"] == "running-svc"


def test_job_detail_page_includes_worker_address(client, state, job_request, resource_spec):
    """Job detail page injects worker address for client-side fetch."""
    state.add_worker(
        ControllerWorker(worker_id=WorkerId("w1"), address="worker-host:9000", resources=resource_spec, healthy=True)
    )
    state.add_job(
        ControllerJob(
            job_id=JobId("j1"), request=job_request, state=cluster_pb2.JOB_STATE_RUNNING, worker_id=WorkerId("w1")
        )
    )

    response = client.get("/job/j1")

    assert response.status_code == 200
    assert "worker-host:9000" in response.text


def test_job_detail_page_empty_worker_for_pending_job(client, state, job_request):
    """Job detail page has empty worker address for unassigned jobs."""
    state.add_job(ControllerJob(job_id=JobId("pending-job"), request=job_request, state=cluster_pb2.JOB_STATE_PENDING))

    response = client.get("/job/pending-job")

    assert response.status_code == 200
    # Worker address placeholder should be empty
    assert "const workerAddress = '';" in response.text


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
        state.add_job(ControllerJob(job_id=JobId(f"j-{proto_state}"), request=job_request, state=proto_state))

    jobs = client.get("/api/jobs").json()
    job_by_id = {j["job_id"]: j["state"] for j in jobs}

    for proto_state, expected_name in state_mapping:
        assert job_by_id[f"j-{proto_state}"] == expected_name


def test_api_jobs_includes_attempt_info(client, state, job_request):
    """Jobs API includes attempt tracking fields."""
    from iris.cluster.controller.job import JobAttempt

    job = ControllerJob(
        job_id=JobId("test-job"),
        request=job_request,
        state=cluster_pb2.JOB_STATE_RUNNING,
        current_attempt_id=2,
        failure_count=1,
        preemption_count=0,
        attempts=[
            JobAttempt(attempt_id=0, state=cluster_pb2.JOB_STATE_FAILED),
            JobAttempt(attempt_id=1, state=cluster_pb2.JOB_STATE_WORKER_FAILED),
        ],
    )
    state.add_job(job)

    jobs = client.get("/api/jobs").json()

    assert len(jobs) == 1
    assert jobs[0]["current_attempt_id"] == 2
    assert jobs[0]["total_attempts"] == 3
    assert jobs[0]["failure_count"] == 1
    assert jobs[0]["preemption_count"] == 0


def test_api_job_attempts_returns_attempt_history(client, state, job_request):
    """Job attempts API returns historical attempts plus current."""
    from iris.cluster.controller.job import JobAttempt

    job = ControllerJob(
        job_id=JobId("test-job"),
        request=job_request,
        state=cluster_pb2.JOB_STATE_RUNNING,
        current_attempt_id=2,
        worker_id=WorkerId("worker2"),
        started_at_ms=3000,
        attempts=[
            JobAttempt(
                attempt_id=0,
                worker_id=WorkerId("worker1"),
                state=cluster_pb2.JOB_STATE_FAILED,
                started_at_ms=1000,
                finished_at_ms=2000,
                exit_code=1,
                error="Task failed",
                is_worker_failure=False,
            ),
            JobAttempt(
                attempt_id=1,
                worker_id=WorkerId("worker1"),
                state=cluster_pb2.JOB_STATE_WORKER_FAILED,
                started_at_ms=2100,
                finished_at_ms=2500,
                error="Worker died",
                is_worker_failure=True,
            ),
        ],
    )
    state.add_job(job)

    attempts = client.get("/api/jobs/test-job/attempts").json()

    assert len(attempts) == 3

    # First attempt
    assert attempts[0]["attempt_id"] == 0
    assert attempts[0]["worker_id"] == "worker1"
    assert attempts[0]["state"] == "failed"
    assert attempts[0]["exit_code"] == 1
    assert attempts[0]["error"] == "Task failed"
    assert attempts[0]["is_worker_failure"] is False

    # Second attempt
    assert attempts[1]["attempt_id"] == 1
    assert attempts[1]["worker_id"] == "worker1"
    assert attempts[1]["state"] == "worker_failed"
    assert attempts[1]["error"] == "Worker died"
    assert attempts[1]["is_worker_failure"] is True

    # Current attempt
    assert attempts[2]["attempt_id"] == 2
    assert attempts[2]["worker_id"] == "worker2"
    assert attempts[2]["state"] == "running"
    assert attempts[2]["is_worker_failure"] is False


def test_api_job_attempts_returns_404_for_missing_job(client, state):
    """Job attempts API returns 404 for non-existent job."""
    response = client.get("/api/jobs/nonexistent/attempts")

    assert response.status_code == 404
    assert response.json()["error"] == "Job not found"
