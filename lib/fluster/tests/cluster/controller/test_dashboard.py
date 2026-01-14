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

"""Tests for controller dashboard with Starlette TestClient."""

from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from fluster import cluster_pb2
from fluster.cluster.controller.dashboard import ControllerDashboard
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.types import JobId, WorkerId


@pytest.fixture
def state():
    return ControllerState()


class MockSchedulerWake:
    """Mock object that just tracks wake() calls."""

    def __init__(self):
        self.wake = Mock()


@pytest.fixture
def mock_scheduler():
    return MockSchedulerWake()


@pytest.fixture
def service(state, mock_scheduler):
    return ControllerServiceImpl(state, mock_scheduler)


@pytest.fixture
def dashboard(service):
    return ControllerDashboard(service)


@pytest.fixture
def client(dashboard):
    return TestClient(dashboard._app)


@pytest.fixture
def make_job_request():
    def _make(name: str = "test-job") -> cluster_pb2.LaunchJobRequest:
        return cluster_pb2.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )

    return _make


@pytest.fixture
def make_resource_spec():
    def _make() -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=1, memory="1g", disk="10g")

    return _make


def test_dashboard_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Fluster Controller Dashboard" in response.text


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "workers" in data
    assert "healthy_workers" in data
    assert "jobs" in data


def test_api_stats_empty(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["jobs_pending"] == 0
    assert data["jobs_running"] == 0
    assert data["jobs_completed"] == 0
    assert data["workers_healthy"] == 0
    assert data["workers_total"] == 0


def test_api_stats_with_data(client, state, make_job_request, make_resource_spec):
    # Add workers
    state.add_worker(
        ControllerWorker(
            worker_id=WorkerId("w1"),
            address="host1:8080",
            resources=make_resource_spec(),
            healthy=True,
        )
    )
    state.add_worker(
        ControllerWorker(
            worker_id=WorkerId("w2"),
            address="host2:8080",
            resources=make_resource_spec(),
            healthy=False,
        )
    )

    # Add jobs in various states
    state.add_job(
        ControllerJob(
            job_id=JobId("j1"),
            request=make_job_request("job1"),
            state=cluster_pb2.JOB_STATE_PENDING,
        )
    )
    state.add_job(
        ControllerJob(
            job_id=JobId("j2"),
            request=make_job_request("job2"),
            state=cluster_pb2.JOB_STATE_RUNNING,
        )
    )
    state.add_job(
        ControllerJob(
            job_id=JobId("j3"),
            request=make_job_request("job3"),
            state=cluster_pb2.JOB_STATE_SUCCEEDED,
        )
    )
    state.add_job(
        ControllerJob(
            job_id=JobId("j4"),
            request=make_job_request("job4"),
            state=cluster_pb2.JOB_STATE_FAILED,
        )
    )

    response = client.get("/api/stats")
    data = response.json()
    assert data["jobs_pending"] == 1
    assert data["jobs_running"] == 1
    assert data["jobs_completed"] == 2
    assert data["workers_healthy"] == 1
    assert data["workers_total"] == 2


def test_api_jobs_returns_all_jobs(client, state, make_job_request):
    state.add_job(
        ControllerJob(
            job_id=JobId("j1"),
            request=make_job_request("job1"),
            state=cluster_pb2.JOB_STATE_RUNNING,
            worker_id=WorkerId("w1"),
        )
    )
    state.add_job(
        ControllerJob(
            job_id=JobId("j2"),
            request=make_job_request("job2"),
            state=cluster_pb2.JOB_STATE_FAILED,
            error="Something went wrong",
        )
    )

    response = client.get("/api/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) == 2

    job_by_id = {j["job_id"]: j for j in jobs}
    assert job_by_id["j1"]["name"] == "job1"
    assert job_by_id["j1"]["state"] == "running"
    assert job_by_id["j1"]["worker_id"] == "w1"
    assert job_by_id["j2"]["name"] == "job2"
    assert job_by_id["j2"]["state"] == "failed"
    assert job_by_id["j2"]["error"] == "Something went wrong"


def test_api_workers_returns_all_workers(client, state, make_resource_spec):
    w1 = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host1:8080",
        resources=make_resource_spec(),
        healthy=True,
        last_heartbeat_ms=1000,
    )
    w1.running_jobs.add(JobId("j1"))
    state.add_worker(w1)

    state.add_worker(
        ControllerWorker(
            worker_id=WorkerId("w2"),
            address="host2:8080",
            resources=make_resource_spec(),
            healthy=False,
            consecutive_failures=3,
        )
    )

    response = client.get("/api/workers")
    assert response.status_code == 200
    workers = response.json()
    assert len(workers) == 2

    worker_by_id = {w["worker_id"]: w for w in workers}
    assert worker_by_id["w1"]["address"] == "host1:8080"
    assert worker_by_id["w1"]["healthy"] is True
    assert worker_by_id["w1"]["running_jobs"] == 1
    assert worker_by_id["w2"]["healthy"] is False
    assert worker_by_id["w2"]["consecutive_failures"] == 3


def test_api_actions_returns_log(client, state):
    state.log_action("job_submitted", job_id=JobId("j1"), details="test job")
    state.log_action("worker_registered", worker_id=WorkerId("w1"))
    state.log_action("job_started", job_id=JobId("j1"), worker_id=WorkerId("w1"))

    response = client.get("/api/actions")
    assert response.status_code == 200
    actions = response.json()
    assert len(actions) == 3

    assert actions[0]["action"] == "job_submitted"
    assert actions[0]["job_id"] == "j1"
    assert actions[0]["details"] == "test job"
    assert actions[1]["action"] == "worker_registered"
    assert actions[1]["worker_id"] == "w1"
    assert actions[2]["action"] == "job_started"


def test_rpc_mounted_at_correct_path(client, state, make_job_request):
    # Add a job so ListJobs returns something
    state.add_job(
        ControllerJob(
            job_id=JobId("test-job"),
            request=make_job_request("test"),
            state=cluster_pb2.JOB_STATE_PENDING,
        )
    )

    # Call ListJobs via Connect RPC
    response = client.post(
        "/fluster.cluster.ControllerService/ListJobs",
        headers={"Content-Type": "application/json"},
        content="{}",
    )
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert len(data["jobs"]) == 1
    assert data["jobs"][0]["jobId"] == "test-job"


def test_rpc_get_job_status(client, state, make_job_request):
    state.add_job(
        ControllerJob(
            job_id=JobId("j1"),
            request=make_job_request("test"),
            state=cluster_pb2.JOB_STATE_RUNNING,
            worker_id=WorkerId("w1"),
        )
    )

    response = client.post(
        "/fluster.cluster.ControllerService/GetJobStatus",
        headers={"Content-Type": "application/json"},
        content='{"jobId": "j1"}',
    )
    assert response.status_code == 200
    data = response.json()
    assert data["job"]["jobId"] == "j1"
    assert data["job"]["state"] == "JOB_STATE_RUNNING"
    assert data["job"]["workerId"] == "w1"


def test_rpc_launch_job(client):
    response = client.post(
        "/fluster.cluster.ControllerService/LaunchJob",
        headers={"Content-Type": "application/json"},
        content="""{
            "name": "new-job",
            "serializedEntrypoint": "dGVzdA==",
            "resources": {"cpu": 1, "memory": "1g"},
            "environment": {"workspace": "/tmp"}
        }""",
    )
    assert response.status_code == 200
    data = response.json()
    assert "jobId" in data
    assert len(data["jobId"]) > 0


def test_dashboard_html_has_javascript(client):
    """Verify dashboard has auto-refresh JavaScript."""
    response = client.get("/")
    assert "setInterval(refresh" in response.text
    assert "fetch('/api/stats')" in response.text
    assert "fetch('/api/jobs')" in response.text
    assert "fetch('/api/workers')" in response.text
    assert "fetch('/api/actions')" in response.text


def test_api_jobs_includes_all_states(client, state, make_job_request):
    """Verify all job states are represented correctly."""
    states = [
        (cluster_pb2.JOB_STATE_PENDING, "pending"),
        (cluster_pb2.JOB_STATE_RUNNING, "running"),
        (cluster_pb2.JOB_STATE_SUCCEEDED, "succeeded"),
        (cluster_pb2.JOB_STATE_FAILED, "failed"),
        (cluster_pb2.JOB_STATE_KILLED, "killed"),
        (cluster_pb2.JOB_STATE_WORKER_FAILED, "worker_failed"),
    ]

    for i, (proto_state, _) in enumerate(states):
        state.add_job(
            ControllerJob(
                job_id=JobId(f"j{i}"),
                request=make_job_request(f"job{i}"),
                state=proto_state,
            )
        )

    response = client.get("/api/jobs")
    jobs = response.json()
    job_states = {j["job_id"]: j["state"] for j in jobs}

    for i, (_, expected_state) in enumerate(states):
        assert job_states[f"j{i}"] == expected_state
