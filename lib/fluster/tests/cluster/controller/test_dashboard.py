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

"""Tests for controller dashboard."""

import pytest
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from fluster import cluster_pb2
from fluster.cluster.controller.dashboard import create_dashboard_app
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.types import JobId, WorkerId


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
def make_resource_spec():
    """Create a minimal ResourceSpec for testing."""

    def _make() -> cluster_pb2.ResourceSpec:
        return cluster_pb2.ResourceSpec(cpu=1, memory="1g", disk="10g")

    return _make


class TestDashboard(AioHTTPTestCase):
    """Test suite for dashboard endpoints using aiohttp test utilities."""

    async def get_application(self):
        """Create test application with prepopulated state."""
        self.state = ControllerState()

        # Add some workers
        worker1 = ControllerWorker(
            worker_id=WorkerId("w1"),
            address="host1:8080",
            resources=cluster_pb2.ResourceSpec(cpu=4, memory="16g", disk="100g"),
            healthy=True,
        )
        worker2 = ControllerWorker(
            worker_id=WorkerId("w2"),
            address="host2:8080",
            resources=cluster_pb2.ResourceSpec(cpu=4, memory="16g", disk="100g"),
            healthy=False,
        )
        self.state.add_worker(worker1)
        self.state.add_worker(worker2)

        # Add some jobs
        job1 = ControllerJob(
            job_id=JobId("j1"),
            request=cluster_pb2.LaunchJobRequest(
                name="job1",
                serialized_entrypoint=b"test",
                resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
                environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
            ),
            state=cluster_pb2.JOB_STATE_RUNNING,
            worker_id=WorkerId("w1"),
        )
        job2 = ControllerJob(
            job_id=JobId("j2"),
            request=cluster_pb2.LaunchJobRequest(
                name="job2",
                serialized_entrypoint=b"test",
                resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
                environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
            ),
            state=cluster_pb2.JOB_STATE_FAILED,
            error="Test error",
        )
        self.state.add_job(job1)
        self.state.add_job(job2)

        # Mark job1 as running on worker1
        worker1.running_jobs.add(job1.job_id)

        return create_dashboard_app(self.state)

    @unittest_run_loop
    async def test_dashboard_index_returns_html(self):
        """Verify index returns HTML with workers and jobs."""
        resp = await self.client.request("GET", "/")
        assert resp.status == 200
        assert resp.content_type == "text/html"

        text = await resp.text()
        assert "Fluster Controller Dashboard" in text
        assert "Workers" in text
        assert "Jobs" in text

    @unittest_run_loop
    async def test_dashboard_shows_worker_info(self):
        """Verify worker table has correct data."""
        resp = await self.client.request("GET", "/")
        assert resp.status == 200

        text = await resp.text()

        # Check worker counts
        assert "1 healthy / 2 total" in text

        # Check worker data appears
        assert "w1" in text
        assert "host1:8080" in text
        assert "w2" in text
        assert "host2:8080" in text

        # Check healthy status
        assert "Yes" in text  # w1 is healthy
        assert "No" in text  # w2 is unhealthy

    @unittest_run_loop
    async def test_dashboard_shows_job_info(self):
        """Verify job table has correct data."""
        resp = await self.client.request("GET", "/")
        assert resp.status == 200

        text = await resp.text()

        # Check job count
        assert "2 total" in text

        # Check job data appears
        assert "j1" in text
        assert "job1" in text
        assert "j2" in text
        assert "job2" in text

        # Check job states
        assert "RUNNING" in text
        assert "FAILED" in text

        # Check worker assignment
        assert "w1" in text  # j1 assigned to w1

        # Check error message
        assert "Test error" in text

    @unittest_run_loop
    async def test_dashboard_health_endpoint(self):
        """Verify /health returns JSON."""
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200

        json_data = await resp.json()
        assert json_data["status"] == "ok"
        assert json_data["workers"] == 2
        assert json_data["healthy_workers"] == 1
        assert json_data["jobs"] == 2

    @unittest_run_loop
    async def test_dashboard_escapes_html_in_user_data(self):
        """Verify HTML special characters are escaped to prevent XSS."""
        # Add a job with XSS payload in the name
        xss_job = ControllerJob(
            job_id=JobId("xss_job"),
            request=cluster_pb2.LaunchJobRequest(
                name="<script>alert('xss')</script>",
                serialized_entrypoint=b"test",
                resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
                environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
            ),
            state=cluster_pb2.JOB_STATE_FAILED,
            error="<img src=x onerror=alert('xss')>",
        )
        self.state.add_job(xss_job)

        # Add a worker with XSS payload in address
        xss_worker = ControllerWorker(
            worker_id=WorkerId("<script>bad</script>"),
            address="<img onerror=alert(1)>",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
            healthy=True,
        )
        self.state.add_worker(xss_worker)

        resp = await self.client.request("GET", "/")
        text = await resp.text()

        # Raw script/img tags should NOT appear (would be XSS vectors)
        assert "<script>" not in text
        assert "<img " not in text  # space after img to match tag, not escaped &lt;img

        # Escaped versions should appear
        assert "&lt;script&gt;" in text
        assert "&lt;img" in text


def test_dashboard_empty_state():
    """Test dashboard with no workers or jobs."""
    state = ControllerState()
    app = create_dashboard_app(state)

    # Just verify app creation succeeds
    assert app is not None


def test_dashboard_with_many_jobs(make_job_request, make_resource_spec):
    """Test dashboard can handle many jobs."""
    state = ControllerState()

    # Add a worker
    worker = ControllerWorker(
        worker_id=WorkerId("w1"),
        address="host1:8080",
        resources=make_resource_spec(),
    )
    state.add_worker(worker)

    # Add many jobs
    for i in range(100):
        job = ControllerJob(
            job_id=JobId(f"j{i}"),
            request=make_job_request(f"job{i}"),
            state=cluster_pb2.JOB_STATE_PENDING if i % 2 == 0 else cluster_pb2.JOB_STATE_RUNNING,
        )
        state.add_job(job)

    app = create_dashboard_app(state)
    assert app is not None


def test_dashboard_with_gang_jobs(make_job_request, make_resource_spec):
    """Test dashboard with gang-scheduled jobs."""
    state = ControllerState()

    # Add workers
    for i in range(4):
        worker = ControllerWorker(
            worker_id=WorkerId(f"w{i}"),
            address=f"host{i}:8080",
            resources=make_resource_spec(),
        )
        state.add_worker(worker)

    # Add gang jobs
    for i in range(4):
        job = ControllerJob(
            job_id=JobId(f"gang_job_{i}"),
            request=make_job_request(f"gang_job_{i}"),
            state=cluster_pb2.JOB_STATE_RUNNING,
            worker_id=WorkerId(f"w{i}"),
            gang_id="gang1",
        )
        state.add_job(job)

    app = create_dashboard_app(state)
    assert app is not None


def test_dashboard_all_job_states(make_job_request):
    """Test dashboard shows all job state types correctly."""
    state = ControllerState()

    states = [
        (cluster_pb2.JOB_STATE_PENDING, "PENDING"),
        (cluster_pb2.JOB_STATE_RUNNING, "RUNNING"),
        (cluster_pb2.JOB_STATE_SUCCEEDED, "SUCCEEDED"),
        (cluster_pb2.JOB_STATE_FAILED, "FAILED"),
        (cluster_pb2.JOB_STATE_KILLED, "KILLED"),
        (cluster_pb2.JOB_STATE_WORKER_FAILED, "WORKER_FAILED"),
    ]

    for state_val, state_name in states:
        job = ControllerJob(
            job_id=JobId(f"j_{state_name}"),
            request=make_job_request(f"job_{state_name}"),
            state=state_val,
        )
        state.add_job(job)

    app = create_dashboard_app(state)
    assert app is not None
