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

"""Tests for controller dashboard behavioral logic.

Tests verify dashboard functionality through the Connect RPC endpoints.
The dashboard serves a web UI that fetches data via RPC calls.
"""

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
    controller_mock.task_schedule_status = scheduler.task_schedule_status
    controller_mock.autoscaler = None  # No autoscaler by default
    return ControllerServiceImpl(state, controller_mock, bundle_prefix="file:///tmp/iris-test-bundles")


@pytest.fixture
def client(service):
    dashboard = ControllerDashboard(service)
    return TestClient(dashboard._app)


@pytest.fixture
def service_with_autoscaler(state, scheduler, mock_autoscaler):
    """Service with autoscaler enabled for tests."""
    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.task_schedule_status = scheduler.task_schedule_status
    controller_mock.autoscaler = mock_autoscaler  # Enable autoscaler
    return ControllerServiceImpl(state, controller_mock, bundle_prefix="file:///tmp/iris-test-bundles")


def rpc_post(client: TestClient, method: str, body: dict | None = None):
    """Helper to call RPC endpoint and return JSON response."""
    resp = client.post(
        f"/iris.cluster.ControllerService/{method}",
        json=body or {},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200, f"RPC {method} failed: {resp.text}"
    return resp.json()


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


def test_list_jobs_returns_job_state_counts(client, state, job_request):
    """ListJobs RPC returns jobs with correct state values."""
    submit_job(state, "pending", job_request)
    # Job is already in PENDING state after submission

    building_id = submit_job(state, "building", job_request)
    state.get_job(building_id).state = cluster_pb2.JOB_STATE_BUILDING

    running_id = submit_job(state, "running", job_request)
    state.get_job(running_id).state = cluster_pb2.JOB_STATE_RUNNING

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])

    jobs_by_state = {}
    for j in jobs:
        state_name = j.get("state", "")
        jobs_by_state[state_name] = jobs_by_state.get(state_name, 0) + 1

    assert jobs_by_state.get("JOB_STATE_PENDING", 0) == 1
    assert jobs_by_state.get("JOB_STATE_BUILDING", 0) == 1
    assert jobs_by_state.get("JOB_STATE_RUNNING", 0) == 1


def test_list_jobs_includes_terminal_states(client, state, job_request):
    """ListJobs RPC returns jobs with terminal states."""
    for job_state in [
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
    ]:
        job_id = submit_job(state, f"job-{job_state}", job_request)
        state.get_job(job_id).state = job_state

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])

    assert len(jobs) == 4
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_KILLED", "JOB_STATE_WORKER_FAILED"}
    for j in jobs:
        assert j.get("state") in terminal_states


def test_list_workers_returns_healthy_status(client, state, make_worker_metadata):
    """ListWorkers RPC returns workers with healthy status."""
    register_worker(state, "healthy1", "h1:8080", make_worker_metadata())
    register_worker(state, "healthy2", "h2:8080", make_worker_metadata())
    register_worker(state, "unhealthy", "h3:8080", make_worker_metadata(), healthy=False)

    resp = rpc_post(client, "ListWorkers")
    workers = resp.get("workers", [])

    assert len(workers) == 3
    healthy_count = sum(1 for w in workers if w.get("healthy", False))
    assert healthy_count == 2


def test_list_endpoints_filters_by_running_jobs(client, state, job_request):
    """ListEndpoints RPC filters out endpoints for non-running jobs."""
    # Running job with endpoint
    running_id = submit_job(state, "running", job_request)
    state.get_job(running_id).state = cluster_pb2.JOB_STATE_RUNNING
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep1", name="svc", address="host:80", job_id=running_id))

    # Pending job with endpoint (should not be returned)
    pending_id = submit_job(state, "pending", job_request)
    state.add_endpoint(ControllerEndpoint(endpoint_id="ep2", name="svc2", address="host:81", job_id=pending_id))

    resp = rpc_post(client, "ListEndpoints", {"prefix": ""})
    endpoints = resp.get("endpoints", [])

    assert len(endpoints) == 1
    assert endpoints[0]["name"] == "svc"


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

    resp = rpc_post(client, "ListEndpoints", {"prefix": ""})
    endpoints = resp.get("endpoints", [])

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


def test_list_jobs_state_names_mapped_correctly(client, state, job_request):
    """Proto state enums are returned as expected."""
    state_mapping = [
        (cluster_pb2.JOB_STATE_PENDING, "JOB_STATE_PENDING"),
        (cluster_pb2.JOB_STATE_BUILDING, "JOB_STATE_BUILDING"),
        (cluster_pb2.JOB_STATE_RUNNING, "JOB_STATE_RUNNING"),
        (cluster_pb2.JOB_STATE_SUCCEEDED, "JOB_STATE_SUCCEEDED"),
        (cluster_pb2.JOB_STATE_FAILED, "JOB_STATE_FAILED"),
        (cluster_pb2.JOB_STATE_KILLED, "JOB_STATE_KILLED"),
        (cluster_pb2.JOB_STATE_WORKER_FAILED, "JOB_STATE_WORKER_FAILED"),
    ]

    for proto_state, _ in state_mapping:
        job_id = submit_job(state, f"j-{proto_state}", job_request)
        state.get_job(job_id).state = proto_state

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])
    # RPC uses camelCase field names (Connect RPC standard)
    job_by_id = {j["jobId"]: j["state"] for j in jobs}

    for proto_state, expected_name in state_mapping:
        assert job_by_id[f"j-{proto_state}"] == expected_name


def test_list_jobs_includes_retry_counts(client, state, job_request):
    """ListJobs RPC includes retry count fields aggregated from tasks."""
    job_id = submit_job(state, "test-job", job_request)
    job = state.get_job(job_id)
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # Set retry counts on tasks (the RPC aggregates from tasks, not job)
    tasks = state.get_job_tasks(job_id)
    tasks[0].failure_count = 1
    tasks[0].preemption_count = 2

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])

    assert len(jobs) == 1
    # RPC uses camelCase field names
    assert jobs[0]["failureCount"] == 1
    assert jobs[0]["preemptionCount"] == 2


def test_list_jobs_includes_task_counts(client, state):
    """ListJobs RPC returns taskCount, completedCount, and taskStateCounts for compact view."""
    # Submit a job with multiple replicas (replicas is on ResourceSpecProto)
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="multi-replica-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    job_id = submit_job(state, "multi", request)
    job = state.get_job(job_id)
    job.state = cluster_pb2.JOB_STATE_RUNNING

    # Get the tasks and set their states
    tasks = state.get_job_tasks(job_id)
    assert len(tasks) == 3
    tasks[0].state = cluster_pb2.TASK_STATE_SUCCEEDED
    tasks[1].state = cluster_pb2.TASK_STATE_RUNNING
    tasks[2].state = cluster_pb2.TASK_STATE_PENDING

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])

    assert len(jobs) == 1
    j = jobs[0]
    # RPC uses camelCase field names
    assert j["taskCount"] == 3
    assert j["completedCount"] == 1  # Only succeeded counts
    assert j["taskStateCounts"]["TASK_STATE_SUCCEEDED"] == 1
    assert j["taskStateCounts"]["TASK_STATE_RUNNING"] == 1
    assert j["taskStateCounts"]["TASK_STATE_PENDING"] == 1


def test_get_job_status_returns_retry_info(client, state, job_request):
    """GetJobStatus RPC returns retry counts and current state.

    Jobs no longer track individual attempts - tasks do. The RPC returns
    aggregate retry information for the job.
    """
    job_id = submit_job(state, "test-job", job_request)
    job = state.get_job(job_id)
    job.state = cluster_pb2.JOB_STATE_RUNNING
    job.started_at_ms = 3000

    # Set retry counts on tasks (the RPC aggregates from tasks)
    tasks = state.get_job_tasks(job_id)
    tasks[0].failure_count = 1
    tasks[0].preemption_count = 1

    # RPC uses camelCase: jobId not job_id
    resp = rpc_post(client, "GetJobStatus", {"jobId": "test-job"})
    job_status = resp.get("job", {})

    # RPC uses camelCase field names
    assert job_status["failureCount"] == 1
    assert job_status["preemptionCount"] == 1
    assert job_status["state"] == "JOB_STATE_RUNNING"
    assert int(job_status["startedAtMs"]) == 3000


def test_get_job_status_returns_error_for_missing_job(client):
    """GetJobStatus RPC returns error for non-existent job."""
    resp = client.post(
        "/iris.cluster.ControllerService/GetJobStatus",
        json={"jobId": "nonexistent"},
        headers={"Content-Type": "application/json"},
    )
    # Connect RPC returns non-200 status for errors
    assert resp.status_code != 200


# =============================================================================
# Autoscaler RPC Tests
# =============================================================================


def test_get_autoscaler_status_returns_disabled_when_no_autoscaler(client):
    """GetAutoscalerStatus RPC returns empty status when autoscaler is not configured."""
    resp = rpc_post(client, "GetAutoscalerStatus")
    status = resp.get("status", {})

    # When no autoscaler, should return empty status
    assert status.get("groups", []) == []


@pytest.fixture
def mock_autoscaler():
    """Create a mock autoscaler that returns a status proto."""
    from iris.rpc import config_pb2, vm_pb2

    autoscaler = Mock()
    autoscaler.get_status.return_value = vm_pb2.AutoscalerStatus(
        groups=[
            vm_pb2.ScaleGroupStatus(
                name="test-group",
                config=config_pb2.ScaleGroupConfig(
                    name="test-group",
                    accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
                    accelerator_variant="v4-8",
                    min_slices=1,
                    max_slices=5,
                ),
                slices=[
                    vm_pb2.SliceInfo(
                        slice_id="slice-1",
                        scale_group="test-group",
                        vms=[vm_pb2.VmInfo(vm_id="vm-1", state=vm_pb2.VM_STATE_READY)],
                    ),
                    vm_pb2.SliceInfo(
                        slice_id="slice-2",
                        scale_group="test-group",
                        vms=[vm_pb2.VmInfo(vm_id="vm-2", state=vm_pb2.VM_STATE_READY)],
                    ),
                    vm_pb2.SliceInfo(
                        slice_id="slice-3",
                        scale_group="test-group",
                        vms=[vm_pb2.VmInfo(vm_id="vm-3", state=vm_pb2.VM_STATE_BOOTING)],
                    ),
                ],
                current_demand=3,
            ),
        ],
        current_demand={"test-group": 3},
        last_evaluation_ms=1000,
        recent_actions=[
            vm_pb2.AutoscalerAction(
                timestamp_ms=1000,
                action_type="scale_up",
                scale_group="test-group",
                slice_id="slice-1",
                reason="demand=3 > capacity=2",
            ),
        ],
    )
    return autoscaler


@pytest.fixture
def client_with_autoscaler(service_with_autoscaler):
    """Dashboard test client with autoscaler enabled."""
    dashboard = ControllerDashboard(service_with_autoscaler)
    return TestClient(dashboard._app)


def test_get_autoscaler_status_returns_status_when_enabled(client_with_autoscaler):
    """GetAutoscalerStatus RPC returns full status when autoscaler is configured."""
    resp = rpc_post(client_with_autoscaler, "GetAutoscalerStatus")
    data = resp.get("status", {})

    # Verify groups data (RPC uses camelCase field names)
    assert len(data["groups"]) == 1
    group = data["groups"][0]
    assert group["name"] == "test-group"
    assert group["currentDemand"] == 3

    # Verify demand tracking
    assert data["currentDemand"] == {"test-group": 3}
    # Int64 fields are serialized as strings in JSON (protobuf convention)
    assert int(data["lastEvaluationMs"]) == 1000

    # Verify recent actions
    assert len(data["recentActions"]) == 1
    action = data["recentActions"][0]
    assert action["actionType"] == "scale_up"
    assert action["scaleGroup"] == "test-group"


def test_get_autoscaler_status_includes_slice_details(client_with_autoscaler):
    """GetAutoscalerStatus RPC returns scale group slice details."""
    resp = rpc_post(client_with_autoscaler, "GetAutoscalerStatus")
    data = resp.get("status", {})

    assert len(data["groups"]) == 1
    group = data["groups"][0]
    assert group["name"] == "test-group"
    # Verify slices are included in response
    assert len(group["slices"]) == 3
    # Verify slice structure (RPC uses camelCase)
    for slice_info in group["slices"]:
        assert "sliceId" in slice_info
        assert "vms" in slice_info
        assert len(slice_info["vms"]) == 1
    assert group["config"]["acceleratorVariant"] == "v4-8"


# =============================================================================
# Health Endpoint Tests
# =============================================================================


def test_health_endpoint_returns_ok(client, state, make_worker_metadata, job_request):
    """Health endpoint returns status ok with worker and job counts."""
    register_worker(state, "w1", "h1:8080", make_worker_metadata())
    register_worker(state, "w2", "h2:8080", make_worker_metadata())
    submit_job(state, "j1", job_request)

    resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["workers"] == 2
    assert data["jobs"] == 1


def test_health_endpoint_empty_cluster(client):
    """Health endpoint returns ok for empty cluster (no workers, no jobs)."""
    resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["workers"] == 0
    assert data["jobs"] == 0
