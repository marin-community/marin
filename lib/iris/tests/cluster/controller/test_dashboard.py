# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller dashboard behavioral logic.

Tests verify dashboard functionality through the Connect RPC endpoints.
The dashboard serves a web UI that fetches data via RPC calls.
"""

import re
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from iris.cluster.bundle import BundleStore
from iris.cluster.controller.codec import constraints_from_json, resource_spec_from_scalars
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.log_server.server import LogServiceImpl
from iris.cluster.controller.db import (
    healthy_active_workers_with_attributes,
)
from iris.cluster.controller.schema import (
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
    EndpointRow,
)
from iris.cluster.controller.scheduler import JobRequirements, Scheduler
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import Assignment, ControllerTransitions, HeartbeatApplyRequest, TaskUpdate
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.providers.k8s.types import K8sResource
from iris.cluster.types import JobName, WorkerId
from iris.rpc import config_pb2, vm_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.time_proto import timestamp_to_proto
from rigging.timing import Timestamp

from .conftest import (
    check_task_can_be_scheduled,
    make_test_entrypoint,
    make_worker_metadata,
    query_tasks_with_attempts as _query_tasks_with_attempts,
    register_worker,
)

# =============================================================================
# Test Helpers
# =============================================================================


def submit_job(
    state: ControllerTransitions,
    job_id: str,
    request: controller_pb2.Controller.LaunchJobRequest,
) -> JobName:
    """Submit a job through the state command API."""
    jid = JobName.from_string(job_id) if job_id.startswith("/") else JobName.root("test-user", job_id)
    request.name = jid.to_wire()
    state.submit_job(jid, request, Timestamp.now())
    return jid


def set_job_state(
    state: ControllerTransitions, job_id: JobName, new_state: int, *, started_at_ms: int | None = None
) -> None:
    """Directly set job state in DB for dashboard-only read-model tests."""
    state._db.execute(
        "UPDATE jobs SET state = ?, started_at_ms = COALESCE(?, started_at_ms) WHERE job_id = ?",
        (new_state, started_at_ms, job_id.to_wire()),
    )


def set_task_retry_counts(
    state: ControllerTransitions,
    task_id: JobName,
    *,
    failure_count: int | None = None,
    preemption_count: int | None = None,
) -> None:
    """Directly set retry counters in DB for read-model aggregate tests."""
    state._db.execute(
        "UPDATE tasks SET failure_count = COALESCE(?, failure_count), preemption_count = COALESCE(?, preemption_count) "
        "WHERE task_id = ?",
        (failure_count, preemption_count, task_id.to_wire()),
    )


def set_task_state(state: ControllerTransitions, task_id: JobName, new_state: int) -> None:
    """Directly set task state in DB for aggregate count tests."""
    state._db.execute(
        "UPDATE tasks SET state = ? WHERE task_id = ?",
        (new_state, task_id.to_wire()),
    )


@pytest.fixture
def scheduler():
    return Scheduler()


def _make_controller_mock(state, scheduler, autoscaler=None):
    """Build a mock that implements the ControllerProtocol for testing.

    The mock delegates create_scheduling_context to the scheduler and computes
    scheduling diagnostics on the fly, mirroring how the real controller caches
    diagnostics per scheduling cycle.
    """

    def _create_scheduling_context(workers):
        with state._db.snapshot() as q:
            rows = q.raw(
                "SELECT a.worker_id, COUNT(*) as c FROM tasks t "
                "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
                "JOIN jobs j ON t.job_id = j.job_id "
                "WHERE t.state IN (?, ?) AND j.is_reservation_holder = 0 "
                "GROUP BY a.worker_id ORDER BY a.worker_id ASC",
                (
                    job_pb2.TASK_STATE_BUILDING,
                    job_pb2.TASK_STATE_ASSIGNED,
                ),
                decoders={"worker_id": WorkerId, "c": int},
            )
        building_counts = {row.worker_id: row.c for row in rows}
        return scheduler.create_scheduling_context(workers, building_counts=building_counts)

    def _get_job_scheduling_diagnostics(job_wire_id):
        """Compute diagnostics on the fly for tests (mirrors real controller cache)."""
        with state._db.snapshot() as q:
            rows = JOB_DETAIL_PROJECTION.decode(
                q.fetchall(
                    f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
                    (job_wire_id,),
                )
            )
        if not rows:
            return None
        job = rows[0]
        if job.state != job_pb2.JOB_STATE_PENDING:
            return None
        req = JobRequirements(
            resources=resource_spec_from_scalars(
                job.res_cpu_millicores, job.res_memory_bytes, job.res_disk_bytes, job.res_device_json
            ),
            constraints=constraints_from_json(job.constraints_json),
            is_coscheduled=job.has_coscheduling,
            coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
        )
        tasks = _query_tasks_with_attempts(state, job.job_id)
        schedulable_task_id = next((t.task_id for t in tasks if check_task_can_be_scheduled(t)), None)
        workers = healthy_active_workers_with_attributes(state._db)
        context = _create_scheduling_context(workers)
        return scheduler.get_job_scheduling_diagnostics(req, context, schedulable_task_id, num_tasks=len(tasks))

    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.create_scheduling_context = _create_scheduling_context
    controller_mock.get_job_scheduling_diagnostics = _get_job_scheduling_diagnostics
    controller_mock.autoscaler = autoscaler
    controller_mock.provider = Mock()
    controller_mock.has_direct_provider = False
    return controller_mock


@pytest.fixture
def service(state, scheduler, tmp_path):
    controller_mock = _make_controller_mock(state, scheduler)
    log_service = LogServiceImpl()
    return ControllerServiceImpl(
        state,
        state._db,
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=log_service,
    )


@pytest.fixture
def client(service):
    dashboard = ControllerDashboard(service, log_service=service._log_service)
    return TestClient(dashboard.app)


@pytest.fixture
def service_with_autoscaler(state, scheduler, mock_autoscaler, tmp_path):
    """Service with autoscaler enabled for tests."""
    controller_mock = _make_controller_mock(state, scheduler, autoscaler=mock_autoscaler)
    log_service = LogServiceImpl()
    return ControllerServiceImpl(
        state,
        state._db,
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=log_service,
    )


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
def job_request():
    return controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "test-job").to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=4 * 1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )


@pytest.fixture
def resource_spec():
    return job_pb2.ResourceSpecProto(cpu_millicores=4000, memory_bytes=8 * 1024**3, disk_bytes=100 * 1024**3)


def test_list_jobs_returns_job_state_counts(client, state, job_request):
    """ListJobs RPC returns jobs with correct state values."""
    submit_job(state, "pending", job_request)
    # Job is already in PENDING state after submission

    building_id = submit_job(state, "building", job_request)
    running_id = submit_job(state, "running", job_request)
    set_job_state(state, building_id, job_pb2.JOB_STATE_BUILDING)
    set_job_state(state, running_id, job_pb2.JOB_STATE_RUNNING)

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
    overrides: list[tuple[JobName, int]] = []
    for job_state in [
        job_pb2.JOB_STATE_SUCCEEDED,
        job_pb2.JOB_STATE_FAILED,
        job_pb2.JOB_STATE_KILLED,
        job_pb2.JOB_STATE_WORKER_FAILED,
    ]:
        job_id = submit_job(state, f"job-{job_state}", job_request)
        overrides.append((job_id, job_state))
    for job_id, job_state in overrides:
        set_job_state(state, job_id, job_state)

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])

    assert len(jobs) == 4
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_KILLED", "JOB_STATE_WORKER_FAILED"}
    for j in jobs:
        assert j.get("state") in terminal_states


def test_list_workers_returns_healthy_status(client, state):
    """ListWorkers RPC returns workers with healthy status."""
    register_worker(state, "healthy1", "h1:8080", make_worker_metadata())
    register_worker(state, "healthy2", "h2:8080", make_worker_metadata())
    register_worker(state, "unhealthy", "h3:8080", make_worker_metadata(), healthy=False)

    resp = rpc_post(client, "ListWorkers")
    workers = resp.get("workers", [])

    assert len(workers) == 3
    healthy_count = sum(1 for w in workers if w.get("healthy", False))
    assert healthy_count == 2


def test_endpoints_only_returned_for_running_jobs(client, state, job_request):
    """ListEndpoints returns endpoints for non-terminal jobs.

    Endpoints are associated with tasks and deleted when tasks reach terminal states,
    so only endpoints for pending/running jobs should exist at query time.
    """
    # Create jobs in various states
    pending_id = submit_job(state, "pending", job_request)

    running_id = submit_job(state, "running", job_request)
    set_job_state(state, running_id, job_pb2.JOB_STATE_RUNNING)

    # No endpoint for succeeded job — endpoints are deleted when tasks go terminal
    succeeded_id = submit_job(state, "succeeded", job_request)
    set_job_state(state, succeeded_id, job_pb2.JOB_STATE_SUCCEEDED)

    # Add endpoints only for non-terminal jobs
    state.add_endpoint(
        EndpointRow(
            endpoint_id="ep1",
            name="pending-svc",
            address="h:1",
            task_id=pending_id.task(0),
            metadata={},
            registered_at=Timestamp.now(),
        ),
    )
    state.add_endpoint(
        EndpointRow(
            endpoint_id="ep2",
            name="running-svc",
            address="h:2",
            task_id=running_id.task(0),
            metadata={},
            registered_at=Timestamp.now(),
        ),
    )

    resp = rpc_post(client, "ListEndpoints", {"prefix": ""})
    endpoints = resp.get("endpoints", [])

    assert len(endpoints) == 2
    endpoint_names = {ep["name"] for ep in endpoints}
    assert endpoint_names == {"pending-svc", "running-svc"}


def test_list_endpoints_returns_task_id(client, state, job_request):
    """ListEndpoints returns the task_id so the dashboard can derive the owning job."""
    job_id = submit_job(state, "ep-job", job_request)
    set_job_state(state, job_id, job_pb2.JOB_STATE_RUNNING)

    task_id = job_id.task(0)
    state.add_endpoint(
        EndpointRow(
            endpoint_id="ep-task",
            name="my-actor",
            address="h:1",
            task_id=task_id,
            metadata={},
            registered_at=Timestamp.now(),
        ),
    )

    resp = rpc_post(client, "ListEndpoints", {"prefix": ""})
    endpoints = resp.get("endpoints", [])
    assert len(endpoints) == 1
    # The response must carry the full task_id (including task index) so the
    # dashboard's jobIdFromTaskId() can strip the index and show the job name.
    assert endpoints[0]["taskId"] == task_id.to_wire()


def test_list_jobs_includes_retry_counts(client, state, job_request):
    """ListJobs RPC includes retry count fields aggregated from tasks."""
    job_id = submit_job(state, "test-job", job_request)
    set_job_state(state, job_id, job_pb2.JOB_STATE_RUNNING)

    # Set retry counts on tasks (the RPC aggregates from tasks, not job)
    tasks = _query_tasks_with_attempts(state, job_id)
    set_task_retry_counts(state, tasks[0].task_id, failure_count=1, preemption_count=2)

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])

    assert len(jobs) == 1
    # RPC uses camelCase field names
    assert jobs[0]["failureCount"] == 1
    assert jobs[0]["preemptionCount"] == 2


def test_list_jobs_includes_task_counts(client, state):
    """ListJobs RPC returns taskCount, completedCount, and taskStateCounts for compact view."""
    # Submit a job with multiple replicas (replicas is on ResourceSpecProto)
    request = controller_pb2.Controller.LaunchJobRequest(
        name="multi-replica-job",
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=3,
        environment=job_pb2.EnvironmentConfig(),
    )
    job_id = submit_job(state, "multi", request)
    set_job_state(state, job_id, job_pb2.JOB_STATE_RUNNING)

    # Get the tasks and set their states
    tasks = _query_tasks_with_attempts(state, job_id)
    assert len(tasks) == 3
    set_task_state(state, tasks[0].task_id, job_pb2.TASK_STATE_SUCCEEDED)
    set_task_state(state, tasks[1].task_id, job_pb2.TASK_STATE_RUNNING)
    set_task_state(state, tasks[2].task_id, job_pb2.TASK_STATE_PENDING)

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])

    assert len(jobs) == 1
    j = jobs[0]
    # RPC uses camelCase field names
    assert j["taskCount"] == 3
    assert j["completedCount"] == 1  # Only succeeded counts
    assert j["taskStateCounts"]["succeeded"] == 1
    assert j["taskStateCounts"]["running"] == 1
    assert j["taskStateCounts"]["pending"] == 1


def test_list_users_returns_aggregates(client, state):
    """ListUsers RPC returns one aggregate row per user."""
    request = controller_pb2.Controller.LaunchJobRequest(
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "/alice/train", request)
    submit_job(state, "/alice/eval", request)
    submit_job(state, "/bob/train", request)

    resp = rpc_post(client, "ListUsers")
    users = {entry["user"]: entry for entry in resp.get("users", [])}

    assert users["alice"]["jobStateCounts"]["pending"] == 2
    assert users["alice"]["taskStateCounts"]["pending"] == 2
    assert users["bob"]["jobStateCounts"]["pending"] == 1
    assert users["bob"]["taskStateCounts"]["pending"] == 1


def test_get_job_status_returns_retry_info(client, state, job_request):
    """GetJobStatus RPC returns retry counts and current state.

    Jobs no longer track individual attempts - tasks do. The RPC returns
    aggregate retry information for the job.
    """
    job_id = submit_job(state, "test-job", job_request)
    set_job_state(state, job_id, job_pb2.JOB_STATE_RUNNING, started_at_ms=3000)

    # Set retry counts on tasks (the RPC aggregates from tasks)
    tasks = _query_tasks_with_attempts(state, job_id)
    set_task_retry_counts(state, tasks[0].task_id, failure_count=1, preemption_count=1)

    # RPC uses camelCase: jobId not job_id
    resp = rpc_post(client, "GetJobStatus", {"jobId": JobName.root("test-user", "test-job").to_wire()})
    job_status = resp.get("job", {})

    # RPC uses camelCase field names
    assert job_status["failureCount"] == 1
    assert job_status["preemptionCount"] == 1
    assert job_status["state"] == "JOB_STATE_RUNNING"
    assert int(job_status["startedAt"]["epochMs"]) == 3000


def test_get_job_status_returns_original_request(client, state):
    """GetJobStatus RPC returns the original LaunchJobRequest for the job detail page."""
    request = controller_pb2.Controller.LaunchJobRequest(
        name="request-detail-job",
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=4000,
            memory_bytes=8 * 1024**3,
            disk_bytes=100 * 1024**3,
        ),
        environment=job_pb2.EnvironmentConfig(
            pip_packages=["torch", "numpy"],
            python_version="3.11",
        ),
        replicas=2,
        constraints=[
            job_pb2.Constraint(
                key=WellKnownAttribute.TPU_NAME,
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="my-tpu"),
            ),
        ],
        coscheduling=job_pb2.CoschedulingConfig(group_by=WellKnownAttribute.TPU_NAME),
    )
    job_id = submit_job(state, "request-detail-job", request)

    resp = rpc_post(client, "GetJobStatus", {"jobId": job_id.to_wire()})
    returned_request = resp.get("request", {})

    assert returned_request is not None
    # Verify entrypoint command is preserved
    ep = returned_request.get("entrypoint", {})
    assert ep.get("runCommand", {}).get("argv") == ["python", "-c", "pass"]
    # Verify resources
    res = returned_request.get("resources", {})
    assert res["cpuMillicores"] == 4000
    assert int(res["memoryBytes"]) == 8 * 1024**3
    assert int(res["diskBytes"]) == 100 * 1024**3
    # Verify environment
    env = returned_request.get("environment", {})
    assert env["pipPackages"] == ["torch", "numpy"]
    assert env["pythonVersion"] == "3.11"
    # Verify replicas
    assert returned_request["replicas"] == 2
    # Verify constraints
    constraints = returned_request.get("constraints", [])
    assert len(constraints) == 1
    assert constraints[0]["key"] == "tpu-name"
    assert constraints[0]["value"]["stringValue"] == "my-tpu"
    # Verify coscheduling
    assert returned_request["coscheduling"]["groupBy"] == "tpu-name"


def test_get_job_status_returns_error_for_missing_job(client):
    """GetJobStatus RPC returns error for non-existent job."""
    resp = client.post(
        "/iris.cluster.ControllerService/GetJobStatus",
        json={"jobId": JobName.root("test-user", "nonexistent").to_wire()},
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
    autoscaler = Mock()
    autoscaler.get_status.return_value = vm_pb2.AutoscalerStatus(
        groups=[
            vm_pb2.ScaleGroupStatus(
                name="test-group",
                config=config_pb2.ScaleGroupConfig(
                    name="test-group",
                    buffer_slices=1,
                    max_slices=5,
                    resources=config_pb2.ScaleGroupResources(
                        device_type=config_pb2.ACCELERATOR_TYPE_TPU,
                        device_variant="v4-8",
                    ),
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
                availability_status="requesting",
                availability_reason="scale-up in progress",
                blocked_until=timestamp_to_proto(Timestamp.from_ms(0)),
            ),
        ],
        current_demand={"test-group": 3},
        last_evaluation=timestamp_to_proto(Timestamp.from_ms(1000)),
        recent_actions=[
            vm_pb2.AutoscalerAction(
                timestamp=timestamp_to_proto(Timestamp.from_ms(1000)),
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
    dashboard = ControllerDashboard(service_with_autoscaler, log_service=service_with_autoscaler._log_service)
    return TestClient(dashboard.app)


def test_get_autoscaler_status_returns_status_when_enabled(client_with_autoscaler):
    """GetAutoscalerStatus RPC returns full status when autoscaler is configured."""
    resp = rpc_post(client_with_autoscaler, "GetAutoscalerStatus")
    data = resp.get("status", {})

    # Verify groups data (RPC uses camelCase field names)
    assert len(data["groups"]) == 1
    group = data["groups"][0]
    assert group["name"] == "test-group"
    assert group["currentDemand"] == 3
    assert group["availabilityStatus"] == "requesting"
    assert group["availabilityReason"] == "scale-up in progress"

    # Verify demand tracking
    assert data["currentDemand"] == {"test-group": 3}
    # Timestamp fields are nested messages
    assert int(data["lastEvaluation"]["epochMs"]) == 1000

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
    assert group["config"]["resources"]["deviceVariant"] == "v4-8"


def test_pending_reason_uses_autoscaler_hint_for_scale_up(
    client_with_autoscaler,
    state,
    job_request,
    mock_autoscaler,
):
    """Pending jobs surface autoscaler scale-up wait hints in job/detail APIs."""
    submit_job(state, "pending-scale", job_request)

    task_id = JobName.root("test-user", "pending-scale").task(0).to_wire()
    mock_autoscaler.get_status.return_value = vm_pb2.AutoscalerStatus(
        last_routing_decision=vm_pb2.RoutingDecision(
            group_to_launch={"tpu_v5e_32": 1},
            routed_entries={
                "tpu_v5e_32": vm_pb2.DemandEntryStatusList(entries=[vm_pb2.DemandEntryStatus(task_ids=[task_id])])
            },
        )
    )

    job_resp = rpc_post(
        client_with_autoscaler, "GetJobStatus", {"jobId": JobName.root("test-user", "pending-scale").to_wire()}
    )
    pending_reason = job_resp.get("job", {}).get("pendingReason", "")
    assert "Waiting for worker scale-up in scale group 'tpu_v5e_32'" in pending_reason

    jobs_resp = rpc_post(client_with_autoscaler, "ListJobs")
    listed = [
        j for j in jobs_resp.get("jobs", []) if j.get("jobId") == JobName.root("test-user", "pending-scale").to_wire()
    ]
    assert listed
    assert "Waiting for worker scale-up in scale group 'tpu_v5e_32'" in listed[0].get("pendingReason", "")


def test_pending_reason_uses_passive_autoscaler_hint_over_scheduler(
    client_with_autoscaler,
    state,
    mock_autoscaler,
):
    """GetJobStatus should use autoscaler passive-wait hint even when no active launch."""
    register_worker(state, "w1", "h1:8080", make_worker_metadata())

    request = controller_pb2.Controller.LaunchJobRequest(
        name="diag-constraint",
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            job_pb2.Constraint(
                key="nonexistent-attr",
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="x"),
            )
        ],
    )
    submit_job(state, "diag-constraint", request)
    task_id = JobName.root("test-user", "diag-constraint").task(0).to_wire()

    mock_autoscaler.get_status.return_value = vm_pb2.AutoscalerStatus(
        last_routing_decision=vm_pb2.RoutingDecision(
            group_to_launch={"tpu_v5e_32": 0},
            routed_entries={
                "tpu_v5e_32": vm_pb2.DemandEntryStatusList(entries=[vm_pb2.DemandEntryStatus(task_ids=[task_id])])
            },
        )
    )

    job_resp = rpc_post(
        client_with_autoscaler, "GetJobStatus", {"jobId": JobName.root("test-user", "diag-constraint").to_wire()}
    )
    pending_reason = job_resp.get("job", {}).get("pendingReason", "")
    assert pending_reason
    assert "Waiting for workers in scale group 'tpu_v5e_32' to become ready" in pending_reason


def test_list_jobs_shows_passive_autoscaler_wait_hint(
    client_with_autoscaler,
    state,
    job_request,
    mock_autoscaler,
):
    """ListJobs should show passive autoscaler wait hints for pending jobs."""
    submit_job(state, "pending-no-launch", job_request)
    task_id = JobName.root("test-user", "pending-no-launch").task(0).to_wire()

    mock_autoscaler.get_status.return_value = vm_pb2.AutoscalerStatus(
        last_routing_decision=vm_pb2.RoutingDecision(
            group_to_launch={"tpu_v5e_32": 0},
            routed_entries={
                "tpu_v5e_32": vm_pb2.DemandEntryStatusList(entries=[vm_pb2.DemandEntryStatus(task_ids=[task_id])])
            },
        )
    )

    jobs_resp = rpc_post(client_with_autoscaler, "ListJobs")
    listed = [
        j
        for j in jobs_resp.get("jobs", [])
        if j.get("jobId") == JobName.root("test-user", "pending-no-launch").to_wire()
    ]
    assert listed
    assert "Waiting for workers in scale group 'tpu_v5e_32' to become ready" in listed[0].get("pendingReason", "")


# =============================================================================
# Health Endpoint Tests
# =============================================================================


def test_worker_detail_page_escapes_id(client):
    """Worker detail page escapes the ID to prevent XSS."""
    response = client.get('/worker/"onmouseover="alert(1)')
    assert response.status_code == 200
    assert "onmouseover" not in response.text or "&quot;" in response.text


def test_get_worker_status_recent_tasks_have_timestamps(client, state, job_request):
    """GetWorkerStatus returns recent_tasks with started_at populated from attempt timestamps."""
    wid = register_worker(state, "w1", "h1:8080", make_worker_metadata())
    job_id = submit_job(state, "ts-job", job_request)
    task_id = job_id.task(0)

    state.queue_assignments([Assignment(task_id=task_id, worker_id=wid)])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=wid,
            worker_resource_snapshot=None,
            updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
        )
    )
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=wid,
            worker_resource_snapshot=None,
            updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
        )
    )

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w1"})
    tasks = resp.get("recentTasks", [])
    assert len(tasks) == 1
    assert tasks[0]["taskId"] == task_id.to_wire()
    assert tasks[0].get("startedAt"), "started_at must be populated from attempt timestamps"
    assert tasks[0].get("finishedAt"), "finished_at must be populated from attempt timestamps"


def test_get_worker_status_by_worker_id(client, state):
    """GetWorkerStatus looks up purely by worker ID — no autoscaler cross-referencing."""
    register_worker(state, "w1", "10.0.0.5:8080", make_worker_metadata())

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w1"})
    assert resp.get("worker", {}).get("workerId") == "w1"
    assert resp.get("worker", {}).get("healthy") is True
    assert resp.get("worker", {}).get("address") == "10.0.0.5:8080"


def test_get_worker_status_includes_running_tasks_and_resource_history(client, state, job_request):
    """GetWorkerStatus assembles running tasks and resource history explicitly."""
    wid = register_worker(state, "w1", "10.0.0.5:8080", make_worker_metadata())
    job_id = submit_job(state, "worker-detail-res", job_request)
    task_id = job_id.task(0)
    state.queue_assignments([Assignment(task_id=task_id, worker_id=wid)])

    first = job_pb2.WorkerResourceSnapshot(host_cpu_percent=25, running_task_count=1)
    second = job_pb2.WorkerResourceSnapshot(host_cpu_percent=50, running_task_count=1)
    state.apply_task_updates(HeartbeatApplyRequest(worker_id=wid, worker_resource_snapshot=first, updates=[]))
    state.apply_task_updates(HeartbeatApplyRequest(worker_id=wid, worker_resource_snapshot=second, updates=[]))

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w1"})
    running_job_ids = resp.get("worker", {}).get("runningJobIds", [])
    assert task_id.to_wire() in running_job_ids
    assert [entry.get("hostCpuPercent") for entry in resp.get("resourceHistory", [])] == [25, 50]
    assert resp.get("currentResources", {}).get("hostCpuPercent") == 50


def test_get_worker_status_unknown_id_returns_error(client):
    """GetWorkerStatus returns 404 for unknown IDs (no VM fallback)."""
    resp = client.post(
        "/iris.cluster.ControllerService/GetWorkerStatus",
        json={"id": "nonexistent-vm-0"},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code != 200


def test_health_endpoint_returns_ok(client):
    """Health endpoint returns a trivial ok response without querying state."""
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# =============================================================================
# Task Logs Proxy Tests
# =============================================================================


def test_fetch_logs_for_missing_task_returns_empty_entries(client):
    """FetchLogs on LogService returns empty entries for a nonexistent task."""
    task_id = JobName.root("test-user", "nonexistent").task(0).to_wire()
    resp = client.post(
        "/iris.logging.LogService/FetchLogs",
        json={"source": re.escape(task_id) + ":.*"},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("entries", []) == []


def test_fetch_logs_backward_compat_proxy(client):
    """The old ControllerService/FetchLogs path proxies to LogService."""
    task_id = JobName.root("test-user", "nonexistent").task(0).to_wire()
    resp = client.post(
        "/iris.cluster.ControllerService/FetchLogs",
        json={"source": re.escape(task_id) + ":.*"},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("entries", []) == []


def test_fetch_logs_backward_compat_proxy_proto_binary(client):
    """Old clients using default Connect proto encoding hit the compat endpoint."""
    from iris.rpc import logging_pb2

    task_id = JobName.root("test-user", "nonexistent").task(0).to_wire()
    req = logging_pb2.FetchLogsRequest(source=re.escape(task_id) + ":.*")
    resp = client.post(
        "/iris.cluster.ControllerService/FetchLogs",
        content=req.SerializeToString(),
        headers={"Content-Type": "application/proto"},
    )
    assert resp.status_code == 200
    parsed = logging_pb2.FetchLogsResponse()
    parsed.ParseFromString(resp.content)
    assert list(parsed.entries) == []


# =============================================================================
# Coscheduling Diagnostic Tests
# =============================================================================


def test_coscheduling_failure_reason_no_workers(client, state):
    """Pending coscheduled job reports diagnostic reason when no workers match constraints.

    Diagnostics are on the job-level (via GetJobStatus), not per-task in ListTasks.
    """
    request = controller_pb2.Controller.LaunchJobRequest(
        name="cosched-job",
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=2,
        environment=job_pb2.EnvironmentConfig(),
        constraints=[
            job_pb2.Constraint(
                key=WellKnownAttribute.TPU_NAME,
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="nonexistent-tpu"),
            ),
        ],
        coscheduling=job_pb2.CoschedulingConfig(group_by=WellKnownAttribute.TPU_NAME),
    )
    submit_job(state, "cosched-job", request)

    resp = rpc_post(client, "GetJobStatus", {"jobId": JobName.root("test-user", "cosched-job").to_wire()})
    job = resp.get("job", {})
    reason = job.get("pendingReason", "")
    assert "no workers match constraints" in reason.lower(), f"Expected constraint failure reason, got: {reason}"


def test_coscheduling_failure_reason_insufficient_group(client, state):
    """Pending coscheduled job reports diagnostic when group is too small.

    Diagnostics are on the job-level (via GetJobStatus), not per-task in ListTasks.
    """
    # Register 2 workers with tpu-name=my-tpu
    for i in range(2):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].CopyFrom(job_pb2.AttributeValue(string_value="my-tpu"))
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].CopyFrom(job_pb2.AttributeValue(int_value=i))
        register_worker(state, f"w{i}", f"h{i}:8080", meta)

    # Submit a coscheduled job needing 4 replicas
    request = controller_pb2.Controller.LaunchJobRequest(
        name="big-cosched",
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        constraints=[
            job_pb2.Constraint(
                key=WellKnownAttribute.TPU_NAME,
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="my-tpu"),
            ),
        ],
        coscheduling=job_pb2.CoschedulingConfig(group_by=WellKnownAttribute.TPU_NAME),
    )
    submit_job(state, "big-cosched", request)

    resp = rpc_post(client, "GetJobStatus", {"jobId": JobName.root("test-user", "big-cosched").to_wire()})
    job = resp.get("job", {})
    reason = job.get("pendingReason", "")
    assert "need 4" in reason, f"Expected 'need 4' in reason, got: {reason}"
    assert "largest group has 2" in reason, f"Expected 'largest group has 2' in reason, got: {reason}"


# =============================================================================
# Worker Attributes Tests
# =============================================================================


def test_worker_attributes_in_list_workers(client, state):
    """ListWorkers RPC returns worker attributes in metadata."""
    meta = make_worker_metadata()
    meta.attributes[WellKnownAttribute.TPU_NAME].CopyFrom(job_pb2.AttributeValue(string_value="v5litepod-16"))
    meta.attributes[WellKnownAttribute.TPU_WORKER_ID].CopyFrom(job_pb2.AttributeValue(int_value=0))
    register_worker(state, "tpu-worker", "h1:8080", meta)

    resp = rpc_post(client, "ListWorkers")
    workers = resp.get("workers", [])
    assert len(workers) == 1

    attrs = workers[0].get("metadata", {}).get("attributes", {})
    assert attrs["tpu-name"]["stringValue"] == "v5litepod-16"
    assert int(attrs["tpu-worker-id"]["intValue"]) == 0


# =============================================================================
# Pagination / Many Jobs Tests
# =============================================================================


def test_list_jobs_returns_all_jobs_for_pagination(client, state):
    """ListJobs RPC returns all jobs even with many entries (pagination is client-side)."""
    for i in range(60):
        request = controller_pb2.Controller.LaunchJobRequest(
            name=f"job-{i:03d}",
            entrypoint=make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
        )
        submit_job(state, f"job-{i:03d}", request)

    resp = rpc_post(client, "ListJobs")
    jobs = resp.get("jobs", [])
    assert len(jobs) == 60


def test_bundle_download_route_serves_bundle_bytes(client, service):
    bundle_id = "a" * 64
    bundle_bytes = b"zip-bytes"
    service.bundle_zip = Mock(return_value=bundle_bytes)

    resp = client.get(f"/bundles/{bundle_id}.zip")
    assert resp.status_code == 200
    assert resp.content == bundle_bytes
    assert resp.headers["content-type"] == "application/zip"


# =============================================================================
# Auth Config Endpoint Tests
# =============================================================================


def test_auth_config_returns_disabled_by_default(client):
    """Auth config endpoint reports auth disabled when no verifier is configured."""
    resp = client.get("/auth/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["auth_enabled"] is False
    assert data["provider"] is None


def test_auth_config_returns_enabled_when_verifier_set(service):
    """Auth config endpoint reports auth enabled with provider name."""
    from iris.rpc.auth import StaticTokenVerifier

    verifier = StaticTokenVerifier({"test-token": "test-user"})
    dashboard = ControllerDashboard(
        service, log_service=service._log_service, auth_verifier=verifier, auth_provider="gcp"
    )
    authed_client = TestClient(dashboard.app)

    resp = authed_client.get("/auth/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["auth_enabled"] is True
    assert data["provider"] == "gcp"


def test_auth_config_worker_provider_kind(client):
    """auth/config returns provider_kind=worker when no direct provider."""
    resp = client.get("/auth/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider_kind"] == "worker"


def test_auth_config_kubernetes_provider_kind(state, scheduler, tmp_path):
    """auth/config returns provider_kind=kubernetes when controller has direct provider."""
    controller_mock = _make_controller_mock(state, scheduler)
    controller_mock.has_direct_provider = True
    log_service = LogServiceImpl()
    svc = ControllerServiceImpl(
        state,
        state._db,
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=log_service,
    )
    dashboard = ControllerDashboard(svc, log_service=svc._log_service)
    k8s_client = TestClient(dashboard.app)

    resp = k8s_client.get("/auth/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider_kind"] == "kubernetes"


# =============================================================================
# Kubernetes Cluster Status RPC
# =============================================================================


def _make_k8s_dashboard_client(state, scheduler, tmp_path):
    """Build a TestClient wired to a real K8sTaskProvider backed by InMemoryK8sService."""
    from iris.cluster.providers.k8s.fake import InMemoryK8sService
    from iris.cluster.providers.k8s.tasks import K8sTaskProvider

    k8s = InMemoryK8sService(namespace="iris")
    provider = K8sTaskProvider(kubectl=k8s, namespace="iris", default_image="img:latest")
    controller_mock = _make_controller_mock(state, scheduler)
    controller_mock.has_direct_provider = True
    controller_mock.provider = provider
    log_service = LogServiceImpl()
    svc = ControllerServiceImpl(
        state,
        state._db,
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=log_service,
    )
    dashboard = ControllerDashboard(svc, log_service=svc._log_service)
    return TestClient(dashboard.app), k8s, provider


def test_k8s_cluster_status_returns_nodes_and_pods(state, scheduler, tmp_path):
    """GetKubernetesClusterStatus returns node capacity and pod statuses after sync."""
    from iris.cluster.controller.transitions import DirectProviderBatch
    from iris.cluster.providers.k8s.tasks import _LABEL_MANAGED, _LABEL_RUNTIME, _RUNTIME_LABEL_VALUE

    client, k8s, provider = _make_k8s_dashboard_client(state, scheduler, tmp_path)

    # Seed nodes and a pod.
    k8s.seed_resource(
        K8sResource.NODES,
        "node-1",
        {
            "kind": "Node",
            "metadata": {"name": "node-1"},
            "spec": {"taints": []},
            "status": {"allocatable": {"cpu": "8", "memory": "16Gi"}},
        },
    )
    k8s.seed_resource(
        K8sResource.PODS,
        "iris-task-0",
        {
            "kind": "Pod",
            "metadata": {
                "name": "iris-task-0",
                "labels": {
                    _LABEL_MANAGED: "true",
                    _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
                    "iris.task_id": "job.0",
                },
            },
            "status": {"phase": "Running"},
        },
    )

    # Sync to populate ClusterState.
    provider.sync(DirectProviderBatch(tasks_to_run=[], running_tasks=[], tasks_to_kill=[]))

    resp = client.post(
        "/iris.cluster.ControllerService/GetKubernetesClusterStatus",
        json={},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["namespace"] == "iris"
    assert data["totalNodes"] == 1
    assert data["schedulableNodes"] == 1
    assert "cores" in data["allocatableCpu"]
    assert len(data["podStatuses"]) == 1
    assert data["podStatuses"][0]["podName"] == "iris-task-0"
    assert data["podStatuses"][0]["phase"] == "Running"

    provider.close()


def test_k8s_cluster_status_empty_before_sync(state, scheduler, tmp_path):
    """GetKubernetesClusterStatus returns empty data when no sync has run yet."""
    client, _k8s, provider = _make_k8s_dashboard_client(state, scheduler, tmp_path)

    resp = client.post(
        "/iris.cluster.ControllerService/GetKubernetesClusterStatus",
        json={},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("totalNodes", 0) == 0
    assert data.get("podStatuses", []) == []

    provider.close()


def test_k8s_cluster_status_without_direct_provider(client):
    """GetKubernetesClusterStatus returns empty response when no K8s provider is configured."""
    resp = client.post(
        "/iris.cluster.ControllerService/GetKubernetesClusterStatus",
        json={},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("totalNodes", 0) == 0
