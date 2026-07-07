# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller dashboard behavioral logic.

Tests verify dashboard functionality through the Connect RPC endpoints.
The dashboard serves a web UI that fetches data via RPC calls.
"""

from unittest.mock import Mock

import pytest
from iris.cluster.backends.k8s.tasks import (
    _KUEUE_POD_GROUP_NAME,
    _KUEUE_QUEUE_NAME,
    _LABEL_MANAGED,
    _LABEL_RUNTIME,
    _RUNTIME_LABEL_VALUE,
    K8sTaskProvider,
)
from iris.cluster.backends.rpc.backend import RpcTaskBackend
from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller import ops, reads
from iris.cluster.controller.autoscaler.status import PendingHint, overlay_worker_usability
from iris.cluster.controller.backend import BackendCapability, BackendRuntime
from iris.cluster.controller.codec import constraints_from_json, device_counts_from_json, device_variant_from_json
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.projections.endpoints import EndpointRow
from iris.cluster.controller.reads import ControlSnapshot, healthy_active_workers_with_attributes
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.scheduling.scheduler import (
    DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
    DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
    JobRequirements,
    Scheduler,
    SchedulingContext,
    worker_snapshot_from_row,
)
from iris.cluster.controller.schema import jobs_table, task_attempts_table, tasks_table
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.types import DEFAULT_BACKEND_ID, JobName, UserBudgetDefaults, WorkerId, WorkerUsability
from iris.rpc import controller_pb2, job_pb2, vm_pb2
from iris.time_proto import timestamp_to_proto
from rigging.server_auth import RequestAuthPolicy
from rigging.testing import MockVerifier
from rigging.timing import Timestamp
from sqlalchemy import func, insert, select
from sqlalchemy import update as sa_update
from starlette.testclient import TestClient
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations

from .conftest import (
    check_task_can_be_scheduled,
    make_test_entrypoint,
    make_worker_metadata,
    register_worker,
)
from .conftest import (
    query_tasks_with_attempts as _query_tasks_with_attempts,
)

# =============================================================================
# Test Helpers
# =============================================================================


def submit_job(
    state: ControllerTestState,
    job_id: str,
    request: controller_pb2.Controller.LaunchJobRequest,
) -> JobName:
    """Submit a job through the state command API."""
    jid = JobName.from_string(job_id) if job_id.startswith("/") else JobName.root("test-user", job_id)
    request.name = jid.to_wire()
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=jid, request=request, ts=Timestamp.now())
    return jid


def set_job_state(
    state: ControllerTestState, job_id: JobName, new_state: int, *, started_at_ms: int | None = None
) -> None:
    """Directly set job state in DB for dashboard-only read-model tests."""
    values: dict = {"state": new_state}
    if started_at_ms is not None:
        values["started_at_ms"] = Timestamp.from_ms(started_at_ms)
    with state._db.transaction() as tx:
        tx.execute(sa_update(jobs_table).where(jobs_table.c.job_id == job_id).values(**values))


def set_task_retry_counts(
    state: ControllerTestState,
    task_id: JobName,
    *,
    failure_count: int | None = None,
    preemption_count: int | None = None,
) -> None:
    """Give ``task_id`` an attempt history yielding the requested derived counts.

    Retry counters are derived from ``task_attempts`` (there are no stored count
    columns), so this appends terminal attempt rows: ``failure_count`` FAILED
    attempts and ``preemption_count`` executing-phase WORKER_FAILED attempts
    (``started_at`` set marks the executing phase that charges the preemption
    budget).
    """
    fc = failure_count or 0
    pc = preemption_count or 0
    if fc == 0 and pc == 0:
        return
    with state._db.transaction() as tx:
        next_id = int(
            tx.execute(
                select(func.coalesce(func.max(task_attempts_table.c.attempt_id), -1)).where(
                    task_attempts_table.c.task_id == task_id
                )
            ).scalar()
            or -1
        )
        rows: list[dict] = []
        for state_value, count in ((job_pb2.TASK_STATE_FAILED, fc), (job_pb2.TASK_STATE_WORKER_FAILED, pc)):
            for _ in range(count):
                next_id += 1
                rows.append(
                    {
                        "task_id": task_id,
                        "attempt_id": next_id,
                        "worker_id": None,
                        "state": state_value,
                        "created_at_ms": 0,
                        "started_at_ms": 1,
                        "finished_at_ms": 2,
                        "attempt_uid": f"{task_id.to_wire()}:{next_id}",
                    }
                )
        tx.execute(insert(task_attempts_table), rows)


def set_task_state(state: ControllerTestState, task_id: JobName, new_state: int) -> None:
    """Directly set task state in DB for aggregate count tests."""
    with state._db.transaction() as tx:
        tx.execute(sa_update(tasks_table).where(tasks_table.c.task_id == task_id).values(state=new_state))


@pytest.fixture
def scheduler():
    return Scheduler()


def _worker_backend(state, autoscaler, backend_id=DEFAULT_BACKEND_ID):
    """A real ``RpcTaskBackend`` bound to the test DB and shared liveness tracker.

    It authors its own ``status()`` / ``autoscaler_status()`` exactly as in
    production — health counts from its tracker, per-VM usability + running-task
    overlay from its own state, groups tagged with its own ``backend_id`` — so the
    controller reads the result verbatim. The stub factory is unused by the status
    paths, so a bare ``Mock`` suffices."""
    backend = RpcTaskBackend(stub_factory=Mock())
    backend.health = state._health
    backend.autoscaler = autoscaler
    backend.bind_runtime(
        BackendRuntime(
            backend_id=backend_id,
            db=state._db,
            owns_scale_group=lambda _scale_group: True,
            budget_defaults=UserBudgetDefaults(),
        )
    )
    return backend


def _make_controller_mock(state, scheduler, autoscaler=None):
    """Build a mock that implements the ControllerProtocol for testing.

    Computes scheduling diagnostics on the fly when the service asks, mirroring
    how the real controller caches diagnostics per scheduling cycle. The
    on-the-fly path constructs a fresh ``SchedulingContext`` from the test DB
    state — the raw-read fields are not consumed by ``get_job_scheduling_diagnostics``
    so they are passed empty.
    """

    def _build_diagnostics_context():
        with state._db.read_snapshot() as tx:
            bc_rows = tx.execute(
                select(task_attempts_table.c.worker_id, func.count().label("c"))
                .join(
                    tasks_table,
                    (tasks_table.c.task_id == task_attempts_table.c.task_id)
                    & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
                )
                .where(tasks_table.c.state.in_([job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_ASSIGNED]))
                .group_by(task_attempts_table.c.worker_id)
                .order_by(task_attempts_table.c.worker_id.asc())
            ).all()
            building_counts = {row.worker_id: int(row.c) for row in bc_rows}
            usage_by_worker = reads.resource_usage_by_worker(tx)
            workers = healthy_active_workers_with_attributes(tx, state._health, state._worker_attrs)
        snapshots = [worker_snapshot_from_row(w, usage_by_worker.get(w.worker_id)) for w in workers]
        return SchedulingContext(
            workers=snapshots,
            building_counts=building_counts,
            max_building_tasks=DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
            max_assignments_per_worker=DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
            pending_tasks=[],
            jobs={},
            pending_task_rows=[],
            user_spend={},
            user_budget_limits={},
            requested_bands={},
            user_budget_defaults=UserBudgetDefaults(),
        )

    def _get_job_scheduling_diagnostics(job_wire_id):
        """Compute diagnostics on the fly for tests (mirrors real controller cache)."""
        job_id = JobName.from_wire(job_wire_id)
        with state._db.read_snapshot() as tx:
            job = reads.get_job_detail(tx, job_id)
        if job is None:
            return None
        if job.state != job_pb2.JOB_STATE_PENDING:
            return None
        dc = device_counts_from_json(job.res_device_json)
        req = JobRequirements(
            req_cpu_millicores=job.res_cpu_millicores,
            req_memory_bytes=job.res_memory_bytes,
            req_gpu_count=dc.gpu,
            req_tpu_count=dc.tpu,
            device_variant=device_variant_from_json(job.res_device_json),
            constraints=constraints_from_json(job.constraints_json),
            is_coscheduled=job.has_coscheduling,
            coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
        )
        tasks = _query_tasks_with_attempts(state, job.job_id)
        schedulable_task_id = next((t.task_id for t in tasks if check_task_can_be_scheduled(t)), None)
        context = _build_diagnostics_context()
        return scheduler.get_job_scheduling_diagnostics(req, context, schedulable_task_id, num_tasks=len(tasks))

    controller_mock = Mock()
    controller_mock.wake = Mock()
    controller_mock.get_job_scheduling_diagnostics = _get_job_scheduling_diagnostics
    controller_mock.last_scheduling_context = None
    worker_caps = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})
    controller_mock.provider = Mock(capabilities=worker_caps)
    controller_mock.provider.name = "worker"
    controller_mock.provider.autoscaler = autoscaler
    # The single backend owns the liveness tracker; the service reads liveness through
    # the controller's union over the backends' trackers.
    controller_mock.provider.health = state._health
    # status()/autoscaler_status() are delegated to a real backend bound to the same
    # DB + tracker, so the provider authors them exactly as production — the service
    # overlays nothing on top.
    _authoring_backend = _worker_backend(state, autoscaler)
    controller_mock.provider.autoscaler_status.side_effect = _authoring_backend.autoscaler_status
    controller_mock.provider.status.side_effect = _authoring_backend.status
    controller_mock.capabilities = worker_caps
    controller_mock.backends = {DEFAULT_BACKEND_ID: controller_mock.provider}
    controller_mock.backend_id_for_scale_group = Mock(return_value=DEFAULT_BACKEND_ID)
    controller_mock.all_liveness = lambda: state._health.all()
    controller_mock.liveness_for_worker = lambda wid: state._health.liveness(wid)
    controller_mock.last_unroutable_jobs = {}
    controller_mock.scale_group_to_backend = {}
    return controller_mock


@pytest.fixture
def service(state, scheduler, tmp_path, embedded_log_server, log_client):
    controller_mock = _make_controller_mock(state, scheduler)
    return ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(
            db=state._db,
            system_endpoints={"/system/log-server": embedded_log_server.address},
        ),
    )


@pytest.fixture
def client(service):
    dashboard = ControllerDashboard(service)
    return TestClient(dashboard.app)


@pytest.fixture
def service_with_autoscaler(state, scheduler, mock_autoscaler, tmp_path, log_client):
    controller_mock = _make_controller_mock(state, scheduler, autoscaler=mock_autoscaler)
    return ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
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
    with state._db.transaction() as cur:
        state._endpoints.add(
            cur,
            EndpointRow(
                endpoint_id="ep1",
                name="pending-svc",
                address="h:1",
                task_id=pending_id.task(0),
                metadata={},
                registered_at=Timestamp.now(),
            ),
        )
    with state._db.transaction() as cur:
        state._endpoints.add(
            cur,
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
    with state._db.transaction() as cur:
        state._endpoints.add(
            cur,
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


def test_list_endpoints_filters_by_task_ids(client, state):
    """ListEndpoints(task_ids=[...]) returns only endpoints owned by those tasks.

    The dashboard's task list and detail pages use this to render a proxy link
    per task without scanning every endpoint in the cluster.
    """
    request = controller_pb2.Controller.LaunchJobRequest(
        name="multi-ep-job",
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=2,
        environment=job_pb2.EnvironmentConfig(),
    )
    job_id = submit_job(state, "multi-ep", request)
    set_job_state(state, job_id, job_pb2.JOB_STATE_RUNNING)

    task0, task1 = job_id.task(0), job_id.task(1)
    with state._db.transaction() as cur:
        for endpoint_id, task in (("ep-0", task0), ("ep-1", task1)):
            state._endpoints.add(
                cur,
                EndpointRow(
                    endpoint_id=endpoint_id,
                    name=f"/svc/{endpoint_id}",
                    address="h:1",
                    task_id=task,
                    metadata={},
                    registered_at=Timestamp.now(),
                ),
            )

    resp = rpc_post(client, "ListEndpoints", {"taskIds": [task0.to_wire()]})
    endpoints = resp.get("endpoints", [])
    assert [e["taskId"] for e in endpoints] == [task0.to_wire()]
    assert endpoints[0]["name"] == "/svc/ep-0"


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
            setup_scripts=["uv sync\n"],
            env_vars={"MY_FLAG": "1"},
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
    assert env["setupScripts"] == ["uv sync\n"]
    assert env["envVars"] == {"MY_FLAG": "1"}
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
    autoscaler.get_pending_hints.return_value = {}
    autoscaler.get_status.return_value = vm_pb2.AutoscalerStatus(
        groups=[
            vm_pb2.ScaleGroupStatus(
                name="test-group",
                device_type="tpu",
                device_variant="v4-8",
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
    dashboard = ControllerDashboard(service_with_autoscaler)
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
    assert group["deviceVariant"] == "v4-8"


def test_get_autoscaler_status_populates_worker_id_for_unrostered_vm(client_with_autoscaler):
    """worker_id (and the running-task lookup) must be populated for every VM in the
    status, even one absent from the liveness roster — otherwise its running tasks
    silently drop out of the dashboard. The mock VMs are not registered workers."""
    resp = rpc_post(client_with_autoscaler, "GetAutoscalerStatus")
    vms = [vm for group in resp["status"]["groups"] for s in group["slices"] for vm in s["vms"]]
    assert vms
    # vm_id IS the worker_id; it is set unconditionally now (previously skipped
    # when the VM was missing from the roster).
    for vm in vms:
        assert vm["workerId"] == vm["vmId"]


def test_overlay_worker_usability_tags_vms_and_per_slice_degraded_count():
    """The overlay tags each VM with usability/worker_healthy/running_task_count and
    records the per-slice count of degraded (reachable-but-failing) hosts."""
    status = vm_pb2.AutoscalerStatus(
        groups=[
            vm_pb2.ScaleGroupStatus(
                name="g",
                slices=[
                    vm_pb2.SliceInfo(
                        slice_id="s1",
                        state="ready",
                        vms=[vm_pb2.VmInfo(vm_id="w-healthy"), vm_pb2.VmInfo(vm_id="w-degraded")],
                    ),
                    vm_pb2.SliceInfo(slice_id="s2", state="ready", vms=[vm_pb2.VmInfo(vm_id="w-unrostered")]),
                ],
            )
        ]
    )
    usability_by_id = {
        "w-healthy": WorkerUsability.HEALTHY,
        "w-degraded": WorkerUsability.DEGRADED,
        # "w-unrostered" intentionally absent from the roster.
    }
    running = {WorkerId("w-healthy"): {"task-1"}}

    overlay_worker_usability(status, usability_by_id, running)

    group = status.groups[0]
    s1, s2 = group.slices
    by_id = {vm.vm_id: vm for vm in (*s1.vms, *s2.vms)}
    assert by_id["w-healthy"].usability == "healthy"
    assert by_id["w-healthy"].worker_healthy is True
    assert by_id["w-healthy"].running_task_count == 1
    assert by_id["w-degraded"].usability == "degraded"
    assert by_id["w-degraded"].worker_healthy is True
    # An unrostered VM keeps worker_id/task count but is left unclassified.
    assert by_id["w-unrostered"].worker_id == "w-unrostered"
    assert by_id["w-unrostered"].usability == ""

    assert s1.degraded_slot_count == 1
    assert s2.degraded_slot_count == 0


def test_overlay_capacity_status_busy_healthy_slice_is_in_use():
    """Regression for '40 schedulable' on fully booked slices: a healthy slice that
    is running tasks is `in_use`, never counted as free/schedulable capacity."""
    status = vm_pb2.AutoscalerStatus(
        groups=[
            vm_pb2.ScaleGroupStatus(
                name="g",
                slices=[
                    vm_pb2.SliceInfo(
                        slice_id="s",
                        state="ready",
                        vms=[vm_pb2.VmInfo(vm_id="a"), vm_pb2.VmInfo(vm_id="b")],
                    )
                ],
            )
        ]
    )
    usability = {"a": WorkerUsability.HEALTHY, "b": WorkerUsability.HEALTHY}
    overlay_worker_usability(status, usability, {WorkerId("a"): {"t1"}, WorkerId("b"): {"t2"}})

    assert status.groups[0].slices[0].capacity_status == "in_use"


def test_pending_reason_uses_autoscaler_hint_for_scale_up(
    client_with_autoscaler,
    state,
    job_request,
    mock_autoscaler,
):
    """Pending jobs surface autoscaler scale-up wait hints in job/detail APIs."""
    submit_job(state, "pending-scale", job_request)

    job_wire = JobName.root("test-user", "pending-scale").to_wire()
    mock_autoscaler.get_pending_hints.return_value = {
        job_wire: PendingHint(
            message="Waiting for worker scale-up in scale group 'tpu_v5e_32' (1 slice(s) requested)",
            is_scaling_up=True,
        )
    }

    # GetJobStatus appends this job's autoscaler hint via the per-cycle hint
    # cache (#4848) — a single dict lookup, no routing-table serialization.
    job_resp = rpc_post(
        client_with_autoscaler, "GetJobStatus", {"jobId": JobName.root("test-user", "pending-scale").to_wire()}
    )
    pending_reason = job_resp.get("job", {}).get("pendingReason", "")
    assert "Waiting for worker scale-up in scale group 'tpu_v5e_32'" in pending_reason
    assert "(scaling up)" in pending_reason

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
    job_wire = JobName.root("test-user", "diag-constraint").to_wire()

    mock_autoscaler.get_pending_hints.return_value = {
        job_wire: PendingHint(
            message="Waiting for workers in scale group 'tpu_v5e_32' to become ready",
            is_scaling_up=False,
        )
    }

    # GetJobStatus appends this job's autoscaler passive-wait hint.
    job_resp = rpc_post(
        client_with_autoscaler, "GetJobStatus", {"jobId": JobName.root("test-user", "diag-constraint").to_wire()}
    )
    pending_reason = job_resp.get("job", {}).get("pendingReason", "")
    assert "Waiting for workers in scale group 'tpu_v5e_32' to become ready" in pending_reason


def test_list_jobs_shows_passive_autoscaler_wait_hint(
    client_with_autoscaler,
    state,
    job_request,
    mock_autoscaler,
):
    """ListJobs should show passive autoscaler wait hints for pending jobs."""
    submit_job(state, "pending-no-launch", job_request)
    job_wire = JobName.root("test-user", "pending-no-launch").to_wire()

    mock_autoscaler.get_pending_hints.return_value = {
        job_wire: PendingHint(
            message="Waiting for workers in scale group 'tpu_v5e_32' to become ready",
            is_scaling_up=False,
        )
    }

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


def test_get_worker_status_recent_attempts_have_timestamps(client, state, job_request):
    """Verify GetWorkerStatus returns per-attempt rows with distinct
    timestamps, preserving retry history."""
    wid = register_worker(state, "w1", "h1:8080", make_worker_metadata())
    job_id = submit_job(state, "ts-job", job_request)
    task_id = job_id.task(0)

    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w1"})
    attempts = resp.get("recentAttempts", [])
    assert len(attempts) == 1
    assert attempts[0]["taskId"] == task_id.to_wire()
    attempt = attempts[0].get("attempt", {})
    assert attempt.get("attemptId") == 0
    assert attempt.get("state") == "TASK_STATE_SUCCEEDED"
    assert attempt.get("startedAt"), "started_at must be populated from attempt timestamps"
    assert attempt.get("finishedAt"), "finished_at must be populated from attempt timestamps"


def test_get_worker_status_recent_attempts_separates_retries(client, state):
    """Two attempts of the same task on the same worker get two distinct rows
    with per-attempt state. Regression for the dashboard rendering bug where
    one task with multiple attempts on a worker showed up as N duplicate
    'RUNNING' rows because the server returned per-task entries that the UI
    rendered with the parent task's state."""
    wid = register_worker(state, "w1", "h1:8080", make_worker_metadata())
    # Need preemption budget so the first WORKER_FAILED retries instead of
    # killing the job; otherwise the second attempt's heartbeat is dropped.
    request = controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "retry-job").to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=4 * 1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        max_retries_preemption=2,
    )
    job_id = submit_job(state, "retry-job", request)
    task_id = job_id.task(0)

    # First attempt: BUILDING -> WORKER_FAILED (retriable, retries to PENDING).
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[
                        TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_BUILDING),
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[
                        TaskUpdate(
                            task_id=task_id,
                            attempt_id=0,
                            new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                            error="TPU init failure",
                        ),
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )
    # Second attempt: re-dispatch to the same worker, RUNNING.
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=1, new_state=job_pb2.TASK_STATE_RUNNING)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w1"})
    attempts = resp.get("recentAttempts", [])
    assert len(attempts) == 2, f"expected one row per attempt, got {len(attempts)}: {attempts}"
    by_attempt_id = {a["attempt"]["attemptId"]: a for a in attempts}
    assert by_attempt_id[0]["attempt"]["state"] == "TASK_STATE_WORKER_FAILED"
    assert by_attempt_id[1]["attempt"]["state"] == "TASK_STATE_RUNNING"
    assert all(a["taskId"] == task_id.to_wire() for a in attempts)


def test_get_worker_status_recent_attempts_carry_attempt_uid(client, state, job_request):
    """GetWorkerStatus per-attempt rows surface the controller-minted
    attempt_uid for operator traceability.

    Covers ``_attempts_for_worker``: the projection reads ``attempt_uid``
    from ``ATTEMPT_COLS`` and stamps it onto each ``TaskAttempt`` proto. The
    UID is minted by ``insert_attempt`` when the attempt row is placed, so it
    must be a non-empty 16-hex-char string on every attempt.
    """
    wid = register_worker(state, "w1", "h1:8080", make_worker_metadata())
    job_id = submit_job(state, "uid-worker-job", job_request)
    task_id = job_id.task(0)
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    # The minted UID lives on the attempt row; read it directly to compare.
    with state._db.read_snapshot() as tx:
        db_uid = tx.execute(
            select(task_attempts_table.c.attempt_uid).where(
                task_attempts_table.c.task_id == task_id,
                task_attempts_table.c.attempt_id == 0,
            )
        ).scalar_one()
    assert len(db_uid) == 16 and all(c in "0123456789abcdef" for c in db_uid)

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w1"})
    attempts = resp.get("recentAttempts", [])
    assert len(attempts) == 1
    attempt = attempts[0]["attempt"]
    assert attempt.get("attemptUid") == db_uid


def test_get_task_status_attempts_carry_attempt_uid(client, state, job_request):
    """GetTaskStatus attempts surface attempt_uid via ``task_to_proto``.

    Each retry mints its own UID, so a task with two attempts yields two
    ``TaskAttempt`` protos with distinct, non-empty UIDs matching the rows.
    """
    wid = register_worker(state, "w1", "h1:8080", make_worker_metadata())
    request = controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "uid-task-job").to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=4 * 1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        max_retries_preemption=2,
    )
    job_id = submit_job(state, "uid-task-job", request)
    task_id = job_id.task(0)

    # Attempt 0: placed then WORKER_FAILED so it retries to a fresh attempt.
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_WORKER_FAILED)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )
    # Attempt 1: re-placed and RUNNING.
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=1, new_state=job_pb2.TASK_STATE_RUNNING)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    with state._db.read_snapshot() as tx:
        db_uids = dict(
            tx.execute(
                select(task_attempts_table.c.attempt_id, task_attempts_table.c.attempt_uid).where(
                    task_attempts_table.c.task_id == task_id
                )
            ).all()
        )
    assert set(db_uids) == {0, 1}
    assert db_uids[0] != db_uids[1]

    resp = rpc_post(client, "GetTaskStatus", {"taskId": task_id.to_wire()})
    attempts = resp.get("task", resp).get("attempts", [])
    assert len(attempts) == 2
    proto_uids = {a["attemptId"]: a.get("attemptUid") for a in attempts}
    assert proto_uids == db_uids


def test_get_worker_status_by_worker_id(client, state):
    """GetWorkerStatus looks up purely by worker ID — no autoscaler cross-referencing."""
    register_worker(state, "w1", "10.0.0.5:8080", make_worker_metadata())

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w1"})
    assert resp.get("worker", {}).get("workerId") == "w1"
    assert resp.get("worker", {}).get("healthy") is True
    assert resp.get("worker", {}).get("address") == "10.0.0.5:8080"


def test_get_worker_status_includes_running_tasks(client, state, job_request):
    """GetWorkerStatus assembles running tasks for the worker.

    Per-tick resource history is populated from the ``iris.worker`` stats
    namespace, not the controller DB; this test covers only DB-backed
    fields.
    """
    wid = register_worker(state, "w1", "10.0.0.5:8080", make_worker_metadata())
    job_id = submit_job(state, "worker-detail-res", job_request)
    task_id = job_id.task(0)
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)

    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [WorkerTaskUpdates(worker_id=wid, updates=[])],
            health=state._health,
            now=Timestamp.now(),
        )

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w1"})
    running_job_ids = resp.get("worker", {}).get("runningJobIds", [])
    assert task_id.to_wire() in running_job_ids
    assert "resourceHistory" not in resp
    assert "currentResources" not in resp


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
    """The endpoint proxy forwards FetchLogs to the registered log server."""
    task_id = JobName.root("test-user", "nonexistent").task(0).to_wire()
    resp = client.post(
        "/proxy/system.log-server/finelog.logging.LogService/FetchLogs",
        json={"source": f"{task_id}:", "match_scope": "MATCH_SCOPE_PREFIX"},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("entries", []) == []


@pytest.mark.parametrize(
    "path",
    [
        "/iris.cluster.ControllerService/FetchLogs",
        "/iris.logging.LogService/FetchLogs",
        "/finelog.logging.LogService/FetchLogs",
    ],
)
def test_fetch_logs_outside_endpoint_proxy_is_not_exposed(client, path):
    resp = client.post(path, json={}, headers={"Content-Type": "application/json"})

    assert resp.status_code == 404


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
    verifier = MockVerifier({"test-token": "test-user"})
    dashboard = ControllerDashboard(
        service,
        auth_provider="iap",
        auth_policy=RequestAuthPolicy.enforcing(verifier=verifier),
    )
    authed_client = TestClient(dashboard.app)

    resp = authed_client.get("/auth/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["auth_enabled"] is True
    assert data["provider"] == "iap"


def test_auth_config_worker_capabilities(client):
    """auth/config advertises worker + autoscaler capabilities for an Iris backend."""
    resp = client.get("/auth/config")
    assert resp.status_code == 200
    backend = resp.json()["backend"]
    assert backend["name"] == "worker"
    assert "placement" not in backend
    assert "manages_capacity" not in backend
    assert "workers" in backend["capabilities"]
    assert "autoscaler" in backend["capabilities"]
    assert "cluster" not in backend["capabilities"]


def test_auth_config_kubernetes_capabilities(state, scheduler, tmp_path, log_client):
    """auth/config advertises the cluster capability for a backend-placed (k8s) backend."""
    controller_mock = _make_controller_mock(state, scheduler)
    cluster_caps = frozenset({BackendCapability.CLUSTER_VIEW})
    controller_mock.capabilities = cluster_caps
    controller_mock.provider = Mock(capabilities=cluster_caps)
    controller_mock.provider.name = "kubernetes"
    controller_mock.backends = {DEFAULT_BACKEND_ID: controller_mock.provider}
    svc = ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )
    dashboard = ControllerDashboard(svc)
    k8s_client = TestClient(dashboard.app)

    resp = k8s_client.get("/auth/config")
    assert resp.status_code == 200
    backend = resp.json()["backend"]
    assert backend["name"] == "kubernetes"
    assert "placement" not in backend
    assert "manages_capacity" not in backend
    assert "cluster" in backend["capabilities"]
    assert "workers" not in backend["capabilities"]
    assert "autoscaler" not in backend["capabilities"]


# =============================================================================
# Kubernetes Cluster Status RPC
# =============================================================================


def _make_k8s_dashboard_client(state, scheduler, tmp_path, log_client):
    """Build a TestClient wired to a real K8sTaskProvider backed by InMemoryK8sService."""
    k8s = InMemoryK8sService(namespace="iris")
    provider = K8sTaskProvider(kubectl=k8s, namespace="iris", default_image="img:latest", cluster_scan_interval=0.0)
    controller_mock = _make_controller_mock(state, scheduler)
    controller_mock.capabilities = frozenset({BackendCapability.CLUSTER_VIEW})
    controller_mock.provider = provider
    controller_mock.backends = {DEFAULT_BACKEND_ID: provider}
    svc = ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )
    dashboard = ControllerDashboard(svc)
    return TestClient(dashboard.app), k8s, provider


def test_k8s_cluster_status_returns_nodes_and_pods(state, scheduler, tmp_path, log_client):
    """GetKubernetesClusterStatus returns node capacity and pod statuses after sync."""
    client, k8s, provider = _make_k8s_dashboard_client(state, scheduler, tmp_path, log_client)

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

    # Reconcile to populate ClusterState.
    provider.sync(
        ControlSnapshot(
            worker_addresses={},
            reconcile_rows=[],
            timeout_rows=[],
        )
    )

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


def test_k8s_cluster_status_enriches_scheduling_gated_pods_with_kueue_workload(state, scheduler, tmp_path, log_client):
    """SchedulingGated pod statuses include Kueue admission diagnostics."""
    client, k8s, provider = _make_k8s_dashboard_client(state, scheduler, tmp_path, log_client)
    queue_name = "iris-local"
    provider.local_queue = queue_name
    pod_group = "iris-pg-test-0"
    workload_message = "gpu-quota-diagnostic-token"

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
                    _KUEUE_POD_GROUP_NAME: pod_group,
                    _KUEUE_QUEUE_NAME: queue_name,
                    "iris.task_id": "job.0",
                },
            },
            "spec": {"schedulingGates": [{"name": "kueue.x-k8s.io/admission"}]},
            "status": {
                "phase": "Pending",
                "conditions": [
                    {
                        "type": "PodScheduled",
                        "status": "False",
                        "reason": "SchedulingGated",
                        "message": "Scheduling is blocked due to non-empty scheduling gates",
                    }
                ],
            },
        },
    )
    k8s.seed_resource(
        K8sResource.WORKLOADS,
        pod_group,
        {
            "kind": "Workload",
            "metadata": {"name": pod_group},
            "spec": {"queueName": queue_name},
            "status": {
                "conditions": [
                    {
                        "type": "QuotaReserved",
                        "status": "False",
                        "reason": "Pending",
                        "message": workload_message,
                    }
                ]
            },
        },
    )

    provider.sync(ControlSnapshot(worker_addresses={}, reconcile_rows=[], timeout_rows=[]))

    resp = client.post(
        "/iris.cluster.ControllerService/GetKubernetesClusterStatus",
        json={},
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 200
    status = resp.json()["podStatuses"][0]
    assert status["reason"] == "SchedulingGated"
    assert pod_group in status["message"]
    assert queue_name in status["message"]
    assert workload_message in status["message"]

    provider.close()


def test_k8s_cluster_status_empty_before_sync(state, scheduler, tmp_path, log_client):
    """GetKubernetesClusterStatus returns empty data when no sync has run yet."""
    client, _k8s, provider = _make_k8s_dashboard_client(state, scheduler, tmp_path, log_client)

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


# =============================================================================
# Multi-backend RPC surface
# =============================================================================


def _backend_mock(name, capabilities, autoscaler=None, cluster_status=None, allowed_users=None, advertised=None):
    backend = Mock(capabilities=capabilities)
    backend.name = name
    backend.autoscaler = autoscaler
    backend.advertised_attributes.return_value = advertised if advertised is not None else {}
    backend.allowed_users = allowed_users if allowed_users is not None else frozenset({"*"})
    # status() authors the BackendStatus variant the backend's capability selects:
    # a cluster view returns ``kubernetes``; everything else returns ``worker``.
    if cluster_status is not None:
        backend.get_cluster_status.return_value = cluster_status
        backend.status.return_value = controller_pb2.Controller.BackendStatus(kubernetes=cluster_status)
    else:
        autoscaler_status = autoscaler.get_status.return_value if autoscaler is not None else vm_pb2.AutoscalerStatus()
        backend.status.return_value = controller_pb2.Controller.BackendStatus(
            worker=controller_pb2.Controller.WorkerFleetDetail(autoscaler=autoscaler_status)
        )

    # autoscaler_status() authors this backend's groups tagged with its own id (the
    # dict key is the backend_id in these tests), mirroring the real backend.
    def _autoscaler_status():
        status = autoscaler.get_status() if autoscaler is not None else vm_pb2.AutoscalerStatus()
        for group in status.groups:
            group.backend_id = name
        return status

    backend.autoscaler_status.side_effect = _autoscaler_status
    return backend


def _multi_backend_client(state, scheduler, tmp_path, log_client, backends):
    """A dashboard client whose controller fronts several backends (representative = first)."""
    controller_mock = _make_controller_mock(state, scheduler)
    controller_mock.backends = dict(backends)
    controller_mock.provider = next(iter(backends.values()))
    controller_mock.capabilities = frozenset(cap for backend in backends.values() for cap in backend.capabilities)
    svc = ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )
    return TestClient(ControllerDashboard(svc).app)


def _status_autoscaler(group_name):
    autoscaler = Mock()
    autoscaler.get_status.return_value = vm_pb2.AutoscalerStatus(
        groups=[vm_pb2.ScaleGroupStatus(name=group_name)],
        current_demand={group_name: 1},
        last_evaluation=timestamp_to_proto(Timestamp.from_ms(5)),
        last_routing_decision=vm_pb2.RoutingDecision(
            group_statuses=[vm_pb2.GroupRoutingStatus(group=group_name, decision="hold")],
        ),
    )
    return autoscaler


def test_auth_config_unions_capabilities_across_backends(state, scheduler, tmp_path, log_client):
    """/auth/config advertises the union of every backend's capabilities, not just the representative's."""
    client = _multi_backend_client(
        state,
        scheduler,
        tmp_path,
        log_client,
        {
            "gcp": _backend_mock("gcp", frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})),
            "eu-k8s": _backend_mock("eu-k8s", frozenset({BackendCapability.CLUSTER_VIEW})),
        },
    )
    config = client.get("/auth/config").json()
    assert set(config["capabilities"]) == {"workers", "autoscaler", "cluster"}
    assert {b["id"] for b in config["backends"]} == {"gcp", "eu-k8s"}


def test_get_autoscaler_status_merges_all_backends(state, scheduler, tmp_path, log_client):
    """GetAutoscalerStatus merges every backend's autoscaler and tags each group with its backend_id."""
    client = _multi_backend_client(
        state,
        scheduler,
        tmp_path,
        log_client,
        {
            "gcp": _backend_mock(
                "gcp", frozenset({BackendCapability.IRIS_AUTOSCALER}), autoscaler=_status_autoscaler("gcp-v5e")
            ),
            "cw": _backend_mock(
                "cw", frozenset({BackendCapability.IRIS_AUTOSCALER}), autoscaler=_status_autoscaler("cw-h100")
            ),
        },
    )
    status = rpc_post(client, "GetAutoscalerStatus")["status"]
    assert {g["name"]: g.get("backendId", "") for g in status["groups"]} == {"gcp-v5e": "gcp", "cw-h100": "cw"}
    # Each backend's routing decision folds into the merged decision (disjoint
    # groups), so the dashboard's pools table sees every backend's pools.
    assert {s["group"] for s in status["lastRoutingDecision"]["groupStatuses"]} == {"gcp-v5e", "cw-h100"}

    # backend_id drill-down restricts the merged view to one backend.
    scoped = rpc_post(client, "GetAutoscalerStatus", {"backendId": "gcp"})["status"]
    assert [g["name"] for g in scoped["groups"]] == ["gcp-v5e"]
    assert [s["group"] for s in scoped["lastRoutingDecision"]["groupStatuses"]] == ["gcp-v5e"]


def test_get_kubernetes_cluster_status_finds_non_representative_backend(state, scheduler, tmp_path, log_client):
    """GetKubernetesClusterStatus locates the CLUSTER_VIEW backend even when it is not the representative one."""
    cluster_status = controller_pb2.Controller.GetKubernetesClusterStatusResponse(namespace="eu", total_nodes=2)
    client = _multi_backend_client(
        state,
        scheduler,
        tmp_path,
        log_client,
        {
            "gcp": _backend_mock("gcp", frozenset({BackendCapability.WORKER_DAEMON})),
            "eu-k8s": _backend_mock(
                "eu-k8s", frozenset({BackendCapability.CLUSTER_VIEW}), cluster_status=cluster_status
            ),
        },
    )
    data = rpc_post(client, "GetKubernetesClusterStatus")
    assert data["namespace"] == "eu"
    assert data["totalNodes"] == 2


def test_task_backend_id_propagated_to_proto(client, state, job_request):
    """GetTaskStatus surfaces backend_id stamped on the tasks row."""
    job_id = submit_job(state, "backend-task-job", job_request)
    tasks = _query_tasks_with_attempts(state, job_id)
    task_id = tasks[0].task_id
    with state._db.transaction() as tx:
        tx.execute(sa_update(tasks_table).where(tasks_table.c.task_id == task_id).values(backend_id="gcp"))

    resp = rpc_post(client, "GetTaskStatus", {"taskId": task_id.to_wire()})
    assert resp["task"]["backendId"] == "gcp"


def test_job_backend_id_propagated_to_list_jobs(client, state, job_request):
    """ListJobs surfaces backend_id stamped on the jobs row."""
    job_id = submit_job(state, "backend-job", job_request)
    with state._db.transaction() as tx:
        tx.execute(sa_update(jobs_table).where(jobs_table.c.job_id == job_id).values(backend_id="gcp"))

    resp = rpc_post(client, "ListJobs")
    matching = [j for j in resp["jobs"] if j["jobId"] == job_id.to_wire()]
    assert len(matching) == 1
    assert matching[0]["backendId"] == "gcp"


def test_job_backend_id_propagated_to_get_job_status(client, state, job_request):
    """GetJobStatus surfaces backend_id stamped on the jobs row (job detail page)."""
    job_id = submit_job(state, "backend-detail-job", job_request)
    with state._db.transaction() as tx:
        tx.execute(sa_update(jobs_table).where(jobs_table.c.job_id == job_id).values(backend_id="gcp"))

    resp = rpc_post(client, "GetJobStatus", {"jobId": job_id.to_wire()})
    assert resp["job"]["backendId"] == "gcp"


def test_list_jobs_filters_by_backend_id(client, state, job_request):
    """ListJobs.query.backendId restricts results to jobs on that backend."""
    gcp_job_id = submit_job(state, "gcp-job", job_request)
    cw_job_id = submit_job(state, "cw-job", job_request)
    with state._db.transaction() as tx:
        tx.execute(sa_update(jobs_table).where(jobs_table.c.job_id == gcp_job_id).values(backend_id="gcp"))
        tx.execute(sa_update(jobs_table).where(jobs_table.c.job_id == cw_job_id).values(backend_id="cw"))

    resp_gcp = rpc_post(client, "ListJobs", {"query": {"backendId": "gcp"}})
    assert len(resp_gcp["jobs"]) == 1
    assert resp_gcp["jobs"][0]["backendId"] == "gcp"

    resp_cw = rpc_post(client, "ListJobs", {"query": {"backendId": "cw"}})
    assert len(resp_cw["jobs"]) == 1
    assert resp_cw["jobs"][0]["backendId"] == "cw"


def test_list_workers_stamps_backend_id_and_scale_group(state, scheduler, tmp_path, log_client, job_request):
    """ListWorkers stamps backend_id (resolved via backend_id_for_scale_group) and scale_group."""
    controller_mock = _make_controller_mock(state, scheduler)
    controller_mock.backend_id_for_scale_group = lambda sg: "gcp" if sg == "tpu-v5e" else DEFAULT_BACKEND_ID
    svc = ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )
    client = TestClient(ControllerDashboard(svc).app)

    register_worker(state, "w-tpu", "10.0.0.1", make_worker_metadata(), scale_group="tpu-v5e")

    resp = rpc_post(client, "ListWorkers")
    workers = resp["workers"]
    tpu_worker = next(w for w in workers if w["workerId"] == "w-tpu")
    assert tpu_worker["backendId"] == "gcp"
    assert tpu_worker["scaleGroup"] == "tpu-v5e"


def test_worker_backend_id_propagated_to_get_worker_status(state, scheduler, tmp_path, log_client):
    """GetWorkerStatus stamps backend_id (resolved via scale_group) and scale_group (worker detail page)."""
    controller_mock = _make_controller_mock(state, scheduler)
    controller_mock.backend_id_for_scale_group = lambda sg: "gcp" if sg == "tpu-v5e" else DEFAULT_BACKEND_ID
    svc = ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )
    client = TestClient(ControllerDashboard(svc).app)

    register_worker(state, "w-tpu", "10.0.0.1", make_worker_metadata(), scale_group="tpu-v5e")

    resp = rpc_post(client, "GetWorkerStatus", {"id": "w-tpu"})
    assert resp["worker"]["backendId"] == "gcp"
    assert resp["worker"]["scaleGroup"] == "tpu-v5e"


def test_list_workers_filters_by_backend_id(state, scheduler, tmp_path, log_client):
    """ListWorkers.query.backendId returns only workers whose scale_group maps to that backend."""
    controller_mock = _make_controller_mock(state, scheduler)
    controller_mock.backend_id_for_scale_group = lambda sg: {"tpu-v5e": "gcp", "h100": "cw"}.get(sg, DEFAULT_BACKEND_ID)
    svc = ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )
    client = TestClient(ControllerDashboard(svc).app)

    register_worker(state, "w-gcp", "10.0.0.1", make_worker_metadata(), scale_group="tpu-v5e")
    register_worker(state, "w-cw", "10.0.0.2", make_worker_metadata(), scale_group="h100")

    resp_gcp = rpc_post(client, "ListWorkers", {"query": {"backendId": "gcp"}})
    assert [w["workerId"] for w in resp_gcp["workers"]] == ["w-gcp"]

    resp_cw = rpc_post(client, "ListWorkers", {"query": {"backendId": "cw"}})
    assert [w["workerId"] for w in resp_cw["workers"]] == ["w-cw"]


def test_list_backends_returns_per_backend_summary(state, scheduler, tmp_path, log_client):
    """ListBackends returns one BackendSummary per backend with correct kind and capabilities."""
    controller_mock = _make_controller_mock(state, scheduler)
    gcp_backend = _backend_mock(
        "gcp",
        frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER}),
        advertised={"device-variant": {"v6e-16", "v5e-4"}},
    )
    k8s_backend = _backend_mock("eu-k8s", frozenset({BackendCapability.CLUSTER_VIEW}))
    controller_mock.backends = {"gcp": gcp_backend, "eu-k8s": k8s_backend}
    controller_mock.scale_group_to_backend = {"tpu-v5e": "gcp"}
    controller_mock.backend_id_for_scale_group = lambda sg: controller_mock.scale_group_to_backend.get(
        sg, DEFAULT_BACKEND_ID
    )
    controller_mock.last_unroutable_jobs = {"/alice/exp": "no backend matches the job's constraints"}
    svc = ControllerServiceImpl(
        controller=controller_mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )
    client = TestClient(ControllerDashboard(svc).app)

    resp = rpc_post(client, "ListBackends")
    summaries = {b["backendId"]: b for b in resp["backends"]}

    assert set(summaries) == {"gcp", "eu-k8s"}
    assert summaries["gcp"]["kind"] == "worker-daemon"
    assert summaries["eu-k8s"]["kind"] == "kubernetes"
    assert "workers" in summaries["gcp"]["capabilities"]
    assert "cluster" in summaries["eu-k8s"]["capabilities"]
    assert summaries["gcp"]["scaleGroups"] == ["tpu-v5e"]
    assert summaries["eu-k8s"].get("scaleGroups", []) == []
    # Advertised attributes round-trip through the proto map<string, StringList>.
    assert summaries["gcp"]["advertisedAttributes"]["device-variant"]["values"] == ["v5e-4", "v6e-16"]
    # Unroutable jobs surface as a structured count + sample, not parsed reason strings.
    assert resp["unroutableJobCount"] == 1
    assert resp["unroutableSample"][0]["reason"] == "no backend matches the job's constraints"


def test_list_backends_worker_detail_reports_autoscaler_and_health_counts(state, scheduler, tmp_path, log_client):
    """ListBackends.detail.worker carries the backend's autoscaler groups plus the
    health counts the backend authors from its own liveness tracker."""
    client = _multi_backend_client(
        state,
        scheduler,
        tmp_path,
        log_client,
        {DEFAULT_BACKEND_ID: _worker_backend(state, _status_autoscaler("tpu-v5e-us"))},
    )
    register_worker(state, "w-healthy-1", "10.0.0.1:8080", make_worker_metadata(), scale_group="tpu-v5e")
    register_worker(state, "w-healthy-2", "10.0.0.2:8080", make_worker_metadata(), scale_group="tpu-v5e")
    register_worker(state, "w-dead", "10.0.0.3:8080", make_worker_metadata(), healthy=False, scale_group="tpu-v5e")

    detail = next(b for b in rpc_post(client, "ListBackends")["backends"] if b["backendId"] == DEFAULT_BACKEND_ID)[
        "detail"
    ]["worker"]
    assert detail["totalWorkerCount"] == 3
    assert detail["healthyWorkerCount"] == 2
    # The backend's own autoscaler groups are surfaced and tagged with its backend_id.
    assert [g["name"] for g in detail["autoscaler"]["groups"]] == ["tpu-v5e-us"]
    assert detail["autoscaler"]["groups"][0]["backendId"] == DEFAULT_BACKEND_ID


def test_list_backends_worker_detail_overlays_running_task_counts(state, scheduler, tmp_path, log_client, job_request):
    """detail.worker runs the same usability overlay as GetAutoscalerStatus, so a VM
    with a running task reports a non-zero running_task_count (not idle)."""
    autoscaler = Mock()
    autoscaler.get_status.return_value = vm_pb2.AutoscalerStatus(
        groups=[
            vm_pb2.ScaleGroupStatus(
                name="tpu-v5e-us",
                slices=[
                    vm_pb2.SliceInfo(
                        slice_id="s1",
                        state="ready",
                        vms=[vm_pb2.VmInfo(vm_id="w-run", state=vm_pb2.VM_STATE_READY)],
                    )
                ],
            )
        ],
    )
    client = _multi_backend_client(
        state,
        scheduler,
        tmp_path,
        log_client,
        {DEFAULT_BACKEND_ID: _worker_backend(state, autoscaler)},
    )
    # Place a running task on the VM's worker so the overlay's DB lookup finds it.
    wid = register_worker(state, "w-run", "10.0.0.9:8080", make_worker_metadata(), scale_group="tpu-v5e")
    task_id = submit_job(state, "run-job", job_request).task(0)
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wid,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    detail = next(b for b in rpc_post(client, "ListBackends")["backends"] if b["backendId"] == DEFAULT_BACKEND_ID)[
        "detail"
    ]["worker"]
    vm = detail["autoscaler"]["groups"][0]["slices"][0]["vms"][0]
    assert vm["runningTaskCount"] == 1


def test_list_backends_kubernetes_detail_from_cluster_state(state, scheduler, tmp_path, log_client):
    """ListBackends.detail.kubernetes carries the CLUSTER_VIEW backend's synced node/pod snapshot."""
    client, k8s, provider = _make_k8s_dashboard_client(state, scheduler, tmp_path, log_client)
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
                "labels": {_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE, "iris.task_id": "job.0"},
            },
            "status": {"phase": "Running"},
        },
    )
    provider.sync(ControlSnapshot(worker_addresses={}, reconcile_rows=[], timeout_rows=[]))

    detail = next(b for b in rpc_post(client, "ListBackends")["backends"] if b["backendId"] == DEFAULT_BACKEND_ID)[
        "detail"
    ]["kubernetes"]
    assert detail["totalNodes"] == 1
    assert detail["podStatuses"][0]["podName"] == "iris-task-0"
    assert detail["podStatuses"][0]["phase"] == "Running"

    provider.close()


def test_get_kubernetes_cluster_status_ambiguous_raises(state, scheduler, tmp_path, log_client):
    """GetKubernetesClusterStatus raises INVALID_ARGUMENT when >1 CLUSTER_VIEW backends and no backend_id."""
    cluster_a = controller_pb2.Controller.GetKubernetesClusterStatusResponse(namespace="eu", total_nodes=2)
    cluster_b = controller_pb2.Controller.GetKubernetesClusterStatusResponse(namespace="us", total_nodes=4)
    client = _multi_backend_client(
        state,
        scheduler,
        tmp_path,
        log_client,
        {
            "eu-k8s": _backend_mock("eu-k8s", frozenset({BackendCapability.CLUSTER_VIEW}), cluster_status=cluster_a),
            "us-k8s": _backend_mock("us-k8s", frozenset({BackendCapability.CLUSTER_VIEW}), cluster_status=cluster_b),
        },
    )
    resp = client.post(
        "/iris.cluster.ControllerService/GetKubernetesClusterStatus",
        json={},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400

    # With explicit backend_id, resolves correctly
    data_eu = rpc_post(client, "GetKubernetesClusterStatus", {"backendId": "eu-k8s"})
    assert data_eu["namespace"] == "eu"
    data_us = rpc_post(client, "GetKubernetesClusterStatus", {"backendId": "us-k8s"})
    assert data_us["namespace"] == "us"
