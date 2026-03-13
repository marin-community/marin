# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller RPC service implementation.

These tests verify the RPC contract (input -> output) of the ControllerServiceImpl.
State changes are verified via RPC calls rather than internal state inspection.
"""

from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.db import JOBS, TASKS, ATTEMPTS, ControllerDB, _tasks_with_attempts
from iris.cluster.log_store import LogStore
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import Assignment, ControllerTransitions, HeartbeatApplyRequest, TaskUpdate
from iris.cluster.constraints import device_variant_constraint
from iris.cluster.types import JobName, WorkerId, tpu_device
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp


def _query_job(state: ControllerTransitions, job_id: JobName):
    """Read a single job from state DB."""
    with state._db.snapshot() as q:
        return q.one(JOBS, where=JOBS.c.job_id == job_id.to_wire())


def _query_tasks_with_attempts(state: ControllerTransitions, job_id: JobName):
    """Read tasks with attempts for a job."""
    with state._db.snapshot() as q:
        tasks = q.select(
            TASKS,
            where=TASKS.c.job_id == job_id.to_wire(),
            order_by=(TASKS.c.task_index.asc(),),
        )
        if not tasks:
            return []
        attempts = q.select(
            ATTEMPTS,
            where=ATTEMPTS.c.task_id.in_([t.task_id.to_wire() for t in tasks]),
            order_by=(ATTEMPTS.c.task_id.asc(), ATTEMPTS.c.attempt_id.asc()),
        )
    return _tasks_with_attempts(tasks, attempts)


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    """Create a minimal RuntimeEntrypoint proto for testing."""
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


# =============================================================================
# Test Helpers
# =============================================================================


def _register_worker(state: ControllerTransitions, worker_id: WorkerId) -> None:
    metadata = cluster_pb2.WorkerMetadata(
        hostname=str(worker_id),
        ip_address="127.0.0.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
    )
    state.register_or_refresh_worker(
        worker_id=worker_id,
        address=f"{worker_id}:8080",
        metadata=metadata,
        ts=Timestamp.now(),
    )


def _set_job_state(state: ControllerTransitions, job_id: JobName, state_value: int) -> None:
    state._db.execute(
        "UPDATE jobs SET state = ? WHERE job_id = ?",
        (state_value, job_id.to_wire()),
    )


def _assign_and_transition(
    state: ControllerTransitions,
    task_id: JobName,
    worker_id: WorkerId,
    target_state: int,
    *,
    error: str | None = None,
) -> None:
    state.queue_assignments([Assignment(task_id=task_id, worker_id=worker_id)])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=cluster_pb2.TASK_STATE_RUNNING)],
        )
    )
    if target_state != cluster_pb2.TASK_STATE_RUNNING:
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=target_state, error=error)],
            )
        )


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(
        name: str = "test-job",
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 0,  # Default to 0 for tests (no implicit retries)
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        job_name = JobName.from_string(name) if name.startswith("/") else JobName.root("test-user", name)
        return cluster_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            replicas=replicas,
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
def state(tmp_path):
    """Create a fresh ControllerTransitions for each test."""
    db_path = tmp_path / "controller.sqlite3"
    db = ControllerDB(db_path=db_path)
    log_store = LogStore(db_path=db_path)
    s = ControllerTransitions(db=db, log_store=log_store)
    yield s
    log_store.close()
    db.close()


class MockSchedulerWake:
    """Mock object that tracks controller protocol calls."""

    def __init__(self):
        self.wake = Mock()
        self.kill_tasks_on_workers = Mock()
        self.create_scheduling_context = Mock(return_value=Mock())
        self.get_job_scheduling_diagnostics = Mock(return_value=None)
        self.autoscaler = None
        self.stub_factory = Mock()


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler with wake() method."""
    return MockSchedulerWake()


@pytest.fixture
def service(state, mock_scheduler, tmp_path):
    """Create a ControllerServiceImpl for testing."""
    from iris.cluster.bundle import BundleStore

    return ControllerServiceImpl(
        state,
        state._db,
        controller=mock_scheduler,
        bundle_store=BundleStore(db_path=tmp_path / "bundles.sqlite3"),
        log_store=state._log_store,
    )


# =============================================================================
# Job Launch Tests
# =============================================================================


def test_launch_job_returns_job_id(service, job_request):
    """Verify launch_job returns a job_id and job can be queried via RPC."""
    request = job_request("test-job")

    response = service.launch_job(request, None)

    assert response.job_id == JobName.root("test-user", "test-job").to_wire()

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "test-job").to_wire()), None
    )
    assert status_response.job.job_id == JobName.root("test-user", "test-job").to_wire()
    assert status_response.job.state == cluster_pb2.JOB_STATE_PENDING


def test_launch_job_bundle_blob_rewrites_to_controller_bundle_id(service, state, job_request):
    request = job_request("bundle-job")
    request.bundle_blob = b"bundle-bytes"
    service.launch_job(request, None)

    job = _query_job(state, JobName.root("test-user", "bundle-job"))
    assert job is not None
    assert job.request.bundle_blob == b""
    assert len(job.request.bundle_id) == 64


def test_launch_job_rejects_duplicate_name(service, job_request):
    """Verify launch_job rejects duplicate job names for running jobs."""
    request = job_request("duplicate-job")

    response = service.launch_job(request, None)
    assert response.job_id == JobName.root("test-user", "duplicate-job").to_wire()

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "still running" in exc_info.value.message


def test_launch_job_replaces_finished_job_by_default(service, state, job_request):
    """Verify launch_job replaces finished jobs by default."""
    request = job_request("replaceable-job")
    job_id = JobName.root("test-user", "replaceable-job")

    # Submit initial job
    response = service.launch_job(request, None)
    assert response.job_id == job_id.to_wire()

    # Mark the job as failed
    job = _query_job(state, job_id)
    assert job is not None
    tasks = _query_tasks_with_attempts(state, job.job_id)
    assert len(tasks) == 1
    _set_job_state(state, job.job_id, cluster_pb2.JOB_STATE_FAILED)

    # Verify job is now failed
    job = _query_job(state, job_id)
    assert job.state == cluster_pb2.JOB_STATE_FAILED

    # Submit again - should succeed (replaces the finished job)
    response = service.launch_job(request, None)
    assert response.job_id == job_id.to_wire()

    # Verify the new job is pending
    job = _query_job(state, job_id)
    assert job.state == cluster_pb2.JOB_STATE_PENDING


def test_launch_job_fail_if_exists_prevents_replacement(service, state, job_request):
    """Verify fail_if_exists=true prevents replacing finished jobs."""
    request = job_request("no-replace-job")
    job_id = JobName.root("test-user", "no-replace-job")

    # Submit initial job
    response = service.launch_job(request, None)
    assert response.job_id == job_id.to_wire()

    # Mark the job as succeeded
    job = _query_job(state, job_id)
    _set_job_state(state, job.job_id, cluster_pb2.JOB_STATE_SUCCEEDED)

    # Verify job is now succeeded
    job = _query_job(state, job_id)
    assert job.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Submit again with fail_if_exists=true - should fail
    request_no_replace = job_request("no-replace-job")
    request_no_replace.fail_if_exists = True

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request_no_replace, None)

    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "SUCCEEDED" in exc_info.value.message


def test_launch_job_rejects_empty_name(service, state):
    """Verify launch_job rejects empty job names."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
    )

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.INVALID_ARGUMENT


# =============================================================================
# Job Status Tests
# =============================================================================


def test_get_job_status_returns_status(service, job_request):
    """Verify get_job_status returns correct status for launched job."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "test-job").to_wire())
    response = service.get_job_status(request, None)

    assert response.job.job_id == JobName.root("test-user", "test-job").to_wire()
    assert response.job.state == cluster_pb2.JOB_STATE_PENDING


def test_get_job_status_not_found(service):
    """Verify get_job_status raises ConnectError for unknown job."""
    request = cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "nonexistent").to_wire())

    with pytest.raises(ConnectError) as exc_info:
        service.get_job_status(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


# =============================================================================
# Job Termination Tests
# =============================================================================


def test_terminate_job_marks_as_killed(service, job_request):
    """Verify terminate_job sets job state to KILLED via get_job_status."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "test-job").to_wire())
    response = service.terminate_job(request, None)

    assert isinstance(response, cluster_pb2.Empty)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "test-job").to_wire()), None
    )
    assert status_response.job.state == cluster_pb2.JOB_STATE_KILLED
    assert status_response.job.finished_at.epoch_ms > 0


def test_terminate_job_not_found(service):
    """Verify terminate_job raises ConnectError for unknown job."""
    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "nonexistent").to_wire())

    with pytest.raises(ConnectError) as exc_info:
        service.terminate_job(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


def test_terminate_pending_job(service, job_request):
    """Verify terminate_job works on pending jobs (not just running)."""
    service.launch_job(job_request("test-job"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "test-job").to_wire())
    service.terminate_job(request, None)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "test-job").to_wire()), None
    )
    assert status_response.job.state == cluster_pb2.JOB_STATE_KILLED
    assert status_response.job.finished_at.epoch_ms > 0


def test_terminate_job_cascades_to_children(service, job_request):
    """Verify terminate_job terminates all children when parent is terminated."""
    service.launch_job(job_request("parent"), None)
    service.launch_job(job_request("/test-user/parent/child1"), None)
    service.launch_job(job_request("/test-user/parent/child2"), None)

    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "parent").to_wire())
    service.terminate_job(request, None)

    # Verify all jobs are killed via get_job_status RPC
    for job_name in [
        JobName.root("test-user", "parent"),
        JobName.from_string("/test-user/parent/child1"),
        JobName.from_string("/test-user/parent/child2"),
    ]:
        status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=job_name.to_wire()), None)
        assert status.job.state == cluster_pb2.JOB_STATE_KILLED, f"Job {job_name} should be KILLED"


def test_terminate_job_only_affects_descendants(service, job_request):
    """Verify terminate_job does not affect sibling jobs."""
    service.launch_job(job_request("parent"), None)
    service.launch_job(job_request("/test-user/parent/child1"), None)
    service.launch_job(job_request("/test-user/parent/child2"), None)

    # Terminate only child1
    request = cluster_pb2.Controller.TerminateJobRequest(
        job_id=JobName.from_string("/test-user/parent/child1").to_wire()
    )
    service.terminate_job(request, None)

    # Verify states via get_job_status RPC
    child1_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/test-user/parent/child1").to_wire()),
        None,
    )
    assert child1_status.job.state == cluster_pb2.JOB_STATE_KILLED

    child2_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/test-user/parent/child2").to_wire()),
        None,
    )
    assert child2_status.job.state == cluster_pb2.JOB_STATE_PENDING

    parent_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "parent").to_wire()), None
    )
    assert parent_status.job.state == cluster_pb2.JOB_STATE_PENDING


def test_terminate_job_skips_already_finished_children(service, state, job_request):
    """Verify terminate_job skips children already in terminal state."""
    # Launch parent via RPC
    service.launch_job(job_request("parent"), None)

    # Create child and transition it to SUCCEEDED.
    service.launch_job(job_request("/test-user/parent/child-succeeded"), None)
    child_succeeded_job = JobName.from_string("/test-user/parent/child-succeeded")
    child_task = _query_tasks_with_attempts(state, child_succeeded_job)[0]
    done_worker = WorkerId("w-child-succeeded")
    _register_worker(state, done_worker)
    _assign_and_transition(state, child_task.task_id, done_worker, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Launch running child via RPC
    service.launch_job(job_request("/test-user/parent/child-running"), None)

    # Terminate parent
    request = cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "parent").to_wire())
    service.terminate_job(request, None)

    # Verify states via get_job_status RPC
    succeeded_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(
            job_id=JobName.from_string("/test-user/parent/child-succeeded").to_wire()
        ),
        None,
    )
    assert succeeded_status.job.state == cluster_pb2.JOB_STATE_SUCCEEDED

    running_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(
            job_id=JobName.from_string("/test-user/parent/child-running").to_wire()
        ),
        None,
    )
    assert running_status.job.state == cluster_pb2.JOB_STATE_KILLED

    parent_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "parent").to_wire()),
        None,
    )
    assert parent_status.job.state == cluster_pb2.JOB_STATE_KILLED


# =============================================================================
# Authorization Tests
# =============================================================================


def test_terminate_job_allowed_by_owner(service, job_request):
    """Job owner can terminate their own job."""
    from iris.rpc.auth import _verified_user

    service.launch_job(job_request("/alice/my-job"), None)

    token = _verified_user.set("alice")
    try:
        request = cluster_pb2.Controller.TerminateJobRequest(job_id="/alice/my-job")
        service.terminate_job(request, None)
    finally:
        _verified_user.reset(token)

    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="/alice/my-job"), None)
    assert status.job.state == cluster_pb2.JOB_STATE_KILLED


def test_terminate_job_rejected_for_non_owner(state, mock_scheduler, tmp_path, job_request):
    """Non-owner gets PERMISSION_DENIED when trying to terminate another user's job."""
    from iris.cluster.bundle import BundleStore
    from iris.cluster.controller.auth import ControllerAuth
    from iris.rpc.auth import _verified_user

    auth_service = ControllerServiceImpl(
        state,
        state._db,
        controller=mock_scheduler,
        bundle_store=BundleStore(db_path=tmp_path / "bundles_owner.sqlite3"),
        log_store=state._log_store,
        auth=ControllerAuth(provider="static"),
    )

    auth_service.launch_job(job_request("/alice/my-job"), None)

    token = _verified_user.set("bob")
    try:
        request = cluster_pb2.Controller.TerminateJobRequest(job_id="/alice/my-job")
        with pytest.raises(ConnectError) as exc_info:
            auth_service.terminate_job(request, None)
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_user.reset(token)

    # Job should still be running
    status = auth_service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="/alice/my-job"), None)
    assert status.job.state == cluster_pb2.JOB_STATE_PENDING


def test_launch_child_job_rejected_for_non_owner(state, mock_scheduler, tmp_path, job_request):
    """Cannot submit a child job under another user's hierarchy."""
    from iris.cluster.bundle import BundleStore
    from iris.cluster.controller.auth import ControllerAuth
    from iris.rpc.auth import _verified_user

    auth_service = ControllerServiceImpl(
        state,
        state._db,
        controller=mock_scheduler,
        bundle_store=BundleStore(db_path=tmp_path / "bundles_child.sqlite3"),
        log_store=state._log_store,
        auth=ControllerAuth(provider="static"),
    )

    auth_service.launch_job(job_request("/alice/parent-job"), None)

    token = _verified_user.set("bob")
    try:
        with pytest.raises(ConnectError) as exc_info:
            auth_service.launch_job(job_request("/alice/parent-job/sneaky-child"), None)
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_user.reset(token)


def test_terminate_job_allowed_when_auth_disabled(service, job_request):
    """When auth is disabled (no verified user), anyone can terminate."""
    service.launch_job(job_request("/alice/my-job"), None)

    # No _verified_user set => auth disabled
    request = cluster_pb2.Controller.TerminateJobRequest(job_id="/alice/my-job")
    service.terminate_job(request, None)

    status = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id="/alice/my-job"), None)
    assert status.job.state == cluster_pb2.JOB_STATE_KILLED


def test_parent_job_failure_cascades_to_children(service, state, job_request):
    """Verify when a parent job fails, all children are automatically cancelled."""
    # Launch parent and children via RPC
    service.launch_job(job_request("parent"), None)
    service.launch_job(job_request("/test-user/parent/child1"), None)
    service.launch_job(job_request("/test-user/parent/child2"), None)

    # Get parent task and mark it as failed
    parent_job = _query_job(state, JobName.root("test-user", "parent"))
    parent_task = _query_tasks_with_attempts(state, parent_job.job_id)[0]
    worker_id = WorkerId("w-parent")
    _register_worker(state, worker_id)
    _assign_and_transition(
        state,
        parent_task.task_id,
        worker_id,
        cluster_pb2.TASK_STATE_FAILED,
        error="Parent task failed",
    )

    # Verify all jobs are now in terminal states via get_job_status RPC
    parent_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "parent").to_wire()), None
    )
    assert parent_status.job.state == cluster_pb2.JOB_STATE_FAILED

    child1_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/test-user/parent/child1").to_wire()),
        None,
    )
    assert child1_status.job.state == cluster_pb2.JOB_STATE_KILLED, "Child 1 should be killed when parent fails"

    child2_status = service.get_job_status(
        cluster_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/test-user/parent/child2").to_wire()),
        None,
    )
    assert child2_status.job.state == cluster_pb2.JOB_STATE_KILLED, "Child 2 should be killed when parent fails"


def test_launch_job_rejects_child_of_failed_parent(service, state, job_request):
    """Verify launch_job rejects submissions to a failed parent's namespace."""
    # Launch and fail parent
    service.launch_job(job_request("failed-parent"), None)
    parent_job = _query_job(state, JobName.root("test-user", "failed-parent"))
    parent_task = _query_tasks_with_attempts(state, parent_job.job_id)[0]
    worker_id = WorkerId("w-failed-parent")
    _register_worker(state, worker_id)
    _assign_and_transition(
        state,
        parent_task.task_id,
        worker_id,
        cluster_pb2.TASK_STATE_FAILED,
        error="Parent task failed",
    )

    # Try to submit a child job - should fail
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(job_request("/test-user/failed-parent/new-child"), None)

    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert "terminated" in exc_info.value.message.lower() or "failed" in exc_info.value.message.lower()


# =============================================================================
# Job List Tests
# =============================================================================


def test_list_jobs_returns_all_jobs(service, job_request):
    """Verify list_jobs returns all jobs launched via RPC."""
    service.launch_job(job_request("job-1"), None)
    service.launch_job(job_request("job-2"), None)
    service.launch_job(job_request("job-3"), None)

    # Terminate one to get different state
    service.terminate_job(
        cluster_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "job-3").to_wire()), None
    )

    request = cluster_pb2.Controller.ListJobsRequest()
    response = service.list_jobs(request, None)

    assert len(response.jobs) == 3
    job_ids = {j.job_id for j in response.jobs}
    assert job_ids == {
        JobName.root("test-user", "job-1").to_wire(),
        JobName.root("test-user", "job-2").to_wire(),
        JobName.root("test-user", "job-3").to_wire(),
    }

    states_by_id = {j.job_id: j.state for j in response.jobs}
    assert states_by_id[JobName.root("test-user", "job-1").to_wire()] == cluster_pb2.JOB_STATE_PENDING
    assert states_by_id[JobName.root("test-user", "job-2").to_wire()] == cluster_pb2.JOB_STATE_PENDING
    assert states_by_id[JobName.root("test-user", "job-3").to_wire()] == cluster_pb2.JOB_STATE_KILLED


# =============================================================================
# Worker Tests
# =============================================================================


def test_list_workers_returns_all(service, state, worker_metadata):
    """Verify list_workers returns all registered workers."""
    from iris.rpc.auth import _verified_user

    db = state._db
    db.ensure_user("system:worker", Timestamp.now(), role="worker")
    token = _verified_user.set("system:worker")
    try:
        for i in range(3):
            request = cluster_pb2.Controller.RegisterRequest(
                address=f"host{i}:8080",
                metadata=worker_metadata(),
                worker_id=f"worker-{i}",
            )
            service.register(request, None)
    finally:
        _verified_user.reset(token)

    request = cluster_pb2.Controller.ListWorkersRequest()
    response = service.list_workers(request, None)

    assert len(response.workers) == 3

    # All workers should be healthy after registration
    for w in response.workers:
        assert w.healthy is True


# =============================================================================
# Constraint Injection Tests
# =============================================================================


def test_launch_job_injects_device_constraints_from_tpu_resource(service, state):
    """Job with TPU resource spec gets auto-injected device-type and device-variant constraints."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "tpu-job").to_wire(),
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
    )
    request.resources.device.CopyFrom(tpu_device("v5litepod-16"))

    service.launch_job(request, None)

    job = _query_job(state, JobName.root("test-user", "tpu-job"))
    stored_constraints = list(job.request.constraints)
    keys = {c.key for c in stored_constraints}
    assert WellKnownAttribute.DEVICE_TYPE in keys
    assert WellKnownAttribute.DEVICE_VARIANT in keys

    dt = next(c for c in stored_constraints if c.key == WellKnownAttribute.DEVICE_TYPE)
    assert dt.value.string_value == "tpu"
    dv = next(c for c in stored_constraints if c.key == WellKnownAttribute.DEVICE_VARIANT)
    assert dv.value.string_value == "v5litepod-16"


def test_launch_job_user_constraints_override_auto(service, state):
    """Explicit user constraints for canonical keys replace auto-generated ones."""
    user_variant = device_variant_constraint(["v5litepod-16", "v6e-16"])

    request = cluster_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "multi-variant-job").to_wire(),
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
    )
    request.resources.device.CopyFrom(tpu_device("v5litepod-16"))
    request.constraints.append(user_variant.to_proto())

    service.launch_job(request, None)

    job = _query_job(state, JobName.root("test-user", "multi-variant-job"))
    stored_constraints = list(job.request.constraints)

    # device-variant should be the user's IN constraint, not the auto EQ
    dv_constraints = [c for c in stored_constraints if c.key == WellKnownAttribute.DEVICE_VARIANT]
    assert len(dv_constraints) == 1
    assert dv_constraints[0].op == cluster_pb2.CONSTRAINT_OP_IN

    # device-type should still be auto-injected
    dt_constraints = [c for c in stored_constraints if c.key == WellKnownAttribute.DEVICE_TYPE]
    assert len(dt_constraints) == 1
    assert dt_constraints[0].value.string_value == "tpu"


def test_launch_job_cpu_resource_no_constraints_injected(service, state):
    """CPU-only jobs get no auto-injected device constraints."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "cpu-job").to_wire(),
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
    )
    request.resources.device.CopyFrom(cluster_pb2.DeviceConfig(cpu=cluster_pb2.CpuDevice()))

    service.launch_job(request, None)

    job = _query_job(state, JobName.root("test-user", "cpu-job"))
    assert len(job.request.constraints) == 0


# =============================================================================
# Register Role-Gating Tests
# =============================================================================


def test_register_requires_worker_role(state, mock_scheduler, tmp_path, worker_metadata):
    """Non-worker user gets PERMISSION_DENIED on register()."""
    from iris.cluster.bundle import BundleStore
    from iris.cluster.controller.auth import ControllerAuth
    from iris.rpc.auth import _verified_user

    db = state._db
    now = Timestamp.now()
    db.ensure_user("alice", now, role="user")

    auth = ControllerAuth()
    service = ControllerServiceImpl(
        state,
        db,
        controller=mock_scheduler,
        bundle_store=BundleStore(db_path=tmp_path / "bundles.sqlite3"),
        log_store=state._log_store,
        auth=auth,
    )

    token = _verified_user.set("alice")
    try:
        with pytest.raises(ConnectError) as exc_info:
            service.register(
                cluster_pb2.Controller.RegisterRequest(
                    worker_id="w-1",
                    address="localhost:8080",
                    metadata=worker_metadata(),
                ),
                None,
            )
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_user.reset(token)


def test_register_allows_worker_role(state, mock_scheduler, tmp_path, worker_metadata):
    """Worker-role user can call register()."""
    from iris.cluster.bundle import BundleStore
    from iris.cluster.controller.auth import ControllerAuth
    from iris.rpc.auth import _verified_user

    db = state._db
    now = Timestamp.now()
    db.ensure_user("system:worker", now, role="worker")

    auth = ControllerAuth()
    service = ControllerServiceImpl(
        state,
        db,
        controller=mock_scheduler,
        bundle_store=BundleStore(db_path=tmp_path / "bundles.sqlite3"),
        log_store=state._log_store,
        auth=auth,
    )

    token = _verified_user.set("system:worker")
    try:
        resp = service.register(
            cluster_pb2.Controller.RegisterRequest(
                worker_id="w-1",
                address="localhost:8080",
                metadata=worker_metadata(),
            ),
            None,
        )
        assert resp.accepted
    finally:
        _verified_user.reset(token)
