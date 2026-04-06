# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for budget admin API: set/get/list RPCs, auth enforcement, and band validation."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import Assignment, HeartbeatApplyRequest, TaskUpdate
from iris.cluster.types import JobName, WorkerId
from iris.log_server.server import LogServiceImpl
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc.auth import VerifiedIdentity, _verified_identity
from rigging.timing import Timestamp

from tests.cluster.controller.conftest import (
    MockController,
    make_controller_state,
    make_test_entrypoint,
)
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.auth import ControllerAuth


@pytest.fixture
def state():
    with make_controller_state() as s:
        yield s


@pytest.fixture
def mock_controller():
    return MockController()


def _make_service(state, mock_controller, tmp_path, auth=None):
    return ControllerServiceImpl(
        state,
        state._db,
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=LogServiceImpl(),
        auth=auth or ControllerAuth(),
    )


def _as_admin(fn, *args, **kwargs):
    """Run fn with admin identity."""
    reset = _verified_identity.set(VerifiedIdentity(user_id="admin", role="admin"))
    try:
        return fn(*args, **kwargs)
    finally:
        _verified_identity.reset(reset)


def _as_user(fn, user_id="alice", *args, **kwargs):
    """Run fn with user identity."""
    reset = _verified_identity.set(VerifiedIdentity(user_id=user_id, role="user"))
    try:
        return fn(*args, **kwargs)
    finally:
        _verified_identity.reset(reset)


def _make_job_request(name: str, band: int = 0) -> controller_pb2.Controller.LaunchJobRequest:
    return controller_pb2.Controller.LaunchJobRequest(
        name=name,
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        priority_band=band,
    )


# ---------------------------------------------------------------------------
# test_admin_can_set_budget
# ---------------------------------------------------------------------------


def test_admin_can_set_budget(state, mock_controller, tmp_path):
    service = _make_service(state, mock_controller, tmp_path)

    # Set budget as admin
    _as_admin(
        service.set_user_budget,
        controller_pb2.Controller.SetUserBudgetRequest(
            user_id="alice",
            budget_limit=5000,
            max_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        ),
        None,
    )

    # Get budget back
    resp = _as_admin(
        service.get_user_budget,
        controller_pb2.Controller.GetUserBudgetRequest(user_id="alice"),
        None,
    )
    assert resp.user_id == "alice"
    assert resp.budget_limit == 5000
    assert resp.max_band == job_pb2.PRIORITY_BAND_INTERACTIVE
    assert resp.budget_spent == 0


# ---------------------------------------------------------------------------
# test_non_admin_cannot_set_budget
# ---------------------------------------------------------------------------


def test_non_admin_cannot_set_budget(state, mock_controller, tmp_path):
    service = _make_service(state, mock_controller, tmp_path)

    with pytest.raises(ConnectError) as exc_info:
        _as_user(
            service.set_user_budget,
            "alice",
            controller_pb2.Controller.SetUserBudgetRequest(
                user_id="alice",
                budget_limit=5000,
                max_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            ),
            None,
        )
    assert exc_info.value.code == Code.PERMISSION_DENIED


# ---------------------------------------------------------------------------
# test_non_admin_cannot_submit_production
# ---------------------------------------------------------------------------


def test_non_admin_cannot_submit_production(state, mock_controller, tmp_path):
    auth = ControllerAuth(provider="static")
    service = _make_service(state, mock_controller, tmp_path, auth=auth)

    # Submit PRODUCTION band as non-admin user -> should fail
    request = _make_job_request("/alice/prod-job", band=job_pb2.PRIORITY_BAND_PRODUCTION)
    with pytest.raises(ConnectError) as exc_info:
        _as_user(
            service.launch_job,
            "alice",
            request,
            None,
        )
    assert exc_info.value.code == Code.PERMISSION_DENIED


# ---------------------------------------------------------------------------
# test_band_validation_max_band
# ---------------------------------------------------------------------------


def test_band_validation_rejects_above_max_band(state, mock_controller, tmp_path):
    """User with max_band=BATCH cannot submit INTERACTIVE jobs."""
    auth = ControllerAuth(provider="static")
    service = _make_service(state, mock_controller, tmp_path, auth=auth)

    # Set alice's max_band to BATCH (sort key 3)
    _as_admin(
        service.set_user_budget,
        controller_pb2.Controller.SetUserBudgetRequest(
            user_id="alice",
            budget_limit=0,
            max_band=job_pb2.PRIORITY_BAND_BATCH,
        ),
        None,
    )

    # Try to submit INTERACTIVE (sort key 2 < 3) -> should fail
    request = _make_job_request("/alice/interactive-job", band=job_pb2.PRIORITY_BAND_INTERACTIVE)
    with pytest.raises(ConnectError) as exc_info:
        _as_user(
            service.launch_job,
            "alice",
            request,
            None,
        )
    assert exc_info.value.code == Code.PERMISSION_DENIED
    assert "cannot submit" in str(exc_info.value.message).lower()


# ---------------------------------------------------------------------------
# test_budget_spend_reflects_running_tasks
# ---------------------------------------------------------------------------


def test_budget_spend_reflects_running_tasks(state, mock_controller, tmp_path):
    service = _make_service(state, mock_controller, tmp_path)

    # Set budget for alice
    _as_admin(
        service.set_user_budget,
        controller_pb2.Controller.SetUserBudgetRequest(
            user_id="alice",
            budget_limit=10000,
            max_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        ),
        None,
    )

    # Submit a job as alice (directly via transitions)
    job_id = JobName.root("alice", "test-job")
    request = _make_job_request(job_id.to_wire())
    state.submit_job(job_id, request, Timestamp.now())

    # Register a worker and assign the task
    worker_id = WorkerId("worker-1")
    state.register_or_refresh_worker(
        worker_id=worker_id,
        address="worker-1:8080",
        metadata=job_pb2.WorkerMetadata(
            hostname="worker-1",
            ip_address="127.0.0.1",
            cpu_count=8,
            memory_bytes=16 * 1024**3,
            disk_bytes=100 * 1024**3,
        ),
        ts=Timestamp.now(),
    )

    task_id = JobName.from_wire(f"{job_id.to_wire()}/0")
    state.queue_assignments([Assignment(task_id=task_id, worker_id=worker_id)])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
        )
    )

    # Get budget — spend should be non-zero
    resp = _as_admin(
        service.get_user_budget,
        controller_pb2.Controller.GetUserBudgetRequest(user_id="alice"),
        None,
    )
    assert resp.budget_spent > 0


# ---------------------------------------------------------------------------
# test_list_user_budgets
# ---------------------------------------------------------------------------


def test_list_user_budgets(state, mock_controller, tmp_path):
    service = _make_service(state, mock_controller, tmp_path)

    # Set budgets for multiple users
    for user_id, limit, band in [
        ("alice", 5000, job_pb2.PRIORITY_BAND_INTERACTIVE),
        ("bob", 3000, job_pb2.PRIORITY_BAND_BATCH),
        ("charlie", 0, job_pb2.PRIORITY_BAND_PRODUCTION),
    ]:
        _as_admin(
            service.set_user_budget,
            controller_pb2.Controller.SetUserBudgetRequest(
                user_id=user_id,
                budget_limit=limit,
                max_band=band,
            ),
            None,
        )

    # List all budgets
    resp = _as_admin(
        service.list_user_budgets,
        controller_pb2.Controller.ListUserBudgetsRequest(),
        None,
    )
    assert len(resp.users) == 3
    user_ids = {u.user_id for u in resp.users}
    assert user_ids == {"alice", "bob", "charlie"}

    # Verify specific budget values
    by_user = {u.user_id: u for u in resp.users}
    assert by_user["alice"].budget_limit == 5000
    assert by_user["bob"].budget_limit == 3000
    assert by_user["charlie"].budget_limit == 0
    assert by_user["charlie"].max_band == job_pb2.PRIORITY_BAND_PRODUCTION


# ---------------------------------------------------------------------------
# test_admin_can_submit_production
# ---------------------------------------------------------------------------


def test_admin_can_submit_production(state, mock_controller, tmp_path):
    """Admin should be able to submit production jobs."""
    auth = ControllerAuth(provider="static")
    service = _make_service(state, mock_controller, tmp_path, auth=auth)

    request = _make_job_request("/admin/prod-job", band=job_pb2.PRIORITY_BAND_PRODUCTION)
    resp = _as_admin(service.launch_job, request, None)
    assert resp.job_id == "/admin/prod-job"


# ---------------------------------------------------------------------------
# test_get_budget_not_found
# ---------------------------------------------------------------------------


def test_get_budget_not_found(state, mock_controller, tmp_path):
    service = _make_service(state, mock_controller, tmp_path)

    with pytest.raises(ConnectError) as exc_info:
        _as_admin(
            service.get_user_budget,
            controller_pb2.Controller.GetUserBudgetRequest(user_id="nonexistent"),
            None,
        )
    assert exc_info.value.code == Code.NOT_FOUND


# ---------------------------------------------------------------------------
# test_set_budget_invalid_max_band
# ---------------------------------------------------------------------------


def test_set_budget_invalid_max_band(state, mock_controller, tmp_path):
    """Setting an invalid max_band value is rejected."""
    service = _make_service(state, mock_controller, tmp_path)

    with pytest.raises(ConnectError) as exc_info:
        _as_admin(
            service.set_user_budget,
            controller_pb2.Controller.SetUserBudgetRequest(
                user_id="alice",
                budget_limit=5000,
                max_band=99,  # invalid
            ),
            None,
        )
    assert exc_info.value.code == Code.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# test_set_budget_empty_user_id
# ---------------------------------------------------------------------------


def test_set_budget_empty_user_id(state, mock_controller, tmp_path):
    """Setting budget with empty user_id is rejected."""
    service = _make_service(state, mock_controller, tmp_path)

    with pytest.raises(ConnectError) as exc_info:
        _as_admin(
            service.set_user_budget,
            controller_pb2.Controller.SetUserBudgetRequest(
                user_id="",
                budget_limit=5000,
                max_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            ),
            None,
        )
    assert exc_info.value.code == Code.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# test_default_band_submission
# ---------------------------------------------------------------------------


def test_unspecified_band_defaults_to_interactive(state, mock_controller, tmp_path):
    """Submitting with UNSPECIFIED (0) band defaults to INTERACTIVE."""
    auth = ControllerAuth(provider="static")
    service = _make_service(state, mock_controller, tmp_path, auth=auth)

    # Submit with band=0 (UNSPECIFIED)
    request = _make_job_request("/alice/default-band-job", band=0)
    resp = _as_user(service.launch_job, "alice", request, None)
    assert resp.job_id == "/alice/default-band-job"


# ---------------------------------------------------------------------------
# test_user_can_read_own_budget
# ---------------------------------------------------------------------------


def test_user_can_read_own_budget(state, mock_controller, tmp_path):
    """Non-admin users can read budget info (require_identity, not require admin)."""
    service = _make_service(state, mock_controller, tmp_path)

    # Set budget as admin first
    _as_admin(
        service.set_user_budget,
        controller_pb2.Controller.SetUserBudgetRequest(
            user_id="alice",
            budget_limit=5000,
            max_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        ),
        None,
    )

    # Read as non-admin user
    resp = _as_user(
        service.get_user_budget,
        "alice",
        controller_pb2.Controller.GetUserBudgetRequest(user_id="alice"),
        None,
    )
    assert resp.user_id == "alice"
    assert resp.budget_limit == 5000
