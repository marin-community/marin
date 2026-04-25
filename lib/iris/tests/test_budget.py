# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for budget tracking (resource_value / compute_user_spend / interleave_by_user)
and the admin API RPCs that expose them."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.bundle import BundleStore
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.budget import (
    UserBudgetDefaults,
    UserTask,
    compute_effective_band,
    compute_user_spend,
    interleave_by_user,
    resource_value,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import Assignment, HeartbeatApplyRequest, TaskUpdate
from iris.cluster.types import JobName, WorkerId
from iris.log_server.server import LogServiceImpl
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import VerifiedIdentity, _verified_identity
from iris.rpc.proto_utils import PRIORITY_BAND_VALUES, priority_band_name, priority_band_value
from rigging.timing import Timestamp

from tests.cluster.controller.conftest import (
    MockController,
    make_controller_state,
    make_test_entrypoint,
)

PRODUCTION = job_pb2.PRIORITY_BAND_PRODUCTION
INTERACTIVE = job_pb2.PRIORITY_BAND_INTERACTIVE
BATCH = job_pb2.PRIORITY_BAND_BATCH

GiB = 1024**3


@pytest.fixture
def state():
    """Fresh ControllerTransitions with a temp DB, cleaned up on exit."""
    with make_controller_state() as s:
        yield s


# ---------------------------------------------------------------------------
# resource_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cpu_millicores,memory_bytes,accelerator_count,expected",
    [
        (4000, 16 * GiB, 0, 5 * 4 + 16),  # CPU only
        (8000, 64 * GiB, 4, 1000 * 4 + 64 + 5 * 8),  # 4 GPUs
        (96000, 320 * GiB, 8, 1000 * 8 + 320 + 5 * 96),  # 8 TPU chips
        (1500, int(1.5 * GiB), 0, 5 * 1 + 1),  # fractional → truncated
    ],
)
def test_resource_value(cpu_millicores, memory_bytes, accelerator_count, expected):
    assert resource_value(cpu_millicores, memory_bytes, accelerator_count) == expected


# ---------------------------------------------------------------------------
# interleave_by_user
# ---------------------------------------------------------------------------


def test_interleave_by_user_empty():
    assert interleave_by_user([], user_spend={}) == []


def test_interleave_by_user_single_user_preserves_order():
    tasks = [UserTask("alice", "t1"), UserTask("alice", "t2"), UserTask("alice", "t3")]
    assert interleave_by_user(tasks, user_spend={}) == ["t1", "t2", "t3"]


def test_interleave_by_user_lower_spend_goes_first():
    tasks = [
        UserTask("alice", "a1"),
        UserTask("alice", "a2"),
        UserTask("bob", "b1"),
        UserTask("bob", "b2"),
    ]
    # Bob has spent less, so his task goes first in each round.
    assert interleave_by_user(tasks, user_spend={"alice": 8000, "bob": 1000}) == [
        "b1",
        "a1",
        "b2",
        "a2",
    ]


def test_interleave_by_user_missing_spend_defaults_to_zero():
    tasks = [UserTask("alice", "a1"), UserTask("bob", "b1")]
    # Alice has no spend row → defaults to 0 (< bob's 5000), so alice goes first.
    assert interleave_by_user(tasks, user_spend={"bob": 5000}) == ["a1", "b1"]


def test_interleave_by_user_three_users_unequal_counts():
    tasks = [
        UserTask("alice", "a1"),
        UserTask("alice", "a2"),
        UserTask("bob", "b1"),
        UserTask("charlie", "c1"),
        UserTask("charlie", "c2"),
        UserTask("charlie", "c3"),
    ]
    # Spend order: bob (100) < charlie (3000) < alice (5000).
    result = interleave_by_user(tasks, user_spend={"alice": 5000, "bob": 100, "charlie": 3000})
    assert result == ["b1", "c1", "a1", "c2", "a2", "c3"]


# ---------------------------------------------------------------------------
# compute_effective_band
# ---------------------------------------------------------------------------


_UNLIMITED_DEFAULTS = UserBudgetDefaults(budget_limit=0, max_band=INTERACTIVE)


@pytest.mark.parametrize(
    "task_band,spend,limit,expected",
    [
        (INTERACTIVE, 10000, 5000, BATCH),  # over budget → demoted
        (INTERACTIVE, 3000, 5000, INTERACTIVE),  # within budget → kept
        (PRODUCTION, 10000, 5000, PRODUCTION),  # production never demoted
        (INTERACTIVE, 999999, 0, INTERACTIVE),  # limit=0 means unlimited
        (BATCH, 10000, 5000, BATCH),  # batch stays batch
    ],
)
def test_effective_band(task_band, spend, limit, expected):
    assert (
        compute_effective_band(task_band, "alice", {"alice": spend}, {"alice": limit}, _UNLIMITED_DEFAULTS) == expected
    )


def test_effective_band_no_limit_row_uses_defaults():
    """Users without a budget row fall back to defaults.budget_limit."""
    # Tight default → over-budget spend demotes.
    tight = UserBudgetDefaults(budget_limit=1000, max_band=INTERACTIVE)
    assert compute_effective_band(INTERACTIVE, "alice", {"alice": 5000}, {}, tight) == BATCH
    # Unlimited default (0) → no demotion regardless of spend.
    assert compute_effective_band(INTERACTIVE, "alice", {"alice": 999999}, {}, _UNLIMITED_DEFAULTS) == INTERACTIVE


# ---------------------------------------------------------------------------
# priority_band helpers
# ---------------------------------------------------------------------------


def test_priority_band_name_roundtrip():
    for band in PRIORITY_BAND_VALUES:
        assert priority_band_value(priority_band_name(band)) == band


# ---------------------------------------------------------------------------
# compute_user_spend
# ---------------------------------------------------------------------------


def _launch_request(
    name: str,
    cpu_millicores: int = 4000,
    memory_bytes: int = 16 * GiB,
    include_resources: bool = True,
    replicas: int = 1,
    band: int = 0,
) -> controller_pb2.Controller.LaunchJobRequest:
    req = controller_pb2.Controller.LaunchJobRequest(
        name=name,
        entrypoint=make_test_entrypoint(),
        environment=job_pb2.EnvironmentConfig(),
        replicas=replicas,
        priority_band=band,
    )
    if include_resources:
        req.resources.CopyFrom(job_pb2.ResourceSpecProto(cpu_millicores=cpu_millicores, memory_bytes=memory_bytes))
    return req


def _start_running_job(
    state,
    user: str,
    job_name: str,
    *,
    cpu_millicores: int = 4000,
    memory_bytes: int = 16 * GiB,
    replicas: int = 1,
    include_resources: bool = True,
) -> None:
    """Submit a job, register a worker, and transition each task to RUNNING."""
    job_id = JobName.root(user, job_name)
    request = _launch_request(
        job_id.to_wire(),
        cpu_millicores=cpu_millicores,
        memory_bytes=memory_bytes,
        include_resources=include_resources,
        replicas=replicas,
    )
    state.submit_job(job_id, request, Timestamp.now())

    worker_id = WorkerId(f"w-{user}")
    state.register_or_refresh_worker(
        worker_id=worker_id,
        address=f"{worker_id}:8080",
        metadata=job_pb2.WorkerMetadata(
            hostname=str(worker_id),
            ip_address="127.0.0.1",
            cpu_count=16,
            memory_bytes=64 * GiB,
            disk_bytes=100 * GiB,
        ),
        ts=Timestamp.now(),
    )
    for idx in range(replicas):
        task_id = job_id.task(idx)
        state.queue_assignments([Assignment(task_id=task_id, worker_id=worker_id)])
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
            )
        )


def test_compute_user_spend_empty(state):
    with state._db.snapshot() as snap:
        assert compute_user_spend(snap) == {}


def test_compute_user_spend_sums_running_tasks(state):
    _start_running_job(state, "alice", "job", cpu_millicores=4000, memory_bytes=16 * GiB, replicas=2)
    with state._db.snapshot() as snap:
        spend = compute_user_spend(snap)
    assert spend["alice"] == resource_value(4000, 16 * GiB, 0) * 2


def test_compute_user_spend_excludes_pending(state):
    """Tasks that never reach RUNNING/ASSIGNED/BUILDING do not contribute."""
    job_id = JobName.root("bob", "pending")
    request = _launch_request(job_id.to_wire(), cpu_millicores=2000, memory_bytes=8 * GiB)
    state.submit_job(job_id, request, Timestamp.now())
    with state._db.snapshot() as snap:
        assert compute_user_spend(snap).get("bob", 0) == 0


def test_compute_user_spend_null_resources_proto(state):
    """Regression: res_device_json is NULL when LaunchJobRequest omits resources."""
    _start_running_job(state, "carol", "no-resources", include_resources=False)
    with state._db.snapshot() as snap:
        assert compute_user_spend(snap).get("carol", 0) == 0


# ---------------------------------------------------------------------------
# Budget admin API (service layer)
# ---------------------------------------------------------------------------


@pytest.fixture
def service(state, tmp_path) -> ControllerServiceImpl:
    """ControllerServiceImpl wired with static-provider auth so that
    priority-band authorization triggers (see launch_job band check)."""
    return ControllerServiceImpl(
        state,
        state._store,
        controller=MockController(),
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=LogServiceImpl(),
        auth=ControllerAuth(provider="static"),
    )


def _as_admin(fn, *args, **kwargs):
    reset = _verified_identity.set(VerifiedIdentity(user_id="admin", role="admin"))
    try:
        return fn(*args, **kwargs)
    finally:
        _verified_identity.reset(reset)


def _as_user(fn, user_id, *args, **kwargs):
    reset = _verified_identity.set(VerifiedIdentity(user_id=user_id, role="user"))
    try:
        return fn(*args, **kwargs)
    finally:
        _verified_identity.reset(reset)


def _set_budget(user_id: str, limit: int = 5000, max_band: int = INTERACTIVE):
    return controller_pb2.Controller.SetUserBudgetRequest(user_id=user_id, budget_limit=limit, max_band=max_band)


def _get_budget(user_id: str):
    return controller_pb2.Controller.GetUserBudgetRequest(user_id=user_id)


def _launch(name: str, band: int = 0):
    return _launch_request(
        name,
        cpu_millicores=1000,
        memory_bytes=GiB,
        band=band,
    )


def test_admin_sets_and_reads_budget(service):
    _as_admin(service.set_user_budget, _set_budget("alice", 5000, INTERACTIVE), None)
    resp = _as_admin(service.get_user_budget, _get_budget("alice"), None)
    assert resp.user_id == "alice"
    assert resp.budget_limit == 5000
    assert resp.max_band == INTERACTIVE
    assert resp.budget_spent == 0


def test_non_admin_cannot_set_budget(service):
    with pytest.raises(ConnectError) as exc:
        _as_user(service.set_user_budget, "alice", _set_budget("alice"), None)
    assert exc.value.code == Code.PERMISSION_DENIED


def test_user_can_read_own_budget(service):
    """get_user_budget requires identity, not admin."""
    _as_admin(service.set_user_budget, _set_budget("alice", 5000), None)
    resp = _as_user(service.get_user_budget, "alice", _get_budget("alice"), None)
    assert resp.user_id == "alice"
    assert resp.budget_limit == 5000


def test_get_budget_not_found(service):
    with pytest.raises(ConnectError) as exc:
        _as_admin(service.get_user_budget, _get_budget("nonexistent"), None)
    assert exc.value.code == Code.NOT_FOUND


def test_set_budget_rejects_invalid_max_band(service):
    with pytest.raises(ConnectError) as exc:
        _as_admin(service.set_user_budget, _set_budget("alice", 5000, max_band=99), None)
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_set_budget_rejects_empty_user_id(service):
    with pytest.raises(ConnectError) as exc:
        _as_admin(service.set_user_budget, _set_budget(""), None)
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_list_user_budgets(service):
    for user_id, limit, band in [
        ("alice", 5000, INTERACTIVE),
        ("bob", 3000, BATCH),
        ("charlie", 0, PRODUCTION),
    ]:
        _as_admin(service.set_user_budget, _set_budget(user_id, limit, band), None)

    resp = _as_admin(
        service.list_user_budgets,
        controller_pb2.Controller.ListUserBudgetsRequest(),
        None,
    )
    by_user = {u.user_id: u for u in resp.users}
    assert set(by_user) == {"alice", "bob", "charlie"}
    assert by_user["alice"].budget_limit == 5000
    assert by_user["bob"].max_band == BATCH
    assert by_user["charlie"].max_band == PRODUCTION


def test_non_admin_cannot_submit_production(service):
    with pytest.raises(ConnectError) as exc:
        _as_user(service.launch_job, "alice", _launch("/alice/prod-job", band=PRODUCTION), None)
    assert exc.value.code == Code.PERMISSION_DENIED


def test_admin_can_submit_production(service):
    resp = _as_admin(service.launch_job, _launch("/admin/prod-job", band=PRODUCTION), None)
    assert resp.job_id == "/admin/prod-job"


def test_launch_job_rejects_band_above_user_max(service):
    """User with max_band=BATCH cannot submit INTERACTIVE (numerically lower) jobs."""
    _as_admin(service.set_user_budget, _set_budget("alice", 0, BATCH), None)
    with pytest.raises(ConnectError) as exc:
        _as_user(
            service.launch_job,
            "alice",
            _launch("/alice/interactive-job", band=INTERACTIVE),
            None,
        )
    assert exc.value.code == Code.PERMISSION_DENIED
    assert "cannot submit" in str(exc.value.message).lower()


def test_launch_job_unspecified_band_accepted(service):
    """Submitting with band=0 (UNSPECIFIED) is accepted and defaults to INTERACTIVE."""
    resp = _as_user(service.launch_job, "alice", _launch("/alice/default-band-job", band=0), None)
    assert resp.job_id == "/alice/default-band-job"


def test_get_budget_spend_reflects_running_task(service, state):
    _as_admin(service.set_user_budget, _set_budget("alice", 10000, INTERACTIVE), None)
    _start_running_job(state, "alice", "running-job", cpu_millicores=1000, memory_bytes=GiB)
    resp = _as_admin(service.get_user_budget, _get_budget("alice"), None)
    assert resp.budget_spent > 0
