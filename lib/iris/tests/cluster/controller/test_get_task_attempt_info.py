# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GetTaskAttemptInfo RPC."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.server import LogServiceImpl
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import (
    Assignment,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import VerifiedIdentity, _verified_identity
from rigging.timing import Timestamp

from tests.cluster.conftest import fake_log_client_from_service

from .conftest import (
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)


def _register(state, worker_id: str) -> WorkerId:
    return register_worker(
        state,
        worker_id=worker_id,
        address=f"{worker_id}:8080",
        metadata=make_worker_metadata(),
    )


def _assign_task(state, task_id: JobName, worker_id: WorkerId) -> None:
    with state._store.transaction() as cur:
        result = state.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=worker_id)])
    assert not result.rejected, f"queue_assignments rejected: {result.rejected}"


def test_get_task_attempt_info_happy_path(controller_service, state):
    """The RPC returns a populated RunTaskRequest for an assigned task."""
    worker_id = _register(state, "w-1")
    submit_job(state, "happy-job", make_job_request("happy-job"))
    job_id = JobName.root("test-user", "happy-job")
    task_id = JobName.from_wire(job_id.to_wire() + "/0")
    _assign_task(state, task_id, worker_id)

    with state._db.read_snapshot() as snap:
        expected = state.run_request_for_attempt(snap, task_id, 0)
    assert expected is not None

    response = controller_service.get_task_attempt_info(
        controller_pb2.Controller.GetTaskAttemptInfoRequest(task_id=task_id.to_wire(), attempt_id=0),
        None,
    )

    assert response.task_id == task_id.to_wire()
    assert response.attempt_id == 0
    assert response.entrypoint == expected.entrypoint
    assert response.environment == expected.environment
    assert response.bundle_id == expected.bundle_id


def test_get_task_attempt_info_not_found_for_unknown_task(controller_service):
    """Unknown task_id surfaces NOT_FOUND."""
    bogus = JobName.from_wire("/test-user/no-such-job/0")
    with pytest.raises(ConnectError) as exc_info:
        controller_service.get_task_attempt_info(
            controller_pb2.Controller.GetTaskAttemptInfoRequest(task_id=bogus.to_wire(), attempt_id=0),
            None,
        )
    assert exc_info.value.code == Code.NOT_FOUND


def test_get_task_attempt_info_stale_attempt(controller_service, state):
    """Asking for a stale attempt_id surfaces FAILED_PRECONDITION.

    Reproduce by assigning, marking the attempt PREEMPTED with retries
    remaining (puts the task back to PENDING), then re-assigning to bump
    current_attempt_id to 1. Fetching attempt_id=0 is now stale.
    """
    worker_id = _register(state, "w-1")
    request = make_job_request("stale-job", max_retries_preemption=5)
    submit_job(state, "stale-job", request)
    job_id = JobName.root("test-user", "stale-job")
    task_id = JobName.from_wire(job_id.to_wire() + "/0")

    _assign_task(state, task_id, worker_id)
    # Transition to RUNNING so preempt_task sees an active state.
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=worker_id,
                updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
            ),
        )
    # Preempt: with retries remaining the task returns to PENDING so we can
    # re-assign it for a fresh attempt_id.
    with state._store.transaction() as cur:
        state.preempt_task(cur, task_id, reason="test stale-attempt rotation")
    # The producer-side preempt leaves the prior attempt unfinalized; that's
    # fine for this test — we only need current_attempt_id to advance past 0.
    # Also finalize the prior attempt so the resource is released before the
    # second assignment.
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=worker_id,
                updates=[
                    TaskUpdate(
                        task_id=task_id,
                        attempt_id=0,
                        new_state=job_pb2.TASK_STATE_PREEMPTED,
                    )
                ],
            ),
        )
    _assign_task(state, task_id, worker_id)

    with state._db.read_snapshot() as snap:
        assert state._store.tasks.get_current_attempt_id(snap, task_id) == 1

    with pytest.raises(ConnectError) as exc_info:
        controller_service.get_task_attempt_info(
            controller_pb2.Controller.GetTaskAttemptInfoRequest(task_id=task_id.to_wire(), attempt_id=0),
            None,
        )
    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert "current_attempt_id=1" in exc_info.value.message

    # The current attempt is fetchable.
    current = controller_service.get_task_attempt_info(
        controller_pb2.Controller.GetTaskAttemptInfoRequest(task_id=task_id.to_wire(), attempt_id=1),
        None,
    )
    assert current.attempt_id == 1
    assert current.task_id == task_id.to_wire()


def test_get_task_attempt_info_same_name_replacement_returns_new_spec(controller_service, state, monkeypatch):
    """Same-name replacement returns the new submission's spec, not the old one.

    Regression guard for the staleness window that opened when an earlier
    revision of this RPC kept a per-(task_id, attempt_id) cache without an
    invalidation hook.
    """
    worker_id = _register(state, "w-1")

    request_v1 = make_job_request("replace-job")
    request_v1.environment.env_vars["SECRET"] = "v1"
    request_v1.entrypoint.run_command.argv[:] = ["echo", "v1"]
    submit_job(state, "replace-job", request_v1)
    job_id = JobName.root("test-user", "replace-job")
    task_id = JobName.from_wire(job_id.to_wire() + "/0")
    _assign_task(state, task_id, worker_id)

    fetch = controller_pb2.Controller.GetTaskAttemptInfoRequest(
        task_id=task_id.to_wire(),
        attempt_id=0,
    )

    first = controller_service.get_task_attempt_info(fetch, None)
    assert first.environment.env_vars["SECRET"] == "v1"
    assert list(first.entrypoint.run_command.argv) == ["echo", "v1"]

    # Replace the job under the same name with new env_vars + entrypoint.
    # ``submit_job`` short-circuits if a row already exists, so emulate the
    # service-layer RECREATE path: cancel + remove the old job, then resubmit.
    with state._store.transaction() as cur:
        state.cancel_job(cur, job_id, "test replacement")
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=worker_id,
                updates=[
                    TaskUpdate(
                        task_id=task_id,
                        attempt_id=0,
                        new_state=job_pb2.TASK_STATE_KILLED,
                    )
                ],
            ),
        )
    with state._store.transaction() as cur:
        state.remove_finished_job(cur, job_id)

    request_v2 = make_job_request("replace-job")
    request_v2.environment.env_vars["SECRET"] = "v2"
    request_v2.entrypoint.run_command.argv[:] = ["echo", "v2"]
    submit_job(state, "replace-job", request_v2)
    _assign_task(state, task_id, worker_id)

    second = controller_service.get_task_attempt_info(fetch, None)
    assert (
        second.environment.env_vars["SECRET"] == "v2"
    ), "Same-name replacement must surface the new submission's env_vars; stale cache would have returned 'v1'"
    assert list(second.entrypoint.run_command.argv) == ["echo", "v2"]


def test_get_task_attempt_info_rejects_unauthorized_caller(state, mock_controller, tmp_path):
    """Non-worker, non-admin identity gets PERMISSION_DENIED.

    ``RunTaskRequest`` embeds ``environment.env_vars``, into which Fray
    injects secrets like ``HF_TOKEN`` and ``WANDB_API_KEY``. Only workers
    (and admins) may fetch it.
    """
    db = state._db
    now = Timestamp.now()
    db.ensure_user("alice", now, role="user")
    db.ensure_user("system:worker", now, role="worker")

    service = ControllerServiceImpl(
        state,
        state._store,
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=fake_log_client_from_service(LogServiceImpl()),
        auth=ControllerAuth(provider="static"),
    )

    # Worker role identity is required to land a job under that user.
    worker_token = _verified_identity.set(VerifiedIdentity(user_id="system:worker", role="worker"))
    try:
        worker_id = _register(state, "w-1")
        submit_job(state, "auth-job", make_job_request("auth-job"))
        job_id = JobName.root("test-user", "auth-job")
        task_id = JobName.from_wire(job_id.to_wire() + "/0")
        _assign_task(state, task_id, worker_id)
    finally:
        _verified_identity.reset(worker_token)

    fetch = controller_pb2.Controller.GetTaskAttemptInfoRequest(
        task_id=task_id.to_wire(),
        attempt_id=0,
    )

    # Non-worker, non-admin must be rejected.
    user_token = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        with pytest.raises(ConnectError) as exc_info:
            service.get_task_attempt_info(fetch, None)
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(user_token)

    # Worker identity passes the gate.
    worker_token = _verified_identity.set(VerifiedIdentity(user_id="system:worker", role="worker"))
    try:
        resp = service.get_task_attempt_info(fetch, None)
        assert resp.task_id == task_id.to_wire()
    finally:
        _verified_identity.reset(worker_token)
