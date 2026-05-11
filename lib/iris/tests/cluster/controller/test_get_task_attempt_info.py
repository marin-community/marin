# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GetTaskAttemptInfo RPC.

The RPC returns the ``RunTaskRequest`` the controller would have dispatched
for the given ``(task_id, attempt_id)``. Reconcile-only workers poll this
after learning about a new attempt from ``PollTasks``. The expensive
per-job assembly is memoized by ``ControllerTransitions._run_template_cache``
(invalidated on same-name replacement in ``submit_job``); per-call
``stamp_attempt_onto_template`` is sub-ms, so no per-attempt cache is needed.
"""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.server import LogServiceImpl
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
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


def _build_dispatch_request(state, task_id: JobName, attempt_id: int) -> job_pb2.RunTaskRequest:
    """Build the RunTaskRequest the dispatch loop would produce for this row."""
    with state._db.read_snapshot() as snap:
        job_id = state._store.tasks.get_job_id(snap, task_id)
        assert job_id is not None
        template = state.run_request_template(snap, job_id)
        assert template is not None
    return ControllerTransitions.stamp_attempt_onto_template(template, task_id, attempt_id)


def test_get_task_attempt_info_happy_path(controller_service, state):
    """The RPC returns a RunTaskRequest matching what the dispatch loop would build."""
    worker_id = _register(state, "w-1")
    submit_job(state, "happy-job", make_job_request("happy-job"))
    job_id = JobName.root("test-user", "happy-job")
    task_id = JobName.from_wire(job_id.to_wire() + "/0")
    _assign_task(state, task_id, worker_id)

    # After assignment, current_attempt_id == 0.
    expected = _build_dispatch_request(state, task_id, 0)

    response = controller_service.get_task_attempt_info(
        controller_pb2.Controller.GetTaskAttemptInfoRequest(task_id=task_id.to_wire(), attempt_id=0),
        None,
    )

    assert response.task_id == expected.task_id
    assert response.attempt_id == expected.attempt_id
    assert response.resources == expected.resources
    assert response.environment == expected.environment
    assert response.entrypoint == expected.entrypoint
    assert response.bundle_id == expected.bundle_id
    assert list(response.ports) == list(expected.ports)


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


def test_get_task_attempt_info_reuses_per_job_template_cache(controller_service, state, monkeypatch):
    """Repeated calls share the per-job ``_run_template_cache``: the expensive
    job_config / workdir_files reads + JSON parsing run only once.

    The stale-attempt check (``tasks.get_current_attempt_id``) and the cheap
    template-stamp run on every call; the per-job template assembly does not.
    """
    worker_id = _register(state, "w-1")
    submit_job(state, "cache-job", make_job_request("cache-job"))
    job_id = JobName.root("test-user", "cache-job")
    task_id = JobName.from_wire(job_id.to_wire() + "/0")
    _assign_task(state, task_id, worker_id)

    request = controller_pb2.Controller.GetTaskAttemptInfoRequest(task_id=task_id.to_wire(), attempt_id=0)

    # Count calls into the heavy DB read that ``run_request_template`` issues
    # on a per-job cache miss. After the first call the per-job cache is warm
    # and subsequent calls must not re-issue the read.
    calls = {"detail": 0}
    real_get_detail = state._store.jobs.get_detail

    def counting_get_detail(tx, jid):
        calls["detail"] += 1
        return real_get_detail(tx, jid)

    monkeypatch.setattr(state._store.jobs, "get_detail", counting_get_detail)

    first = controller_service.get_task_attempt_info(request, None)
    assert first.task_id == task_id.to_wire()
    assert calls["detail"] == 1, "First call should hit the DB once"

    second = controller_service.get_task_attempt_info(request, None)
    assert second.SerializeToString() == first.SerializeToString()
    assert calls["detail"] == 1, "Per-job template cache should suppress repeated job_config reads"


def test_get_task_attempt_info_evicts_cached_template_on_same_name_replacement(controller_service, state, monkeypatch):
    """Same-name RECREATE replacement returns the NEW submission's spec.

    Codex review repro: with a per-(task_id, attempt_id) cache and no
    invalidation hook, the second fetch for the same key would serve the
    previous job's ``RunTaskRequest`` (including its env_vars). The
    per-job ``_run_template_cache`` is correctly evicted on ``submit_job``,
    so dropping the per-attempt cache closes the staleness window.
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
