# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GetTaskAttemptInfo RPC and its per-attempt LRU cache.

The RPC returns the cached ``RunTaskRequest`` that the controller would have
dispatched for the given ``(task_id, attempt_id)``. Reconcile-only workers
poll this after learning about a new attempt from ``PollTasks``.
"""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2

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


def test_get_task_attempt_info_cache_hit(controller_service, state, monkeypatch):
    """A second call for the same key serves from the LRU and skips DB lookup."""
    worker_id = _register(state, "w-1")
    submit_job(state, "cache-job", make_job_request("cache-job"))
    job_id = JobName.root("test-user", "cache-job")
    task_id = JobName.from_wire(job_id.to_wire() + "/0")
    _assign_task(state, task_id, worker_id)

    request = controller_pb2.Controller.GetTaskAttemptInfoRequest(task_id=task_id.to_wire(), attempt_id=0)

    # Prime the cache.
    first = controller_service.get_task_attempt_info(request, None)
    assert first.task_id == task_id.to_wire()

    # Count cache-miss DB lookups via the store's get_current_attempt_id.
    calls = {"count": 0}
    real_get = state._store.tasks.get_current_attempt_id

    def counting_get(tx, tid):
        calls["count"] += 1
        return real_get(tx, tid)

    monkeypatch.setattr(state._store.tasks, "get_current_attempt_id", counting_get)

    second = controller_service.get_task_attempt_info(request, None)
    assert second.SerializeToString() == first.SerializeToString()
    assert calls["count"] == 0, "Cache hit should not trigger a DB lookup"
