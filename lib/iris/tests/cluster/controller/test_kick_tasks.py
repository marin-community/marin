# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPC target resolution and validation for the KickTasks override."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller import ops
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations

from .conftest import (
    make_direct_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)

JOB_USER = "test-user"


def _observe(state, worker_id, task_id, attempt_id, new_state, error=None):
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=new_state, error=error)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )


def _assign_and_run(state, task_id, worker_id):
    """Drive a PENDING task to RUNNING on ``worker_id`` (attempt 0)."""
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=worker_id)], health=state._health)
    _observe(state, worker_id, task_id, 0, job_pb2.TASK_STATE_RUNNING)


# =============================================================================
# RPC handler: target resolution + validation
# =============================================================================


def _kick(controller_service, targets, desired_state=job_pb2.TASK_STATE_PREEMPTED, reason=""):
    return controller_service.kick_tasks(
        controller_pb2.Controller.KickTasksRequest(targets=targets, desired_state=desired_state, reason=reason),
        None,
    )


def _two_task_job_with_one_running(state):
    """Job with two tasks: task 0 RUNNING on a worker, task 1 left PENDING."""
    submit_job(state, "job-a", make_direct_job_request("job-a", replicas=2))
    register_worker(state, "w0", "10.0.0.1", make_worker_metadata())
    job_id = JobName.root(JOB_USER, "job-a")
    _assign_and_run(state, job_id.task(0), WorkerId("w0"))
    return job_id


def test_kick_handler_rejects_pending_task(controller_service, state):
    job_id = _two_task_job_with_one_running(state)
    response = _kick(controller_service, [job_id.task(1).to_wire()])

    assert not response.results[0].queued
    assert "not running on a worker" in response.results[0].detail


def test_kick_handler_rejects_missing_task(controller_service, state):
    job_id = _two_task_job_with_one_running(state)
    response = _kick(controller_service, [job_id.task(9).to_wire()])

    assert not response.results[0].queued
    assert "not found" in response.results[0].detail


def test_kick_handler_honors_matching_attempt(controller_service, state):
    job_id = _two_task_job_with_one_running(state)
    response = _kick(controller_service, [f"{job_id.task(0).to_wire()}:0"])

    assert response.results[0].queued


def test_kick_handler_rejects_stale_attempt(controller_service, state):
    job_id = _two_task_job_with_one_running(state)
    response = _kick(controller_service, [f"{job_id.task(0).to_wire()}:5"])

    assert not response.results[0].queued
    assert "not current" in response.results[0].detail


def test_kick_handler_job_target_expands_to_active_tasks(controller_service, state):
    job_id = _two_task_job_with_one_running(state)
    # The job has two tasks but only task 0 is active; the PENDING task 1 is excluded.
    response = _kick(controller_service, [job_id.to_wire()])

    queued = [r for r in response.results if r.queued]
    assert [r.task_id for r in queued] == [job_id.task(0).to_wire()]


def test_kick_handler_rejects_unsupported_state(controller_service, state):
    job_id = _two_task_job_with_one_running(state)
    with pytest.raises(ConnectError) as exc:
        _kick(controller_service, [job_id.task(0).to_wire()], desired_state=job_pb2.TASK_STATE_KILLED)
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_kick_handler_requires_a_target(controller_service, state):
    with pytest.raises(ConnectError) as exc:
        _kick(controller_service, [])
    assert exc.value.code == Code.INVALID_ARGUMENT
