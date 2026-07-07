# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the KickTasks administrative override.

Two layers are exercised:

- The control-loop mechanism: a kick queued via ``Controller.request_task_kicks``
  is buffered and only applied on the next ``_control_tick``, finalizing the
  task's current attempt through the same path the scheduler's preemptions use.
- The ``kick_tasks`` RPC handler: target resolution and validation, observable
  through the per-target ``KickResult`` rows it returns.
"""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller import ops
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalKind
from iris.cluster.controller.service import PendingKick
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations

from .conftest import (
    make_direct_job_request,
    make_worker_metadata,
    query_attempt,
    query_task,
    query_tasks_for_job,
    reconcile_once,
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
# Control-loop mechanism (real Controller, kick -> tick -> finalize)
# =============================================================================


def _running_task_on_controller(ctrl, request=None):
    """Submit a single-replica job, register a worker, and drive its task to RUNNING."""
    state = ControllerTestState(
        ctrl._db,
        health=ctrl.provider.health,
    )
    register_worker(state, "w0", "10.0.0.1", make_worker_metadata())
    submit_job(state, "job-a", request or make_direct_job_request("job-a", replicas=1))
    ctrl._run_scheduling()
    task_id = JobName.root(JOB_USER, "job-a").task(0)
    [task] = query_tasks_for_job(state, JobName.root(JOB_USER, "job-a"))
    _observe(state, task.current_worker_id, task_id, task.current_attempt_id, job_pb2.TASK_STATE_RUNNING)
    return state, task_id


def test_kick_is_buffered_until_next_tick(make_controller):
    ctrl = make_controller(remote_state_dir="file:///tmp/iris-kick-buffer")
    state, task_id = _running_task_on_controller(ctrl)
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING

    ctrl.request_task_kicks([PendingKick(task_id=task_id, attempt_id=None, kind=TerminalKind.PREEMPT, reason="x")])
    # Queued, not applied: the task stays RUNNING until a control tick drains the buffer.
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING

    reconcile_once(ctrl)
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_PREEMPTED


def test_kick_preempt_with_budget_requeues_for_retry(make_controller):
    ctrl = make_controller(remote_state_dir="file:///tmp/iris-kick-retry")
    request = make_direct_job_request("job-a", replicas=1)
    request.max_retries_preemption = 3
    state, task_id = _running_task_on_controller(ctrl, request)

    ctrl.request_task_kicks([PendingKick(task_id=task_id, attempt_id=None, kind=TerminalKind.PREEMPT, reason="x")])
    reconcile_once(ctrl)

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.preemption_count == 1
    # The kicked attempt itself is recorded as PREEMPTED; the retry runs as a new attempt.
    assert query_attempt(state, task_id, 0).state == job_pb2.TASK_STATE_PREEMPTED


def test_kick_failed_finalizes_without_retry(make_controller):
    ctrl = make_controller(remote_state_dir="file:///tmp/iris-kick-failed")
    state, task_id = _running_task_on_controller(ctrl)

    ctrl.request_task_kicks([PendingKick(task_id=task_id, attempt_id=None, kind=TerminalKind.TIMEOUT, reason="x")])
    reconcile_once(ctrl)

    assert query_task(state, task_id).state == job_pb2.TASK_STATE_FAILED


def test_kick_targeting_superseded_attempt_is_dropped(make_controller):
    """A kick naming a specific attempt is dropped once that attempt is no longer
    current, so it cannot land on a retry that started after the RPC."""
    ctrl = make_controller(remote_state_dir="file:///tmp/iris-kick-superseded")
    state, task_id = _running_task_on_controller(ctrl)
    current = query_task(state, task_id).current_attempt_id

    ctrl.request_task_kicks(
        [PendingKick(task_id=task_id, attempt_id=current + 1, kind=TerminalKind.PREEMPT, reason="stale")]
    )
    reconcile_once(ctrl)

    # The targeted attempt was never current, so the task is untouched.
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING

    # Targeting the actual current attempt does take effect.
    ctrl.request_task_kicks([PendingKick(task_id=task_id, attempt_id=current, kind=TerminalKind.PREEMPT, reason="x")])
    reconcile_once(ctrl)
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_PREEMPTED


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


def test_kick_handler_queues_running_task(controller_service, state):
    job_id = _two_task_job_with_one_running(state)
    response = _kick(controller_service, [job_id.task(0).to_wire()])

    assert len(response.results) == 1
    result = response.results[0]
    assert result.queued
    assert result.task_id == job_id.task(0).to_wire()


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
