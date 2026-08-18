# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf import any_pb2
from iris.rpc import job_pb2, resource_job_pb2, resource_pb2, resource_task_pb2
from iris.testing.journeys.backend import ScriptedObservation


def _update(ref, request_id: str, body, *, reason: str = "") -> resource_pb2.UpdateResourceRequest:
    packed = any_pb2.Any()
    packed.Pack(body)
    return resource_pb2.UpdateResourceRequest(
        mutation=resource_pb2.MutationMetadata(request_id=request_id, reason=reason),
        ref=ref,
        update=packed,
    )


def _operation(journey, ref: resource_pb2.ResourceRef) -> resource_pb2.Operation:
    response = journey.controller.get_resource(resource_pb2.GetResourceRequest(ref=ref))
    operation = resource_pb2.Operation()
    assert response.resource.body.Unpack(operation)
    return operation


def test_task_preemption_operation_survives_restart_and_verifies_runtime_release(journey):
    job = journey.submit("durable-preempt", preemption_retries=1)
    journey.settle()
    before = journey.task(job[0])
    before_uid = before.attempts[-1].attempt_uid
    request = _update(
        resource_pb2.ResourceRef(
            authority_cluster_id="journey",
            type="iris/task",
            id=job[0].wire_id,
        ),
        "durable-preempt",
        resource_task_pb2.TaskUpdate(preempt=resource_task_pb2.PreemptAttempt()),
    )

    accepted = journey.controller.update_resource(request)

    assert accepted.phase == resource_pb2.OPERATION_PHASE_VERIFYING
    assert [(ref.id, ref.uid) for ref in accepted.affected] == [(f"{job[0].wire_id}:0", before_uid)]

    journey.settle()
    journey.restart()
    duplicate = journey.controller.update_resource(request)
    completed = _operation(journey, accepted.ref)
    after = journey.task(job[0])

    with pytest.raises(ConnectError) as reused:
        journey.controller.update_resource(
            _update(
                request.ref,
                request.mutation.request_id,
                resource_task_pb2.TaskUpdate(terminate=resource_task_pb2.TerminateAttempt()),
            )
        )

    assert duplicate.ref == accepted.ref
    assert reused.value.code == Code.ALREADY_EXISTS
    assert completed.phase == resource_pb2.OPERATION_PHASE_VERIFIED
    assert after.state == job_pb2.TASK_STATE_RUNNING
    assert [attempt.state for attempt in after.attempts] == [
        job_pb2.TASK_STATE_PREEMPTED,
        job_pb2.TASK_STATE_RUNNING,
    ]
    assert [(event.task_id, event.attempt_id) for event in journey.backend_events(kind="stopped")] == [
        (job[0].wire_id, 0)
    ]


def test_exact_attempt_update_cannot_follow_a_retry(journey):
    job = journey.submit("stale-exact-attempt", preemption_retries=1)
    journey.settle()
    initial = journey.task(job[0]).attempts[-1]
    journey.preempt(job[0])
    journey.settle()
    events = journey.backend_events()

    with pytest.raises(ConnectError) as raised:
        journey.controller.update_resource(
            _update(
                resource_pb2.ResourceRef(
                    authority_cluster_id="journey",
                    type="iris/attempt",
                    id=f"{job[0].wire_id}:{initial.attempt_id}",
                    uid=initial.attempt_uid,
                ),
                "stale-exact-attempt",
                resource_task_pb2.AttemptUpdate(fail=resource_task_pb2.FailAttempt()),
            )
        )

    assert raised.value.code == Code.FAILED_PRECONDITION
    assert journey.task(job[0]).state == job_pb2.TASK_STATE_RUNNING
    assert journey.backend_events() == events


def test_job_cancel_operation_is_idempotent_and_fenced_from_replacement(journey):
    original = journey.submit("replace-after-cancel", tasks=2)
    journey.settle()
    request = _update(
        resource_pb2.ResourceRef(
            authority_cluster_id="journey",
            type="iris/job",
            id=original.wire_id,
        ),
        "cancel-original",
        resource_job_pb2.JobUpdate(cancel=resource_job_pb2.CancelJob()),
        reason="operator cancel",
    )
    accepted = journey.controller.update_resource(request)

    assert accepted.phase == resource_pb2.OPERATION_PHASE_VERIFYING
    assert journey.job(original).state == job_pb2.JOB_STATE_KILLED

    journey.settle()
    completed = _operation(journey, accepted.ref)
    assert len(journey.backend_events(kind="stopped")) == 2
    replacement = journey.submit("replace-after-cancel")
    journey.settle()
    duplicate = journey.controller.update_resource(request)

    assert completed.phase == resource_pb2.OPERATION_PHASE_VERIFIED
    assert duplicate.ref == accepted.ref
    assert journey.job(replacement).state == job_pb2.JOB_STATE_RUNNING

    with pytest.raises(ConnectError) as raised:
        journey.controller.update_resource(
            _update(
                accepted.resolved_ref,
                "cancel-stale-incarnation",
                resource_job_pb2.JobUpdate(cancel=resource_job_pb2.CancelJob()),
            )
        )

    assert raised.value.code == Code.FAILED_PRECONDITION
    assert journey.job(replacement).state == job_pb2.JOB_STATE_RUNNING


def test_task_update_during_reconcile_discards_the_stale_result(journey):
    job = journey.submit("update-during-reconcile", preemption_retries=1)
    journey.settle()
    task = job[0]
    journey.backend.observe(task.wire_id, ScriptedObservation(job_pb2.TASK_STATE_RUNNING))
    reconcile_started = threading.Event()
    release_reconcile = threading.Event()
    journey.backend.pause_next_reconcile(started=reconcile_started, release=release_reconcile)
    tick_errors: list[Exception] = []
    action_errors: list[Exception] = []
    operations: list[resource_pb2.Operation] = []

    def run_tick() -> None:
        try:
            journey.step()
        except Exception as error:
            tick_errors.append(error)

    def update_task() -> None:
        try:
            operations.append(
                journey.controller.update_resource(
                    _update(
                        resource_pb2.ResourceRef(
                            authority_cluster_id="journey",
                            type="iris/task",
                            id=task.wire_id,
                        ),
                        "update-during-reconcile",
                        resource_task_pb2.TaskUpdate(preempt=resource_task_pb2.PreemptAttempt()),
                    )
                )
            )
        except Exception as error:
            action_errors.append(error)

    tick = threading.Thread(target=run_tick)
    tick.start()
    assert reconcile_started.wait(timeout=5)
    action = threading.Thread(target=update_task)
    action.start()
    action.join(timeout=1)
    action_completed_while_reconcile_was_blocked = not action.is_alive()
    release_reconcile.set()
    action.join(timeout=5)
    tick.join(timeout=5)

    assert action_completed_while_reconcile_was_blocked
    assert not action.is_alive()
    assert not tick.is_alive()
    assert not action_errors
    assert not tick_errors
    journey.settle()
    completed = _operation(journey, operations[0].ref)
    attempts = journey.task(task).attempts
    assert completed.phase == resource_pb2.OPERATION_PHASE_VERIFIED
    assert [attempt.state for attempt in attempts] == [
        job_pb2.TASK_STATE_PREEMPTED,
        job_pb2.TASK_STATE_RUNNING,
    ]


def test_operation_fails_when_worker_disappears_before_release_is_confirmed(journey):
    job = journey.submit("worker-gone-during-terminate")
    journey.settle()
    attempt_uid = journey.task(job[0]).attempts[-1].attempt_uid
    journey.backend.lose_runtime(attempt_uid)

    accepted = journey.controller.update_resource(
        _update(
            resource_pb2.ResourceRef(
                authority_cluster_id="journey",
                type="iris/task",
                id=job[0].wire_id,
            ),
            "worker-gone-during-terminate",
            resource_task_pb2.TaskUpdate(terminate=resource_task_pb2.TerminateAttempt()),
        )
    )
    journey.step()
    completed = _operation(journey, accepted.ref)

    assert completed.phase == resource_pb2.OPERATION_PHASE_FAILED
    assert completed.error.code == Code.UNAVAILABLE.value
    assert journey.task(job[0]).state == job_pb2.TASK_STATE_KILLED
