# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from google.protobuf import any_pb2
from iris.cluster.controller.composition import wire_resource_service
from iris.cluster.controller.persistence.pruning import prune_old_data
from iris.resources.action import ActionKind, ActionResult, ActionState
from iris.resources.errors import ResourceNotFound, ResourceReplaced
from iris.resources.identity import AttemptIdentity
from iris.rpc import job_pb2, resource_action_pb2, resource_pb2, resource_task_pb2
from iris.rpc.resource_service import ResourceServiceImpl
from iris.rpc.resource_types import ATTEMPT, TASK
from rigging.timing import Duration


def _service(journey) -> ResourceServiceImpl:
    return wire_resource_service(journey.controller.controller)


def _pack(value) -> any_pb2.Any:
    result = any_pb2.Any()
    result.Pack(value)
    return result


def _current_task_update(
    task_id: str,
    *,
    request_id: str,
    update: resource_task_pb2.TaskUpdate,
    reason: str = "",
) -> resource_pb2.UpdateResourceRequest:
    return resource_pb2.UpdateResourceRequest(
        mutation=resource_pb2.MutationMetadata(request_id=request_id, reason=reason),
        ref=resource_pb2.ResourceRef(
            authority_cluster_id="journey",
            type=TASK,
            id=task_id,
        ),
        update=_pack(update),
    )


def _exact_attempt_update(
    identity: AttemptIdentity,
    *,
    request_id: str,
    update: resource_task_pb2.AttemptUpdate,
) -> resource_pb2.UpdateResourceRequest:
    return resource_pb2.UpdateResourceRequest(
        mutation=resource_pb2.MutationMetadata(request_id=request_id),
        ref=resource_pb2.ResourceRef(
            authority_cluster_id=identity.task.cluster_id,
            type=ATTEMPT,
            id=f"{identity.task.resource_id}:{identity.attempt_number}",
            uid=identity.attempt_uid,
        ),
        update=_pack(update),
    )


def test_cancel_receipt_survives_restart_and_duplicate_request(journey):
    job = journey.submit("cancel-restart", tasks=2)
    journey.settle()
    identity = journey.job(job).summary.identity

    accepted = journey.cancel_job(identity, idempotency_key="cancel-restart")
    journey.restart()
    duplicate = journey.cancel_job(identity, idempotency_key="cancel-restart")
    completed = journey.settle_action(duplicate)

    assert duplicate.action_id == accepted.action_id
    assert completed.kind is ActionKind.CANCEL_JOB
    assert completed.state is ActionState.SUCCEEDED
    assert completed.result_code is ActionResult.SATISFIED
    assert journey.job(job).summary.state == job_pb2.JOB_STATE_KILLED


def test_pruning_a_job_removes_its_action_receipts(journey):
    job = journey.submit("pruned-action")
    journey.settle()
    receipt = journey.cancel_job(journey.job(job).summary.identity, idempotency_key="pruned-action")
    completed = journey.settle_action(receipt)
    journey.clock.advance(2 * 86_400)

    result = prune_old_data(
        journey.database,
        journey.backends.values(),
        job_retention=Duration.from_seconds(86_400),
        worker_retention=Duration.from_seconds(86_400),
        slice_retention=Duration.from_seconds(86_400),
        pause_between_s=0,
    )

    assert result.jobs_deleted == 1
    with pytest.raises(ResourceNotFound):
        journey.action_receipt(completed.action_id)


def test_retry_receipt_replaces_exact_attempt_and_survives_restart(journey):
    job = journey.submit("retry-exact", preemption_retries=1)
    journey.settle()
    before = journey.task(job[0])
    identity = before.summary.identity
    current = before.summary.current_attempt
    assert current is not None

    accepted = journey.retry_task(
        identity,
        expected_attempt_uid=current.attempt_uid,
        idempotency_key="retry-exact",
    )
    completed = journey.settle_action(accepted)
    journey.settle()
    journey.restart()
    duplicate = journey.retry_task(
        identity,
        expected_attempt_uid=current.attempt_uid,
        idempotency_key="retry-exact",
    )
    after = journey.task(job[0])

    assert duplicate.action_id == accepted.action_id
    assert completed.kind is ActionKind.RETRY_TASK
    assert completed.state is ActionState.SUCCEEDED
    assert completed.result_code is ActionResult.TARGET_ABSENT
    assert after.summary.identity == identity
    assert after.summary.current_attempt is not None
    assert after.summary.current_attempt.attempt_uid != current.attempt_uid
    assert [(attempt.identity.attempt_uid, attempt.state) for attempt in after.attempts] == [
        (current.attempt_uid, job_pb2.TASK_STATE_PREEMPTED),
        (after.summary.current_attempt.attempt_uid, job_pb2.TASK_STATE_RUNNING),
    ]


def test_terminate_receipt_finishes_exact_attempt_without_retry(journey):
    job = journey.submit("terminate-exact", preemption_retries=3)
    journey.settle()
    current = journey.attempt(job[0]).summary.identity

    accepted = journey.terminate_attempt(current, idempotency_key="terminate-exact")
    completed = journey.settle_action(accepted)
    journey.settle()
    after = journey.task(job[0])

    assert completed.kind is ActionKind.TERMINATE_ATTEMPT
    assert completed.state is ActionState.SUCCEEDED
    assert completed.result_code is ActionResult.SATISFIED
    assert completed.expected_attempt_uid == current.attempt_uid
    assert len(after.attempts) == 1
    assert after.attempts[0].identity == current
    assert after.attempts[0].state == job_pb2.TASK_STATE_KILLED


def test_replaced_job_rejects_stale_job_and_task_actions_without_mutating_current_run(journey):
    original = journey.submit("resource-replacement", preemption_retries=1)
    journey.settle()
    original_summary = journey.job(original).summary
    original_job = original_summary.identity
    original_task = journey.task(original[0]).summary
    original_attempt = original_task.current_attempt
    assert original_attempt is not None
    journey.succeed(original[0])
    journey.settle()
    journey.restart()

    journey.clock.epoch_ms = original_summary.submitted_at.epoch_ms()
    replacement = journey.submit("resource-replacement", preemption_retries=1)
    journey.settle()
    replacement_job = journey.job(replacement)
    replacement_task = journey.task(replacement[0])
    assert replacement_job.summary.identity != original_job
    assert replacement_task.summary.identity != original_task.identity
    backend_events = journey.backend_events()

    with pytest.raises(ResourceReplaced):
        journey.cancel_job(original_job, idempotency_key="stale-job")
    with pytest.raises(ResourceReplaced):
        journey.retry_task(
            original_task.identity,
            expected_attempt_uid=original_attempt.attempt_uid,
            idempotency_key="stale-task",
        )

    assert journey.job(replacement) == replacement_job
    after = journey.task(replacement[0])
    assert after.summary == replacement_task.summary
    assert after.attempts == replacement_task.attempts
    assert journey.backend_events() == backend_events


def test_generic_current_task_update_replays_before_resolving_a_new_attempt(journey) -> None:
    job = journey.submit("generic-current-replay", preemption_retries=2)
    journey.settle()
    before = journey.attempt(job[0]).summary.identity
    service = _service(journey)
    request = _current_task_update(
        job[0].wire_id,
        request_id="generic-current-replay",
        update=resource_task_pb2.TaskUpdate(preempt=resource_task_pb2.PreemptTaskUpdate()),
    )

    accepted = service.update_resource(request, None)
    journey.settle()
    replacement = journey.attempt(job[0]).summary.identity
    duplicate = service.update_resource(request, None)

    assert replacement.attempt_uid != before.attempt_uid
    assert duplicate.ref == accepted.ref
    assert duplicate.resolved_ref.type == TASK
    assert len(duplicate.affected) == 1
    assert duplicate.affected[0].type == "iris/attempt"
    assert duplicate.affected[0].id == f"{job[0].wire_id}:{before.attempt_number}"
    assert duplicate.affected[0].uid == before.attempt_uid
    receipt = resource_action_pb2.ActionReceipt.FromString(duplicate.result.value)
    assert receipt.action_id == accepted.ref.id
    assert receipt.expected_attempt_uid == before.attempt_uid
    assert receipt.expected_attempt_number == before.attempt_number


def test_generic_failed_task_update_fails_current_attempt_without_retry(journey) -> None:
    job = journey.submit("generic-force-fail", preemption_retries=3)
    journey.settle()
    before = journey.attempt(job[0]).summary.identity

    operation = _service(journey).update_resource(
        _current_task_update(
            job[0].wire_id,
            request_id="generic-force-fail",
            update=resource_task_pb2.TaskUpdate(fail=resource_task_pb2.FailTaskUpdate()),
            reason="operator diagnosed corrupt state",
        ),
        None,
    )
    journey.settle()
    task = journey.task(job[0])
    receipt = resource_action_pb2.ActionReceipt.FromString(operation.result.value)

    assert operation.requested_ref.type == TASK
    assert operation.resolved_ref.type == TASK
    assert operation.resolved_ref.uid == task.summary.identity.task_uid
    assert operation.affected[0].uid == before.attempt_uid
    assert receipt.kind == resource_action_pb2.ACTION_KIND_FAIL_ATTEMPT
    assert task.summary.state == job_pb2.TASK_STATE_FAILED
    assert len(task.attempts) == 1
    assert task.summary.error_message == "operator diagnosed corrupt state"


def test_generic_cancelled_task_update_terminates_its_exact_current_attempt(journey) -> None:
    job = journey.submit("generic-cancel-task", preemption_retries=3)
    journey.settle()
    before = journey.attempt(job[0]).summary.identity

    operation = _service(journey).update_resource(
        _current_task_update(
            job[0].wire_id,
            request_id="generic-cancel-task",
            update=resource_task_pb2.TaskUpdate(terminate=resource_task_pb2.TerminateTaskUpdate()),
        ),
        None,
    )
    journey.settle()

    task = journey.task(job[0])
    assert task.summary.state == job_pb2.TASK_STATE_KILLED
    assert len(task.attempts) == 1
    assert task.attempts[0].identity == before
    assert operation.affected[0].id == f"{job[0].wire_id}:{before.attempt_number}"
    assert operation.affected[0].uid == before.attempt_uid


def test_generic_preempted_attempt_update_preserves_retry_policy(journey) -> None:
    job = journey.submit("generic-preempt-attempt", preemption_retries=1)
    journey.settle()
    before = journey.attempt(job[0]).summary.identity
    request = _exact_attempt_update(
        before,
        request_id="generic-preempt-attempt",
        update=resource_task_pb2.AttemptUpdate(preempt=resource_task_pb2.PreemptAttemptUpdate()),
    )

    service = _service(journey)
    operation = service.update_resource(request, None)
    journey.settle()
    duplicate = service.update_resource(request, None)

    task = journey.task(job[0])
    assert task.summary.current_attempt is not None
    assert task.summary.current_attempt.attempt_uid != before.attempt_uid
    assert task.attempts[0].identity == before
    assert task.attempts[0].state == job_pb2.TASK_STATE_PREEMPTED
    assert operation.resolved_ref == operation.requested_ref
    assert operation.affected[0] == operation.requested_ref
    assert duplicate.ref == operation.ref
    assert duplicate.resolved_ref == operation.resolved_ref


def test_service_info_reports_backend_contributions_to_registered_resources(journey) -> None:
    response = _service(journey).get_service_info(resource_pb2.GetServiceInfoRequest(), None)
    installed = {
        (capability.backend_id, capability.type): tuple(capability.verbs) for capability in response.backend_resources
    }

    assert installed[("default", "iris/attempt")] == ("update",)
    assert installed[("default", "iris/exec-session")] == ("create",)
    assert installed[("default", "iris/profile-capture")] == ("create",)
