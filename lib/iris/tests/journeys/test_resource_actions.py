# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.resources.action import ActionKind, ActionResult, ActionState
from iris.rpc import job_pb2


def test_cancel_receipt_survives_restart_and_duplicate_request(journey):
    job = journey.submit("cancel-restart", tasks=2)
    journey.settle()
    identity = journey.resource_job(job).summary.identity

    accepted = journey.cancel_resource_job(identity, idempotency_key="cancel-restart")
    journey.restart()
    duplicate = journey.cancel_resource_job(identity, idempotency_key="cancel-restart")
    completed = journey.settle_action(duplicate)

    assert duplicate.action_id == accepted.action_id
    assert completed.kind is ActionKind.CANCEL_JOB
    assert completed.state is ActionState.SUCCEEDED
    assert completed.result_code is ActionResult.SATISFIED
    assert journey.job(job).state == job_pb2.JOB_STATE_KILLED


def test_retry_receipt_replaces_only_the_exact_current_attempt_and_is_idempotent(journey):
    job = journey.submit("retry-exact", preemption_retries=1)
    journey.settle()
    before = journey.resource_task(job[0])
    identity = before.summary.identity
    current = before.summary.current_attempt
    assert current is not None

    accepted = journey.retry_resource_task(
        identity,
        expected_attempt_uid=current.attempt_uid,
        idempotency_key="retry-exact",
    )
    completed = journey.settle_action(accepted)
    journey.settle()
    duplicate = journey.retry_resource_task(
        identity,
        expected_attempt_uid=current.attempt_uid,
        idempotency_key="retry-exact",
    )
    after = journey.resource_task(job[0])

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
    current = journey.resource_attempt(job[0]).summary.identity

    accepted = journey.terminate_resource_attempt(current, idempotency_key="terminate-exact")
    completed = journey.settle_action(accepted)
    journey.settle()
    after = journey.resource_task(job[0])

    assert completed.kind is ActionKind.TERMINATE_ATTEMPT
    assert completed.state is ActionState.SUCCEEDED
    assert completed.result_code is ActionResult.SATISFIED
    assert completed.expected_attempt_uid == current.attempt_uid
    assert len(after.attempts) == 1
    assert after.attempts[0].identity == current
    assert after.attempts[0].state == job_pb2.TASK_STATE_KILLED
