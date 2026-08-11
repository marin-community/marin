# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cluster.controller.persistence.pruning import prune_old_data
from iris.resources.action import ActionKind, ActionResult, ActionState
from iris.resources.errors import ResourceNotFound, ResourceReplaced
from iris.rpc import job_pb2
from rigging.timing import Duration


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
