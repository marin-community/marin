# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Administrative operations expressed through public Task state."""

from iris.rpc import job_pb2


def test_kick_preemption_is_buffered_then_retries_the_exact_attempt(journey):
    job = journey.submit("operator-preempt", preemption_retries=1)
    journey.settle()
    current_attempt = journey.task(job[0]).current_attempt_id

    result = journey.kick(job[0], attempt_id=current_attempt)

    assert result.queued
    assert journey.task(job[0]).state == job_pb2.TASK_STATE_RUNNING

    journey.settle()

    detail = journey.task(job[0])
    assert detail.state == job_pb2.TASK_STATE_RUNNING
    assert [attempt.state for attempt in detail.attempts] == [
        job_pb2.TASK_STATE_PREEMPTED,
        job_pb2.TASK_STATE_RUNNING,
    ]


def test_kick_failure_finalizes_without_retry_and_stops_runtime(journey):
    job = journey.submit("operator-fail")
    journey.settle()

    result = journey.kick(job[0], desired_state=job_pb2.TASK_STATE_FAILED)
    journey.settle()

    assert result.queued
    assert journey.task(job[0]).state == job_pb2.TASK_STATE_FAILED
    assert [(event.task_id, event.attempt_id) for event in journey.backend_events(kind="stopped")] == [
        (job[0].wire_id, 0)
    ]


def test_kick_for_a_superseded_attempt_cannot_change_its_replacement(journey):
    job = journey.submit("stale-operator-action", preemption_retries=1)
    journey.settle()
    journey.preempt(job[0])
    journey.settle()

    result = journey.kick(job[0], desired_state=job_pb2.TASK_STATE_FAILED, attempt_id=0)
    journey.settle()

    assert not result.queued
    assert result.detail == "attempt 0 is not current (current is 1)"
    assert journey.task(job[0]).state == job_pb2.TASK_STATE_RUNNING
    assert [attempt.state for attempt in journey.task(job[0]).attempts] == [
        job_pb2.TASK_STATE_PREEMPTED,
        job_pb2.TASK_STATE_RUNNING,
    ]
