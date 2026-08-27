# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2


def test_controller_when_reopened_preserves_attempt_and_resumes_without_duplicate_launch(journey):
    job = journey.submit("restart", failure_retries=1)
    journey.settle()
    journey.fail(job[0])
    journey.settle()

    before = journey.task(job[0])
    journey.restart()
    after = journey.task(job[0])
    journey.succeed(job[0])
    journey.settle()

    assert [(attempt.attempt_id, attempt.state) for attempt in after.attempts] == [
        (attempt.attempt_id, attempt.state) for attempt in before.attempts
    ]
    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED
    assert [(event.task_id, event.attempt_id) for event in journey.backend_events(kind="launched")] == [
        (job[0].wire_id, 0),
        (job[0].wire_id, 1),
    ]
