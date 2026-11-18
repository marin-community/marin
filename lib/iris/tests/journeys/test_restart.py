# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2
from rigging.timing import Duration


def test_topology_assignment_survives_restart_and_stale_attempt_cannot_replace_it(journey):
    job = journey.submit("topology", failure_retries=1)
    journey.settle()
    domain = {"ds.coreweave.com/nvlink.domain": "rack-a"}
    journey.assign_topology(job[0], domain)
    journey.settle()
    journey.restart()
    assert dict(journey.task(job[0]).attempts[0].topology) == domain

    journey.fail(job[0])
    journey.settle()
    assert dict(journey.task(job[0]).attempts[-1].topology) == {}
    journey.assign_topology(job[0], {"ds.coreweave.com/nvlink.domain": "rack-stale"}, attempt_id=0)
    journey.settle()
    attempts = journey.task(job[0]).attempts
    assert dict(attempts[0].topology) == domain
    assert dict(attempts[-1].topology) == {}


def test_controller_when_reopened_preserves_attempt_and_resumes_without_duplicate_launch(journey):
    job = journey.submit("restart", failure_retries=1, termination_grace=Duration.from_seconds(60))
    journey.settle()
    journey.fail(job[0])
    journey.settle()

    before = journey.task(job[0])
    journey.restart()
    after = journey.task(job[0])
    stored = journey.controller.get_job_status(job.wire_id).request
    assert stored.termination_grace.milliseconds == 60_000
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
