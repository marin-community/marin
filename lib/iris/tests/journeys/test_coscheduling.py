# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.rpc import job_pb2


@pytest.mark.parametrize("groups", [(8, 16), (16, 16, 16)])
def test_locality_groups_after_restart_retry_and_cancel_remain_one_job(journey, groups):
    job = journey.submit("locality-groups", tasks=sum(groups), locality_groups=groups, failure_retries=1)
    journey.settle()
    journey.restart()
    assert tuple(journey.controller.get_job_status(job.wire_id).request.coscheduling.group_sizes) == groups

    journey.fail(job[groups[0]])
    journey.settle()
    launched = journey.backend_events(kind="launched")
    assert len(launched) == 2 * sum(groups)
    assert all(event.group_sizes == groups for event in launched)
    assert all(len(journey.task(job[index]).attempts) == 2 for index in range(job.tasks))

    journey.cancel(job)
    journey.settle()
    assert journey.job(job).state == job_pb2.JOB_STATE_KILLED
    assert all(journey.task(job[index]).state == job_pb2.TASK_STATE_KILLED for index in range(job.tasks))


def test_coscheduled_job_when_one_task_exhausts_retries_stops_the_gang(journey):
    job = journey.submit("coscheduled-failure", tasks=4, coscheduled=True)
    journey.settle()

    journey.clock.advance(3600)
    journey.fail(job[2])
    journey.settle()

    assert journey.job(job).state == job_pb2.JOB_STATE_FAILED
    assert journey.task(job[2]).state == job_pb2.TASK_STATE_FAILED
    assert [journey.task(job[index]).state for index in (0, 1, 3)] == [
        job_pb2.TASK_STATE_COSCHED_FAILED,
    ] * 3
    assert all(journey.task(job[index]).attempts[0].HasField("finished_at") for index in range(4))


def test_coscheduled_job_when_one_task_retries_restarts_the_whole_gang(journey):
    job = journey.submit(
        "coscheduled-retry",
        tasks=4,
        failure_retries=1,
        max_task_failures=1,
        coscheduled=True,
    )
    journey.settle()

    journey.clock.advance(3600)
    journey.fail(job[2])
    journey.settle()

    assert [journey.task(job[index]).state for index in range(4)] == [job_pb2.TASK_STATE_RUNNING] * 4
    assert all(len(journey.task(job[index]).attempts) == 2 for index in range(4))
    first_attempt = journey.task(job[0]).attempts[0]
    assert first_attempt.HasField("finished_at")
    assert 3_600_000 <= first_attempt.finished_at.epoch_ms - first_attempt.started_at.epoch_ms < 3_601_000

    journey.clock.advance(9 * 3600)
    journey.settle()
    assert journey.task(job[0]).attempts[0].finished_at == first_attempt.finished_at

    journey.succeed_all(job)
    journey.settle()
    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED
