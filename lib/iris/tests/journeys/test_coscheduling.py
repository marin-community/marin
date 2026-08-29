# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2


def test_coscheduled_job_when_one_task_exhausts_retries_stops_the_gang(journey):
    job = journey.submit("coscheduled-failure", tasks=4, coscheduled=True)
    journey.settle()

    journey.fail(job[2])
    journey.settle()

    assert journey.job(job).state == job_pb2.JOB_STATE_FAILED
    assert journey.task(job[2]).state == job_pb2.TASK_STATE_FAILED
    assert [journey.task(job[index]).state for index in (0, 1, 3)] == [
        job_pb2.TASK_STATE_COSCHED_FAILED,
    ] * 3


def test_coscheduled_job_when_one_task_retries_restarts_the_whole_gang(journey):
    job = journey.submit(
        "coscheduled-retry",
        tasks=4,
        failure_retries=1,
        max_task_failures=1,
        coscheduled=True,
    )
    journey.settle()

    journey.fail(job[2])
    journey.settle()

    assert [journey.task(job[index]).state for index in range(4)] == [job_pb2.TASK_STATE_RUNNING] * 4
    assert all(len(journey.task(job[index]).attempts) == 2 for index in range(4))

    journey.succeed_all(job)
    journey.settle()
    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED


def test_coscheduled_job_when_workers_succeed_before_head_preemption_restarts_the_whole_gang(journey):
    job = journey.submit(
        "coscheduled-preempted-head",
        tasks=4,
        preemption_retries=1,
        coscheduled=True,
    )
    journey.settle()

    journey.succeed(job[1])
    journey.succeed(job[2])
    journey.settle()
    journey.preempt(job[0])
    journey.settle()

    tasks = journey.tasks(job)
    assert [task.state for task in tasks] == [job_pb2.TASK_STATE_RUNNING] * 4
    assert all(len(journey.task(job[index]).attempts) == 2 for index in range(4))
    assert [attempt.state for attempt in journey.task(job[1]).attempts] == [
        job_pb2.TASK_STATE_SUCCEEDED,
        job_pb2.TASK_STATE_RUNNING,
    ]
