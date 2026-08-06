# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2


def test_job_when_every_task_succeeds_reports_exact_public_fold(journey):
    job = journey.submit("success", tasks=8)
    journey.settle()

    journey.succeed_all(job)
    journey.settle()

    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED
    assert [task.state for task in journey.tasks(job)] == [job_pb2.TASK_STATE_SUCCEEDED] * 8
    assert len(journey.backend_events(kind="launched")) == 8


def test_task_7_when_application_fails_once_retries_without_affecting_siblings(journey):
    job = journey.submit("retry", tasks=8, failure_retries=1)
    journey.settle()

    journey.fail(job[7])
    journey.settle()
    for task_index in range(7):
        journey.succeed(job[task_index])
    journey.succeed(job[7])
    journey.settle()

    task_7 = journey.task(job[7])
    assert [attempt.state for attempt in task_7.attempts] == [
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_SUCCEEDED,
    ]
    assert all(len(journey.task(job[index]).attempts) == 1 for index in range(7))
    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED


def test_task_when_failure_budget_is_exhausted_fails_job(journey):
    job = journey.submit("exhausted", failure_retries=1)
    journey.settle()

    journey.fail(job[0], error="first")
    journey.settle()
    journey.fail(job[0], error="second")
    journey.settle()

    task = journey.task(job[0])
    assert [attempt.state for attempt in task.attempts] == [
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_FAILED,
    ]
    assert journey.job(job).state == job_pb2.JOB_STATE_FAILED


def test_task_when_runtime_disappears_retries_without_charging_failure_budget(journey):
    job = journey.submit("runtime-loss", failure_retries=0, preemption_retries=1)
    journey.settle()

    journey.lose_runtime(job[0])
    journey.settle()
    journey.succeed(job[0])
    journey.settle()

    task = journey.task(job[0])
    assert [attempt.state for attempt in task.attempts] == [
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_SUCCEEDED,
    ]
    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED


def test_task_when_preempted_retries_under_separate_budget(journey):
    job = journey.submit("preempted", preemption_retries=1)
    journey.settle()

    journey.preempt(job[0])
    journey.settle()
    journey.succeed(job[0])
    journey.settle()

    assert [attempt.state for attempt in journey.task(job[0]).attempts] == [
        job_pb2.TASK_STATE_PREEMPTED,
        job_pb2.TASK_STATE_SUCCEEDED,
    ]
    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED


def test_job_when_cumulative_failure_budget_is_crossed_stops_live_siblings(journey):
    job = journey.submit("cumulative-budget", tasks=3, failure_retries=1, max_task_failures=1)
    journey.settle()

    journey.fail(job[0])
    journey.settle()
    journey.fail(job[1])
    journey.settle()

    assert journey.job(job).state == job_pb2.JOB_STATE_FAILED
    assert journey.task(job[2]).state == job_pb2.TASK_STATE_KILLED


def test_job_with_128_tasks_converges_without_duplicate_launches(journey):
    job = journey.submit("wide", tasks=128)
    journey.settle()

    journey.succeed_all(job)
    journey.settle()

    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED
    assert len(journey.backend_events(kind="launched")) == 128


def test_late_observation_from_abandoned_attempt_cannot_finish_replacement(journey):
    job = journey.submit("late-observation", preemption_retries=1)
    journey.settle()
    journey.lose_runtime(job[0])
    journey.settle()

    journey.succeed(job[0], attempt_id=0)
    journey.settle()
    assert journey.task(job[0]).state == job_pb2.TASK_STATE_RUNNING

    journey.succeed(job[0])
    journey.settle()
    assert [attempt.state for attempt in journey.task(job[0]).attempts] == [
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_SUCCEEDED,
    ]
