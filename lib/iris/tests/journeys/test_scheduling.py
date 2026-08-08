# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2


def test_job_waiting_for_capacity_becomes_unschedulable_at_deadline(journey_without_capacity):
    journey = journey_without_capacity
    job = journey.submit(
        "unroutable",
        scheduling_timeout=5,
    )
    journey.settle()
    assert journey.job(job).summary.state == job_pb2.JOB_STATE_PENDING

    journey.clock.advance(6)
    journey.settle()

    assert journey.job(job).summary.state == job_pb2.JOB_STATE_UNSCHEDULABLE
    assert journey.task(job[0]).summary.state == job_pb2.TASK_STATE_UNSCHEDULABLE


def test_sequential_jobs_reuse_one_backend_without_cross_job_state(journey):
    for index in range(3):
        job = journey.submit(f"sequential-{index}")
        journey.settle()
        journey.succeed(job[0])
        journey.settle()
        assert journey.job(job).summary.state == job_pb2.JOB_STATE_SUCCEEDED


def test_transient_backend_outage_redrives_one_attempt_without_duplicate_launch(journey):
    journey.backend_outage(ticks=2)
    job = journey.submit("backend-outage")

    journey.wait_through_outage(ticks=2)
    assert journey.backend_events(kind="launched") == []

    journey.settle()
    journey.succeed(job[0])
    journey.settle()

    assert journey.job(job).summary.state == job_pb2.JOB_STATE_SUCCEEDED
    assert [(event.task_id, event.attempt_id) for event in journey.backend_events(kind="launched")] == [
        (job[0].wire_id, 0)
    ]


def test_running_task_crossing_execution_deadline_fails_and_stops(journey):
    job = journey.submit("execution-timeout", execution_timeout=5)
    journey.settle()

    journey.clock.advance(6)
    journey.settle()

    assert journey.job(job).summary.state == job_pb2.JOB_STATE_FAILED
    assert journey.task(job[0]).summary.state == job_pb2.TASK_STATE_FAILED
    assert [(event.task_id, event.attempt_id) for event in journey.backend_events(kind="stopped")] == [
        (job[0].wire_id, 0)
    ]


def test_cancel_during_backend_outage_is_delivered_after_recovery(journey):
    job = journey.submit("cancel-during-outage")
    journey.settle()
    journey.backend_outage(ticks=1)

    journey.cancel(job)
    journey.wait_through_outage(ticks=1)
    journey.settle()

    assert journey.job(job).summary.state == job_pb2.JOB_STATE_KILLED
    assert [(event.task_id, event.attempt_id) for event in journey.backend_events(kind="stopped")] == [
        (job[0].wire_id, 0)
    ]
