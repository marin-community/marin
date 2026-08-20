# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2


def test_job_when_cancelled_stops_every_attempt_and_is_idempotent(journey):
    job = journey.submit("cancel", tasks=4)
    journey.settle()

    journey.cancel(job)
    journey.settle()
    journey.cancel(job)
    journey.settle()

    assert journey.job(job).summary.state == job_pb2.JOB_STATE_KILLED
    assert [task.state for task in journey.tasks(job)] == [job_pb2.TASK_STATE_KILLED] * 4
    assert {(event.task_id, event.attempt_id) for event in journey.backend_events(kind="stopped")} == {
        (job[index].wire_id, 0) for index in range(4)
    }


def test_pending_job_when_cancelled_never_reaches_the_backend(journey):
    job = journey.submit("cancel-pending")

    journey.cancel(job)
    journey.settle()

    assert journey.job(job).summary.state == job_pb2.JOB_STATE_KILLED
    assert journey.backend_events(kind="launched") == []
