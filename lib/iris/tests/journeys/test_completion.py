# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2


def test_active_job_when_completed_succeeds_and_stops_every_attempt(journey):
    job = journey.submit("complete", tasks=4)
    journey.settle()

    journey.complete(job)
    journey.settle()
    journey.complete(job)
    journey.settle()

    assert journey.job(job).state == job_pb2.JOB_STATE_SUCCEEDED
    assert [task.state for task in journey.tasks(job)] == [job_pb2.TASK_STATE_SUCCEEDED] * 4
    assert {(event.task_id, event.attempt_id) for event in journey.backend_events(kind="stopped")} == {
        (job[index].wire_id, 0) for index in range(4)
    }
