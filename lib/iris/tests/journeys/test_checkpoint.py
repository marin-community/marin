# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.resources.errors import ResourceNotFound
from iris.rpc import job_pb2


def test_checkpoint_restore_preserves_snapshot_and_discards_later_writes(journey):
    durable = journey.submit("before-checkpoint")
    journey.settle()
    journey.succeed(durable[0])
    journey.settle()
    checkpoint, result = journey.checkpoint()

    later = journey.submit("after-checkpoint")
    journey.settle()
    journey.succeed(later[0])
    journey.settle()

    journey.restore(checkpoint)

    assert result.job_count == 1
    assert journey.job(durable).summary.state == job_pb2.JOB_STATE_SUCCEEDED
    with pytest.raises(ResourceNotFound):
        journey.job(later)


def test_active_task_restored_from_checkpoint_retries_after_runtime_loss(journey):
    job = journey.submit("active-at-checkpoint", preemption_retries=1)
    journey.settle()
    checkpoint, _ = journey.checkpoint()

    journey.restore(checkpoint)
    journey.lose_runtime(job[0])
    journey.settle()
    journey.succeed(job[0])
    journey.settle()

    assert [attempt.state for attempt in journey.task(job[0]).attempts] == [
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_SUCCEEDED,
    ]
    assert journey.job(job).summary.state == job_pb2.JOB_STATE_SUCCEEDED
