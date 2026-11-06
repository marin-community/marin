# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #4117: _check_worker_group false abort.

The race (before fix):
1. Last stage completes — all shards done, workers get SHUTDOWN, exit
2. Main thread is in _collect_results / _regroup_result_refs (between
   _wait_for_stage returning and self.shutdown())
3. Background coordinator loop calls _check_worker_group
4. worker_group.status() returns SUCCEEDED (workers exited cleanly!)
5. Coordinator calls abort("Worker job terminated permanently...")

Fix: a draining coordinator ignores expected worker exit after all shards complete.
"""

from unittest.mock import MagicMock

import pytest
from fray.types import JobStatus
from zephyr.coordinator import PullStatus
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask, TaskResult, ZephyrPoolLostError, ZephyrWorkerError
from zephyr.testing.coordinator import (
    TEST_TASK_COST,
    TEST_WORKER_AVAILABLE,
    make_test_coordinator,
    start_test_stage,
)
from zephyr.worker_context import CounterSnapshot


@pytest.fixture
def draining_coordinator(tmp_path, actor_context):
    coordinator = make_test_coordinator(tmp_path, drain_idle_workers=True)
    yield coordinator
    coordinator.shutdown()


def test_check_worker_group_skips_after_completed_stage(draining_coordinator):
    """Worker group finishing after completed stage must not abort. #4117."""
    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        cost=TEST_TASK_COST,
        stage_name="test",
    )
    run = start_test_stage(draining_coordinator, [task], stage_name="last-stage", is_last_stage=True)
    status, work = draining_coordinator.pull_task("worker-0", TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    assert work is not None
    draining_coordinator.report_result(
        "worker-0",
        run.execution_id,
        work.task.shard_idx,
        work.attempt,
        TaskResult(shard=ListShard(refs=[])),
        CounterSnapshot.empty(),
        run.stage_generation,
    )
    assert run.completed_shards == run.total_shards

    mock_group = MagicMock()
    mock_group.status.return_value = JobStatus.SUCCEEDED
    draining_coordinator._worker_group = mock_group
    draining_coordinator._check_worker_group()

    assert draining_coordinator.get_fatal_error() is None


@pytest.mark.parametrize(
    ("worker_job_status", "error_type"),
    [
        # Iris gave up on a worker: the failure may repeat, so the driver must not retry.
        (JobStatus.FAILED, ZephyrWorkerError),
        # Every worker exited cleanly, so the pool released its last worker too early.
        (JobStatus.SUCCEEDED, ZephyrPoolLostError),
    ],
)
def test_check_worker_group_aborts_mid_stage(draining_coordinator, worker_job_status, error_type):
    """A worker job that ends with a shard outstanding aborts the stage.

    The error type tells the driver whether a fresh pool may retry the pipeline.
    """
    tasks = [
        ShardTask(
            shard_idx=shard_idx,
            total_shards=2,
            shard=ListShard(refs=[]),
            operations=[],
            cost=TEST_TASK_COST,
            stage_name="test",
        )
        for shard_idx in range(2)
    ]
    run = start_test_stage(draining_coordinator, tasks, stage_name="mid-stage", is_last_stage=True)
    status, work = draining_coordinator.pull_task("worker-0", TEST_WORKER_AVAILABLE)
    assert status == PullStatus.RUN_TASK
    assert work is not None
    draining_coordinator.report_result(
        "worker-0",
        run.execution_id,
        work.task.shard_idx,
        work.attempt,
        TaskResult(shard=ListShard(refs=[])),
        CounterSnapshot.empty(),
        run.stage_generation,
    )

    mock_group = MagicMock()
    mock_group.status.return_value = worker_job_status
    draining_coordinator._worker_group = mock_group
    draining_coordinator._check_worker_group()

    with pytest.raises(error_type):
        draining_coordinator._wait_for_stage(run)
