# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #4117: _check_worker_group false abort.

The race (before fix):
1. Last stage completes — all shards done, workers get SHUTDOWN, exit
2. Main thread is in _collect_results / _regroup_result_refs (between
   _wait_for_stage returning and self.shutdown())
3. Background coordinator loop calls _check_worker_group
4. worker_group.is_done() returns True (workers exited cleanly!)
5. Coordinator calls abort("Worker job terminated permanently...")

Fix: _check_worker_group skips while no shard is outstanding.

Only a draining pool can hit this — its workers exit once the last stage
drains. A standing pool's workers live until coordinator shutdown, so there a
terminated worker job really is a crash.
"""

import threading
import time
from unittest.mock import MagicMock

from conftest import _TEST_EXECUTION_ID, _TEST_TASK_COST, _make_test_coordinator, start_test_stage
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask, TaskResult
from zephyr.worker_context import CounterSnapshot


def test_check_worker_group_skips_after_completed_stage(tmp_path, actor_context):
    """Worker group finishing after completed stage must not abort. #4117."""
    coordinator = _make_test_coordinator(tmp_path, drain_idle_workers=True)
    mock_group = MagicMock()
    mock_group.is_done.return_value = True
    coordinator.set_worker_group(mock_group)

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        cost=_TEST_TASK_COST,
        stage_name="test",
    )
    run = start_test_stage(coordinator, [task], stage_name="last-stage")
    coordinator.report_result(
        "worker-0", _TEST_EXECUTION_ID, 0, 0, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
    )

    assert run.completed_shards >= run.total_shards

    coordinator._check_worker_group()

    assert coordinator.get_fatal_error() is None


def test_check_worker_group_still_aborts_mid_stage(coordinator):
    """Worker group dying while shards are still in-flight must abort."""
    mock_group = MagicMock()
    mock_group.is_done.return_value = True
    coordinator.set_worker_group(mock_group)

    task = ShardTask(
        shard_idx=0,
        total_shards=2,
        shard=ListShard(refs=[]),
        operations=[],
        cost=_TEST_TASK_COST,
        stage_name="test",
    )
    start_test_stage(coordinator, [task, task], stage_name="mid-stage")
    # Only 1 of 2 shards completed
    coordinator.report_result(
        "worker-0", _TEST_EXECUTION_ID, 0, 0, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
    )

    coordinator._check_worker_group()

    assert coordinator.get_fatal_error() is not None
    assert "Worker job terminated permanently" in coordinator.get_fatal_error()


def test_coordinator_loop_no_abort_during_result_collection(tmp_path, actor_context):
    """Background loop must not abort during post-stage result collection. #4117."""
    coordinator = _make_test_coordinator(tmp_path, drain_idle_workers=True)
    mock_group = MagicMock()
    call_count = 0

    def is_done_with_delay():
        nonlocal call_count
        call_count += 1
        return call_count > 2

    mock_group.is_done.side_effect = is_done_with_delay
    coordinator.set_worker_group(mock_group)

    task = ShardTask(
        shard_idx=0,
        total_shards=1,
        shard=ListShard(refs=[]),
        operations=[],
        cost=_TEST_TASK_COST,
        stage_name="test",
    )
    start_test_stage(coordinator, [task], stage_name="last-stage")
    coordinator.report_result(
        "worker-0", _TEST_EXECUTION_ID, 0, 0, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty()
    )

    t = threading.Thread(target=coordinator._coordinator_loop, daemon=True)
    t.start()

    # Simulate the post-stage window where main thread collects/regroups results
    time.sleep(2)

    fatal = coordinator.get_fatal_error()
    coordinator.shutdown()
    t.join(timeout=2.0)

    assert fatal is None
