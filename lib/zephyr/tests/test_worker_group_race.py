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

Fix: _check_worker_group skips when all shards are completed.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from zephyr.execution import CounterSnapshot, ListShard, ShardTask, TaskResult, ZephyrCoordinator


@pytest.fixture
def coordinator(actor_context, tmp_path):
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    yield coord
    coord.shutdown()


def test_check_worker_group_skips_after_completed_stage(coordinator):
    """Worker group finishing after completed stage must not abort. #4117."""
    mock_group = MagicMock()
    mock_group.is_done.return_value = True
    coordinator.set_worker_group(mock_group)

    task = ShardTask(shard_idx=0, total_shards=1, shard=ListShard(refs=[]), operations=[], stage_name="test")
    coordinator._start_stage("last-stage", 0, [task], is_last_stage=True)
    coordinator.report_result("worker-0", 0, 0, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())

    assert coordinator._completed_shards >= coordinator._total_shards

    coordinator._check_worker_group()

    assert coordinator.get_fatal_error() is None


def test_check_worker_group_still_aborts_mid_stage(coordinator):
    """Worker group dying while shards are still in-flight must abort."""
    mock_group = MagicMock()
    mock_group.is_done.return_value = True
    coordinator.set_worker_group(mock_group)

    task = ShardTask(shard_idx=0, total_shards=2, shard=ListShard(refs=[]), operations=[], stage_name="test")
    coordinator._start_stage("mid-stage", 0, [task, task], is_last_stage=False)
    # Only 1 of 2 shards completed
    coordinator.report_result("worker-0", 0, 0, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())

    coordinator._check_worker_group()

    assert coordinator.get_fatal_error() is not None
    assert "Worker job terminated permanently" in coordinator.get_fatal_error()


def test_coordinator_loop_no_abort_during_result_collection(coordinator):
    """Background loop must not abort during post-stage result collection. #4117."""
    mock_group = MagicMock()
    call_count = 0

    def is_done_with_delay():
        nonlocal call_count
        call_count += 1
        return call_count > 2

    mock_group.is_done.side_effect = is_done_with_delay
    coordinator.set_worker_group(mock_group)

    task = ShardTask(shard_idx=0, total_shards=1, shard=ListShard(refs=[]), operations=[], stage_name="test")
    coordinator._start_stage("last-stage", 0, [task], is_last_stage=True)
    coordinator.report_result("worker-0", 0, 0, TaskResult(shard=ListShard(refs=[])), CounterSnapshot.empty())

    t = threading.Thread(target=coordinator._coordinator_loop, daemon=True)
    t.start()

    # Simulate the post-stage window where main thread collects/regroups results
    time.sleep(2)

    fatal = coordinator.get_fatal_error()
    coordinator.shutdown()
    t.join(timeout=2.0)

    assert fatal is None
