# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduction for issue #4117: _check_worker_group aborts after last stage completes.

The race:
1. Last stage completes — all shards done, workers get SHUTDOWN, exit
2. Main thread is in _collect_results / _regroup_result_refs (between
   _wait_for_stage returning and self.shutdown())
3. Background coordinator loop calls _check_worker_group
4. worker_group.is_done() returns True (workers exited cleanly!)
5. Coordinator calls abort("Worker job terminated permanently...")
6. Next _wait_for_stage (or the caller reading results) sees fatal_error

_check_worker_group does NOT check whether the stage actually completed.
It unconditionally treats is_done()==True as a crash.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from zephyr.execution import ListShard, ShardTask, TaskResult, ZephyrCoordinator


@pytest.fixture
def coordinator(actor_context, tmp_path):
    coord = ZephyrCoordinator()
    coord.set_chunk_config(str(tmp_path / "chunks"), "test-exec")
    yield coord
    coord.shutdown()


def test_check_worker_group_aborts_after_completed_stage(coordinator):
    """Reproduce: worker group finishing after last stage triggers false abort.

    This is the exact race from issue #4117. The coordinator background loop
    sees worker_group.is_done() == True and aborts, even though the stage
    completed successfully.
    """
    # Set up a worker group that reports is_done=True (workers exited after SHUTDOWN)
    mock_group = MagicMock()
    mock_group.is_done.return_value = True
    coordinator.set_worker_group(mock_group)

    # Simulate a completed stage: 1 task, 1 completed
    task = ShardTask(shard_idx=0, total_shards=1, shard=ListShard(refs=[]), operations=[], stage_name="test")
    coordinator.start_stage("last-stage", [task], is_last_stage=True)

    # Simulate task completion (worker finished before exiting)
    coordinator.report_result("worker-0", 0, 0, TaskResult(shard=ListShard(refs=[])))

    # Verify stage is complete
    assert coordinator._completed_shards >= coordinator._total_shards

    # Now _check_worker_group fires — this is the bug
    coordinator._check_worker_group()

    fatal = coordinator.get_fatal_error()
    # BUG: fatal_error is set even though the stage completed successfully
    assert fatal is not None, "Bug not triggered — _check_worker_group should have aborted"
    assert "Worker job terminated permanently" in fatal
    print(f"\nBUG REPRODUCED: {fatal!r}")
    print("Workers exited cleanly after SHUTDOWN, but coordinator treated it as a crash.")


def test_check_worker_group_aborts_during_result_collection(coordinator):
    """Reproduce the full race: background loop fires during post-stage processing."""
    mock_group = MagicMock()
    # Workers alive during stage, then exit after SHUTDOWN
    call_count = 0

    def is_done_with_delay():
        nonlocal call_count
        call_count += 1
        # First few calls: workers still running
        # After that: workers have exited
        return call_count > 2

    mock_group.is_done.side_effect = is_done_with_delay
    coordinator.set_worker_group(mock_group)

    task = ShardTask(shard_idx=0, total_shards=1, shard=ListShard(refs=[]), operations=[], stage_name="test")
    coordinator.start_stage("last-stage", [task], is_last_stage=True)
    coordinator.report_result("worker-0", 0, 0, TaskResult(shard=ListShard(refs=[])))

    # Start coordinator background loop
    t = threading.Thread(target=coordinator._coordinator_loop, daemon=True)
    t.start()

    # Simulate the main thread doing post-stage work (collect_results, regroup, etc.)
    # During this time, the background loop keeps calling _check_worker_group
    time.sleep(2)

    fatal = coordinator.get_fatal_error()
    coordinator.shutdown()
    t.join(timeout=2.0)

    assert fatal is not None, "Bug not triggered — expected abort during result collection window"
    assert "Worker job terminated permanently" in fatal
    print(f"\nBUG REPRODUCED: {fatal!r}")
    print("Background loop aborted during post-stage result collection.")
