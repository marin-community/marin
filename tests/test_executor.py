# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import time

from marin.execution.executor_step_status import (
    StatusFile,
)


def test_status_file_reads_legacy_format(tmp_path):
    output_dir = tmp_path / "step"
    output_dir.mkdir()
    status_path = output_dir / ".executor_status"

    events = [
        {"date": "2024-09-28T13:29:20.780705", "status": "WAITING", "message": None},
        {"date": "2024-09-28T13:29:21.091470", "status": "RUNNING", "message": None},
        {"date": "2024-09-28T13:29:47.559614", "status": "SUCCESS", "message": None},
    ]

    with open(status_path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    status_file = StatusFile(str(output_dir), worker_id="legacy-reader")
    assert status_file.status == "SUCCESS"


def test_status_file_takeover_stale_lock_then_refresh(tmp_path):
    """Test taking over a stale lock from a dead worker and then refreshing it."""
    from marin.execution.executor_step_status import HEARTBEAT_TIMEOUT, Lease

    # Simulate worker A creating a stale lock (as if it died)
    dead_worker = StatusFile(tmp_path, worker_id="dead-worker")
    dead_worker.try_acquire_lock()

    # Manually backdate the lock to make it stale
    generation, _ = dead_worker._read_lock_with_generation()
    stale_lease = Lease(worker_id="dead-worker", timestamp=time.time() - HEARTBEAT_TIMEOUT - 10)
    dead_worker._write_lock(stale_lease, if_generation_match=generation)

    # Worker B comes along and takes over
    live_worker = StatusFile(tmp_path, worker_id="live-worker")

    # Verify the lock is stale
    _, lease = live_worker._read_lock_with_generation()
    assert lease is not None
    assert lease.is_stale()

    # Take over the stale lock
    assert live_worker.try_acquire_lock()

    # Verify we now own the lock
    _, lease_after_takeover = live_worker._read_lock_with_generation()
    assert lease_after_takeover.worker_id == "live-worker"

    # Now try to refresh
    time.sleep(0.1)
    live_worker.refresh_lock()

    _, lease_after_refresh = live_worker._read_lock_with_generation()
    assert lease_after_refresh.worker_id == "live-worker"
    assert lease_after_refresh.timestamp > lease_after_takeover.timestamp
