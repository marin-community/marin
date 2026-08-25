# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import pytest
import rigging.filesystem.distributed_lock as distributed_lock
from rigging.filesystem.distributed_lock import (
    HEARTBEAT_TIMEOUT,
    Lease,
    LeaseLostError,
    create_lock,
)


@pytest.mark.parametrize(
    "age,is_stale",
    [
        (HEARTBEAT_TIMEOUT - 1, False),
        (HEARTBEAT_TIMEOUT + 1, True),
    ],
)
def test_lease_age_determines_staleness(age: float, is_stale: bool):
    lease = Lease(worker_id="worker-1", timestamp=time.time() - age)
    assert lease.is_stale() is is_stale


def test_acquire_and_release_local_lock(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")

    assert lock.try_acquire()
    # Lock file should exist with correct holder
    with open(lock_path) as f:
        data = json.loads(f.read())
    assert data["worker_id"] == "holder-a"

    lock.release()
    assert not os.path.exists(lock_path)


def test_second_holder_blocked_while_lock_held(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock_a = create_lock(lock_path, worker_id="holder-a")
    lock_b = create_lock(lock_path, worker_id="holder-b")

    assert lock_a.try_acquire()
    assert not lock_b.try_acquire()

    lock_a.release()
    assert lock_b.try_acquire()
    lock_b.release()


def test_stale_lock_can_be_taken_over(tmp_path):
    lock_path = str(tmp_path / "test.lock")

    # Write a stale lock directly
    stale_lease = Lease(worker_id="dead-worker", timestamp=time.time() - HEARTBEAT_TIMEOUT - 10)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as f:
        f.write(json.dumps({"worker_id": stale_lease.worker_id, "timestamp": stale_lease.timestamp}))

    lock = create_lock(lock_path, worker_id="new-worker")
    assert lock.active_holder_id() is None
    assert lock.try_acquire()
    lock.release()


def test_refresh_updates_timestamp(tmp_path, monkeypatch):
    lock_path = str(tmp_path / "test.lock")
    current_time = 100.0
    monkeypatch.setattr(distributed_lock.time, "time", lambda: current_time)
    lock = create_lock(lock_path, worker_id="holder-a")
    assert lock.try_acquire()

    # Read initial timestamp
    with open(lock_path) as f:
        t1 = json.loads(f.read())["timestamp"]

    current_time = 101.0
    lock.refresh()

    with open(lock_path) as f:
        t2 = json.loads(f.read())["timestamp"]

    assert t2 > t1
    lock.release()


def test_refresh_raises_lease_lost_error_if_not_holder(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock_a = create_lock(lock_path, worker_id="holder-a")
    lock_b = create_lock(lock_path, worker_id="holder-b")

    assert lock_a.try_acquire()
    with pytest.raises(LeaseLostError, match="held by"):
        lock_b.refresh()
    lock_a.release()


def test_refresh_raises_lease_lost_when_lock_disappears(tmp_path):
    """If the lock file vanishes under the holder, refresh() must raise LeaseLostError.

    A disappearing lock file means another worker took over the stale lease and
    subsequently released it, so the original holder has irrecoverably lost
    ownership.  Silently re-acquiring would let two workers run the same step.
    """
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")
    assert lock.try_acquire()

    # Simulate the lock file being deleted (e.g. by another worker that took
    # over a stale lease and then released it).
    os.remove(lock_path)

    with pytest.raises(LeaseLostError, match="disappeared"):
        lock.refresh()


def test_has_active_holder(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")

    assert not lock.has_active_holder()
    assert lock.active_holder_id() is None
    assert lock.try_acquire()
    assert lock.has_active_holder()
    assert lock.active_holder_id() == "holder-a"
    lock.release()
    assert not lock.has_active_holder()
    assert lock.active_holder_id() is None


def test_same_holder_can_reacquire(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")

    assert lock.try_acquire()
    assert lock.try_acquire()  # idempotent
    lock.release()


def test_release_idempotent_when_lock_already_gone(tmp_path):
    """Release should not crash if the lock file was already deleted."""
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")
    assert lock.try_acquire()

    # Delete manually
    os.remove(lock_path)

    # An already-absent lock is the documented idempotent release case.
    lock.release()
    assert not os.path.exists(lock_path)


def test_stale_takeover_then_refresh_detects_new_holder(tmp_path):
    """Simulates a full stale-takeover race:

    1. Worker A acquires.
    2. Worker A stops heartbeating; its lease goes stale.
    3. Worker B takes over the stale lock.
    4. Worker A wakes up and tries to refresh — must get LeaseLostError.
    """
    lock_path = str(tmp_path / "test.lock")
    lock_a = create_lock(lock_path, worker_id="worker-A")
    lock_b = create_lock(lock_path, worker_id="worker-B")

    assert lock_a.try_acquire()

    # Artificially age Worker A's lease past the timeout.
    with open(lock_path) as f:
        data = json.loads(f.read())
    data["timestamp"] = time.time() - HEARTBEAT_TIMEOUT - 10
    with open(lock_path, "w") as f:
        f.write(json.dumps(data))

    # Worker B sees the stale lock and takes over.
    assert lock_b.try_acquire()

    # Worker A tries to refresh — should fail because B now holds the lock.
    with pytest.raises(LeaseLostError, match="worker-B"):
        lock_a.refresh()

    lock_b.release()
