# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import pytest

from iris.distributed_lock import (
    HEARTBEAT_TIMEOUT,
    DistributedLock,
    Lease,
)


def test_lease_is_stale_after_timeout():
    lease = Lease(holder_id="worker-1", timestamp=time.time() - HEARTBEAT_TIMEOUT - 1)
    assert lease.is_stale()


def test_lease_is_fresh_within_timeout():
    lease = Lease(holder_id="worker-1", timestamp=time.time())
    assert not lease.is_stale()


def test_acquire_and_release_local_lock(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = DistributedLock(lock_path, holder_id="holder-a")

    assert lock.try_acquire()
    # Lock file should exist with correct holder
    with open(lock_path) as f:
        data = json.loads(f.read())
    assert data["holder_id"] == "holder-a"

    lock.release()
    assert not os.path.exists(lock_path)


def test_second_holder_blocked_while_lock_held(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock_a = DistributedLock(lock_path, holder_id="holder-a")
    lock_b = DistributedLock(lock_path, holder_id="holder-b")

    assert lock_a.try_acquire()
    assert not lock_b.try_acquire()

    lock_a.release()
    assert lock_b.try_acquire()
    lock_b.release()


def test_stale_lock_can_be_taken_over(tmp_path):
    lock_path = str(tmp_path / "test.lock")

    # Write a stale lock directly
    stale_lease = Lease(holder_id="dead-worker", timestamp=time.time() - HEARTBEAT_TIMEOUT - 10)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as f:
        f.write(json.dumps({"holder_id": stale_lease.holder_id, "timestamp": stale_lease.timestamp}))

    lock = DistributedLock(lock_path, holder_id="new-worker")
    assert lock.try_acquire()
    lock.release()


def test_refresh_updates_timestamp(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = DistributedLock(lock_path, holder_id="holder-a")
    assert lock.try_acquire()

    # Read initial timestamp
    with open(lock_path) as f:
        t1 = json.loads(f.read())["timestamp"]

    time.sleep(0.05)
    lock.refresh()

    with open(lock_path) as f:
        t2 = json.loads(f.read())["timestamp"]

    assert t2 > t1
    lock.release()


def test_refresh_fails_if_not_holder(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock_a = DistributedLock(lock_path, holder_id="holder-a")
    lock_b = DistributedLock(lock_path, holder_id="holder-b")

    assert lock_a.try_acquire()
    with pytest.raises(ValueError, match="held by"):
        lock_b.refresh()
    lock_a.release()


def test_has_active_holder(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = DistributedLock(lock_path, holder_id="holder-a")

    assert not lock.has_active_holder()
    assert lock.try_acquire()
    assert lock.has_active_holder()
    lock.release()
    assert not lock.has_active_holder()


def test_same_holder_can_reacquire(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = DistributedLock(lock_path, holder_id="holder-a")

    assert lock.try_acquire()
    assert lock.try_acquire()  # idempotent
    lock.release()
