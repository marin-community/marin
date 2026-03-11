# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from unittest.mock import patch

import pytest

from iris.distributed_lock import (
    HEARTBEAT_TIMEOUT,
    GcsLease,
    Lease,
    LeaseLostError,
    LocalFileLease,
    create_lock,
)


def test_lease_is_stale_after_timeout():
    lease = Lease(worker_id="worker-1", timestamp=time.time() - HEARTBEAT_TIMEOUT - 1)
    assert lease.is_stale()


def test_lease_is_fresh_within_timeout():
    lease = Lease(worker_id="worker-1", timestamp=time.time())
    assert not lease.is_stale()


def test_create_lock_returns_local_for_local_path(tmp_path):
    lock = create_lock(str(tmp_path / "test.lock"), worker_id="w")
    assert isinstance(lock, LocalFileLease)


def test_create_lock_returns_gcs_for_gcs_path():
    lock = create_lock("gs://bucket/test.lock", worker_id="w")
    assert isinstance(lock, GcsLease)


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
    assert lock.try_acquire()
    lock.release()


def test_refresh_updates_timestamp(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")
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


def test_refresh_raises_lease_lost_error_if_not_holder(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock_a = create_lock(lock_path, worker_id="holder-a")
    lock_b = create_lock(lock_path, worker_id="holder-b")

    assert lock_a.try_acquire()
    with pytest.raises(LeaseLostError, match="held by"):
        lock_b.refresh()
    lock_a.release()


def test_has_active_holder(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")

    assert not lock.has_active_holder()
    assert lock.try_acquire()
    assert lock.has_active_holder()
    lock.release()
    assert not lock.has_active_holder()


def test_same_holder_can_reacquire(tmp_path):
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")

    assert lock.try_acquire()
    assert lock.try_acquire()  # idempotent
    lock.release()


def test_refresh_reacquires_when_lock_disappears(tmp_path):
    """Refresh should re-acquire the lock if it was deleted externally."""
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")
    assert lock.try_acquire()

    # Simulate another worker deleting the lock file
    os.remove(lock_path)

    # Refresh should re-acquire instead of crashing
    lock.refresh()

    # Lock should exist again with the same holder
    with open(lock_path) as f:
        data = json.loads(f.read())
    assert data["worker_id"] == "holder-a"
    lock.release()


def test_release_idempotent_when_lock_already_gone(tmp_path):
    """Release should not crash if the lock file was already deleted."""
    lock_path = str(tmp_path / "test.lock")
    lock = create_lock(lock_path, worker_id="holder-a")
    assert lock.try_acquire()

    # Delete manually
    os.remove(lock_path)

    # Release should not raise
    lock.release()


def test_has_active_holder_when_lock_missing(tmp_path):
    """has_active_holder should return False if lock file doesn't exist."""
    lock_path = str(tmp_path / "nonexistent.lock")
    lock = create_lock(lock_path, worker_id="holder-a")
    assert not lock.has_active_holder()


# ---------------------------------------------------------------------------
# GCS mock tests — validate NotFound handling and generation semantics
# ---------------------------------------------------------------------------


class _FakeBlob:
    """Minimal mock for google.cloud.storage.Blob."""

    def __init__(self, data: str | None, generation: int):
        self._data = data
        self.generation = generation

    def download_as_string(self) -> bytes:
        if self._data is None:
            from google.api_core.exceptions import NotFound

            raise NotFound("blob gone")
        return self._data.encode()

    def upload_from_string(self, data: str, if_generation_match: int = 0) -> None:
        pass

    def delete(self) -> None:
        pass


def _make_gcs_lease(lock_path: str = "gs://bucket/test.lock", worker_id: str = "w") -> GcsLease:
    return GcsLease(lock_path, worker_id)


@patch("iris.distributed_lock.GcsLease._read_with_generation")
def test_gcs_refresh_raises_lease_lost_on_different_holder(mock_read):
    """GcsLease.refresh raises LeaseLostError when another worker holds the lock."""
    lease = _make_gcs_lease(worker_id="worker-A")
    mock_read.return_value = (42, Lease(worker_id="worker-B", timestamp=time.time()))

    with pytest.raises(LeaseLostError, match="worker-B"):
        lease.refresh()


@patch("iris.distributed_lock.GcsLease._write")
@patch("iris.distributed_lock.GcsLease._read_with_generation")
def test_gcs_refresh_reacquires_when_lock_gone(mock_read, mock_write):
    """GcsLease.refresh re-acquires when lock disappears (NotFound race)."""
    lease = _make_gcs_lease(worker_id="worker-A")
    mock_read.return_value = (0, None)

    lease.refresh()

    mock_write.assert_called_once()
    args = mock_write.call_args
    assert args[0][0].worker_id == "worker-A"
    assert args[0][1] == 0  # generation=0 → create-if-absent


@patch("iris.distributed_lock.GcsLease._read_with_generation")
def test_gcs_has_active_holder_returns_false_on_missing(mock_read):
    """has_active_holder returns False when the lock blob is absent."""
    lease = _make_gcs_lease()
    mock_read.return_value = (0, None)

    assert not lease.has_active_holder()


@patch("iris.distributed_lock.GcsLease._delete")
@patch("iris.distributed_lock.GcsLease._read_with_generation")
def test_gcs_release_idempotent_when_gone(mock_read, mock_delete):
    """release() is a no-op when the lock is already gone."""
    lease = _make_gcs_lease(worker_id="worker-A")
    mock_read.return_value = (0, None)

    # Should not raise
    lease.release()
    mock_delete.assert_not_called()


@patch("iris.distributed_lock.GcsLease._read_with_generation")
def test_gcs_try_acquire_returns_false_on_precondition_failed(mock_read):
    """try_acquire returns False on PreconditionFailed (GCS generation race)."""
    lease = _make_gcs_lease(worker_id="worker-A")
    mock_read.return_value = (0, None)

    # Simulate PreconditionFailed from _write
    class PreconditionFailed(Exception):
        pass

    with patch.object(lease, "_write", side_effect=PreconditionFailed("gen mismatch")):
        assert not lease.try_acquire()
