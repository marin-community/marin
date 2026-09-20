# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import PreconditionFailed
from rigging.filesystem.distributed_lock import DistributedLease, GcsLease, LeaseLostError, lease_refresh


class RecordingLease:
    lock_path = "memory://test.lock"

    def __init__(self) -> None:
        self.refresh_started = threading.Event()
        self.finish_refresh = threading.Event()
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1
        self.refresh_started.set()
        assert self.finish_refresh.wait(timeout=1)


class LostLease:
    lock_path = "memory://lost.lock"

    def __init__(self) -> None:
        self.refresh_started = threading.Event()

    def refresh(self) -> None:
        self.refresh_started.set()
        raise LeaseLostError("lease changed owners")


def _gcs_client_with_blob(blob: MagicMock | None) -> MagicMock:
    client = MagicMock()
    client.bucket.return_value.get_blob.return_value = blob
    return client


def test_gcs_refresh_when_holder_changed_raises_lease_lost() -> None:
    lease = GcsLease("gs://bucket/test.lock", "worker-A")
    blob = MagicMock(generation=42)
    blob.download_as_bytes.return_value = json.dumps({"worker_id": "worker-B", "timestamp": time.time()}).encode()

    with (
        patch("rigging.filesystem.conditional_object._gcs_client", return_value=_gcs_client_with_blob(blob)),
        pytest.raises(LeaseLostError, match="worker-B"),
    ):
        lease.refresh()


def test_gcs_refresh_when_lock_disappears_raises_lease_lost() -> None:
    lease = GcsLease("gs://bucket/test.lock", "worker-A")

    with (
        patch("rigging.filesystem.conditional_object._gcs_client", return_value=_gcs_client_with_blob(None)),
        pytest.raises(LeaseLostError, match="disappeared"),
    ):
        lease.refresh()


def test_gcs_acquire_when_conditional_write_loses_returns_false() -> None:
    lease = GcsLease("gs://bucket/test.lock", "worker-A")

    client = _gcs_client_with_blob(None)
    client.bucket.return_value.blob.return_value.upload_from_string.side_effect = PreconditionFailed("gen mismatch")

    with patch("rigging.filesystem.conditional_object._gcs_client", return_value=client):
        assert not lease.try_acquire()


def test_lease_refresh_keeps_lease_fresh_and_checks_ownership_on_exit() -> None:
    lease = RecordingLease()

    with lease_refresh(cast(DistributedLease, lease), interval=0.001):
        assert lease.refresh_started.wait(timeout=1)
        lease.finish_refresh.set()

    assert lease.refresh_count >= 2


def test_lease_refresh_raises_background_refresh_failure() -> None:
    lease = LostLease()

    with pytest.raises(LeaseLostError, match="changed owners"):
        with lease_refresh(cast(DistributedLease, lease), interval=0.001):
            assert lease.refresh_started.wait(timeout=1)
