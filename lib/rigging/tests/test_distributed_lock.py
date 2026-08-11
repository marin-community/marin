# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import PreconditionFailed
from rigging.filesystem.distributed_lock import GcsLease, LeaseLostError


def _gcs_client_with_blob(blob: MagicMock | None) -> MagicMock:
    client = MagicMock()
    client.bucket.return_value.get_blob.return_value = blob
    return client


def test_gcs_refresh_when_holder_changed_raises_lease_lost() -> None:
    lease = GcsLease("gs://bucket/test.lock", "worker-A")
    blob = MagicMock(generation=42)
    blob.download_as_string.return_value = json.dumps({"worker_id": "worker-B", "timestamp": time.time()})

    with (
        patch("rigging.filesystem.distributed_lock.storage.Client", return_value=_gcs_client_with_blob(blob)),
        pytest.raises(LeaseLostError, match="worker-B"),
    ):
        lease.refresh()


def test_gcs_refresh_when_lock_disappears_raises_lease_lost() -> None:
    lease = GcsLease("gs://bucket/test.lock", "worker-A")

    with (
        patch("rigging.filesystem.distributed_lock.storage.Client", return_value=_gcs_client_with_blob(None)),
        pytest.raises(LeaseLostError, match="disappeared"),
    ):
        lease.refresh()


def test_gcs_acquire_when_conditional_write_loses_returns_false() -> None:
    lease = GcsLease("gs://bucket/test.lock", "worker-A")

    client = _gcs_client_with_blob(None)
    client.bucket.return_value.blob.return_value.upload_from_string.side_effect = PreconditionFailed("gen mismatch")

    with patch("rigging.filesystem.distributed_lock.storage.Client", return_value=client):
        assert not lease.try_acquire()


def test_gcs_refresh_when_conditional_write_loses_raises_lease_lost() -> None:
    lease = GcsLease("gs://bucket/test.lock", "worker-A")
    blob = MagicMock(generation=42)
    blob.download_as_string.return_value = json.dumps({"worker_id": "worker-A", "timestamp": time.time()})
    client = _gcs_client_with_blob(blob)
    client.bucket.return_value.blob.return_value.upload_from_string.side_effect = PreconditionFailed("gen mismatch")

    with (
        patch("rigging.filesystem.distributed_lock.storage.Client", return_value=client),
        pytest.raises(LeaseLostError, match="refreshing"),
    ):
        lease.refresh()
