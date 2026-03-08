# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller-side BundleStore behavior."""

import hashlib

from connectrpc.errors import ConnectError
import pytest

from iris.cluster.bundle import BundleStore


@pytest.fixture
def store(tmp_path):
    return BundleStore(db_path=tmp_path / "bundles.sqlite3")


def test_write_zip_returns_content_hash_id(store):
    blob = b"test bundle content"

    bundle_id = store.write_zip(blob)
    assert bundle_id == hashlib.sha256(blob).hexdigest()
    assert store.get_zip(bundle_id) == blob


def test_write_zip_is_idempotent(store):
    blob = b"same bytes"

    id1 = store.write_zip(blob)
    id2 = store.write_zip(blob)
    assert id1 == id2


def test_get_zip_reads_stored_bytes(store):
    blob = b"bundle data"

    bundle_id = store.write_zip(blob)
    assert store.get_zip(bundle_id) == blob


def test_get_zip_missing_raises_not_found(store):
    with pytest.raises(ConnectError, match="Bundle not found"):
        store.get_zip("a" * 64)
