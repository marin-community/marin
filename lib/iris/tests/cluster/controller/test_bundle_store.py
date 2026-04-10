# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller-side BundleStore behavior."""

import hashlib

import pytest

from iris.cluster.bundle import BundleStore


@pytest.fixture
def store(tmp_path):
    return BundleStore(storage_dir=str(tmp_path / "bundles"))


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
    with pytest.raises(FileNotFoundError, match="Bundle not found"):
        store.get_zip("a" * 64)


def test_store_survives_restart(tmp_path):
    """Re-creating BundleStore from same directory recovers bundles via fsspec."""
    storage_dir = str(tmp_path / "bundles")
    store = BundleStore(storage_dir=storage_dir)
    blob = b"persist me"
    bundle_id = store.write_zip(blob)
    store.close()

    store2 = BundleStore(storage_dir=storage_dir)
    assert store2.get_zip(bundle_id) == blob


def test_get_or_fetch_missing_no_controller_raises_not_found(store):
    """get_or_fetch raises FileNotFoundError when content is absent and no controller is configured."""
    with pytest.raises(FileNotFoundError):
        store.get_or_fetch("b" * 64, "blobs/" + "b" * 64)


def test_write_zip_skips_upload_when_already_in_storage(tmp_path):
    """write_zip should not re-upload if bundle exists in storage but was evicted from cache."""
    storage_dir = str(tmp_path / "bundles")
    store = BundleStore(storage_dir=storage_dir, max_cache_items=1)

    blob_a = b"bundle A"
    blob_b = b"bundle B"
    id_a = store.write_zip(blob_a)
    store.write_zip(blob_b)  # evicts blob_a from in-memory cache

    # blob_a should still be in storage; re-submitting should not call _write_to_storage
    original_write = store._write_to_storage
    write_calls = []

    def tracking_write(bundle_id, blob):
        write_calls.append(bundle_id)
        return original_write(bundle_id, blob)

    store._write_to_storage = tracking_write
    id_a2 = store.write_zip(blob_a)
    assert id_a2 == id_a
    assert write_calls == [], "write_zip should not re-upload when bundle exists in storage"


def test_write_blob_returns_content_hash(store):
    data = b"large pickle payload"

    blob_id = store.write_blob(data)
    assert blob_id == hashlib.sha256(data).hexdigest()
    assert store.get_blob(blob_id) == data


def test_write_blob_is_idempotent(store):
    data = b"same blob bytes"

    id1 = store.write_blob(data)
    id2 = store.write_blob(data)
    assert id1 == id2


def test_get_blob_missing_raises(store):
    with pytest.raises(RuntimeError, match="not cached and controller address is not configured"):
        store.get_blob("b" * 64)


def test_blob_survives_restart(tmp_path):
    storage_dir = str(tmp_path / "bundles")
    store = BundleStore(storage_dir=storage_dir)
    data = b"persist this blob"
    blob_id = store.write_blob(data)
    store.close()

    store2 = BundleStore(storage_dir=storage_dir)
    assert store2.get_blob(blob_id) == data


def test_blob_and_bundle_namespaces_independent(store):
    """Blobs and bundles with identical content get separate storage paths."""
    data = b"shared content"

    bundle_id = store.write_zip(data)
    blob_id = store.write_blob(data)
    # Same hash since content is identical
    assert bundle_id == blob_id
    # Both retrievable via their respective methods
    assert store.get_zip(bundle_id) == data
    assert store.get_blob(blob_id) == data
