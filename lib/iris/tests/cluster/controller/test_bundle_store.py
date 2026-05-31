# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller-side BundleStore behavior."""

import hashlib

import pytest
from iris.cluster.bundle import BundleStore


@pytest.fixture
def store(tmp_path):
    return BundleStore(storage_dir=str(tmp_path / "bundles"))


def test_write_returns_content_hash_id(store):
    blob = b"test bundle content"

    bundle_id = store.write(blob)
    assert bundle_id == hashlib.sha256(blob).hexdigest()
    assert store.get(bundle_id) == blob


def test_write_is_idempotent(store):
    blob = b"same bytes"

    id1 = store.write(blob)
    id2 = store.write(blob)
    assert id1 == id2


def test_get_reads_stored_bytes(store):
    blob = b"bundle data"

    bundle_id = store.write(blob)
    assert store.get(bundle_id) == blob


def test_get_missing_raises_not_found(store):
    with pytest.raises(FileNotFoundError, match="not found and no controller configured"):
        store.get("a" * 64)


def test_store_survives_restart(tmp_path):
    """Re-creating BundleStore from same directory recovers bundles via fsspec."""
    storage_dir = str(tmp_path / "bundles")
    store = BundleStore(storage_dir=storage_dir)
    blob = b"persist me"
    bundle_id = store.write(blob)
    store.close()

    store2 = BundleStore(storage_dir=storage_dir)
    assert store2.get(bundle_id) == blob


def test_write_skips_upload_when_already_in_storage(tmp_path):
    """write should not re-upload if content exists in storage but was evicted from cache."""
    storage_dir = str(tmp_path / "bundles")
    store = BundleStore(storage_dir=storage_dir, max_cache_items=1)

    blob_a = b"bundle A"
    blob_b = b"bundle B"
    id_a = store.write(blob_a)
    store.write(blob_b)  # evicts blob_a from in-memory cache

    original_open = store._fs.open
    write_paths: list[str] = []

    def tracking_open(path, mode="rb", *args, **kwargs):
        if "w" in mode:
            write_paths.append(path)
        return original_open(path, mode, *args, **kwargs)

    store._fs.open = tracking_open
    id_a2 = store.write(blob_a)
    assert id_a2 == id_a
    assert write_paths == [], "write should not re-upload when content exists in storage"


def test_bundle_and_blob_share_storage(store):
    """Identical bytes written via any path share one disk entry and id."""
    data = b"shared content"

    id_first = store.write(data)
    id_second = store.write(data)
    assert id_first == id_second
    assert store.get(id_first) == data


def test_cache_hit_does_not_skip_disk_write(tmp_path):
    """A second write of identical bytes must still ensure disk persistence
    even when the in-memory cache already has the entry — otherwise the
    content would be unreachable after controller restart or eviction.
    """
    storage_dir = str(tmp_path / "bundles")
    store = BundleStore(storage_dir=storage_dir)
    data = b"cached but maybe undisked"

    cid = store.write(data)

    # Simulate eviction by deleting the file under the cache.
    path = store._path_for(cid)
    store._fs.rm(path)
    assert not store._fs.exists(path)

    # Re-write should detect the missing file and restore it.
    store.write(data)
    assert store._fs.exists(path)

    # Fresh store (cold cache) should still load from disk.
    store2 = BundleStore(storage_dir=storage_dir)
    assert store2.get(cid) == data
