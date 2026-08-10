# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller-side BundleStore behavior."""

import hashlib

import pytest
from fsspec.implementations.local import LocalFileSystem
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


def test_write_when_content_is_persisted_skips_second_filesystem_write(tmp_path, monkeypatch):
    storage_path = tmp_path / "bundles"
    storage_dir = str(storage_path)
    store = BundleStore(storage_dir=storage_dir, max_cache_items=1)

    blob_a = b"bundle A"
    blob_b = b"bundle B"
    id_a = store.write(blob_a)
    store.write(blob_b)  # evicts blob_a from in-memory cache
    persisted_path = storage_path / id_a
    assert persisted_path.read_bytes() == blob_a

    original_open = LocalFileSystem.open
    writes: list[str] = []

    def tracking_open(filesystem, path, mode="rb", *args, **kwargs):
        if "w" in mode and path == str(persisted_path):
            writes.append(path)
        return original_open(filesystem, path, mode, *args, **kwargs)

    monkeypatch.setattr(LocalFileSystem, "open", tracking_open)
    id_a2 = store.write(blob_a)
    assert id_a2 == id_a
    assert writes == []


def test_write_when_cached_file_is_missing_restores_persistent_content(tmp_path):
    """A second write of identical bytes must still ensure disk persistence
    even when the in-memory cache already has the entry — otherwise the
    content would be unreachable after controller restart or eviction.
    """
    storage_path = tmp_path / "bundles"
    storage_dir = str(storage_path)
    store = BundleStore(storage_dir=storage_dir)
    data = b"cached but maybe undisked"

    cid = store.write(data)

    # Simulate eviction by deleting the file under the cache.
    path = storage_path / cid
    path.unlink()
    assert not path.exists()

    # Re-write should detect the missing file and restore it.
    store.write(data)
    assert path.read_bytes() == data

    # Fresh store (cold cache) should still load from disk.
    store2 = BundleStore(storage_dir=storage_dir)
    assert store2.get(cid) == data
