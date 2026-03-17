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


def test_migration_script_helpers(tmp_path):
    """Test the offline migration helpers from migrate_bundle_store.py."""
    import sqlite3
    import time

    from scripts.migrations.migrate_bundle_store import (
        read_bundles_from_sqlite,
        verify_bundles,
        write_bundles_to_fsspec,
    )

    # Create a legacy SQLite bundle store
    sqlite_path = tmp_path / "bundles.sqlite3"
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute(
        "CREATE TABLE bundles ("
        "bundle_id TEXT PRIMARY KEY, zip_bytes BLOB NOT NULL, "
        "created_at_ms INTEGER NOT NULL, last_access_ms INTEGER NOT NULL, size_bytes INTEGER NOT NULL)"
    )
    blob = b"legacy bundle data"
    bundle_id = hashlib.sha256(blob).hexdigest()
    now_ms = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO bundles (bundle_id, zip_bytes, created_at_ms, last_access_ms, size_bytes) VALUES (?, ?, ?, ?, ?)",
        (bundle_id, blob, now_ms, now_ms, len(blob)),
    )
    conn.commit()
    conn.close()

    # Read from sqlite
    bundles = read_bundles_from_sqlite(sqlite_path)
    assert len(bundles) == 1
    assert bundles[bundle_id] == blob

    # Write to fsspec storage
    storage_dir = str(tmp_path / "new_bundles")
    written = write_bundles_to_fsspec(bundles, storage_dir)
    assert written == 1

    # Idempotent: second write should skip existing
    written2 = write_bundles_to_fsspec(bundles, storage_dir)
    assert written2 == 0

    # Verify via BundleStore
    failures = verify_bundles(bundles, storage_dir)
    assert failures == []
