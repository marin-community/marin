# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for bundle storage functionality."""

import hashlib
import tempfile
from pathlib import Path

import fsspec
import pytest
from connectrpc.errors import ConnectError
from iris.cluster.bundle import ControllerBundleStore


@pytest.fixture
def temp_bundle_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_write_zip_returns_content_hash_id(temp_bundle_dir):
    bundle_prefix = f"file://{temp_bundle_dir}/bundles"
    store = ControllerBundleStore(bundle_prefix)
    blob = b"test bundle content"

    bundle_id = store.write_zip(blob)
    assert bundle_id == hashlib.sha256(blob).hexdigest()

    bundle_uri = store.bundle_uri_for_id(bundle_id)
    fs, path = fsspec.core.url_to_fs(bundle_uri)
    assert fs.exists(path)


def test_write_zip_is_idempotent(temp_bundle_dir):
    bundle_prefix = f"file://{temp_bundle_dir}/bundles"
    store = ControllerBundleStore(bundle_prefix)
    blob = b"same bytes"

    id1 = store.write_zip(blob)
    id2 = store.write_zip(blob)
    assert id1 == id2


def test_get_zip_reads_stored_bytes(temp_bundle_dir):
    bundle_prefix = f"file://{temp_bundle_dir}/bundles"
    store = ControllerBundleStore(bundle_prefix)
    blob = b"bundle data"

    bundle_id = store.write_zip(blob)
    assert store.get_zip(bundle_id) == blob


def test_get_zip_missing_raises_not_found(temp_bundle_dir):
    bundle_prefix = f"file://{temp_bundle_dir}/bundles"
    store = ControllerBundleStore(bundle_prefix)
    with pytest.raises(ConnectError, match="Bundle not found"):
        store.get_zip("a" * 64)
