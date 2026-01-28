# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for bundle storage functionality."""

import tempfile
from pathlib import Path

import pytest

from iris.cluster.controller.bundle_store import BundleStore


@pytest.fixture
def temp_bundle_dir():
    """Create a temporary directory for bundle storage testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_write_bundle_creates_parent_directories_for_local_path(temp_bundle_dir):
    """Verify write_bundle creates parent directories for local file:// paths."""
    bundle_prefix = f"file://{temp_bundle_dir}/bundles"
    store = BundleStore(bundle_prefix)

    job_id = "test-job-123"
    blob = b"test bundle content"

    # Write bundle - should create parent directories automatically
    bundle_path = store.write_bundle(job_id, blob)

    # Verify the bundle was written
    expected_path = temp_bundle_dir / "bundles" / job_id / "bundle.zip"
    assert expected_path.exists()
    assert expected_path.read_bytes() == blob
    assert bundle_path == f"{bundle_prefix}/{job_id}/bundle.zip"


def test_write_bundle_to_existing_directory(temp_bundle_dir):
    """Verify write_bundle works when parent directory already exists."""
    bundle_prefix = f"file://{temp_bundle_dir}/bundles"
    store = BundleStore(bundle_prefix)

    job_id = "test-job-456"

    # Pre-create the parent directory structure
    parent_dir = temp_bundle_dir / "bundles" / job_id
    parent_dir.mkdir(parents=True, exist_ok=True)

    blob = b"test bundle content"

    # Write bundle - should work with existing directory
    bundle_path = store.write_bundle(job_id, blob)

    # Verify the bundle was written
    expected_path = parent_dir / "bundle.zip"
    assert expected_path.exists()
    assert expected_path.read_bytes() == blob
    assert bundle_path == f"{bundle_prefix}/{job_id}/bundle.zip"


def test_write_multiple_bundles_different_jobs(temp_bundle_dir):
    """Verify write_bundle handles multiple jobs correctly."""
    bundle_prefix = f"file://{temp_bundle_dir}/bundles"
    store = BundleStore(bundle_prefix)

    jobs = [
        ("job-1", b"bundle content 1"),
        ("job-2", b"bundle content 2"),
        ("job-3", b"bundle content 3"),
    ]

    for job_id, blob in jobs:
        store.write_bundle(job_id, blob)

    # Verify all bundles were written
    for job_id, blob in jobs:
        expected_path = temp_bundle_dir / "bundles" / job_id / "bundle.zip"
        assert expected_path.exists()
        assert expected_path.read_bytes() == blob


def test_write_bundle_overwrites_existing(temp_bundle_dir):
    """Verify write_bundle overwrites existing bundle for same job."""
    bundle_prefix = f"file://{temp_bundle_dir}/bundles"
    store = BundleStore(bundle_prefix)

    job_id = "overwrite-job"

    # Write first version
    store.write_bundle(job_id, b"original content")

    # Write second version (should overwrite)
    store.write_bundle(job_id, b"updated content")

    # Verify only the updated content exists
    expected_path = temp_bundle_dir / "bundles" / job_id / "bundle.zip"
    assert expected_path.read_bytes() == b"updated content"
