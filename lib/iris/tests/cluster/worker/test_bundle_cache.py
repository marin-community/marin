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

"""Tests for BundleCache."""

import hashlib
import os
import zipfile

import pytest
from iris.cluster.worker.bundle_cache import BundleCache


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def test_bundle(tmp_path):
    """Create a test bundle zip file."""
    bundle_dir = tmp_path / "test_bundle"
    bundle_dir.mkdir()

    # Create some test files
    (bundle_dir / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (bundle_dir / "main.py").write_text("print('hello')\n")

    src_dir = bundle_dir / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text("def foo(): pass\n")

    # Create zip file
    zip_path = tmp_path / "test_bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in bundle_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(bundle_dir))

    return zip_path


@pytest.fixture
def test_bundle_hash(test_bundle):
    """Compute hash of test bundle."""
    h = hashlib.sha256()
    with open(test_bundle, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_download_local_bundle(temp_cache_dir, test_bundle):
    """Test downloading a local bundle using file:// path."""
    cache = BundleCache(temp_cache_dir)

    # Use file:// protocol
    file_url = f"file://{test_bundle}"

    # Get bundle
    extract_path = cache.get_bundle(file_url)

    # Verify extraction
    assert extract_path.exists()
    assert extract_path.is_dir()
    assert (extract_path / "pyproject.toml").exists()
    assert (extract_path / "main.py").exists()
    assert (extract_path / "src" / "module.py").exists()


def test_caching_behavior(temp_cache_dir, test_bundle):
    """Test that bundles are cached and not re-downloaded."""
    cache = BundleCache(temp_cache_dir)

    file_url = f"file://{test_bundle}"

    # First download
    extract_path1 = cache.get_bundle(file_url)

    # Second request - should use cache and return same path
    extract_path2 = cache.get_bundle(file_url)

    assert extract_path1 == extract_path2


def test_hash_verification_success(temp_cache_dir, test_bundle, test_bundle_hash):
    """Test that hash verification passes with correct hash."""
    cache = BundleCache(temp_cache_dir)

    file_url = f"file://{test_bundle}"

    # Get bundle with correct hash - should succeed without raising
    extract_path = cache.get_bundle(file_url, expected_hash=test_bundle_hash)

    # Verify path is valid by checking we got something back
    assert extract_path is not None


def test_hash_verification_failure(temp_cache_dir, test_bundle):
    """Test that hash verification fails with incorrect hash."""
    cache = BundleCache(temp_cache_dir)

    file_url = f"file://{test_bundle}"

    # Use wrong hash
    wrong_hash = "a" * 64

    # Should raise ValueError
    with pytest.raises(ValueError, match="Bundle hash mismatch"):
        cache.get_bundle(file_url, expected_hash=wrong_hash)


def test_lru_eviction(temp_cache_dir, tmp_path):
    """Test LRU eviction when cache exceeds max_bundles."""
    # Create cache with max 2 bundles
    cache = BundleCache(temp_cache_dir, max_bundles=2)

    # Create 3 test bundles
    bundles = []
    for i in range(3):
        bundle_dir = tmp_path / f"bundle_{i}"
        bundle_dir.mkdir()
        (bundle_dir / "test.txt").write_text(f"bundle {i}")

        zip_path = tmp_path / f"bundle_{i}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(bundle_dir / "test.txt", "test.txt")

        bundles.append(zip_path)

    # Download bundles one at a time. Set distinct mtimes before the third
    # download so eviction (triggered inside get_bundle) sees deterministic order.
    paths = []
    for _i, bundle in enumerate(bundles):
        # Backdate earlier bundles so eviction picks the oldest
        for j, p in enumerate(paths):
            os.utime(p, (j, j))
        file_url = f"file://{bundle}"
        path = cache.get_bundle(file_url)
        paths.append(path)

    # First bundle should be evicted (only 2 should remain)
    assert not paths[0].exists(), "First bundle should be evicted"
    assert paths[1].exists(), "Second bundle should still exist"
    assert paths[2].exists(), "Third bundle should still exist"

    # Verify only 2 extracts exist
    extracts = list((temp_cache_dir / "extracts").iterdir())
    assert len(extracts) == 2


def test_concurrent_downloads(temp_cache_dir, test_bundle):
    """Test that concurrent downloads work correctly."""
    import concurrent.futures

    cache = BundleCache(temp_cache_dir)

    file_url = f"file://{test_bundle}"

    # Request same bundle multiple times concurrently using threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(cache.get_bundle, file_url) for _ in range(5)]
        paths = [f.result() for f in futures]

    # All should return the same path
    assert all(p == paths[0] for p in paths)
