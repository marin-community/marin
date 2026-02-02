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

import subprocess

import pytest
from iris.cluster.worker.builder import ImageCache

# ImageCache Tests


@pytest.fixture
def docker_bundle(tmp_path):
    """Create a minimal test bundle for Docker builds."""
    bundle_dir = tmp_path / "docker_bundle"
    bundle_dir.mkdir()

    # Create package directory structure
    src_dir = bundle_dir / "src" / "test_app"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text('"""Test app."""\n')
    (src_dir / "main.py").write_text(
        'def main():\n    print("Hello from Docker!")\n\nif __name__ == "__main__":\n    main()\n'
    )

    # Create pyproject.toml
    pyproject = """[project]
name = "test-app"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/test_app"]
"""
    (bundle_dir / "pyproject.toml").write_text(pyproject)

    # Create uv.lock
    uv_lock = """version = 1
requires-python = ">=3.11"

[[package]]
name = "test-app"
version = "0.1.0"
source = { editable = "." }
"""
    (bundle_dir / "uv.lock").write_text(uv_lock)

    return bundle_dir


def check_docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.slow
def test_image_caching(tmp_path, docker_bundle, docker_cleanup_scope):
    """Test that subsequent builds use the cache."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    cache_dir = tmp_path / "cache"
    builder = ImageCache(cache_dir, registry="localhost:5000")

    job_id = "cache-test-456"
    base_image = "python:3.11-slim"

    # First build - not from cache
    result1 = builder.build(
        bundle_path=docker_bundle,
        base_image=base_image,
        extras=[],
        job_id=job_id,
    )

    assert result1.from_cache is False

    # Second build - should be from cache
    result2 = builder.build(
        bundle_path=docker_bundle,
        base_image=base_image,
        extras=[],
        job_id=job_id,
    )

    assert result2.from_cache is True
    assert result2.image_tag == result1.image_tag


@pytest.mark.slow
def test_image_build_with_extras(tmp_path, docker_cleanup_scope):
    """Test building image with extras."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Create bundle with extras
    bundle_dir = tmp_path / "extras_bundle"
    bundle_dir.mkdir()

    src_dir = bundle_dir / "src" / "test_app"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text('"""Test app."""\n')

    pyproject = """[project]
name = "test-app"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = []
test = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/test_app"]
"""
    (bundle_dir / "pyproject.toml").write_text(pyproject)

    # Generate proper lock file
    try:
        subprocess.run(
            ["uv", "lock"],
            cwd=bundle_dir,
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("uv not available or failed to create lock file")

    cache_dir = tmp_path / "cache"
    builder = ImageCache(cache_dir, registry="localhost:5000")

    # Build with extras
    result = builder.build(
        bundle_path=bundle_dir,
        base_image="python:3.11-slim",
        extras=["dev", "test"],
        job_id="extras-test",
    )

    assert result.from_cache is False


def test_protect_unprotect_refcounting(tmp_path):
    """Test that protect/unprotect correctly manages refcounts."""
    cache_dir = tmp_path / "cache"
    builder = ImageCache(cache_dir, registry="localhost:5000")

    tag = "test-image:v1"

    # Initially no refcount
    assert tag not in builder._image_refcounts

    # Protect once
    builder.protect(tag)
    assert builder._image_refcounts[tag] == 1

    # Protect again (multiple jobs using same image)
    builder.protect(tag)
    assert builder._image_refcounts[tag] == 2

    # Unprotect once
    builder.unprotect(tag)
    assert builder._image_refcounts[tag] == 1

    # Unprotect again - should be removed from dict
    builder.unprotect(tag)
    assert tag not in builder._image_refcounts

    # Unprotect on non-existent tag should be safe
    builder.unprotect(tag)
    assert tag not in builder._image_refcounts


def test_cache_eviction_enforces_max_images(tmp_path):
    """Test that cache eviction respects max_images limit."""
    cache_dir = tmp_path / "cache"
    max_images = 3
    builder = ImageCache(cache_dir, registry="localhost:5000", max_images=max_images)

    from unittest.mock import MagicMock
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class MockImage:
        tag: str
        created_at: datetime

    # Mock docker to simulate having many images
    builder._docker.exists = MagicMock(return_value=False)
    builder._docker.build = MagicMock()

    # Create mock images (oldest to newest)
    mock_images = [
        MockImage(tag="localhost:5000/iris-job-old1:abc123", created_at=datetime(2025, 1, 1, 10, 0)),
        MockImage(tag="localhost:5000/iris-job-old2:def456", created_at=datetime(2025, 1, 1, 11, 0)),
        MockImage(tag="localhost:5000/iris-job-old3:ghi789", created_at=datetime(2025, 1, 1, 12, 0)),
        MockImage(tag="localhost:5000/iris-job-new1:jkl012", created_at=datetime(2025, 1, 1, 13, 0)),
        MockImage(tag="localhost:5000/iris-job-new2:mno345", created_at=datetime(2025, 1, 1, 14, 0)),
    ]

    builder._docker.list_images = MagicMock(return_value=mock_images)
    builder._docker.remove = MagicMock()

    # Trigger eviction
    builder._evict_old_images()

    # Should have removed 2 oldest images (5 total - 3 max = 2 to remove)
    assert builder._docker.remove.call_count == 2
    removed_tags = {call[0][0] for call in builder._docker.remove.call_args_list}
    assert removed_tags == {
        "localhost:5000/iris-job-old1:abc123",
        "localhost:5000/iris-job-old2:def456",
    }


def test_cache_eviction_respects_protected_images(tmp_path):
    """Test that protected images are not evicted even when max_images exceeded."""
    cache_dir = tmp_path / "cache"
    max_images = 2
    builder = ImageCache(cache_dir, registry="localhost:5000", max_images=max_images)

    from unittest.mock import MagicMock
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class MockImage:
        tag: str
        created_at: datetime

    builder._docker.exists = MagicMock(return_value=False)
    builder._docker.build = MagicMock()

    # Create 4 images: 1 protected, 3 unprotected
    # With max_images=2, we should evict 1 unprotected (oldest)
    mock_images = [
        MockImage(tag="localhost:5000/iris-job-protected:abc123", created_at=datetime(2025, 1, 1, 10, 0)),
        MockImage(tag="localhost:5000/iris-job-old:def456", created_at=datetime(2025, 1, 1, 11, 0)),
        MockImage(tag="localhost:5000/iris-job-mid:ghi789", created_at=datetime(2025, 1, 1, 12, 0)),
        MockImage(tag="localhost:5000/iris-job-new:jkl012", created_at=datetime(2025, 1, 1, 13, 0)),
    ]

    # Protect the oldest image
    builder.protect("localhost:5000/iris-job-protected:abc123")

    builder._docker.list_images = MagicMock(return_value=mock_images)
    builder._docker.remove = MagicMock()

    # Trigger eviction
    builder._evict_old_images()

    # Should only remove the oldest unprotected image
    # 3 evictable images - 2 max_images = 1 to remove
    builder._docker.remove.assert_called_once_with("localhost:5000/iris-job-old:def456")
