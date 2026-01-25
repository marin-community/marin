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

# Test configuration for iris

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--use-docker",
        action="store_true",
        default=False,
        help="Use real Docker containers instead of in-process mocks",
    )


@pytest.fixture(scope="session")
def use_docker(request):
    return request.config.getoption("--use-docker")


# =============================================================================
# Docker Cleanup Utilities
# =============================================================================


def _get_container_ids() -> set[str]:
    """Get all container IDs (running and stopped)."""
    result = subprocess.run(["docker", "ps", "-aq"], capture_output=True, text=True, check=False)
    return set(result.stdout.strip().split()) if result.stdout.strip() else set()


def _get_image_ids() -> set[str]:
    """Get all image IDs."""
    result = subprocess.run(["docker", "images", "-q"], capture_output=True, text=True, check=False)
    return set(result.stdout.strip().split()) if result.stdout.strip() else set()


@contextmanager
def _docker_cleanup(*, cleanup_images: bool = True) -> Iterator[None]:
    """Context manager that cleans up Docker artifacts created during its scope."""
    containers_before = _get_container_ids()
    images_before = _get_image_ids() if cleanup_images else set()

    try:
        yield
    finally:
        for cid in _get_container_ids() - containers_before:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)
        if cleanup_images:
            for iid in _get_image_ids() - images_before:
                subprocess.run(["docker", "rmi", "-f", iid], capture_output=True, check=False)


@pytest.fixture
def docker_cleanup_scope():
    """Cleans up all Docker containers and images created during the test."""
    with _docker_cleanup(cleanup_images=True):
        yield


@pytest.fixture(scope="session")
def docker_cleanup_session():
    """Session-scoped cleanup for e2e tests that use real Docker."""
    with _docker_cleanup(cleanup_images=True):
        yield
