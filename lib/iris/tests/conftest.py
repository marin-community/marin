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

import logging
import subprocess
import sys
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
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Update golden snapshot files instead of comparing",
    )


@pytest.fixture(scope="session")
def use_docker(request):
    return request.config.getoption("--use-docker")


# =============================================================================
# Docker Cleanup Utilities
# =============================================================================
# TODO: This before/after snapshot approach can accidentally delete containers
# created by other processes during the test window. Fix by patching
# DockerRuntime -> TrackingDockerRuntime at the module level so all Docker
# operations are tracked explicitly. See review.md for details.


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
def docker_cleanup_session(use_docker):
    """Session-scoped cleanup for e2e tests that use real Docker."""
    if use_docker:
        with _docker_cleanup(cleanup_images=True):
            yield
    else:
        # Skip Docker cleanup when not using Docker
        yield


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that silently ignores writes to closed streams.

    Background threads may continue running briefly after pytest closes
    stdout/stderr during test cleanup. This handler prevents
    "ValueError: I/O operation on closed file" errors from third-party
    library logging (httpx, uvicorn, etc.) that happens in those threads.
    """

    def emit(self, record):
        try:
            super().emit(record)
        except ValueError as e:
            if "I/O operation on closed file" not in str(e):
                raise
            # Silently ignore - pytest closed our streams during cleanup


def pytest_configure(config):
    """Install safe stream handler to prevent closed file errors during cleanup."""
    # Replace all StreamHandlers with SafeStreamHandler in root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, _SafeStreamHandler):
            # Preserve handler configuration
            safe_handler = _SafeStreamHandler(handler.stream)
            safe_handler.setLevel(handler.level)
            safe_handler.setFormatter(handler.formatter)
            root_logger.removeHandler(handler)
            root_logger.addHandler(safe_handler)


def pytest_sessionfinish(session, exitstatus):
    """Hook to debug pytest exit status - helps diagnose silent failures."""
    if exitstatus == 0:
        print("\nPytest thinks everything is fine...", file=sys.stderr)
    else:
        print(f"\nPytest is exiting with status {exitstatus}", file=sys.stderr)
