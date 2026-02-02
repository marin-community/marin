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
import os
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from iris.managed_thread import thread_registry_scope
from iris.test_util import SentinelFile

# httpx logs every HTTP request at INFO level, which floods test output
# during polling loops (status checks, log fetching).
logging.getLogger("httpx").setLevel(logging.WARNING)


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


@pytest.fixture
def sentinel(tmp_path) -> SentinelFile:
    """Per-test sentinel file for blocking/unblocking job threads."""
    return SentinelFile(str(tmp_path / "sentinel"))


@pytest.fixture
def registry():
    """Provides an isolated thread registry for the test.

    Automatically cleans up all threads spawned within the registry when
    the test completes. This is the recommended way to use ThreadContainer
    in tests that need explicit thread management.

    Example:
        def test_with_threads(registry):
            registry.spawn(my_thread_fn, name="test-thread")
            # Test code...
            # Cleanup happens automatically
    """
    with thread_registry_scope("test") as reg:
        yield reg


@pytest.fixture(autouse=True)
def _thread_cleanup():
    """Ensure no new non-daemon threads leak from each test.

    Takes a snapshot of threads before the test and checks that no new
    non-daemon threads remain after teardown. Waits briefly for threads
    that are in the process of shutting down.

    This fixture helps catch tests that don't properly clean up their threads,
    which can cause tests to hang or interfere with each other.
    """
    before = {t.ident for t in threading.enumerate()}
    yield

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        leaked = [
            t
            for t in threading.enumerate()
            if t.is_alive() and not t.daemon and t.name != "MainThread" and t.ident not in before
        ]
        if not leaked:
            return
        time.sleep(0.1)

    # Generate detailed warning about leaked threads
    thread_info = []
    for t in leaked:
        thread_info.append(f"{t.name} (daemon={t.daemon}, ident={t.ident})")

    warnings.warn(
        f"Threads leaked from test: {thread_info}\n"
        "All threads should be stopped via ThreadContainer.stop() or similar cleanup.\n"
        "See lib/iris/tests/test_utils.py for best practices.",
        stacklevel=1,
    )


def pytest_sessionfinish(session, exitstatus):
    """Hook to debug pytest exit status - dump any non-daemon threads still alive.

    Uses os._exit() to force-terminate if orphaned non-daemon threads would
    otherwise prevent process exit.
    """
    alive = [t for t in threading.enumerate() if t.is_alive() and not t.daemon and t.name != "MainThread"]
    if alive:
        tty = os.fdopen(os.dup(2), "w")
        tty.write(f"\n⚠ {len(alive)} non-daemon threads still alive at session end:\n")
        frames = sys._current_frames()
        for t in alive:
            tty.write(f"\n  Thread: {t.name} (daemon={t.daemon}, ident={t.ident})\n")
            frame = frames.get(t.ident)
            if frame:
                for line in traceback.format_stack(frame):
                    tty.write(f"    {line.rstrip()}\n")
        tty.flush()
        tty.close()
        # Force exit to prevent orphaned non-daemon threads from blocking
        os._exit(exitstatus)
