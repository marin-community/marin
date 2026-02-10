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

import pytest
from iris.cluster.types import DEFAULT_BASE_IMAGE
from iris.test_util import SentinelFile
from iris.time_utils import Deadline, Duration


def _docker_image_exists(tag: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "images", "-q", tag],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_task_image_available: bool | None = None


def pytest_collection_modifyitems(config, items):
    """Skip docker-marked tests if the task image isn't available."""
    global _task_image_available
    if _task_image_available is None:
        _task_image_available = _docker_image_exists(DEFAULT_BASE_IMAGE)

    if _task_image_available:
        return

    skip = pytest.mark.skip(reason=f"Docker image {DEFAULT_BASE_IMAGE} not built")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


# httpx logs every HTTP request at INFO level, which floods test output
# during polling loops (status checks, log fetching).
logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture(autouse=True, scope="function")
def _ensure_logging_health():
    """Ensure logging handlers are healthy before and after each test.

    Removes any closed or invalid handlers before test setup to prevent
    "I/O operation on closed file" errors. Flushes after test completion
    to ensure buffered messages are written.
    """
    # Before test: remove any closed handlers from previous tests
    for handler in logging.root.handlers[:]:
        if hasattr(handler, "stream"):
            try:
                # Test if stream is writable
                handler.stream.closed  # noqa: B018
                if handler.stream.closed:
                    logging.root.removeHandler(handler)
            except (AttributeError, ValueError):
                # Handler doesn't have a stream or stream is invalid
                pass

    yield

    # After test: flush all handlers
    for handler in logging.root.handlers[:]:
        try:
            handler.flush()
        except (OSError, ValueError):
            # Handler may be closed or invalid
            pass


@pytest.fixture
def sentinel(tmp_path) -> SentinelFile:
    """Per-test sentinel file for blocking/unblocking job threads."""
    return SentinelFile(str(tmp_path / "sentinel"))


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

    deadline = Deadline.from_now(Duration.from_seconds(5.0))
    while not deadline.expired():
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
        # Only force-exit on failure to avoid skipping atexit handlers on clean runs
        if exitstatus != 0:
            os._exit(exitstatus)
