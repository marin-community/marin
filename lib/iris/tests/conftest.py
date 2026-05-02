# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Test configuration for iris

import logging
import os
import subprocess
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import pytest
from iris.cluster.config import load_config, make_local_config
from iris.managed_thread import thread_container_scope
from iris.rpc import config_pb2
from iris.test_util import SentinelFile
from rigging.timing import Duration, ExponentialBackoff

IRIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "test.yaml"


def _make_controller_only_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with no auto-scaled workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    sg = config.scale_groups["placeholder"]
    sg.name = "placeholder"
    sg.num_vms = 1
    sg.buffer_slices = 0
    sg.max_slices = 0
    sg.resources.cpu_millicores = 1000
    sg.resources.memory_bytes = 1 * 1024**3
    sg.resources.disk_bytes = 10 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.resources.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
    sg.slice_template.local.SetInParent()
    return make_local_config(config)


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
        _task_image_available = _docker_image_exists("iris-task:latest")

    if _task_image_available:
        return

    skip = pytest.mark.skip(reason="Docker image iris-task:latest not built")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="function")
def _ensure_logging_health():
    """Ensure logging handlers are healthy before and after each test.

    Removes any closed or invalid handlers before test setup to prevent
    "I/O operation on closed file" errors. Flushes after test completion
    to ensure buffered messages are written.
    """
    # Before test: remove any closed handlers from previous tests
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            try:
                if handler.stream.closed:
                    logging.root.removeHandler(handler)
            except ValueError:
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
def _thread_cleanup(request):
    """Isolate each test's managed threads and warn on leaks.

    Installs a fresh ThreadContainer via thread_container_scope() so every
    component that calls get_thread_container() (e.g. Controller._start_local_log_server)
    registers its threads into a per-test container that is stopped on teardown.
    This ensures log-server uvicorn threads and other managed threads are joined
    even when a test constructs a Controller without calling stop().

    As a safety net, takes a snapshot of threads before the test and warns
    about any non-daemon threads created outside any container that survive
    teardown.
    """
    before = {t.ident for t in threading.enumerate()}
    with thread_container_scope(name=f"test:{request.node.name}"):
        yield

    def _no_leaked_threads() -> bool:
        return not any(
            t.is_alive() and not t.daemon and t.name != "MainThread" and t.ident not in before
            for t in threading.enumerate()
        )

    if ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(_no_leaked_threads, timeout=Duration.from_seconds(5.0)):
        return

    leaked = [
        t
        for t in threading.enumerate()
        if t.is_alive() and not t.daemon and t.name != "MainThread" and t.ident not in before
    ]

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
    """Dump any non-daemon threads still alive at session end.

    Groups threads by stack trace so identical stacks are shown once with all
    thread names listed, rather than repeating the same trace for each thread.

    Registers an atexit handler so the force-exit happens only after pytest has
    finished printing the FAILURES section and test summary.
    """
    alive = [t for t in threading.enumerate() if t.is_alive() and not t.daemon and t.name != "MainThread"]
    if not alive:
        return

    tty = os.fdopen(os.dup(2), "w")
    tty.write(f"\n⚠ {len(alive)} non-daemon threads still alive at session end:\n")
    frames = sys._current_frames()

    # Group threads by stack trace so duplicate stacks are shown only once.
    groups: dict[str, list[str]] = {}
    for t in alive:
        frame = frames.get(t.ident)
        stack_key = "".join(traceback.format_stack(frame)) if frame else "<no stack>"
        groups.setdefault(stack_key, []).append(t.name)

    for stack, names in groups.items():
        tty.write(f"\n  Threads: {', '.join(names)}\n")
        for line in stack.splitlines():
            tty.write(f"    {line}\n")

    tty.flush()
    tty.close()

    if exitstatus != 0:
        # Spawn a daemon thread to force-exit after pytest prints its summary.
        # atexit won't work here: Python joins non-daemon threads before running
        # atexit handlers, so leaked controller threads would deadlock shutdown.
        def _force_exit():
            time.sleep(5)
            os._exit(exitstatus)

        threading.Thread(target=_force_exit, daemon=True).start()
