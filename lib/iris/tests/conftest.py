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
from finelog.client import LogClient
from finelog.embedded import is_available as finelog_native_available
from finelog.embedded import require_embedded_server
from iris.client.local_client import make_local_client
from iris.cluster.config import (
    AuthConfig,
    IrisClusterConfig,
    LocalSliceConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    SliceConfig,
    StaticAuthConfig,
    load_config,
    make_local_config,
)
from iris.cluster.types import AcceleratorType, CapacityType
from iris.managed_thread import thread_container_scope
from iris.test_util import SentinelFile
from rigging.timing import Duration, ExponentialBackoff

IRIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = IRIS_ROOT / "config" / "ci-test.yaml"


@pytest.fixture
def embedded_log_server(tmp_path):
    """A fresh in-process native finelog server for tests that exercise logs/stats.

    Boots the same engine the ``finelog-server`` binary runs over a per-test
    on-disk ``log_dir``. (In-memory mode spawns no maintenance task, so its RAM
    buffer never flushes to a readable segment — written logs would never be
    queryable; a disk-backed store serves reads.) Function-scoped so every test
    gets an isolated store. Tests talk to it over the normal RPC contract via
    ``finelog.client.LogClient`` or the generated ``LogServiceClientSync``
    against ``embedded_log_server.address``. Skips when the native extension is
    unavailable (e.g. a pure-Python install).
    """
    if not finelog_native_available():
        pytest.skip("finelog native server extension (finelog_server) not available")
    server = require_embedded_server()(log_dir=str(tmp_path / "log-server"))
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def log_client(embedded_log_server):
    """A ``finelog.client.LogClient`` connected to the per-test embedded server."""
    client = LogClient.connect(embedded_log_server.address)
    try:
        yield client
    finally:
        client.close()


def _make_controller_only_config() -> IrisClusterConfig:
    """Build a local config with no auto-scaled workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups = {
        "placeholder": ScaleGroupConfig(
            name="placeholder",
            num_vms=1,
            buffer_slices=0,
            max_slices=0,
            resources=ScaleGroupResources(
                cpu_millicores=1000,
                memory_bytes=1 * 1024**3,
                disk_bytes=10 * 1024**3,
                device_type=AcceleratorType.CPU,
                capacity_type=CapacityType.ON_DEMAND,
            ),
            slice_template=SliceConfig(local=LocalSliceConfig()),
        )
    }
    # Provide an empty static-auth block so auth tests can populate tokens
    # (config.auth.static.tokens[...]) the way the proto config auto-vivified.
    config.auth = AuthConfig(static=StaticAuthConfig())
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


@pytest.fixture
def local_iris_client():
    """Boot an in-process LocalCluster and yield an IrisClient connected to it.

    Cluster is torn down on teardown even if the test raises. For module-scoped
    reuse, override this fixture in your test file with ``scope="module"`` and
    the same body — ``make_local_client`` does not depend on per-test state.
    """

    client = make_local_client()
    try:
        yield client
    finally:
        client.shutdown()


@pytest.fixture(autouse=True)
def _thread_cleanup(request):
    """Isolate each test's managed threads and warn on leaks.

    Installs a fresh ThreadContainer via thread_container_scope() so every
    component that calls get_thread_container() registers its threads into a
    per-test container that is stopped on teardown. This ensures log-server
    uvicorn threads and other managed threads are joined even when a test
    constructs a Controller without calling stop().

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
