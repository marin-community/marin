# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for zephyr tests."""
import atexit
import os
import sys
import tempfile
import threading
import time
import traceback
import warnings
from unittest.mock import MagicMock

import pytest
from fray.actor import ActorContext, _reset_current_actor, _set_current_actor
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from rigging.timing import ExponentialBackoff
from zephyr.execution import ZephyrContext
from zephyr.readers import load_file
from zephyr.testing.coordinator import make_test_coordinator


@pytest.fixture(scope="session")
def local_client():
    client = LocalClient()
    yield client
    client.shutdown(wait=True)


@pytest.fixture(scope="session")
def zephyr_ctx(local_client, tmp_path_factory):
    """Local-only ZephyrContext for functional tests."""
    tmp_path = tmp_path_factory.mktemp("zephyr")
    ctx = ZephyrContext(
        client=local_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="test-ctx",
    )
    yield ctx
    ctx.shutdown()


@pytest.fixture
def coordinator(actor_context, tmp_path):
    coord = make_test_coordinator(tmp_path)
    yield coord
    coord.shutdown()


@pytest.fixture
def actor_context():
    """Provide a fake actor context so ZephyrCoordinator can call current_actor()."""

    token = _set_current_actor(ActorContext(handle=MagicMock(), index=0, group_name="test-coord"))
    yield
    _reset_current_actor(token)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return list(range(1, 11))  # [1, 2, 3, ..., 10]


class CallCounter:
    """Helper to track function calls across test scenarios."""

    def __init__(self):
        self.flat_map_count = 0
        self.map_count = 0
        self.processed_ids = []

    def reset(self):
        self.flat_map_count = 0
        self.map_count = 0
        self.processed_ids = []

    def counting_flat_map(self, path):
        self.flat_map_count += 1
        return load_file(path)

    def counting_map(self, x):
        self.map_count += 1
        self.processed_ids.append(x["id"])
        return {**x, "processed": True}


@pytest.fixture(autouse=True)
def _configure_marin_prefix():
    """Set MARIN_PREFIX to a temp directory for tests that rely on it."""
    if "MARIN_PREFIX" in os.environ:
        yield
        return

    with tempfile.TemporaryDirectory(prefix="marin_prefix") as temp_dir:
        os.environ["MARIN_PREFIX"] = temp_dir
        yield
        del os.environ["MARIN_PREFIX"]


# Thread name prefixes for infrastructure threads managed by long-lived
# fixtures (iris, fray). These persist across tests within a module/session
# and are not leaks.
_INFRA_THREAD_PREFIXES = (
    "worker-server",
    "worker-lifecycle",
    "AnyIO worker thread",
    "ThreadPoolExecutor",
    "asyncio_",
    "grpc_",
    "monitoring",
    # Iris worker task/log threads, spawned the first time a test touches the
    # cluster fixture and torn down with the cluster.
    "task-/",
    "logs-/",
    # Iris worker profiling thread and controller RPC thread pool — created
    # lazily when the cluster handles its first job but torn down with the
    # module-scoped iris_cluster fixture, not with individual tests.
    "profile-loop",
    "rpc-handler",
)


@pytest.fixture(autouse=True)
def _thread_cleanup():
    """Ensure no new non-daemon threads leak from each test.

    Takes a snapshot of threads before the test and checks that no new
    non-daemon threads remain after teardown. Waits briefly for threads
    that are in the process of shutting down.

    Infrastructure threads from long-lived fixtures (iris cluster, fray) are
    excluded — they persist for the lifetime of the fixture and are not leaks.
    """
    before = {t.ident for t in threading.enumerate()}
    yield

    def _is_leaked(t: threading.Thread) -> bool:
        if not t.is_alive() or t.daemon or t.name == "MainThread":
            return False
        if t.ident in before:
            return False
        if any(t.name.startswith(prefix) for prefix in _INFRA_THREAD_PREFIXES):
            return False
        return True

    backoff = ExponentialBackoff(initial=0.1, maximum=1.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        leaked = [t for t in threading.enumerate() if _is_leaked(t)]
        if not leaked:
            return
        time.sleep(backoff.next_interval())

    thread_info = [f"{t.name} (daemon={t.daemon}, ident={t.ident})" for t in leaked]
    warnings.warn(
        f"Threads leaked from test: {thread_info}\n" "All threads should be stopped via shutdown() or similar cleanup.",
        stacklevel=1,
    )


def pytest_sessionfinish(session, exitstatus):
    """Dump any non-daemon threads still alive at session end."""
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
        if exitstatus != 0:
            atexit.register(os._exit, exitstatus)
