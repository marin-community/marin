# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for zephyr tests."""
import tempfile

import atexit
import os
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

from rigging.timing import ExponentialBackoff

import pytest
from fray.v2 import ResourceConfig
from fray.v2.iris_backend import FrayIrisClient
from fray.v2.local_backend import LocalClient
from zephyr import load_file
from zephyr.execution import ZephyrContext

# Path to zephyr root (from tests/conftest.py -> tests -> lib/zephyr)
ZEPHYR_ROOT = Path(__file__).resolve().parents[1]

# Use Iris demo config as base
IRIS_CONFIG = Path(__file__).resolve().parents[2] / "iris" / "examples" / "test.yaml"


@pytest.fixture(scope="session")
def iris_cluster():
    """Start local Iris cluster for testing - reused across all tests."""
    from iris.cluster.config import load_config, make_local_config, connect_cluster

    config = load_config(IRIS_CONFIG)
    config = make_local_config(config)
    with connect_cluster(config) as url:
        yield url


# --- Local-only fixtures (functional tests) ---


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


# --- Multi-backend fixtures (integration tests) ---


@pytest.fixture(params=["local", "iris"], scope="session")
def integration_client(request):
    """Parametrized fixture providing Local and Iris clients.

    Session-scoped to reuse clusters across all test modules.
    """
    if request.param == "local":
        client = LocalClient()
        yield client
        client.shutdown(wait=True)
    elif request.param == "iris":
        from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
        from iris.cluster.types import JobName

        iris_cluster = request.getfixturevalue("iris_cluster")
        iris_client = IrisClient.remote(iris_cluster, workspace=ZEPHYR_ROOT)
        client = FrayIrisClient.from_iris_client(iris_client)

        ctx = IrisContext(job_id=JobName.root("test-user", "test"), client=iris_client)
        with iris_ctx_scope(ctx):
            yield client
        client.shutdown(wait=True)
    else:
        raise ValueError(f"Unknown backend: {request.param}")


@pytest.fixture(scope="session")
def integration_ctx(integration_client, tmp_path_factory):
    """ZephyrContext on all backends for integration tests."""
    tmp_path = tmp_path_factory.mktemp("zephyr-integration")
    ctx = ZephyrContext(
        client=integration_client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="test-integration",
    )
    yield ctx
    ctx.shutdown()


@pytest.fixture
def actor_context():
    """Provide a fake actor context so ZephyrCoordinator can call current_actor()."""
    from unittest.mock import MagicMock

    from fray.v2.actor import ActorContext, _reset_current_actor, _set_current_actor

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


# Thread name prefixes for infrastructure threads managed by session-scoped
# clusters (iris, fray). These persist across tests and are not leaks.
_INFRA_THREAD_PREFIXES = (
    "worker-server",
    "worker-lifecycle",
    "AnyIO worker thread",
    "ThreadPoolExecutor",
    "asyncio_",
    "grpc_",
    "monitoring",
)


@pytest.fixture(autouse=True)
def _thread_cleanup():
    """Ensure no new non-daemon threads leak from each test.

    Takes a snapshot of threads before the test and checks that no new
    non-daemon threads remain after teardown. Waits briefly for threads
    that are in the process of shutting down.

    Infrastructure threads from session-scoped clusters (iris) are
    excluded — they persist for the session and are not leaks.
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
