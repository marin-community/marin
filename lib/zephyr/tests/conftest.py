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

"""Pytest fixtures for zephyr tests."""

import logging
import os
import sys
import threading
import uuid
import time
import traceback
import warnings
from pathlib import Path

# Disable Ray's automatic UV runtime env propagation BEFORE importing ray.
# This prevents Ray from packaging the entire working directory (~38MB) for actors.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
os.environ["MARIN_CI_DISABLE_RUNTIME_ENVS"] = "1"

import pytest
import ray
from fray.v2 import ResourceConfig
from fray.v2.iris_backend import FrayIrisClient
from fray.v2.local_backend import LocalClient
from fray.v2.ray_backend.backend import RayClient
from zephyr import load_file
from zephyr.execution import ZephyrContext

# Path to zephyr root (from tests/conftest.py -> tests -> lib/zephyr)
ZEPHYR_ROOT = Path(__file__).resolve().parents[1]

# Use Iris demo config as base
IRIS_CONFIG = Path(__file__).resolve().parents[2] / "iris" / "examples" / "demo.yaml"


@pytest.fixture(scope="session")
def iris_cluster():
    """Start local Iris cluster for testing - reused across all tests."""
    from iris.cluster.vm.cluster_manager import ClusterManager
    from iris.cluster.vm.config import load_config, make_local_config

    try:
        config = load_config(IRIS_CONFIG)
        config = make_local_config(config)
        manager = ClusterManager(config)
        with manager.connect() as url:
            yield url
    except Exception as e:
        pytest.skip(f"Failed to start local Iris cluster: {e}")


@pytest.fixture(scope="session")
def ray_cluster():
    """Initialize Ray cluster for testing - reused across all tests."""
    if not ray.is_initialized():
        logging.info("Initializing Ray cluster for zephyr tests")
        ray.init(
            address="local",
            num_cpus=8,
            ignore_reinit_error=True,
            logging_level="info",
            log_to_driver=True,
            resources={"head_node": 1},
        )
    yield
    # Don't shutdown - Ray will be reused across test sessions


@pytest.fixture(params=["local", "iris", "ray"], scope="module")
def fray_client(request):
    """Parametrized fixture providing Local, Iris, and Ray clients.

    Fixtures are requested lazily to avoid initializing Ray when running
    Iris tests (and vice-versa), since ray.is_initialized() being true
    causes current_client() auto-detection to pick Ray.
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

        # Set up IrisContext so actor handles can resolve
        ctx = IrisContext(job_id=JobName.root("test"), client=iris_client)
        with iris_ctx_scope(ctx):
            yield client
        client.shutdown(wait=True)
    elif request.param == "ray":
        request.getfixturevalue("ray_cluster")
        client = RayClient()
        yield client
        client.shutdown(wait=True)
    else:
        raise ValueError(f"Unknown backend: {request.param}")


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return list(range(1, 11))  # [1, 2, 3, ..., 10]


@pytest.fixture(scope="module")
def zephyr_ctx(fray_client, tmp_path_factory):
    """ZephyrContext running on all backends with temp chunk storage.

    Module-scoped to reuse coordinator/workers across tests in the same file.
    """
    tmp_path = tmp_path_factory.mktemp("zephyr")
    chunk_prefix = str(tmp_path / "chunks")
    with ZephyrContext(
        client=fray_client,
        num_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
        name=f"test-{uuid.uuid4().hex[:8]}",
    ) as ctx:
        yield ctx


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
def _thread_cleanup():
    """Ensure no new non-daemon threads leak from each test.

    Takes a snapshot of threads before the test and checks that no new
    non-daemon threads remain after teardown. Waits briefly for threads
    that are in the process of shutting down.
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
            os._exit(exitstatus)
