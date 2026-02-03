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
from pathlib import Path

import pytest
import ray
from fray.v2 import ResourceConfig
from fray.v2.iris_backend import FrayIrisClient
from fray.v2.local import LocalClient
from fray.v2.ray.backend import RayClient
from zephyr import load_file
from zephyr.execution import ZephyrContext

# Path to zephyr root (from tests/conftest.py -> tests -> lib/zephyr)
ZEPHYR_ROOT = Path(__file__).resolve().parents[1]

# Use Iris demo config as base
IRIS_CONFIG = Path(__file__).resolve().parents[2] / "iris" / "examples" / "demo.yaml"


@pytest.fixture(scope="session")
def iris_cluster():
    """Start local Iris cluster for testing - reused across all tests."""
    from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
    from iris.cluster.vm.config import load_config

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
        import os

        # Disable Ray's automatic UV runtime env propagation which packages the
        # entire working directory (~38MB) for every actor. Not needed for local tests.
        os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
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


@pytest.fixture(params=["local", "iris", "ray"])
def fray_client(request):
    """Parametrized fixture providing Local, Iris, and Ray clients.

    Fixtures are requested lazily to avoid initializing Ray when running
    Iris tests (and vice-versa), since ray.is_initialized() being true
    causes current_client() auto-detection to pick Ray.
    """
    if request.param == "local":
        client = LocalClient()
    elif request.param == "iris":
        iris_cluster = request.getfixturevalue("iris_cluster")
        client = FrayIrisClient(controller_address=iris_cluster, workspace=ZEPHYR_ROOT)
    elif request.param == "ray":
        request.getfixturevalue("ray_cluster")
        client = RayClient()
    else:
        raise ValueError(f"Unknown backend: {request.param}")

    yield client
    client.shutdown(wait=True)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return list(range(1, 11))  # [1, 2, 3, ..., 10]


@pytest.fixture
def zephyr_ctx(fray_client, tmp_path):
    """ZephyrContext running on all backends with temp chunk storage."""
    chunk_prefix = str(tmp_path / "chunks")
    with ZephyrContext(
        client=fray_client,
        num_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=chunk_prefix,
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
