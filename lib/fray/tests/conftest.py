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

"""Pytest fixtures for fray tests."""


import logging
import socket
import time

import pytest
import ray
from fray.cluster.local_cluster import LocalCluster, LocalClusterConfig

# Configure logging for RPC tests
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Enable DEBUG logging for RPC during tests
logging.getLogger("fray.rpc").setLevel(logging.DEBUG)


@pytest.fixture(scope="module")
def ray_cluster():
    from fray.cluster.ray import RayCluster

    if not ray.is_initialized():
        logging.info("Initializing Ray cluster")
        ray.init(
            address="local",
            num_cpus=8,
            ignore_reinit_error=True,
            logging_level="info",
            log_to_driver=True,
            resources={"head_node": 1},
        )
    yield RayCluster()


@pytest.fixture(scope="module")
def local_cluster():
    yield LocalCluster(LocalClusterConfig(use_isolated_env=True))


@pytest.fixture(scope="module", params=["local", "ray"])
def cluster(request, local_cluster, ray_cluster):
    if request.param == "local":
        return local_cluster
    elif request.param == "ray":
        return ray_cluster


@pytest.fixture(scope="module")
def rpc_coordinator():
    """Start RPC coordinator server for testing."""
    # Find free port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    addr = f"127.0.0.1:{port}"

    # Start coordinator using the Rust function
    from fray.fray_rpc import start_coordinator_server

    start_coordinator_server(addr)

    # Wait a bit more for server to be fully ready
    time.sleep(1)

    yield addr

    # Cleanup happens automatically with daemon thread
