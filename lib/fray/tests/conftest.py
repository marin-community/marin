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

import pytest
import ray
from fray.cluster.local_cluster import LocalCluster, LocalClusterConfig


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
    yield LocalCluster(LocalClusterConfig(use_isolated_env=False))


@pytest.fixture(scope="module", params=["local", "ray"])
def cluster(request, local_cluster, ray_cluster):
    if request.param == "local":
        return local_cluster
    elif request.param == "ray":
        return ray_cluster
