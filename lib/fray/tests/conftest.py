# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for fray tests."""


import logging

import pytest
import ray
from fray.v1.cluster.local_cluster import LocalCluster, LocalClusterConfig


@pytest.fixture(scope="module")
def ray_cluster():
    from fray.v1.cluster.ray import RayCluster

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
