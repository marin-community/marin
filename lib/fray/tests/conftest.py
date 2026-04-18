# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for fray tests."""


import pytest
from fray.v1.cluster.local_cluster import LocalCluster, LocalClusterConfig


@pytest.fixture(scope="module")
def local_cluster():
    yield LocalCluster(LocalClusterConfig(use_isolated_env=False))


@pytest.fixture(scope="module")
def cluster(local_cluster):
    return local_cluster
