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

import pytest
import ray
from fray.cluster import LocalCluster, RayCluster


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield RayCluster()
    # Don't shutdown - let pytest handle cleanup


@pytest.fixture(scope="module")
def local_cluster():
    yield LocalCluster()


@pytest.fixture(scope="module", params=["local", "ray"])
def cluster_type(request):
    return request.param


@pytest.fixture
def cluster(cluster_type, local_cluster, ray_cluster):
    if cluster_type == "local":
        return local_cluster
    elif cluster_type == "ray":
        return ray_cluster


@pytest.fixture(params=["sync", "thread", "ray"])
def context_type(request):
    """Parameterized context type fixture.

    This fixture will run tests with SyncContext, ThreadContext, and RayContext.
    """
    if request.param == "ray":
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    return request.param
