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

import pytest
import ray

from zephyr import create_backend, load_file


@pytest.fixture(scope="module")
def ray_cluster():
    """Start Ray cluster for tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    # Don't shutdown - let pytest handle cleanup


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return list(range(1, 11))  # [1, 2, 3, ..., 10]


@pytest.fixture(
    params=[
        pytest.param(create_backend("sync"), id="sync"),
        pytest.param(create_backend("threadpool", max_parallelism=2), id="thread"),
        pytest.param(create_backend("ray", max_parallelism=2), id="ray"),
    ]
)
def backend(request):
    """Parametrized fixture providing all backend types."""
    return request.param


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
