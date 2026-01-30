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

from fray.v2.local import LocalClient
from zephyr import load_file
from zephyr.execution import ZephyrContext


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return list(range(1, 11))  # [1, 2, 3, ..., 10]


@pytest.fixture
def zephyr_ctx():
    """ZephyrContext fixture using LocalClient for testing."""
    client = LocalClient()
    ctx = ZephyrContext(client=client, num_workers=2)
    yield ctx
    ctx.shutdown()


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
