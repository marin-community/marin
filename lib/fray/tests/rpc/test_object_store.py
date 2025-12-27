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

"""Tests for RPC object store operations."""

import pytest
from fray.fray_rpc import RustyContext


@pytest.fixture
def rpc_context(rpc_coordinator):
    """Create RPC context for tests."""
    return RustyContext(coordinator_addr=rpc_coordinator)


def test_object_store_put_get(rpc_context):
    """Test basic put and get operations via RPC."""
    data = {"test": "data", "value": 123}
    ref = rpc_context.put(data)
    retrieved = rpc_context.get(ref)
    assert retrieved == data


def test_object_store_multiple_objects(rpc_context):
    """Test storing and retrieving multiple objects."""
    objects = [
        {"id": 1, "value": "first"},
        {"id": 2, "value": "second"},
        {"id": 3, "value": "third"},
    ]

    refs = [rpc_context.put(obj) for obj in objects]
    retrieved = [rpc_context.get(ref) for ref in refs]

    assert retrieved == objects


def test_object_store_large_object(rpc_context):
    """Test storing large objects."""
    # 1MB of data
    large_data = {"data": "x" * 1_000_000}
    ref = rpc_context.put(large_data)
    retrieved = rpc_context.get(ref)
    assert retrieved == large_data
