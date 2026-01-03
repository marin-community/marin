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

"""Tests for RPC task execution."""

import pytest
from fray_rpc import RustyContext


@pytest.fixture
def rpc_context(rpc_coordinator):
    """Create RPC context for tests."""
    return RustyContext(coordinator_addr=rpc_coordinator)


def test_task_execution_simple(rpc_context):
    """Test executing a simple task."""
    future = rpc_context.run(lambda x: x * 2, 21)
    result = rpc_context.get(future)
    assert result == 42


def test_task_execution_with_error(rpc_context):
    """Test task execution with error handling."""

    def failing_task():
        raise ValueError("Expected error")

    # RPC context raises errors synchronously during submission
    with pytest.raises(RuntimeError):
        rpc_context.run(failing_task)


def test_task_execution_multiple_tasks(rpc_context):
    """Test executing multiple tasks concurrently."""
    futures = [rpc_context.run(lambda x: x + 10, i) for i in range(5)]
    results = [rpc_context.get(f) for f in futures]
    assert results == [10, 11, 12, 13, 14]


def test_task_wait_partial(rpc_context):
    """Test waiting for partial task completion."""
    futures = [rpc_context.run(lambda x: x, i) for i in range(5)]
    ready, pending = rpc_context.wait(futures, num_returns=2)

    assert len(ready) == 2
    assert len(pending) == 3

    # Verify ready futures can be retrieved
    results = [rpc_context.get(f) for f in ready]
    assert all(isinstance(r, int) for r in results)
