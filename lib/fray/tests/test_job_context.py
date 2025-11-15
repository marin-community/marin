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

"""Tests for execution contexts."""

import pytest

from fray import SyncContext, ThreadContext, create_context


def test_context_put_get(execution_context):
    """Test execution context basic put/get operations.

    This test runs for SyncContext, ThreadContext, and RayContext.
    """
    obj = {"key": "value"}
    ref = execution_context.put(obj)
    assert execution_context.get(ref) == obj


def test_context_run(execution_context):
    """Test execution context run operation.

    This test runs for SyncContext, ThreadContext, and RayContext.
    """
    future = execution_context.run(lambda x: x * 2, 5)
    assert execution_context.get(future) == 10


def test_context_wait(execution_context):
    """Test execution context wait operation.

    This test runs for SyncContext, ThreadContext, and RayContext.
    """
    futures = [execution_context.run(lambda x: x, i) for i in range(5)]
    ready, pending = execution_context.wait(futures, num_returns=2)
    assert len(ready) == 2
    assert len(pending) == 3


def test_create_context_sync():
    """Test factory for sync context."""
    ctx = create_context("sync")
    assert isinstance(ctx, SyncContext)


def test_create_context_threadpool():
    """Test factory for threadpool context."""
    ctx = create_context("threadpool", max_workers=4)
    assert isinstance(ctx, ThreadContext)


def test_create_context_invalid():
    """Test factory with invalid type."""
    with pytest.raises(ValueError, match="Unknown context type"):
        create_context("invalid")  # type: ignore
