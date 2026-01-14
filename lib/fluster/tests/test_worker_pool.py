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

"""Tests for TaskExecutorActor."""

import cloudpickle
import pytest

from fluster.worker_pool import TaskExecutorActor


def test_execute_basic():
    """Test basic function execution."""
    executor = TaskExecutorActor()

    fn_bytes = cloudpickle.dumps(lambda x, y: x + y)
    args_bytes = cloudpickle.dumps((1, 2))
    kwargs_bytes = cloudpickle.dumps({})

    result = executor.execute(fn_bytes, args_bytes, kwargs_bytes)
    assert result == 3


def test_execute_with_kwargs():
    """Test execution with keyword arguments."""
    executor = TaskExecutorActor()

    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    fn_bytes = cloudpickle.dumps(greet)
    args_bytes = cloudpickle.dumps(("World",))
    kwargs_bytes = cloudpickle.dumps({"greeting": "Hi"})

    result = executor.execute(fn_bytes, args_bytes, kwargs_bytes)
    assert result == "Hi, World!"


def test_execute_returns_complex_object():
    """Test that complex objects can be returned."""
    executor = TaskExecutorActor()

    def create_dict():
        return {"a": [1, 2, 3], "b": {"nested": True}}

    fn_bytes = cloudpickle.dumps(create_dict)
    args_bytes = cloudpickle.dumps(())
    kwargs_bytes = cloudpickle.dumps({})

    result = executor.execute(fn_bytes, args_bytes, kwargs_bytes)
    assert result == {"a": [1, 2, 3], "b": {"nested": True}}


def test_execute_propagates_exception():
    """Test that exceptions are propagated."""
    executor = TaskExecutorActor()

    def raise_error():
        raise ValueError("test error")

    fn_bytes = cloudpickle.dumps(raise_error)
    args_bytes = cloudpickle.dumps(())
    kwargs_bytes = cloudpickle.dumps({})

    with pytest.raises(ValueError, match="test error"):
        executor.execute(fn_bytes, args_bytes, kwargs_bytes)


def test_execute_with_closure():
    """Test execution of closures that capture variables."""
    executor = TaskExecutorActor()

    multiplier = 10

    def multiply(x):
        return x * multiplier

    fn_bytes = cloudpickle.dumps(multiply)
    args_bytes = cloudpickle.dumps((5,))
    kwargs_bytes = cloudpickle.dumps({})

    result = executor.execute(fn_bytes, args_bytes, kwargs_bytes)
    assert result == 50
