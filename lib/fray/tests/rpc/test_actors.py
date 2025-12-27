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

"""Tests for RPC actor creation and method calls."""

import pytest
from fray.job import SimpleActor
from fray.fray_rpc import RustyContext


@pytest.fixture
def rpc_context(rpc_coordinator):
    """Create RPC context for tests."""
    return RustyContext(coordinator_addr=rpc_coordinator)


def test_actor_creation(rpc_context):
    """Test basic actor creation."""
    actor = rpc_context.create_actor(SimpleActor, 10)
    result = rpc_context.get(actor.get_value.remote())
    assert result == 10


def test_actor_method_call(rpc_context):
    """Test calling methods on an actor."""
    actor = rpc_context.create_actor(SimpleActor, 0)

    rpc_context.get(actor.increment.remote(5))
    rpc_context.get(actor.increment.remote(3))

    result = rpc_context.get(actor.get_value.remote())
    assert result == 8


def test_actor_state_persistence(rpc_context):
    """Test that actor state persists across method calls."""
    actor = rpc_context.create_actor(SimpleActor, 100)

    # Multiple increments
    for _ in range(10):
        rpc_context.get(actor.increment.remote(1))

    final = rpc_context.get(actor.get_value.remote())
    assert final == 110


def test_actor_named_creation(rpc_context):
    """Test creating named actors."""
    actor = rpc_context.create_actor(SimpleActor, 42, name="test_named_actor", get_if_exists=False)
    result = rpc_context.get(actor.get_value.remote())
    assert result == 42


def test_actor_named_duplicate_error(rpc_context):
    """Test that creating duplicate named actors fails."""
    rpc_context.create_actor(SimpleActor, 10, name="duplicate_actor", get_if_exists=False)

    with pytest.raises(ValueError):
        rpc_context.create_actor(SimpleActor, 20, name="duplicate_actor", get_if_exists=False)


def test_actor_named_get_if_exists(rpc_context):
    """Test get_if_exists flag for named actors."""
    actor1 = rpc_context.create_actor(SimpleActor, 100, name="shared_actor", get_if_exists=True)
    rpc_context.get(actor1.increment.remote(10))

    actor2 = rpc_context.create_actor(SimpleActor, 999, name="shared_actor", get_if_exists=True)
    result = rpc_context.get(actor2.get_value.remote())

    # Should get the existing actor's value
    assert result == 110


def test_multiple_actors_isolation(rpc_context):
    """Test that multiple unnamed actors are isolated."""
    actor1 = rpc_context.create_actor(SimpleActor, 10)
    actor2 = rpc_context.create_actor(SimpleActor, 20)

    rpc_context.get(actor1.increment.remote(5))
    rpc_context.get(actor2.increment.remote(3))

    val1 = rpc_context.get(actor1.get_value.remote())
    val2 = rpc_context.get(actor2.get_value.remote())

    assert val1 == 15
    assert val2 == 23
