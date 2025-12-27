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

"""Integration tests for RPC combining tasks, actors, and object store."""

import pytest
from fray.job import SimpleActor
from fray.fray_rpc import RustyContext


@pytest.fixture
def rpc_context(rpc_coordinator):
    """Create RPC context for tests."""
    return RustyContext(coordinator_addr=rpc_coordinator)


def test_actors_and_wait_integration(rpc_context):
    """Test actor method calls with wait."""
    actor = rpc_context.create_actor(SimpleActor, 0)

    futures = [actor.increment.remote(1) for _ in range(10)]
    ready, pending = rpc_context.wait(futures, num_returns=5)

    assert len(ready) == 5
    assert len(pending) == 5

    # Wait for rest
    ready2, pending2 = rpc_context.wait(pending, num_returns=len(pending))
    assert len(ready2) == 5
    assert len(pending2) == 0


def test_mixed_tasks_and_actors(rpc_context):
    """Test mixing regular tasks with actor method calls."""
    actor = rpc_context.create_actor(SimpleActor, 0)

    # Mix task futures and actor method futures
    task_future = rpc_context.run(lambda x: x * 2, 10)
    actor_future = actor.increment.remote(5)

    task_result = rpc_context.get(task_future)
    actor_result = rpc_context.get(actor_future)

    assert task_result == 20
    assert actor_result == 5
