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

"""Tests for ActorPool round-robin and broadcast functionality."""

from fluster.actor.pool import ActorPool
from fluster.actor.resolver import FixedResolver
from fluster.actor.server import ActorServer


class Counter:
    """Test actor with stateful counter."""

    def __init__(self, start: int = 0):
        self._value = start

    def get(self) -> int:
        """Get current counter value."""
        return self._value

    def increment(self) -> int:
        """Increment and return new value."""
        self._value += 1
        return self._value


def test_pool_round_robin():
    """Test that pool.call() cycles through endpoints in round-robin fashion."""
    servers = []
    urls = []

    # Create 3 servers with different starting counters
    for i in range(3):
        server = ActorServer(host="127.0.0.1")
        server.register("counter", Counter(start=i * 100))
        port = server.serve_background()
        servers.append(server)
        urls.append(f"http://127.0.0.1:{port}")

    resolver = FixedResolver({"counter": urls})
    pool = ActorPool(resolver, "counter")

    assert pool.size == 3

    # Round-robin should cycle through servers
    results = [pool.call().get() for _ in range(6)]
    # Should see values from all three servers (0, 100, 200, 0, 100, 200)
    assert set(results) == {0, 100, 200}


def test_pool_broadcast():
    """Test that pool.broadcast() sends to all endpoints."""
    servers = []
    urls = []

    # Create 3 servers with different starting counters
    for i in range(3):
        server = ActorServer(host="127.0.0.1")
        server.register("counter", Counter(start=i))
        port = server.serve_background()
        servers.append(server)
        urls.append(f"http://127.0.0.1:{port}")

    resolver = FixedResolver({"counter": urls})
    pool = ActorPool(resolver, "counter")

    # Broadcast get() to all endpoints
    broadcast = pool.broadcast().get()
    results = broadcast.wait_all()

    assert len(results) == 3
    assert all(r.success for r in results)
    assert {r.value for r in results} == {0, 1, 2}
