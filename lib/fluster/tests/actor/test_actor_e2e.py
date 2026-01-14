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

"""End-to-end tests for actor server and client."""

import pytest

from fluster.actor.client import ActorClient
from fluster.actor.server import ActorServer
from fluster.actor.types import ActorContext, current_ctx


class Calculator:
    """Test actor with basic arithmetic operations."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def divide(self, a: int, b: int) -> float:
        return a / b  # May raise ZeroDivisionError


class ContextAwareActor:
    """Test actor that accesses the injected context."""

    def get_job_id(self) -> str:
        return current_ctx().job_id


def test_basic_actor_call():
    """Test basic actor method calls work correctly."""
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    client = ActorClient(f"http://127.0.0.1:{port}", "calc")
    assert client.add(2, 3) == 5
    assert client.multiply(4, 5) == 20


def test_actor_exception_propagation():
    """Test that exceptions from actor methods propagate to the client."""
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    client = ActorClient(f"http://127.0.0.1:{port}", "calc")
    with pytest.raises(ZeroDivisionError):
        client.divide(1, 0)


def test_actor_context_injection():
    """Test that ActorContext is properly injected and accessible."""
    server = ActorServer(host="127.0.0.1")
    server.register("ctx_actor", ContextAwareActor())

    ctx = ActorContext(cluster=None, resolver=None, job_id="test-job-123", namespace="<local>")
    port = server.serve_background(context=ctx)

    client = ActorClient(f"http://127.0.0.1:{port}", "ctx_actor")
    assert client.get_job_id() == "test-job-123"
