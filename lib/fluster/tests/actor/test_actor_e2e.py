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

from fluster.actor import ActorClient, ActorServer, FixedResolver
from fluster.client import FlusterContext, fluster_ctx, fluster_ctx_scope


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
        return fluster_ctx().job_id


def test_basic_actor_call():
    """Test basic actor method calls work correctly."""
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    resolver = FixedResolver({"calc": f"http://127.0.0.1:{port}"})
    client = ActorClient(resolver, "calc")
    assert client.add(2, 3) == 5
    assert client.multiply(4, 5) == 20


def test_actor_exception_propagation():
    """Test that exceptions from actor methods propagate to the client."""
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    resolver = FixedResolver({"calc": f"http://127.0.0.1:{port}"})
    client = ActorClient(resolver, "calc")
    with pytest.raises(ZeroDivisionError):
        client.divide(1, 0)


def test_actor_context_injection():
    """Test that FlusterContext is properly injected and accessible."""
    ctx = FlusterContext(
        job_id="test-job-123",
        worker_id="test-worker",
        controller=None,
    )

    # Set up context before starting server (server captures context at serve_background time)
    with fluster_ctx_scope(ctx):
        server = ActorServer(host="127.0.0.1")
        server.register("ctx_actor", ContextAwareActor())
        port = server.serve_background()

    resolver = FixedResolver({"ctx_actor": f"http://127.0.0.1:{port}"})
    client = ActorClient(resolver, "ctx_actor")
    assert client.get_job_id() == "test-job-123"


@pytest.mark.asyncio
async def test_list_actors():
    """Test that list_actors returns registered actors."""
    from fluster.rpc import actor_pb2

    server = ActorServer(host="127.0.0.1")
    actor_id1 = server.register("calc", Calculator())
    actor_id2 = server.register("ctx", ContextAwareActor())
    server.serve_background()

    request = actor_pb2.ListActorsRequest()
    response = await server.list_actors(request, None)

    assert len(response.actors) == 2

    actor_names = {a.name for a in response.actors}
    assert "calc" in actor_names
    assert "ctx" in actor_names

    actor_ids = {a.actor_id for a in response.actors}
    assert actor_id1 in actor_ids
    assert actor_id2 in actor_ids

    for actor in response.actors:
        assert actor.registered_at_ms > 0


@pytest.mark.asyncio
async def test_list_methods():
    """Test that list_methods returns method info for an actor."""
    from fluster.rpc import actor_pb2

    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    server.serve_background()

    request = actor_pb2.ListMethodsRequest(actor_name="calc")
    response = await server.list_methods(request, None)

    method_names = {m.name for m in response.methods}
    assert "add" in method_names
    assert "multiply" in method_names
    assert "divide" in method_names

    for method in response.methods:
        assert method.signature
        assert "(" in method.signature


@pytest.mark.asyncio
async def test_list_methods_with_docstring():
    """Test that list_methods includes docstrings when present."""
    from fluster.rpc import actor_pb2

    class DocumentedActor:
        def documented_method(self) -> str:
            """This method has documentation."""
            return "result"

        def undocumented_method(self) -> str:
            return "result"

    server = ActorServer(host="127.0.0.1")
    server.register("doc", DocumentedActor())
    server.serve_background()

    request = actor_pb2.ListMethodsRequest(actor_name="doc")
    response = await server.list_methods(request, None)

    methods_by_name = {m.name: m for m in response.methods}

    assert "documented_method" in methods_by_name
    assert "This method has documentation" in methods_by_name["documented_method"].docstring

    assert "undocumented_method" in methods_by_name
    assert methods_by_name["undocumented_method"].docstring == ""


@pytest.mark.asyncio
async def test_list_methods_missing_actor():
    """Test that list_methods returns empty response for missing actor."""
    from fluster.rpc import actor_pb2

    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    server.serve_background()

    request = actor_pb2.ListMethodsRequest(actor_name="nonexistent")
    response = await server.list_methods(request, None)

    assert len(response.methods) == 0
