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

"""Tests for FrayWorker actor instantiation and execution."""

import cloudpickle
import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from uvicorn import Config, Server

from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayWorkerASGIApplication, FrayWorkerClient
from fray.job.rpc.worker import FrayWorkerServicer


@pytest.fixture
def worker_server():
    """Start a test worker server."""
    import threading
    import time
    from unittest.mock import Mock

    controller_client_mock = Mock()
    servicer = FrayWorkerServicer(worker_id="test-worker-1", controller_client=controller_client_mock)
    app = FrayWorkerASGIApplication(servicer)
    config = Config(app=app, host="127.0.0.1", port=0, log_level="error")
    server = Server(config=config)

    # Start server in background
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(0.5)

    # Get actual port
    actual_port = server.servers[0].sockets[0].getsockname()[1]

    yield server, actual_port

    # Cleanup
    server.should_exit = True
    server_thread.join(timeout=2.0)


@pytest.fixture
def worker_url(worker_server):
    """Get worker server URL."""
    _, port = worker_server
    return f"http://localhost:{port}"


class Counter:
    """Test actor class with state."""

    def __init__(self, initial_value: int = 0):
        self.count = initial_value

    def increment(self):
        self.count += 1
        return self.count

    def add(self, value: int):
        self.count += value
        return self.count

    def get_count(self):
        return self.count

    def raise_error(self):
        raise ValueError("Intentional error for testing")


@pytest.mark.asyncio
async def test_instantiate_actor(worker_url):
    """Test basic actor instantiation."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Serialize actor spec
        actor_data = {"cls": Counter, "args": (10,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)

        spec = fray_pb2.ActorSpec(
            actor_id="actor-1",
            serialized_actor=serialized_actor,
            name="counter",
        )

        # Instantiate actor
        handle = await client.instantiate_actor(spec)

        assert handle.actor_id == "actor-1"
        assert handle.worker_id == "test-worker-1"
        assert handle.name == "counter"
        assert handle.status == fray_pb2.ACTOR_STATUS_READY


@pytest.mark.asyncio
async def test_execute_actor_method(worker_url):
    """Test executing methods on actor instances."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Instantiate actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        spec = fray_pb2.ActorSpec(
            actor_id="actor-1",
            serialized_actor=serialized_actor,
            name="",
        )
        await client.instantiate_actor(spec)

        # Call increment method
        call_data = {"method": "increment", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        call_request = fray_pb2.ActorCall(
            actor_id="actor-1",
            serialized_call=serialized_call,
        )

        result = await client.execute_actor_method(call_request)

        assert not result.error
        assert result.serialized_result

        # Deserialize result
        count = cloudpickle.loads(result.serialized_result)
        assert count == 1

        # Call increment again to verify state persistence
        result2 = await client.execute_actor_method(call_request)
        count2 = cloudpickle.loads(result2.serialized_result)
        assert count2 == 2


@pytest.mark.asyncio
async def test_execute_actor_method_with_args(worker_url):
    """Test executing actor methods with arguments."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Instantiate actor
        actor_data = {"cls": Counter, "args": (5,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        spec = fray_pb2.ActorSpec(
            actor_id="actor-1",
            serialized_actor=serialized_actor,
            name="",
        )
        await client.instantiate_actor(spec)

        # Call add method with argument
        call_data = {"method": "add", "args": (3,), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        call_request = fray_pb2.ActorCall(
            actor_id="actor-1",
            serialized_call=serialized_call,
        )

        result = await client.execute_actor_method(call_request)

        count = cloudpickle.loads(result.serialized_result)
        assert count == 8  # 5 + 3


@pytest.mark.asyncio
async def test_execute_actor_method_not_found(worker_url):
    """Test executing method on non-existent actor."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Try to call method on non-existent actor
        call_data = {"method": "increment", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        call_request = fray_pb2.ActorCall(
            actor_id="nonexistent-actor",
            serialized_call=serialized_call,
        )

        with pytest.raises(ConnectError) as exc_info:
            await client.execute_actor_method(call_request)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_execute_actor_method_error(worker_url):
    """Test that actor method errors are properly serialized."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Instantiate actor
        actor_data = {"cls": Counter, "args": (), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        spec = fray_pb2.ActorSpec(
            actor_id="actor-1",
            serialized_actor=serialized_actor,
            name="",
        )
        await client.instantiate_actor(spec)

        # Call method that raises error
        call_data = {"method": "raise_error", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        call_request = fray_pb2.ActorCall(
            actor_id="actor-1",
            serialized_call=serialized_call,
        )

        result = await client.execute_actor_method(call_request)

        # Should have error, not success
        assert result.error
        assert "ValueError" in result.error
        assert "Intentional error for testing" in result.error
        assert result.serialized_error

        # Deserialize error
        error = cloudpickle.loads(result.serialized_error)
        assert isinstance(error, ValueError)


@pytest.mark.asyncio
async def test_destroy_actor(worker_url):
    """Test destroying actor instances."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Instantiate actor
        actor_data = {"cls": Counter, "args": (), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        spec = fray_pb2.ActorSpec(
            actor_id="actor-1",
            serialized_actor=serialized_actor,
            name="",
        )
        await client.instantiate_actor(spec)

        # Verify actor exists
        call_data = {"method": "get_count", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        call_request = fray_pb2.ActorCall(
            actor_id="actor-1",
            serialized_call=serialized_call,
        )
        result = await client.execute_actor_method(call_request)
        assert not result.error

        # Destroy actor
        delete_request = fray_pb2.ActorDeleteRequest(actor_id="actor-1")
        empty_response = await client.destroy_actor(delete_request)
        assert isinstance(empty_response, fray_pb2.Empty)

        # Verify actor no longer exists
        with pytest.raises(ConnectError) as exc_info:
            await client.execute_actor_method(call_request)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_destroy_actor_idempotent(worker_url):
    """Test that destroying non-existent actor is idempotent."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Destroy non-existent actor (should not raise error)
        delete_request = fray_pb2.ActorDeleteRequest(actor_id="nonexistent-actor")
        empty_response = await client.destroy_actor(delete_request)
        assert isinstance(empty_response, fray_pb2.Empty)


@pytest.mark.asyncio
async def test_list_actors(worker_url):
    """Test listing actors on worker."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Initially no actors
        actor_list = await client.list_actors(fray_pb2.Empty())
        assert len(actor_list.actors) == 0

        # Instantiate first actor
        actor_data_1 = {"cls": Counter, "args": (1,), "kwargs": {}}
        serialized_actor_1 = cloudpickle.dumps(actor_data_1)
        spec_1 = fray_pb2.ActorSpec(
            actor_id="actor-1",
            serialized_actor=serialized_actor_1,
            name="",
        )
        await client.instantiate_actor(spec_1)

        # Instantiate second actor
        actor_data_2 = {"cls": Counter, "args": (2,), "kwargs": {}}
        serialized_actor_2 = cloudpickle.dumps(actor_data_2)
        spec_2 = fray_pb2.ActorSpec(
            actor_id="actor-2",
            serialized_actor=serialized_actor_2,
            name="",
        )
        await client.instantiate_actor(spec_2)

        # List actors
        actor_list = await client.list_actors(fray_pb2.Empty())
        assert len(actor_list.actors) == 2

        actor_ids = {actor.actor_id for actor in actor_list.actors}
        assert "actor-1" in actor_ids
        assert "actor-2" in actor_ids

        # All actors should be READY
        for actor in actor_list.actors:
            assert actor.status == fray_pb2.ACTOR_STATUS_READY
            assert actor.worker_id == "test-worker-1"


@pytest.mark.asyncio
async def test_actor_state_isolation(worker_url):
    """Test that different actor instances maintain separate state."""
    async with httpx.AsyncClient() as http_client:
        client = FrayWorkerClient(worker_url, session=http_client)

        # Instantiate two counters with different initial values
        for actor_id, initial_value in [("actor-1", 10), ("actor-2", 20)]:
            actor_data = {"cls": Counter, "args": (initial_value,), "kwargs": {}}
            serialized_actor = cloudpickle.dumps(actor_data)
            spec = fray_pb2.ActorSpec(
                actor_id=actor_id,
                serialized_actor=serialized_actor,
                name="",
            )
            await client.instantiate_actor(spec)

        # Increment actor-1
        call_data = {"method": "increment", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)

        call_request_1 = fray_pb2.ActorCall(
            actor_id="actor-1",
            serialized_call=serialized_call,
        )
        result_1 = await client.execute_actor_method(call_request_1)
        count_1 = cloudpickle.loads(result_1.serialized_result)
        assert count_1 == 11

        # Get count from actor-2 (should be unchanged)
        get_call_data = {"method": "get_count", "args": (), "kwargs": {}}
        serialized_get_call = cloudpickle.dumps(get_call_data)

        call_request_2 = fray_pb2.ActorCall(
            actor_id="actor-2",
            serialized_call=serialized_get_call,
        )
        result_2 = await client.execute_actor_method(call_request_2)
        count_2 = cloudpickle.loads(result_2.serialized_result)
        assert count_2 == 20  # Unchanged
