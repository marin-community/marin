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

"""Tests for FrayController Connect RPC service."""

import cloudpickle
import threading
import time

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from uvicorn import Config, Server

from fray.job.rpc.controller import FrayControllerServer
from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayControllerClient, FrayWorkerASGIApplication
from fray.job.rpc.worker import FrayWorkerServicer


class Counter:
    """Test actor class with state."""

    def __init__(self, initial_value: int = 0):
        self.count = initial_value

    def increment(self):
        self.count += 1
        return self.count

    def get_count(self):
        return self.count


@pytest.fixture
def server():
    """Start a test controller server."""
    server = FrayControllerServer(port=0)
    port = server.start()
    yield server, port
    server.stop()


@pytest.fixture
def client_url(server):
    """Get the server URL for tests."""
    _, port = server
    return f"http://localhost:{port}"


@pytest.fixture
def worker_server():
    """Start a test worker server."""
    servicer = FrayWorkerServicer(worker_id="worker-1")
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
def two_worker_servers():
    """Start two test worker servers."""
    servers = []
    threads = []
    ports = []

    for worker_id in ["worker-1", "worker-2"]:
        servicer = FrayWorkerServicer(worker_id=worker_id)
        app = FrayWorkerASGIApplication(servicer)
        config = Config(app=app, host="127.0.0.1", port=0, log_level="error")
        server = Server(config=config)

        # Start server in background
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        servers.append(server)
        threads.append(server_thread)

    # Wait for servers to start
    time.sleep(0.5)

    # Get actual ports
    for server in servers:
        actual_port = server.servers[0].sockets[0].getsockname()[1]
        ports.append(actual_port)

    yield servers, ports

    # Cleanup
    for server, thread in zip(servers, threads, strict=False):
        server.should_exit = True
        thread.join(timeout=2.0)


@pytest.mark.asyncio
async def test_submit_and_get_task_status(client_url):
    """Test submitting a task and retrieving its status."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Submit a task
        task_spec = fray_pb2.TaskSpec(
            serialized_fn=b"test_function_data",
            resources={"cpu": 1},
            max_retries=3,
        )
        handle = await client.submit_task(task_spec)

        assert handle.task_id
        assert handle.status == fray_pb2.TASK_STATUS_PENDING

        # Get status
        status_handle = await client.get_task_status(handle)
        assert status_handle.task_id == handle.task_id
        assert status_handle.status == fray_pb2.TASK_STATUS_PENDING


@pytest.mark.asyncio
async def test_worker_registration(client_url):
    """Test worker registration."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        worker_info = fray_pb2.WorkerInfo(
            worker_id="worker-1",
            address="localhost:9999",
            num_cpus=4,
            memory_bytes=8 * 1024 * 1024 * 1024,
        )

        result = await client.register_worker(worker_info)
        assert isinstance(result, fray_pb2.Empty)


@pytest.mark.asyncio
async def test_get_next_task(client_url):
    """Test worker getting next task from queue."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Submit a task first
        task_spec = fray_pb2.TaskSpec(
            serialized_fn=b"test_function_data",
            resources={"cpu": 2},
            max_retries=1,
        )
        handle = await client.submit_task(task_spec)

        # Get next task as a worker
        request = fray_pb2.GetTaskRequest(worker_id="worker-1")
        next_task = await client.get_next_task(request)

        assert next_task.task_id == handle.task_id
        assert next_task.serialized_fn == b"test_function_data"
        assert next_task.resources == {"cpu": 2}
        assert next_task.max_retries == 1

        # Verify task status is now RUNNING
        status_handle = await client.get_task_status(handle)
        assert status_handle.status == fray_pb2.TASK_STATUS_RUNNING
        assert status_handle.worker_id == "worker-1"


@pytest.mark.asyncio
async def test_get_next_task_timeout(client_url):
    """Test that get_next_task times out when no tasks available."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        request = fray_pb2.GetTaskRequest(worker_id="worker-1")

        # Should timeout since no tasks are pending
        with pytest.raises(ConnectError) as exc_info:
            await client.get_next_task(request)

        assert exc_info.value.code == Code.DEADLINE_EXCEEDED


@pytest.mark.asyncio
async def test_report_task_complete(client_url):
    """Test reporting task completion."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Submit and get a task
        task_spec = fray_pb2.TaskSpec(serialized_fn=b"test_fn")
        handle = await client.submit_task(task_spec)
        request = fray_pb2.GetTaskRequest(worker_id="worker-1")
        await client.get_next_task(request)

        # Report completion
        result = fray_pb2.TaskResult(
            task_id=handle.task_id,
            serialized_result=b"result_data",
            error="",
        )
        await client.report_task_complete(result)

        # Verify status
        status = await client.get_task_status(handle)
        assert status.status == fray_pb2.TASK_STATUS_COMPLETED

        # Get result
        task_result = await client.get_task_result(handle)
        assert task_result.serialized_result == b"result_data"


@pytest.mark.asyncio
async def test_report_task_failed(client_url):
    """Test reporting task failure."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Submit and get a task
        task_spec = fray_pb2.TaskSpec(serialized_fn=b"test_fn")
        handle = await client.submit_task(task_spec)
        request = fray_pb2.GetTaskRequest(worker_id="worker-1")
        await client.get_next_task(request)

        # Report failure
        result = fray_pb2.TaskResult(
            task_id=handle.task_id,
            serialized_result=b"",
            error="Task failed: division by zero",
        )
        await client.report_task_failed(result)

        # Verify status
        status = await client.get_task_status(handle)
        assert status.status == fray_pb2.TASK_STATUS_FAILED
        assert status.error == "Task failed: division by zero"


@pytest.mark.asyncio
async def test_unregister_worker(client_url):
    """Test worker unregistration."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register worker
        worker_info = fray_pb2.WorkerInfo(
            worker_id="worker-1",
            address="localhost:9999",
        )
        await client.register_worker(worker_info)

        # Unregister worker
        result = await client.unregister_worker(worker_info)
        assert isinstance(result, fray_pb2.Empty)


@pytest.mark.asyncio
async def test_multiple_tasks_fifo_order(client_url):
    """Test that tasks are retrieved in FIFO order."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Submit multiple tasks
        task_ids = []
        for i in range(3):
            task_spec = fray_pb2.TaskSpec(serialized_fn=f"task_{i}".encode())
            handle = await client.submit_task(task_spec)
            task_ids.append(handle.task_id)

        # Get tasks in order
        request = fray_pb2.GetTaskRequest(worker_id="worker-1")
        for expected_id in task_ids:
            task = await client.get_next_task(request)
            assert task.task_id == expected_id


@pytest.mark.asyncio
async def test_get_status_nonexistent_task(client_url):
    """Test getting status of a task that doesn't exist."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Try to get status of non-existent task
        handle = fray_pb2.TaskHandle(task_id="nonexistent-task")

        with pytest.raises(ConnectError) as exc_info:
            await client.get_task_status(handle)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_get_result_nonexistent_task(client_url):
    """Test getting result of a task that doesn't exist."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Try to get result of non-existent task
        handle = fray_pb2.TaskHandle(task_id="nonexistent-task")

        with pytest.raises(ConnectError) as exc_info:
            await client.get_task_result(handle)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_report_complete_nonexistent_task(client_url):
    """Test reporting completion for a task that doesn't exist."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Try to report completion for non-existent task
        result = fray_pb2.TaskResult(task_id="nonexistent-task", serialized_result=b"result")

        with pytest.raises(ConnectError) as exc_info:
            await client.report_task_complete(result)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_report_failed_nonexistent_task(client_url):
    """Test reporting failure for a task that doesn't exist."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Try to report failure for non-existent task
        result = fray_pb2.TaskResult(task_id="nonexistent-task", error="some error")

        with pytest.raises(ConnectError) as exc_info:
            await client.report_task_failed(result)

        assert exc_info.value.code == Code.NOT_FOUND


# Actor tests


@pytest.mark.asyncio
async def test_create_actor_basic(client_url, worker_server):
    """Test creating an actor with basic parameters."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker first
        worker_info = fray_pb2.WorkerInfo(
            worker_id="worker-1",
            address=f"localhost:{worker_port}",
            num_cpus=4,
            memory_bytes=8 * 1024 * 1024 * 1024,
        )
        await client.register_worker(worker_info)

        # Create an actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor,
            name="",
            get_if_exists=False,
        )
        handle = await client.create_actor(actor_spec)

        assert handle.actor_id
        assert handle.worker_id == "worker-1"
        assert handle.status == fray_pb2.ACTOR_STATUS_READY


@pytest.mark.asyncio
async def test_create_actor_no_workers(client_url):
    """Test that creating an actor fails when no workers are available."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Try to create an actor without any workers
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor,
            name="",
            get_if_exists=False,
        )

        with pytest.raises(ConnectError) as exc_info:
            await client.create_actor(actor_spec)

        assert exc_info.value.code == Code.UNAVAILABLE


@pytest.mark.asyncio
async def test_create_named_actor(client_url, worker_server):
    """Test creating a named actor."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker
        worker_info = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{worker_port}")
        await client.register_worker(worker_info)

        # Create a named actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor,
            name="my_counter",
            get_if_exists=False,
        )
        handle = await client.create_actor(actor_spec)

        assert handle.actor_id
        assert handle.name == "my_counter"
        assert handle.status == fray_pb2.ACTOR_STATUS_READY


@pytest.mark.asyncio
async def test_create_named_actor_duplicate(client_url, worker_server):
    """Test that creating a duplicate named actor fails."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker
        worker_info = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{worker_port}")
        await client.register_worker(worker_info)

        # Create a named actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor,
            name="my_counter",
            get_if_exists=False,
        )
        await client.create_actor(actor_spec)

        # Try to create another actor with the same name
        with pytest.raises(ConnectError) as exc_info:
            await client.create_actor(actor_spec)

        assert exc_info.value.code == Code.ALREADY_EXISTS


@pytest.mark.asyncio
async def test_create_named_actor_get_if_exists(client_url, worker_server):
    """Test that get_if_exists returns existing actor."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker
        worker_info = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{worker_port}")
        await client.register_worker(worker_info)

        # Create a named actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor,
            name="my_counter",
            get_if_exists=False,
        )
        handle1 = await client.create_actor(actor_spec)

        # Create with get_if_exists=True should return existing actor
        actor_data2 = {"cls": Counter, "args": (10,), "kwargs": {}}
        serialized_actor2 = cloudpickle.dumps(actor_data2)
        actor_spec2 = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor2,
            name="my_counter",
            get_if_exists=True,
        )
        handle2 = await client.create_actor(actor_spec2)

        assert handle1.actor_id == handle2.actor_id
        assert handle2.name == "my_counter"


@pytest.mark.asyncio
async def test_actor_placement_least_loaded(client_url, two_worker_servers):
    """Test that actors are placed on least-loaded workers."""
    _, ports = two_worker_servers

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register two workers
        worker1 = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{ports[0]}")
        worker2 = fray_pb2.WorkerInfo(worker_id="worker-2", address=f"localhost:{ports[1]}")
        await client.register_worker(worker1)
        await client.register_worker(worker2)

        # Create first actor - should go to worker-1
        actor_data1 = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor1 = cloudpickle.dumps(actor_data1)
        actor_spec1 = fray_pb2.ActorSpec(serialized_actor=serialized_actor1)
        handle1 = await client.create_actor(actor_spec1)
        assert handle1.worker_id in ["worker-1", "worker-2"]

        # Create second actor - should go to the other worker
        actor_data2 = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor2 = cloudpickle.dumps(actor_data2)
        actor_spec2 = fray_pb2.ActorSpec(serialized_actor=serialized_actor2)
        handle2 = await client.create_actor(actor_spec2)
        assert handle2.worker_id in ["worker-1", "worker-2"]

        # Create third actor - should go to whichever has fewer (should balance)
        actor_data3 = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor3 = cloudpickle.dumps(actor_data3)
        actor_spec3 = fray_pb2.ActorSpec(serialized_actor=serialized_actor3)
        handle3 = await client.create_actor(actor_spec3)
        assert handle3.worker_id in ["worker-1", "worker-2"]


@pytest.mark.asyncio
async def test_get_actor_status(client_url, worker_server):
    """Test getting actor status."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker
        worker_info = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{worker_port}")
        await client.register_worker(worker_info)

        # Create an actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(serialized_actor=serialized_actor, name="test_actor")
        handle = await client.create_actor(actor_spec)

        # Get status
        status = await client.get_actor_status(handle)
        assert status.actor_id == handle.actor_id
        assert status.worker_id == "worker-1"
        assert status.name == "test_actor"
        assert status.status == fray_pb2.ACTOR_STATUS_READY


@pytest.mark.asyncio
async def test_get_actor_status_nonexistent(client_url):
    """Test getting status of a nonexistent actor."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Try to get status of nonexistent actor
        handle = fray_pb2.ActorHandle(actor_id="nonexistent-actor")

        with pytest.raises(ConnectError) as exc_info:
            await client.get_actor_status(handle)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_delete_actor(client_url, worker_server):
    """Test deleting an actor."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker
        worker_info = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{worker_port}")
        await client.register_worker(worker_info)

        # Create an actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(serialized_actor=serialized_actor)
        handle = await client.create_actor(actor_spec)

        # Delete the actor
        delete_req = fray_pb2.ActorDeleteRequest(actor_id=handle.actor_id)
        result = await client.delete_actor(delete_req)
        assert isinstance(result, fray_pb2.Empty)

        # Verify actor is gone
        with pytest.raises(ConnectError) as exc_info:
            await client.get_actor_status(handle)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_delete_actor_nonexistent(client_url):
    """Test deleting a nonexistent actor."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Try to delete nonexistent actor
        delete_req = fray_pb2.ActorDeleteRequest(actor_id="nonexistent-actor")

        with pytest.raises(ConnectError) as exc_info:
            await client.delete_actor(delete_req)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_delete_named_actor(client_url, worker_server):
    """Test deleting a named actor removes it from the name registry."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker
        worker_info = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{worker_port}")
        await client.register_worker(worker_info)

        # Create a named actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor,
            name="my_actor",
            get_if_exists=False,
        )
        handle = await client.create_actor(actor_spec)

        # Delete the actor
        delete_req = fray_pb2.ActorDeleteRequest(actor_id=handle.actor_id)
        await client.delete_actor(delete_req)

        # Should be able to create new actor with same name
        handle2 = await client.create_actor(actor_spec)
        assert handle2.actor_id != handle.actor_id
        assert handle2.name == "my_actor"


@pytest.mark.asyncio
async def test_call_actor(client_url, worker_server):
    """Test calling an actor method."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker
        worker_info = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{worker_port}")
        await client.register_worker(worker_info)

        # Create an actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(serialized_actor=serialized_actor)
        actor_handle = await client.create_actor(actor_spec)

        # Call actor method
        call_data = {"method": "increment", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        actor_call = fray_pb2.ActorCall(
            actor_id=actor_handle.actor_id,
            serialized_call=serialized_call,
        )
        task_handle = await client.call_actor(actor_call)

        assert task_handle.task_id
        assert task_handle.status == fray_pb2.TASK_STATUS_PENDING


@pytest.mark.asyncio
async def test_call_actor_nonexistent(client_url):
    """Test calling a nonexistent actor."""
    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Try to call nonexistent actor
        call_data = {"method": "increment", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        actor_call = fray_pb2.ActorCall(
            actor_id="nonexistent-actor",
            serialized_call=serialized_call,
        )

        with pytest.raises(ConnectError) as exc_info:
            await client.call_actor(actor_call)

        assert exc_info.value.code == Code.NOT_FOUND


@pytest.mark.asyncio
async def test_call_actor_worker_unavailable(client_url, worker_server):
    """Test calling an actor whose worker is unavailable."""
    _, worker_port = worker_server

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(client_url, session=http_client)

        # Register a worker
        worker_info = fray_pb2.WorkerInfo(worker_id="worker-1", address=f"localhost:{worker_port}")
        await client.register_worker(worker_info)

        # Create an actor
        actor_data = {"cls": Counter, "args": (0,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(serialized_actor=serialized_actor)
        actor_handle = await client.create_actor(actor_spec)

        # Unregister the worker (simulating worker failure)
        await client.unregister_worker(worker_info)

        # Try to call actor - should fail with unavailable
        call_data = {"method": "increment", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        actor_call = fray_pb2.ActorCall(
            actor_id=actor_handle.actor_id,
            serialized_call=serialized_call,
        )

        with pytest.raises(ConnectError) as exc_info:
            await client.call_actor(actor_call)

        assert exc_info.value.code == Code.UNAVAILABLE
