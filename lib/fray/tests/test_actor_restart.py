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

"""Integration tests for actor restart on worker failure (Phase 2)."""

import asyncio
import threading
import time

import cloudpickle
import httpx
import pytest
from uvicorn import Config, Server

from fray.job.rpc.controller import FrayControllerServer
from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayControllerClient, FrayWorkerASGIApplication
from fray.job.rpc.worker import FrayWorker, FrayWorkerServicer


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


@pytest.fixture
def controller():
    """Start a test controller server."""
    server = FrayControllerServer(port=0)
    port = server.start()
    yield server, port
    server.stop()


@pytest.fixture
def controller_url(controller):
    """Get controller URL."""
    _, port = controller
    return f"http://localhost:{port}"


@pytest.fixture
def worker_server():
    """Start a test worker server for actor instantiation."""
    from unittest.mock import Mock

    controller_client_mock = Mock()
    servicer = FrayWorkerServicer(worker_id="worker-1", controller_client=controller_client_mock)
    app = FrayWorkerASGIApplication(servicer)
    config = Config(app=app, host="127.0.0.1", port=0, log_level="error")
    server = Server(config=config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to start
    max_wait = 5.0
    start_time = time.time()
    while not server.started:
        if time.time() - start_time > max_wait:
            raise RuntimeError(f"Worker server failed to start within {max_wait}s")
        time.sleep(0.01)

    actual_port = server.servers[0].sockets[0].getsockname()[1]

    yield server, actual_port

    server.should_exit = True
    server_thread.join(timeout=2.0)


@pytest.fixture
def two_worker_servers():
    """Start two test worker servers for actor instantiation."""
    from unittest.mock import Mock

    servers = []
    threads = []
    ports = []

    for worker_id in ["worker-1", "worker-2"]:
        controller_client_mock = Mock()
        servicer = FrayWorkerServicer(worker_id=worker_id, controller_client=controller_client_mock)
        app = FrayWorkerASGIApplication(servicer)
        config = Config(app=app, host="127.0.0.1", port=0, log_level="error")
        server = Server(config=config)

        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        servers.append(server)
        threads.append(server_thread)

    # Wait for servers to start
    max_wait = 5.0
    start_time = time.time()
    for server in servers:
        while not server.started:
            if time.time() - start_time > max_wait:
                raise RuntimeError(f"Worker server failed to start within {max_wait}s")
            time.sleep(0.01)

    for server in servers:
        actual_port = server.servers[0].sockets[0].getsockname()[1]
        ports.append(actual_port)

    yield servers, ports

    for server, thread in zip(servers, threads, strict=False):
        server.should_exit = True
        thread.join(timeout=2.0)


@pytest.mark.asyncio
async def test_actor_restart_on_worker_failure_integration(controller):
    """
    Integration test: Start controller + 2 workers, create actor, kill worker, verify restart.

    This is the main Phase 2 test demonstrating automatic actor restart.
    """
    controller_server, port = controller
    controller_url = f"http://localhost:{port}"
    # Start two worker processes
    worker1 = FrayWorker(controller_address=controller_url, worker_id="worker-1", port=0)
    worker2 = FrayWorker(controller_address=controller_url, worker_id="worker-2", port=0)

    # Start workers in background threads
    import threading

    worker1_thread = threading.Thread(target=worker1.run, daemon=True)
    worker2_thread = threading.Thread(target=worker2.run, daemon=True)
    worker1_thread.start()
    worker2_thread.start()

    # Wait for workers to register
    await asyncio.sleep(1.0)

    async with httpx.AsyncClient() as http_client:
        client = FrayControllerClient(controller_url, session=http_client)

        # Create an actor
        actor_data = {"cls": Counter, "args": (10,), "kwargs": {}}
        serialized_actor = cloudpickle.dumps(actor_data)
        actor_spec = fray_pb2.ActorSpec(
            serialized_actor=serialized_actor,
            name="counter",
            get_if_exists=False,
        )
        handle = await client.create_actor(actor_spec)

        # Verify we got a valid address
        assert handle.worker_id
        # The worker_id field now contains the worker address for client connections
        original_worker_address = handle.worker_id

        # Map address back to worker instance
        if worker1.address == original_worker_address:
            original_worker = worker1
            original_worker_id = "worker-1"
        else:
            original_worker = worker2
            original_worker_id = "worker-2"

        # Call actor method to verify it works
        call_data = {"method": "increment", "args": (), "kwargs": {}}
        serialized_call = cloudpickle.dumps(call_data)
        actor_call = fray_pb2.ActorCall(
            actor_id=handle.actor_id,
            serialized_call=serialized_call,
        )
        task_handle = await client.call_actor(actor_call)

        # Wait for task to complete
        for _ in range(50):
            status = await client.get_task_status(task_handle)
            if status.status == fray_pb2.TASK_STATUS_COMPLETED:
                break
            await asyncio.sleep(0.1)

        result = await client.get_task_result(task_handle)
        count = cloudpickle.loads(result.serialized_result)
        assert count == 11  # Initial 10 + 1

        # Kill the worker hosting the actor
        original_worker.stop()

        # Wait for health check to detect failure and restart actor
        # Health check runs every 10s and worker timeout is 30s, so we need to wait
        # For testing, we'll manually trigger worker removal and restart
        # In production, this happens automatically via health checks

        # Manually unregister the failed worker
        await client.unregister_worker(fray_pb2.WorkerInfo(worker_id=original_worker_id))

        # Wait a moment for cleanup
        await asyncio.sleep(0.5)

        # Try to call actor - should fail because worker is gone
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        with pytest.raises(ConnectError) as exc_info:
            await client.call_actor(actor_call)
        assert exc_info.value.code == Code.UNAVAILABLE

        # Stop the other worker that's still running
        other_worker_id = "worker-2" if original_worker_id == "worker-1" else "worker-1"
        if other_worker_id == "worker-1":
            worker1.stop()
        else:
            worker2.stop()

        # Add a NEW worker to the cluster
        worker3 = FrayWorker(controller_address=controller_url, worker_id="worker-3", port=0)
        worker3_thread = threading.Thread(target=worker3.run, daemon=True)
        worker3_thread.start()

        # Wait for new worker to register
        await asyncio.sleep(1.0)

        # Manually trigger actor restart on the new worker
        # In production, this would happen automatically via health check loop after 30s
        # For testing, we manually trigger it to avoid long waits
        await controller_server.servicer._handle_worker_failure(original_worker_id)

        # Wait for actor to restart and verify it's on worker-3
        # The actor should be moved to the new worker
        for _ in range(50):
            actor_status = await client.get_actor_status(handle)
            # Check if the actor has been assigned to worker-3
            if actor_status.worker_id == worker3.address:
                break
            await asyncio.sleep(0.1)

        # Verify the actor has been restarted on worker-3 (check worker location)
        assert actor_status.worker_id == worker3.address

        # Verify the actor works by calling increment method
        call_data_recovery = {"method": "increment", "args": (), "kwargs": {}}
        serialized_call_recovery = cloudpickle.dumps(call_data_recovery)
        actor_call_recovery = fray_pb2.ActorCall(
            actor_id=handle.actor_id,
            serialized_call=serialized_call_recovery,
        )

        task_handle_recovery = await client.call_actor(actor_call_recovery)

        # Wait for task to complete
        for _ in range(50):
            status = await client.get_task_status(task_handle_recovery)
            if status.status == fray_pb2.TASK_STATUS_COMPLETED:
                break
            await asyncio.sleep(0.1)

        result = await client.get_task_result(task_handle_recovery)
        count = cloudpickle.loads(result.serialized_result)
        # Actor state was reset on restart, so count should be initial (10) + 1 = 11
        assert count == 11

        # Clean up the new worker
        worker3.stop()

    # Cleanup
    worker1.stop()
    worker2.stop()
