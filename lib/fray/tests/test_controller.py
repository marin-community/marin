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

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from fray.job.rpc.controller import FrayControllerServer
from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayControllerClient


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
