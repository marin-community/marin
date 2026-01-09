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

"""Tests for execution contexts."""

import threading
import time
import uuid

import pytest
from fray.job import RayContext, SimpleActor, SyncContext, ThreadContext, create_job_ctx
from fray.job.rpc.context import FrayContext
from fray.job.rpc.controller import FrayControllerServer
from fray.job.rpc.worker import FrayWorker


@pytest.fixture(scope="module")
def rpc_controller():
    """Start a controller server for RPC tests."""
    server = FrayControllerServer(port=0)
    port = server.start()

    # Start a worker for the controller
    worker = FrayWorker(f"http://localhost:{port}", port=0)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    time.sleep(2.0)  # Give worker time to register and start its RPC server

    yield port

    # Cleanup
    worker.stop()
    server.stop()
    thread.join(timeout=2.0)


@pytest.fixture(params=["sync", "threadpool", "ray", "rpc"])
def job_context(request, ray_cluster, rpc_controller):
    if request.param == "sync":
        return SyncContext()
    elif request.param == "threadpool":
        return ThreadContext(max_workers=2)
    elif request.param == "rpc":
        return FrayContext(f"http://localhost:{rpc_controller}")

    return RayContext()


def test_context_put_get(job_context):
    """Test that put() raises NotImplementedError."""
    obj = {"key": "value"}
    with pytest.raises(NotImplementedError):
        job_context.put(obj)


def test_context_run(job_context):
    future = job_context.run(lambda x: x * 2, 5)
    assert job_context.get(future) == 10


def test_context_wait(job_context):
    futures = [job_context.run(lambda x: x, i) for i in range(5)]
    ready, pending = job_context.wait(futures, num_returns=2)
    assert len(ready) == 2
    assert len(pending) == 3


def test_fray_job_ctx_invalid():
    with pytest.raises(ValueError, match="Unknown context type"):
        create_job_ctx("invalid")  # type: ignore


def test_actor_named_get_if_exists(job_context):
    actor1 = job_context.create_actor(SimpleActor, 100, name="test_actor", get_if_exists=True)
    future1 = actor1.increment.remote(10)
    job_context.get(future1)

    actor2 = job_context.create_actor(SimpleActor, 999, name="test_actor", get_if_exists=True)
    future2 = actor2.increment.remote(0)
    assert job_context.get(future2) == 110


def test_actor_thread_safety(job_context):
    actor = job_context.create_actor(SimpleActor, 0)

    futures = [actor.increment.remote(1) for _ in range(100)]
    [job_context.get(f) for f in futures]

    final_value = actor.get_value.remote()
    assert job_context.get(final_value) == 100


def test_actor_unnamed_isolation(job_context):
    """Test that unnamed actors are isolated instances."""
    actor1 = job_context.create_actor(SimpleActor, 10)
    actor2 = job_context.create_actor(SimpleActor, 20)

    job_context.get(actor1.increment.remote(5))
    job_context.get(actor2.increment.remote(3))

    assert job_context.get(actor1.get_value.remote()) == 15
    assert job_context.get(actor2.get_value.remote()) == 23


def test_actor_named_without_get_if_exists(job_context):
    """Test that named actors without get_if_exists create new instances."""
    unique_name = f"actor_{uuid.uuid4().hex[:8]}"
    actor1 = job_context.create_actor(SimpleActor, 10, name=unique_name, get_if_exists=False)
    job_context.get(actor1.increment.remote(5))

    with pytest.raises(ValueError):
        job_context.create_actor(SimpleActor, 20, name=unique_name, get_if_exists=False)


# Retry tests for RPC actor method calls


def test_actor_retry_on_unavailable(rpc_controller):
    """Test that actor method calls retry on UNAVAILABLE errors with exponential backoff."""
    from unittest.mock import MagicMock, patch

    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from fray.job.rpc.context import _FrayActorMethod
    from fray.job.rpc.proto import fray_pb2

    # Create a mock controller client
    mock_client = MagicMock()
    mock_actor_handle = fray_pb2.ActorHandle(actor_id="actor-1", worker_id="http://localhost:8000")
    mock_client.get_actor_status.return_value = mock_actor_handle

    unavailable_error = ConnectError(Code.UNAVAILABLE, "Actor restarting")

    # Mock worker stub to fail first, then succeed
    import cloudpickle

    mock_task_result = fray_pb2.TaskResult(serialized_result=cloudpickle.dumps(42))
    mock_worker_stub = MagicMock()
    mock_worker_stub.execute_actor_method.side_effect = [unavailable_error, mock_task_result]

    # Create actor method with retry enabled
    method = _FrayActorMethod(
        actor_id="actor-1",
        method_name="test_method",
        client=mock_client,
        max_retries=5,
        base_delay=0.001,  # 1ms for fast test
        max_delay=0.1,
    )

    # Patch FrayWorkerClientSync to return our mock
    with patch("fray.job.rpc.proto.fray_connect.FrayWorkerClientSync", return_value=mock_worker_stub):
        # Call should succeed after retry
        future = method.remote(arg1=1, arg2=2)

    # Verify execute_actor_method was called twice (1 failure + 1 success)
    assert mock_worker_stub.execute_actor_method.call_count == 2
    assert future._done


def test_actor_retry_max_retries_exceeded(rpc_controller):
    """Test that max retries exceeded raises the original error."""
    from unittest.mock import MagicMock, patch

    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from fray.job.rpc.context import _FrayActorMethod
    from fray.job.rpc.proto import fray_pb2

    mock_client = MagicMock()
    mock_actor_handle = fray_pb2.ActorHandle(actor_id="actor-1", worker_id="http://localhost:8000")
    mock_client.get_actor_status.return_value = mock_actor_handle

    unavailable_error = ConnectError(Code.UNAVAILABLE, "Actor restarting")

    # Always fail with UNAVAILABLE
    mock_worker_stub = MagicMock()
    mock_worker_stub.execute_actor_method.side_effect = unavailable_error

    method = _FrayActorMethod(
        actor_id="actor-1",
        method_name="test_method",
        client=mock_client,
        max_retries=3,
        base_delay=0.001,  # 1ms for fast test
        max_delay=0.1,
    )

    # Should raise ConnectError after max retries
    with patch("fray.job.rpc.proto.fray_connect.FrayWorkerClientSync", return_value=mock_worker_stub):
        with pytest.raises(ConnectError) as exc_info:
            method.remote()

    assert exc_info.value.code == Code.UNAVAILABLE
    assert "Actor restarting" in str(exc_info.value)

    # Verify we tried exactly max_retries times
    assert mock_worker_stub.execute_actor_method.call_count == 3


def test_actor_retry_non_unavailable_error_immediate_raise(rpc_controller):
    """Test that non-UNAVAILABLE errors are raised immediately without retry."""
    from unittest.mock import MagicMock, patch

    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    from fray.job.rpc.context import _FrayActorMethod
    from fray.job.rpc.proto import fray_pb2

    mock_client = MagicMock()
    mock_actor_handle = fray_pb2.ActorHandle(actor_id="actor-1", worker_id="http://localhost:8000")
    mock_client.get_actor_status.return_value = mock_actor_handle

    not_found_error = ConnectError(Code.NOT_FOUND, "Actor not found")

    # Fail with NOT_FOUND (should not retry)
    mock_worker_stub = MagicMock()
    mock_worker_stub.execute_actor_method.side_effect = not_found_error

    method = _FrayActorMethod(
        actor_id="actor-1",
        method_name="test_method",
        client=mock_client,
        max_retries=5,
        base_delay=0.001,
        max_delay=0.1,
    )

    # Should raise immediately without retry
    with patch("fray.job.rpc.proto.fray_connect.FrayWorkerClientSync", return_value=mock_worker_stub):
        start_time = time.time()
        with pytest.raises(ConnectError) as exc_info:
            method.remote()
        elapsed_time = time.time() - start_time

    assert exc_info.value.code == Code.NOT_FOUND
    assert "Actor not found" in str(exc_info.value)

    # Should only try once (no retry)
    assert mock_worker_stub.execute_actor_method.call_count == 1

    # Should be very fast (no retry delay)
    assert elapsed_time < 0.1
