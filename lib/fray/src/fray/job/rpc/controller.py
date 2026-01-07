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

"""FrayController Connect RPC service implementation for task queue and worker management.

This module implements the central coordinator for distributed task execution. The controller
maintains a queue of pending tasks and a registry of active workers. It assigns tasks to
workers on demand and tracks task status and results.

Architecture
------------
The controller consists of two main classes:

1. **FrayControllerServicer**: Implements the Connect RPC service interface, handling
   client requests to submit tasks, check status, and retrieve results. Workers poll
   for tasks and report completion/failure.

2. **FrayControllerServer**: Wraps the servicer in a uvicorn ASGI server for deployment.
   Manages server lifecycle including port binding and graceful shutdown.

Usage
-----
Starting a controller server::

    from fray.job.rpc.controller import FrayControllerServer

    server = FrayControllerServer(port=50051)
    port = server.start()
    print(f"Controller listening on port {port}")

    # Keep server running...
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()

Task Lifecycle
--------------
1. Client submits task via submit_task() - task enters PENDING state and joins queue
2. Worker calls get_next_task() - task transitions to RUNNING and assigned to worker
3. Worker executes task and calls either:
   - report_task_complete() with result - task transitions to COMPLETED
   - report_task_failed() with error - task transitions to FAILED
4. Client polls get_task_status() until complete, then calls get_task_result()

Thread Safety
-------------
All state modifications are protected by threading.Lock. The get_next_task() method
uses threading.Condition for efficient blocking with timeout, allowing workers to
wait for tasks without busy polling.

Deployment
----------
The controller uses uvicorn's ASGI interface and can be deployed with:
- uvicorn directly: `uvicorn --host :: --port 50051 app:app`
- Embedded server (as shown above) for testing and simple deployments
- Production ASGI servers like gunicorn with uvicorn workers
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayController, FrayControllerASGIApplication

if TYPE_CHECKING:
    import uvicorn


@dataclass
class Task:
    """Tracks the state of a task in the controller."""

    task_id: str
    serialized_fn: bytes
    status: int  # fray_pb2.TaskStatus enum value
    result: bytes | None = None
    error: str | None = None
    worker_id: str | None = None
    resources: dict[str, int] = field(default_factory=dict)
    max_retries: int = 0


class FrayControllerServicer(FrayController):
    """
    Connect RPC servicer implementing the FrayController service.

    Manages a task queue and worker registry for distributed task execution.
    Thread-safe using locks and condition variables for coordination.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._pending_queue: deque[str] = deque()
        self._workers: dict[str, fray_pb2.WorkerInfo] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    async def submit_task(self, request: fray_pb2.TaskSpec, ctx: RequestContext) -> fray_pb2.TaskHandle:
        """
        Client submits a new task for execution.

        Generates a unique task ID, stores the task as PENDING, adds it to the queue,
        and notifies waiting workers.
        """
        task_id = str(uuid.uuid4())

        task = Task(
            task_id=task_id,
            serialized_fn=request.serialized_fn,
            status=fray_pb2.TASK_STATUS_PENDING,
            resources=dict(request.resources),
            max_retries=request.max_retries,
        )

        with self._lock:
            self._tasks[task_id] = task
            self._pending_queue.append(task_id)
            self._condition.notify_all()

        return fray_pb2.TaskHandle(
            task_id=task_id,
            status=fray_pb2.TASK_STATUS_PENDING,
        )

    async def get_task_status(self, request: fray_pb2.TaskHandle, ctx: RequestContext) -> fray_pb2.TaskHandle:
        """Returns the current status of a task."""
        with self._lock:
            task = self._tasks.get(request.task_id)
            if task is None:
                raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

            return fray_pb2.TaskHandle(
                task_id=task.task_id,
                status=cast(fray_pb2.TaskStatus, task.status),
                worker_id=task.worker_id or "",
                error=task.error or "",
            )

    async def get_task_result(self, request: fray_pb2.TaskHandle, ctx: RequestContext) -> fray_pb2.TaskResult:
        """Returns the result of a completed task."""
        with self._lock:
            task = self._tasks.get(request.task_id)
            if task is None:
                raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

            return fray_pb2.TaskResult(
                task_id=task.task_id,
                serialized_result=task.result or b"",
                error=task.error or "",
            )

    async def register_worker(self, request: fray_pb2.WorkerInfo, ctx: RequestContext) -> fray_pb2.Empty:
        """Registers a new worker with the controller."""
        with self._lock:
            self._workers[request.worker_id] = request

        return fray_pb2.Empty()

    async def get_next_task(self, request: fray_pb2.GetTaskRequest, ctx: RequestContext) -> fray_pb2.TaskSpec:
        """
        Worker requests the next pending task.

        Waits for up to 1 second for a task to become available. If a task is available,
        pops it from the queue, marks it as RUNNING, assigns the worker, and returns it.
        """
        # Run the blocking wait in a thread pool to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._get_next_task_blocking, request)
        return result

    def _get_next_task_blocking(self, request: fray_pb2.GetTaskRequest) -> fray_pb2.TaskSpec:
        """Helper method to handle blocking wait for next task."""
        with self._condition:
            # Wait for a pending task with 1s timeout to allow graceful shutdown
            while not self._pending_queue:
                if not self._condition.wait(timeout=1.0):
                    # Timeout - raise error to allow worker to retry
                    raise ConnectError(Code.DEADLINE_EXCEEDED, "No tasks available")

            task_id = self._pending_queue.popleft()
            task = self._tasks[task_id]
            task.status = fray_pb2.TASK_STATUS_RUNNING
            task.worker_id = request.worker_id

            return fray_pb2.TaskSpec(
                task_id=task.task_id,
                serialized_fn=task.serialized_fn,
                resources=task.resources,
                max_retries=task.max_retries,
            )

    async def report_task_complete(self, request: fray_pb2.TaskResult, ctx: RequestContext) -> fray_pb2.Empty:
        """Worker reports successful task completion."""
        with self._lock:
            task = self._tasks.get(request.task_id)
            if task is None:
                raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

            task.status = fray_pb2.TASK_STATUS_COMPLETED
            task.result = request.serialized_result

        return fray_pb2.Empty()

    async def report_task_failed(self, request: fray_pb2.TaskResult, ctx: RequestContext) -> fray_pb2.Empty:
        """Worker reports task failure."""
        with self._lock:
            task = self._tasks.get(request.task_id)
            if task is None:
                raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

            task.status = fray_pb2.TASK_STATUS_FAILED
            task.error = request.error

        return fray_pb2.Empty()

    async def unregister_worker(self, request: fray_pb2.WorkerInfo, ctx: RequestContext) -> fray_pb2.Empty:
        """Removes a worker from the registry."""
        with self._lock:
            self._workers.pop(request.worker_id, None)

        return fray_pb2.Empty()


class FrayControllerServer:
    """
    Wraps the FrayControllerServicer in an ASGI server using uvicorn.

    Manages server lifecycle including starting, stopping, and port binding.
    """

    def __init__(self, port: int = 0) -> None:
        """
        Initialize the controller server.

        Args:
            port: Port to bind to. 0 means auto-assign an available port.
        """
        self._port = port
        self._servicer = FrayControllerServicer()
        self._app = FrayControllerASGIApplication(self._servicer)
        self._server: uvicorn.Server | None = None

    def start(self) -> int:
        """
        Start the ASGI server.

        Returns:
            The actual port the server is listening on.
        """
        import socket

        import uvicorn

        # If port is 0, find an available port
        if self._port == 0:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.listen(1)
                self._port = s.getsockname()[1]

        config = uvicorn.Config(self._app, host="::", port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)

        # Start the server in a background thread
        def run_server():
            asyncio.run(self._server.serve())

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

        # Wait for server to start with proper readiness checking
        import time

        max_wait = 5.0
        start_time = time.time()
        while not self._server.started:
            if time.time() - start_time > max_wait:
                raise RuntimeError(f"Server failed to start within {max_wait}s")
            time.sleep(0.01)

        return self._port

    def stop(self) -> None:
        """Stop the ASGI server."""
        if self._server:
            self._server.should_exit = True

    @property
    def servicer(self) -> FrayControllerServicer:
        """Access the underlying servicer for testing or inspection."""
        return self._servicer
