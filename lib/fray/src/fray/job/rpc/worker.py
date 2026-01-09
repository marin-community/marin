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

"""Fray RPC worker implementation.

This module implements the worker process that executes tasks pushed from the FrayController.
Workers register with the controller, receive task assignments via RPC, execute them using
cloudpickle deserialization, and report results back.

Architecture
------------
Each FrayWorker consists of:

1. **FrayWorkerServicer**: Exposes a Connect RPC service for receiving task assignments,
   health checks, and task monitoring. Controllers push tasks to workers and can query
   worker status.

2. **FrayWorker**: Main worker class that maintains a connection to the controller,
   manages a thread pool executor for task execution, and handles worker lifecycle.

Usage
-----
Starting a worker::

    from fray.job.rpc.worker import FrayWorker

    worker = FrayWorker(
        controller_address="http://localhost:50051",
        worker_id="worker-1",  # Optional, UUID generated if not provided
        port=0  # Port for worker's own RPC server, 0 = auto-assign
    )
    worker.run()  # Blocks until stopped

Worker Lifecycle
----------------
1. Worker registers with controller via register_worker()
2. Starts RPC server and waits for task assignments from controller
3. When controller calls assign_task():
   - Deserializes task function and arguments using cloudpickle
   - Submits to thread pool executor
   - Registers callback to report result when complete
4. On shutdown:
   - Unregisters from controller
   - Stops RPC server
   - Exits gracefully

Task Execution
--------------
Tasks are serialized as cloudpickle payloads containing:
- fn: The function to execute
- args: Positional arguments to pass to the function
- For actor tasks: actor_id and serialized method call

The worker deserializes the payload, executes via thread pool, and reports the result
back using cloudpickle. Any exceptions during execution are caught and reported as task failures.

Error Handling
--------------
- RPC errors (connection failures, timeouts) are logged but don't crash the worker
- Task execution errors are caught and reported as failed tasks
- Graceful shutdown on SIGINT/SIGTERM via stop() method

Health Monitoring
-----------------
Workers expose their own Connect RPC service providing:
- health_check(): Returns worker status including uptime and current tasks
- list_tasks(): Returns detailed information about running tasks
- probe_task(): Returns status of a specific task

This allows controllers and monitoring tools to track worker health and task execution.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import cloudpickle
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from uvicorn import Config, Server

from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import (
    FrayControllerClientSync,
    FrayWorker as FrayWorkerProtocol,
    FrayWorkerASGIApplication,
)
from fray.job.rpc.proto.fray_pb2 import TaskStatus

logger = logging.getLogger(__name__)


class FrayWorkerServicer(FrayWorkerProtocol):
    """
    Implements the FrayWorker Connect RPC service for receiving task assignments,
    health checks, and task monitoring.

    This servicer runs on each worker to provide status information about the worker's
    health, uptime, and currently running tasks. It also receives task assignments from
    the controller and manages task execution via a thread pool executor.
    """

    def __init__(self, worker_id: str, controller_client: FrayControllerClientSync, max_workers: int = 4):
        self.worker_id = worker_id
        self.start_time = time.time_ns() // 1_000_000
        self.current_tasks: list[tuple[str, TaskStatus, int]] = []
        self._actor_instances: dict[str, Any] = {}
        self._running_tasks: dict[str, Future] = {}
        self._controller_client = controller_client
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    async def health_check(self, request: fray_pb2.Empty, ctx: RequestContext) -> fray_pb2.WorkerStatus:
        return self._get_status()

    async def list_tasks(self, request: fray_pb2.Empty, ctx: RequestContext) -> fray_pb2.WorkerStatus:
        return self._get_status()

    def _get_status(self) -> fray_pb2.WorkerStatus:
        """Generate current worker status."""
        current_time_ms = time.time_ns() // 1_000_000
        uptime_ms = current_time_ms - self.start_time

        with self._lock:
            worker_tasks = [
                fray_pb2.WorkerTask(
                    task_id=task_id,
                    status=status,
                    started_at_ms=started_at_ms,
                )
                for task_id, status, started_at_ms in self.current_tasks
            ]

        return fray_pb2.WorkerStatus(
            worker_id=self.worker_id,
            healthy=True,
            current_tasks=worker_tasks,
            uptime_ms=uptime_ms,
        )

    def add_task(self, task_id: str, status: TaskStatus) -> None:
        """Add a task to the current tasks list."""
        started_at_ms = time.time_ns() // 1_000_000
        with self._lock:
            self.current_tasks.append((task_id, status, started_at_ms))

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update the status of a task."""
        with self._lock:
            for i, (tid, _, started_at_ms) in enumerate(self.current_tasks):
                if tid == task_id:
                    self.current_tasks[i] = (task_id, status, started_at_ms)
                    break

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the current tasks list."""
        with self._lock:
            self.current_tasks = [
                (tid, status, started_at_ms) for tid, status, started_at_ms in self.current_tasks if tid != task_id
            ]

    async def instantiate_actor(self, request: fray_pb2.ActorSpec, ctx: RequestContext) -> fray_pb2.ActorHandle:
        """
        Instantiate an actor from serialized specification.

        Deserializes the actor class, arguments, and keyword arguments from the request,
        creates an instance, and stores it in the actor_instances dictionary.

        Args:
            request: ActorSpec containing actor_id and serialized actor data
            ctx: Request context from Connect RPC

        Returns:
            ActorHandle with actor_id, worker_id, name, and READY status

        Raises:
            ConnectError with Code.INTERNAL if instantiation fails
        """
        # Deserialize actor spec
        actor_data = cloudpickle.loads(request.serialized_actor)
        cls = actor_data["cls"]
        args = actor_data["args"]
        kwargs = actor_data["kwargs"]

        # Create instance
        try:
            instance = cls(*args, **kwargs)

            with self._lock:
                self._actor_instances[request.actor_id] = instance

            logger.info(f"Worker {self.worker_id} instantiated actor {request.actor_id}")

            return fray_pb2.ActorHandle(
                actor_id=request.actor_id,
                worker_id=self.worker_id,
                name=request.name,
                status=fray_pb2.ACTOR_STATUS_READY,
            )

        except Exception as e:
            error_msg = f"Failed to instantiate actor: {type(e).__name__}: {e!s}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ConnectError(Code.INTERNAL, error_msg) from e

    async def execute_actor_method(self, request: fray_pb2.ActorCall, ctx: RequestContext) -> fray_pb2.TaskResult:
        """
        Execute a method on an actor instance.

        Deserializes the method call (method name, args, kwargs), retrieves the actor
        instance, executes the method, and returns the serialized result.

        Args:
            request: ActorCall containing actor_id and serialized method call
            ctx: Request context from Connect RPC

        Returns:
            TaskResult with serialized result or error information

        Raises:
            ConnectError with Code.NOT_FOUND if actor not found
        """
        with self._lock:
            if request.actor_id not in self._actor_instances:
                raise ConnectError(Code.NOT_FOUND, f"Actor {request.actor_id} not found on this worker")

            instance = self._actor_instances[request.actor_id]

        # Deserialize method call
        call_data = cloudpickle.loads(request.serialized_call)
        method_name = call_data["method"]
        args = call_data["args"]
        kwargs = call_data["kwargs"]

        # Execute method
        try:
            method = getattr(instance, method_name)
            result = method(*args, **kwargs)

            serialized_result = cloudpickle.dumps(result)

            logger.info(f"Worker {self.worker_id} executed {method_name} on actor {request.actor_id}")

            return fray_pb2.TaskResult(
                task_id="",  # Not used for direct calls
                serialized_result=serialized_result,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e!s}\n{traceback.format_exc()}"
            serialized_error = cloudpickle.dumps(e)

            logger.error(
                f"Worker {self.worker_id} failed executing {method_name} on actor {request.actor_id}: {error_msg}"
            )

            return fray_pb2.TaskResult(
                task_id="",
                error=error_msg,
                serialized_error=serialized_error,
            )

    async def destroy_actor(self, request: fray_pb2.ActorDeleteRequest, ctx: RequestContext) -> fray_pb2.Empty:
        """Destroy an actor instance."""
        with self._lock:
            if request.actor_id in self._actor_instances:
                del self._actor_instances[request.actor_id]
                logger.info(f"Worker {self.worker_id} destroyed actor {request.actor_id}")

        return fray_pb2.Empty()

    async def list_actors(self, request: fray_pb2.Empty, ctx: RequestContext) -> fray_pb2.ActorList:
        """
        List all actor instances on this worker.

        Returns information about all actors currently hosted on this worker.

        Args:
            request: Empty request
            ctx: Request context from Connect RPC

        Returns:
            ActorList containing actor handles for all actors on this worker
        """
        with self._lock:
            actor_handles = [
                fray_pb2.ActorHandle(
                    actor_id=actor_id,
                    worker_id=self.worker_id,
                    name="",  # Name is not stored in worker state
                    status=fray_pb2.ACTOR_STATUS_READY,
                )
                for actor_id in self._actor_instances.keys()
            ]

        return fray_pb2.ActorList(actors=actor_handles)

    async def assign_task(self, request: fray_pb2.TaskAssignment, ctx: RequestContext) -> fray_pb2.Empty:
        """
        Receive a task assignment from the controller.

        Deserializes the task payload, submits it to the executor (handling both actor
        and regular tasks), and registers a callback to report the result when complete.

        Args:
            request: TaskAssignment containing task_id and serialized function/args
            ctx: Request context from Connect RPC

        Returns:
            Empty response (result is reported asynchronously via callback)
        """
        task_id = request.task_id
        logger.info(f"Worker {self.worker_id} received task assignment {task_id}")

        # Track task as running
        self.add_task(task_id, fray_pb2.TASK_STATUS_RUNNING)

        # Deserialize the task payload
        task_data = cloudpickle.loads(request.serialized_fn)

        # Submit to executor based on task type
        if "actor_id" in task_data:
            # Actor method call
            future = self._executor.submit(self._execute_actor_task, task_id, task_data)
        else:
            # Regular task
            future = self._executor.submit(self._execute_regular_task, task_id, task_data)

        # Store future
        with self._lock:
            self._running_tasks[task_id] = future

        # Register callback to report result
        future.add_done_callback(lambda f: self._report_task_result(task_id, f))

        return fray_pb2.Empty()

    async def probe_task(self, request: fray_pb2.TaskHandle, ctx: RequestContext) -> fray_pb2.WorkerTask:
        """
        Query the status of a specific task.

        Args:
            request: TaskHandle containing task_id to query
            ctx: Request context from Connect RPC

        Returns:
            WorkerTask with task status and start time

        Raises:
            ConnectError with Code.NOT_FOUND if task not found
        """
        with self._lock:
            for task_id, status, started_at_ms in self.current_tasks:
                if task_id == request.task_id:
                    return fray_pb2.WorkerTask(
                        task_id=task_id,
                        status=status,
                        started_at_ms=started_at_ms,
                    )

        raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found on this worker")

    def _execute_regular_task(self, task_id: str, task_data: dict) -> Any:
        """
        Execute a regular (non-actor) task.

        Args:
            task_id: Task identifier
            task_data: Deserialized task payload containing 'fn' and 'args'

        Returns:
            Result of executing the function
        """
        fn = task_data["fn"]
        args = task_data["args"]
        logger.info(f"Worker {self.worker_id} executing task {task_id}")
        return fn(*args)

    def _execute_actor_task(self, task_id: str, task_data: dict) -> Any:
        """
        Execute an actor method call task.

        Args:
            task_id: Task identifier
            task_data: Deserialized task payload containing 'actor_id' and 'serialized_call'

        Returns:
            Result of executing the actor method

        Raises:
            RuntimeError: If actor not found on this worker
        """
        actor_id = task_data["actor_id"]
        serialized_call = task_data["serialized_call"]

        # Get actor instance
        with self._lock:
            if actor_id not in self._actor_instances:
                raise RuntimeError(f"Actor {actor_id} not found on this worker")
            instance = self._actor_instances[actor_id]

        # Deserialize method call
        call_data = cloudpickle.loads(serialized_call)
        method_name = call_data["method"]
        args = call_data["args"]
        kwargs = call_data["kwargs"]

        # Execute method
        logger.info(f"Worker {self.worker_id} executing {method_name} on actor {actor_id} (task {task_id})")
        method = getattr(instance, method_name)
        return method(*args, **kwargs)

    def _report_task_result(self, task_id: str, future: Future) -> None:
        """
        Report task result to controller.

        Retrieves result or exception from the completed future, creates appropriate
        TaskResultPayload, and sends to controller via report_task_result RPC.

        Args:
            task_id: Task identifier
            future: Completed future from executor
        """
        try:
            result = future.result()
            serialized_value = cloudpickle.dumps(result)

            payload = fray_pb2.TaskResultPayload(
                task_id=task_id,
                serialized_value=serialized_value,
            )

            logger.info(f"Worker {self.worker_id} completed task {task_id}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e!s}\n{traceback.format_exc()}"
            serialized_error = cloudpickle.dumps(e)

            payload = fray_pb2.TaskResultPayload(
                task_id=task_id,
                serialized_error=serialized_error,
            )

            logger.error(f"Worker {self.worker_id} failed task {task_id}: {error_msg}")

        finally:
            # Remove from tracking
            self.remove_task(task_id)
            with self._lock:
                self._running_tasks.pop(task_id, None)

        # Report to controller
        try:
            self._controller_client.report_task_result(payload)
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to report result for task {task_id}: {e}")

    def shutdown(self) -> None:
        """Shutdown the executor and wait for pending tasks."""
        self._executor.shutdown(wait=True)


class FrayWorker:
    """
    Worker that connects to a FrayController and receives pushed task assignments.

    The worker maintains a connection to the controller and exposes its own Connect RPC
    service for receiving task assignments, health checks, and monitoring.
    """

    def __init__(self, controller_address: str, worker_id: str | None = None, port: int = 0, max_workers: int = 4):
        """
        Initialize a FrayWorker.

        Args:
            controller_address: Address of the controller to connect to (e.g., "localhost:50051")
            worker_id: Optional worker ID. If not provided, a UUID will be generated.
            port: Port to run the worker's own Connect RPC server on. If 0, an available port will be chosen.
            max_workers: Maximum number of concurrent tasks to execute in thread pool.
        """
        self.worker_id = worker_id or str(uuid.uuid4())
        self._port = port
        self._running = False

        # Ensure controller address has http:// prefix
        if not controller_address.startswith("http://") and not controller_address.startswith("https://"):
            controller_address = f"http://{controller_address}"
        self.controller_address = controller_address

        # Create Connect RPC client to controller
        self._controller_client = FrayControllerClientSync(controller_address)

        # Create our own Connect RPC service
        self._servicer = FrayWorkerServicer(self.worker_id, self._controller_client, max_workers)
        app = FrayWorkerASGIApplication(self._servicer)

        # Create uvicorn server for ASGI app
        self._server_config = Config(app=app, host="0.0.0.0", port=port, log_level="error")
        self._server = Server(config=self._server_config)
        self._server_thread: threading.Thread | None = None
        self._port = port

    @property
    def address(self) -> str:
        """Return the address of this worker's RPC server."""
        # Get actual port from server after it starts
        if hasattr(self._server, "servers") and self._server.servers:
            # Extract port from running server
            actual_port = self._server.servers[0].sockets[0].getsockname()[1]
            return f"localhost:{actual_port}"
        return f"localhost:{self._port}"

    def register(self) -> None:
        """Register this worker with the controller."""
        worker_info = fray_pb2.WorkerInfo(
            worker_id=self.worker_id,
            address=self.address,
            num_cpus=os.cpu_count() or 1,
            memory_bytes=0,  # Not tracking memory for now
        )
        self._controller_client.register_worker(worker_info)
        logger.info(f"Worker {self.worker_id} registered with controller at {self.controller_address}")

    def run(self) -> None:
        """
        Main worker loop.

        Registers with the controller, starts the RPC server to receive task assignments,
        and blocks until stopped. Tasks are pushed from the controller and executed
        asynchronously in a thread pool.
        """
        self._running = True

        # Start our own Connect RPC server in a background thread BEFORE registering
        # This ensures the server is ready to accept requests when the controller
        # starts pushing tasks immediately after registration
        def run_server():
            self._server.run()

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Wait for server to be ready
        max_wait = 5.0
        start_time = time.time()
        while not self._server.started:
            if time.time() - start_time > max_wait:
                raise RuntimeError(f"Worker RPC server failed to start within {max_wait}s")
            time.sleep(0.01)

        logger.info(f"Worker {self.worker_id} Connect RPC server started on {self.address}")

        # Now register with controller (server is ready to accept requests)
        self.register()

        try:
            # Block until stopped - tasks are received via assign_task() RPC
            while self._running:
                time.sleep(0.1)

        finally:
            # Cleanup
            self._cleanup()

    def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self.worker_id}")
        self._running = False

    def _cleanup(self) -> None:
        """Clean up resources when stopping."""
        # Shutdown executor
        try:
            self._servicer.shutdown()
            logger.info(f"Worker {self.worker_id} executor shutdown")
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")

        # Unregister from controller
        try:
            worker_info = fray_pb2.WorkerInfo(
                worker_id=self.worker_id,
                address=self.address,
                num_cpus=os.cpu_count() or 1,
                memory_bytes=0,
            )
            self._controller_client.unregister_worker(worker_info)
            logger.info(f"Worker {self.worker_id} unregistered from controller")
        except Exception as e:
            logger.error(f"Error unregistering worker: {e}")

        # Stop our Connect RPC server
        self._server.should_exit = True
        if self._server_thread:
            self._server_thread.join(timeout=5.0)
        logger.info(f"Worker {self.worker_id} Connect RPC server stopped")
