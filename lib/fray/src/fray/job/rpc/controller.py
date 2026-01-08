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

"""FrayController Connect RPC service implementation for task queue, actor registry, and worker management.

This module implements the central coordinator for distributed task execution and actor management.
The controller maintains a queue of pending tasks, a registry of actors, and a registry of active
workers. It assigns tasks and actors to workers on demand and tracks their status and results.

Architecture
------------
The controller consists of two main classes:

1. **FrayControllerServicer**: Implements the Connect RPC service interface, handling
   client requests to submit tasks, create/manage actors, check status, and retrieve results.
   Workers poll for tasks and report completion/failure.

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

Actor Lifecycle
---------------
1. Client calls create_actor() with actor spec - controller assigns actor to least-loaded worker
2. Actor enters CREATING state, then transitions to READY once instantiated
3. Client calls call_actor() with method calls - routed as tasks to the hosting worker
4. If worker becomes unavailable, actor status reflects UNAVAILABLE and restart is needed
5. Client calls delete_actor() to remove actor and free resources

Actor Placement
---------------
Actors are placed on workers using a least-loaded strategy, counting the number of actors
per worker and assigning new actors to the worker with the fewest. Named actors support
singleton patterns with get_if_exists=True to retrieve existing instances.

Thread Safety
-------------
All state modifications are protected by threading.Lock. The get_next_task() method
uses threading.Condition for efficient blocking with timeout, allowing workers to
wait for tasks without busy polling. Actor state access is also protected by the same lock.

Deployment
----------
The controller uses uvicorn's ASGI interface and can be deployed with:
- uvicorn directly: `uvicorn --host :: --port 50051 app:app`
- Embedded server (as shown above) for testing and simple deployments
- Production ASGI servers like gunicorn with uvicorn workers
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import cloudpickle
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayController, FrayControllerASGIApplication, FrayWorkerClient

if TYPE_CHECKING:
    import uvicorn

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Tracks the state of a task in the controller."""

    task_id: str
    serialized_fn: bytes
    status: int  # fray_pb2.TaskStatus enum value
    result: bytes | None = None
    error: str | None = None
    serialized_error: bytes | None = None
    worker_id: str | None = None
    resources: dict[str, int] = field(default_factory=dict)
    max_retries: int = 0
    actor_id: str | None = None  # Set if this is an actor method call task


@dataclass
class ActorInfo:
    """Tracks the state of an actor in the controller."""

    actor_id: str
    worker_id: str
    name: str | None
    created_at: float
    last_used: float
    status: int  # fray_pb2.ActorStatus enum value


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

        # Actor state management
        self._actors: dict[str, ActorInfo] = {}
        self._named_actors: dict[str, str] = {}  # name -> actor_id
        self._actor_specs: dict[str, bytes] = {}  # actor_id -> serialized spec for restart

        # Worker health tracking
        self._worker_last_seen: dict[str, float] = {}  # worker_id -> timestamp
        self._health_check_task: asyncio.Task | None = None
        self._running = False

    def _create_worker_client(self, worker_id: str) -> FrayWorkerClient:
        """Create a new client stub for the specified worker (not cached)."""
        worker_info = self._workers[worker_id]
        address = worker_info.address
        # Add http:// prefix if missing
        if not address.startswith("http"):
            address = f"http://{address}"
        return FrayWorkerClient(address)

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
                serialized_error=task.serialized_error or b"",
            )

    async def register_worker(self, request: fray_pb2.WorkerInfo, ctx: RequestContext) -> fray_pb2.Empty:
        """Registers a new worker with the controller."""
        with self._lock:
            self._workers[request.worker_id] = request
            self._worker_last_seen[request.worker_id] = time.time()

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
            # Update worker last seen timestamp
            self._worker_last_seen[request.worker_id] = time.time()

            # Wait for a pending task with 1s timeout to allow graceful shutdown
            # For actor tasks, only return them to the worker hosting the actor
            while True:
                while not self._pending_queue:
                    if not self._condition.wait(timeout=1.0):
                        # Timeout - raise error to allow worker to retry
                        raise ConnectError(Code.DEADLINE_EXCEEDED, "No tasks available")

                # Find a task this worker can execute
                for i, task_id in enumerate(self._pending_queue):
                    task = self._tasks[task_id]

                    # If this is an actor task, only the hosting worker can execute it
                    if task.actor_id:
                        actor_info = self._actors.get(task.actor_id)
                        if actor_info and actor_info.worker_id != request.worker_id:
                            # This actor task is for a different worker, skip it
                            continue

                    # Found a suitable task - remove from queue and assign
                    del self._pending_queue[i]
                    task.status = fray_pb2.TASK_STATUS_RUNNING
                    task.worker_id = request.worker_id

                    return fray_pb2.TaskSpec(
                        task_id=task.task_id,
                        serialized_fn=task.serialized_fn,
                        resources=task.resources,
                        max_retries=task.max_retries,
                    )

                # No suitable task found in queue - wait for more
                if not self._condition.wait(timeout=1.0):
                    raise ConnectError(Code.DEADLINE_EXCEEDED, "No tasks available")

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
            task.serialized_error = request.serialized_error

        return fray_pb2.Empty()

    async def unregister_worker(self, request: fray_pb2.WorkerInfo, ctx: RequestContext) -> fray_pb2.Empty:
        """Removes a worker from the registry."""
        with self._lock:
            self._workers.pop(request.worker_id, None)

        return fray_pb2.Empty()

    async def create_actor(self, request: fray_pb2.ActorSpec, ctx: RequestContext) -> fray_pb2.ActorHandle:
        """
        Create actor and assign to worker using least-loaded placement.

        Named actors can be retrieved with get_if_exists=True to support singleton patterns.
        The actor spec is stored for restart capability on worker failure.
        """
        # Check for existing named actor
        if request.name:
            with self._lock:
                if request.name in self._named_actors:
                    if request.get_if_exists:
                        actor_id = self._named_actors[request.name]
                        actor = self._actors[actor_id]
                        return fray_pb2.ActorHandle(
                            actor_id=actor_id,
                            worker_id=actor.worker_id,
                            name=request.name,
                            status=cast(fray_pb2.ActorStatus, actor.status),
                        )
                    else:
                        raise ConnectError(Code.ALREADY_EXISTS, f"Actor {request.name} already exists")

        # Generate actor ID
        actor_id = str(uuid.uuid4())

        with self._lock:
            # Select worker with fewest actors (least-loaded placement)
            if not self._workers:
                raise ConnectError(Code.UNAVAILABLE, "No workers available")

            worker_id = min(
                self._workers.keys(), key=lambda w: sum(1 for a in self._actors.values() if a.worker_id == w)
            )

            # Store spec for restart
            self._actor_specs[actor_id] = request.serialized_actor

            # Create ActorInfo
            current_time = time.time()
            actor_info = ActorInfo(
                actor_id=actor_id,
                worker_id=worker_id,
                name=request.name or None,
                status=fray_pb2.ACTOR_STATUS_CREATING,
                created_at=current_time,
                last_used=current_time,
            )
            self._actors[actor_id] = actor_info

            if request.name:
                self._named_actors[request.name] = actor_id

        # Instantiate on worker (async RPC to worker)
        # Create a new client for each request (httpx async clients need proper lifecycle)
        actor_spec = fray_pb2.ActorSpec(
            actor_id=actor_id,
            serialized_actor=request.serialized_actor,
            name=request.name,
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create worker client - using async context manager for proper httpx client lifecycle
                worker_client = self._create_worker_client(worker_id)
                async with worker_client:
                    await worker_client.instantiate_actor(actor_spec)

                with self._lock:
                    self._actors[actor_id].status = fray_pb2.ACTOR_STATUS_READY
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait a bit before retrying (worker server might still be starting)
                    await asyncio.sleep(0.1 * (attempt + 1))
                else:
                    with self._lock:
                        self._actors[actor_id].status = fray_pb2.ACTOR_STATUS_FAILED
                    raise ConnectError(
                        Code.INTERNAL, f"Failed to instantiate actor after {max_retries} attempts: {e}"
                    ) from e

        return fray_pb2.ActorHandle(
            actor_id=actor_id,
            worker_id=worker_id,
            name=request.name,
            status=fray_pb2.ACTOR_STATUS_READY,
        )

    async def call_actor(self, request: fray_pb2.ActorCall, ctx: RequestContext) -> fray_pb2.TaskHandle:
        """
        Route actor method call to hosting worker as a task.

        Actor method calls are converted to tasks and routed to the worker hosting the actor.
        If the actor's worker is unavailable, the actor will need to be restarted.
        """
        with self._lock:
            if request.actor_id not in self._actors:
                raise ConnectError(Code.NOT_FOUND, f"Actor {request.actor_id} not found")

            actor_info = self._actors[request.actor_id]

            # Check if worker is alive
            if actor_info.worker_id not in self._workers:
                # Worker died - trigger restart
                raise ConnectError(Code.UNAVAILABLE, f"Actor {request.actor_id} worker unavailable, restart needed")

            # Update last used timestamp
            actor_info.last_used = time.time()

        # Create task for method call
        task_id = str(uuid.uuid4())

        # Wrap as special actor task with actor_id
        actor_task_payload = {
            "actor_id": request.actor_id,
            "serialized_call": request.serialized_call,
        }
        serialized_fn = cloudpickle.dumps(actor_task_payload)

        task = Task(
            task_id=task_id,
            serialized_fn=serialized_fn,
            status=fray_pb2.TASK_STATUS_PENDING,
            actor_id=request.actor_id,  # Mark as actor task
        )

        with self._lock:
            self._tasks[task_id] = task
            # Add to pending queue - worker will pick it up
            self._pending_queue.append(task_id)
            self._condition.notify_all()

        return fray_pb2.TaskHandle(
            task_id=task_id,
            status=fray_pb2.TASK_STATUS_PENDING,
        )

    async def get_actor_status(self, request: fray_pb2.ActorHandle, ctx: RequestContext) -> fray_pb2.ActorHandle:
        """Returns the current status of an actor."""
        with self._lock:
            actor = self._actors.get(request.actor_id)
            if actor is None:
                raise ConnectError(Code.NOT_FOUND, f"Actor {request.actor_id} not found")

            return fray_pb2.ActorHandle(
                actor_id=actor.actor_id,
                worker_id=actor.worker_id,
                name=actor.name or "",
                status=cast(fray_pb2.ActorStatus, actor.status),
            )

    async def delete_actor(self, request: fray_pb2.ActorDeleteRequest, ctx: RequestContext) -> fray_pb2.Empty:
        """
        Delete actor and clean up resources.

        Removes the actor from the registry and cleans up its spec. In a full implementation,
        this would also call the worker to destroy the actor instance.
        """
        with self._lock:
            actor = self._actors.get(request.actor_id)
            if actor is None:
                raise ConnectError(Code.NOT_FOUND, f"Actor {request.actor_id} not found")

            # Remove from named actors if it has a name
            if actor.name and actor.name in self._named_actors:
                del self._named_actors[actor.name]

            # Remove from registries
            del self._actors[request.actor_id]
            self._actor_specs.pop(request.actor_id, None)

        # TODO: Call worker.destroy_actor() when worker client stubs are available

        return fray_pb2.Empty()

    async def _restart_actor(self, actor_id: str) -> None:
        """
        Restart actor on new worker after failure.

        This method re-instantiates an actor on a different worker when its current
        worker becomes unavailable. The actor's original spec is used to recreate it.
        Note: Actor state is lost on restart (Phase 1 limitation).
        """
        with self._lock:
            if actor_id not in self._actor_specs:
                # Can't restart without spec
                if actor_id in self._actors:
                    del self._actors[actor_id]
                return

            if actor_id not in self._actors:
                # Actor was already deleted
                return

            actor_info = self._actors[actor_id]
            old_worker_id = actor_info.worker_id
            actor_spec = self._actor_specs[actor_id]

            # Select new worker (exclude failed worker)
            available_workers = [w_id for w_id in self._workers.keys() if w_id != old_worker_id]

            if not available_workers:
                actor_info.status = fray_pb2.ACTOR_STATUS_FAILED
                return

            new_worker_id = min(
                available_workers, key=lambda w_id: sum(1 for a in self._actors.values() if a.worker_id == w_id)
            )

            # Update status to RESTARTING
            actor_info.status = fray_pb2.ACTOR_STATUS_RESTARTING

        # Instantiate on new worker
        worker_client = self._create_worker_client(new_worker_id)
        spec = fray_pb2.ActorSpec(
            actor_id=actor_id,
            serialized_actor=actor_spec,
            name=actor_info.name or "",
        )

        try:
            async with worker_client:
                await worker_client.instantiate_actor(spec)

            with self._lock:
                actor_info.worker_id = new_worker_id
                actor_info.status = fray_pb2.ACTOR_STATUS_READY

        except Exception as e:
            with self._lock:
                actor_info.status = fray_pb2.ACTOR_STATUS_FAILED
            logger.error(f"Failed to restart actor {actor_id} on worker {new_worker_id}: {e}")

    def start_health_checks(self) -> None:
        """
        Start background health checking of workers.

        This should be called when the controller starts to enable automatic
        detection of worker failures and actor restarts.
        """
        self._running = True
        # Create event loop if none exists, otherwise use current
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, health checks will be started when server starts
            logger.warning("No event loop running, health checks will start when server starts")
            return

        self._health_check_task = loop.create_task(self._health_check_loop())

    async def _health_check_loop(self) -> None:
        """
        Background loop that periodically checks worker health and restarts actors on failed workers.

        Workers are considered failed if they haven't been seen for >30 seconds.
        When a worker fails, all its actors are restarted on other workers.
        """
        while self._running:
            await asyncio.sleep(10)  # Check every 10 seconds

            current_time = time.time()
            failed_workers = []

            with self._lock:
                # Find workers that haven't been seen in >30 seconds
                for worker_id, last_seen in list(self._worker_last_seen.items()):
                    if current_time - last_seen > 30.0:
                        failed_workers.append(worker_id)
                        # Remove from workers registry
                        self._workers.pop(worker_id, None)
                        self._worker_last_seen.pop(worker_id, None)
                        logger.warning(f"Worker {worker_id} failed (not seen for {current_time - last_seen:.1f}s)")

            # Restart actors from failed workers (outside lock to avoid deadlock)
            for worker_id in failed_workers:
                await self._handle_worker_failure(worker_id)

    async def _handle_worker_failure(self, worker_id: str) -> None:
        """
        Handle worker failure by restarting all actors hosted on that worker.

        Args:
            worker_id: ID of the failed worker
        """
        # Find all actors on this worker
        actors_to_restart = []
        with self._lock:
            for actor_id, actor_info in self._actors.items():
                if actor_info.worker_id == worker_id:
                    actors_to_restart.append(actor_id)

        logger.info(f"Restarting {len(actors_to_restart)} actors from failed worker {worker_id}")

        # Restart each actor
        for actor_id in actors_to_restart:
            try:
                await self._restart_actor(actor_id)
                logger.info(f"Successfully restarted actor {actor_id} after worker {worker_id} failure")
            except Exception as e:
                logger.error(f"Failed to restart actor {actor_id} after worker {worker_id} failure: {e}")

    def stop_health_checks(self) -> None:
        """Stop background health checking."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()


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
            # Start health checks when event loop starts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._servicer._running = True
            self._servicer._health_check_task = loop.create_task(self._servicer._health_check_loop())
            assert self._server is not None
            loop.run_until_complete(self._server.serve())

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

        # Wait for server to start with proper readiness checking
        import time

        assert self._server is not None
        max_wait = 5.0
        start_time = time.time()
        while not self._server.started:
            if time.time() - start_time > max_wait:
                raise RuntimeError(f"Server failed to start within {max_wait}s")
            time.sleep(0.01)

        return self._port

    def stop(self) -> None:
        """Stop the ASGI server."""
        self._servicer.stop_health_checks()
        if self._server:
            self._server.should_exit = True

    @property
    def servicer(self) -> FrayControllerServicer:
        """Access the underlying servicer for testing or inspection."""
        return self._servicer
