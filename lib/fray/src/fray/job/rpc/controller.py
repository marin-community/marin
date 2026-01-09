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
from fray.job.rpc.proto.fray_connect import (
    FrayController,
    FrayControllerASGIApplication,
    FrayWorkerClient,
    FrayWorkerClientSync,
)

if TYPE_CHECKING:
    import uvicorn

logger = logging.getLogger(__name__)

ACTOR_INSTANTIATION_TIMEOUT_MS = 10_000  # 10 seconds default
TASK_ASSIGNMENT_TIMEOUT_MS = 5_000  # 5 seconds default for pushing tasks to workers


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

        # Worker stub management for pushing tasks
        self._worker_stubs: dict[str, FrayWorkerClientSync] = {}
        self._worker_stub_lock = threading.Lock()

        # Task scheduler thread
        self._scheduler_thread: threading.Thread | None = None

    def _create_worker_client(self, worker_id: str) -> FrayWorkerClient:
        """Create a new client stub for the specified worker (not cached)."""
        worker_info = self._workers[worker_id]
        address = worker_info.address
        # Add http:// prefix if missing
        if not address.startswith("http"):
            address = f"http://{address}"
        return FrayWorkerClient(address)

    def _get_or_create_worker_stub(self, worker_id: str) -> FrayWorkerClientSync:
        """Get or create a synchronous client stub for the specified worker."""
        with self._worker_stub_lock:
            if worker_id not in self._worker_stubs:
                worker_info = self._workers[worker_id]
                address = worker_info.address
                # Add http:// prefix if missing
                if not address.startswith("http"):
                    address = f"http://{address}"
                self._worker_stubs[worker_id] = FrayWorkerClientSync(address)
            return self._worker_stubs[worker_id]

    def _assign_task_to_worker(self, task_id: str, worker_id: str) -> None:
        """
        Push a task to a worker using the AssignTask RPC.

        This method runs in the scheduler thread and handles errors by re-queueing
        the task if assignment fails.
        """
        # Get task info outside the lock
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning(f"Task {task_id} not found, skipping assignment")
                return

            # Mark task as running and assign to worker
            task.status = fray_pb2.TASK_STATUS_RUNNING
            task.worker_id = worker_id

            # Prepare assignment request
            assignment = fray_pb2.TaskAssignment(
                task_id=task.task_id,
                serialized_fn=task.serialized_fn,
                resources=task.resources,
                max_retries=task.max_retries,
                actor_id=task.actor_id or "",
            )

        # Push to worker outside the lock
        try:
            stub = self._get_or_create_worker_stub(worker_id)
            stub.assign_task(assignment, timeout_ms=TASK_ASSIGNMENT_TIMEOUT_MS)
            logger.debug(f"Assigned task {task_id} to worker {worker_id}")
        except Exception as e:
            logger.error(f"Failed to assign task {task_id} to worker {worker_id}: {e}")
            # Re-queue the task and mark as pending
            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None:
                    task.status = fray_pb2.TASK_STATUS_PENDING
                    task.worker_id = None
                    self._pending_queue.append(task_id)

    def _task_scheduler_loop(self) -> None:
        """
        Background thread that continuously assigns pending tasks to workers.

        This thread pops tasks from the pending queue and pushes them to workers,
        routing actor tasks to their hosting worker and regular tasks to the
        least-loaded worker.
        """
        logger.info("Task scheduler loop started")
        while self._running:
            task_id = None
            worker_id = None
            should_wait = False

            # Get next task from queue
            with self._condition:
                if self._pending_queue:
                    task_id = self._pending_queue.popleft()
                    task = self._tasks.get(task_id)

                    if task is None:
                        continue

                    # Determine target worker
                    if task.actor_id:
                        # Actor task - route to hosting worker
                        actor_info = self._actors.get(task.actor_id)
                        if actor_info:
                            worker_id = actor_info.worker_id
                        else:
                            # Actor not found - mark task as failed
                            task.status = fray_pb2.TASK_STATUS_FAILED
                            task.error = f"Actor {task.actor_id} not found"
                            logger.error(f"Actor {task.actor_id} not found for task {task_id}")
                            continue
                    else:
                        # Regular task - assign to least-loaded worker
                        if not self._workers:
                            # No workers available - re-queue and wait
                            self._pending_queue.append(task_id)
                            logger.debug("No workers available, waiting for worker registration")
                            should_wait = True
                            task_id = None  # Don't assign yet
                        else:
                            # Count tasks per worker
                            worker_loads = {w_id: 0 for w_id in self._workers.keys()}
                            for t in self._tasks.values():
                                if t.status == fray_pb2.TASK_STATUS_RUNNING and t.worker_id:
                                    worker_loads[t.worker_id] = worker_loads.get(t.worker_id, 0) + 1

                            worker_id = min(worker_loads.keys(), key=lambda w: worker_loads[w])

                # Wait for notification if no workers available
                if should_wait:
                    self._condition.wait(timeout=0.1)
                    continue

                # If no tasks, wait for new tasks or workers
                if not task_id:
                    self._condition.wait(timeout=0.01)
                    continue

            # Assign task outside the lock
            if task_id and worker_id:
                self._assign_task_to_worker(task_id, worker_id)

        logger.info("Task scheduler loop stopped")

    async def submit_task(self, request: fray_pb2.TaskSpec, ctx: RequestContext) -> fray_pb2.TaskHandle:
        """
        Client submits a new task for execution.

        Generates a unique task ID, stores the task as PENDING, and adds it to the queue.
        The scheduler thread will push it to a worker.
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
            # Notify scheduler that a new worker is available (helps with pending tasks)
            self._condition.notify_all()

        logger.info(f"Registered worker {request.worker_id} at {request.address}")
        return fray_pb2.Empty()

    async def report_task_result(self, request: fray_pb2.TaskResultPayload, ctx: RequestContext) -> fray_pb2.Empty:
        """
        Worker reports task completion or failure using unified result payload.

        Uses the oneof field in TaskResultPayload to distinguish between success and error.
        For errors, deserializes the exception to extract error message for backward compatibility.
        """
        with self._lock:
            task = self._tasks.get(request.task_id)
            if task is None:
                raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")

            # Check which field is set in the oneof
            if request.HasField("serialized_value"):
                task.status = fray_pb2.TASK_STATUS_COMPLETED
                task.result = request.serialized_value
            elif request.HasField("serialized_error"):
                task.status = fray_pb2.TASK_STATUS_FAILED
                task.serialized_error = request.serialized_error
                # Deserialize error to extract message for backward compatibility
                try:
                    exc = cloudpickle.loads(request.serialized_error)
                    task.error = f"{type(exc).__name__}: {exc!s}"
                except Exception:
                    task.error = "Failed to deserialize error"
            else:
                raise ConnectError(
                    Code.INVALID_ARGUMENT, "TaskResultPayload must have either serialized_value or serialized_error"
                )

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
                        # Get worker address for client to connect to
                        worker_info = self._workers.get(actor.worker_id)
                        worker_address = worker_info.address if worker_info else ""
                        return fray_pb2.ActorHandle(
                            actor_id=actor_id,
                            worker_id=worker_address,
                            name=request.name,
                            status=cast(fray_pb2.ActorStatus, actor.status),
                        )
                    else:
                        raise ConnectError(Code.ALREADY_EXISTS, f"Actor {request.name} already exists")

        # Generate actor ID
        actor_id = str(uuid.uuid4())

        # Store spec for restart
        with self._lock:
            if not self._workers:
                raise ConnectError(Code.UNAVAILABLE, "No workers available")

            self._actor_specs[actor_id] = request.serialized_actor

            # Create ActorInfo in CREATING state
            current_time = time.time()
            actor_info = ActorInfo(
                actor_id=actor_id,
                worker_id="",  # Will be set after successful placement
                name=request.name or None,
                status=fray_pb2.ACTOR_STATUS_CREATING,
                created_at=current_time,
                last_used=current_time,
            )
            self._actors[actor_id] = actor_info

            if request.name:
                self._named_actors[request.name] = actor_id

        # Retry loop with exponential backoff and worker re-selection
        actor_spec = fray_pb2.ActorSpec(
            actor_id=actor_id,
            serialized_actor=request.serialized_actor,
            name=request.name,
        )

        max_retries = 3
        final_worker_id = None
        for attempt in range(max_retries):
            try:
                # Re-pick worker on each retry for better load balancing
                with self._lock:
                    if not self._workers:
                        raise ConnectError(Code.UNAVAILABLE, "No workers available")

                    worker_id = min(
                        self._workers.keys(), key=lambda w: sum(1 for a in self._actors.values() if a.worker_id == w)
                    )
                    # Update actor info with selected worker
                    self._actors[actor_id].worker_id = worker_id

                # Create worker client and instantiate actor
                worker_client = self._create_worker_client(worker_id)
                async with worker_client:
                    await worker_client.instantiate_actor(actor_spec, timeout_ms=ACTOR_INSTANTIATION_TIMEOUT_MS)

                with self._lock:
                    self._actors[actor_id].status = fray_pb2.ACTOR_STATUS_READY
                final_worker_id = worker_id
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 100ms, 200ms, 400ms
                    backoff_ms = 100 * (2**attempt)
                    await asyncio.sleep(backoff_ms / 1000.0)
                else:
                    with self._lock:
                        self._actors[actor_id].status = fray_pb2.ACTOR_STATUS_FAILED
                    raise ConnectError(
                        Code.INTERNAL, f"Failed to instantiate actor after {max_retries} attempts: {e}"
                    ) from e

        # Get worker address for client to connect to
        if final_worker_id is None:
            raise ConnectError(Code.INTERNAL, "Actor instantiation succeeded but no worker ID was assigned")

        with self._lock:
            worker_info = self._workers.get(final_worker_id)
            worker_address = worker_info.address if worker_info else ""

        return fray_pb2.ActorHandle(
            actor_id=actor_id,
            worker_id=worker_address,
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
            # Add to pending queue - scheduler will push it to worker
            self._pending_queue.append(task_id)

        return fray_pb2.TaskHandle(
            task_id=task_id,
            status=fray_pb2.TASK_STATUS_PENDING,
        )

    async def get_actor_status(self, request: fray_pb2.ActorHandle, ctx: RequestContext) -> fray_pb2.ActorHandle:
        """Returns the worker location for an actor (not the status)."""
        with self._lock:
            actor = self._actors.get(request.actor_id)
            if actor is None:
                raise ConnectError(Code.NOT_FOUND, f"Actor {request.actor_id} not found")

            # Get worker address for client to connect to
            worker_info = self._workers.get(actor.worker_id)
            worker_address = worker_info.address if worker_info else ""

            return fray_pb2.ActorHandle(
                actor_id=actor.actor_id,
                worker_id=worker_address,
                name=actor.name or "",
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

    def start_scheduler(self) -> None:
        """Start the task scheduler thread."""
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._task_scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Task scheduler thread started")

    def stop_scheduler(self) -> None:
        """Stop the task scheduler thread."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2.0)
            logger.info("Task scheduler thread stopped")


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

        # Start the scheduler thread before server
        self._servicer.start_scheduler()

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
        """Stop the ASGI server and scheduler."""
        self._servicer.stop_health_checks()
        self._servicer.stop_scheduler()
        if self._server:
            self._server.should_exit = True

    @property
    def servicer(self) -> FrayControllerServicer:
        """Access the underlying servicer for testing or inspection."""
        return self._servicer
