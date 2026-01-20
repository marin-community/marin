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

"""Unified controller managing background scheduling and worker health checks."""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import uvicorn

from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.scheduler import Scheduler, SchedulingResult
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.events import Event, EventType
from iris.cluster.controller.state import ControllerState, ControllerTask, ControllerWorker
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.time_utils import ExponentialBackoff, now_ms

logger = logging.getLogger(__name__)


class WorkerClient(Protocol):
    """Protocol for worker RPC client.

    Matches client-side WorkerServiceClientSync signature. The server Protocol has different signatures.
    """

    def run_task(
        self,
        request: cluster_pb2.Worker.RunTaskRequest,
    ) -> cluster_pb2.Worker.RunTaskResponse: ...

    def get_task_status(
        self,
        request: cluster_pb2.Worker.GetTaskStatusRequest,
    ) -> cluster_pb2.TaskStatus: ...

    def list_tasks(
        self,
        request: cluster_pb2.Worker.ListTasksRequest,
    ) -> cluster_pb2.Worker.ListTasksResponse: ...

    def health_check(
        self,
        request: cluster_pb2.Empty,
    ) -> cluster_pb2.Worker.HealthResponse: ...


class WorkerStubFactory(Protocol):
    """Factory for getting worker RPC stubs. Allows injecting mock stubs for testing."""

    def get_stub(self, address: str) -> WorkerClient:
        """Get a worker stub for the given address.

        Args:
            address: Worker address in "host:port" format

        Returns:
            A WorkerClient stub for making RPC calls
        """
        ...


class DefaultWorkerStubFactory:
    """Default factory that creates real RPC client stubs."""

    def get_stub(self, address: str) -> WorkerClient:
        return WorkerServiceClientSync(
            address=f"http://{address}",
            timeout_ms=10000,
        )


@dataclass
class ControllerConfig:
    """Controller configuration.

    Args:
        host: Host to bind the HTTP server to (default: "127.0.0.1")
        port: Port to bind the HTTP server to (default: 0 for auto-assign)
        bundle_dir: Directory for storing uploaded job bundles (optional)
        scheduler_interval_seconds: How often the scheduler checks for pending tasks (default: 0.5)
        worker_timeout_seconds: Time after which a worker without heartbeat is marked unhealthy (default: 60.0)
    """

    host: str = "127.0.0.1"
    port: int = 0
    bundle_dir: Path | None = None
    scheduler_interval_seconds: float = 0.5
    worker_timeout_seconds: float = 60.0


class Controller:
    """Unified controller managing all components and lifecycle.

    Owns a single background loop that periodically:
    1. Runs the scheduler to find task assignments
    2. Dispatches assigned tasks to workers
    3. Checks worker health via heartbeats
    4. Applies state changes from heartbeat results

    Example:
        ```python
        config = ControllerConfig(port=8080)
        controller = Controller(
            config=config,
            worker_stub_factory=DefaultWorkerStubFactory(),
        )
        controller.start()
        try:
            job_id = controller.launch_job(request)
            status = controller.get_job_status(job_id)
        finally:
            controller.stop()
        ```

    Args:
        config: Controller configuration
        worker_stub_factory: Factory for creating worker RPC stubs
    """

    def __init__(
        self,
        config: ControllerConfig,
        worker_stub_factory: WorkerStubFactory,
    ):
        self._config = config
        self._stub_factory = worker_stub_factory

        self._state = ControllerState()
        self._scheduler = Scheduler(self._state)
        self._service = ControllerServiceImpl(self._state, self, bundle_dir=config.bundle_dir)
        self._dashboard = ControllerDashboard(self._service, self._scheduler, host=config.host, port=config.port)

        # Background loop state
        self._stop = False
        self._wake_event = threading.Event()
        self._scheduling_loop_thread: threading.Thread | None = None
        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

    def wake(self) -> None:
        """Signal the controller loop to run immediately.

        Called when events occur that may make scheduling possible:
        - New job submitted
        - New worker registered
        - Task finished (freeing capacity)
        """
        self._wake_event.set()

    def start(self) -> None:
        """Start main controller loop and dashboard server in background daemon threads."""
        self._stop = False

        # Start main controller loop
        self._scheduling_loop_thread = threading.Thread(
            target=self._run_scheduling_loop,
            daemon=True,
        )
        self._scheduling_loop_thread.start()

        # Start dashboard server in background thread
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self._server_thread.start()

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=5.0,
        )

    def stop(self) -> None:
        """Stop all background components gracefully."""
        self._stop = True
        self._wake_event.set()
        if self._scheduling_loop_thread:
            self._scheduling_loop_thread.join(timeout=5.0)

        # Stop uvicorn server
        if self._server:
            self._server.should_exit = True
        if self._server_thread:
            self._server_thread.join(timeout=5.0)

    def _run_scheduling_loop(self) -> None:
        """Main controller loop running scheduling and worker timeout checks."""
        while not self._stop:
            # Wait for wake signal or timeout (use scheduler interval)
            self._wake_event.wait(timeout=self._config.scheduler_interval_seconds)
            self._wake_event.clear()

            if self._stop:
                break

            # Run scheduling
            self._run_scheduling()

            # Check for timed-out workers
            self._check_worker_timeouts()

    def _run_scheduling(self) -> None:
        """Run one scheduling cycle."""
        pending_tasks = self._state.peek_pending_tasks()
        workers = self._state.get_available_workers()

        if not pending_tasks:
            return

        result = self._scheduler.find_assignments(pending_tasks, workers)
        self._apply_schedule_result(result)

    def _apply_schedule_result(self, result: SchedulingResult) -> None:
        """Apply scheduling results: dispatch tasks and handle timeouts."""
        for task in result.timed_out_tasks:
            self._mark_task_unschedulable(task)

        for task, worker in result.assignments:
            self._dispatch_task(task, worker)

    def _dispatch_task(
        self,
        task: ControllerTask,
        worker: ControllerWorker,
    ) -> None:
        """Dispatch a task to a worker via RPC.

        Creates attempt before RPC so worker reports are always valid:
        1. Fire TASK_ASSIGNED event (commits resources + creates attempt)
        2. Send RPC with valid attempt_id
        3. On failure: revert attempt and mark worker failed
        """
        job = self._state.get_job(task.job_id)
        if not job:
            return

        # Create attempt BEFORE RPC so worker state reports have a valid attempt_id
        self._state.handle_event(
            Event(
                event_type=EventType.TASK_ASSIGNED,
                task_id=task.task_id,
                worker_id=worker.worker_id,
            )
        )

        try:
            stub = self._stub_factory.get_stub(worker.address)
            request = cluster_pb2.Worker.RunTaskRequest(
                job_id=str(task.job_id),
                task_id=str(task.task_id),
                task_index=task.task_index,
                num_tasks=len(self._state.get_job_tasks(task.job_id)),
                serialized_entrypoint=job.request.serialized_entrypoint,
                environment=cluster_pb2.EnvironmentConfig(
                    workspace=job.request.environment.workspace,
                    env_vars=dict(job.request.environment.env_vars),
                ),
                bundle_gcs_path=job.request.bundle_gcs_path,
                resources=cluster_pb2.ResourceSpecProto(
                    cpu=job.request.resources.cpu,
                    memory_bytes=job.request.resources.memory_bytes,
                ),
                ports=list(job.request.ports),
                attempt_id=task.current_attempt_id,
            )

            logger.info(
                f"Dispatching task {task.task_id} to worker {worker.worker_id} at {worker.address} "
                f"(cpu={job.request.resources.cpu}, memory={job.request.resources.memory_bytes})"
            )
            stub.run_task(request)

            logger.info(f"Successfully dispatched task {task.task_id} to worker {worker.worker_id}")

        except Exception as e:
            # FAILURE: Revert attempt and mark worker failed
            task.revert_attempt()
            self._state.handle_event(
                Event(
                    event_type=EventType.WORKER_FAILED,
                    worker_id=worker.worker_id,
                    error=f"Dispatch RPC failed: {e}",
                )
            )
            logger.exception("Failed to dispatch task %s to worker %s", task.task_id, worker.address)

    def _mark_task_unschedulable(self, task: ControllerTask) -> None:
        """Mark a task as unschedulable due to timeout."""
        job = self._state.get_job(task.job_id)
        timeout_seconds = job.request.scheduling_timeout_seconds if job else 0
        logger.warning(f"Task {task.task_id} exceeded scheduling timeout ({timeout_seconds}s), marking as UNSCHEDULABLE")
        self._state.handle_event(
            Event(
                EventType.TASK_STATE_CHANGED,
                task_id=task.task_id,
                new_state=cluster_pb2.TASK_STATE_UNSCHEDULABLE,
                error=f"Scheduling timeout exceeded ({timeout_seconds}s)",
            )
        )

    def _check_worker_timeouts(self) -> None:
        """Mark workers as unhealthy if they haven't sent heartbeat within worker_timeout_seconds."""
        current_time_ms = now_ms()
        timeout_ms = int(self._config.worker_timeout_seconds * 1000)

        for worker in self._state.list_all_workers():
            if worker.healthy and (current_time_ms - worker.last_heartbeat_ms) > timeout_ms:
                logger.warning(
                    f"Worker {worker.worker_id} timed out (no heartbeat for {self._config.worker_timeout_seconds}s)"
                )
                # Use WORKER_FAILED event which will cascade to running tasks
                self._state.handle_event(
                    Event(
                        EventType.WORKER_FAILED,
                        worker_id=worker.worker_id,
                        error=f"Worker {worker.worker_id} timed out",
                    )
                )

    def _run_server(self) -> None:
        try:
            config = uvicorn.Config(
                self._dashboard._app,
                host=self._config.host,
                port=self._config.port,
                log_level="error",
            )
            self._server = uvicorn.Server(config)
            self._server.run()
        except Exception as e:
            logger.exception("Controller server error: %s", e)

    def launch_job(
        self,
        request: cluster_pb2.Controller.LaunchJobRequest,
    ) -> cluster_pb2.Controller.LaunchJobResponse:
        """Submit a job to the controller."""
        return self._service.launch_job(request, None)

    def get_job_status(
        self,
        job_id: str,
    ) -> cluster_pb2.Controller.GetJobStatusResponse:
        """Get the status of a job."""
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        return self._service.get_job_status(request, None)

    def register_worker(
        self,
        request: cluster_pb2.Controller.RegisterWorkerRequest,
    ) -> cluster_pb2.Controller.RegisterWorkerResponse:
        """Register a worker with the controller."""
        return self._service.register_worker(request, None)

    def terminate_job(
        self,
        job_id: str,
    ) -> cluster_pb2.Empty:
        """Terminate a running job."""
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        return self._service.terminate_job(request, None)

    # Properties

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def port(self) -> int:
        """Actual bound port (may differ from config if port=0 was specified)."""
        if self._server and self._server.servers:
            # Get actual port from the first server socket
            sockets = self._server.servers[0].sockets
            if sockets:
                return sockets[0].getsockname()[1]
        return self._config.port

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self.port}"
