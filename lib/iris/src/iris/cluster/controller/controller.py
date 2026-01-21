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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import uvicorn

from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.scheduler import Scheduler, SchedulingResult
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.events import (
    TaskAssignedEvent,
    TaskStateChangedEvent,
    WorkerFailedEvent,
)
from iris.cluster.controller.state import ControllerJob, ControllerState, ControllerTask, ControllerWorker
from iris.cluster.types import JobId, TaskId
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync
from iris.time_utils import ExponentialBackoff, now_ms

logger = logging.getLogger(__name__)

# RPC timeout for dispatch - worker should respond immediately since execution is async
DISPATCH_RPC_TIMEOUT_SECONDS = 5.0


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

    def kill_task(
        self,
        request: cluster_pb2.Worker.KillTaskRequest,
    ) -> cluster_pb2.Empty: ...


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
        max_dispatch_parallelism: Maximum concurrent dispatch RPCs (default: 32)
    """

    host: str = "127.0.0.1"
    port: int = 0
    bundle_dir: Path | None = None
    scheduler_interval_seconds: float = 0.5
    worker_timeout_seconds: float = 60.0
    max_dispatch_parallelism: int = 32


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
        # Prevents concurrent scheduling cycles from double-assigning the same task.
        # The entire peek → assign → dispatch sequence runs under this lock.
        self._scheduler_lock = threading.Lock()
        self._scheduling_loop_thread: threading.Thread | None = None
        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

        # Thread pool for parallel task dispatch
        self._dispatch_executor = ThreadPoolExecutor(
            max_workers=config.max_dispatch_parallelism,
            thread_name_prefix="dispatch",
        )

    def wake(self) -> None:
        """Signal the controller loop to run immediately.

        Called when events occur that may make scheduling possible:
        - New job submitted
        - New worker registered
        - Task finished (freeing capacity)
        """
        self._wake_event.set()

    def kill_tasks_on_workers(self, task_ids: set[TaskId]) -> None:
        """Send KILL RPCs to workers for killed tasks.

        Called by the service after state transitions that kill tasks.
        Delegates to _kill_tasks_on_workers which has access to the stub factory.
        """
        if task_ids:
            self._kill_tasks_on_workers(task_ids)

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

        # Shutdown dispatch executor
        self._dispatch_executor.shutdown(wait=True, cancel_futures=True)

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
        with self._scheduler_lock:
            pending_tasks = self._state.peek_pending_tasks()
            workers = self._state.get_available_workers()

            if not pending_tasks:
                return

            result = self._scheduler.find_assignments(pending_tasks, workers)
            self._apply_schedule_result(result)

    def _apply_schedule_result(self, result: SchedulingResult) -> None:
        """Apply scheduling results: dispatch tasks and handle timeouts.

        Groups assignments by job for coscheduled handling. All tasks are
        dispatched in parallel for performance, with resource commits
        happening synchronously before RPC dispatch.
        """
        for task in result.timed_out_tasks:
            self._mark_task_unschedulable(task)

        if not result.assignments:
            return

        # Group assignments by job for coscheduled handling
        by_job: dict[JobId, list[tuple[ControllerTask, ControllerWorker]]] = defaultdict(list)
        for task, worker in result.assignments:
            by_job[task.job_id].append((task, worker))

        for job_id, job_assignments in by_job.items():
            job = self._state.get_job(job_id)
            if job is None:
                continue

            if job.is_coscheduled:
                self._dispatch_coscheduled_group(job_assignments, job)
            else:
                self._dispatch_tasks_parallel(job_assignments)

    def _dispatch_tasks_parallel(
        self,
        assignments: list[tuple[ControllerTask, ControllerWorker]],
    ) -> None:
        """Dispatch multiple independent tasks in parallel.

        Flow:
        1. Commit all resources synchronously (fire events sequentially)
        2. Send all RPCs in parallel with timeout
        3. Handle failures by reverting attempt and marking worker failed
        """
        if not assignments:
            return

        # Phase 1: Commit all resources synchronously and build RPC requests
        committed: list[tuple[ControllerTask, ControllerWorker, cluster_pb2.Worker.RunTaskRequest]] = []
        for task, worker in assignments:
            job = self._state.get_job(task.job_id)
            if not job:
                continue

            # Create attempt BEFORE RPC so worker state reports have a valid attempt_id
            self._state.handle_event(
                TaskAssignedEvent(
                    task_id=task.task_id,
                    worker_id=worker.worker_id,
                )
            )

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
            committed.append((task, worker, request))

        if not committed:
            return

        # Phase 2: Send RPCs in parallel
        futures = {
            self._dispatch_executor.submit(self._send_run_task_rpc, task, worker, request): (task, worker)
            for task, worker, request in committed
        }

        # Phase 3: Collect results and handle failures
        failed: list[tuple[ControllerTask, ControllerWorker, str]] = []
        try:
            for future in as_completed(futures, timeout=DISPATCH_RPC_TIMEOUT_SECONDS + 1):
                task, worker = futures[future]
                try:
                    future.result(timeout=0)
                except Exception as e:
                    failed.append((task, worker, str(e)))
        except TimeoutError:
            # Some futures didn't complete in time - check all futures
            for future, (task, worker) in futures.items():
                if not future.done():
                    failed.append((task, worker, "Dispatch RPC timed out"))
                else:
                    # Future completed - check if it raised an exception we missed
                    exc = future.exception()
                    if exc is not None:
                        failed.append((task, worker, str(exc)))

        # Phase 4: Handle failures sequentially (fire events)
        # WorkerFailedEvent will cascade to all tasks on the worker, transitioning them
        # to WORKER_FAILED state. Don't call revert_attempt() - the event handler manages
        # the full task lifecycle.
        all_tasks_to_kill: set[TaskId] = set()
        for task, worker, error in failed:
            logger.warning(f"Dispatch failed for {task.task_id}: {error}")
            txn = self._state.handle_event(
                WorkerFailedEvent(
                    worker_id=worker.worker_id,
                    error=f"Dispatch RPC failed: {error}",
                )
            )
            all_tasks_to_kill.update(txn.tasks_to_kill)

        if all_tasks_to_kill:
            self._kill_tasks_on_workers(all_tasks_to_kill)

    def _dispatch_coscheduled_group(
        self,
        assignments: list[tuple[ControllerTask, ControllerWorker]],
        job: ControllerJob,
    ) -> None:
        """Dispatch all tasks in a coscheduled group in parallel.

        Commits all resources first, then sends all RPCs in parallel.
        On any RPC failure, releases resources for failed tasks.
        Successfully dispatched tasks continue running.
        """
        if not assignments:
            return

        # Phase 1: Commit all resources synchronously and build RPC requests
        committed: list[tuple[ControllerTask, ControllerWorker, cluster_pb2.Worker.RunTaskRequest]] = []
        for task, worker in assignments:
            self._state.handle_event(
                TaskAssignedEvent(
                    task_id=task.task_id,
                    worker_id=worker.worker_id,
                )
            )

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
            committed.append((task, worker, request))

        # Phase 2: Send all RPCs in parallel
        futures = {
            self._dispatch_executor.submit(self._send_run_task_rpc, task, worker, request): (task, worker)
            for task, worker, request in committed
        }

        # Phase 3: Collect results
        failed: list[tuple[ControllerTask, ControllerWorker, str]] = []
        try:
            for future in as_completed(futures, timeout=DISPATCH_RPC_TIMEOUT_SECONDS + 1):
                task, worker = futures[future]
                try:
                    future.result(timeout=0)
                except Exception as e:
                    failed.append((task, worker, str(e)))
        except TimeoutError:
            # Some futures didn't complete in time - check all futures
            for future, (task, worker) in futures.items():
                if not future.done():
                    failed.append((task, worker, "Dispatch RPC timed out"))
                else:
                    # Future completed - check if it raised an exception we missed
                    exc = future.exception()
                    if exc is not None:
                        failed.append((task, worker, str(exc)))

        # Phase 4: Handle failures
        # For coscheduled jobs, if ANY task fails to dispatch, we have a problem.
        # The successfully dispatched tasks will start running, but the group is incomplete.
        # WorkerFailedEvent will cascade to tasks, marking them WORKER_FAILED. The running
        # tasks will eventually fail due to missing peers (e.g., collective ops will timeout).
        all_tasks_to_kill: set[TaskId] = set()
        for task, worker, error in failed:
            logger.warning(f"Coscheduled dispatch failed for {task.task_id}: {error}")
            txn = self._state.handle_event(
                WorkerFailedEvent(
                    worker_id=worker.worker_id,
                    error=f"Dispatch RPC failed: {error}",
                )
            )
            all_tasks_to_kill.update(txn.tasks_to_kill)

        if all_tasks_to_kill:
            self._kill_tasks_on_workers(all_tasks_to_kill)

    def _send_run_task_rpc(
        self,
        task: ControllerTask,
        worker: ControllerWorker,
        request: cluster_pb2.Worker.RunTaskRequest,
    ) -> None:
        """Send run_task RPC with timeout. Called from thread pool."""
        logger.info(f"Dispatching task {task.task_id} to worker {worker.worker_id} at {worker.address}")
        stub = self._stub_factory.get_stub(worker.address)
        stub.run_task(request)
        logger.info(f"Successfully dispatched task {task.task_id} to worker {worker.worker_id}")

    def _mark_task_unschedulable(self, task: ControllerTask) -> None:
        """Mark a task as unschedulable due to timeout."""
        job = self._state.get_job(task.job_id)
        timeout_seconds = job.request.scheduling_timeout_seconds if job else 0
        logger.warning(f"Task {task.task_id} exceeded scheduling timeout ({timeout_seconds}s), marking as UNSCHEDULABLE")
        txn = self._state.handle_event(
            TaskStateChangedEvent(
                task_id=task.task_id,
                new_state=cluster_pb2.TASK_STATE_UNSCHEDULABLE,
                error=f"Scheduling timeout exceeded ({timeout_seconds}s)",
            )
        )
        if txn.tasks_to_kill:
            self._kill_tasks_on_workers(txn.tasks_to_kill)

    def _kill_tasks_on_workers(self, task_ids: set[TaskId]) -> None:
        """Send KILL RPCs to workers for tasks that were running.

        Called after state has marked tasks as killed. For each task that had
        a worker assigned, sends a kill RPC to that worker to terminate the
        task process.

        Kill RPCs are fire-and-forget: we log failures but don't retry since
        the task is already marked as killed in our state.
        """
        for task_id in task_ids:
            task = self._state.get_task(task_id)
            if not task:
                continue

            worker_id = task.worker_id
            if not worker_id:
                continue

            worker = self._state.get_worker(worker_id)
            if not worker:
                continue

            try:
                stub = self._stub_factory.get_stub(worker.address)
                request = cluster_pb2.Worker.KillTaskRequest(
                    task_id=str(task_id),
                    term_timeout_ms=5000,
                )
                logger.info(f"Sending kill RPC for task {task_id} to worker {worker_id}")
                stub.kill_task(request)
                logger.info(f"Successfully sent kill RPC for task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to send kill RPC for task {task_id} to worker {worker_id}: {e}")

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
                txn = self._state.handle_event(
                    WorkerFailedEvent(
                        worker_id=worker.worker_id,
                        error=f"Worker {worker.worker_id} timed out",
                    )
                )
                # Send kill RPCs for any other tasks that were killed as a result
                if txn.tasks_to_kill:
                    self._kill_tasks_on_workers(txn.tasks_to_kill)

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
