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

"""Unified Controller class for managing all controller components.

Provides a single Controller class that encapsulates and manages the lifecycle
of all controller components:
- ControllerState: In-memory job and worker state
- Scheduler: Pure scheduling logic (shallow interface)
- WorkerHealthTracker: Worker health logic (shallow interface)
- ControllerServiceImpl: RPC service implementation
- ControllerDashboard: Web dashboard and HTTP server

The Controller owns the background thread that drives scheduling and heartbeat
loops, calling the shallow Scheduler and WorkerHealthTracker as needed.

This simplifies controller initialization and ensures consistent lifecycle
management across all components.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import uvicorn

from fluster.time_utils import wait_until

from fluster.cluster.controller.dashboard import ControllerDashboard
from fluster.cluster.controller.retry import handle_job_failure
from fluster.cluster.controller.scheduler import ScheduleResult, Scheduler
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import WorkerServiceClientSync

logger = logging.getLogger(__name__)


class WorkerClient(Protocol):
    """Protocol for worker RPC client.

    This matches the WorkerServiceClientSync signature (client-side).
    The server Protocol (WorkerServiceSync) has different signatures.
    """

    def run_job(
        self,
        request: cluster_pb2.Worker.RunJobRequest,
    ) -> cluster_pb2.Worker.RunJobResponse: ...

    def get_job_status(
        self,
        request: cluster_pb2.Worker.GetJobStatusRequest,
    ) -> cluster_pb2.JobStatus: ...

    def list_jobs(
        self,
        request: cluster_pb2.Worker.ListJobsRequest,
    ) -> cluster_pb2.Worker.ListJobsResponse: ...

    def health_check(
        self,
        request: cluster_pb2.Empty,
    ) -> cluster_pb2.Worker.HealthResponse: ...


class WorkerStubFactory(Protocol):
    """Factory for getting worker RPC stubs.

    This protocol allows injecting mock stubs for testing. In production,
    use DefaultWorkerStubFactory which creates real RPC clients.
    """

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
        """Create a real RPC client for the given address."""
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
        scheduler_interval_seconds: How often the scheduler checks for pending jobs (default: 0.5)
        worker_timeout_seconds: Time after which a worker without heartbeat is marked unhealthy (default: 60.0)
    """

    host: str = "127.0.0.1"
    port: int = 0
    bundle_dir: Path | None = None
    scheduler_interval_seconds: float = 0.5
    worker_timeout_seconds: float = 60.0


class Controller:
    """Unified controller managing all components and lifecycle.

    Encapsulates all controller components and provides a clean API for
    job submission, status queries, and worker registration. The controller
    handles all background threads and ensures proper cleanup on shutdown.

    Components managed:
    - ControllerState: Thread-safe state for jobs, workers, and endpoints
    - Scheduler: Pure job-to-worker matching (shallow interface)
    - ControllerServiceImpl: RPC service implementation
    - ControllerDashboard: Web dashboard and HTTP server

    The Controller owns a single background loop that periodically:
    1. Runs the scheduler to find job assignments
    2. Dispatches assigned jobs to workers
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
        worker_stub_factory: Factory for creating worker RPC stubs. Use
            DefaultWorkerStubFactory for production or inject a mock for testing.
    """

    def __init__(
        self,
        config: ControllerConfig,
        worker_stub_factory: WorkerStubFactory,
    ):
        """Initialize controller components.

        Args:
            config: Controller configuration
            worker_stub_factory: Factory for creating worker RPC stubs
        """
        self._config = config
        self._stub_factory = worker_stub_factory

        # Initialize state first
        self._state = ControllerState()

        # Scheduler: shallow interface, no threads, no callbacks
        self._scheduler = Scheduler(self._state)

        # Service and dashboard
        self._service = ControllerServiceImpl(
            self._state,
            self,  # Controller implements the scheduling wake interface
            bundle_dir=config.bundle_dir,
        )
        self._dashboard = ControllerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        # Background loop state
        self._stop = False
        self._wake_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

    def wake(self) -> None:
        """Signal the controller loop to run immediately.

        Called when events occur that may make scheduling possible:
        - New job submitted
        - New worker registered
        - Job finished (freeing capacity)
        """
        self._wake_event.set()

    def start(self) -> None:
        """Start all background components.

        Starts the main controller loop and dashboard server.
        Both run in background daemon threads.
        """
        self._stop = False

        # Start main controller loop
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )
        self._loop_thread.start()

        # Start dashboard server in background thread
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self._server_thread.start()

        # Wait for server startup with exponential backoff
        wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=5.0,
            initial_interval=0.05,
            max_interval=0.5,
        )

    def stop(self) -> None:
        """Stop all background components gracefully.

        Signals the loop to stop, wakes it, and waits for termination.
        The dashboard server stops automatically when the daemon thread exits.
        """
        self._stop = True
        self._wake_event.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)

    def _run_loop(self) -> None:
        """Main controller loop.

        Runs scheduling and worker timeout checks. Uses an event for wake
        signaling to allow immediate scheduling after job submissions or
        worker registrations.
        """
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
        """Run one scheduling cycle.

        Gets pending jobs and available workers, calls the scheduler to
        find assignments, then dispatches each assignment.
        """
        now_ms = int(time.time() * 1000)
        pending_jobs = self._state.peek_pending_jobs()
        workers = self._state.get_available_workers()

        if not pending_jobs:
            return

        result = self._scheduler.find_assignments(pending_jobs, workers, now_ms)
        self._apply_schedule_result(result, now_ms)

    def _apply_schedule_result(self, result: ScheduleResult, now_ms: int) -> None:
        """Apply scheduling results: dispatch jobs and handle timeouts.

        Args:
            result: ScheduleResult from scheduler
            now_ms: Current timestamp in milliseconds
        """
        for job in result.timed_out_jobs:
            self._mark_job_unschedulable(job, now_ms)

        for job, worker in result.assignments:
            self._dispatch_job(job, worker, now_ms)

    def _dispatch_job(self, job: ControllerJob, worker: ControllerWorker, now_ms: int) -> None:
        """Dispatch a job to a worker via RPC.

        All state is set BEFORE the RPC call. The worker reports all state
        transitions (BUILDING, RUNNING, SUCCEEDED, etc.) via ReportJobState.

        On failure: resets state and marks worker unhealthy.
        """
        # Set all state BEFORE RPC - worker may complete before RPC returns
        job.worker_id = worker.worker_id
        job.started_at_ms = now_ms
        worker.running_jobs.add(job.job_id)
        self._state.remove_from_queue(job.job_id)

        try:
            stub = self._stub_factory.get_stub(worker.address)
            request = cluster_pb2.Worker.RunJobRequest(
                job_id=str(job.job_id),
                serialized_entrypoint=job.request.serialized_entrypoint,
                environment=cluster_pb2.EnvironmentConfig(
                    workspace=job.request.environment.workspace,
                    env_vars=dict(job.request.environment.env_vars),
                ),
                bundle_gcs_path=job.request.bundle_gcs_path,
                resources=cluster_pb2.ResourceSpec(
                    cpu=job.request.resources.cpu,
                    memory=job.request.resources.memory,
                ),
                ports=list(job.request.ports),
            )
            stub.run_job(request)

            logger.info(f"Dispatched job {job.job_id} to worker {worker.worker_id}")
            self._state.log_action(
                "job_dispatched",
                job_id=job.job_id,
                worker_id=worker.worker_id,
            )

        except Exception:
            # Failure: reset all state changes
            job.worker_id = None
            job.started_at_ms = None
            job.state = cluster_pb2.JOB_STATE_PENDING
            worker.running_jobs.discard(job.job_id)
            self._state.add_to_queue(job)  # Re-add to queue for retry
            worker.healthy = False
            logger.exception("Failed to dispatch job %s to worker %s", job.job_id, worker.address)
            self._state.log_action(
                "dispatch_failed",
                job_id=job.job_id,
                worker_id=worker.worker_id,
            )

    def _mark_job_unschedulable(self, job: ControllerJob, now_ms: int) -> None:
        """Mark job as unschedulable and remove from queue."""
        logger.warning(
            f"Job {job.job_id} exceeded scheduling timeout "
            f"({job.request.scheduling_timeout_seconds}s), marking as UNSCHEDULABLE"
        )
        job.state = cluster_pb2.JOB_STATE_UNSCHEDULABLE
        job.finished_at_ms = now_ms
        job.error = f"Scheduling timeout exceeded ({job.request.scheduling_timeout_seconds}s)"
        self._state.remove_from_queue(job.job_id)
        self._state.log_action(
            "job_unschedulable",
            job_id=job.job_id,
            details=f"timeout={job.request.scheduling_timeout_seconds}s",
        )

    def _check_worker_timeouts(self) -> None:
        """Mark workers as unhealthy if they haven't sent heartbeat recently.

        Workers send periodic heartbeats to the controller. If a worker hasn't
        sent a heartbeat within worker_timeout_seconds, it's marked unhealthy
        and its running jobs are failed for retry.
        """
        now_ms = int(time.time() * 1000)
        timeout_ms = int(self._config.worker_timeout_seconds * 1000)

        for worker in self._state.list_all_workers():
            if worker.healthy and (now_ms - worker.last_heartbeat_ms) > timeout_ms:
                worker.healthy = False
                logger.warning(
                    f"Worker {worker.worker_id} timed out (no heartbeat for {self._config.worker_timeout_seconds}s)"
                )
                self._state.log_action(
                    "worker_timeout",
                    worker_id=worker.worker_id,
                    details=f"No heartbeat for {self._config.worker_timeout_seconds}s",
                )

                # Retry jobs that were running on the timed-out worker
                for job_id in list(worker.running_jobs):
                    handle_job_failure(self._state, job_id, is_worker_failure=True)

    def _run_server(self) -> None:
        """Run dashboard server (blocking, for thread)."""
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
            print(f"Controller server error: {e}")

    # Delegate key service methods

    def launch_job(
        self,
        request: cluster_pb2.Controller.LaunchJobRequest,
    ) -> cluster_pb2.Controller.LaunchJobResponse:
        """Submit a job to the controller.

        Creates a new job, adds it to the queue, and wakes the scheduler
        to attempt immediate dispatch.

        Args:
            request: Job launch request with entrypoint and resources

        Returns:
            LaunchJobResponse containing the assigned job_id
        """
        return self._service.launch_job(request, None)

    def get_job_status(
        self,
        job_id: str,
    ) -> cluster_pb2.Controller.GetJobStatusResponse:
        """Get the status of a job.

        Args:
            job_id: Job identifier

        Returns:
            GetJobStatusResponse with current job status
        """
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        return self._service.get_job_status(request, None)

    def register_worker(
        self,
        request: cluster_pb2.Controller.RegisterWorkerRequest,
    ) -> cluster_pb2.Controller.RegisterWorkerResponse:
        """Register a worker with the controller.

        Adds the worker to the registry and wakes the scheduler to
        potentially dispatch pending jobs.

        Args:
            request: Worker registration request

        Returns:
            RegisterWorkerResponse with acceptance status
        """
        return self._service.register_worker(request, None)

    def terminate_job(
        self,
        job_id: str,
    ) -> cluster_pb2.Empty:
        """Terminate a running job.

        Marks the job as killed in the controller state.

        Args:
            job_id: Job identifier

        Returns:
            Empty response
        """
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        return self._service.terminate_job(request, None)

    # Properties

    @property
    def state(self) -> ControllerState:
        """Access to controller state (for advanced usage).

        Returns:
            The controller's internal state
        """
        return self._state

    @property
    def url(self) -> str:
        """Controller URL.

        Returns:
            HTTP URL for the controller dashboard and RPC service
        """
        return f"http://{self._config.host}:{self._config.port}"
