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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import uvicorn

from fluster import cluster_pb2
from fluster.cluster.controller.dashboard import ControllerDashboard
from fluster.cluster.controller.retry import handle_job_failure
from fluster.cluster.controller.scheduler import ScheduleResult, Scheduler
from fluster.cluster.types import JobId
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster_connect import WorkerServiceClientSync

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


MAX_CONSECUTIVE_HEARTBEAT_FAILURES = 3


@dataclass
class JobStateUpdate:
    """A job state change from heartbeat response."""

    job_id: JobId
    new_state: int  # JobState enum value
    exit_code: int = 0
    error: str | None = None
    finished_at_ms: int = 0


@dataclass
class HeartbeatResult:
    """Result of processing a single heartbeat.

    Contains all information to update state after a heartbeat check.
    """

    worker_id: str
    success: bool
    consecutive_failures: int = 0
    worker_failed: bool = False
    failed_job_ids: list[JobId] = field(default_factory=list)
    job_updates: list[JobStateUpdate] = field(default_factory=list)


@dataclass
class ControllerConfig:
    """Controller configuration.

    Args:
        host: Host to bind the HTTP server to (default: "127.0.0.1")
        port: Port to bind the HTTP server to (default: 0 for auto-assign)
        bundle_dir: Directory for storing uploaded job bundles (optional)
        scheduler_interval_seconds: How often the scheduler checks for pending jobs (default: 0.5)
        heartbeat_interval_seconds: How often to check worker health (default: 2.0)
    """

    host: str = "127.0.0.1"
    port: int = 0
    bundle_dir: Path | None = None
    scheduler_interval_seconds: float = 0.5
    heartbeat_interval_seconds: float = 2.0


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

        # Track timing for scheduler and heartbeat
        self._last_heartbeat_time = 0.0

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

        # Wait for server startup
        time.sleep(1.0)

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

        Runs scheduling and heartbeat checks on configured intervals.
        Uses an event for wake signaling to allow immediate scheduling
        after job submissions or worker registrations.
        """
        while not self._stop:
            # Wait for wake signal or timeout (use scheduler interval)
            self._wake_event.wait(timeout=self._config.scheduler_interval_seconds)
            self._wake_event.clear()

            if self._stop:
                break

            # Run scheduling
            self._run_scheduling()

            # Check if heartbeats are due
            now = time.time()
            if now - self._last_heartbeat_time >= self._config.heartbeat_interval_seconds:
                self._run_heartbeats()
                self._last_heartbeat_time = now

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
        # Handle timed-out jobs
        for job in result.timed_out_jobs:
            self._mark_job_unschedulable(job, now_ms)

        # Dispatch assignments
        for job, worker in result.assignments:
            success = self._dispatch_job(job, worker)
            if success:
                self._handle_successful_dispatch(job, worker, now_ms)
            else:
                self._handle_failed_dispatch(job, worker)

    def _dispatch_job(self, job: ControllerJob, worker: ControllerWorker) -> bool:
        """Dispatch a job to a worker via RPC.

        Args:
            job: Job to dispatch
            worker: Worker to dispatch to

        Returns:
            True if dispatch succeeded, False on failure
        """
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
            return True
        except Exception:
            return False

    def _handle_successful_dispatch(self, job: ControllerJob, worker: ControllerWorker, now_ms: int) -> None:
        """Update state after successful dispatch."""
        job.state = cluster_pb2.JOB_STATE_RUNNING
        job.worker_id = worker.worker_id
        job.started_at_ms = now_ms

        worker.running_jobs.add(job.job_id)
        self._state.remove_from_queue(job.job_id)

        logger.info(f"Dispatched job {job.job_id} to worker {worker.worker_id}")
        self._state.log_action(
            "job_dispatched",
            job_id=job.job_id,
            worker_id=worker.worker_id,
        )

    def _handle_failed_dispatch(self, job: ControllerJob, worker: ControllerWorker) -> None:
        """Handle dispatch failure - mark worker unhealthy, keep job in queue."""
        worker.healthy = False
        logger.warning(f"Failed to dispatch job {job.job_id} to {worker.worker_id}, " "marking worker unhealthy")
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

    def _run_heartbeats(self) -> None:
        """Run heartbeat checks for all workers."""
        workers = self._state.list_all_workers()
        now_ms = int(time.time() * 1000)

        for worker in workers:
            if not worker.healthy:
                continue

            response = self._send_heartbeat(worker.address)
            result = self._process_heartbeat(worker, response)
            self._apply_heartbeat_result(worker, result, now_ms)

    def _send_heartbeat(self, address: str) -> cluster_pb2.Worker.ListJobsResponse | None:
        """Send heartbeat to a worker and get job status.

        Args:
            address: Worker address in "host:port" format

        Returns:
            ListJobsResponse with job statuses, or None on failure
        """
        try:
            stub = self._stub_factory.get_stub(address)
            stub.health_check(cluster_pb2.Empty())

            jobs_response = stub.list_jobs(cluster_pb2.Worker.ListJobsRequest())
            return jobs_response
        except Exception:
            return None

    def _process_heartbeat(
        self,
        worker: ControllerWorker,
        response: cluster_pb2.Worker.ListJobsResponse | None,
    ) -> HeartbeatResult:
        """Process heartbeat response and return what changed."""
        if response is None:
            new_failure_count = worker.consecutive_failures + 1
            worker_failed = new_failure_count >= MAX_CONSECUTIVE_HEARTBEAT_FAILURES

            return HeartbeatResult(
                worker_id=worker.worker_id,
                success=False,
                consecutive_failures=new_failure_count,
                worker_failed=worker_failed,
                failed_job_ids=list(worker.running_jobs) if worker_failed else [],
            )

        job_updates = []
        for status in response.jobs:
            if status.state in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                job_updates.append(
                    JobStateUpdate(
                        job_id=JobId(status.job_id),
                        new_state=status.state,
                        exit_code=status.exit_code,
                        error=status.error or None,
                        finished_at_ms=status.finished_at_ms,
                    )
                )

        return HeartbeatResult(
            worker_id=worker.worker_id,
            success=True,
            consecutive_failures=0,
            worker_failed=False,
            job_updates=job_updates,
        )

    def _apply_heartbeat_result(self, worker: ControllerWorker, result: HeartbeatResult, now_ms: int) -> None:
        """Apply heartbeat result to worker and job state.

        Args:
            worker: Worker that was checked
            result: HeartbeatResult from health tracker
            now_ms: Current timestamp in milliseconds
        """
        worker.consecutive_failures = result.consecutive_failures

        if result.success:
            worker.last_heartbeat_ms = now_ms

        if result.worker_failed:
            worker.healthy = False
            logger.warning(f"Worker {worker.worker_id} failed health check")
            self._state.log_action("worker_failed", worker_id=worker.worker_id)

            # Retry jobs that were running on the failed worker
            for job_id in result.failed_job_ids:
                handle_job_failure(self._state, job_id, is_worker_failure=True)

        # Apply job state updates from heartbeat response
        for update in result.job_updates:
            job = self._state.get_job(update.job_id)
            if job:
                job.state = update.new_state
                job.exit_code = update.exit_code
                job.error = update.error
                job.finished_at_ms = update.finished_at_ms

                # Remove from worker's running jobs
                worker.running_jobs.discard(update.job_id)

                self._state.log_action(
                    "job_completed",
                    job_id=update.job_id,
                    details=f"state={update.new_state}, exit_code={update.exit_code}",
                )

    def _run_server(self) -> None:
        """Run dashboard server (blocking, for thread)."""
        try:
            uvicorn.run(
                self._dashboard._app,
                host=self._config.host,
                port=self._config.port,
                log_level="error",
            )
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
