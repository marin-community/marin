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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import uvicorn

from iris.time_utils import ExponentialBackoff

from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.job import Job
from iris.cluster.controller.scheduler import Scheduler, SchedulingTransaction
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.state import ControllerState, ControllerWorker
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import WorkerServiceClientSync

logger = logging.getLogger(__name__)


class WorkerClient(Protocol):
    """Protocol for worker RPC client.

    Matches client-side WorkerServiceClientSync signature. The server Protocol has different signatures.
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

    Owns a single background loop that periodically:
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
        worker_stub_factory: Factory for creating worker RPC stubs
    """

    def __init__(
        self,
        config: ControllerConfig,
        worker_stub_factory: WorkerStubFactory,
    ):
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
        self._scheduling_loop_thread: threading.Thread | None = None
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
        now_ms = int(time.time() * 1000)
        pending_jobs = self._state.peek_pending_jobs()
        workers = self._state.get_available_workers()

        if not pending_jobs:
            return

        result = self._scheduler.find_assignments(pending_jobs, workers, now_ms)
        self._apply_schedule_result(result, now_ms)

    def _apply_schedule_result(self, transaction: SchedulingTransaction, now_ms: int) -> None:
        """Apply scheduling results: dispatch jobs and handle timeouts."""
        for job in transaction.timed_out_jobs:
            self._mark_job_unschedulable(job, now_ms)

        for job, worker in transaction.assignments:
            self._dispatch_job(transaction, job, worker, now_ms)

    def _dispatch_job(
        self,
        transaction: SchedulingTransaction,
        job: Job,
        worker: ControllerWorker,
        now_ms: int,
    ) -> None:
        """Dispatch a job to a worker via RPC.

        State is already updated by tentatively_assign(). The worker reports all state
        transitions (BUILDING, RUNNING, SUCCEEDED, etc.) via ReportJobState.
        """
        job.mark_dispatched(worker.worker_id, now_ms)

        try:
            stub = self._stub_factory.get_stub(worker.address)
            request = cluster_pb2.Worker.RunJobRequest(
                job_id=str(job.job_id),
                attempt_id=job.current_attempt_id,
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
            job.revert_dispatch()
            transaction.rollback_assignment(job, worker)
            self._state.mark_worker_unhealthy(worker.worker_id)
            logger.exception("Failed to dispatch job %s to worker %s", job.job_id, worker.address)
            self._state.log_action(
                "dispatch_failed",
                job_id=job.job_id,
                worker_id=worker.worker_id,
            )

    def _mark_job_unschedulable(self, job: Job, now_ms: int) -> None:
        logger.warning(
            f"Job {job.job_id} exceeded scheduling timeout "
            f"({job.request.scheduling_timeout_seconds}s), marking as UNSCHEDULABLE"
        )
        self._state.transition_job(job.job_id, cluster_pb2.JOB_STATE_UNSCHEDULABLE, now_ms)
        self._state.log_action(
            "job_unschedulable",
            job_id=job.job_id,
            details=f"timeout={job.request.scheduling_timeout_seconds}s",
        )

    def _check_worker_timeouts(self) -> None:
        """Mark workers as unhealthy if they haven't sent heartbeat within worker_timeout_seconds."""
        now_ms = int(time.time() * 1000)
        timeout_ms = int(self._config.worker_timeout_seconds * 1000)

        for worker in self._state.list_all_workers():
            if worker.healthy and (now_ms - worker.last_heartbeat_ms) > timeout_ms:
                self._state.mark_worker_unhealthy(worker.worker_id)
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
                    self._state.transition_job(
                        job_id,
                        cluster_pb2.JOB_STATE_WORKER_FAILED,
                        now_ms,
                        is_worker_failure=True,
                        error=f"Worker {worker.worker_id} timed out",
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
            print(f"Controller server error: {e}")

    # Delegate key service methods

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
    def url(self) -> str:
        return f"http://{self._config.host}:{self._config.port}"
