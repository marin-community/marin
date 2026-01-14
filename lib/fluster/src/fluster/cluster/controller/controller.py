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
- Scheduler: Background thread for job scheduling
- HeartbeatMonitor: Background thread for worker health checks
- ControllerServiceImpl: RPC service implementation
- ControllerDashboard: Web dashboard and HTTP server

This simplifies controller initialization and ensures consistent lifecycle
management across all components.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from fluster import cluster_pb2
from fluster.cluster.controller.dashboard import ControllerDashboard
from fluster.cluster.controller.heartbeat import HeartbeatMonitor
from fluster.cluster.controller.scheduler import Scheduler
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from fluster.cluster.types import JobId, WorkerId


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
    - Scheduler: Background thread matching jobs to workers
    - HeartbeatMonitor: Background thread checking worker health
    - ControllerServiceImpl: RPC service implementation
    - ControllerDashboard: Web dashboard and HTTP server

    Example:
        ```python
        config = ControllerConfig(port=8080)
        controller = Controller(
            config=config,
            dispatch_fn=my_dispatch_fn,
            heartbeat_fn=my_heartbeat_fn,
            on_worker_failed=my_failure_handler,
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
        dispatch_fn: Callback to dispatch jobs to workers. Takes a job and worker,
            returns True on success, False on failure.
        heartbeat_fn: Callback to send heartbeats to workers. Takes worker address,
            returns HeartbeatResponse on success or None on failure.
        on_worker_failed: Callback when worker fails. Takes worker_id and list of
            failed job IDs.
    """

    def __init__(
        self,
        config: ControllerConfig,
        dispatch_fn: Callable[[ControllerJob, ControllerWorker], bool],
        heartbeat_fn: Callable[[str], cluster_pb2.HeartbeatResponse | None],
        on_worker_failed: Callable[[WorkerId, list[JobId]], None],
    ):
        """Initialize controller components.

        Args:
            config: Controller configuration
            dispatch_fn: Callback to dispatch jobs to workers
            heartbeat_fn: Callback to send heartbeats to workers
            on_worker_failed: Callback when worker fails
        """
        self._config = config

        # Initialize components
        self._state = ControllerState()
        self._scheduler = Scheduler(
            self._state,
            dispatch_fn,
            interval_seconds=config.scheduler_interval_seconds,
        )
        self._heartbeat_monitor = HeartbeatMonitor(
            self._state,
            heartbeat_fn,
            on_worker_failed,
            interval_seconds=config.heartbeat_interval_seconds,
        )
        self._service = ControllerServiceImpl(
            self._state,
            self._scheduler,
            bundle_dir=config.bundle_dir,
        )
        self._dashboard = ControllerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        # Track actual port after binding
        self._actual_port: int | None = None
        self._server_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start all background components.

        Starts the scheduler, heartbeat monitor, and dashboard server.
        The dashboard server runs in a background daemon thread.
        """
        # Start scheduler and heartbeat monitor
        self._scheduler.start()
        self._heartbeat_monitor.start()

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

        Stops the heartbeat monitor and scheduler. The dashboard server
        stops automatically when the daemon thread exits.
        """
        if self._heartbeat_monitor:
            self._heartbeat_monitor.stop()
        if self._scheduler:
            self._scheduler.stop()
        # Dashboard stops when thread exits (daemon)

    def _run_server(self) -> None:
        """Run dashboard server (blocking, for thread).

        This method is called in a background thread and runs the uvicorn
        server until the process exits or an error occurs.
        """
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
        request: cluster_pb2.LaunchJobRequest,
    ) -> cluster_pb2.LaunchJobResponse:
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
    ) -> cluster_pb2.GetJobStatusResponse:
        """Get the status of a job.

        Args:
            job_id: Job identifier

        Returns:
            GetJobStatusResponse with current job status
        """
        request = cluster_pb2.GetJobStatusRequest(job_id=job_id)
        return self._service.get_job_status(request, None)

    def register_worker(
        self,
        request: cluster_pb2.RegisterWorkerRequest,
    ) -> cluster_pb2.RegisterWorkerResponse:
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
        request = cluster_pb2.TerminateJobRequest(job_id=job_id)
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
