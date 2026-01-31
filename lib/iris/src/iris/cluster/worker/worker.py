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

"""Unified worker managing all components and lifecycle."""

import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from iris.chaos import chaos
from iris.cluster.worker.builder import ImageCache, ImageProvider
from iris.cluster.worker.bundle_cache import BundleCache, BundleProvider
from iris.cluster.worker.dashboard import WorkerDashboard
from iris.cluster.worker.docker import ContainerRuntime, DockerRuntime
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider, EnvironmentProvider
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.task_attempt import TaskAttempt, TaskAttemptConfig
from iris.cluster.worker.worker_types import TaskInfo
from iris.logging import get_global_buffer
from iris.managed_thread import ManagedThread, ThreadRegistry
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Worker configuration."""

    host: str = "127.0.0.1"
    port: int = 0
    cache_dir: Path | None = None
    registry: str = "localhost:5000"
    port_range: tuple[int, int] = (30000, 40000)
    controller_address: str | None = None
    worker_id: str | None = None
    poll_interval_seconds: float = 5.0
    heartbeat_timeout_seconds: float = 60.0


class Worker:
    """Unified worker managing all components and lifecycle."""

    def __init__(
        self,
        config: WorkerConfig,
        cache_dir: Path | None = None,
        bundle_provider: BundleProvider | None = None,
        image_provider: ImageProvider | None = None,
        container_runtime: ContainerRuntime | None = None,
        environment_provider: EnvironmentProvider | None = None,
        port_allocator: PortAllocator | None = None,
        thread_registry: ThreadRegistry | None = None,
    ):
        self._config = config

        # Setup cache directory
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if cache_dir:
            self._cache_dir = cache_dir
        elif config.cache_dir:
            self._cache_dir = config.cache_dir
        else:
            # Create temporary cache
            self._temp_dir = tempfile.TemporaryDirectory(prefix="worker_cache_")
            self._cache_dir = Path(self._temp_dir.name)

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Use overrides if provided, otherwise create defaults
        self._bundle_cache = bundle_provider or BundleCache(self._cache_dir, max_bundles=100)
        self._image_cache = image_provider or ImageCache(
            self._cache_dir,
            registry=config.registry,
            max_images=50,
        )
        self._runtime = container_runtime or DockerRuntime()
        self._environment_provider = environment_provider or DefaultEnvironmentProvider()
        self._port_allocator = port_allocator or PortAllocator(config.port_range)

        # Probe worker metadata eagerly so it's available before any task arrives.
        self._worker_metadata = self._environment_provider.probe()

        # Task state
        self._tasks: dict[str, TaskAttempt] = {}
        self._lock = threading.Lock()

        self._service = WorkerServiceImpl(self, log_buffer=get_global_buffer())
        self._dashboard = WorkerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        self._threads = thread_registry or ThreadRegistry()
        self._server: uvicorn.Server | None = None
        self._server_managed: ManagedThread | None = None
        self._lifecycle_managed: ManagedThread | None = None

        self._worker_id: str | None = config.worker_id
        self._controller_client: ControllerServiceClientSync | None = None

        # Heartbeat tracking for timeout detection
        self._last_heartbeat_time = time.monotonic()

    def start(self) -> None:
        # Clean up any orphaned containers from previous runs
        self._cleanup_all_iris_containers()

        # Start HTTP server
        self._server_managed = self._threads.spawn(target=self._run_server, name="worker-server")

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=5.0,
        )

        # Create controller client if controller configured
        if self._config.controller_address:
            self._controller_client = ControllerServiceClientSync(
                address=self._config.controller_address,
                timeout_ms=5000,
            )

            # Start lifecycle thread: register + serve + reset loop
            self._lifecycle_managed = self._threads.spawn(target=self._run_lifecycle, name="worker-lifecycle")

    def _cleanup_all_iris_containers(self) -> None:
        """Remove all iris-managed containers at startup.

        This handles crash recovery cleanly without tracking complexity.
        """
        removed = self._runtime.remove_all_iris_containers()
        if removed > 0:
            logger.info("Startup cleanup: removed %d iris containers", removed)

    def wait(self) -> None:
        """Block until the server thread exits."""
        if self._server_managed:
            self._server_managed.join()

    def stop(self) -> None:
        # Stop uvicorn server
        if self._server:
            self._server.should_exit = True

        # Shutdown all managed threads
        self._threads.shutdown(timeout=5.0)

        # Kill and remove all containers
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            if task.container_id:
                try:
                    self._runtime.kill(task.container_id, force=True)
                except RuntimeError:
                    pass
                try:
                    self._runtime.remove(task.container_id)
                except RuntimeError:
                    pass

        # Cleanup temp directory (best-effort)
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except OSError:
                pass

    def _run_server(self, stop_event: threading.Event) -> None:
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
            print(f"Worker server error: {e}")

    def _run_lifecycle(self, stop_event: threading.Event) -> None:
        """Main lifecycle: register, serve, reset, repeat.

        This loop runs continuously until shutdown. On each iteration:
        1. Reset worker state (kill all containers)
        2. Register with controller (retry until accepted)
        3. Serve (wait for heartbeats from controller)
        4. If heartbeat timeout expires, return to step 1
        """
        while not stop_event.is_set():
            self._reset_worker_state()
            worker_id = self._register(stop_event)
            if worker_id is None:
                # Shutdown requested during registration
                break
            self._worker_id = worker_id
            self._serve(stop_event)

    def _register(self, stop_event: threading.Event) -> str | None:
        """Register with controller. Retries until accepted or shutdown.

        Returns the assigned worker_id, or None if shutdown was requested.
        """
        metadata = self._worker_metadata
        address = self._resolve_address()

        # Controller client is created in start() before this thread starts
        assert self._controller_client is not None

        logger.info("Attempting to register with controller at %s", self._config.controller_address)

        while not stop_event.is_set():
            try:
                # Chaos injection for testing delayed registration
                if rule := chaos("worker.register"):
                    time.sleep(rule.delay_seconds)
                    if rule.error:
                        raise rule.error

                response = self._controller_client.register(
                    cluster_pb2.Controller.RegisterRequest(
                        address=address,
                        metadata=metadata,
                    )
                )
                if response.accepted:
                    logger.info("Registered with controller: %s", response.worker_id)
                    return response.worker_id
                else:
                    logger.warning("Registration rejected by controller, retrying in 5s")
            except Exception as e:
                logger.warning("Registration failed: %s, retrying in 5s", e)
            stop_event.wait(5.0)

        return None

    def _resolve_address(self) -> str:
        """Resolve the address to advertise to the controller."""
        metadata = self._worker_metadata

        # Determine the address to advertise to the controller.
        # If host is 0.0.0.0 (bind to all interfaces), use the probed IP for external access.
        # Otherwise, use the configured host.
        address_host = self._config.host
        if address_host == "0.0.0.0":
            address_host = metadata.ip_address

        # Get VM address from probe (injected by ManagedVm bootstrap via IRIS_VM_ADDRESS)
        # For non-cloud workers, use host:port as both worker_id and vm_address
        vm_address = metadata.vm_address
        if not vm_address:
            vm_address = f"{address_host}:{self._config.port}"
            metadata.vm_address = vm_address

        return f"{address_host}:{self._config.port}"

    def _serve(self, stop_event: threading.Event) -> None:
        """Wait for heartbeats from controller. Returns when heartbeat timeout expires.

        This method blocks in a loop, checking the time since last heartbeat.
        When the timeout expires, it returns, triggering a reset and re-registration.
        """
        self._last_heartbeat_time = time.monotonic()
        heartbeat_timeout = self._config.heartbeat_timeout_seconds
        logger.info("Serving (waiting for controller heartbeats)")

        while not stop_event.is_set():
            elapsed = time.monotonic() - self._last_heartbeat_time
            if elapsed > heartbeat_timeout:
                logger.warning("No heartbeat from controller for %.0fs, resetting", elapsed)
                return
            # Check every second
            stop_event.wait(1.0)

    def _reset_worker_state(self) -> None:
        """Reset worker state: wipe all containers and clear tracking."""
        logger.info("Resetting worker state")

        # Clear task tracking
        with self._lock:
            self._tasks.clear()

        # Wipe ALL iris containers (simple, no tracking needed)
        self._cleanup_all_iris_containers()

        logger.info("Worker state reset complete")

    def _notify_task_update(self, task: TaskAttempt) -> None:
        """Notify controller that task state changed.

        Sends a lightweight ping to the controller, triggering a priority heartbeat.
        """
        if not self._controller_client or not self._worker_id:
            return

        # Send a lightweight ping to trigger priority heartbeat (best-effort)
        try:
            self._controller_client.notify_task_update(
                cluster_pb2.Controller.NotifyTaskUpdateRequest(
                    worker_id=self._worker_id,
                )
            )
        except Exception as e:
            # Best-effort ping; if it fails, the next regular heartbeat will deliver the update
            logger.debug("notify_task_update failed (update will be delivered via next heartbeat): %s", e, exc_info=True)

    # Task management methods

    def submit_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> str:
        """Submit a new task for execution."""
        if rule := chaos("worker.submit_task"):
            time.sleep(rule.delay_seconds)
            raise RuntimeError("chaos: worker rejecting task")
        task_id = request.task_id
        job_id = request.job_id
        task_index = request.task_index
        num_tasks = request.num_tasks
        attempt_id = request.attempt_id

        # Allocate requested ports
        port_names = list(request.ports)
        allocated_ports = self._port_allocator.allocate(len(port_names)) if port_names else []
        ports = dict(zip(port_names, allocated_ports, strict=True))

        # Create task working directory with attempt isolation
        # Use safe path component for hierarchical task IDs (e.g., "my-exp/task-0" -> "my-exp__task-0")
        safe_task_id = task_id.replace("/", "__")
        workdir = Path(tempfile.gettempdir()) / "iris-worker" / "tasks" / f"{safe_task_id}_attempt_{attempt_id}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Create TaskAttempt to handle the full execution lifecycle
        config = TaskAttemptConfig(
            task_id=task_id,
            job_id=job_id,
            task_index=task_index,
            num_tasks=num_tasks,
            attempt_id=attempt_id,
            request=request,
            ports=ports,
            workdir=workdir,
        )

        attempt = TaskAttempt(
            config=config,
            bundle_provider=self._bundle_cache,
            image_provider=self._image_cache,
            container_runtime=self._runtime,
            worker_metadata=self._worker_metadata,
            worker_id=self._worker_id,
            controller_address=self._config.controller_address,
            port_allocator=self._port_allocator,
            report_state=lambda: self._notify_task_update(attempt),
            poll_interval_seconds=self._config.poll_interval_seconds,
        )

        with self._lock:
            self._tasks[task_id] = attempt

        # Start execution in background
        thread = threading.Thread(target=attempt.run, daemon=True)
        attempt.thread = thread
        thread.start()

        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get a task by ID.

        Returns TaskInfo view (implemented by TaskAttempt) to decouple callers
        from execution internals.
        """
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskInfo]:
        """List all tasks.

        Returns TaskInfo views (implemented by TaskAttempt) to decouple callers
        from execution internals.
        """
        return list(self._tasks.values())

    def handle_heartbeat(self, request: cluster_pb2.HeartbeatRequest) -> cluster_pb2.HeartbeatResponse:
        """Handle controller-initiated heartbeat with reconciliation.

        Processes tasks_to_run and tasks_to_kill, reconciles expected_tasks against
        actual state, and returns current running/completed tasks.
        """
        # Update heartbeat timestamp
        self._last_heartbeat_time = time.monotonic()

        # Start new tasks
        for run_req in request.tasks_to_run:
            try:
                self.submit_task(run_req)
                logger.info("Heartbeat: submitted task %s", run_req.task_id)
            except Exception as e:
                logger.warning("Heartbeat: failed to submit task %s: %s", run_req.task_id, e)

        # Kill requested tasks
        for task_id in request.tasks_to_kill:
            try:
                self.kill_task(task_id)
                logger.info("Heartbeat: killed task %s", task_id)
            except Exception as e:
                logger.warning("Heartbeat: failed to kill task %s: %s", task_id, e)

        # Build expected_tasks lookup
        expected_task_ids = {entry.task_id: entry for entry in request.expected_tasks}

        # Terminal states
        terminal_states = {
            cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_KILLED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        }

        running_tasks = []
        completed_tasks = []

        with self._lock:
            # Reconcile expected_tasks against actual state
            for expected_entry in request.expected_tasks:
                task_id = expected_entry.task_id
                task = self._tasks.get(task_id)

                if task is None:
                    # Task not found - report as WORKER_FAILED
                    completed_tasks.append(
                        cluster_pb2.Controller.CompletedTaskEntry(
                            task_id=task_id,
                            job_id="",  # We don't have this information
                            task_index=0,
                            state=cluster_pb2.TASK_STATE_WORKER_FAILED,
                            exit_code=0,
                            error="Task not found on worker",
                            finished_at_ms=int(time.time() * 1000),
                            attempt_id=expected_entry.attempt_id,
                        )
                    )
                elif task.status in terminal_states:
                    # Task is terminal - report as completed
                    task_proto = task.to_proto()
                    completed_tasks.append(
                        cluster_pb2.Controller.CompletedTaskEntry(
                            task_id=task_proto.task_id,
                            job_id=task_proto.job_id,
                            task_index=task_proto.task_index,
                            state=task_proto.state,
                            exit_code=task_proto.exit_code,
                            error=task_proto.error,
                            finished_at_ms=task_proto.finished_at_ms,
                            attempt_id=task_proto.current_attempt_id,
                        )
                    )
                else:
                    # Task is running/building - include in running_tasks
                    running_tasks.append(
                        cluster_pb2.Controller.RunningTaskEntry(
                            task_id=task_id,
                            attempt_id=task.to_proto().current_attempt_id,
                        )
                    )

            # Report all non-terminal tasks (including unexpected ones)
            for task_id, task in self._tasks.items():
                if task_id not in expected_task_ids and task.status not in terminal_states:
                    running_tasks.append(
                        cluster_pb2.Controller.RunningTaskEntry(
                            task_id=task_id,
                            attempt_id=task.to_proto().current_attempt_id,
                        )
                    )

        return cluster_pb2.HeartbeatResponse(
            running_tasks=running_tasks,
            completed_tasks=completed_tasks,
        )

    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool:
        """Kill a running task."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        # Check if already in terminal state
        if task.status not in (
            cluster_pb2.TASK_STATE_RUNNING,
            cluster_pb2.TASK_STATE_BUILDING,
            cluster_pb2.TASK_STATE_PENDING,
        ):
            return False

        # Set flag to signal thread to stop
        task.should_stop = True

        # If container exists, try to kill it
        if task.container_id:
            try:
                # Send SIGTERM
                self._runtime.kill(task.container_id, force=False)

                # Wait for shutdown
                running_states = (cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_BUILDING)
                stopped = ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
                    lambda: task.status not in running_states,
                    timeout=term_timeout_ms / 1000,
                )

                # Force kill if graceful shutdown timed out
                if not stopped:
                    try:
                        self._runtime.kill(task.container_id, force=True)
                    except RuntimeError:
                        pass
            except RuntimeError:
                # Container may have already been removed or stopped
                pass

        return True

    def get_logs(self, task_id: str, start_line: int = 0) -> list[cluster_pb2.Worker.LogEntry]:
        """Get logs for a task."""
        task = self._tasks.get(task_id)
        if not task:
            return []

        logs: list[cluster_pb2.Worker.LogEntry] = []

        # Add build logs from task.logs (these have proper timestamps)
        for log_line in task.logs.lines:
            logs.append(log_line.to_proto())

        # Fetch container stdout/stderr from Docker if container exists
        if task.container_id:
            container_logs = self._runtime.get_logs(task.container_id)
            for log_line in container_logs:
                logs.append(log_line.to_proto())

        # Sort by timestamp
        logs.sort(key=lambda x: x.timestamp_ms)

        return logs[start_line:]

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self._config.port}"
