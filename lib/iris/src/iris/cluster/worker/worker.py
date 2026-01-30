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
        # Completions that failed to report via report_task_state RPC.
        # Piggybacked on the next successful heartbeat.
        self._unreported_completions: list[cluster_pb2.Controller.CompletedTaskEntry] = []

        self._service = WorkerServiceImpl(self, log_buffer=get_global_buffer())
        self._dashboard = WorkerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

        # Heartbeat/registration thread
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()
        self._worker_id: str | None = config.worker_id
        self._controller_client: ControllerServiceClientSync | None = None

    def start(self) -> None:
        # Clean up any orphaned containers from previous runs
        self._cleanup_all_iris_containers()

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

        # Create controller client synchronously (before any tasks can be dispatched)
        if self._config.controller_address:
            self._controller_client = ControllerServiceClientSync(
                address=self._config.controller_address,
                timeout_ms=5000,
            )

        # Start heartbeat loop if controller configured
        if self._config.controller_address:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _cleanup_all_iris_containers(self) -> None:
        """Remove all iris-managed containers at startup.

        This handles crash recovery cleanly without tracking complexity.
        """
        removed = self._runtime.remove_all_iris_containers()
        if removed > 0:
            logger.info("Startup cleanup: removed %d iris containers", removed)

    def wait(self) -> None:
        """Block until the server thread exits."""
        if self._server_thread:
            self._server_thread.join()

    def stop(self) -> None:
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)

        # Stop uvicorn server
        if self._server:
            self._server.should_exit = True
        if self._server_thread:
            self._server_thread.join(timeout=5.0)

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
            print(f"Worker server error: {e}")

    def _heartbeat_loop(self) -> None:
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

        # Derive worker_id from vm_address (no UUID generation)
        if not self._worker_id:
            self._worker_id = vm_address

        # Controller client is created in start() before this thread starts
        assert self._controller_client is not None

        address = f"{address_host}:{self._config.port}"

        # Retry registration until successful (or reset requested)
        attempt = 0
        while not self._stop_heartbeat.is_set():
            attempt += 1

            request, num_completions = self._build_heartbeat_request(address, metadata)

            try:
                if rule := chaos("worker.heartbeat"):
                    time.sleep(rule.delay_seconds)
                    logger.debug("chaos: skipping heartbeat")
                    continue
                logger.debug("Registration attempt %d for worker %s", attempt, self._worker_id)
                response = self._controller_client.register_worker(request)
                self._clear_delivered_completions(num_completions)

                if response.should_reset:
                    logger.warning("Controller signaled reset for worker %s, cleaning up", self._worker_id)
                    self._reset_worker_state()
                    continue  # Re-register with empty task list

                if response.accepted:
                    logger.info("Registered with controller: %s", self._worker_id)
                    break
            except Exception as e:
                logger.warning("Registration attempt %d failed, retrying in 5s: %s", attempt, e)
            self._stop_heartbeat.wait(5.0)

        # Periodic heartbeat (re-registration)
        heartbeat_interval = 10.0  # seconds
        while not self._stop_heartbeat.is_set():
            self._stop_heartbeat.wait(heartbeat_interval)
            if self._stop_heartbeat.is_set():
                break
            try:
                if rule := chaos("worker.heartbeat"):
                    time.sleep(rule.delay_seconds)
                    logger.debug("chaos: skipping heartbeat")
                    continue

                request, num_completions = self._build_heartbeat_request(address, metadata)
                response = self._controller_client.register_worker(request)
                self._clear_delivered_completions(num_completions)

                if response.should_reset:
                    logger.warning("Controller signaled reset during heartbeat, cleaning up")
                    self._reset_worker_state()
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    def _build_heartbeat_request(
        self,
        address: str,
        metadata: cluster_pb2.WorkerMetadata,
    ) -> tuple[cluster_pb2.Controller.RegisterWorkerRequest, int]:
        """Build a heartbeat request with current task state and pending completions.

        Returns the request and the count of pending completions included,
        so the caller can clear them after a successful RPC.
        """
        with self._lock:
            active_tasks = [
                t
                for t in self._tasks.values()
                if t.status in (cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_BUILDING)
            ]
            running_tasks = [
                cluster_pb2.Controller.RunningTaskEntry(
                    task_id=t.task_id,
                    attempt_id=t.attempt_id,
                )
                for t in active_tasks
            ]
            num_completions = len(self._unreported_completions)
            completed_tasks = list(self._unreported_completions)

        request = cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id=self._worker_id,
            address=address,
            metadata=metadata,
            running_tasks=running_tasks,
            completed_tasks=completed_tasks,
        )
        return request, num_completions

    def _clear_delivered_completions(self, num_delivered: int) -> None:
        """Remove completions that were successfully delivered via heartbeat.

        Since completions are only appended, the delivered entries are always
        a prefix of _unreported_completions.
        """
        if num_delivered > 0:
            with self._lock:
                self._unreported_completions = self._unreported_completions[num_delivered:]

    def _reset_worker_state(self) -> None:
        """Reset worker state: wipe all containers and clear tracking.

        Called when controller signals should_reset (e.g., after controller restart
        when worker claims tasks the controller doesn't know about).
        """
        logger.info("Resetting worker state")

        # Clear task tracking
        with self._lock:
            self._tasks.clear()

        # Wipe ALL iris containers (simple, no tracking needed)
        self._cleanup_all_iris_containers()

        logger.info("Worker state reset complete")

    def _report_task_state(self, task: TaskAttempt) -> None:
        """Report task state to controller.

        Terminal completions are always delivered via heartbeat. This RPC is just
        a ping so the controller processes the next heartbeat sooner.
        """
        self._buffer_completion_if_terminal(task)

        if not self._controller_client or not self._worker_id:
            return

        if rule := chaos("worker.report_task_state"):
            time.sleep(rule.delay_seconds)
            logger.debug("chaos: skipping report_task_state")
            return

        try:
            request = cluster_pb2.Controller.ReportTaskStateRequest(
                worker_id=self._worker_id,
                task_id=task.task_id,
                job_id=task.job_id,
                task_index=task.task_index,
                state=task.status,
                exit_code=task.exit_code or 0,
                error=task.error or "",
                finished_at_ms=task.finished_at.epoch_ms if task.finished_at else 0,
                attempt_id=task.attempt_id,
            )
            self._controller_client.report_task_state(request)
        except Exception as e:
            logger.warning(f"Failed to report task state to controller: {e}")

    def _buffer_completion_if_terminal(self, task: TaskAttempt) -> None:
        """Buffer a terminal task completion for heartbeat delivery."""
        terminal_states = {
            cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_KILLED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        }
        if task.status not in terminal_states:
            return
        with self._lock:
            self._unreported_completions.append(
                cluster_pb2.Controller.CompletedTaskEntry(
                    task_id=task.task_id,
                    job_id=task.job_id,
                    task_index=task.task_index,
                    state=task.status,
                    exit_code=task.exit_code or 0,
                    error=task.error or "",
                    finished_at_ms=task.finished_at.epoch_ms if task.finished_at else 0,
                    attempt_id=task.attempt_id,
                )
            )

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
            report_state=self._report_task_state,
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
