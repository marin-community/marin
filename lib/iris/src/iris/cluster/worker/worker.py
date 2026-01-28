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
import shutil
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from iris.rpc import cluster_pb2
from iris.time_utils import ExponentialBackoff, now_ms
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.rpc.errors import format_exception_with_traceback
from iris.cluster.types import Entrypoint
from iris.cluster.worker.builder import ImageCache, ImageProvider
from iris.cluster.worker.bundle_cache import BundleCache, BundleProvider
from iris.cluster.worker.dashboard import WorkerDashboard
from iris.cluster.worker.docker import ContainerConfig, ContainerRuntime, DockerRuntime
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider, EnvironmentProvider, collect_workdir_size_mb
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.worker_types import Task

logger = logging.getLogger(__name__)


def _rewrite_address_for_container(address: str) -> str:
    """Rewrite localhost addresses to host.docker.internal for container access.

    Docker containers on Mac/Windows cannot reach host localhost directly.
    Using host.docker.internal works cross-platform when combined with
    --add-host=host.docker.internal:host-gateway on Linux.
    """
    for localhost in ("127.0.0.1", "localhost", "0.0.0.0"):
        if localhost in address:
            return address.replace(localhost, "host.docker.internal")
    return address


class PortAllocator:
    """Allocate ephemeral ports for tasks."""

    def __init__(self, port_range: tuple[int, int] = (30000, 40000)):
        self._range = port_range
        self._allocated: set[int] = set()
        self._lock = threading.Lock()

    def allocate(self, count: int = 1) -> list[int]:
        with self._lock:
            ports = []
            for _ in range(count):
                port = self._find_free_port()
                self._allocated.add(port)
                ports.append(port)
            return ports

    def release(self, ports: list[int]) -> None:
        with self._lock:
            for port in ports:
                self._allocated.discard(port)

    def _find_free_port(self) -> int:
        for port in range(self._range[0], self._range[1]):
            if port in self._allocated:
                continue
            if self._is_port_free(port):
                return port
        logger.warning("Port allocation exhausted: no free ports in range %d-%d", self._range[0], self._range[1])
        raise RuntimeError("No free ports available")

    def _is_port_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False


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

        # Task state
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

        self._service = WorkerServiceImpl(self)
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
        metadata = self._environment_provider.probe()

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

        # Retry registration until successful (or reset requested)
        attempt = 0
        while not self._stop_heartbeat.is_set():
            attempt += 1

            # Get active task IDs for controller restart recovery (RUNNING or BUILDING)
            with self._lock:
                running_task_ids = [
                    t.task_id
                    for t in self._tasks.values()
                    if t.status in (cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_BUILDING)
                ]

            request = cluster_pb2.Controller.RegisterWorkerRequest(
                worker_id=self._worker_id,
                address=f"{address_host}:{self._config.port}",
                metadata=metadata,
                running_task_ids=running_task_ids,
            )

            try:
                logger.debug("Registration attempt %d for worker %s", attempt, self._worker_id)
                response = self._controller_client.register_worker(request)

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
                # Update active task IDs for each heartbeat
                with self._lock:
                    running_task_ids = [
                        t.task_id
                        for t in self._tasks.values()
                        if t.status in (cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_BUILDING)
                    ]
                request = cluster_pb2.Controller.RegisterWorkerRequest(
                    worker_id=self._worker_id,
                    address=f"{address_host}:{self._config.port}",
                    metadata=metadata,
                    running_task_ids=running_task_ids,
                )
                response = self._controller_client.register_worker(request)

                if response.should_reset:
                    logger.warning("Controller signaled reset during heartbeat, cleaning up")
                    self._reset_worker_state()
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

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

    def _report_task_state(self, task: Task) -> None:
        """Report task state to controller."""
        if not self._controller_client or not self._worker_id:
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
                finished_at_ms=task.finished_at_ms or 0,
                attempt_id=task.attempt_id,
            )
            self._controller_client.report_task_state(request)
        except Exception as e:
            logger.warning(f"Failed to report task state to controller: {e}")

    # Task management methods

    def submit_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> str:
        """Submit a new task for execution."""
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

        task = Task(
            task_id=task_id,
            job_id=job_id,
            task_index=task_index,
            num_tasks=num_tasks,
            attempt_id=attempt_id,
            request=request,
            status=cluster_pb2.TASK_STATE_PENDING,
            ports=ports,
            workdir=workdir,
        )

        with self._lock:
            self._tasks[task_id] = task

        # Start execution in background
        task.thread = threading.Thread(target=self._execute_task, args=(task,), daemon=True)
        task.thread.start()

        return task_id

    def _execute_task(self, task: Task) -> None:
        import sys

        try:
            # Phase 1: Download bundle
            task.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="downloading bundle")
            task.started_at_ms = now_ms()

            bundle_path = self._bundle_cache.get_bundle(
                task.request.bundle_gcs_path,
                expected_hash=None,
            )

            # Phase 2: Build image
            task.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="building image")
            task.build_started_ms = now_ms()
            env_config = task.request.environment
            extras = list(env_config.extras)

            task.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="populating uv cache")
            task.logs.add("build", "Building Docker image...")

            # Detect host Python version for container compatibility
            # cloudpickle serializes bytecode which is version-specific
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            base_image = f"python:{py_version}-slim"

            build_result = self._image_cache.build(
                bundle_path=bundle_path,
                base_image=base_image,
                extras=extras,
                job_id=task.job_id,
                task_logs=task.logs,
            )

            task.build_finished_ms = now_ms()
            task.build_from_cache = build_result.from_cache
            task.image_tag = build_result.image_tag

            # Protect image from eviction while task is running
            if self._image_cache:
                self._image_cache.protect(build_result.image_tag)

            # Phase 3: Create and start container
            task.transition_to(cluster_pb2.TASK_STATE_RUNNING)

            # Build environment from user-provided vars + EnvironmentConfig
            env = dict(env_config.env_vars)

            # Build iris system environment based on runtime
            iris_env = self._build_iris_env(task)
            env.update(iris_env)

            # Convert proto entrypoint to typed Entrypoint
            entrypoint = Entrypoint.from_proto(task.request.entrypoint)

            config = ContainerConfig(
                image=build_result.image_tag,
                entrypoint=entrypoint,
                env=env,
                resources=task.request.resources if task.request.HasField("resources") else None,
                timeout_seconds=task.request.timeout_seconds or None,
                ports=task.ports,
                mounts=[(str(task.workdir), "/workdir", "rw")],
                task_id=task.task_id,
                job_id=task.job_id,
            )

            # Create and start container with retry on port binding failures
            container_id = None
            max_port_retries = 3
            for attempt in range(max_port_retries):
                try:
                    container_id = self._runtime.create_container(config)
                    task.container_id = container_id
                    self._runtime.start_container(container_id)
                    break
                except RuntimeError as e:
                    if "address already in use" in str(e) and attempt < max_port_retries - 1:
                        logger.warning(
                            "Port conflict for task %s, retrying with new ports (attempt %d)", task.task_id, attempt + 2
                        )
                        task.logs.add("build", f"Port conflict, retrying with new ports (attempt {attempt + 2})")
                        # Release current ports and allocate new ones
                        self._port_allocator.release(list(task.ports.values()))
                        port_names = list(task.ports.keys())
                        new_ports = self._port_allocator.allocate(len(port_names))
                        task.ports = dict(zip(port_names, new_ports, strict=True))

                        # Update config with new ports
                        config.ports = task.ports
                        for name, port in task.ports.items():
                            config.env[f"IRIS_PORT_{name.upper()}"] = str(port)

                        # Try to remove failed container if it was created
                        if container_id:
                            try:
                                self._runtime.remove(container_id)
                            except RuntimeError:
                                pass
                            container_id = None
                    else:
                        raise

            # container_id is guaranteed to be set here (loop breaks on success, raises on failure)
            assert container_id is not None

            # Report RUNNING state to controller so endpoints become visible
            self._report_task_state(task)

            # Phase 4: Monitor task execution
            self._monitor_task(task, container_id, config.timeout_seconds)

        except Exception as e:
            error_msg = format_exception_with_traceback(e)
            task.logs.add("error", f"Task failed:\n{error_msg}")
            task.transition_to(cluster_pb2.TASK_STATE_FAILED, error=error_msg)
        finally:
            # Report task state to controller if in terminal state
            if task.status in (
                cluster_pb2.TASK_STATE_SUCCEEDED,
                cluster_pb2.TASK_STATE_FAILED,
                cluster_pb2.TASK_STATE_KILLED,
            ):
                self._report_task_state(task)

            # Cleanup: release ports, remove workdir (keep container for logs)
            if not task.cleanup_done:
                task.cleanup_done = True
                self._port_allocator.release(list(task.ports.values()))
                # Unprotect image from eviction now that task is done
                if self._image_cache and task.image_tag:
                    self._image_cache.unprotect(task.image_tag)
                # Keep container around for log retrieval via docker logs
                # Remove working directory (no longer needed since logs come from Docker)
                if task.workdir and task.workdir.exists():
                    shutil.rmtree(task.workdir, ignore_errors=True)

    def _monitor_task(self, task: Task, container_id: str, timeout_seconds: int | None) -> None:
        """Monitor task execution: check status, collect stats, handle timeouts.

        Polls container status at regular intervals until the container stops.
        Collects runtime statistics (CPU, memory, disk) and handles timeout enforcement.
        Updates task state to terminal status (SUCCEEDED/FAILED/KILLED) when container stops.
        """
        start_time = time.time()

        while True:
            # Check if we should stop
            if task.should_stop:
                task.transition_to(cluster_pb2.TASK_STATE_KILLED)
                break

            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                self._runtime.kill(container_id, force=True)
                task.transition_to(
                    cluster_pb2.TASK_STATE_FAILED,
                    error="Timeout exceeded",
                    exit_code=-1,
                )
                break

            # Check container status
            status = self._runtime.inspect(container_id)
            if not status.running:
                # Read result file only if container succeeded
                if status.exit_code == 0 and task.workdir:
                    result_path = task.workdir / "_result.pkl"
                    if result_path.exists():
                        try:
                            task.result = result_path.read_bytes()
                        except Exception as e:
                            task.logs.add("error", f"Failed to read result file: {e}")

                # Container has stopped
                if status.error:
                    task.transition_to(
                        cluster_pb2.TASK_STATE_FAILED,
                        error=status.error,
                        exit_code=status.exit_code or -1,
                    )
                elif status.exit_code == 0:
                    task.transition_to(cluster_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
                else:
                    task.transition_to(
                        cluster_pb2.TASK_STATE_FAILED,
                        error=f"Exit code: {status.exit_code}",
                        exit_code=status.exit_code or -1,
                    )
                break

            # Collect stats
            try:
                stats = self._runtime.get_stats(container_id)
                if stats.available:
                    task.current_memory_mb = stats.memory_mb
                    task.current_cpu_percent = stats.cpu_percent
                    task.process_count = stats.process_count
                    if stats.memory_mb > task.peak_memory_mb:
                        task.peak_memory_mb = stats.memory_mb

                if task.workdir:
                    task.disk_mb = collect_workdir_size_mb(task.workdir)
            except Exception:
                pass  # Don't fail task on stats collection errors

            # Sleep before next poll
            time.sleep(self._config.poll_interval_seconds)

    def _build_iris_env(self, task: Task) -> dict[str, str]:
        """Build Iris system environment variables for the task container.

        Auto-injects task metadata and configuration that tasks need to interact
        with the Iris cluster (task ID, job ID, worker ID, controller address, ports).
        These override user-provided values.
        """
        env = {}

        # Core task metadata
        env["IRIS_JOB_ID"] = task.job_id
        env["IRIS_TASK_ID"] = task.task_id
        env["IRIS_TASK_INDEX"] = str(task.task_index)
        env["IRIS_NUM_TASKS"] = str(task.num_tasks)
        env["IRIS_ATTEMPT_ID"] = str(task.attempt_id)

        if self._worker_id:
            env["IRIS_WORKER_ID"] = self._worker_id

        if self._config.controller_address:
            # Only rewrite localhost addresses for Docker containers
            if isinstance(self._runtime, DockerRuntime):
                env["IRIS_CONTROLLER_ADDRESS"] = _rewrite_address_for_container(self._config.controller_address)
            else:
                env["IRIS_CONTROLLER_ADDRESS"] = self._config.controller_address

        # Inject bundle path for sub-task inheritance
        if task.request.bundle_gcs_path:
            env["IRIS_BUNDLE_GCS_PATH"] = task.request.bundle_gcs_path

        # Inject bind host - 0.0.0.0 for Docker (so port mapping works), 127.0.0.1 otherwise
        # Also inject advertise host - the address other containers should use to reach this one
        if isinstance(self._runtime, DockerRuntime):
            env["IRIS_BIND_HOST"] = "0.0.0.0"
            env["IRIS_ADVERTISE_HOST"] = "host.docker.internal"
        else:
            env["IRIS_BIND_HOST"] = "127.0.0.1"
            env["IRIS_ADVERTISE_HOST"] = "127.0.0.1"

        # Inject allocated ports
        for name, port in task.ports.items():
            env[f"IRIS_PORT_{name.upper()}"] = str(port)

        return env

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        """List all tasks."""
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
