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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn

from iris.chaos import chaos
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import ContainerRuntime
from iris.cluster.types import JobName
from iris.cluster.worker.bundle_cache import BundleCache, BundleProvider
from iris.cluster.worker.dashboard import WorkerDashboard
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider, EnvironmentProvider
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.task_attempt import TaskAttempt, TaskAttemptConfig
from iris.cluster.worker.worker_types import TaskInfo
from iris.logging import get_global_buffer
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Deadline, Duration, ExponentialBackoff, Timestamp

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Worker configuration."""

    host: str = "127.0.0.1"
    port: int = 0
    cache_dir: Path | None = None
    port_range: tuple[int, int] = (30000, 40000)
    controller_address: str | None = None
    worker_id: str | None = None
    poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5.0))
    heartbeat_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(60.0))
    uv_cache_dir: Path | None = None


class Worker:
    """Unified worker managing all components and lifecycle."""

    def __init__(
        self,
        config: WorkerConfig,
        bundle_provider: BundleProvider | None = None,
        container_runtime: ContainerRuntime | None = None,
        environment_provider: EnvironmentProvider | None = None,
        port_allocator: PortAllocator | None = None,
        threads: ThreadContainer | None = None,
    ):
        self._config = config

        if not config.cache_dir:
            raise ValueError("WorkerConfig.cache_dir is required")
        self._cache_dir = config.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Use external uv_cache_dir if provided, otherwise create inside cache_dir
        if config.uv_cache_dir:
            self._uv_cache_dir = config.uv_cache_dir
        else:
            self._uv_cache_dir = self._cache_dir / "uv-cache"
        self._uv_cache_dir.mkdir(parents=True, exist_ok=True)

        # Use overrides if provided, otherwise create defaults
        self._bundle_cache = bundle_provider or BundleCache(self._cache_dir, max_bundles=100)
        self._runtime = container_runtime or DockerRuntime()
        self._environment_provider = environment_provider or DefaultEnvironmentProvider()
        self._port_allocator = port_allocator or PortAllocator(config.port_range)

        # Probe worker metadata eagerly so it's available before any task arrives.
        self._worker_metadata = self._environment_provider.probe()

        # Task state: maps (task_id, attempt_id) -> TaskAttempt.
        # Preserves all attempts so logs for historical attempts remain accessible.
        self._tasks: dict[tuple[str, int], TaskAttempt] = {}
        self._lock = threading.Lock()

        self._service = WorkerServiceImpl(self, log_buffer=get_global_buffer())
        self._dashboard = WorkerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        self._server: uvicorn.Server | None = None
        self._threads = threads if threads is not None else get_thread_container()
        self._task_threads = self._threads.create_child("tasks")

        self._worker_id: str | None = config.worker_id
        self._controller_client: ControllerServiceClientSync | None = None

        # Heartbeat tracking for timeout detection
        self._heartbeat_deadline = Deadline.from_seconds(float("inf"))

    def start(self) -> None:
        # Clean up any orphaned containers from previous runs
        self._cleanup_all_iris_containers()

        # Start HTTP server
        self._server = uvicorn.Server(
            uvicorn.Config(
                self._dashboard._app,
                host=self._config.host,
                port=self._config.port,
                log_level="error",
            )
        )
        self._threads.spawn_server(self._server, name="worker-server")

        # Wait for server startup with exponential backoff
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server.started,
            timeout=Duration.from_seconds(5.0),
        )

        # Create controller client if controller configured
        if self._config.controller_address:
            self._controller_client = ControllerServiceClientSync(
                address=self._config.controller_address,
                timeout_ms=5000,
            )

            # Start lifecycle thread: register + serve + reset loop
            self._threads.spawn(target=self._run_lifecycle, name="worker-lifecycle")

    def _cleanup_all_iris_containers(self) -> None:
        """Remove all iris-managed containers at startup.

        This handles crash recovery cleanly without tracking complexity.
        """
        removed = self._runtime.remove_all_iris_containers()
        if removed > 0:
            logger.info("Startup cleanup: removed %d iris containers", removed)

    def wait(self) -> None:
        self._threads.wait()

    def stop(self) -> None:
        # Stop task threads first so running tasks exit before infrastructure
        # tears down. ThreadContainer.stop() signals each thread's stop_event,
        # which the _run_task watcher bridges to attempt.should_stop + container kill.
        self._task_threads.stop()

        if self._server:
            self._server.should_exit = True
        self._threads.stop()

        # Remove any remaining containers (tasks already killed above via stop_event)
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            if task.container_id:
                try:
                    self._runtime.remove(task.container_id)
                except RuntimeError:
                    pass

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
        self._heartbeat_deadline = Deadline.from_seconds(self._config.heartbeat_timeout.to_seconds())
        logger.info("Serving (waiting for controller heartbeats)")

        while not stop_event.is_set():
            if self._heartbeat_deadline.expired():
                logger.warning("No heartbeat from controller, resetting")
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

    _TERMINAL_STATES = frozenset(
        {
            cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_KILLED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        }
    )

    def _get_current_attempt(self, task_id_wire: str) -> TaskAttempt | None:
        """Get the most recent attempt for a task, or None if no attempts exist."""
        # Find all attempts for this task and return the one with highest attempt_id
        matching = [(key, task) for key, task in self._tasks.items() if key[0] == task_id_wire]
        if not matching:
            return None
        # Return the attempt with the highest attempt_id
        matching.sort(key=lambda x: x[0][1], reverse=True)
        return matching[0][1]

    def submit_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> str:
        """Submit a new task for execution.

        If a non-terminal task with the same task_id already exists:
        - Same or older attempt_id: rejected as duplicate
        - Newer attempt_id: old attempt is killed and new one starts

        Raises:
            RuntimeError: If a non-terminal attempt already exists (sanity check)
        """
        if rule := chaos("worker.submit_task"):
            time.sleep(rule.delay_seconds)
            raise RuntimeError("chaos: worker rejecting task")
        task_id_wire = request.task_id
        task_id = JobName.from_wire(task_id_wire)
        attempt_id = request.attempt_id
        key = (task_id_wire, attempt_id)

        should_kill_existing = False
        with self._lock:
            # Check if this exact (task_id, attempt_id) already exists
            if key in self._tasks:
                existing = self._tasks[key]
                logger.info(
                    "Rejecting duplicate task %s attempt %d (status=%s)",
                    task_id,
                    attempt_id,
                    existing.status,
                )
                return task_id_wire

            # Sanity check: find any non-terminal attempt for this task
            current = self._get_current_attempt(task_id_wire)
            if current is not None and current.status not in self._TERMINAL_STATES:
                if attempt_id <= current.attempt_id:
                    logger.info(
                        "Rejecting duplicate task %s (attempt %d, current attempt %d status=%s)",
                        task_id,
                        attempt_id,
                        current.attempt_id,
                        current.status,
                    )
                    return task_id_wire
                # New attempt with higher ID supersedes old one - kill the old attempt
                logger.info(
                    "Superseding task %s: attempt %d -> %d, killing old attempt",
                    task_id,
                    current.attempt_id,
                    attempt_id,
                )
                should_kill_existing = True

        if should_kill_existing:
            self._kill_task_attempt(task_id_wire, current.attempt_id)  # type: ignore[union-attr]

        task_id.require_task()
        num_tasks = request.num_tasks
        attempt_id = request.attempt_id

        # Allocate requested ports
        port_names = list(request.ports)
        allocated_ports = self._port_allocator.allocate(len(port_names)) if port_names else []
        ports = dict(zip(port_names, allocated_ports, strict=True))

        # Create task working directory with attempt isolation
        # Use safe path component for hierarchical task IDs (e.g., "/my-exp/0" -> "__my-exp__0")
        safe_task_id = task_id.to_safe_token()
        workdir = self._cache_dir / "workdirs" / f"{safe_task_id}_attempt_{attempt_id}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Create TaskAttempt to handle the full execution lifecycle
        config = TaskAttemptConfig(
            task_id=task_id,
            num_tasks=num_tasks,
            attempt_id=attempt_id,
            request=request,
            ports=ports,
            workdir=workdir,
            cache_dir=self._cache_dir,
            uv_cache_dir=self._uv_cache_dir,
        )

        attempt = TaskAttempt(
            config=config,
            bundle_provider=self._bundle_cache,
            container_runtime=self._runtime,
            worker_metadata=self._worker_metadata,
            worker_id=self._worker_id,
            controller_address=self._config.controller_address,
            port_allocator=self._port_allocator,
            report_state=lambda: self._notify_task_update(attempt),
            poll_interval_seconds=self._config.poll_interval.to_seconds(),
        )

        with self._lock:
            self._tasks[key] = attempt

        # Start execution in a monitored non-daemon thread. When stop() is called,
        # the on_stop callback kills the container so attempt.run() exits promptly.
        def _run_task(stop_event: threading.Event) -> None:
            attempt.run()

        def _stop_task() -> None:
            try:
                attempt.stop(force=True)
            except RuntimeError:
                pass

        mt = self._task_threads.spawn(target=_run_task, name=f"task-{task_id_wire}", on_stop=_stop_task)
        attempt.thread = mt._thread

        return task_id_wire

    def get_task(self, task_id: str, attempt_id: int = -1) -> TaskInfo | None:
        """Get a task by ID and optionally attempt ID.

        Args:
            task_id: Task identifier
            attempt_id: Specific attempt ID, or -1 to get the most recent attempt

        Returns:
            TaskInfo view (implemented by TaskAttempt) to decouple callers
            from execution internals.
        """
        if attempt_id >= 0:
            return self._tasks.get((task_id, attempt_id))
        return self._get_current_attempt(task_id)

    def list_tasks(self) -> list[TaskInfo]:
        """List all task attempts.

        Returns TaskInfo views (implemented by TaskAttempt) to decouple callers
        from execution internals. Returns all attempts, not just current ones.
        """
        return list(self._tasks.values())

    def list_current_tasks(self) -> list[TaskInfo]:
        """List only the most recent attempt for each task.

        Returns TaskInfo views for the current (highest attempt_id) attempt of each task.
        """
        # Group by task_id and return only the highest attempt_id for each
        by_task: dict[str, TaskAttempt] = {}
        for (task_id, attempt_id), task in self._tasks.items():
            existing = by_task.get(task_id)
            if existing is None or attempt_id > existing.attempt_id:
                by_task[task_id] = task
        return list(by_task.values())

    def handle_heartbeat(self, request: cluster_pb2.HeartbeatRequest) -> cluster_pb2.HeartbeatResponse:
        """Handle controller-initiated heartbeat with reconciliation.

        Processes tasks_to_run and tasks_to_kill, reconciles expected_tasks against
        actual state, and returns current running/completed tasks.
        """
        # Reset heartbeat deadline
        self._heartbeat_deadline = Deadline.from_seconds(self._config.heartbeat_timeout.to_seconds())

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
                expected_attempt_id = expected_entry.attempt_id
                key = (task_id, expected_attempt_id)
                task = self._tasks.get(key)

                if task is None:
                    # Task/attempt not found - report as WORKER_FAILED
                    completed_tasks.append(
                        cluster_pb2.Controller.CompletedTaskEntry(
                            task_id=task_id,
                            state=cluster_pb2.TASK_STATE_WORKER_FAILED,
                            exit_code=0,
                            error="Task not found on worker",
                            finished_at=Timestamp.now().to_proto(),
                            attempt_id=expected_attempt_id,
                        )
                    )
                elif task.status in terminal_states:
                    # Task is terminal - report as completed
                    task_proto = task.to_proto()
                    completed_tasks.append(
                        cluster_pb2.Controller.CompletedTaskEntry(
                            task_id=task_proto.task_id,
                            state=task_proto.state,
                            exit_code=task_proto.exit_code,
                            error=task_proto.error,
                            finished_at=task_proto.finished_at,
                            attempt_id=task_proto.current_attempt_id,
                        )
                    )
                else:
                    # Task is running/building - include in running_tasks
                    running_tasks.append(
                        cluster_pb2.Controller.RunningTaskEntry(
                            task_id=task_id,
                            attempt_id=task.to_proto().current_attempt_id,
                            state=task.status,
                        )
                    )

            # Kill tasks not in expected_tasks - the controller has decided these
            # tasks should no longer run (e.g., job was killed, task was reassigned)
            expected_keys = {(entry.task_id, entry.attempt_id) for entry in request.expected_tasks}
            tasks_to_kill: list[tuple[str, int]] = []
            for key, task in self._tasks.items():
                if key not in expected_keys and task.status not in terminal_states:
                    tasks_to_kill.append(key)

        # Kill removed tasks outside lock to avoid deadlock
        for task_id, attempt_id in tasks_to_kill:
            logger.warning("Killing task %s attempt %d (no longer in expected_tasks)", task_id, attempt_id)
            self._kill_task_attempt(task_id, attempt_id)

        return cluster_pb2.HeartbeatResponse(
            running_tasks=running_tasks,
            completed_tasks=completed_tasks,
        )

    def _kill_task_attempt(self, task_id: str, attempt_id: int, term_timeout_ms: int = 5000) -> bool:
        """Kill a specific task attempt."""
        task = self._tasks.get((task_id, attempt_id))
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

        # If container handle exists, try to stop it
        if task._container_handle:
            try:
                # Send SIGTERM (graceful stop)
                task._container_handle.stop(force=False)

                # Wait for shutdown
                running_states = (cluster_pb2.TASK_STATE_RUNNING, cluster_pb2.TASK_STATE_BUILDING)
                stopped = ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
                    lambda: task.status not in running_states,
                    timeout=Duration.from_ms(term_timeout_ms),
                )

                # Force kill if graceful shutdown timed out
                if not stopped:
                    try:
                        task._container_handle.stop(force=True)
                    except RuntimeError:
                        pass
            except RuntimeError:
                # Container may have already been removed or stopped
                pass

        return True

    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool:
        """Kill the current (most recent) attempt of a task."""
        current = self._get_current_attempt(task_id)
        if not current:
            return False
        return self._kill_task_attempt(task_id, current.attempt_id, term_timeout_ms)

    def get_logs(
        self,
        task_id: str,
        start_line: int = 0,
        attempt_id: int = -1,
    ) -> list[cluster_pb2.Worker.LogEntry]:
        """Get logs for a task.

        Logs are streamed into task.logs during execution (single source of truth).
        Container is removed after task completion to release TPU devices.

        Args:
            task_id: ID of the task to get logs for
            start_line: Line offset (supports negative indexing: -1000 = last 1000 lines)
            attempt_id: Specific attempt to get logs for (-1 = all attempts for this task)

        Returns:
            List of LogEntry protos with attempt_id populated.
        """
        if attempt_id >= 0:
            # Specific attempt requested
            task = self._tasks.get((task_id, attempt_id))
            if not task:
                return []
            logs = [log_line.to_proto() for log_line in task.logs.lines]
            for log in logs:
                log.attempt_id = attempt_id
            logs.sort(key=lambda x: x.timestamp.epoch_ms)
            return logs[start_line:]

        # All attempts for this task
        all_logs: list[cluster_pb2.Worker.LogEntry] = []
        for (tid, aid), task in self._tasks.items():
            if tid != task_id:
                continue
            for log_line in task.logs.lines:
                proto = log_line.to_proto()
                proto.attempt_id = aid
                all_logs.append(proto)

        all_logs.sort(key=lambda x: x.timestamp.epoch_ms)
        return all_logs[start_line:]

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self._config.port}"
