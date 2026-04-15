# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified worker managing all components and lifecycle."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn

from iris.chaos import chaos
from iris.cluster.log_store import worker_log_key
from iris.cluster.runtime.docker import DockerRuntime
from iris.log_server.client import LogPusher, RemoteLogHandler
from iris.cluster.runtime.types import ContainerRuntime, ExecutionStage
from iris.cluster.types import JobName, TaskAttempt as TaskAttemptId
from iris.cluster.bundle import BundleStore
from iris.cluster.worker.dashboard import WorkerDashboard
from iris.cluster.worker.env_probe import (
    EnvironmentProvider,
    HardwareProbe,
    HostMetricsCollector,
    build_worker_metadata,
    check_worker_health,
    construct_worker_id,
    infer_worker_id,
    probe_disk_writable,
    probe_hardware,
)
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.task_attempt import TaskAttempt, TaskAttemptConfig
from iris.cluster.worker.worker_types import TaskInfo
from rigging.log_setup import slow_log
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import config_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc import worker_pb2
from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.time_proto import timestamp_to_proto
from rigging.timing import Deadline, Duration, ExponentialBackoff, Timestamp

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
    slice_id: str | None = None
    worker_attributes: dict[str, str] = field(default_factory=dict)
    task_env: dict[str, str] = field(default_factory=dict)
    default_task_image: str | None = None
    resolve_image: Callable[[str], str] = field(default_factory=lambda: lambda image: image)
    poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5.0))
    heartbeat_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(600.0))
    accelerator_type: int = 0
    accelerator_variant: str = ""
    gpu_count: int = 0
    capacity_type: int = 0
    storage_prefix: str = ""
    auth_token: str = ""


def worker_config_from_proto(
    proto: config_pb2.WorkerConfig,
    resolve_image: Callable[[str], str] | None = None,
) -> WorkerConfig:
    """Create internal WorkerConfig from WorkerConfig proto.

    Translates the proto representation into the internal dataclass,
    applying defaults where proto fields are unset.
    """
    port_start, port_end = 30000, 40000
    if proto.port_range:
        port_start, port_end = map(int, proto.port_range.split("-"))

    controller_address = proto.controller_address
    if controller_address and not controller_address.startswith("http"):
        controller_address = f"http://{controller_address}"

    return WorkerConfig(
        host=proto.host or "0.0.0.0",
        port=proto.port or 8080,
        cache_dir=Path(proto.cache_dir) if proto.cache_dir else None,
        port_range=(port_start, port_end),
        controller_address=controller_address or None,
        worker_id=proto.worker_id or None,
        slice_id=proto.slice_id or None,
        worker_attributes=dict(proto.worker_attributes),
        task_env=dict(proto.task_env),
        default_task_image=proto.default_task_image or None,
        resolve_image=resolve_image or (lambda image: image),
        poll_interval=(
            Duration.from_ms(proto.poll_interval.milliseconds)
            if proto.HasField("poll_interval")
            else Duration.from_seconds(5.0)
        ),
        heartbeat_timeout=(
            Duration.from_ms(proto.heartbeat_timeout.milliseconds)
            if proto.HasField("heartbeat_timeout")
            else Duration.from_seconds(600.0)
        ),
        accelerator_type=proto.accelerator_type,
        accelerator_variant=proto.accelerator_variant,
        gpu_count=proto.gpu_count,
        capacity_type=proto.capacity_type,
        storage_prefix=proto.storage_prefix,
        auth_token=proto.auth_token,
    )


class Worker:
    """Unified worker managing all components and lifecycle."""

    def __init__(
        self,
        config: WorkerConfig,
        bundle_store: BundleStore | None = None,
        container_runtime: ContainerRuntime | None = None,
        environment_provider: EnvironmentProvider | None = None,
        port_allocator: PortAllocator | None = None,
        threads: ThreadContainer | None = None,
        worker_metadata: job_pb2.WorkerMetadata | None = None,
    ):
        self._config = config

        if not config.cache_dir:
            raise ValueError("WorkerConfig.cache_dir is required")
        self._cache_dir = config.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Probe cache-dir writability once at startup. Failures propagate so
        # the worker aborts and the controller reaps the machine; heartbeats
        # deliberately do not repeat this probe (see #4732).
        probe_disk_writable(str(self._cache_dir))

        # Use overrides if provided, otherwise create defaults
        self._bundle_store = bundle_store or BundleStore(
            storage_dir=str(self._cache_dir / "bundles"),
            controller_address=config.controller_address,
            max_cache_items=100,
        )
        self._runtime = container_runtime or DockerRuntime(cache_dir=self._cache_dir)
        self._port_allocator = port_allocator or PortAllocator(config.port_range)

        # Resolve worker metadata: explicit > environment_provider > hardware probe
        hardware: HardwareProbe | None = None
        if worker_metadata is not None:
            self._worker_metadata = worker_metadata
        elif environment_provider is not None:
            self._worker_metadata = environment_provider.probe()
        else:
            hardware = probe_hardware()
            self._worker_metadata = build_worker_metadata(
                hardware=hardware,
                accelerator_type=config.accelerator_type,
                accelerator_variant=config.accelerator_variant,
                gpu_count_override=config.gpu_count,
                capacity_type=config.capacity_type,
                worker_attributes=config.worker_attributes,
            )

        # Task state: maps (task_id, attempt_id) -> TaskAttempt.
        # Preserves all attempts so logs for historical attempts remain accessible.
        self._tasks: dict[tuple[str, int], TaskAttempt] = {}
        self._lock = threading.Lock()

        self._host_metrics = HostMetricsCollector(disk_path=str(self._cache_dir))

        # LogPusher and RemoteLogHandler are created before registration so
        # pre-register failures (container bring-up, disk/health probes,
        # registration rejection) leave remote logs. Attachment relies on
        # ``self._worker_id`` having been resolved locally (IRIS_WORKER_ID,
        # slice_id + TPU index, or GCE instance name); the rare case where
        # the controller assigns the id is handled by re-attaching post-
        # register.
        self._log_pusher: LogPusher | None = None
        self._log_handler: RemoteLogHandler | None = None

        self._service = WorkerServiceImpl(self)
        self._dashboard = WorkerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        self._server: uvicorn.Server | None = None
        self._threads = threads if threads is not None else get_thread_container()
        self._task_threads = self._threads.create_child("tasks")

        # Resolve worker_id: config > slice_id + TPU index > GCP metadata inference > assigned by controller
        worker_id = config.worker_id
        if worker_id is None and config.slice_id and hardware is not None:
            worker_index = int(hardware.tpu_worker_id) if hardware.tpu_worker_id else 0
            worker_id = construct_worker_id(config.slice_id, worker_index)
        elif worker_id is None and hardware is not None:
            worker_id = infer_worker_id(hardware)
        self._worker_id: str | None = worker_id
        self._controller_client: ControllerServiceClientSync | None = None

        # Heartbeat tracking for timeout detection
        self._heartbeat_deadline = Deadline.from_seconds(float("inf"))

    def start(self) -> None:
        # Try to adopt running containers from a previous worker process.
        # If adoption succeeds, skip the destructive cleanup that would kill them.
        adopted = self.adopt_running_containers()
        if adopted == 0:
            self._cleanup_all_iris_containers()

        # Start HTTP server
        # timeout_keep_alive=120: default 5s races with controller heartbeat intervals,
        # causing TCP resets on idle connections.
        self._server = uvicorn.Server(
            uvicorn.Config(
                self._dashboard.app,
                host=self._config.host,
                port=self._config.port,
                log_level="error",
                log_config=None,
                timeout_keep_alive=120,
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
            interceptors = ()
            if self._config.auth_token:
                interceptors = (AuthTokenInjector(StaticTokenProvider(self._config.auth_token)),)
            self._controller_client = ControllerServiceClientSync(
                address=self._config.controller_address,
                timeout_ms=5000,
                interceptors=interceptors,
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

    def adopt_running_containers(self) -> int:
        """Discover and adopt running containers from a previous worker process.

        Inspects Docker containers labeled with iris metadata. Running containers
        whose worker_id matches this worker are adopted — a TaskAttempt is created
        in RUNNING state and a monitoring thread is spawned. Non-adoptable
        containers (build-phase, exited, wrong worker_id) are removed.

        Returns the count of successfully adopted containers.
        """
        discovered = self._runtime.discover_containers()
        if not discovered:
            return 0

        adopted = 0
        to_remove: list[str] = []

        for container in discovered:
            if container.phase != ExecutionStage.RUN or not container.running:
                to_remove.append(container.container_id)
                continue

            # Only adopt containers from this worker. Containers with no worker_id
            # label (pre-adoption-era or unset) are adopted by any worker — this is
            # intentional for backward compatibility with containers created before
            # the worker_id label was added.
            if self._worker_id and container.worker_id and container.worker_id != self._worker_id:
                to_remove.append(container.container_id)
                continue

            # Create a handle wrapping the existing container
            try:
                handle = self._runtime.adopt_container(container.container_id)
            except NotImplementedError:
                logger.warning("Container adoption not supported by this runtime")
                continue
            attempt = TaskAttempt.adopt(
                discovered=container,
                container_handle=handle,
                log_pusher=self._log_pusher,
                port_allocator=self._port_allocator,
                poll_interval_seconds=self._config.poll_interval.to_seconds(),
            )

            key = (container.task_id, container.attempt_id)
            with self._lock:
                self._tasks[key] = attempt

            # Spawn monitoring thread
            def _run_adopted(stop_event: threading.Event, a: TaskAttempt = attempt) -> None:
                a.resume_monitoring()

            def _stop_adopted(a: TaskAttempt = attempt) -> None:
                try:
                    a.stop(force=True)
                except RuntimeError:
                    pass

            self._task_threads.spawn(
                target=_run_adopted,
                name=f"adopted-{container.task_id}",
                on_stop=_stop_adopted,
            )

            adopted += 1
            logger.info(
                "Adopted container %s for task %s attempt %d",
                container.container_id[:12],
                container.task_id,
                container.attempt_id,
            )

        # Clean up non-adoptable containers
        if to_remove:
            removed = self._runtime.remove_containers(to_remove)
            logger.info("Cleaned up %d non-adoptable containers", removed)

        if adopted > 0:
            logger.info("Adopted %d running containers from previous worker process", adopted)

        return adopted

    def wait(self) -> None:
        self._threads.wait()

    def stop(self, preserve_containers: bool = False) -> None:
        """Stop the worker.

        Args:
            preserve_containers: When True, stop the worker process but leave
                Docker containers running so a new worker can adopt them.
                Used during rolling restarts.
        """
        if not preserve_containers:
            # Stop task threads first so running tasks exit before infrastructure
            # tears down. ThreadContainer.stop() signals each thread's stop_event,
            # which the _run_task watcher bridges to attempt.should_stop + container kill.
            self._task_threads.stop()
        else:
            logger.info("Preserving %d running containers for adoption by new worker", len(self._tasks))
            # Detach task threads from the parent so _threads.stop() won't
            # cascade into _task_threads and trigger on_stop container kills.
            self._threads.detach_child(self._task_threads)

        if self._server:
            self._server.should_exit = True
        self._threads.stop()
        if self._controller_client:
            self._controller_client.close()
        self._detach_log_handler()
        if self._log_pusher is not None:
            self._log_pusher.close()
        self._bundle_store.close()

    def _run_lifecycle(self, stop_event: threading.Event) -> None:
        """Main lifecycle: register, serve, reset, repeat.

        This loop runs continuously until shutdown. On each iteration:
        1. Reset worker state (kill all containers)
        2. Attach the remote log handler so pre-registration log lines ship
           to the central log server (and a refreshed /system/log-server
           endpoint is picked up after any log-server failover)
        3. Register with controller (retry until accepted)
        4. If the controller assigned a worker_id we didn't know locally,
           re-attach the handler under the canonical key
        5. Serve (wait for heartbeats from controller)
        6. If heartbeat timeout expires, return to step 1

        On the first iteration after a restart with adopted containers,
        step 1 is skipped to preserve the running tasks.
        """
        first_iteration = True
        try:
            while not stop_event.is_set():
                # Skip reset on the first iteration if we adopted containers,
                # to avoid killing the tasks we just took over.
                if first_iteration and self._tasks:
                    logger.info("Skipping reset: %d adopted tasks present", len(self._tasks))
                else:
                    self._reset_worker_state()
                first_iteration = False
                self._attach_log_handler()
                worker_id = self._register(stop_event)
                if worker_id is None:
                    # Shutdown requested during registration
                    break
                if worker_id != self._worker_id:
                    self._worker_id = worker_id
                    self._attach_log_handler()
                self._serve(stop_event)
        except Exception:
            logger.exception("Worker lifecycle crashed")
            raise

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
                    controller_pb2.Controller.RegisterRequest(
                        address=address,
                        metadata=metadata,
                        worker_id=self._worker_id or "",
                        slice_id=self._config.slice_id or "",
                        scale_group=self._config.worker_attributes.get("scale-group", ""),
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

    def _resolve_log_service(self) -> str | None:
        """Resolve the LogService address via the /system/log-server endpoint.

        Called before registration, so the controller may not yet be reachable.
        Treats RPC errors and missing endpoints the same: log a warning and
        return None so the caller can skip remote log attachment without
        crashing the lifecycle thread.
        """
        if not self._controller_client:
            return None
        try:
            resp = self._controller_client.list_endpoints(
                controller_pb2.Controller.ListEndpointsRequest(
                    prefix="/system/log-server",
                    exact=True,
                ),
            )
        except Exception as e:
            logger.warning("Failed to resolve /system/log-server: %s", e)
            return None
        if not resp.endpoints:
            logger.warning("No /system/log-server endpoint registered on controller")
            return None
        addr = resp.endpoints[0].address
        logger.info("Resolved /system/log-server -> %s", addr)
        return addr

    def _attach_log_handler(self) -> None:
        """Create LogPusher and attach RemoteLogHandler under ``worker_log_key``.

        Always tears down any existing handler first so each lifecycle cycle
        re-resolves /system/log-server (picking up log-server failover) and
        rebuilds the LogPusher against the fresh address.

        Skipped when ``self._worker_id`` is not yet known locally — in that
        (rare) case the controller will assign an id during ``_register`` and
        the lifecycle loop re-calls this method with the canonical id.
        """
        self._detach_log_handler()
        if not self._worker_id:
            return
        log_addr = self._resolve_log_service()
        if not log_addr:
            return
        log_interceptors = ()
        if self._config.auth_token:
            log_interceptors = (AuthTokenInjector(StaticTokenProvider(self._config.auth_token)),)
        self._log_pusher = LogPusher(log_addr, interceptors=log_interceptors)
        self._log_handler = RemoteLogHandler(
            self._log_pusher,
            key=worker_log_key(self._worker_id),
        )
        self._log_handler.setLevel(logging.INFO)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger().addHandler(self._log_handler)

    def _detach_log_handler(self) -> None:
        """Remove and close the current RemoteLogHandler and LogPusher if any."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None
        if self._log_pusher is not None:
            self._log_pusher.close()
            self._log_pusher = None

    def _resolve_address(self) -> str:
        """Resolve the address to advertise to the controller."""
        metadata = self._worker_metadata

        # Determine the address to advertise to the controller.
        # If host is 0.0.0.0 (bind to all interfaces), use the probed IP for external access.
        # Otherwise, use the configured host.
        address_host = self._config.host
        if address_host == "0.0.0.0":
            address_host = metadata.ip_address

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
        """Reset worker state: stop task threads, wipe containers, clear tracking."""
        logger.info("Resetting worker state")

        # Stop all running task threads so they exit cleanly before we
        # kill containers.  Without this, orphaned threads discover their
        # containers are gone and log confusing "Container not found" errors.
        self._task_threads.stop()

        # Clear task tracking
        with self._lock:
            self._tasks.clear()

        # Replace the task thread container so new tasks get a fresh group.
        self._task_threads = self._threads.create_child("tasks")

        # Wipe ALL iris containers (simple, no tracking needed)
        self._cleanup_all_iris_containers()

        logger.info("Worker state reset complete")

    # Task management methods

    _TERMINAL_STATES = frozenset(
        {
            job_pb2.TASK_STATE_SUCCEEDED,
            job_pb2.TASK_STATE_FAILED,
            job_pb2.TASK_STATE_KILLED,
            job_pb2.TASK_STATE_WORKER_FAILED,
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

    def submit_task(self, request: job_pb2.RunTaskRequest) -> str:
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

        # Create a minimal TaskAttemptConfig. Expensive setup (port allocation,
        # workdir creation, log sink init) is deferred to TaskAttempt.run() so
        # the heartbeat RPC returns quickly.
        config = TaskAttemptConfig(
            task_attempt=TaskAttemptId(task_id=task_id, attempt_id=attempt_id),
            num_tasks=request.num_tasks,
            request=request,
            cache_dir=self._cache_dir,
        )

        attempt = TaskAttempt(
            config=config,
            bundle_store=self._bundle_store,
            container_runtime=self._runtime,
            worker_metadata=self._worker_metadata,
            worker_id=self._worker_id,
            controller_address=self._config.controller_address,
            task_env=self._config.task_env,
            default_task_image=self._config.default_task_image,
            resolve_image=self._config.resolve_image,
            port_allocator=self._port_allocator,
            log_pusher=self._log_pusher,
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

    def handle_heartbeat(self, request: job_pb2.HeartbeatRequest) -> job_pb2.HeartbeatResponse:
        """Handle controller-initiated heartbeat with reconciliation.

        Processing order (sequential, not concurrent):
        1. Submit tasks_to_run — registers each task in self._tasks
        2. Kill tasks_to_kill — synchronously, blocks until old process is stopped
           so the controller does not assign new work while old tasks hold resources
        3. Reconcile expected_tasks — for each expected task, report its current
           state. If not found in self._tasks, report WORKER_FAILED ("Task not
           found on worker"). This happens when the worker has reset its state
           (_tasks.clear() in _reset_worker_state) between heartbeats — from
           the controller's perspective this is equivalent to a worker restart.
        4. Kill unexpected tasks — any task in self._tasks that is NOT in
           expected_tasks or tasks_to_run is killed (controller no longer wants it)

        The ordering guarantee between steps 1 and 3 is critical: a task that
        appears in both tasks_to_run and expected_tasks (which is always the case
        for newly-assigned tasks) will be submitted before reconciliation checks
        for it, so it will be found.

        Reconciliation kills (step 4) remain async to avoid blocking the heartbeat
        for stale-task cleanup that is not part of a preemption handoff.
        """
        # Reset heartbeat deadline
        self._heartbeat_deadline = Deadline.from_seconds(self._config.heartbeat_timeout.to_seconds())

        with slow_log(logger, "handle_heartbeat", threshold_ms=2000):
            # Start new tasks
            with slow_log(logger, "heartbeat submit_tasks", threshold_ms=200):
                for run_req in request.tasks_to_run:
                    try:
                        with slow_log(logger, f"heartbeat submit_task[{run_req.task_id}]", threshold_ms=500):
                            self.submit_task(run_req)
                        logger.info("Heartbeat: submitted task %s", run_req.task_id)
                    except Exception as e:
                        logger.warning("Heartbeat: failed to submit task %s: %s", run_req.task_id, e)

            # Kill requested tasks synchronously so the old process is fully stopped
            # before the heartbeat returns. This prevents the controller from assigning
            # new work while the old task still holds accelerator resources (TPU/GPU chips).
            with slow_log(logger, "heartbeat kill_tasks", threshold_ms=5000):
                for task_id in request.tasks_to_kill:
                    try:
                        current = self._get_current_attempt(task_id)
                        if current:
                            with slow_log(logger, f"heartbeat kill_task[{task_id}]", threshold_ms=2000):
                                self._kill_task_attempt(task_id, current.attempt_id, async_kill=False)
                            logger.info("Heartbeat: killed task %s", task_id)
                    except Exception as e:
                        logger.warning("Heartbeat: failed to kill task %s: %s", task_id, e)

            tasks: list[job_pb2.WorkerTaskStatus] = []

            with slow_log(logger, "heartbeat reconciliation", threshold_ms=200):
                with self._lock:
                    # Reconcile expected_tasks against actual state
                    for expected_entry in request.expected_tasks:
                        task_id = expected_entry.task_id
                        expected_attempt_id = expected_entry.attempt_id
                        key = (task_id, expected_attempt_id)
                        task = self._tasks.get(key)

                        if task is None:
                            tasks.append(
                                job_pb2.WorkerTaskStatus(
                                    task_id=task_id,
                                    attempt_id=expected_attempt_id,
                                    state=job_pb2.TASK_STATE_WORKER_FAILED,
                                    exit_code=0,
                                    error="Task not found on worker",
                                    finished_at=timestamp_to_proto(Timestamp.now()),
                                )
                            )
                        else:
                            task_proto = task.to_proto()
                            reported_state = task.status
                            if reported_state == job_pb2.TASK_STATE_PENDING:
                                reported_state = job_pb2.TASK_STATE_BUILDING

                            entry = job_pb2.WorkerTaskStatus(
                                task_id=task_id,
                                attempt_id=task_proto.current_attempt_id,
                                state=reported_state,
                                exit_code=task_proto.exit_code,
                                error=task_proto.error or "",
                                container_id=task_proto.container_id or "",
                            )
                            if task.status in self._TERMINAL_STATES:
                                entry.finished_at.CopyFrom(task_proto.finished_at)
                            if task_proto.resource_usage.ByteSize() > 0:
                                entry.resource_usage.CopyFrom(task_proto.resource_usage)
                            tasks.append(entry)

                    # Kill tasks not in expected_tasks - the controller has decided these
                    # tasks should no longer run (e.g., job was killed, task was reassigned).
                    # Include tasks_to_run in the expected set: these were just submitted
                    # in this heartbeat and may not yet appear in expected_tasks if the
                    # controller excludes unconfirmed tasks.
                    expected_keys = {(entry.task_id, entry.attempt_id) for entry in request.expected_tasks}
                    for run_req in request.tasks_to_run:
                        expected_keys.add((run_req.task_id, run_req.attempt_id))
                    tasks_to_kill: list[tuple[str, int]] = []
                    for key, task in self._tasks.items():
                        if key not in expected_keys and task.status not in self._TERMINAL_STATES:
                            tasks_to_kill.append(key)

                # Kill removed tasks asynchronously outside lock to avoid deadlock
                for task_id, attempt_id in tasks_to_kill:
                    logger.warning("Killing task %s attempt %d (no longer in expected_tasks)", task_id, attempt_id)
                    self._kill_task_attempt(task_id, attempt_id, async_kill=True)

            # Collect host metrics and aggregate task stats
            with slow_log(logger, "heartbeat host_metrics", threshold_ms=100):
                resource_snapshot = self._host_metrics.collect()
                running_count = 0
                total_processes = 0
                with self._lock:
                    for task in self._tasks.values():
                        if task.status == job_pb2.TASK_STATE_RUNNING:
                            running_count += 1
                            total_processes += task.process_count
                resource_snapshot.running_task_count = running_count
                resource_snapshot.total_process_count = total_processes

            # Run health checks to detect local faults (disk full, write failure)
            with slow_log(logger, "heartbeat health_check", threshold_ms=100):
                health = check_worker_health(disk_path=str(self._cache_dir))
                if not health.healthy:
                    logger.warning("Worker health check failed: %s", health.error)

            return job_pb2.HeartbeatResponse(
                tasks=tasks,
                resource_snapshot=resource_snapshot,
                worker_healthy=health.healthy,
                health_error=health.error,
            )

    def _kill_task_attempt(
        self,
        task_id: str,
        attempt_id: int,
        term_timeout_ms: int = 5000,
        async_kill: bool = False,
    ) -> bool:
        """Kill a specific task attempt.

        Args:
            task_id: Wire-format task ID.
            attempt_id: Attempt number to kill.
            term_timeout_ms: Time to wait for graceful shutdown before SIGKILL.
            async_kill: If True, signal the task immediately but perform the
                container stop/wait/force-kill sequence in a daemon thread.
                Used by heartbeat to avoid blocking the RPC response.
        """
        task = self._tasks.get((task_id, attempt_id))
        if not task:
            return False

        # Check if already in terminal state
        if task.status not in (
            job_pb2.TASK_STATE_RUNNING,
            job_pb2.TASK_STATE_BUILDING,
            job_pb2.TASK_STATE_PENDING,
        ):
            return False

        # Set flag to signal the task's execution thread to stop.
        # This is always done immediately regardless of async_kill.
        task.should_stop = True

        if async_kill:
            thread = threading.Thread(
                target=self._do_kill_container,
                args=(task, term_timeout_ms),
                name=f"kill-{task_id}-{attempt_id}",
                daemon=True,
            )
            thread.start()
        else:
            self._do_kill_container(task, term_timeout_ms)

        return True

    @staticmethod
    def _do_kill_container(task: TaskAttempt, term_timeout_ms: int) -> None:
        """Perform the SIGTERM -> wait -> SIGKILL sequence for a task's container."""
        if not task.has_container:
            return

        try:
            task.stop(force=False)

            running_states = (job_pb2.TASK_STATE_RUNNING, job_pb2.TASK_STATE_BUILDING)
            stopped = ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
                lambda: task.status not in running_states,
                timeout=Duration.from_ms(term_timeout_ms),
            )

            if not stopped:
                try:
                    task.stop(force=True)
                except RuntimeError:
                    pass
        except RuntimeError:
            # Container may have already been removed or stopped
            pass

    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool:
        """Kill the current (most recent) attempt of a task."""
        current = self._get_current_attempt(task_id)
        if not current:
            return False
        return self._kill_task_attempt(task_id, current.attempt_id, term_timeout_ms)

    def profile_task(
        self,
        task_id: str,
        duration_seconds: int,
        profile_type: job_pb2.ProfileType,
        attempt_id: int | None = None,
    ) -> bytes:
        """Profile a running task by delegating to its container handle.

        Args:
            task_id: Bare task ID (e.g. ``/alice/job/0``).
            duration_seconds: How long to sample.
            profile_type: CPU, memory, or threads profiler config.
            attempt_id: Specific attempt to profile.  When ``None``, the
                current (most recent) attempt is used.
        """
        if attempt_id is not None:
            attempt = self._tasks.get((task_id, attempt_id))
            if not attempt:
                raise ValueError(f"Task {task_id} attempt {attempt_id} not found")
        else:
            attempt = self._get_current_attempt(task_id)
            if not attempt:
                raise ValueError(f"Task {task_id} not found")
        if attempt.status != job_pb2.TASK_STATE_RUNNING:
            raise ValueError(f"Task {task_id} is not running (state={job_pb2.TaskState.Name(attempt.status)})")
        return attempt.profile(duration_seconds, profile_type)

    def exec_in_container(
        self, task_id: str, command: list[str], timeout_seconds: int = 60
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        """Execute a command in a running task's container.

        Delegates to the container handle's underlying runtime (docker exec, subprocess, kubectl exec).
        """
        attempt = self._get_current_attempt(task_id)
        if not attempt:
            return worker_pb2.Worker.ExecInContainerResponse(error=f"Task {task_id} not found")
        if attempt.status != job_pb2.TASK_STATE_RUNNING:
            return worker_pb2.Worker.ExecInContainerResponse(
                error=f"Task {task_id} is not running (state={job_pb2.TaskState.Name(attempt.status)})"
            )
        container_id = attempt.container_id
        if not container_id:
            return worker_pb2.Worker.ExecInContainerResponse(error=f"Task {task_id} has no container")
        return attempt.exec_in_container(command, timeout_seconds)

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self._config.port}"
