# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified worker managing all components and lifecycle."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

import uvicorn
from connectrpc.errors import ConnectError
from finelog.client import LogClient, RemoteLogHandler, StatsError, Table
from rigging.timing import Deadline, Duration, ExponentialBackoff, Timestamp

from iris.chaos import chaos
from iris.cluster.bundle import BundleStore
from iris.cluster.log_store_helpers import worker_log_key
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import ContainerRuntime, ExecutionStage
from iris.cluster.types import JobName
from iris.cluster.types import TaskAttempt as TaskAttemptId
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
from iris.cluster.worker.stats import (
    WORKER_STATS_NAMESPACE,
    WORKER_STATUS_IDLE,
    WORKER_STATUS_RUNNING,
    IrisWorkerStat,
    build_worker_stat,
)
from iris.cluster.worker.task_attempt import TaskAttempt, TaskAttemptConfig
from iris.cluster.worker.worker_types import TaskInfo
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import config_pb2, controller_pb2, job_pb2, worker_pb2
from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.time_proto import timestamp_to_proto

logger = logging.getLogger(__name__)


class _StatsState(StrEnum):
    """Lifecycle of the worker's iris.worker stats Table registration."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    FAILED = "failed"


def _now_dt() -> datetime:
    """Tz-naive UTC datetime for the stats namespace's TIMESTAMP_MS column."""
    return datetime.fromtimestamp(Timestamp.now().epoch_seconds(), tz=timezone.utc).replace(tzinfo=None)


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

    # Grace period during which a freshly-submitted task is treated as
    # "expected" by reconciliation, even if it hasn't yet appeared in the
    # controller's expected_tasks list. Protects against the StartTasks →
    # PollTasks race where the controller polls before its internal view
    # catches up with the task it just assigned.
    _RECENT_SUBMISSION_GRACE_SECONDS = 30.0

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
        # Freshly-submitted tasks -> monotonic submission time. Used by
        # reconciliation to grant a grace period before a task becomes
        # eligible for "unexpected, kill" if the controller hasn't yet
        # listed it in expected_tasks. See _RECENT_SUBMISSION_GRACE_SECONDS.
        self._recent_submissions: dict[tuple[str, int], float] = {}
        self._lock = threading.Lock()

        self._host_metrics = HostMetricsCollector(disk_path=str(self._cache_dir))

        # LogClient and RemoteLogHandler are created in start() before container
        # adoption and registration. Building before adoption ensures adopted
        # attempts capture a live client (regression #5261). Building before
        # registration ensures pre-register failures (container bring-up,
        # disk/health probes, registration rejection) leave remote logs.
        # Attachment relies on ``self._worker_id`` having been resolved locally
        # (IRIS_WORKER_ID, slice_id + TPU index, or GCE instance name); the rare
        # case where the controller assigns the id is handled by re-attaching
        # post-register.
        self._log_client: LogClient | None = None
        self._log_handler: RemoteLogHandler | None = None
        # Stats Table for the iris.worker namespace. Created lazily on the
        # first successful register; if registration ever fails (server
        # unreachable, version skew) we log the failure once and leave the
        # field None so subsequent heartbeats silently no-op. The dashboard
        # pane treats an empty roster as "no workers" rather than blocking.
        self._stats_table: Table | None = None
        self._stats_state: _StatsState = _StatsState.UNINITIALIZED

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
        # Build the LogClient *before* adopting containers so adopted attempts
        # capture a live client rather than a permanent ``None`` (regression #5261).
        # The client only depends on auth + the resolver callback, neither of
        # which need the uvicorn server or controller client to exist yet.
        interceptors: tuple[AuthTokenInjector, ...] = ()
        if self._config.controller_address and self._config.auth_token:
            interceptors = (AuthTokenInjector(StaticTokenProvider(self._config.auth_token)),)
        if self._config.controller_address:
            self._log_client = LogClient.connect(
                "/system/log-server",
                interceptors=interceptors,
                resolver=self._resolve_log_service,
            )

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
            self._controller_client = ControllerServiceClientSync(
                address=self._config.controller_address,
                timeout_ms=10_000,
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
                log_client=self._log_client,
                port_allocator=self._port_allocator,
                poll_interval_seconds=self._config.poll_interval.to_seconds(),
            )
            attempt.on_state_change = self._make_state_change_callback(attempt)

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
        if self._log_client is not None:
            self._log_client.close()
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

    def _resolve_log_service(self, server_url: str) -> str:
        """Look up ``server_url`` on the controller's endpoint registry."""
        if self._controller_client is None:
            raise ConnectionError("worker controller client not yet initialized")
        resp = self._controller_client.list_endpoints(
            controller_pb2.Controller.ListEndpointsRequest(prefix=server_url, exact=True),
        )
        if not resp.endpoints:
            raise ConnectionError(f"No {server_url!r} endpoint registered on controller")
        return resp.endpoints[0].address

    def _attach_log_handler(self) -> None:
        """Attach or rename the remote log handler under ``worker_log_key(self._worker_id)``."""
        if not self._worker_id or self._log_client is None:
            return
        key = worker_log_key(self._worker_id)
        if self._log_handler is None:
            self._log_handler = RemoteLogHandler(self._log_client, key=key)
            self._log_handler.setLevel(logging.INFO)
            self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
            logging.getLogger().addHandler(self._log_handler)
        else:
            self._log_handler.key = key

    def _detach_log_handler(self) -> None:
        """Remove and close the current RemoteLogHandler and LogClient if any.

        ``LogClient.close`` drains every open Table including the stats
        Table, so stats rows queued before shutdown still ship.
        """
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None
        # The stats Table belongs to LogClient; LogClient.close() drains it.
        # Drop our cached reference so a subsequent register starts fresh.
        self._stats_table = None
        self._stats_state = _StatsState.UNINITIALIZED
        if self._log_client is not None:
            self._log_client.close()
            self._log_client = None

    def _make_state_change_callback(self, attempt: TaskAttempt) -> Callable[[job_pb2.TaskState], None]:
        """Build a closure that pushes a WorkerTaskStatus to the controller on transition.

        Runs synchronously on the TaskAttempt's own thread. RPC failures are
        dropped — the controller's poll loop reconciles missed transitions.
        """

        def _on_state_change(new_state: job_pb2.TaskState) -> None:
            client = self._controller_client
            if client is None or not self._worker_id:
                return
            reported_state = new_state
            if reported_state == job_pb2.TASK_STATE_PENDING:
                reported_state = job_pb2.TASK_STATE_BUILDING
            entry = job_pb2.WorkerTaskStatus(
                task_id=attempt.task_id.to_wire(),
                attempt_id=attempt.attempt_id,
                state=reported_state,
                exit_code=attempt.exit_code or 0,
                error=attempt.error or "",
                container_id=attempt.platform_container_id or "",
            )
            if attempt.finished_at is not None:
                entry.finished_at.CopyFrom(timestamp_to_proto(attempt.finished_at))
            usage = job_pb2.ResourceUsage(
                memory_mb=attempt.current_memory_mb,
                memory_peak_mb=attempt.peak_memory_mb,
                disk_mb=attempt.disk_mb,
                cpu_millicores=attempt.current_cpu_millicores,
                process_count=attempt.process_count,
            )
            if usage.ByteSize() > 0:
                entry.resource_usage.CopyFrom(usage)
            try:
                client.update_task_status(
                    controller_pb2.Controller.UpdateTaskStatusRequest(
                        worker_id=self._worker_id,
                        updates=[entry],
                    )
                )
            except Exception as e:
                logger.warning("UpdateTaskStatus push failed for %s: %s", attempt.task_id, e)

        return _on_state_change

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
        """Wait for RPCs from controller. Returns when the controller-contact timeout expires.

        This method blocks in a loop, checking the time since the last
        controller RPC (Ping / PollTasks / StartTasks / StopTasks). When the
        timeout expires it returns, triggering a reset and re-registration.
        """
        self._heartbeat_deadline = Deadline.from_seconds(self._config.heartbeat_timeout.to_seconds())
        logger.info("Serving (waiting for controller RPCs)")

        while not stop_event.is_set():
            if self._heartbeat_deadline.expired():
                logger.warning("No contact from controller, resetting")
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
            self._recent_submissions.clear()

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
            log_client=self._log_client,
            poll_interval_seconds=self._config.poll_interval.to_seconds(),
        )
        attempt.on_state_change = self._make_state_change_callback(attempt)

        with self._lock:
            self._tasks[key] = attempt
            self._recent_submissions[key] = time.monotonic()

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

    def _encode_task_status(self, task: TaskAttempt, task_id: str) -> job_pb2.WorkerTaskStatus:
        """Build a WorkerTaskStatus proto from a worker-side TaskAttempt.

        Maps PENDING → BUILDING because the controller treats PENDING as
        "not yet picked up by a worker"; once a worker holds the task, the
        earliest visible state to the controller is BUILDING.
        """
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
        return entry

    @staticmethod
    def _missing_task_status(task_id: str, expected_attempt_id: int) -> job_pb2.WorkerTaskStatus:
        """Status for an expected task that the worker has no record of (lost state)."""
        return job_pb2.WorkerTaskStatus(
            task_id=task_id,
            attempt_id=expected_attempt_id,
            state=job_pb2.TASK_STATE_WORKER_FAILED,
            exit_code=0,
            error="Task not found on worker",
            finished_at=timestamp_to_proto(Timestamp.now()),
        )

    def _prune_and_get_recent_submission_keys(self) -> set[tuple[str, int]]:
        """Return keys submitted within the grace window, pruning stale entries.

        Caller must hold ``self._lock``. Stale entries (older than the grace
        window) are removed from ``self._recent_submissions`` so the dict
        doesn't grow unbounded.
        """
        now = time.monotonic()
        cutoff = now - self._RECENT_SUBMISSION_GRACE_SECONDS
        stale = [key for key, ts in self._recent_submissions.items() if ts < cutoff]
        for key in stale:
            del self._recent_submissions[key]
        return set(self._recent_submissions)

    def _reconcile_expected_tasks(
        self,
        expected_entries,
    ) -> tuple[list[job_pb2.WorkerTaskStatus], list[tuple[str, int]]]:
        """Build status entries for expected tasks; collect non-terminal local tasks
        not in the expected set as targets to kill.

        Caller must hold ``self._lock``.

        Freshly-submitted tasks (``self._recent_submissions``) are protected
        from reconciliation kills via the grace window, which covers the
        StartTasks → PollTasks race where the controller polls before its
        internal view catches up with a task it just assigned.
        """
        tasks: list[job_pb2.WorkerTaskStatus] = []
        expected_keys: set[tuple[str, int]] = set()
        for expected_entry in expected_entries:
            task_id = expected_entry.task_id
            expected_attempt_id = expected_entry.attempt_id
            key = (task_id, expected_attempt_id)
            expected_keys.add(key)
            task = self._tasks.get(key)
            if task is None:
                tasks.append(self._missing_task_status(task_id, expected_attempt_id))
            else:
                tasks.append(self._encode_task_status(task, task_id))
        expected_keys |= self._prune_and_get_recent_submission_keys()
        tasks_to_kill: list[tuple[str, int]] = []
        for key, task in self._tasks.items():
            if key not in expected_keys and task.status not in self._TERMINAL_STATES:
                tasks_to_kill.append(key)
        return tasks, tasks_to_kill

    def _collect_resource_metrics(self) -> job_pb2.WorkerResourceSnapshot:
        """Collect host metrics with running-task and process aggregates filled in."""
        snapshot = self._host_metrics.collect()
        running_count = 0
        total_processes = 0
        with self._lock:
            for task in self._tasks.values():
                if task.status == job_pb2.TASK_STATE_RUNNING:
                    running_count += 1
                    total_processes += task.process_count
        snapshot.running_task_count = running_count
        snapshot.total_process_count = total_processes
        return snapshot

    def handle_ping(self, request: worker_pb2.Worker.PingRequest) -> worker_pb2.Worker.PingResponse:
        """Liveness check. Resets heartbeat deadline, returns resource snapshot and health."""
        if rule := chaos("worker.ping"):
            if rule.delay_seconds > 0:
                time.sleep(rule.delay_seconds)
            if rule.error:
                raise rule.error
            if not rule.delay_seconds:
                raise RuntimeError("chaos: worker.ping")
        self._heartbeat_deadline = Deadline.from_seconds(self._config.heartbeat_timeout.to_seconds())
        resource_snapshot = self._collect_resource_metrics()
        health = check_worker_health(disk_path=str(self._cache_dir))
        if not health.healthy:
            logger.warning("Worker health check failed: %s", health.error)
        self._emit_worker_stat(resource_snapshot, healthy=health.healthy)
        return worker_pb2.Worker.PingResponse(
            resource_snapshot=resource_snapshot,
            healthy=health.healthy,
            health_error=health.error,
        )

    def _emit_worker_stat(self, snapshot: job_pb2.WorkerResourceSnapshot, *, healthy: bool) -> None:
        """Append one heartbeat row to the ``iris.worker`` stats namespace.

        Non-blocking: ``Table.write`` queues for the bg flush thread, so the
        ping path never waits on the stats service. Schema-register
        failures (server unreachable, namespace conflict) are logged once
        and the writer is dropped — the dashboard treats an empty roster
        as "no workers" rather than blocking the rest of the page.

        Schema-validation bugs (a missing/extra/typo'd field on
        ``IrisWorkerStat``) are intentionally NOT swallowed here — they
        propagate so the worker fails fast and the bug surfaces in tests
        rather than silently producing an empty stats namespace.
        """
        table = self._get_or_register_stats_table()
        if table is None:
            return
        if self._worker_id is None:
            return
        running = int(snapshot.running_task_count) if snapshot is not None else 0
        if running > 0 or self._tasks:
            status = WORKER_STATUS_RUNNING
        else:
            status = WORKER_STATUS_IDLE
        try:
            stat = build_worker_stat(
                worker_id=self._worker_id,
                ts=_now_dt(),
                status=status,
                address=self._resolve_address(),
                healthy=healthy,
                snapshot=snapshot,
                metadata=self._worker_metadata,
            )
            table.write([stat])
        except (StatsError, ConnectError, ConnectionError, OSError, TimeoutError) as exc:
            # Transport / stats-protocol failure: log and move on. The
            # heartbeat path must not block on the stats service.
            logger.warning("worker stat write failed: %s: %s", type(exc).__name__, exc)

    def _get_or_register_stats_table(self) -> Table | None:
        """Lazily register the iris.worker namespace and cache the Table.

        Returns ``None`` if registration has previously failed for this
        worker process (we do not retry register-time failures within one
        process — a misconfigured schema or unavailable stats endpoint is
        a startup-time problem).
        """
        if self._stats_state is _StatsState.READY:
            return self._stats_table
        if self._stats_state is _StatsState.FAILED:
            return None
        if self._log_client is None:
            return None
        try:
            self._stats_table = self._log_client.get_table(WORKER_STATS_NAMESPACE, IrisWorkerStat)
        except (StatsError, ConnectError, ConnectionError, OSError, TimeoutError) as exc:
            # Transport / stats-protocol failure: one-shot give up. Schema
            # bugs (TypeError, KeyError out of schema_from_dataclass) are
            # intentionally NOT caught — the worker should crash visibly so
            # the bug surfaces in tests / startup logs rather than the
            # dashboard silently rendering empty.
            logger.warning(
                "worker stats namespace register failed (giving up): %s: %s",
                type(exc).__name__,
                exc,
            )
            self._stats_state = _StatsState.FAILED
            return None
        self._stats_state = _StatsState.READY
        return self._stats_table

    def handle_start_tasks(self, request: worker_pb2.Worker.StartTasksRequest) -> worker_pb2.Worker.StartTasksResponse:
        """Start task attempts on this worker. Returns per-task ack."""
        acks = []
        for run_req in request.tasks:
            try:
                self.submit_task(run_req)
                logger.info("StartTasks: submitted task %s", run_req.task_id)
                acks.append(worker_pb2.Worker.TaskAck(task_id=run_req.task_id, accepted=True))
            except Exception as e:
                logger.warning("StartTasks: failed to submit task %s: %s", run_req.task_id, e)
                acks.append(worker_pb2.Worker.TaskAck(task_id=run_req.task_id, accepted=False, error=str(e)))
        return worker_pb2.Worker.StartTasksResponse(acks=acks)

    def handle_stop_tasks(self, request: worker_pb2.Worker.StopTasksRequest) -> worker_pb2.Worker.StopTasksResponse:
        """Stop given tasks on this worker."""
        for task_id in request.task_ids:
            try:
                current = self._get_current_attempt(task_id)
                if current:
                    self._kill_task_attempt(task_id, current.attempt_id, async_kill=True)
                    logger.info("StopTasks: initiated async kill for task %s", task_id)
            except Exception as e:
                logger.warning("StopTasks: failed to kill task %s: %s", task_id, e)
        return worker_pb2.Worker.StopTasksResponse()

    def handle_poll_tasks(self, request: worker_pb2.Worker.PollTasksRequest) -> worker_pb2.Worker.PollTasksResponse:
        """Report status of expected tasks and kill unexpected tasks.

        Freshly-submitted tasks (via StartTasks) are protected from the
        StartTasks → PollTasks race by the recent-submission grace window
        applied in _reconcile_expected_tasks.
        """
        with self._lock:
            tasks, tasks_to_kill = self._reconcile_expected_tasks(request.expected_tasks)
        for task_id, attempt_id in tasks_to_kill:
            logger.warning("PollTasks: killing task %s attempt %d (unexpected)", task_id, attempt_id)
            self._kill_task_attempt(task_id, attempt_id, async_kill=True)
        return worker_pb2.Worker.PollTasksResponse(tasks=tasks)

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
