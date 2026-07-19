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
from finelog.client import LogClient, RemoteLogHandler, Table
from rigging.auth import BearerTokenInjector, StaticTokenProvider
from rigging.timing import Deadline, Duration, ExponentialBackoff, RateLimiter

from iris.chaos import chaos
from iris.cluster.bundle import BundleStore
from iris.cluster.config import WorkerConfig as WorkerWireConfig
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.log_keys import worker_log_key
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.profile import (
    PROFILE_NAMESPACE,
    IrisProfile,
    ProfileTrigger,
    build_profile_row,
    profile_local_process,
)
from iris.cluster.runtime.types import ContainerRuntime, ExecutionStage
from iris.cluster.types import AcceleratorType, AttemptUid, CapacityType, JobName
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
from iris.cluster.worker.port_allocator import DEFAULT_TASK_PORT_RANGE, PortAllocator
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.stats import (
    TASK_STATS_NAMESPACE,
    WORKER_STATS_NAMESPACE,
    IrisTaskStat,
    IrisWorkerStat,
    WorkerStatus,
    build_worker_stat,
)
from iris.cluster.worker.task_attempt import TaskAttempt, TaskAttemptConfig
from iris.cluster.worker.worker_types import TaskInfo
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.time_proto import timestamp_to_proto

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Worker configuration."""

    host: str = "127.0.0.1"
    port: int = 0
    cache_dir: Path | None = None
    port_range: tuple[int, int] = DEFAULT_TASK_PORT_RANGE
    controller_address: str | None = None
    worker_id: str | None = None
    slice_id: str | None = None
    worker_attributes: dict[str, str] = field(default_factory=dict)
    task_env: dict[str, str] = field(default_factory=dict)
    default_task_image: str | None = None
    resolve_image: Callable[[str], str] = field(default_factory=lambda: lambda image: image)
    poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5.0))
    heartbeat_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(600.0))
    accelerator_type: AcceleratorType | None = None
    accelerator_variant: str = ""
    gpu_count: int = 0
    capacity_type: CapacityType | None = None
    cpu_millicores: int = 0
    storage_prefix: str = ""
    auth_token: str = ""


def worker_config_from_wire(
    wire: WorkerWireConfig,
    resolve_image: Callable[[str], str] | None = None,
) -> WorkerConfig:
    """Create the internal WorkerConfig from the wire (bootstrap) WorkerConfig.

    Translates the parsed config model into the internal dataclass, applying
    defaults where fields are unset.
    """
    port_start, port_end = DEFAULT_TASK_PORT_RANGE
    if wire.port_range:
        port_start, port_end = map(int, wire.port_range.split("-"))

    controller_address = wire.controller_address
    if controller_address and not controller_address.startswith("http"):
        controller_address = f"http://{controller_address}"

    return WorkerConfig(
        host=wire.host or "0.0.0.0",
        port=wire.port or 8080,
        cache_dir=Path(wire.cache_dir) if wire.cache_dir else None,
        port_range=(port_start, port_end),
        controller_address=controller_address or None,
        worker_id=wire.worker_id or None,
        slice_id=wire.slice_id or None,
        worker_attributes=dict(wire.worker_attributes),
        task_env=dict(wire.task_env),
        default_task_image=wire.default_task_image or None,
        resolve_image=resolve_image or (lambda image: image),
        poll_interval=wire.poll_interval if wire.poll_interval is not None else Duration.from_seconds(5.0),
        heartbeat_timeout=wire.heartbeat_timeout if wire.heartbeat_timeout is not None else Duration.from_seconds(600.0),
        accelerator_type=wire.accelerator_type,
        accelerator_variant=wire.accelerator_variant,
        gpu_count=wire.gpu_count,
        capacity_type=wire.capacity_type,
        cpu_millicores=wire.cpu_millicores,
        storage_prefix=wire.storage_prefix,
        auth_token=wire.auth_token,
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
        profile_interval: Duration = Duration.from_seconds(600),
        profile_duration_seconds: int = 10,
    ):
        self._config = config
        self._profile_interval = profile_interval
        self._profile_duration_seconds = profile_duration_seconds

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
        self._runtime = container_runtime or DockerRuntime(cache_dir=self._cache_dir, capacity_type=config.capacity_type)
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
                cpu_millicores=config.cpu_millicores,
            )

        # Task state: a flat list of TaskAttempt. Each attempt carries its
        # own attempt_uid and is resolved by UID. Preserves all attempts so
        # logs for historical attempts remain accessible. O(10) elements;
        # linear scans are cheap.
        self._tasks: list[TaskAttempt] = []
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
        # Stats Tables for the iris.worker / iris.task / iris.profile namespaces.
        # Set in start() after the controller client is built so the LogClient
        # resolver works.
        self._worker_stats_table: Table | None = None
        self._task_stats_table: Table | None = None
        self._profile_table: Table | None = None

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
        # Ordering matters here. Three invariants drive it:
        #   1. LogClient must exist before adoption so adopted attempts capture
        #      a live client (regression #5261). LogClient.connect is pure
        #      construction — no I/O — so it can come first cheaply.
        #   2. iris.worker / iris.task tables must be registered before adoption
        #      runs. TaskAttempt.__init__ eagerly calls log_client.get_table,
        #      which goes through the resolver (_resolve_log_service) — and the
        #      resolver requires self._controller_client to be set. After the
        #      controller_client is built and the tables are registered once,
        #      the per-attempt get_table inside adoption is a cache hit.
        #   3. The uvicorn server must be up before we register with the
        #      controller, so the controller's first ping lands on a ready
        #      worker. Lifecycle thread is spawned last for that reason.
        interceptors: tuple[BearerTokenInjector, ...] = ()
        if self._config.controller_address and self._config.auth_token:
            interceptors = (BearerTokenInjector(StaticTokenProvider(self._config.auth_token), "authorization"),)

        if self._config.controller_address:
            self._log_client = LogClient.connect(
                LOG_SERVER_ENDPOINT_NAME,
                interceptors=interceptors,
                resolver=self._resolve_log_service,
            )
            self._controller_client = ControllerServiceClientSync(
                address=self._config.controller_address,
                timeout_ms=10_000,
                interceptors=interceptors,
                accept_compression=IRIS_RPC_COMPRESSIONS,
                send_compression=None,
            )
            # Register stats namespaces eagerly. Schema bugs surface here at
            # startup rather than silently producing empty namespaces.
            assert self._log_client is not None
            self._worker_stats_table = self._log_client.get_table(WORKER_STATS_NAMESPACE, IrisWorkerStat)
            self._task_stats_table = self._log_client.get_table(TASK_STATS_NAMESPACE, IrisTaskStat)
            self._profile_table = self._log_client.get_table(PROFILE_NAMESPACE, IrisProfile)

        # Try to adopt running containers from a previous worker process.
        # If adoption succeeds, skip the destructive cleanup that would kill them.
        adopted = self.adopt_running_containers()
        if adopted == 0:
            self._cleanup_all_iris_containers()

        # Bring the HTTP server up last so the worker is ready to serve the
        # controller's Reconcile RPC the moment registration completes.
        # timeout_keep_alive=120: default 5s races with the controller's reconcile
        # cadence, causing TCP resets on idle connections.
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
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server.started,
            timeout=Duration.from_seconds(5.0),
        )

        # Start lifecycle thread: register + serve + reset loop
        if self._config.controller_address:
            self._threads.spawn(target=self._run_lifecycle, name="worker-lifecycle")
            self._threads.spawn(target=self._run_profile_loop, name="profile-loop")

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

            with self._lock:
                self._tasks.append(attempt)

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
        """Remove and close the current RemoteLogHandler and LogClient if any."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None
        # Stats Tables belong to the LogClient; drop our cached references so
        # post-shutdown writes are no-ops.
        self._worker_stats_table = None
        self._task_stats_table = None
        self._profile_table = None
        if self._log_client is not None:
            self._log_client.close()
            self._log_client = None

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
        controller Reconcile RPC. When the timeout expires it returns,
        triggering a reset and re-registration.
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

    def _prefer_live(self, attempts: list[TaskAttempt]) -> TaskAttempt | None:
        """Pick the highest-attempt_id member of ``attempts``, preferring non-terminal.

        Two attempts can share a ``(task_id, attempt_id)`` when a fresh-UID
        resubmit runs alongside a retained terminal attempt. Callers always
        want the live one, so terminal attempts are considered only when no
        live attempt matches.
        """
        if not attempts:
            return None
        live = [t for t in attempts if t.status not in self._TERMINAL_STATES]
        return max(live or attempts, key=lambda t: t.attempt_id)

    def current_attempt(self, task_id: str) -> TaskAttempt | None:
        """Most recent attempt for a task, preferring a live attempt over a terminal twin."""
        return self._prefer_live([t for t in self._tasks if t.task_id.to_wire() == task_id])

    def task_by_uid(self, uid: AttemptUid) -> TaskAttempt | None:
        """Resolve an attempt by its controller-minted UID.

        Returns None for an empty UID — an empty UID never identifies an attempt.
        """
        if not uid:
            return None
        return next((t for t in self._tasks if t.attempt_uid == uid), None)

    def task_by_attempt(self, task_id: str, attempt_id: int) -> TaskAttempt | None:
        """Resolve an attempt by its ``(task_id, attempt_id)`` composite key.

        Prefers a live attempt over a retained terminal twin sharing the key.
        Used by ``get_task`` and ``capture_and_log_profile`` to address an
        attempt by its task-relative identifiers (e.g. for profiling).
        """
        return self._prefer_live(
            [t for t in self._tasks if t.task_id.to_wire() == task_id and t.attempt_id == attempt_id]
        )

    def submit_task(self, request: job_pb2.RunTaskRequest) -> str:
        """Submit a new task for execution.

        Identity is the controller-minted ``attempt_uid``: a request whose UID
        already names a known attempt is rejected as a duplicate. A re-submitted
        ``(task_id, attempt_id)`` with a *fresh* UID is a distinct attempt and
        runs even though a terminal attempt with that composite is retained.

        A higher-attempt-id submission supersedes a non-terminal current
        attempt for the same task — the old attempt is killed.

        Raises:
            ValueError: If ``request.attempt_uid`` is empty.
            RuntimeError: If a non-terminal attempt already exists (sanity check)
        """
        if rule := chaos("worker.submit_task"):
            time.sleep(rule.delay_seconds)
            raise RuntimeError("chaos: worker rejecting task")
        task_id_wire = request.task_id
        task_id = JobName.from_wire(task_id_wire)
        attempt_id = request.attempt_id
        attempt_uid = AttemptUid(request.attempt_uid)
        if not attempt_uid:
            raise ValueError("attempt_uid is required")

        should_kill_existing = False
        with self._lock:
            # Identity is the UID; a re-submitted composite carrying a fresh
            # UID is a new attempt.
            existing = self.task_by_uid(attempt_uid)
            if existing is not None:
                logger.info(
                    "Rejecting duplicate task %s attempt %d (status=%s)",
                    task_id,
                    attempt_id,
                    existing.status,
                )
                return task_id_wire

            # Sanity check: find any non-terminal attempt for this task
            current = self.current_attempt(task_id_wire)
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
            assert current is not None  # set only on the supersede branch above
            current.kill()

        task_id.require_task()

        # Create a minimal TaskAttemptConfig. Expensive setup (port allocation,
        # workdir creation, log sink init) is deferred to TaskAttempt.run() so
        # the heartbeat RPC returns quickly.
        config = TaskAttemptConfig(
            task_attempt=TaskAttemptId(task_id=task_id, attempt_id=attempt_id),
            num_tasks=request.num_tasks,
            request=request,
            cache_dir=self._cache_dir,
            attempt_uid=attempt_uid,
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

        with self._lock:
            self._tasks.append(attempt)

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
            return self.task_by_attempt(task_id, attempt_id)
        return self.current_attempt(task_id)

    def list_tasks(self) -> list[TaskInfo]:
        """List all task attempts.

        Returns TaskInfo views (implemented by TaskAttempt) to decouple callers
        from execution internals. Returns all attempts, not just current ones.
        """
        return list(self._tasks)

    def _collect_resource_metrics(self) -> job_pb2.WorkerResourceSnapshot:
        """Collect host metrics with running-task and process aggregates filled in."""
        snapshot = self._host_metrics.collect()
        running_count = 0
        total_processes = 0
        with self._lock:
            for task in self._tasks:
                if task.status == job_pb2.TASK_STATE_RUNNING:
                    running_count += 1
                    total_processes += task.process_count
        snapshot.running_task_count = running_count
        snapshot.total_process_count = total_processes
        return snapshot

    def _emit_worker_stat(self, snapshot: job_pb2.WorkerResourceSnapshot) -> None:
        """Append one heartbeat row to the ``iris.worker`` stats namespace.

        Non-blocking: ``Table.write`` queues for the bg flush thread, so the
        reconcile path never waits on the stats service. Schema-validation
        ``TypeError`` bugs from the row encoder deliberately propagate.
        """
        table = self._worker_stats_table
        if table is None or self._worker_id is None:
            return
        status = WorkerStatus.RUNNING if self._tasks else WorkerStatus.IDLE
        stat = build_worker_stat(
            worker_id=self._worker_id,
            status=status,
            address=self._resolve_address(),
            snapshot=snapshot,
            metadata=self._worker_metadata,
        )
        table.write([stat])

    def handle_reconcile(self, request: worker_pb2.Worker.ReconcileRequest) -> worker_pb2.Worker.ReconcileResponse:
        """Process desired state from the controller and return observed state.

        Reconcile is the sole controller→worker channel, so it is also the
        worker's keep-alive: every reconcile resets the heartbeat deadline (the
        worker self-resets only after ``heartbeat_timeout`` with no contact) and
        emits a host-metrics stat row. The response carries the worker's
        self-health bit so the controller can reap a responsive-but-unhealthy
        worker (e.g. failed disk).

        Routing is by ``attempt_uid`` only: every DesiredAttempt and
        AttemptObservation is keyed by UID.
        """
        # Identity guard: only act on reconciles addressed to *this* worker.
        # After a worker's VM is deleted GCP can recycle its internal IP onto a
        # new VM; the controller may still hold the dead worker's stale address
        # and reconcile us under the dead worker's id. Because routing is by
        # attempt_uid, processing it would run the dead worker's tasks and
        # zombie-kill our own attempts (absent from the misrouted plan). Refuse,
        # but report our real id so the controller detects the recycled address
        # and reaps the stale worker (a wrong-worker reply is not a heartbeat, so
        # we also skip the deadline reset and the host-metrics stat below).
        if request.worker_id and self._worker_id and request.worker_id != self._worker_id:
            logger.warning(
                "Reconcile addressed to %s but this worker is %s; refusing (recycled address?)",
                request.worker_id,
                self._worker_id,
            )
            return worker_pb2.Worker.ReconcileResponse(
                worker_id=self._worker_id,
                observed=[],
                health=worker_pb2.Worker.WorkerHealth(healthy=True),
            )

        self._heartbeat_deadline = Deadline.from_seconds(self._config.heartbeat_timeout.to_seconds())

        for desired in request.desired:
            attempt_uid = AttemptUid(desired.attempt_uid)
            if desired.HasField("run"):
                self._process_run_intent(attempt_uid, desired.run)
            else:
                self._process_stop_intent(attempt_uid)

        # An attempt is desired if a DesiredAttempt's UID resolves to it.
        with self._lock:
            desired_attempts: set[int] = set()
            for desired in request.desired:
                match = self.task_by_uid(AttemptUid(desired.attempt_uid))
                if match is not None:
                    desired_attempts.add(id(match))
            snapshot = list(self._tasks)

        zombie_attempts: set[int] = set()
        for task in snapshot:
            if id(task) in desired_attempts or task.status in self._TERMINAL_STATES:
                continue
            logger.info(
                "Reconcile: killing zombie attempt %s (uid=%s, not in desired set)",
                task.task_id,
                task.attempt_uid,
            )
            zombie_attempts.add(id(task))
            self._kill_async(task)

        # Observations are bounded by what the controller asked about. Emitting
        # terminal local history the controller did not request is wasted wire
        # bandwidth and a wasted DB write on the apply side. We report:
        #   - attempts that resolve to a DesiredAttempt, and
        #   - zombies we are killing this tick, so the controller can confirm
        #     the kill it implicitly requested by omitting the attempt.
        observations: list[worker_pb2.Worker.AttemptObservation] = []
        with self._lock:
            snapshot = list(self._tasks)
            for task in snapshot:
                if id(task) not in desired_attempts and id(task) not in zombie_attempts:
                    continue
                observations.append(self._build_observation(task))

            # Synthesize MISSING for any desired attempt that resolved to no
            # local attempt — both 'run' and 'stop' intents. A 'run' miss means
            # the worker never received the spec; a 'stop' miss means the worker
            # has forgotten a (e.g. controller-terminated) attempt it was asked
            # to kill. In both cases the attempt is gone, so reporting MISSING
            # lets the controller finalize it (stamp finished_at_ms) instead of
            # re-polling forever.
            for desired in request.desired:
                if self.task_by_uid(AttemptUid(desired.attempt_uid)) is None:
                    observations.append(
                        worker_pb2.Worker.AttemptObservation(
                            attempt_uid=desired.attempt_uid,
                            state=job_pb2.TASK_STATE_MISSING,
                        )
                    )

        resource_snapshot = self._collect_resource_metrics()
        self._emit_worker_stat(resource_snapshot)
        health = check_worker_health(disk_path=str(self._cache_dir))
        if not health.healthy:
            logger.warning("Reconcile: worker health check failed: %s", health.error)

        worker_health = worker_pb2.Worker.WorkerHealth(
            healthy=health.healthy,
            health_error=health.error,
            resources=resource_snapshot,
        )

        return worker_pb2.Worker.ReconcileResponse(
            worker_id=self._worker_id or "",
            observed=observations,
            health=worker_health,
        )

    def _process_run_intent(
        self,
        attempt_uid: AttemptUid,
        attempt_spec: worker_pb2.Worker.AttemptSpec,
    ) -> None:
        """Handle a single DesiredAttempt with intent=run.

        Resolves the target attempt by ``attempt_uid``. If the worker already
        holds it, this is a no-op. Otherwise enqueue when an inline spec is
        provided; without a spec, leave the attempt absent so the observation
        loop reports MISSING.
        """
        with self._lock:
            task = self.task_by_uid(attempt_uid)

        if task is not None:
            return

        if attempt_spec.HasField("request"):
            request = attempt_spec.request
            logger.info(
                "Reconcile: enqueuing attempt uid=%s task=%s attempt=%d (spec inline)",
                attempt_uid,
                request.task_id,
                request.attempt_id,
            )
            self.submit_task(request)
        else:
            logger.info("Reconcile: attempt uid=%s unknown and no spec; will report MISSING", attempt_uid)

    def _process_stop_intent(self, attempt_uid: AttemptUid) -> None:
        """Handle a single DesiredAttempt with intent=stop.

        Resolves the target attempt by ``attempt_uid``. Idempotent: silently
        does nothing if the attempt is already terminal or not present locally.
        """
        with self._lock:
            task = self.task_by_uid(attempt_uid)
            if task is None:
                return

        logger.info("Reconcile: stopping attempt uid=%s (stop intent)", attempt_uid)
        self._kill_async(task)

    def _build_observation(
        self,
        task: TaskAttempt,
    ) -> worker_pb2.Worker.AttemptObservation:
        """Build an AttemptObservation from a local TaskAttempt.

        The observation is keyed by ``attempt_uid``.
        """
        state = task.status
        # Workers never report PENDING to the controller; map it to BUILDING.
        if state == job_pb2.TASK_STATE_PENDING:
            state = job_pb2.TASK_STATE_BUILDING

        obs = worker_pb2.Worker.AttemptObservation(
            attempt_uid=task.attempt_uid,
            state=state,
            exit_code=task.exit_code or 0,
            error=task.error or "",
            container_id=task.platform_container_id or "",
        )
        if task.status in self._TERMINAL_STATES and task.finished_at is not None:
            obs.finished_at.CopyFrom(timestamp_to_proto(task.finished_at))
        return obs

    def _kill_async(self, attempt: TaskAttempt, term_timeout_ms: int = 5000) -> None:
        """Kill ``attempt`` in a daemon thread so an RPC handler does not block on it.

        ``TaskAttempt.kill`` runs a stop/wait/force-kill sequence that can take
        up to ``term_timeout_ms``; heartbeat and reconcile paths offload it so
        the RPC response returns promptly.
        """
        threading.Thread(
            target=attempt.kill,
            args=(term_timeout_ms,),
            name=f"kill-{attempt.task_id.to_wire()}-{attempt.attempt_id}",
            daemon=True,
        ).start()

    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool:
        """Kill the current (most recent) attempt of a task."""
        current = self.current_attempt(task_id)
        if not current:
            return False
        return current.kill(term_timeout_ms)

    def _run_profile_loop(self, stop_event: threading.Event) -> None:
        """Tick at ``profile_interval`` and capture a CPU profile per running attempt.

        Captures run sequentially within a worker; across workers they run in
        parallel automatically. Per-attempt exceptions are logged and dropped.
        """
        limiter = RateLimiter(interval_seconds=self._profile_interval.to_seconds())
        cpu_request = job_pb2.ProfileTaskRequest(
            duration_seconds=self._profile_duration_seconds,
            profile_type=job_pb2.ProfileType(cpu=job_pb2.CpuProfile(format=job_pb2.CpuProfile.SPEEDSCOPE)),
        )
        while not stop_event.is_set():
            remaining = limiter.time_until_next()
            if remaining > 0:
                stop_event.wait(timeout=remaining)
            if stop_event.is_set():
                break
            limiter.mark_run()
            with self._lock:
                running = [a for a in self._tasks if a.status == job_pb2.TASK_STATE_RUNNING]
            for attempt in running:
                if stop_event.is_set():
                    break
                target = TaskAttemptId(task_id=attempt.task_id, attempt_id=attempt.attempt_id).to_wire()
                try:
                    self.capture_and_log_profile(
                        target=target,
                        request=cpu_request,
                        trigger=ProfileTrigger.PERIODIC,
                    )
                except Exception:
                    logger.exception(
                        "profile capture failed for %s attempt=%s",
                        attempt.task_id,
                        attempt.attempt_id,
                    )

    def capture_and_log_profile(
        self,
        *,
        target: str,
        request: job_pb2.ProfileTaskRequest,
        trigger: ProfileTrigger,
    ) -> bytes:
        """Profile ``target`` and write one ``IrisProfile`` row; returns the captured bytes.

        ``target`` is one of:
          - ``"/system/process"``: this worker process. The row's ``source`` is
            rewritten to ``"/system/worker/<id>"``.
          - ``"/job/.../task/N"``: bare task wire. Falls back to the most recent
            attempt for that task.
          - ``"/job/.../task/N:<attempt_id>"``: a specific attempt.

        For task targets the resolved attempt must be ``RUNNING``.
        """
        duration = request.duration_seconds or self._profile_duration_seconds
        assert self._worker_id, "worker_id required before capturing profiles"

        if target == "/system/process":
            data = profile_local_process(duration, request.profile_type)
            row_source = f"/system/worker/{self._worker_id}"
            row_attempt_id: int | None = None
        else:
            parsed = TaskAttemptId.from_wire(target)
            task_id_wire = parsed.task_id.to_wire()
            resolved_attempt_id = parsed.attempt_id
            if resolved_attempt_id is None:
                current = self.current_attempt(task_id_wire)
                if current is None:
                    raise RuntimeError(f"no attempts for task {task_id_wire}")
                resolved_attempt_id = current.attempt_id
            attempt = self.task_by_attempt(task_id_wire, resolved_attempt_id)
            if attempt is None or attempt.status != job_pb2.TASK_STATE_RUNNING:
                raise RuntimeError("attempt no longer running")
            data = attempt.profile(duration, request.profile_type)
            row_source = task_id_wire
            row_attempt_id = resolved_attempt_id

        if not data:
            logger.debug("Empty profile for %s (trigger=%s); skipping iris.profile write", row_source, trigger.value)
            return data

        assert self._profile_table is not None, "profile_table must be initialized before capture"
        self._profile_table.write(
            [
                build_profile_row(
                    source=row_source,
                    attempt_id=row_attempt_id,
                    vm_id=self._worker_id,
                    duration_seconds=duration,
                    profile_type=request.profile_type,
                    profile_data=data,
                    trigger=trigger,
                )
            ]
        )
        return data

    def exec_in_container(
        self, task_id: str, command: list[str], timeout_seconds: int = 60
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        """Execute a command in a running task's container.

        Delegates to the container handle's underlying runtime (docker exec, subprocess, kubectl exec).
        """
        attempt = self.current_attempt(task_id)
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
