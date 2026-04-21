# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task execution attempt handling.

This module encapsulates the full lifecycle of a single task execution attempt:
bundle download -> image build -> container run -> monitor -> cleanup.
"""

import logging
import shutil
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from iris.chaos import chaos, chaos_raise
from iris.cluster.runtime.types import (
    ContainerConfig,
    ContainerErrorKind,
    ContainerHandle,
    ContainerInfraError,
    ContainerPhase,
    ContainerRuntime,
    DiscoveredContainer,
    RuntimeLogReader,
    MountKind,
    MountSpec,
)
from iris.cluster.types import (
    JobName,
    TaskAttempt as TaskAttemptIdentity,
    is_task_finished,
)
from iris.cluster.bundle import BundleStore
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.tpu_health import detect_tpu_init_failure
from iris.cluster.worker.worker_types import LogLine
from iris.cluster.log_store._types import task_log_key
from iris.log_server.client import LogPusher
from iris.logging import str_to_log_level
from rigging.log_setup import parse_log_level
from iris.rpc import logging_pb2
from iris.rpc import job_pb2
from iris.rpc import worker_pb2
from iris.rpc.job_pb2 import TaskState, WorkerMetadata
from iris.rpc.errors import format_exception_with_traceback
from iris.time_proto import timestamp_to_proto
from rigging.timing import Duration, Timestamp

logger = logging.getLogger(__name__)

# Trailing stderr lines scanned for TPU bad-node signatures on non-zero exit.
_TPU_STDERR_TAIL_LINES = 200

# Signal numbers for interpreting exit codes > 128
_SIGNAL_NAMES = {
    6: "SIGABRT",
    9: "SIGKILL",
    11: "SIGSEGV",
    15: "SIGTERM",
}


def _format_exit_error(exit_code: int | None, oom_killed: bool = False) -> str:
    """Format an exit code into a human-readable error message.

    Exit codes > 128 typically indicate the process was killed by a signal,
    where signal_number = exit_code - 128.
    """
    if exit_code is None:
        return "Unknown exit code"

    # Check for OOM first (most specific)
    if oom_killed:
        return f"Exit code {exit_code}: OOM killed (container exceeded memory limit)"

    # Interpret signal-based exit codes
    if exit_code > 128:
        signal_num = exit_code - 128
        signal_name = _SIGNAL_NAMES.get(signal_num, f"signal {signal_num}")
        # Exit 137 (SIGKILL) without OOMKilled flag could still be resource-related
        if signal_num == 9:
            return f"Exit code {exit_code}: killed by {signal_name} (possibly OOM or resource limit)"
        return f"Exit code {exit_code}: killed by {signal_name}"

    return f"Exit code: {exit_code}"


_DISK_CHECK_INTERVAL_SECONDS = 60.0


class TaskCancelled(Exception):
    """Raised when a task is cancelled during execution."""

    pass


@dataclass
class TaskAttemptConfig:
    """Immutable configuration for a task attempt, derived from the RPC request."""

    task_attempt: TaskAttemptIdentity
    num_tasks: int
    request: job_pb2.RunTaskRequest
    cache_dir: Path

    @property
    def task_id(self) -> JobName:
        return self.task_attempt.task_id

    @property
    def attempt_id(self) -> int:
        return self.task_attempt.require_attempt()


def _get_host_ip() -> str:
    """Get the routable IP of this host via the default route.

    Opens a UDP socket to a public IP (no traffic sent) and reads back the
    local address the OS selected. With --network=host this returns the real
    machine IP visible to other machines in the same VPC.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def build_iris_env(
    task: "TaskAttempt",
    worker_id: str | None,
    controller_address: str | None,
) -> dict[str, str]:
    """Build Iris system environment variables for the task container.

    Thin wrapper around build_common_iris_env() that adds worker-specific
    variables (IRIS_WORKER_ID, IRIS_ADVERTISE_HOST) and overrides port values
    with real allocated ports.
    """
    from iris.cluster.runtime.env import build_common_iris_env

    req = task.request
    env = build_common_iris_env(
        task_id=req.task_id,
        attempt_id=task.attempt_id,
        num_tasks=task.num_tasks,
        bundle_id=req.bundle_id,
        controller_address=controller_address,
        environment=req.environment,
        constraints=req.constraints,
        ports=req.ports,
        resources=req.resources if req.HasField("resources") else None,
    )

    if worker_id:
        env["IRIS_WORKER_ID"] = worker_id

    # With --network=host, containers share the host's network stack.
    # Compute the host's routable IP so container code can read it via
    # get_job_info().advertise_host without needing its own socket tricks.
    env["IRIS_ADVERTISE_HOST"] = _get_host_ip()

    # Override port placeholders with real allocated values
    for name, port in task.ports.items():
        env[f"IRIS_PORT_{name.upper()}"] = str(port)

    return env


class TaskAttempt:
    """Manages the lifecycle of a single task execution attempt.

    Owns the full pipeline: bundle download -> image build -> container run -> monitor.
    Reports state changes back to the worker via a callback. Also serves as the
    single source of truth for all task state (status, logs, resource usage, etc.).

    This class is module-internal to iris.cluster.worker and has no external
    consumers. It encapsulates the complex task execution logic that was
    previously interleaved in the Worker class.

    Thread safety: This object is mutated by its execution thread (run()) and
    read concurrently by RPC handlers via the TaskInfo protocol. Python's GIL
    ensures atomic field assignments. State transitions are one-way (PENDING →
    BUILDING → RUNNING → terminal), preventing inconsistent states. External
    code should only read via TaskInfo protocol (status, to_proto()).
    """

    def __init__(
        self,
        config: TaskAttemptConfig,
        bundle_store: BundleStore | None,
        container_runtime: ContainerRuntime | None,
        worker_metadata: WorkerMetadata | None,
        worker_id: str | None,
        controller_address: str | None,
        task_env: dict[str, str] | None,
        default_task_image: str | None,
        resolve_image: Callable[[str], str] | None,
        port_allocator: PortAllocator,
        log_pusher: LogPusher | None,
        poll_interval_seconds: float = 5.0,
        *,
        container_handle: ContainerHandle | None = None,
        initial_status: TaskState | None = None,
    ):
        """Initialize a TaskAttempt.

        Construction is intentionally cheap (no I/O, no port allocation) so
        that submit_task() can return quickly on the heartbeat thread. Expensive
        setup (port allocation, working directory creation) is deferred to run().

        For adopted tasks (container already running from a previous worker),
        pass container_handle and initial_status=TASK_STATE_RUNNING. The
        bundle_store, container_runtime, worker_metadata, task_env, and
        resolve_image params can be None since the run pipeline is skipped.

        Args:
            config: Immutable configuration for this attempt
            bundle_store: Bundle store for resolving task bundles (None for adopted tasks)
            container_runtime: Runtime for creating containers (None for adopted tasks)
            worker_metadata: Worker's hardware/environment metadata (None for adopted tasks)
            worker_id: Worker identifier for env injection
            controller_address: Controller address for env injection
            task_env: Worker-level default env vars (None for adopted tasks)
            default_task_image: Fully-qualified task container image from cluster config
            resolve_image: Resolves image tags for the current platform (None for adopted tasks)
            port_allocator: Port allocator for releasing ports on cleanup
            log_pusher: Pushes log entries to the central LogService.
            poll_interval_seconds: How often to poll container status
            container_handle: Pre-existing container handle for adopted tasks
            initial_status: Starting status (default PENDING, use RUNNING for adopted tasks)
        """
        self._bundle_store = bundle_store
        self._runtime = container_runtime
        self._worker_metadata = worker_metadata or job_pb2.WorkerMetadata()
        self._worker_id = worker_id
        self._controller_address = controller_address
        self._task_env = task_env or {}
        self._default_task_image = default_task_image
        self._resolve_image_fn = resolve_image or (lambda x: x)
        self._port_allocator = port_allocator
        self._poll_interval_seconds = poll_interval_seconds
        self._log_pusher = log_pusher
        self._log_key = task_log_key(config.task_attempt)

        # Task identity (from config)
        self.task_attempt: TaskAttemptIdentity = config.task_attempt
        self.task_id: JobName = config.task_id
        self.num_tasks: int = config.num_tasks
        self.attempt_id: int = config.attempt_id
        self.request: job_pb2.RunTaskRequest = config.request
        self.ports: dict[str, int] = {}
        self.workdir: Path | None = None
        self._cache_dir: Path = config.cache_dir
        # Task state
        self.status: TaskState = initial_status or job_pb2.TASK_STATE_PENDING
        self.exit_code: int | None = None
        self.error: str | None = None
        self.started_at: Timestamp | None = None
        self.finished_at: Timestamp | None = None
        self.status_message: str = ""

        # Resource tracking
        self.current_memory_mb: int = 0
        self.peak_memory_mb: int = 0
        self.current_cpu_millicores: int = 0
        self.process_count: int = 0
        self.disk_mb: int = 0

        # Build tracking
        self.build_started: Timestamp | None = None
        self.build_finished: Timestamp | None = None
        self.build_from_cache: bool = False
        self.image_tag: str = ""
        self._build_phase_start: float = 0.0

        # Internals
        self._container_handle: ContainerHandle | None = container_handle
        self.thread: threading.Thread | None = None
        self.cleanup_done: bool = False
        self.should_stop: bool = False
        self.on_state_change: Callable[[TaskState], None] | None = None

    @classmethod
    def adopt(
        cls,
        discovered: DiscoveredContainer,
        container_handle: ContainerHandle,
        log_pusher: LogPusher | None,
        port_allocator: PortAllocator,
        poll_interval_seconds: float = 5.0,
    ) -> "TaskAttempt":
        """Create a TaskAttempt that adopts an already-running container.

        Used after worker restart to resume monitoring a container started by
        the previous worker process. Calls the normal __init__ with None for
        run-pipeline-only dependencies (bundle_store, runtime, etc.) and
        injects the existing container handle.
        """
        task_id = JobName.from_wire(discovered.task_id)
        attempt_id = discovered.attempt_id
        identity = TaskAttemptIdentity(task_id=task_id, attempt_id=attempt_id)

        request = job_pb2.RunTaskRequest(
            task_id=discovered.task_id,
            attempt_id=attempt_id,
        )
        config = TaskAttemptConfig(
            task_attempt=identity,
            num_tasks=1,
            request=request,
            cache_dir=Path(discovered.workdir_host_path).parent.parent if discovered.workdir_host_path else Path("/tmp"),
        )

        instance = cls(
            config=config,
            bundle_store=None,
            container_runtime=None,
            worker_metadata=None,
            worker_id=discovered.worker_id,
            controller_address=None,
            task_env=None,
            default_task_image=None,
            resolve_image=None,
            port_allocator=port_allocator,
            log_pusher=log_pusher,
            poll_interval_seconds=poll_interval_seconds,
            container_handle=container_handle,
            initial_status=job_pb2.TASK_STATE_RUNNING,
        )
        instance.started_at = Timestamp.now()
        instance.status_message = "adopted"
        instance.workdir = Path(discovered.workdir_host_path) if discovered.workdir_host_path else None
        return instance

    def resume_monitoring(self) -> None:
        """Monitor an adopted container until completion.

        Enters the monitor loop directly, skipping bundle download, image
        resolve, container create, and build phases. Used after adopt().
        """
        assert self._container_handle is not None
        handle = self._container_handle

        logger.info(
            "Resuming monitoring for adopted task %s attempt %d (container=%s)",
            self.task_id,
            self.attempt_id,
            self.container_id,
        )

        try:
            log_reader = handle.log_reader()
            self._monitor_loop(handle, log_reader)
        except Exception as e:
            error_msg = format_exception_with_traceback(e)
            self._append_log(source="error", data=f"Monitoring failed:\n{error_msg}")
            self.transition_to(job_pb2.TASK_STATE_FAILED, error=error_msg)
        finally:
            self._cleanup()
            logger.info(
                "Adopted task finished: task_id=%s attempt=%s state=%s exit_code=%s",
                self.task_id,
                self.attempt_id,
                self.status,
                self.exit_code,
            )

    @property
    def container_id(self) -> str | None:
        """Return the container ID from the handle, if available."""
        if self._container_handle:
            return self._container_handle.container_id
        return None

    @property
    def platform_container_id(self) -> str | None:
        """Return the platform container ID from the handle, if available.

        Docker: container hash. K8s: pod name. Process: local-<uuid>.
        """
        if self._container_handle:
            return self._container_handle.container_id
        return None

    def stop(self, force: bool = False) -> None:
        """Stop the container, if running."""
        self.should_stop = True
        if self._container_handle:
            self._container_handle.stop(force=force)

    @property
    def has_container(self) -> bool:
        """Whether this attempt has an active container handle."""
        return self._container_handle is not None

    def profile(self, duration_seconds: int, profile_type: job_pb2.ProfileType) -> bytes:
        """Profile the running container process.

        Args:
            duration_seconds: How long to sample
            profile_type: ProfileType message with oneof cpu/memory profiler config

        Returns:
            Raw profile output

        Raises:
            ValueError: If no container handle is available
        """
        if not self._container_handle:
            raise ValueError(f"Task {self.task_id} has no container handle")
        return self._container_handle.profile(duration_seconds, profile_type)

    def exec_in_container(
        self, command: list[str], timeout_seconds: int = 60
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        """Execute a command in this task's container.

        Uses docker exec for Docker containers, subprocess for process containers.
        A negative timeout_seconds means no timeout.
        """
        if not self._container_handle:
            return worker_pb2.Worker.ExecInContainerResponse(error=f"Task {self.task_id} has no container handle")

        import subprocess as _subprocess

        container_id = self._container_handle.container_id
        if not container_id:
            return worker_pb2.Worker.ExecInContainerResponse(error="No container ID available")

        effective_timeout: float | None = timeout_seconds if timeout_seconds >= 0 else None

        # Use docker exec for Docker containers, direct exec for process containers
        from iris.cluster.runtime.docker import DockerContainerHandle

        if isinstance(self._container_handle, DockerContainerHandle):
            result = _subprocess.run(
                ["docker", "exec", container_id, *command],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return worker_pb2.Worker.ExecInContainerResponse(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        # Process runtime: run command directly
        result = _subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
        return worker_pb2.Worker.ExecInContainerResponse(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def transition_to(
        self,
        state: TaskState,
        *,
        message: str = "",
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        self.status = state
        self.status_message = message
        if is_task_finished(state):
            self.finished_at = Timestamp.now()
            if error:
                self.error = error
            if exit_code is not None:
                self.exit_code = exit_code
        if self.on_state_change is not None:
            try:
                self.on_state_change(state)
            except Exception:
                logger.debug("on_state_change callback failed", exc_info=True)

    def duration(self) -> Duration | None:
        """Calculate how long the attempt ran.

        Returns:
            Duration from started_at to finished_at, or None if not finished
        """
        if self.finished_at is None:
            return None
        elapsed_ms = self.finished_at.epoch_ms() - self.started_at.epoch_ms()
        return Duration.from_ms(elapsed_ms)

    def to_proto(self) -> job_pb2.TaskStatus:
        proto = job_pb2.TaskStatus(
            task_id=self.task_id.to_wire(),
            state=self.status,
            exit_code=self.exit_code or 0,
            error=self.error or "",
            ports=self.ports,
            current_attempt_id=self.attempt_id,
            container_id=self.platform_container_id or "",
            resource_usage=job_pb2.ResourceUsage(
                memory_mb=self.current_memory_mb,
                memory_peak_mb=self.peak_memory_mb,
                disk_mb=self.disk_mb,
                cpu_millicores=self.current_cpu_millicores,
                process_count=self.process_count,
            ),
            build_metrics=job_pb2.BuildMetrics(
                from_cache=self.build_from_cache,
                image_tag=self.image_tag,
            ),
        )

        # Set timestamp fields using proto Timestamp messages
        if self.started_at is not None:
            proto.started_at.CopyFrom(timestamp_to_proto(self.started_at))
        if self.finished_at is not None:
            proto.finished_at.CopyFrom(timestamp_to_proto(self.finished_at))
        if self.build_started is not None:
            proto.build_metrics.build_started.CopyFrom(timestamp_to_proto(self.build_started))
        if self.build_finished is not None:
            proto.build_metrics.build_finished.CopyFrom(timestamp_to_proto(self.build_finished))
        return proto

    def _check_cancelled(self) -> None:
        """Check if task has been cancelled and raise if so."""
        if self.should_stop:
            raise TaskCancelled("Task was cancelled")

    def _setup(self) -> None:
        """Perform expensive setup work that was deferred from submit_task().

        Allocates ports and creates the working directory. Runs at the start of
        run() on the task thread so the heartbeat RPC returns immediately.
        """
        # Allocate requested ports
        port_names = list(self.request.ports)
        allocated_ports = self._port_allocator.allocate(len(port_names)) if port_names else []
        self.ports = dict(zip(port_names, allocated_ports, strict=True))

        # Create task working directory with attempt isolation
        safe_task_id = self.task_id.to_safe_token()
        self.workdir = self._cache_dir / "workdirs" / f"{safe_task_id}_attempt_{self.attempt_id}"
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Mount tmpfs on workdir for quota enforcement (Docker only; no-op for process/k8s).
        # Must happen before _download_bundle() so staged files land on the tmpfs.
        disk_bytes = self.request.resources.disk_bytes if self.request.HasField("resources") else 0
        self._runtime.prepare_workdir(self.workdir, disk_bytes)

    def run(self) -> None:
        """Execute the full task lifecycle. Intended to run in a background thread.

        The lifecycle is:
        0. Setup: allocate ports, create workdir
        1. Download bundle by bundle ID
        2. Resolve base image
        3. Create container handle
        4. Build phase: run setup_commands (uv sync) - BUILDING state
        5. Run phase: start main command - RUNNING state
        6. Monitor until completion

        Not valid for adopted tasks — use resume_monitoring() instead.
        """
        if self._bundle_store is None or self._runtime is None:
            raise RuntimeError("Cannot run() an adopted TaskAttempt — use resume_monitoring()")
        logger.info(
            "TaskAttempt starting: task_id=%s attempt=%s num_tasks=%s",
            self.task_id,
            self.attempt_id,
            self.num_tasks,
        )
        try:
            self._check_cancelled()
            self._setup()
            self._check_cancelled()
            self._download_bundle()
            self._check_cancelled()
            self._resolve_image()
            self._check_cancelled()
            self._create_container()
            self._check_cancelled()
            self._build_container()
            self._check_cancelled()
            self._run_container()
            self._monitor()
        except TaskCancelled:
            self.transition_to(job_pb2.TASK_STATE_KILLED)
        except ContainerInfraError as e:
            error_msg = format_exception_with_traceback(e)
            self._append_log(source="error", data=f"Infrastructure error:\n{error_msg}")
            self.transition_to(job_pb2.TASK_STATE_WORKER_FAILED, error=error_msg)
        except Exception as e:
            error_msg = format_exception_with_traceback(e)
            self._append_log(source="error", data=f"Task failed:\n{error_msg}")
            self.transition_to(job_pb2.TASK_STATE_FAILED, error=error_msg)
        finally:
            self._cleanup()
            logger.info(
                "TaskAttempt finished: task_id=%s attempt=%s state=%s exit_code=%s",
                self.task_id,
                self.attempt_id,
                self.status,
                self.exit_code,
            )

    def _download_bundle(self) -> None:
        """Stage the code bundle from the configured bundle ID.

        Transitions task to BUILDING state and performs chaos injection checks
        for testing delayed builds.
        """
        self.transition_to(job_pb2.TASK_STATE_BUILDING, message="downloading bundle")
        self.started_at = Timestamp.now()
        self._build_phase_start = time.monotonic()

        download_start = time.monotonic()

        # Chaos injection for testing failures during download
        chaos_raise("worker.bundle_download")

        # Chaos injection for testing delayed builds (for screenshot tests)
        if rule := chaos("worker.building_delay"):
            time.sleep(rule.delay_seconds)

        # Periodically check should_stop during download to support kill during BUILDING
        # (RF-3: For now, we defer kill handling until container starts, as bundle
        # downloads are typically fast. Future work could add cancellation support
        # to BundleStore.extract_bundle_to if long downloads become a problem.)

        assert self.workdir is not None
        workdir_files = dict(self.request.entrypoint.workdir_files)
        for name, blob_id in self.request.entrypoint.workdir_file_refs.items():
            workdir_files[name] = self._bundle_store.get_or_fetch(blob_id, f"blobs/{blob_id}")
        self._runtime.stage_bundle(
            bundle_id=self.request.bundle_id,
            workdir=self.workdir,
            workdir_files=workdir_files,
            bundle_store=self._bundle_store,
        )

        logger.info(
            "Bundle staged for task %s in %.2fs",
            self.task_id,
            time.monotonic() - download_start,
        )

    def _resolve_image(self) -> None:
        """Resolve the task image from the request override or cluster config.

        Per-task ``task_image`` on the RunTaskRequest takes precedence over the
        worker's cluster-configured ``default_task_image``. This lets jobs that
        need a custom runtime (e.g. runsc/skopeo for sandboxing untrusted child
        workloads) supply their own image without reconfiguring the cluster.

        No per-job Docker build — the chosen image must already exist in the
        registry. The remote client wraps the entrypoint with uv sync.
        """
        requested = self.request.task_image or self._default_task_image
        if not requested:
            raise ValueError(
                "No task image configured. Pass task_image to submit() or set "
                "defaults.default_task_image in cluster config."
            )
        self.image_tag = self._resolve_image_fn(requested)

        logger.info("Using task image %s for task %s", self.image_tag, self.task_id)

    def _create_container(self) -> None:
        """Create container handle from config.

        Prepares the container configuration including environment variables,
        mounts, and workdir setup. The actual container is not started yet.
        """
        iris_env = build_iris_env(
            self,
            self._worker_id,
            self._controller_address,
        )
        env = dict(iris_env)

        # Expose the worker's region so child jobs can inherit a region
        # constraint (e.g. when the parent holds a reservation).
        from iris.cluster.constraints import WellKnownAttribute

        region_attr = self._worker_metadata.attributes.get(WellKnownAttribute.REGION)
        if region_attr and region_attr.string_value:
            env["IRIS_WORKER_REGION"] = region_attr.string_value

        env.update(self._task_env)
        env.update(dict(self.request.environment.env_vars))

        # Get RuntimeEntrypoint proto directly
        rt_ep = self.request.entrypoint

        # Extract timeout from proto (0 or unset means no timeout)
        timeout_seconds = None
        if self.request.HasField("timeout") and self.request.timeout.milliseconds > 0:
            timeout_seconds = self.request.timeout.milliseconds / 1000

        assert self.workdir is not None
        job_id, _ = self.task_id.require_task()

        mounts = [
            MountSpec("/app", kind=MountKind.WORKDIR),
            MountSpec("/tmp", kind=MountKind.TMPFS),
            MountSpec("/uv/cache", kind=MountKind.CACHE),
            MountSpec("/root/.cargo/registry", kind=MountKind.CACHE),
            MountSpec("/root/.cargo/target", kind=MountKind.CACHE),
        ]

        config = ContainerConfig(
            image=self.image_tag,
            entrypoint=rt_ep,
            env=env,
            resources=self.request.resources if self.request.HasField("resources") else None,
            timeout_seconds=timeout_seconds,
            mounts=mounts,
            workdir_host_path=self.workdir,
            task_id=self.task_id.to_wire(),
            attempt_id=self.attempt_id,
            job_id=job_id.to_wire(),
            worker_id=self._worker_id,
            worker_metadata=self._worker_metadata,
        )

        chaos_raise("worker.create_container")
        self._container_handle = self._runtime.create_container(config)
        logger.info("Container handle created for task %s", self.task_id)

    def _build_container(self) -> None:
        """Run setup commands (uv sync, pip install, etc) during BUILDING state.

        This is the build phase where dependencies are synced. The container
        handle runs setup_commands in a blocking fashion. If there are no
        setup_commands, this is a no-op.
        """
        assert self._container_handle is not None

        if self.request.entrypoint.setup_commands:
            self.transition_to(job_pb2.TASK_STATE_BUILDING, message="syncing dependencies")
            self.build_started = Timestamp.now()

        def on_build_logs(lines: list[LogLine]) -> None:
            entries = [self._make_log_entry(source=line.source, data=line.data) for line in lines]
            self._push_logs(entries)

        self._container_handle.build(on_logs=on_build_logs)

        self.build_finished = Timestamp.now()
        if self.request.entrypoint.setup_commands:
            logger.info("Build phase completed for task %s", self.task_id)

    def _run_container(self) -> None:
        """Start the container. Task stays in BUILDING until _monitor() confirms readiness."""
        assert self._container_handle is not None

        self._container_handle.run()
        logger.info(
            "Container started for task %s (container_id=%s, ports=%s)",
            self.task_id,
            self.container_id,
            self.ports,
        )

    def _monitor(self) -> None:
        """Monitor task execution: check status, collect stats, stream logs.

        Polls container status at regular intervals until the container stops.
        Streams logs incrementally into task.logs (single source of truth).
        Collects runtime statistics (CPU, memory, disk).
        Updates task state to terminal status (SUCCEEDED/FAILED/KILLED) when container stops.

        Execution timeouts are enforced by the controller, not the worker.
        Profiling is handled centrally by the controller's profile loop thread.
        """
        assert self._container_handle is not None
        assert self.workdir is not None
        handle = self._container_handle

        log_reader = handle.log_reader()
        self._monitor_loop(handle, log_reader)

    def _monitor_loop(
        self,
        handle: ContainerHandle,
        log_reader: RuntimeLogReader,
    ) -> None:
        last_disk_check = 0.0
        while True:
            if rule := chaos("worker.task_monitor"):
                time.sleep(rule.delay_seconds)
                self.transition_to(job_pb2.TASK_STATE_FAILED, error="chaos: monitor crashed")
                break

            # Check if we should stop
            if self.should_stop:
                handle.stop(force=True)
                logger.info("Task %s requested stop; killing container %s", self.task_id, self.container_id)
                self._stream_logs(log_reader)  # Capture final logs
                self.transition_to(job_pb2.TASK_STATE_KILLED)
                break

            # Check container status
            status = handle.status()

            if self.status == job_pb2.TASK_STATE_BUILDING and status.phase == ContainerPhase.RUNNING:
                building_duration = time.monotonic() - self._build_phase_start
                logger.info("Task %s BUILDING→RUNNING after %.1fs", self.task_id, building_duration)
                self.transition_to(job_pb2.TASK_STATE_RUNNING)

            if status.phase == ContainerPhase.STOPPED:
                logger.info(
                    "Container exited for task %s (container_id=%s, exit_code=%s, error=%s)",
                    self.task_id,
                    self.container_id,
                    status.exit_code,
                    status.error,
                )
                # Final log fetch before container stops
                self._stream_logs(log_reader)

                # Container has stopped
                if status.error:
                    failure_state = job_pb2.TASK_STATE_FAILED
                    if status.error_kind == ContainerErrorKind.INFRA_NOT_FOUND:
                        failure_state = job_pb2.TASK_STATE_WORKER_FAILED
                    self.transition_to(failure_state, error=status.error, exit_code=status.exit_code or -1)
                elif status.exit_code == 0:
                    self.transition_to(job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
                else:
                    stderr_tail: list[str] = [
                        entry.data for entry in log_reader.read_all() if entry.source == "stderr" and entry.data
                    ]
                    stderr_line = stderr_tail[-1] if stderr_tail else None
                    error = _format_exit_error(status.exit_code, status.oom_killed)
                    if stderr_line:
                        error = f"{error}. stderr: {stderr_line}"
                    if status.oom_killed:
                        self._append_log(source="error", data="Container was OOM killed by the kernel")
                    # Promote known TPU bad-node signatures to WORKER_FAILED.
                    tpu_pattern = detect_tpu_init_failure(stderr_tail[-_TPU_STDERR_TAIL_LINES:])
                    if tpu_pattern is not None:
                        logger.warning(
                            "Task %s: TPU bad-node signature %r; promoting FAILED -> WORKER_FAILED",
                            self.task_id,
                            tpu_pattern,
                        )
                        self._append_log(
                            source="error",
                            data=f"iris: TPU bad-node signature detected ({tpu_pattern!r}); "
                            "reporting as worker failure",
                        )
                        self.transition_to(
                            job_pb2.TASK_STATE_WORKER_FAILED,
                            error=f"TPU init failure ({tpu_pattern!r}): {error}",
                            exit_code=status.exit_code or -1,
                        )
                    else:
                        self.transition_to(
                            job_pb2.TASK_STATE_FAILED,
                            error=error,
                            exit_code=status.exit_code or -1,
                        )
                break

            # Stream logs incrementally
            self._stream_logs(log_reader)

            # Collect stats
            try:
                stats = handle.stats()
                if stats.available:
                    self.current_memory_mb = stats.memory_mb
                    self.current_cpu_millicores = stats.cpu_millicores
                    self.process_count = stats.process_count
                    if stats.memory_mb > self.peak_memory_mb:
                        self.peak_memory_mb = stats.memory_mb

                now = time.monotonic()
                if now - last_disk_check >= _DISK_CHECK_INTERVAL_SECONDS:
                    self.disk_mb = handle.disk_usage_mb()
                    last_disk_check = now
            except Exception:
                logger.debug("Stats collection failed for task %s", self.task_id, exc_info=True)

            # Sleep before next poll
            time.sleep(self._poll_interval_seconds)

    def _make_log_entry(self, *, source: str, data: str) -> logging_pb2.LogEntry:
        """Build a LogEntry proto from a source/data pair, parsing the level prefix."""
        level_name = parse_log_level(data)
        level = str_to_log_level(level_name) if level_name else 0
        entry = logging_pb2.LogEntry(source=source, data=data, level=level)
        entry.timestamp.epoch_ms = Timestamp.now().epoch_ms()
        return entry

    def _push_logs(self, entries: list[logging_pb2.LogEntry]) -> None:
        """Push a batch of log entries to the central LogService."""
        if not self._log_pusher or not entries:
            return
        try:
            self._log_pusher.push(self._log_key, entries)
        except Exception:
            logger.debug("Failed to push %d logs for task %s", len(entries), self.task_id, exc_info=True)

    def _append_log(self, *, source: str, data: str) -> None:
        """Push a single log entry (for rare events like errors)."""
        self._push_logs([self._make_log_entry(source=source, data=data)])

    def _stream_logs(self, reader: RuntimeLogReader) -> None:
        """Fetch new logs from container and push as a batch."""
        try:
            entries = [self._make_log_entry(source=line.source, data=line.data) for line in reader.read()]
            self._push_logs(entries)
        except Exception:
            logger.debug("Log streaming failed for task %s", self.task_id, exc_info=True)

    def _cleanup(self) -> None:
        """Clean up task resources: container, ports, image protection, workdir.

        Idempotent - safe to call multiple times. Logs errors instead of
        silently swallowing them (RF-5 fix).

        Container is removed here because logs are already streamed into task.logs
        during monitoring. This releases TPU devices that would otherwise remain
        busy until the container is removed.
        """
        if self.cleanup_done:
            return
        self.cleanup_done = True

        # Flush buffered log entries so they reach the server before the task
        # is reported as complete. The pusher is shared across tasks so we
        # flush rather than close.
        if self._log_pusher is not None:
            try:
                self._log_pusher.flush()
            except Exception as e:
                logger.debug("Failed to flush logs for task %s: %s", self.task_id, e)

        # Clean up container handle (logs already captured in monitor loop)
        if self._container_handle:
            try:
                self._container_handle.cleanup()
            except Exception as e:
                logger.warning("Failed to cleanup container handle for task %s: %s", self.task_id, e)

        # Release ports
        try:
            self._port_allocator.release(list(self.ports.values()))
        except Exception as e:
            logger.warning("Failed to release ports for task %s: %s", self.task_id, e)

        # Remove working directory (handle.cleanup() already released backing storage)
        if self.workdir and self.workdir.exists():
            try:
                shutil.rmtree(self.workdir)
            except Exception as e:
                logger.warning("Failed to remove %s: %s", self.workdir, e)
