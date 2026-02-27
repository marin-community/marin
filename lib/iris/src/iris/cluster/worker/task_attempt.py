# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task execution attempt handling.

This module encapsulates the full lifecycle of a single task execution attempt:
bundle download -> image build -> container run -> monitor -> cleanup.
"""

import json
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
    ContainerRuntime,
    RuntimeLogReader,
)
from google.protobuf import json_format

from iris.cluster.types import (
    JobName,
    is_task_finished,
)
from iris.cluster.worker.bundle_cache import BundleProvider
from iris.cluster.worker.env_probe import collect_workdir_size_mb
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.task_logging import LogSink
from iris.rpc import cluster_pb2, logging_pb2
from iris.rpc.cluster_pb2 import TaskState, WorkerMetadata
from iris.rpc.errors import format_exception_with_traceback
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)

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


class TaskCancelled(Exception):
    """Raised when a task is cancelled during execution."""

    pass


@dataclass
class TaskAttemptConfig:
    """Immutable configuration for a task attempt, derived from the RPC request."""

    task_id: JobName
    num_tasks: int
    attempt_id: int
    request: cluster_pb2.Worker.RunTaskRequest
    ports: dict[str, int]
    workdir: Path
    cache_dir: Path


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

    Auto-injects task metadata and configuration that tasks need to interact
    with the Iris cluster (task ID, job ID, worker ID, controller address, ports).
    These override user-provided values.

    Args:
        task: TaskAttempt object with metadata
        worker_id: Worker identifier, if registered with controller
        controller_address: Controller RPC address, if configured

    Returns:
        Dictionary of environment variables to inject into the task container
    """
    env = {}

    # N.B. This needs to mirror JobInfo.from_env()
    # XXX: Should we move this code there instead?
    # Core task metadata
    env["IRIS_JOB_ID"] = task.task_id.to_wire()
    env["IRIS_NUM_TASKS"] = str(task.num_tasks)
    env["IRIS_ATTEMPT_ID"] = str(task.attempt_id)
    env["IRIS_BUNDLE_GCS_PATH"] = task.request.bundle_gcs_path

    if worker_id:
        env["IRIS_WORKER_ID"] = worker_id

    if controller_address:
        # With --network=host, containers share the host's network directly,
        # so no address rewriting is needed.
        env["IRIS_CONTROLLER_ADDRESS"] = controller_address

    # Inject bundle path for sub-task inheritance
    if task.request.bundle_gcs_path:
        env["IRIS_BUNDLE_GCS_PATH"] = task.request.bundle_gcs_path

    # With --network=host, containers share the host's network stack.
    # Compute the host's routable IP so container code can read it via
    # get_job_info().advertise_host without needing its own socket tricks.
    env["IRIS_BIND_HOST"] = "0.0.0.0"
    env["IRIS_ADVERTISE_HOST"] = _get_host_ip()
    env["IRIS_WORKDIR"] = "/app"
    env["IRIS_PYTHON"] = "python"

    # Propagate extras and pip_packages so child jobs can inherit them
    extras = list(task.request.environment.extras)
    if extras:
        env["IRIS_JOB_EXTRAS"] = json.dumps(extras)
    pip_packages = list(task.request.environment.pip_packages)
    if pip_packages:
        env["IRIS_JOB_PIP_PACKAGES"] = json.dumps(pip_packages)

    # Serialize the explicit user env vars so child jobs can inherit them
    # via JobInfo.env without picking up infrastructure vars from os.environ.
    user_env_vars = dict(task.request.environment.env_vars)
    if user_env_vars:
        env["IRIS_JOB_ENV"] = json.dumps(user_env_vars)
    if task.request.constraints:
        env["IRIS_JOB_CONSTRAINTS"] = json.dumps(
            [json_format.MessageToDict(c, preserving_proto_field_name=True) for c in task.request.constraints]
        )

    # Inject allocated ports
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
        bundle_provider: BundleProvider,
        container_runtime: ContainerRuntime,
        worker_metadata: WorkerMetadata,
        worker_id: str | None,
        controller_address: str | None,
        default_task_env: dict[str, str],
        default_task_image: str | None,
        resolve_image: Callable[[str], str],
        port_allocator: PortAllocator,
        report_state: Callable[[], None],
        log_sink: LogSink,
        poll_interval_seconds: float = 5.0,
    ):
        """Initialize a TaskAttempt.

        Args:
            config: Immutable configuration for this attempt
            bundle_provider: Provider for downloading code bundles
            container_runtime: Runtime for creating and managing containers
            worker_metadata: Worker's hardware/environment metadata
            worker_id: Worker identifier for env injection
            controller_address: Controller address for env injection
            default_task_env: Worker-level default env vars injected into task containers
            default_task_image: Fully-qualified task container image from cluster config
            resolve_image: Resolves image tags for the current platform
                (e.g. GHCR→AR rewriting on GCP). Zone is pre-bound by the worker.
            port_allocator: Port allocator for retry logic
            report_state: Callback to report task state changes to Worker
            poll_interval_seconds: How often to poll container status
            log_sink: Persistent log sink for this task attempt
        """
        self._bundle_provider = bundle_provider
        self._runtime = container_runtime
        self._worker_metadata = worker_metadata
        self._worker_id = worker_id
        self._controller_address = controller_address
        self._default_task_env = default_task_env
        self._default_task_image = default_task_image
        self._resolve_image_fn = resolve_image
        self._port_allocator = port_allocator
        self._report_state = report_state
        self._poll_interval_seconds = poll_interval_seconds
        self._log_sink = log_sink

        # Task identity (from config)
        self.task_id: JobName = config.task_id
        self.num_tasks: int = config.num_tasks
        self.attempt_id: int = config.attempt_id
        self.request: cluster_pb2.Worker.RunTaskRequest = config.request
        self.ports: dict[str, int] = config.ports
        self.workdir: Path | None = config.workdir
        self._cache_dir: Path = config.cache_dir
        # Task state
        self.status: TaskState = cluster_pb2.TASK_STATE_PENDING
        self.exit_code: int | None = None
        self.error: str | None = None
        self.started_at: Timestamp | None = None
        self.finished_at: Timestamp | None = None
        self.status_message: str = ""

        # Resource tracking
        self.current_memory_mb: int = 0
        self.peak_memory_mb: int = 0
        self.current_cpu_percent: int = 0
        self.process_count: int = 0
        self.disk_mb: int = 0

        # Build tracking
        self.build_started: Timestamp | None = None
        self.build_finished: Timestamp | None = None
        self.build_from_cache: bool = False
        self.image_tag: str = ""

        # Internals
        self._container_handle: ContainerHandle | None = None
        self.thread: threading.Thread | None = None
        self.cleanup_done: bool = False
        self.should_stop: bool = False

    @property
    def container_id(self) -> str | None:
        """Return the container ID from the handle, if available."""
        if self._container_handle:
            return self._container_handle.container_id
        return None

    @property
    def log_directory(self) -> str:
        """Return the storage directory for this task's logs."""
        return self._log_sink.log_path

    def recent_logs(self, max_entries: int = 0) -> list[logging_pb2.LogEntry]:
        """Return recent logs from the canonical log sink."""
        return self._log_sink.query_recent(max_entries=max_entries)

    def stop(self, force: bool = False) -> None:
        """Stop the container, if running."""
        self.should_stop = True
        if self._container_handle:
            self._container_handle.stop(force=force)

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

    def duration(self) -> Duration | None:
        """Calculate how long the attempt ran.

        Returns:
            Duration from started_at to finished_at, or None if not finished
        """
        if self.finished_at is None:
            return None
        elapsed_ms = self.finished_at.epoch_ms() - self.started_at.epoch_ms()
        return Duration.from_ms(elapsed_ms)

    def to_proto(self) -> cluster_pb2.TaskStatus:
        proto = cluster_pb2.TaskStatus(
            task_id=self.task_id.to_wire(),
            state=self.status,
            exit_code=self.exit_code or 0,
            error=self.error or "",
            ports=self.ports,
            current_attempt_id=self.attempt_id,
            resource_usage=cluster_pb2.ResourceUsage(
                memory_mb=self.current_memory_mb,
                memory_peak_mb=self.peak_memory_mb,
                disk_mb=self.disk_mb,
                cpu_millicores=self.current_cpu_percent * 10,
                cpu_percent=self.current_cpu_percent,
                process_count=self.process_count,
            ),
            build_metrics=cluster_pb2.BuildMetrics(
                from_cache=self.build_from_cache,
                image_tag=self.image_tag,
            ),
        )

        # Set timestamp fields using proto Timestamp messages
        if self.started_at is not None:
            proto.started_at.CopyFrom(self.started_at.to_proto())
        if self.finished_at is not None:
            proto.finished_at.CopyFrom(self.finished_at.to_proto())
        if self.build_started is not None:
            proto.build_metrics.build_started.CopyFrom(self.build_started.to_proto())
        if self.build_finished is not None:
            proto.build_metrics.build_finished.CopyFrom(self.build_finished.to_proto())

        return proto

    def _check_cancelled(self) -> None:
        """Check if task has been cancelled and raise if so."""
        if self.should_stop:
            raise TaskCancelled("Task was cancelled")

    def run(self) -> None:
        """Execute the full task lifecycle. Intended to run in a background thread.

        The lifecycle is:
        1. Download bundle from GCS
        2. Resolve base image
        3. Create container handle
        4. Build phase: run setup_commands (uv sync) - BUILDING state
        5. Run phase: start main command - RUNNING state
        6. Monitor until completion
        """
        logger.info(
            "TaskAttempt starting: task_id=%s attempt=%s num_tasks=%s",
            self.task_id,
            self.attempt_id,
            self.num_tasks,
        )
        try:
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
            self.transition_to(cluster_pb2.TASK_STATE_KILLED)
        except Exception as e:
            error_msg = format_exception_with_traceback(e)
            self._log_sink.append(source="error", data=f"Task failed:\n{error_msg}")
            self.transition_to(cluster_pb2.TASK_STATE_FAILED, error=error_msg)
        finally:
            if is_task_finished(self.status):
                # Flush logs to storage before notifying controller, so that
                # clients who fetch logs immediately after seeing the terminal
                # state will find complete data on disk.
                self._log_sink.sync()
                self._report_state()
            self._cleanup()
            logger.info(
                "TaskAttempt finished: task_id=%s attempt=%s state=%s exit_code=%s",
                self.task_id,
                self.attempt_id,
                self.status,
                self.exit_code,
            )

    def _download_bundle(self) -> None:
        """Download the code bundle from GCS.

        Transitions task to BUILDING state and performs chaos injection checks
        for testing delayed builds.
        """
        self.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="downloading bundle")
        self.started_at = Timestamp.now()
        self._building_start_monotonic = time.monotonic()
        self._report_state()  # Report BUILDING state to controller

        download_start = time.monotonic()

        # Chaos injection for testing failures during download
        chaos_raise("worker.bundle_download")

        # Chaos injection for testing delayed builds (for screenshot tests)
        if rule := chaos("worker.building_delay"):
            time.sleep(rule.delay_seconds)

        # Periodically check should_stop during download to support kill during BUILDING
        # (RF-3: For now, we defer kill handling until container starts, as bundle
        # downloads are typically fast. Future work could add cancellation support
        # to BundleProvider.get_bundle if long downloads become a problem.)

        assert self.workdir is not None
        self._runtime.stage_bundle(
            bundle_gcs_path=self.request.bundle_gcs_path,
            workdir=self.workdir,
            workdir_files=dict(self.request.entrypoint.workdir_files),
            fetch_bundle=lambda path: self._bundle_provider.get_bundle(path, expected_hash=None),
        )
        logger.info(
            "Bundle staged for task %s in %.2fs",
            self.task_id,
            time.monotonic() - download_start,
        )

    def _resolve_image(self) -> None:
        """Resolve the task image from cluster config.

        No per-job Docker build — the pre-built base image has a pre-warmed
        uv cache. The remote client wraps the entrypoint with uv sync.
        """
        if not self._default_task_image:
            raise ValueError("No task image configured. Set defaults.default_task_image in cluster config.")
        self.image_tag = self._resolve_image_fn(self._default_task_image)

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
        env.update(self._default_task_env)
        env.update(dict(self.request.environment.env_vars))

        # uv needs a writable directory for Python downloads.
        # Use a subdirectory of the cache which is bind-mounted from the worker.
        env["UV_PYTHON_INSTALL_DIR"] = "/uv/cache/python"
        env["CARGO_TARGET_DIR"] = "/root/.cargo/target"

        # Get RuntimeEntrypoint proto directly
        rt_ep = self.request.entrypoint

        # Extract timeout from proto (0 or unset means no timeout)
        timeout_seconds = None
        if self.request.HasField("timeout") and self.request.timeout.milliseconds > 0:
            timeout_seconds = self.request.timeout.milliseconds / 1000

        assert self.workdir is not None
        job_id, _ = self.task_id.require_task()

        # Pre-create cache mount directories so Docker doesn't create them as root
        uv_cache = self._cache_dir / "uv"
        cargo_cache = self._cache_dir / "cargo"
        cargo_target = self._cache_dir / "cargo-target"
        uv_cache.mkdir(parents=True, exist_ok=True)
        cargo_cache.mkdir(parents=True, exist_ok=True)
        cargo_target.mkdir(parents=True, exist_ok=True)

        mounts = [
            (str(self.workdir), "/app", "rw"),
            (str(uv_cache), "/uv/cache", "rw"),
            (str(cargo_cache), "/root/.cargo/registry", "rw"),
            (str(cargo_target), "/root/.cargo/target", "rw"),
        ]

        config = ContainerConfig(
            image=self.image_tag,
            entrypoint=rt_ep,
            env=env,
            resources=self.request.resources if self.request.HasField("resources") else None,
            timeout_seconds=timeout_seconds,
            mounts=mounts,
            task_id=self.task_id.to_wire(),
            attempt_id=self.attempt_id,
            job_id=job_id.to_wire(),
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
            self.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="syncing dependencies")
            self.build_started = Timestamp.now()
            self._report_state()

        build_logs = self._container_handle.build()

        # Capture build logs into log sink
        for log_line in build_logs:
            self._log_sink.append(source=log_line.source, data=log_line.data)

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
        """Monitor task execution: check status, collect stats, stream logs, handle timeouts.

        Polls container status at regular intervals until the container stops.
        Streams logs incrementally into task.logs (single source of truth).
        Collects runtime statistics (CPU, memory, disk) and handles timeout enforcement.
        Updates task state to terminal status (SUCCEEDED/FAILED/KILLED) when container stops.
        """
        assert self._container_handle is not None
        handle = self._container_handle

        # Create deadline from timeout if specified (0 or unset means no timeout)
        deadline = None
        if self.request.HasField("timeout") and self.request.timeout.milliseconds > 0:
            timeout_seconds = self.request.timeout.milliseconds / 1000
            deadline = Deadline.from_seconds(timeout_seconds)

        log_reader = handle.log_reader()

        while True:
            if rule := chaos("worker.task_monitor"):
                time.sleep(rule.delay_seconds)
                self.transition_to(cluster_pb2.TASK_STATE_FAILED, error="chaos: monitor crashed")
                break

            # Check if we should stop
            if self.should_stop:
                handle.stop(force=True)
                logger.info("Task %s requested stop; killing container %s", self.task_id, self.container_id)
                self._stream_logs(log_reader)  # Capture final logs
                self.transition_to(cluster_pb2.TASK_STATE_KILLED)
                break

            # Check timeout
            if deadline and deadline.expired():
                handle.stop(force=True)
                self._stream_logs(log_reader)  # Capture final logs
                self.transition_to(
                    cluster_pb2.TASK_STATE_FAILED,
                    error="Timeout exceeded",
                    exit_code=-1,
                )
                break

            # Check container status
            status = handle.status()

            if self.status == cluster_pb2.TASK_STATE_BUILDING and status.running and status.ready:
                building_duration = time.monotonic() - self._building_start_monotonic
                logger.info("Task %s BUILDING→RUNNING after %.1fs", self.task_id, building_duration)
                self.transition_to(cluster_pb2.TASK_STATE_RUNNING)
                self._report_state()

            if not status.running:
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
                    failure_state = cluster_pb2.TASK_STATE_FAILED
                    if status.error_kind == ContainerErrorKind.INFRA_NOT_FOUND:
                        failure_state = cluster_pb2.TASK_STATE_WORKER_FAILED
                    self.transition_to(failure_state, error=status.error, exit_code=status.exit_code or -1)
                elif status.exit_code == 0:
                    self.transition_to(cluster_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
                else:
                    stderr_line = None
                    for entry in reversed(log_reader.read_all()):
                        if entry.source == "stderr" and entry.data:
                            stderr_line = entry.data
                            break
                    error = _format_exit_error(status.exit_code, status.oom_killed)
                    if stderr_line:
                        error = f"{error}. stderr: {stderr_line}"
                    if status.oom_killed:
                        self._log_sink.append(source="error", data="Container was OOM killed by the kernel")
                    self.transition_to(
                        cluster_pb2.TASK_STATE_FAILED,
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
                    self.current_cpu_percent = stats.cpu_percent
                    self.process_count = stats.process_count
                    if stats.memory_mb > self.peak_memory_mb:
                        self.peak_memory_mb = stats.memory_mb

                if self.workdir:
                    self.disk_mb = collect_workdir_size_mb(self.workdir)
            except Exception:
                pass  # Don't fail task on stats collection errors

            # Sleep before next poll
            time.sleep(self._poll_interval_seconds)

    def _stream_logs(self, reader: RuntimeLogReader) -> None:
        """Fetch new logs from container and append to log sink."""
        try:
            for log_line in reader.read():
                self._log_sink.append(source=log_line.source, data=log_line.data)
        except Exception:
            pass  # Don't fail task on log streaming errors

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

        # Final log sync and metadata write
        if is_task_finished(self.status):
            try:
                # Write metadata
                metadata = logging_pb2.TaskAttemptMetadata(
                    task_id=self.task_id.to_wire(),
                    attempt_id=self.attempt_id,
                    worker_id=self._worker_id or "unknown",
                    status=self.status,
                    exit_code=self.exit_code or 0,
                    oom_killed=self._container_handle.status().oom_killed if self._container_handle else False,
                    error_message=self.error or "",
                )

                # Add timestamps if available
                if self.started_at:
                    metadata.start_time.CopyFrom(self.started_at.to_proto())
                if self.finished_at:
                    metadata.end_time.CopyFrom(self.finished_at.to_proto())

                # Add resource usage if available
                if self.peak_memory_mb > 0:
                    metadata.resource_usage.CopyFrom(
                        logging_pb2.ResourceUsage(
                            peak_memory_bytes=self.peak_memory_mb * 1024 * 1024,
                            cpu_seconds=0,  # Not tracked currently
                            gpu_memory_bytes=0,  # Not tracked currently
                        )
                    )

                self._log_sink.write_metadata(metadata)
            except Exception as e:
                logger.error(f"Failed to write final logs/metadata: {e}")

        self._log_sink.close()

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

        # Remove working directory
        if self.workdir and self.workdir.exists():
            try:
                shutil.rmtree(self.workdir)
            except Exception as e:
                logger.warning("Failed to remove %s: %s", self.workdir, e)
