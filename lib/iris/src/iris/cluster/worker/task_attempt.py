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
from iris.cluster.types import Entrypoint, JobName, is_task_finished
from iris.cluster.worker.builder import ImageProvider
from iris.cluster.worker.bundle_cache import BundleProvider
from iris.cluster.worker.docker import ContainerConfig, ContainerRuntime
from iris.cluster.worker.env_probe import collect_workdir_size_mb
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.worker_types import TaskLogs
from iris.rpc import cluster_pb2
from iris.rpc.cluster_pb2 import TaskState, WorkerMetadata
from iris.rpc.errors import format_exception_with_traceback
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)


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

    # Propagate the dockerfile so child jobs can inherit it via get_job_info().dockerfile
    # instead of regenerating (which would lose extras like --package marin --extra cpu).
    dockerfile = task.request.environment.dockerfile
    if dockerfile:
        env["IRIS_DOCKERFILE"] = dockerfile

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
    code should only read via TaskInfo protocol (status, result, to_proto()).
    """

    def __init__(
        self,
        config: TaskAttemptConfig,
        bundle_provider: BundleProvider,
        image_provider: ImageProvider | None,
        container_runtime: ContainerRuntime,
        worker_metadata: WorkerMetadata,
        worker_id: str | None,
        controller_address: str | None,
        port_allocator: PortAllocator,
        report_state: Callable[[], None],
        poll_interval_seconds: float = 5.0,
    ):
        """Initialize a TaskAttempt.

        Args:
            config: Immutable configuration for this attempt
            bundle_provider: Provider for downloading code bundles
            image_provider: Provider for building container images (may be None)
            container_runtime: Runtime for creating and managing containers
            worker_metadata: Worker's hardware/environment metadata
            worker_id: Worker identifier for env injection
            controller_address: Controller address for env injection
            port_allocator: Port allocator for retry logic
            report_state: Callback to report task state changes to Worker
            poll_interval_seconds: How often to poll container status
        """
        self._bundle_provider = bundle_provider
        self._image_provider = image_provider
        self._runtime = container_runtime
        self._worker_metadata = worker_metadata
        self._worker_id = worker_id
        self._controller_address = controller_address
        self._port_allocator = port_allocator
        self._report_state = report_state
        self._poll_interval_seconds = poll_interval_seconds

        # Task identity (from config)
        self.task_id: JobName = config.task_id
        self.num_tasks: int = config.num_tasks
        self.attempt_id: int = config.attempt_id
        self.request: cluster_pb2.Worker.RunTaskRequest = config.request
        self.ports: dict[str, int] = config.ports
        self.workdir: Path | None = config.workdir

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
        self.container_id: str | None = None
        self.thread: threading.Thread | None = None
        self.cleanup_done: bool = False
        self.should_stop: bool = False

        # Structured logs (build logs stored here, container logs fetched from Docker)
        self.logs: TaskLogs = TaskLogs()

        self.result: bytes | None = None  # cloudpickle serialized return value from container

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
        """Execute the full task lifecycle. Intended to run in a background thread."""
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
            self._build_image()
            self._check_cancelled()
            container_id = self._start_container()
            self._monitor(container_id)
        except TaskCancelled:
            self.transition_to(cluster_pb2.TASK_STATE_KILLED)
        except Exception as e:
            error_msg = format_exception_with_traceback(e)
            self.logs.add("error", f"Task failed:\n{error_msg}")
            self.transition_to(cluster_pb2.TASK_STATE_FAILED, error=error_msg)
        finally:
            if is_task_finished(self.status):
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

        self._bundle_path = self._bundle_provider.get_bundle(
            self.request.bundle_gcs_path,
            expected_hash=None,
        )
        logger.info(
            "Bundle downloaded for task %s in %.2fs",
            self.task_id,
            time.monotonic() - download_start,
        )

    def _build_image(self) -> None:
        """Build the container image from the downloaded bundle.

        Uses the pre-generated dockerfile from the client rather than
        regenerating it, which preserves extras like package:extra syntax.
        """
        self.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="building image")
        self.build_started = Timestamp.now()
        self._report_state()  # Report BUILDING state to controller
        build_start = time.monotonic()

        env_config = self.request.environment
        dockerfile = env_config.dockerfile
        if not dockerfile:
            raise RuntimeError(
                f"Task {self.task_id} has no dockerfile in environment config. "
                "The client must provide a dockerfile when submitting jobs."
            )

        self.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="populating uv cache")
        self.logs.add("build", "Building Docker image...")
        self._report_state()

        if not self._image_provider:
            raise RuntimeError("Image provider not configured")

        job_id, _ = self.task_id.require_task()
        build_result = self._image_provider.build(
            bundle_path=self._bundle_path,
            dockerfile=dockerfile,
            job_id=job_id.to_wire(),
            task_logs=self.logs,
        )

        self.build_finished = Timestamp.now()
        self.build_from_cache = build_result.from_cache
        self.image_tag = build_result.image_tag
        logger.info(
            "Image build complete for task %s in %.2fs (cached=%s, tag=%s)",
            self.task_id,
            time.monotonic() - build_start,
            build_result.from_cache,
            build_result.image_tag,
        )

        # Protect image from eviction while task is running
        self._image_provider.protect(build_result.image_tag)

    def _start_container(self) -> str:
        """Create and start the container for the task.

        Builds environment variables, creates container config, and handles
        port binding retries. Returns the container ID.

        Returns:
            Container ID string

        Raises:
            RuntimeError: If container creation fails after retries
        """
        self.transition_to(cluster_pb2.TASK_STATE_RUNNING)
        self._report_state()

        # Build environment from user-provided vars + EnvironmentConfig
        env_config = self.request.environment
        env = dict(env_config.env_vars)

        iris_env = build_iris_env(
            self,
            self._worker_id,
            self._controller_address,
        )
        env.update(iris_env)

        # Convert proto entrypoint to typed Entrypoint
        entrypoint = Entrypoint.from_proto(self.request.entrypoint)

        # Extract timeout from proto (0 or unset means no timeout)
        timeout_seconds = None
        if self.request.HasField("timeout") and self.request.timeout.milliseconds > 0:
            timeout_seconds = self.request.timeout.milliseconds / 1000

        job_id, _ = self.task_id.require_task()
        config = ContainerConfig(
            image=self.image_tag,
            entrypoint=entrypoint,
            env=env,
            resources=self.request.resources if self.request.HasField("resources") else None,
            timeout_seconds=timeout_seconds,
            ports=self.ports,
            mounts=[(str(self.workdir), "/workdir", "rw")],
            task_id=self.task_id.to_wire(),
            job_id=job_id.to_wire(),
            worker_metadata=self._worker_metadata,
        )

        # Create and start container with retry on port binding failures
        container_id = None
        max_port_retries = 3
        for attempt in range(max_port_retries):
            try:
                chaos_raise("worker.create_container")
                container_id = self._runtime.create_container(config)
                self.container_id = container_id
                self._runtime.start_container(container_id)
                logger.info(
                    "Container started for task %s (container_id=%s, ports=%s, timeout=%s)",
                    self.task_id,
                    container_id,
                    self.ports,
                    timeout_seconds,
                )
                break
            except RuntimeError as e:
                if "address already in use" in str(e) and attempt < max_port_retries - 1:
                    logger.warning(
                        "Port conflict for task %s, retrying with new ports (attempt %d)",
                        self.task_id,
                        attempt + 2,
                    )
                    self.logs.add("build", f"Port conflict, retrying with new ports (attempt {attempt + 2})")
                    # Release current ports and allocate new ones
                    self._port_allocator.release(list(self.ports.values()))
                    port_names = list(self.ports.keys())
                    new_ports = self._port_allocator.allocate(len(port_names))
                    self.ports = dict(zip(port_names, new_ports, strict=True))

                    # Update config with new ports
                    config.ports = self.ports
                    for name, port in self.ports.items():
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
        return container_id

    def _monitor(self, container_id: str) -> None:
        """Monitor task execution: check status, collect stats, stream logs, handle timeouts.

        Polls container status at regular intervals until the container stops.
        Streams logs incrementally into task.logs (single source of truth).
        Collects runtime statistics (CPU, memory, disk) and handles timeout enforcement.
        Updates task state to terminal status (SUCCEEDED/FAILED/KILLED) when container stops.

        Args:
            container_id: Container to monitor
        """
        # Create deadline from timeout if specified (0 or unset means no timeout)
        deadline = None
        if self.request.HasField("timeout") and self.request.timeout.milliseconds > 0:
            timeout_seconds = self.request.timeout.milliseconds / 1000
            deadline = Deadline.from_seconds(timeout_seconds)

        # Track last log timestamp for incremental fetching
        last_log_time: Timestamp | None = None

        while True:
            if rule := chaos("worker.task_monitor"):
                time.sleep(rule.delay_seconds)
                self.transition_to(cluster_pb2.TASK_STATE_FAILED, error="chaos: monitor crashed")
                break

            # Check if we should stop
            if self.should_stop:
                self._runtime.kill(container_id, force=True)
                logger.info("Task %s requested stop; killing container %s",
                            self.task_id, container_id)
                self.transition_to(cluster_pb2.TASK_STATE_KILLED)
                break

            # Check timeout
            if deadline and deadline.expired():
                self._runtime.kill(container_id, force=True)
                self.transition_to(
                    cluster_pb2.TASK_STATE_FAILED,
                    error="Timeout exceeded",
                    exit_code=-1,
                )
                break

            # Check container status
            status = self._runtime.inspect(container_id)
            if not status.running:
                logger.info(
                    "Container exited for task %s (container_id=%s, exit_code=%s, error=%s)",
                    self.task_id,
                    container_id,
                    status.exit_code,
                    status.error,
                )
                # Final log fetch before container stops
                last_log_time = self._stream_logs(container_id, last_log_time)

                # Read result file only if container succeeded
                if status.exit_code == 0 and self.workdir:
                    result_path = self.workdir / "_result.pkl"
                    if result_path.exists():
                        try:
                            self.result = result_path.read_bytes()
                        except Exception as e:
                            self.logs.add("error", f"Failed to read result file: {e}")

                # Container has stopped
                if status.error:
                    self.transition_to(
                        cluster_pb2.TASK_STATE_FAILED,
                        error=status.error,
                        exit_code=status.exit_code or -1,
                    )
                elif status.exit_code == 0:
                    self.transition_to(cluster_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
                else:
                    stderr_line = None
                    for entry in reversed(self._runtime.get_logs(container_id)):
                        if entry.source == "stderr" and entry.data:
                            stderr_line = entry.data
                            break
                    error = f"Exit code: {status.exit_code}"
                    if stderr_line:
                        error = f"{error}. stderr: {stderr_line}"
                    self.transition_to(
                        cluster_pb2.TASK_STATE_FAILED,
                        error=error,
                        exit_code=status.exit_code or -1,
                    )
                break

            # Stream logs incrementally
            last_log_time = self._stream_logs(container_id, last_log_time)

            # Collect stats
            try:
                stats = self._runtime.get_stats(container_id)
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

    def _stream_logs(self, container_id: str, since: Timestamp | None) -> Timestamp | None:
        """Fetch new logs from container and append to task.logs.

        Args:
            container_id: Container to fetch logs from
            since: Timestamp to fetch logs after (None for all logs)

        Returns:
            Timestamp of the last log line, or the input 'since' if no new logs
        """
        try:
            new_logs = self._runtime.get_logs(container_id, since=since)
            for log_line in new_logs:
                ts = Timestamp.from_seconds(log_line.timestamp.timestamp())
                self.logs.add(log_line.source, log_line.data, timestamp=ts)
            if new_logs:
                return Timestamp.from_seconds(new_logs[-1].timestamp.timestamp())
        except Exception:
            pass  # Don't fail task on log streaming errors
        return since

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

        # Remove container (logs already captured in monitor loop)
        if self.container_id:
            try:
                self._runtime.remove(self.container_id)
            except Exception as e:
                logger.warning("Failed to remove container %s: %s", self.container_id, e)

        # Release ports
        try:
            self._port_allocator.release(list(self.ports.values()))
        except Exception as e:
            logger.warning("Failed to release ports for task %s: %s", self.task_id, e)

        # Unprotect image from eviction now that task is done
        if self._image_provider and self.image_tag:
            try:
                self._image_provider.unprotect(self.image_tag)
            except Exception as e:
                logger.warning("Failed to unprotect image %s: %s", self.image_tag, e)

        # Remove working directory
        if self.workdir and self.workdir.exists():
            try:
                shutil.rmtree(self.workdir)
            except Exception as e:
                logger.warning("Failed to remove workdir %s: %s", self.workdir, e)
