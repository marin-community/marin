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
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from iris.chaos import chaos, chaos_raise
from iris.cluster.types import Entrypoint, is_task_finished
from iris.cluster.worker.bundle_cache import BundleProvider
from iris.cluster.worker.builder import ImageProvider
from iris.cluster.worker.docker import ContainerConfig, ContainerRuntime, DockerRuntime
from iris.cluster.worker.env_probe import collect_workdir_size_mb
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.worker_types import Task
from iris.rpc import cluster_pb2
from iris.rpc.cluster_pb2 import WorkerMetadata
from iris.rpc.errors import format_exception_with_traceback
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)


@dataclass
class TaskAttemptConfig:
    """Immutable configuration for a task attempt, derived from the RPC request."""

    task_id: str
    job_id: str
    task_index: int
    num_tasks: int
    attempt_id: int
    request: cluster_pb2.Worker.RunTaskRequest
    ports: dict[str, int]
    workdir: Path


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


def build_iris_env(
    task: Task,
    worker_id: str | None,
    controller_address: str | None,
    runtime_type: type[ContainerRuntime],
) -> dict[str, str]:
    """Build Iris system environment variables for the task container.

    Auto-injects task metadata and configuration that tasks need to interact
    with the Iris cluster (task ID, job ID, worker ID, controller address, ports).
    These override user-provided values.

    Args:
        task: Task object with metadata
        worker_id: Worker identifier, if registered with controller
        controller_address: Controller RPC address, if configured
        runtime_type: Type of container runtime (e.g., DockerRuntime)

    Returns:
        Dictionary of environment variables to inject into the task container
    """
    env = {}

    # Core task metadata
    env["IRIS_JOB_ID"] = task.job_id
    env["IRIS_TASK_ID"] = task.task_id
    env["IRIS_TASK_INDEX"] = str(task.task_index)
    env["IRIS_NUM_TASKS"] = str(task.num_tasks)
    env["IRIS_ATTEMPT_ID"] = str(task.attempt_id)

    if worker_id:
        env["IRIS_WORKER_ID"] = worker_id

    if controller_address:
        # Only rewrite localhost addresses for Docker containers
        if runtime_type is DockerRuntime:
            env["IRIS_CONTROLLER_ADDRESS"] = _rewrite_address_for_container(controller_address)
        else:
            env["IRIS_CONTROLLER_ADDRESS"] = controller_address

    # Inject bundle path for sub-task inheritance
    if task.request.bundle_gcs_path:
        env["IRIS_BUNDLE_GCS_PATH"] = task.request.bundle_gcs_path

    # Inject bind host - 0.0.0.0 for Docker (so port mapping works), 127.0.0.1 otherwise
    # Also inject advertise host - the address other containers should use to reach this one
    if runtime_type is DockerRuntime:
        env["IRIS_BIND_HOST"] = "0.0.0.0"
        env["IRIS_ADVERTISE_HOST"] = "host.docker.internal"
    else:
        env["IRIS_BIND_HOST"] = "127.0.0.1"
        env["IRIS_ADVERTISE_HOST"] = "127.0.0.1"

    # Inject allocated ports
    for name, port in task.ports.items():
        env[f"IRIS_PORT_{name.upper()}"] = str(port)

    return env


class TaskAttempt:
    """Manages the lifecycle of a single task execution attempt.

    Owns the full pipeline: bundle download -> image build -> container run -> monitor.
    Reports state changes back to the worker via a callback.

    This class is module-internal to iris.cluster.worker and has no external
    consumers. It encapsulates the complex task execution logic that was
    previously interleaved in the Worker class.
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
        report_state: Callable[[Task], None],
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
        self._config = config
        self._bundle_provider = bundle_provider
        self._image_provider = image_provider
        self._runtime = container_runtime
        self._worker_metadata = worker_metadata
        self._worker_id = worker_id
        self._controller_address = controller_address
        self._port_allocator = port_allocator
        self._report_state = report_state
        self._poll_interval_seconds = poll_interval_seconds

        # IMPORTANT: This task object is shared with Worker._tasks[task_id].
        # Both TaskAttempt and Worker hold references to the same mutable Task object.
        # The TaskAttempt mutates it, and Worker reads it for status reporting.
        # No synchronization is needed because Python's GIL ensures atomicity for
        # single-statement assignments, and we only transition state in one direction.
        self.task = Task(
            task_id=config.task_id,
            job_id=config.job_id,
            task_index=config.task_index,
            num_tasks=config.num_tasks,
            attempt_id=config.attempt_id,
            request=config.request,
            status=cluster_pb2.TASK_STATE_PENDING,
            ports=config.ports,
            workdir=config.workdir,
        )

    def run(self) -> None:
        """Execute the full task lifecycle. Intended to run in a background thread."""
        try:
            self._download_bundle()
            self._build_image()
            container_id = self._start_container()
            self._report_state(self.task)
            self._monitor(container_id)
        except Exception as e:
            error_msg = format_exception_with_traceback(e)
            self.task.logs.add("error", f"Task failed:\n{error_msg}")
            self.task.transition_to(cluster_pb2.TASK_STATE_FAILED, error=error_msg)
        finally:
            if is_task_finished(self.task.status):
                self._report_state(self.task)
            self._cleanup()

    def _download_bundle(self) -> None:
        """Download the code bundle from GCS.

        Transitions task to BUILDING state and performs chaos injection checks
        for testing delayed builds.
        """
        self.task.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="downloading bundle")
        self.task.started_at_ms = now_ms()
        self._report_state(self.task)  # Report BUILDING state to controller

        # Chaos injection for testing failures during download
        chaos_raise("worker.bundle_download")

        # Chaos injection for testing delayed builds (for screenshot tests)
        # This checks for a delay rule and sleeps if configured, without raising
        chaos("worker.building_delay")

        # Periodically check should_stop during download to support kill during BUILDING
        # (RF-3: For now, we defer kill handling until container starts, as bundle
        # downloads are typically fast. Future work could add cancellation support
        # to BundleProvider.get_bundle if long downloads become a problem.)

        self._bundle_path = self._bundle_provider.get_bundle(
            self.task.request.bundle_gcs_path,
            expected_hash=None,
        )

    def _build_image(self) -> None:
        """Build the container image from the downloaded bundle.

        Handles uv cache population and builds with appropriate Python version
        and base image selection.
        """
        self.task.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="building image")
        self.task.build_started_ms = now_ms()

        # Periodically check should_stop during build to support kill during BUILDING
        # (RF-3: Similar to bundle download, we defer kill handling for now since
        # image builds are handled by ImageProvider which doesn't expose cancellation.
        # Most builds are fast due to caching.)

        env_config = self.task.request.environment
        extras = list(env_config.extras)

        self.task.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="populating uv cache")
        self.task.logs.add("build", "Building Docker image...")

        # Use the client's Python version for the task container so cloudpickle
        # bytecode is deserialized with the same Python that serialized it.
        py_version = env_config.python_version or f"{sys.version_info.major}.{sys.version_info.minor}"

        pip_packages = list(env_config.pip_packages) if env_config.pip_packages else None

        # Use the full Python image when additional pip packages are requested,
        # since native wheels (e.g. libtpu for jax[tpu]) often depend on system
        # libraries (libstdc++, libgomp, etc.) that python:*-slim strips out.
        if pip_packages:
            base_image = f"python:{py_version}"
        else:
            base_image = f"python:{py_version}-slim"

        if not self._image_provider:
            raise RuntimeError("Image provider not configured")

        build_result = self._image_provider.build(
            bundle_path=self._bundle_path,
            base_image=base_image,
            extras=extras,
            job_id=self.task.job_id,
            task_logs=self.task.logs,
            pip_packages=pip_packages,
        )

        self.task.build_finished_ms = now_ms()
        self.task.build_from_cache = build_result.from_cache
        self.task.image_tag = build_result.image_tag

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
        self.task.transition_to(cluster_pb2.TASK_STATE_RUNNING)

        # Build environment from user-provided vars + EnvironmentConfig
        env_config = self.task.request.environment
        env = dict(env_config.env_vars)

        # Build iris system environment based on runtime type
        iris_env = build_iris_env(
            self.task,
            self._worker_id,
            self._controller_address,
            type(self._runtime),
        )
        env.update(iris_env)

        # Convert proto entrypoint to typed Entrypoint
        entrypoint = Entrypoint.from_proto(self.task.request.entrypoint)

        config = ContainerConfig(
            image=self.task.image_tag,
            entrypoint=entrypoint,
            env=env,
            resources=self.task.request.resources if self.task.request.HasField("resources") else None,
            timeout_seconds=self.task.request.timeout_seconds or None,
            ports=self.task.ports,
            mounts=[(str(self.task.workdir), "/workdir", "rw")],
            task_id=self.task.task_id,
            job_id=self.task.job_id,
            worker_metadata=self._worker_metadata,
        )

        # Create and start container with retry on port binding failures
        container_id = None
        max_port_retries = 3
        for attempt in range(max_port_retries):
            try:
                chaos_raise("worker.create_container")
                container_id = self._runtime.create_container(config)
                self.task.container_id = container_id
                self._runtime.start_container(container_id)
                break
            except RuntimeError as e:
                if "address already in use" in str(e) and attempt < max_port_retries - 1:
                    logger.warning(
                        "Port conflict for task %s, retrying with new ports (attempt %d)",
                        self.task.task_id,
                        attempt + 2,
                    )
                    self.task.logs.add("build", f"Port conflict, retrying with new ports (attempt {attempt + 2})")
                    # Release current ports and allocate new ones
                    self._port_allocator.release(list(self.task.ports.values()))
                    port_names = list(self.task.ports.keys())
                    new_ports = self._port_allocator.allocate(len(port_names))
                    self.task.ports = dict(zip(port_names, new_ports, strict=True))

                    # Update config with new ports
                    config.ports = self.task.ports
                    for name, port in self.task.ports.items():
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
        """Monitor task execution: check status, collect stats, handle timeouts.

        Polls container status at regular intervals until the container stops.
        Collects runtime statistics (CPU, memory, disk) and handles timeout enforcement.
        Updates task state to terminal status (SUCCEEDED/FAILED/KILLED) when container stops.

        Args:
            container_id: Container to monitor
        """
        timeout_seconds = self.task.request.timeout_seconds or None
        start_time = time.time()

        while True:
            if chaos("worker.task_monitor"):
                self.task.transition_to(cluster_pb2.TASK_STATE_FAILED, error="chaos: monitor crashed")
                break

            # Check if we should stop
            if self.task.should_stop:
                self.task.transition_to(cluster_pb2.TASK_STATE_KILLED)
                break

            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                self._runtime.kill(container_id, force=True)
                self.task.transition_to(
                    cluster_pb2.TASK_STATE_FAILED,
                    error="Timeout exceeded",
                    exit_code=-1,
                )
                break

            # Check container status
            status = self._runtime.inspect(container_id)
            if not status.running:
                # Read result file only if container succeeded
                if status.exit_code == 0 and self.task.workdir:
                    result_path = self.task.workdir / "_result.pkl"
                    if result_path.exists():
                        try:
                            self.task.result = result_path.read_bytes()
                        except Exception as e:
                            self.task.logs.add("error", f"Failed to read result file: {e}")

                # Container has stopped
                if status.error:
                    self.task.transition_to(
                        cluster_pb2.TASK_STATE_FAILED,
                        error=status.error,
                        exit_code=status.exit_code or -1,
                    )
                elif status.exit_code == 0:
                    self.task.transition_to(cluster_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
                else:
                    self.task.transition_to(
                        cluster_pb2.TASK_STATE_FAILED,
                        error=f"Exit code: {status.exit_code}",
                        exit_code=status.exit_code or -1,
                    )
                break

            # Collect stats
            try:
                stats = self._runtime.get_stats(container_id)
                if stats.available:
                    self.task.current_memory_mb = stats.memory_mb
                    self.task.current_cpu_percent = stats.cpu_percent
                    self.task.process_count = stats.process_count
                    if stats.memory_mb > self.task.peak_memory_mb:
                        self.task.peak_memory_mb = stats.memory_mb

                if self.task.workdir:
                    self.task.disk_mb = collect_workdir_size_mb(self.task.workdir)
            except Exception:
                pass  # Don't fail task on stats collection errors

            # Sleep before next poll
            time.sleep(self._poll_interval_seconds)

    def _cleanup(self) -> None:
        """Clean up task resources: ports, image protection, workdir.

        Idempotent - safe to call multiple times. Logs errors instead of
        silently swallowing them (RF-5 fix).
        """
        if self.task.cleanup_done:
            return
        self.task.cleanup_done = True

        # Release ports
        try:
            self._port_allocator.release(list(self.task.ports.values()))
        except Exception as e:
            logger.warning("Failed to release ports for task %s: %s", self.task.task_id, e)

        # Unprotect image from eviction now that task is done
        if self._image_provider and self.task.image_tag:
            try:
                self._image_provider.unprotect(self.task.image_tag)
            except Exception as e:
                logger.warning("Failed to unprotect image %s: %s", self.task.image_tag, e)

        # Remove working directory
        # Keep container around for log retrieval via docker logs
        if self.task.workdir and self.task.workdir.exists():
            try:
                shutil.rmtree(self.task.workdir)
            except Exception as e:
                logger.warning("Failed to remove workdir %s: %s", self.task.workdir, e)
