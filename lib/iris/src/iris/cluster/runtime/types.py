# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Container runtime protocols and data types.

Defines the ContainerHandle, ContainerRuntime and ImageBuilder protocols that all
runtime implementations (Docker, process) must satisfy, plus shared data types
for container configuration, status, and statistics.

ContainerHandle provides a two-phase execution model:
- build(): Run setup commands (uv sync, pip install) synchronously
- run(): Start the main command (non-blocking)

This separation enables scheduler back-pressure: the controller can limit how many
tasks are in the BUILDING state per worker, preventing resource exhaustion from
too many concurrent uv sync operations.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from iris.cluster.worker.worker_types import LogLine, TaskLogs
from iris.rpc import cluster_pb2


@dataclass
class ContainerConfig:
    """Configuration for running a container."""

    image: str
    entrypoint: cluster_pb2.RuntimeEntrypoint
    env: dict[str, str]
    workdir: str = "/app"
    resources: cluster_pb2.ResourceSpecProto | None = None
    timeout_seconds: int | None = None
    mounts: list[tuple[str, str, str]] = field(default_factory=list)  # (host, container, mode)
    network_mode: str = "host"  # e.g. "host" for --network=host
    task_id: str | None = None
    job_id: str | None = None
    worker_metadata: cluster_pb2.WorkerMetadata | None = None

    def get_cpu_millicores(self) -> int | None:
        if not self.resources or not self.resources.cpu:
            return None
        return self.resources.cpu * 1000

    def get_memory_mb(self) -> int | None:
        if not self.resources or not self.resources.memory_bytes:
            return None
        return self.resources.memory_bytes // (1024 * 1024)

    def get_disk_bytes(self) -> int | None:
        if not self.resources or not self.resources.disk_bytes:
            return None
        return self.resources.disk_bytes


@dataclass
class ContainerResult:
    container_id: str
    exit_code: int
    started_at: float
    finished_at: float
    error: str | None = None


@dataclass
class ContainerStats:
    """Parsed container statistics."""

    memory_mb: int
    cpu_percent: int
    process_count: int
    available: bool


@dataclass
class ContainerStatus:
    """Container state from runtime inspection."""

    running: bool
    exit_code: int | None = None
    error: str | None = None
    oom_killed: bool = False


@dataclass
class ImageInfo:
    tag: str
    created_at: str


class RuntimeLogReader(Protocol):
    """Opaque incremental log reader created by a ContainerHandle.

    Each runtime owns the deduplication strategy (byte offsets, timestamps,
    list indices, etc.). Callers simply call read() in a loop to get new
    lines without duplicates.
    """

    def read(self) -> list[LogLine]:
        """Return new log lines since the last read. Advances the cursor."""
        ...

    def read_all(self) -> list[LogLine]:
        """Return all logs from the beginning (for error reporting)."""
        ...


class ContainerHandle(Protocol):
    """Handle for a logical container with build/run lifecycle.

    The runtime implementation decides whether this maps to one or multiple
    physical containers. From the worker's perspective, it's a single unit
    with a two-phase lifecycle:

    1. build() - Run setup_commands (uv sync, pip install, etc.)
       Called during BUILDING state. Creates .venv in workdir.
       Blocks until setup completes.

    2. run() - Start the main command
       Called during RUNNING state. Non-blocking - returns immediately
       after starting. Use status() to monitor.

    This separation enables scheduler back-pressure: limiting how many tasks
    are in BUILDING state per worker prevents resource exhaustion from
    concurrent uv sync operations.
    """

    @property
    def container_id(self) -> str | None:
        """Return the current container ID, if any."""
        ...

    def build(self) -> list[LogLine]:
        """Run setup_commands (uv sync, pip install, etc).

        Blocks until setup completes. If there are no setup_commands,
        this is a no-op.

        Returns:
            List of log lines captured during the build phase.

        Raises:
            RuntimeError: If setup fails (non-zero exit code)
        """
        ...

    def run(self) -> None:
        """Start the main command.

        Non-blocking - returns immediately after starting.
        Use status() to monitor execution progress.
        """
        ...

    def stop(self, force: bool = False) -> None:
        """Stop the container.

        Args:
            force: If True, use SIGKILL. Otherwise use SIGTERM.
        """
        ...

    def status(self) -> ContainerStatus:
        """Check container status (running, exit code, error)."""
        ...

    def log_reader(self) -> RuntimeLogReader:
        """Create an incremental log reader for this container."""
        ...

    def stats(self) -> ContainerStats:
        """Get resource usage statistics."""
        ...

    def profile(self, duration_seconds: int, profile_type: cluster_pb2.ProfileType) -> bytes:
        """Profile the running process using py-spy (CPU) or memray (memory).

        Args:
            duration_seconds: How long to sample
            profile_type: ProfileType message with oneof cpu/memory profiler config

        Returns:
            Raw profile output (SVG/HTML/JSON/text depending on profiler and format)

        Raises:
            RuntimeError: If profiling fails or container is not running
        """
        ...

    def cleanup(self) -> None:
        """Remove the container and clean up resources."""
        ...


class ContainerRuntime(Protocol):
    """Protocol for container runtimes (Docker, process, etc.).

    The runtime creates ContainerHandle instances which manage the full
    container lifecycle including build and run phases.
    """

    def create_container(self, config: ContainerConfig) -> ContainerHandle:
        """Create a container handle from config.

        The handle is not started - call handle.build() then handle.run()
        to execute the container.
        """
        ...

    def stage_bundle(
        self,
        *,
        bundle_gcs_path: str,
        workdir: Path,
        workdir_files: dict[str, bytes],
        fetch_bundle: Callable[[str], Path],
    ) -> None:
        """Materialize task bundle/workdir files for this runtime.

        Runtimes that execute from worker-local paths (docker/process)
        stage the bundle into ``workdir`` directly. Kubernetes runtime may no-op
        and materialize inside the task Pod instead.
        """
        ...

    def list_containers(self) -> list[ContainerHandle]:
        """List all managed containers."""
        ...

    def remove_all_iris_containers(self) -> int:
        """Force remove all iris-managed containers/sandboxes. Returns count removed."""
        ...

    def cleanup(self) -> None:
        """Clean up all containers managed by this runtime."""
        ...


class ImageBuilder(Protocol):
    """Protocol for image building (Docker build, rootfs creation, etc.)."""

    def build(
        self,
        dockerfile_content: str,
        tag: str,
        task_logs: TaskLogs | None = None,
        context: Path | None = None,
    ) -> None: ...

    def exists(self, tag: str) -> bool: ...

    def remove(self, tag: str) -> None: ...

    def list_images(self, pattern: str) -> list[ImageInfo]: ...
