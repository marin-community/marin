# Copyright The Marin Authors
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
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from iris.cluster.bundle import BundleStore
from iris.cluster.worker.worker_types import LogLine, TaskLogs
from iris.rpc import job_pb2


class ContainerInfraError(RuntimeError):
    """Container operation failed due to infrastructure issues (expired credentials,
    unreachable registry, docker daemon problems). Uses preemption retry budget."""

    pass


class ContainerErrorKind(StrEnum):
    """Structured category for container/runtime errors."""

    NONE = "none"
    USER_CODE = "user_code"
    INFRA_NOT_FOUND = "infra_not_found"
    RUNTIME_ERROR = "runtime_error"


class ContainerPhase(StrEnum):
    """Lifecycle phase of a container from the runtime's perspective.

    PENDING: container created but not yet executing (K8s pod scheduling, image pull).
    RUNNING: container is executing the main command.
    STOPPED: container has exited (check exit_code/error for details).
    """

    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"


class ExecutionStage(StrEnum):
    """Which pipeline stage a container belongs to (for adoption filtering)."""

    BUILD = "build"
    RUN = "run"


class MountKind(StrEnum):
    WORKDIR = "workdir"  # task working directory (/app); tmpfs on Docker, emptyDir on K8s
    TMPFS = "tmpfs"  # volatile fast storage; tmpfs on Docker, emptyDir on K8s
    CACHE = "cache"  # persistent cross-task cache (uv, cargo); hostPath bind mount


@dataclass(frozen=True)
class MountSpec:
    container_path: str
    kind: MountKind = MountKind.CACHE
    read_only: bool = False
    size_bytes: int = 0  # 0 = no limit; tmpfs size / emptyDir sizeLimit


@dataclass
class ContainerConfig:
    """Configuration for running a container."""

    image: str
    entrypoint: job_pb2.RuntimeEntrypoint
    env: dict[str, str]
    workdir: str = "/app"
    resources: job_pb2.ResourceSpecProto | None = None
    timeout_seconds: int | None = None
    mounts: list[MountSpec] = field(default_factory=list)
    network_mode: str = "host"  # e.g. "host" for --network=host
    workdir_host_path: Path | None = None
    task_id: str | None = None
    attempt_id: int | None = None
    job_id: str | None = None
    worker_id: str | None = None
    worker_metadata: job_pb2.WorkerMetadata | None = None

    def get_cpu_millicores(self) -> int | None:
        if not self.resources or not self.resources.cpu_millicores:
            return None
        return self.resources.cpu_millicores

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
    cpu_millicores: int
    process_count: int
    available: bool


@dataclass
class ContainerStatus:
    """Container state from runtime inspection."""

    phase: ContainerPhase
    exit_code: int | None = None
    error: str | None = None
    error_kind: ContainerErrorKind = ContainerErrorKind.NONE
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
        """Return the platform container identifier.

        Docker: the container ID (hash from docker create).
        K8s: the pod name.
        Process: a local-<uuid> string.
        """
        ...

    def build(self, on_logs: Callable[[list[LogLine]], None] | None = None) -> list[LogLine]:
        """Run setup_commands (uv sync, pip install, etc).

        Blocks until setup completes. If there are no setup_commands,
        this is a no-op.

        Args:
            on_logs: Optional callback invoked with each incremental batch of
                log lines as they arrive. Enables streaming logs to callers
                during long builds.

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

    def disk_usage_mb(self) -> int:
        """Return disk usage in MB for this container's workdir.

        Docker/Process: shutil.disk_usage on the host workdir path.
        K8s: 0 (workdir lives inside the pod, not on the worker node).
        """
        ...

    def profile(self, duration_seconds: int, profile_type: job_pb2.ProfileType) -> bytes:
        """Profile the running process using py-spy (CPU), memray (memory), or thread dump.

        Args:
            duration_seconds: How long to sample (ignored for threads)
            profile_type: ProfileType message with oneof cpu/memory/threads profiler config

        Returns:
            Raw profile output (SVG/HTML/JSON/text depending on profiler and format)

        Raises:
            RuntimeError: If profiling fails or container is not running
        """
        ...

    def cleanup(self) -> None:
        """Remove the container and clean up resources."""
        ...


@dataclass(frozen=True)
class DiscoveredContainer:
    """Metadata for a container discovered on the host after a worker restart.

    Extracted from Docker labels + inspect. Provides enough information for
    a new worker process to adopt the container and resume monitoring.
    """

    container_id: str
    task_id: str
    attempt_id: int
    job_id: str
    worker_id: str
    phase: ExecutionStage
    running: bool
    exit_code: int | None
    started_at: str  # ISO 8601 timestamp from Docker
    workdir_host_path: str  # host path of the /app mount


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

    def prepare_workdir(self, workdir: Path, disk_bytes: int) -> None:
        """Prepare the task workdir before bundle staging.

        Docker: mounts a per-task tmpfs for quota enforcement.
        Process/K8s: no-op.
        """
        ...

    def stage_bundle(
        self,
        *,
        bundle_id: str,
        workdir: Path,
        workdir_files: dict[str, bytes],
        bundle_store: BundleStore,
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

    def list_iris_containers(self, all_states: bool = True) -> list[str]:
        """List IDs of all iris-managed containers/sandboxes."""
        ...

    def remove_all_iris_containers(self) -> int:
        """Force remove all iris-managed containers/sandboxes. Returns count removed."""
        ...

    def remove_containers(self, container_ids: list[str]) -> int:
        """Force remove specific containers by ID. Returns count removed."""
        ...

    def discover_containers(self) -> list[DiscoveredContainer]:
        """Discover iris-managed containers from a previous worker process.

        Returns metadata for containers that can potentially be adopted.
        Used during worker restart to find running containers that should
        be monitored instead of killed.
        """
        ...

    def adopt_container(self, container_id: str) -> ContainerHandle:
        """Create a handle wrapping an existing container for adoption.

        Returns a handle that supports status(), stop(), log_reader(), stats(),
        and cleanup(). build()/run() should not be called on adopted handles.
        """
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
