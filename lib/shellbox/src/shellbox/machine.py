# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine contract shared by local and remote task runtimes."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import ClassVar, Protocol, runtime_checkable

from shellbox.image import DockerfileSource, PreparedImage, RegistryImage

HARBOR_EXEC_OUTPUT_LIMIT_BYTES = 128 * 1024 * 1024
DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES = 1_048_576


class NetworkPolicy(StrEnum):
    DENY = "deny"
    ALLOW = "allow"


class Backend(StrEnum):
    DOCKER = "docker"
    GVISOR = "gvisor"
    QEMU = "qemu"
    SHELLSIM = "shellsim"
    DAYTONA = "daytona"
    LOCAL = "local"


class CommandSemantics(StrEnum):
    """Execution behavior required by a task, independent of deployment backend."""

    LINUX_PROCESS = "linux_process"
    SHELL_SIMULATOR = "shell_simulator"


SHELL_SIMULATOR_CAPABILITIES = frozenset({"shell", "filesystem", "python"})
LINUX_PROCESS_CAPABILITIES = SHELL_SIMULATOR_CAPABILITIES | {"process", "network", "git_repository"}


class ExecutionRequirements(Protocol):
    """The task contract factories inspect without importing a task schema."""

    @property
    def command_semantics(self) -> CommandSemantics | None: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...

    @property
    def docker_image(self) -> str | None: ...


class ExitReason(StrEnum):
    EXITED = "exited"
    TIMED_OUT = "timed_out"


class ShellStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    RESET = "reset"


@dataclass(frozen=True)
class ShellUpdate:
    output: bytes
    status: ShellStatus
    exit_code: int | None = None
    truncated: bool = False


class ShellSession(Protocol):
    """One persistent shell owned by an agent trial."""

    @property
    def interactive(self) -> bool: ...

    async def execute(
        self, command: str, *, wait: float = 120, output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
    ) -> ShellUpdate: ...

    async def read(
        self, *, wait: float = 0, output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
    ) -> ShellUpdate: ...

    async def write(self, data: bytes) -> None: ...

    async def interrupt(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class BashSessionProvider(Protocol):
    """An environment that can open an agent-owned persistent Bash session."""

    async def open_bash_session(self) -> ShellSession: ...


@dataclass(frozen=True)
class QemuBundle:
    """A staged runtime directory and its loaded registry origin, when present."""

    path: Path
    image_reference: str | None = None


@dataclass(frozen=True)
class DockerImage:
    reference: str


@dataclass(frozen=True)
class PackageSpec:
    """Python requirements and tools resolved by the selected execution backend."""

    __pydantic_config__: ClassVar[dict[str, str]] = {"extra": "forbid"}

    pypi_packages: tuple[str, ...] | None = None
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mount:
    """A host file or directory that commands see read-only at ``guest``."""

    host: Path
    guest: PurePosixPath


@dataclass(frozen=True)
class HostImage:
    """The host's programs, with optional per-machine read-only mounts and program paths.

    ``read_only`` directories appear at their own host paths; each ``Mount`` appears at its guest path.
    """

    read_only: tuple[Path, ...] = ()
    bin_dirs: tuple[Path, ...] = ()
    mounts: tuple[Mount, ...] = ()


@dataclass(frozen=True)
class MachineSpec:
    """Machine inputs, with a provider startup timeout for Daytona.

    Other factories do not apply startup_timeout. Callers enforce their own
    deadline for the complete create operation.
    """

    source: QemuBundle | DockerImage | PreparedImage | RegistryImage | DockerfileSource | PackageSpec | HostImage
    workdir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    network: NetworkPolicy = NetworkPolicy.DENY
    memory_mb: int | None = None
    cpus: int | None = None
    storage_mb: int | None = None
    gpus: int = 0
    startup_timeout: float | None = None

    def __post_init__(self) -> None:
        if self.cpus is not None and self.cpus <= 0:
            raise ValueError("cpus must be positive")
        if self.storage_mb is not None and self.storage_mb <= 0:
            raise ValueError("storage_mb must be positive")


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    stdin: bytes = b""
    timeout: float | None = None
    output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
    user: str | None = None


@dataclass(frozen=True)
class Result:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    reason: ExitReason


class UnsupportedMachineSpec(ValueError):
    """The selected backend cannot create the requested machine."""


def check_execution_requirements(
    requirements: ExecutionRequirements,
    spec: MachineSpec,
    *,
    semantics: CommandSemantics,
    capabilities: frozenset[str],
    image_reference: str | None = None,
) -> None:
    """Check declared execution support and image origin, without acquiring resources.

    Backends check their source and allocation features separately. This does not
    promise installed executables, available capacity or successful acquisition.
    """
    if requirements.command_semantics != semantics:
        raise UnsupportedMachineSpec(f"The selected factory requires {semantics} semantics")
    unsupported = set(requirements.capabilities) - capabilities
    if unsupported:
        raise UnsupportedMachineSpec(f"Unsupported capabilities: {', '.join(sorted(unsupported))}")
    if "network" in requirements.capabilities and spec.network == NetworkPolicy.DENY:
        raise UnsupportedMachineSpec("Network capability requires an allow network policy")
    if requirements.docker_image is not None and requirements.docker_image != image_reference:
        raise UnsupportedMachineSpec("Machine must use the task's required image")


class MachineTerminated(RuntimeError):
    """The machine ended before ``close``: killed, expired, preempted, or lost with its host."""


class CommandStartTimeout(RuntimeError):
    """The machine could not start a command in time, an infrastructure failure; ``Command.timeout`` had not begun."""


class Machine(Protocol):
    """One writable task environment. Files persist until close."""

    async def run(self, command: Command) -> Result: ...

    async def upload(self, source: Path, target: str) -> None: ...

    async def download(self, source: str, target: Path) -> None: ...

    async def close(self) -> None: ...


class MachineFactory(Protocol):
    """Check execution support and create a fresh machine from declared software."""

    @property
    def backend(self) -> Backend: ...

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        """Reject unsupported task contracts before preparation or acquisition.

        Checks declared features, not available capacity or installed executables.
        """
        ...

    async def create(self, spec: MachineSpec) -> Machine: ...
