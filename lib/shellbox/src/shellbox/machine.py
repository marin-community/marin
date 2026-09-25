# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine contract shared by local and remote task runtimes."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from shellbox.image import DockerfileSource, PreparedImage, RegistryImage

HARBOR_EXEC_OUTPUT_LIMIT_BYTES = 128 * 1024 * 1024
DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES = 1_048_576


class NetworkPolicy(StrEnum):
    DENY = "deny"
    ALLOW = "allow"


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
    path: Path


@dataclass(frozen=True)
class DockerImage:
    reference: str


@dataclass(frozen=True)
class ShellSimBuiltins:
    """ShellSim's built-in commands and virtual filesystem, without an OCI image."""


@dataclass(frozen=True)
class MachineSpec:
    source: QemuBundle | DockerImage | PreparedImage | RegistryImage | DockerfileSource | ShellSimBuiltins
    workdir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    network: NetworkPolicy = NetworkPolicy.DENY
    memory_mb: int | None = None


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    stdin: bytes = b""
    timeout: float | None = None
    output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES


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


class Machine(Protocol):
    """One writable task environment. Files persist until close."""

    async def run(self, command: Command) -> Result: ...

    async def upload(self, source: Path, target: str) -> None: ...

    async def download(self, source: str, target: Path) -> None: ...

    async def close(self) -> None: ...


class MachineFactory(Protocol):
    """Create a fresh machine from an image source."""

    async def create(self, spec: MachineSpec) -> Machine: ...
