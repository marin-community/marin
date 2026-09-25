# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ShellSim implementation of the machine and persistent shell contracts."""

import asyncio
import io
import re
import shlex
import tarfile
import uuid
from pathlib import Path, PurePosixPath

import shellsim

from shellbox.machine import (
    DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    Command,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    Result,
    ShellSimBuiltins,
    ShellStatus,
    ShellUpdate,
    UnsupportedMachineSpec,
)

DEFAULT_CPU_LIMIT = 10_000_000_000
DEFAULT_DISK_LIMIT = 256 * 1024 * 1024
DEFAULT_OUTPUT_LIMIT = 128 * 1024 * 1024
DEFAULT_MEMORY_MB = 256
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ShellSimShellSession:
    """One persistent simulated shell with complete, synchronous commands."""

    interactive = False

    def __init__(self, machine: "ShellSimMachine"):
        self.machine = machine

    async def execute(
        self, command: str, *, wait: float = 120, output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
    ) -> ShellUpdate:
        result = await self.machine._run_source(command)
        output = result.stdout + result.stderr
        return ShellUpdate(
            output[:output_limit_bytes],
            ShellStatus.COMPLETED,
            result.returncode,
            len(output) > output_limit_bytes,
        )

    async def read(
        self, *, wait: float = 0, output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
    ) -> ShellUpdate:
        raise UnsupportedMachineSpec("ShellSim completes each Bash action; no command remains to read")

    async def write(self, data: bytes) -> None:
        raise UnsupportedMachineSpec("ShellSim has no interactive stdin")

    async def interrupt(self) -> None:
        raise UnsupportedMachineSpec("ShellSim has no live foreground action to interrupt")

    async def close(self) -> None:
        return


class ShellSimMachine:
    """One in-memory simulated machine and its private virtual filesystem."""

    def __init__(self, spec: MachineSpec, *, cpu: int, disk: int, output: int):
        self.spec = spec
        self.simulation = shellsim.Environment(
            cpu=cpu,
            memory=(spec.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024,
            disk=disk,
            output=output,
        )
        self._lock = asyncio.Lock()
        self._closed = False
        self._shell = ShellSimShellSession(self)
        self.simulation.mkdir(spec.workdir, parents=True)
        for log_dir in ("/logs/agent", "/logs/verifier", "/logs/artifacts"):
            self.simulation.mkdir(log_dir, parents=True)
        for name in spec.env:
            if ENV_NAME.fullmatch(name) is None:
                raise ValueError(f"Invalid environment variable name: {name}")
        environment = " ".join(f"export {name}={shlex.quote(value)};" for name, value in spec.env.items())
        initial = self.simulation.run(f"cd {shlex.quote(spec.workdir)}; {environment}")
        initial.check_returncode()

    async def _run_source(self, source: str, stdin: bytes = b"") -> shellsim.RunResult:
        if self._closed:
            raise RuntimeError("Machine is closed")
        async with self._lock:
            result = await asyncio.to_thread(self.simulation.run, source, stdin)
            if self.simulation.terminated:
                raise RuntimeError(f"ShellSim terminated: {result.stop_reason}")
            return result

    async def open_shell(self) -> ShellSimShellSession:
        if self._closed:
            raise RuntimeError("Machine is closed")
        return self._shell

    async def run(self, command: Command) -> Result:
        if not command.argv:
            raise ValueError("Command argv is empty")
        for name in command.env:
            if ENV_NAME.fullmatch(name) is None:
                raise ValueError(f"Invalid environment variable name: {name}")
        cwd = command.cwd or self.spec.workdir
        environment = " ".join(f"export {name}={shlex.quote(value)};" for name, value in command.env.items())
        source = f"sh -c {shlex.quote(f'cd {shlex.quote(cwd)}; {environment} {shlex.join(command.argv)}')}"
        try:
            result = await asyncio.wait_for(self._run_source(source, command.stdin), timeout=command.timeout)
        except TimeoutError:
            self._closed = True
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        limit = command.output_limit_bytes
        return Result(
            result.returncode,
            result.stdout[:limit],
            result.stderr[:limit],
            len(result.stdout) > limit,
            len(result.stderr) > limit,
            ExitReason.EXITED,
        )

    async def upload(self, source: Path, target: str) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        async with self._lock:
            if source.is_dir():
                await asyncio.to_thread(self.simulation.mount, source, target)
                return
            self.simulation.mkdir(str(PurePosixPath(target).parent), parents=True)
            self.simulation.write_file(target, source.read_bytes(), mode=source.stat().st_mode & 0o7777)

    async def download(self, source: str, target: Path) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        async with self._lock:
            if target.exists() and target.is_dir():
                archive = f"/__harbor_download_{uuid.uuid4().hex}.tar"
                command = f"tar -cf {shlex.quote(archive)} -C {shlex.quote(source)} ."
                result = await asyncio.to_thread(self.simulation.run, command)
                result.check_returncode()
                payload = self.simulation.read_file(archive)
                self.simulation.run(f"rm -f {shlex.quote(archive)}")
                with tarfile.open(fileobj=io.BytesIO(payload)) as contents:
                    contents.extractall(target, filter="data")
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.simulation.read_file(source))

    async def close(self) -> None:
        self._closed = True


class ShellSimMachineFactory:
    """Start a fresh ShellSim built-in environment for each trial."""

    def __init__(
        self,
        *,
        cpu: int = DEFAULT_CPU_LIMIT,
        disk: int = DEFAULT_DISK_LIMIT,
        output: int = DEFAULT_OUTPUT_LIMIT,
    ):
        self.cpu = cpu
        self.disk = disk
        self.output = output

    async def create(self, spec: MachineSpec) -> ShellSimMachine:
        if not isinstance(spec.source, ShellSimBuiltins):
            raise UnsupportedMachineSpec("ShellSim uses only its built-in commands; select ShellSimBuiltins")
        if spec.network is not NetworkPolicy.DENY:
            raise UnsupportedMachineSpec("ShellSim has no guest network")
        return ShellSimMachine(spec, cpu=self.cpu, disk=self.disk, output=self.output)
