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
    Backend,
    Command,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    PackageSpec,
    Result,
    ShellStatus,
    ShellUpdate,
    UnsupportedMachineSpec,
)

DEFAULT_CPU_LIMIT = 10_000_000_000
DEFAULT_DISK_LIMIT = 512 * 1024 * 1024
DEFAULT_OUTPUT_LIMIT = 128 * 1024 * 1024
DEFAULT_MEMORY_MB = 2048
SIMULATOR_PANIC_EXIT_CODE = 1
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ShellSimShellSession:
    """One persistent simulated shell with complete, synchronous commands."""

    interactive = False

    def __init__(self, machine: "ShellSimMachine"):
        self.machine = machine
        self.shell = machine.simulation.create_shell()

    async def execute(
        self, command: str, *, wait: float = 120, output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
    ) -> ShellUpdate:
        try:
            result = await self.machine._run_source(command, shell=self.shell)
        except shellsim.SimulationError as error:
            output = str(error).encode()
            return ShellUpdate(
                output[:output_limit_bytes],
                ShellStatus.COMPLETED,
                SIMULATOR_PANIC_EXIT_CODE,
                len(output) > output_limit_bytes,
            )
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

    def __init__(
        self, spec: MachineSpec, *, source: PackageSpec, context: shellsim.Context, cpu: int, disk: int, output: int
    ):
        self.spec = spec
        self.simulation = context.Environment(
            pypi_packages=source.pypi_packages,
            tools=source.tools,
            cpu=cpu,
            memory=(spec.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024,
            disk=spec.storage_mb * 1024 * 1024 if spec.storage_mb is not None else disk,
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
        initial = self._shell.shell.run(f"cd {shlex.quote(spec.workdir)}; {environment}")
        initial.check_returncode()

    async def _run_source(
        self, source: str, stdin: bytes = b"", *, shell: shellsim.ShellSession | None = None
    ) -> shellsim.RunResult:
        if self._closed:
            raise RuntimeError("Machine is closed")
        async with self._lock:
            runner = self.simulation.run if shell is None else shell.run
            result = await asyncio.to_thread(runner, source, stdin)
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
        if command.user not in (None, "0", "root"):
            raise UnsupportedMachineSpec("ShellSim does not provide separate execution users")
        for name in command.env:
            if ENV_NAME.fullmatch(name) is None:
                raise ValueError(f"Invalid environment variable name: {name}")
        cwd = command.cwd or self.spec.workdir
        environment = " ".join(
            f"export {name}={shlex.quote(value)};" for name, value in {**self.spec.env, **command.env}.items()
        )
        source = f"sh -c {shlex.quote(f'cd {shlex.quote(cwd)}; {environment} {shlex.join(command.argv)}')}"
        try:
            result = await asyncio.wait_for(self._run_source(source, command.stdin), timeout=command.timeout)
        except TimeoutError:
            self._closed = True
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except shellsim.SimulationError as error:
            stderr = str(error).encode()
            limit = command.output_limit_bytes
            return Result(
                SIMULATOR_PANIC_EXIT_CODE,
                b"",
                stderr[:limit],
                False,
                len(stderr) > limit,
                ExitReason.EXITED,
            )
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
    """Share installation policy and prepared software across fresh worker machines."""

    backend: Backend = Backend.SHELLSIM

    def __init__(
        self,
        *,
        context: shellsim.Context | None = None,
        cpu: int = DEFAULT_CPU_LIMIT,
        disk: int = DEFAULT_DISK_LIMIT,
        output: int = DEFAULT_OUTPUT_LIMIT,
    ):
        self.context = context if context is not None else shellsim.Context()
        self.cpu = cpu
        self.disk = disk
        self.output = output

    async def create(self, spec: MachineSpec) -> ShellSimMachine:
        if spec.cpus is not None or spec.gpus:
            raise UnsupportedMachineSpec("ShellSim does not provide host CPU or GPU allocations")
        if not isinstance(spec.source, PackageSpec):
            raise UnsupportedMachineSpec("ShellSim requires a PackageSpec source")
        if spec.network is not NetworkPolicy.DENY:
            raise UnsupportedMachineSpec("ShellSim has no guest network")
        return await asyncio.to_thread(
            ShellSimMachine,
            spec,
            source=spec.source,
            context=self.context,
            cpu=self.cpu,
            disk=self.disk,
            output=self.output,
        )
