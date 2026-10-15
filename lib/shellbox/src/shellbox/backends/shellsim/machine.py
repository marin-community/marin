# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ShellSim implementation of the machine and persistent shell contracts."""

import asyncio
import io
import re
import shlex
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import shellsim

from shellbox.machine import (
    DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    SHELL_SIMULATOR_CAPABILITIES,
    Backend,
    Command,
    CommandSemantics,
    ExecutionRequirements,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    PackageSpec,
    Result,
    ShellStatus,
    ShellUpdate,
    UnsupportedMachineSpec,
    check_execution_requirements,
)

DEFAULT_CPU_LIMIT = 10_000_000_000
DEFAULT_DISK_LIMIT = 512 * 1024 * 1024
DEFAULT_OUTPUT_LIMIT = 128 * 1024 * 1024
DEFAULT_MEMORY_MB = 2048
SIMULATOR_PANIC_EXIT_CODE = 1
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


async def _await_worker[T](worker: asyncio.Task[T], cancellation: shellsim.CancellationToken | None = None) -> T:
    """Keep the machine lock until native work exits, including during caller cancellation."""
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        if cancellation is not None:
            cancellation.cancel()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        worker.result()
        raise


@dataclass(frozen=True)
class _RunningAction:
    cancellation: shellsim.CancellationToken
    worker: asyncio.Task[shellsim.RunResult]
    shell: shellsim.ShellSession | None


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
            result = await asyncio.wait_for(self.machine._run_source(command, shell=self.shell), timeout=wait)
        except TimeoutError:
            output = b"ShellSim command timed out\n"
            return ShellUpdate(output[:output_limit_bytes], ShellStatus.COMPLETED, 130, len(output) > output_limit_bytes)
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
        action = self.machine._active
        if action is not None and action.shell is self.shell:
            action.cancellation.cancel()
            await _await_worker(action.worker, action.cancellation)

    async def close(self) -> None:
        await self.interrupt()


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
        self._active: _RunningAction | None = None
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
        async with self._lock:
            if self._closed:
                raise RuntimeError("Machine is closed")
            return await self._run_source_locked(source, stdin, shell=shell)

    async def _run_source_locked(
        self, source: str, stdin: bytes = b"", *, shell: shellsim.ShellSession | None = None
    ) -> shellsim.RunResult:
        """Execute with the caller holding the machine lock through native cleanup."""
        runner = self.simulation.run if shell is None else shell.run
        cancellation = shellsim.CancellationToken()
        worker = asyncio.create_task(asyncio.to_thread(runner, source, stdin, cancellation=cancellation))
        self._active = _RunningAction(cancellation, worker, shell)
        try:
            result = await _await_worker(worker, cancellation)
            if self.simulation.terminated:
                raise RuntimeError(f"ShellSim terminated: {result.stop_reason}")
            return result
        finally:
            self._active = None

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
        async with self._lock:
            if self._closed:
                raise RuntimeError("Machine is closed")
            if source.is_dir():
                await _await_worker(asyncio.create_task(asyncio.to_thread(self.simulation.mount, source, target)))
                return
            self.simulation.mkdir(str(PurePosixPath(target).parent), parents=True)
            self.simulation.write_file(target, source.read_bytes(), mode=source.stat().st_mode & 0o7777)

    async def download(self, source: str, target: Path) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Machine is closed")
            if target.exists() and target.is_dir():
                archive = f"/__harbor_download_{uuid.uuid4().hex}.tar"
                command = f"tar -cf {shlex.quote(archive)} -C {shlex.quote(source)} ."
                result = await self._run_source_locked(command)
                result.check_returncode()
                payload = self.simulation.read_file(archive)
                await self._run_source_locked(f"rm -f {shlex.quote(archive)}")
                with tarfile.open(fileobj=io.BytesIO(payload)) as contents:
                    contents.extractall(target, filter="data")
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.simulation.read_file(source))

    async def close(self) -> None:
        self._closed = True
        if self._active is not None:
            self._active.cancellation.cancel()
        async with self._lock:
            pass


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
        self.context = context if context is not None else shellsim.Context(trusted_precompiled=True)
        self.cpu = cpu
        self.disk = disk
        self.output = output

    def _check_spec(self, spec: MachineSpec) -> PackageSpec:
        if spec.cpus is not None or spec.gpus:
            raise UnsupportedMachineSpec("ShellSim does not provide host CPU or GPU allocations")
        if not isinstance(spec.source, PackageSpec):
            raise UnsupportedMachineSpec("ShellSim requires a PackageSpec source")
        if spec.network is not NetworkPolicy.DENY:
            raise UnsupportedMachineSpec("ShellSim has no guest network")
        return spec.source

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self._check_spec(spec)
        check_execution_requirements(
            requirements,
            spec,
            semantics=CommandSemantics.SHELL_SIMULATOR,
            capabilities=SHELL_SIMULATOR_CAPABILITIES,
            image_reference=None,
        )

    async def create(self, spec: MachineSpec) -> ShellSimMachine:
        source = self._check_spec(spec)
        return await asyncio.to_thread(
            ShellSimMachine,
            spec,
            source=source,
            context=self.context,
            cpu=self.cpu,
            disk=self.disk,
            output=self.output,
        )
