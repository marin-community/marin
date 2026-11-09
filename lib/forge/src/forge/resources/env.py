# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-writer environments with replayable command and binary-file effects."""

import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import ClassVar, Protocol, override

from rolloutengine.machines import prepare_machine
from rolloutengine.models import MachineRuntimeSpec
from shellbox.machine import Command, ExitReason, Machine, MachineFactory, MachineTerminated
from taskcompendium.models import EnvironmentRequirements

from forge.effects.events import Event, Fingerprint, Note, ReplayPolicy, Resolved, canonical_digest
from forge.effects.issues import Hold, Restart, Transient
from forge.resources.execution import Notify, retry_effect


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    wall: float
    truncated: bool


@dataclass(frozen=True, slots=True)
class RunCommand(Event[CommandResult]):
    result_type: ClassVar[object] = CommandResult
    cmd: str
    timeout: float


@dataclass(frozen=True, slots=True)
class WriteFile(Event[None]):
    result_type: ClassVar[object] = type(None)
    path: PurePosixPath
    data: bytes


@dataclass(frozen=True, slots=True)
class ReadFile(Event[bytes]):
    result_type: ClassVar[object] = bytes
    path: PurePosixPath


type Mutation = Resolved[CommandResult] | Resolved[None]


class Env(Protocol):
    def fingerprint(self) -> Fingerprint: ...
    async def run(self, cmd: str, *, timeout: float) -> CommandResult: ...
    async def read(self, path: PurePosixPath) -> bytes: ...
    async def write(self, path: PurePosixPath, data: bytes) -> None: ...
    async def restore(self, events: Sequence[Mutation]) -> None:
        """Replay mutations, excluding ReadFile; mismatch Notes leave the log authoritative.

        A fingerprint Hold requires human inspection of the inconsistent log and environment.
        """
        ...

    async def close(self) -> None: ...


class EnvFactory(Protocol):
    async def create(self, requirements: EnvironmentRequirements) -> Env: ...


def _stable(result: CommandResult) -> tuple[str, ...]:
    return str(result.exit_code), result.stdout, result.stderr


class _RecordedEnv(Env):
    def __init__(self, image: str, notify: Notify) -> None:
        self.digest = canonical_digest((image,))
        self.notify = notify
        self.closed = False

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("env", self.digest, ReplayPolicy.REEXECUTE)

    def _record_command(self, cmd: str, timeout: float, result: CommandResult) -> None:
        self.digest = canonical_digest((self.digest, "command", cmd, float(timeout), *_stable(result)))

    def _record_write(self, path: PurePosixPath, data: bytes) -> None:
        self.digest = canonical_digest((self.digest, "write", str(path), canonical_digest(data)))

    @override
    async def restore(self, events: Sequence[Mutation]) -> None:
        for index, resolved in enumerate(events):
            current = self.fingerprint()
            expected = next((fp for fp in resolved.fingerprints if fp.resource == "env"), None)
            if expected != current:
                raise Hold(f"Restore step {index}: expected {expected}, found {current.digest}")
            request = resolved.request
            if isinstance(request, WriteFile):
                await self.write(request.path, request.data)
            elif isinstance(request, RunCommand) and isinstance(resolved.value, CommandResult):
                result = await self.run(request.cmd, timeout=request.timeout)
                if _stable(result) != _stable(resolved.value):
                    self.notify(Note(f"Restore step {index}: command result differs from the log"))
                    # Keep the logical replay position despite the explicitly reported divergence.
                    self.digest = current.digest
                    self._record_command(request.cmd, request.timeout, resolved.value)
            else:
                raise Hold(f"Restore step {index} is not a resolved environment mutation")

    def _check_open(self) -> None:
        if self.closed:
            raise Restart("Environment is closed")

    @override
    async def close(self) -> None:
        self.closed = True


class ShellboxEnv(_RecordedEnv):
    """An owned Shellbox machine; nonzero command exits remain ordinary results."""

    def __init__(self, machine: Machine, image: str, notify: Notify, owned: AsyncExitStack) -> None:
        self.machine, self.owned = machine, owned
        super().__init__(image, notify)

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        self._check_open()
        try:
            yield
        except MachineTerminated as error:
            raise Restart("Environment machine terminated") from error
        except RuntimeError as error:
            if str(error) != "Machine is closed":
                raise
            raise Restart("Environment machine is closed") from error
        except (TimeoutError, ConnectionError) as error:
            raise Transient("Environment operation did not resolve") from error

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        started = time.monotonic()
        async with self._operation():
            result = await self.machine.run(Command(("sh", "-c", cmd), timeout=timeout))
        code = 124 if result.reason == ExitReason.TIMED_OUT else result.exit_code
        if code is None:
            raise Restart("Environment command ended without an exit status")
        value = CommandResult(
            code,
            result.stdout.decode(errors="replace"),
            result.stderr.decode(errors="replace"),
            time.monotonic() - started,
            result.stdout_truncated or result.stderr_truncated,
        )
        self._record_command(cmd, timeout, value)
        return value

    @override
    async def read(self, path: PurePosixPath) -> bytes:
        async with self._operation():
            with TemporaryDirectory() as directory:
                target = Path(directory) / "file"
                await self.machine.download(str(path), target)
                return target.read_bytes()

    @override
    async def write(self, path: PurePosixPath, data: bytes) -> None:
        async with self._operation():
            with TemporaryDirectory() as directory:
                source = Path(directory) / "file"
                source.write_bytes(data)
                await self.machine.upload(source, str(path))
        self._record_write(path, data)

    @override
    async def close(self) -> None:
        await super().close()
        await self.owned.aclose()


@dataclass(frozen=True, slots=True)
class ShellboxEnvFactory(EnvFactory):
    """Use local or ShellSim factories with RolloutEngine's setup and cleanup lifecycle."""

    factory: MachineFactory
    runtime: MachineRuntimeSpec
    cleanup_timeout: float
    notify: Notify

    @override
    async def create(self, requirements: EnvironmentRequirements) -> ShellboxEnv:
        owned = AsyncExitStack()
        try:
            machine = await owned.enter_async_context(
                prepare_machine(requirements, (), self.factory, self.runtime, cleanup_timeout=self.cleanup_timeout)
            )
        except MachineTerminated as error:
            raise Restart("Environment machine terminated during startup") from error
        except (TimeoutError, ConnectionError) as error:
            raise Transient("Environment startup did not resolve") from error
        image = canonical_digest((requirements, self.runtime))
        return ShellboxEnv(machine, image, self.notify, owned)


@dataclass(frozen=True, slots=True)
class FakeCommand:
    result: CommandResult
    writes: Mapping[PurePosixPath, bytes]


class FakeEnv(_RecordedEnv):
    """An in-memory filesystem and explicitly scripted shell effects, without a shell parser."""

    def __init__(self, image: str, commands: Mapping[str, FakeCommand | Transient | Restart], notify: Notify) -> None:
        super().__init__(image, notify)
        self.commands = commands
        self.files: dict[PurePosixPath, bytes] = {}

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        self._check_open()
        scripted = self.commands.get(cmd)
        if scripted is None:
            raise ValueError(f"Unscripted fake command: {cmd}")
        if isinstance(scripted, Transient | Restart):
            raise scripted
        self.files.update(scripted.writes)
        self._record_command(cmd, timeout, scripted.result)
        return scripted.result

    @override
    async def read(self, path: PurePosixPath) -> bytes:
        self._check_open()
        return self.files[path]

    @override
    async def write(self, path: PurePosixPath, data: bytes) -> None:
        self._check_open()
        self.files[path] = data
        self._record_write(path, data)


async def run_command(event: RunCommand, env: Env, *, notify: Notify) -> CommandResult:
    return await retry_effect(event, lambda: env.run(event.cmd, timeout=event.timeout), notify)


async def write_file(event: WriteFile, env: Env, *, notify: Notify) -> None:
    await retry_effect(event, lambda: env.write(event.path, event.data), notify)


async def read_file(event: ReadFile, env: Env, *, notify: Notify) -> bytes:
    return await retry_effect(event, lambda: env.read(event.path), notify)
