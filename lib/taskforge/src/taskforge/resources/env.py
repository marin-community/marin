# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-writer environments with replayable command and binary-file effects."""

import errno
import os
import shlex
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Protocol, override
from uuid import uuid4

from rolloutengine.machines import prepare_machine
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.iris.machine import IrisMachine, never_connected
from shellbox.machine import (
    Command,
    CommandOutcomeUnknown,
    ExecSetupFailed,
    ExitReason,
    GuestHelperFailed,
    GuestPathError,
    Machine,
    MachineCancelled,
    MachineFactory,
    MachineOutputLimitExceeded,
    MachineResourceExhausted,
    MachineTerminated,
)
from shellsim import SimulationError
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.staging import StagedFile, write_archive

from taskforge.effects.events import Event, Fingerprint, ReplayPolicy, digest
from taskforge.effects.issues import Hold, OutputBudgetExhausted, Restart, Transient, describe_exception
from taskforge.resources.execution import Notify, retry_effect
from taskforge.resources.sandbox.factories import DEFAULT_WORKDIR


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    wall: float
    truncated: bool


@dataclass(frozen=True, slots=True)
class RunCommand(Event[CommandResult]):
    cmd: str
    timeout: float


@dataclass(frozen=True, slots=True)
class WriteFile(Event[None]):
    path: PurePosixPath
    data: bytes


@dataclass(frozen=True, slots=True)
class WriteFiles(Event[None]):
    files: tuple[tuple[PurePosixPath, bytes], ...]


@dataclass(frozen=True, slots=True)
class ReadFile(Event[bytes]):
    path: PurePosixPath
    offset: int = 0
    limit: int | None = None


class Env(Protocol):
    def fingerprint(self) -> Fingerprint: ...
    async def run(self, cmd: str, *, timeout: float) -> CommandResult: ...
    async def read(self, path: PurePosixPath, *, offset: int = 0, limit: int | None = None) -> bytes: ...
    async def write(self, path: PurePosixPath, data: bytes) -> None: ...
    async def write_many(self, files: Sequence[tuple[PurePosixPath, bytes]]) -> None: ...
    async def close(self) -> None: ...


class EnvFactory(Protocol):
    async def create(self, requirements: EnvironmentRequirements) -> Env: ...


def _stable(result: CommandResult) -> tuple[str, ...]:
    return str(result.exit_code), result.stdout, result.stderr


class _RecordedEnv(Env):
    def __init__(self, image: str, notify: Notify) -> None:
        self.digest = digest((image,))
        self.notify = notify
        self.closed = False

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("env", self.digest, ReplayPolicy.REEXECUTE)

    def _record_command(self, cmd: str, timeout: float, result: CommandResult) -> None:
        self.digest = digest((self.digest, "command", cmd, float(timeout), *_stable(result)))

    def _record_write(self, path: PurePosixPath, data: bytes) -> None:
        self.digest = digest((self.digest, "write", str(path), digest(data)))

    @override
    async def write_many(self, files: Sequence[tuple[PurePosixPath, bytes]]) -> None:
        for path, data in files:
            await self.write(path, data)

    def _check_open(self) -> None:
        if self.closed:
            raise Restart("Environment is closed")

    @override
    async def close(self) -> None:
        self.closed = True


CONTAINED_TIMEOUT = 'if command -v timeout >/dev/null 2>&1; then exec timeout -k 5 "$1" sh -c "$2"; fi; exec sh -c "$2"'
"""Run ``$2`` under an in-container ``timeout`` of ``$1`` where the image has one."""
BACKSTOP = 30.0
"""Machine deadline beyond the command timeout: the kill-after grace plus exec and capture latency."""


# ShellSim 0.1.39's path error text. It reports ENOENT, not ENOTDIR, for a read below a regular file.
SIMULATION_ERRNOS = {
    "no such file or directory": errno.ENOENT,
    "not a directory": errno.ENOTDIR,
    "is a directory": errno.EISDIR,
    "permission denied": errno.EACCES,
    "too many levels of symbolic links": errno.ELOOP,
    "file name too long": errno.ENAMETOOLONG,
    "file exists": errno.EEXIST,
    "directory not empty": errno.ENOTEMPTY,
    "read-only filesystem": errno.EROFS,
}


def _machine_issue(error: RuntimeError) -> Hold | Restart | Transient:
    if isinstance(error, MachineOutputLimitExceeded):
        return OutputBudgetExhausted("output budget exhausted")
    if isinstance(error, MachineTerminated):
        kind = Hold if isinstance(error, MachineResourceExhausted) else Restart
        return kind(f"environment terminated: {error}")
    return Transient(str(error))


class ShellboxEnv(_RecordedEnv):
    """An owned Shellbox machine; nonzero command exits remain ordinary results."""

    def __init__(
        self, machine: Machine, image: str, notify: Notify, owned: AsyncExitStack, workdir: str = DEFAULT_WORKDIR
    ) -> None:
        self.machine, self.owned = machine, owned
        self.workdir = PurePosixPath(workdir)
        super().__init__(image, notify)

    @asynccontextmanager
    async def _operation(self, path: PurePosixPath | None = None) -> AsyncIterator[None]:
        self._check_open()
        try:
            yield
        except (MachineTerminated, ExecSetupFailed, MachineCancelled) as error:
            raise _machine_issue(error) from error
        except CommandOutcomeUnknown as error:
            # Retrying in place could run the command twice; a stage replay issues it once on a fresh machine.
            # That is safe only for roles without network: one that can mutate an external service needs a
            # durable command identity, or a hold on an unknown outcome.
            raise Restart(f"command outcome unknown: {error}") from error
        except GuestHelperFailed as error:
            raise Hold(f"environment helper failed: {describe_exception(error)}") from error
        except (TimeoutError, ConnectionError) as error:
            raise Transient("Environment operation did not resolve") from error
        except GuestPathError as error:
            # Report the guest's errno against the path the caller asked for.
            if error.errno is None:
                raise
            raise OSError(error.errno, os.strerror(error.errno), str(path or error.filename)) from error
        except OSError as error:
            # Any other OS error is the host's own I/O, such as staging a transfer, never the caller's path.
            raise Hold(f"environment host I/O failed: {describe_exception(error)}") from error
        except SimulationError as error:
            # ShellSim reports path failures as "<strerror>: <path>" text only.
            message, _, name = str(error).partition(": ")
            code = SIMULATION_ERRNOS.get(message.lower())
            if code is None:
                raise
            raise OSError(code, os.strerror(code), str(path) if path is not None else name) from error

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        started = time.monotonic()
        # Iris closes the sandbox when a command outlives its machine deadline, so bound the command
        # inside the container and keep the machine deadline as a backstop.
        contained = isinstance(self.machine, IrisMachine) and timeout > 0
        command = (
            Command(("sh", "-c", CONTAINED_TIMEOUT, "forge-timeout", f"{timeout:g}s", cmd), timeout=timeout + BACKSTOP)
            if contained
            else Command(("sh", "-c", cmd), timeout=timeout)
        )
        async with self._operation():
            result = await self.machine.run(command)
        elapsed = time.monotonic() - started
        # GNU timeout exits 124 at the limit, or 137 when its kill-after escalation was needed.
        killed = contained and result.exit_code == 137 and elapsed >= timeout
        code = 124 if result.reason == ExitReason.TIMED_OUT or killed else result.exit_code
        if code is None:
            raise Hold("environment terminated: command ended without an exit status")
        value = CommandResult(
            code,
            result.stdout.decode(errors="replace"),
            result.stderr.decode(errors="replace"),
            elapsed,
            result.stdout_truncated or result.stderr_truncated,
        )
        self._record_command(cmd, timeout, value)
        return value

    @override
    async def read(self, path: PurePosixPath, *, offset: int = 0, limit: int | None = None) -> bytes:
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("offset and limit must be nonnegative")
        async with self._operation(path):
            with TemporaryDirectory() as directory:
                target, source = Path(directory) / "file", str(self.workdir / path)
                # A file read must not archive a directory the model named by mistake.
                if isinstance(self.machine, IrisMachine):
                    await self.machine.download_file(source, target)
                else:
                    await self.machine.download(source, target)
                with target.open("rb") as source:
                    source.seek(offset)
                    return source.read() if limit is None else source.read(limit)

    @override
    async def write(self, path: PurePosixPath, data: bytes) -> None:
        try:
            async with self._operation(path):
                with TemporaryDirectory() as directory:
                    source = Path(directory) / "file"
                    source.write_bytes(data)
                    await self.machine.upload(source, str(self.workdir / path))
        except Transient as issue:
            # A transfer helper whose reply was lost may still truncate or append after a resend rewrote the file,
            # so only a request that never connected is resent here; a late helper cannot reach a fresh sandbox.
            if never_connected(issue):
                raise
            raise Restart(f"write outcome unknown: {issue}") from issue
        self._record_write(path, data)

    @override
    async def write_many(self, files: Sequence[tuple[PurePosixPath, bytes]]) -> None:
        # Small/native writes are cheaper individually. Iris pays two RPCs per small file.
        if len(files) < 4 or not isinstance(self.machine, IrisMachine):
            await super().write_many(files)
            return
        remote = f"/tmp/forge-hydration-{uuid4().hex}"
        archive = remote + ".tar"
        commands = [shlex.join(("mkdir", "-p", remote)), shlex.join(("tar", "-xf", archive, "-C", remote))]
        for index, (path, _) in enumerate(files):
            target = self.workdir / path
            commands.extend(
                (
                    shlex.join(("mkdir", "-p", str(target.parent))),
                    f"cat {shlex.quote(remote + '/' + str(index))} > {shlex.quote(str(target))}",
                )
            )
        # Ordered redirection retains duplicate-write, symlink and existing-mode semantics of Iris upload.
        cleanup = shlex.join(("rm", "-rf", remote, archive))
        script = " && ".join(commands) + f'; status=$?; {cleanup}; exit "$status"'
        async with self._operation():
            with TemporaryDirectory() as directory:
                source = Path(directory) / "files.tar"
                write_archive(source, [StagedFile(str(index), data) for index, (_, data) in enumerate(files)], [])
                await self.machine.upload(source, archive)
                result = await self.machine.run(Command(("sh", "-c", script), cwd="/"))
                if result.exit_code != 0:
                    raise RuntimeError(f"Could not hydrate files: {result.stderr.decode(errors='replace')}")
        for path, data in files:
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
        except (MachineTerminated, ExecSetupFailed, MachineCancelled) as error:
            raise _machine_issue(error) from error
        except (TimeoutError, ConnectionError) as error:
            raise Transient("Environment startup did not resolve") from error
        image = digest((requirements, self.runtime))
        return ShellboxEnv(machine, image, self.notify, owned, requirements.working_directory or DEFAULT_WORKDIR)


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
    async def read(self, path: PurePosixPath, *, offset: int = 0, limit: int | None = None) -> bytes:
        self._check_open()
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("offset and limit must be nonnegative")
        try:
            return self.files[path][offset : None if limit is None else offset + limit]
        except KeyError as error:
            raise FileNotFoundError(errno.ENOENT, "No such file", str(path)) from error

    @override
    async def write(self, path: PurePosixPath, data: bytes) -> None:
        self._check_open()
        self.files[path] = data
        self._record_write(path, data)


async def run_command(event: RunCommand, env: Env, *, notify: Notify) -> CommandResult:
    return await retry_effect(event, lambda: env.run(event.cmd, timeout=event.timeout), notify)


async def write_file(event: WriteFile, env: Env, *, notify: Notify) -> None:
    await retry_effect(event, lambda: env.write(event.path, event.data), notify)


async def write_files(event: WriteFiles, env: Env, *, notify: Notify) -> None:
    await retry_effect(event, lambda: env.write_many(event.files), notify)


async def read_file(event: ReadFile, env: Env, *, notify: Notify) -> bytes:
    return await retry_effect(event, lambda: env.read(event.path, offset=event.offset, limit=event.limit), notify)
