# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-writer environments with live command and binary-file effects."""

import errno
import os
import shlex
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tarfile import TarError
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

from taskforge.effects.events import ENVIRONMENT, Event, Fingerprint, ReplayPolicy, digest
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
    kind = "env.run_command"
    cmd: str
    timeout: float


@dataclass(frozen=True, slots=True)
class WriteFile(Event[None]):
    kind = "env.write_file"
    path: PurePosixPath
    data: bytes


@dataclass(frozen=True, slots=True)
class WriteFiles(Event[None]):
    kind = "env.write_files"
    files: tuple[tuple[PurePosixPath, bytes], ...]


@dataclass(frozen=True, slots=True)
class ReadFile(Event[bytes]):
    kind = "env.read_file"
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


class RecordedEnv(Env):
    """An environment fingerprinted by its initial sandbox definition, which each of its outcome rows records."""

    def __init__(self, image: str) -> None:
        self.digest = digest((image,))
        self.closed = False

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(ENVIRONMENT, self.digest, ReplayPolicy.REEXECUTE)

    @override
    async def write_many(self, files: Sequence[tuple[PurePosixPath, bytes]]) -> None:
        for path, data in files:
            await self.write(path, data)

    def check_open(self) -> None:
        if self.closed:
            raise Restart("Environment is closed")

    @override
    async def close(self) -> None:
        self.closed = True


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


class ShellboxEnv(RecordedEnv):
    """An owned Shellbox machine; nonzero command exits remain ordinary results."""

    def __init__(self, machine: Machine, image: str, owned: AsyncExitStack, workdir: str = DEFAULT_WORKDIR) -> None:
        self.machine, self.owned = machine, owned
        self.workdir = PurePosixPath(workdir)
        super().__init__(image)

    @asynccontextmanager
    async def _operation(self, path: PurePosixPath | None = None) -> AsyncIterator[None]:
        self.check_open()
        try:
            yield
        except (MachineTerminated, ExecSetupFailed, MachineCancelled) as error:
            raise _machine_issue(error) from error
        except CommandOutcomeUnknown as error:
            # Retrying in place could run the command twice; a stage restart issues it once on a fresh machine.
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
        # Every backend Taskforge uses bounds the command and keeps the machine; the model sees timeout's 124.
        async with self._operation():
            result = await self.machine.run(Command(("sh", "-c", cmd), timeout=timeout))
        elapsed = time.monotonic() - started
        code = 124 if result.reason == ExitReason.TIMED_OUT else result.exit_code
        if code is None:
            raise Hold("environment terminated: command ended without an exit status")
        value = CommandResult(
            code,
            result.stdout.decode(errors="replace"),
            result.stderr.decode(errors="replace"),
            elapsed,
            result.stdout_truncated or result.stderr_truncated,
        )
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

    @override
    async def write_many(self, files: Sequence[tuple[PurePosixPath, bytes]]) -> None:
        # Small/native writes are cheaper individually. Iris pays two RPCs per small file.
        if len(files) < 4 or not isinstance(self.machine, IrisMachine):
            await super().write_many(files)
            return
        remote = f"/tmp/forge-hydration-{uuid4().hex}"
        archive = remote + ".tar"
        commands = [shlex.join(("mkdir", "-p", remote)), shlex.join(("tar", "-xf", archive, "-C", remote)), "staged=1"]
        for index, (path, _) in enumerate(files):
            target = self.workdir / path
            commands.extend(
                (
                    shlex.join(("mkdir", "-p", str(target.parent))),
                    f"cat {shlex.quote(remote + '/' + str(index))} > {shlex.quote(str(target))}",
                )
            )
        # Ordered redirection retains duplicate-write, symlink and existing-mode semantics of Iris upload.
        cleanup = shlex.join(("rm", "-rf", remote, archive)) + "; [ $staged = 1 ] || exit 125"  # 125: unpack failed
        script = "staged=0; " + " && ".join(commands) + f'; status=$?; {cleanup}; exit "$status"'
        async with self._operation():
            with TemporaryDirectory() as directory:
                source = Path(directory) / "files.tar"
                write_archive(source, [StagedFile(str(index), data) for index, (_, data) in enumerate(files)], [])
                # Staging Taskforge's own scratch archive fails as a guest helper, never on the caller's paths.
                try:
                    await self.machine.upload(source, archive)
                except GuestPathError as error:
                    raise GuestHelperFailed(f"Could not stage hydration archive: {error}") from error
                result = await self.machine.run(Command(("sh", "-c", script), cwd="/"))
                stderr = result.stderr.decode(errors="replace")
                if result.exit_code == 125:
                    raise GuestHelperFailed(f"Could not unpack hydration archive: {stderr}")
                if result.exit_code != 0:
                    raise RuntimeError(f"Could not hydrate files: {stderr}")

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
        return ShellboxEnv(machine, image, owned, requirements.working_directory or DEFAULT_WORKDIR)


async def live[T](event: Event[T], operation: Callable[[], Awaitable[T]], notify: Notify) -> T:
    """Apply ``event`` to this machine, on every launch; a transfer archive the backend cannot unpack restarts."""
    try:
        return await retry_effect(event, operation, notify)
    except TarError as error:
        raise Restart(f"{event.kind} transfer failed: {describe_exception(error)}") from error


async def run_command(event: RunCommand, env: Env, *, notify: Notify) -> CommandResult:
    return await live(event, lambda: env.run(event.cmd, timeout=event.timeout), notify)


async def write_file(event: WriteFile, env: Env, *, notify: Notify) -> None:
    await live(event, lambda: env.write(event.path, event.data), notify)


async def write_files(event: WriteFiles, env: Env, *, notify: Notify) -> None:
    await live(event, lambda: env.write_many(event.files), notify)


async def read_file(event: ReadFile, env: Env, *, notify: Notify) -> bytes:
    return await live(event, lambda: env.read(event.path, offset=event.offset, limit=event.limit), notify)
