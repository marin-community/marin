# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run trusted commands as subprocesses on the host, without a container."""

import asyncio
import contextlib
import fcntl
import os
import pwd
import shutil
import signal
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from shellbox.machine import Backend, Command, ExitReason, HostImage, MachineSpec, Result, UnsupportedMachineSpec

DEFAULT_LOCK_PATH = Path("/tmp/shellbox-local-machine.lock")
SCRATCH_ROOT = PurePosixPath("/tmp")
"""Uploads may also target this directory; the machine removes them on close."""
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
LOCK_POLL_INTERVAL = 0.1
READ_CHUNK_BYTES = 64 * 1024
# Resolve the program through sh, as the container backends do: a missing or unexecutable program
# or working directory becomes a failed command (127 or 126 for the program) rather than an exception.
RUN_SCRIPT = 'cd "$1" || exit; shift; exec "$@"'


def _absolute_path(path: str) -> PurePosixPath:
    if not path.startswith("/"):
        raise ValueError(f"Path must be absolute: {path}")
    return PurePosixPath(os.path.normpath(path))


def _within(path: PurePosixPath, roots: Iterable[PurePosixPath]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _remove(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _copy(source: Path, target: Path) -> None:
    """Copy a file, or a directory's contents, to ``target``, creating its parents."""
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _first_new_path(path: Path) -> Path:
    """The highest of ``path`` and its ancestors that does not exist yet; removing it undoes a copy to ``path``."""
    while not path.parent.exists():
        path = path.parent
    return path


def _reset_roots(roots: Iterable[PurePosixPath], workdir: PurePosixPath | None) -> None:
    _remove(map(Path, roots))
    for root in roots:
        Path(root).mkdir(parents=True)
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)


def _command_account(user: str | None) -> pwd.struct_passwd | None:
    """The host account a command switches to, or None to run as this process's user."""
    uid = os.geteuid()
    if user is None or user == str(uid):
        return None
    try:
        account = pwd.getpwuid(int(user)) if user.isdigit() else pwd.getpwnam(user)
    except KeyError as error:
        raise UnsupportedMachineSpec(f"The local backend found no host account for user {user}") from error
    if account.pw_uid == uid:
        return None
    if uid != 0:
        raise UnsupportedMachineSpec(f"The local backend runs as uid {uid}; only root can run a command as {user}")
    return account


async def _exclusive_lock(path: Path) -> int:
    """Open ``path`` and hold an exclusive ``flock`` on it.

    Each call opens its own descriptor, and flock excludes other descriptors in the same process
    as well as other processes. Polling keeps a waiting caller cancellable on any event loop.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return descriptor
            except BlockingIOError:
                await asyncio.sleep(LOCK_POLL_INTERVAL)
    except BaseException:
        os.close(descriptor)
        raise


async def _send_input(stdin: asyncio.StreamWriter, data: bytes) -> None:
    try:
        stdin.write(data)
        await stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass  # The command exited without reading all its input; asyncio's communicate() permits this too.
    stdin.close()


async def _bounded_output(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    """Read ``stream`` to EOF, keeping its first ``limit`` bytes and whether it had more."""
    kept = bytearray()
    truncated = False
    while chunk := await stream.read(READ_CHUNK_BYTES):
        room = limit - len(kept)
        truncated = truncated or len(chunk) > room
        kept += chunk[:room]
    return bytes(kept), truncated


class LocalMachine:
    """Host subprocesses that own the factory's roots until ``close``."""

    def __init__(
        self,
        spec: MachineSpec,
        *,
        owned_roots: tuple[PurePosixPath, ...],
        environment: dict[str, str],
        home: Path,
        lock: int,
    ):
        self.spec = spec
        self.owned_roots = owned_roots
        self.environment = environment
        self.home = home
        self._lock = lock
        self._scratch: list[Path] = []
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")

    async def run(self, command: Command) -> Result:
        self._check_open()
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        account = _command_account(command.user)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            RUN_SCRIPT,
            "shellbox-local",
            command.cwd or self.spec.workdir or "/",
            *command.argv,
            env={**self.environment, **self.spec.env, **command.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            # Popen switches users in its forked child without running Python there, unlike preexec_fn.
            user=None if account is None else account.pw_uid,
            group=None if account is None else account.pw_gid,
            extra_groups=None if account is None else [],
        )
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        limit = command.output_limit_bytes
        try:
            async with asyncio.timeout(command.timeout):
                _, (stdout, stdout_truncated), (stderr, stderr_truncated) = await asyncio.gather(
                    _send_input(process.stdin, command.stdin),
                    _bounded_output(process.stdout, limit),
                    _bounded_output(process.stderr, limit),
                )
                returncode = await process.wait()
        except BaseException as interruption:
            with contextlib.suppress(ProcessLookupError):  # The whole group may have exited already.
                os.killpg(process.pid, signal.SIGKILL)
            # A reader that stopped with a full buffer pauses its pipe, and wait() needs both pipes at EOF.
            await process.stdout.read()
            await process.stderr.read()
            await process.wait()
            if isinstance(interruption, TimeoutError):
                return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
            raise
        # Report a signal as a shell does, matching the backends that run commands under sh.
        exit_code = 128 - returncode if returncode < 0 else returncode
        return Result(exit_code, stdout, stderr, stdout_truncated, stderr_truncated, ExitReason.EXITED)

    async def upload(self, source: Path, target: str) -> None:
        self._check_open()
        path = _absolute_path(target)
        if not _within(path, self.owned_roots):
            if path == SCRATCH_ROOT or not path.is_relative_to(SCRATCH_ROOT):
                raise UnsupportedMachineSpec(
                    f"The local backend uploads only into its owned roots or {SCRATCH_ROOT}, not {target}"
                )
            self._scratch.append(_first_new_path(Path(path)))
        await asyncio.to_thread(_copy, source, Path(path))

    async def download(self, source: str, target: Path) -> None:
        self._check_open()
        path = Path(source)
        if not path.exists():
            raise RuntimeError(f"No such file or directory: {source}")
        await asyncio.to_thread(_copy, path, target)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.to_thread(_remove, [*map(Path, self.owned_roots), *self._scratch, self.home])
        finally:
            os.close(self._lock)


class LocalMachineFactory:
    """Run commands as subprocesses of this host process, one machine per host at a time.

    Use this backend only for trusted commands: they share the host's filesystem, processes,
    and network. ``NetworkPolicy.DENY`` is accepted but not enforced, and ``memory_mb`` is
    ignored. Each machine owns ``owned_roots``: ``create`` empties them and ``close`` removes
    them with the machine's uploads. An exclusive ``flock`` on ``lock_path`` lets only one
    machine exist at a time, across processes that share the path. Commands never inherit the
    host's environment: they get ``bin_dirs`` ahead of a standard ``PATH``, a private ``HOME``,
    ``LANG``, the host's ``PYTHONHASHSEED`` if set, and the spec's and command's variables.
    """

    backend: Backend = Backend.LOCAL

    def __init__(
        self,
        owned_roots: tuple[str, ...],
        *,
        bin_dirs: tuple[Path, ...] = (),
        lock_path: Path = DEFAULT_LOCK_PATH,
    ):
        roots = tuple(_absolute_path(root) for root in owned_roots)
        if PurePosixPath("/") in roots:
            raise ValueError("The filesystem root cannot be an owned root")
        self.owned_roots = roots
        self.bin_dirs = tuple(directory.absolute() for directory in bin_dirs)
        self.lock_path = lock_path

    async def create(self, spec: MachineSpec) -> LocalMachine:
        if not isinstance(spec.source, HostImage):
            raise UnsupportedMachineSpec(f"The local backend requires HostImage, not {type(spec.source).__name__}")
        if spec.cpus is not None or spec.storage_mb is not None or spec.gpus:
            raise UnsupportedMachineSpec("The local backend does not provide CPU, storage, or GPU allocations")
        workdir = _absolute_path(spec.workdir) if spec.workdir else None
        if workdir is not None and not _within(workdir, self.owned_roots):
            raise UnsupportedMachineSpec(f"The local backend's workdir {spec.workdir} must lie in an owned root")
        lock = await _exclusive_lock(self.lock_path)
        try:
            await asyncio.to_thread(_reset_roots, self.owned_roots, workdir)
            home = Path(tempfile.mkdtemp(prefix="shellbox-local-home-"))
        except BaseException:
            os.close(lock)
            raise
        environment = {
            "PATH": os.pathsep.join([*map(str, self.bin_dirs), DEFAULT_PATH]),
            "HOME": str(home),
            "LANG": "C.UTF-8",
        }
        if "PYTHONHASHSEED" in os.environ:
            environment["PYTHONHASHSEED"] = os.environ["PYTHONHASHSEED"]
        return LocalMachine(spec, owned_roots=self.owned_roots, environment=environment, home=home, lock=lock)
