# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run trusted commands as subprocesses on the host, without a container."""

import asyncio
import contextlib
import dataclasses
import fcntl
import json
import logging
import math
import os
import pwd
import resource
import shutil
import signal
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shellbox.backends.local import launch
from shellbox.machine import (
    Backend,
    Command,
    ExitReason,
    HostImage,
    MachineSpec,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
)

logger = logging.getLogger(__name__)

DEFAULT_LOCK_PATH = Path("/tmp/shellbox-local-machine.lock")
SCRATCH_ROOT = PurePosixPath("/tmp")
"""Uploads may also target this directory; the machine removes them on close."""
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
LOCK_POLL_INTERVAL = 0.1
READ_CHUNK_BYTES = 64 * 1024
# Resolve the program through sh, as the container backends do: a missing or unexecutable program
# or working directory becomes a failed command (127 or 126 for the program) rather than an exception.
RUN_SCRIPT = 'cd "$1" || exit; shift; exec "$@"'
LAUNCH_SOURCE = Path(launch.__file__).read_text()
NPROC_HEADROOM = 256
"""Tasks a command may add beyond the host's task count when it starts; RLIMIT_NPROC counts all of a user's tasks."""
FILE_SIZE_LIMIT = 1 << 30
CPU_GRACE = 5
"""CPU seconds a command may use beyond the most its timeout allows on the CPUs it may run on."""
# Commands may read and execute these, and the venvs and interpreters of the factory's bin_dirs, but not write them.
SYSTEM_READ_ROOTS = (
    "/bin",
    "/dev",
    "/etc",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/opt",
    "/proc",
    "/run/systemd/resolve",  # The target of /etc/resolv.conf on hosts that run systemd-resolved.
    "/sbin",
    "/sys",
    "/usr",
)
WRITABLE_DEVICES = ("/dev/null", "/dev/zero", "/dev/full")
SHARED_MEMORY = "/dev/shm"
"""Python's multiprocessing creates its semaphores here."""
LANDLOCK_READ = launch.LANDLOCK_EXECUTE | launch.LANDLOCK_READ_FILE | launch.LANDLOCK_READ_DIR
LANDLOCK_DEVICE_WRITE = launch.LANDLOCK_WRITE_FILE | launch.LANDLOCK_TRUNCATE


@dataclass(frozen=True)
class Lockdown:
    """The confinement this host's kernel lets the local backend apply to commands, beyond resource limits."""

    no_new_privs: bool
    """Setuid programs and file capabilities cannot raise a command's privileges."""
    filesystem: bool
    """Landlock limits writes to the owned roots, ``HOME``, ``/tmp`` and ``/dev/shm``, and reads to system and
    interpreter directories. It also stops commands from reading or tracing processes outside the command."""
    tcp: bool
    """Landlock refuses TCP bind and connect under ``NetworkPolicy.DENY``."""
    scopes: bool
    """Landlock stops commands from signalling processes outside the command or reaching their abstract sockets."""


@dataclass(frozen=True)
class LandlockRuleset:
    """The keyword arguments of the launcher's ``restrict``."""

    handled_fs: int
    handled_net: int
    scoped: int
    rules: tuple[tuple[str, int], ...]


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


def _reset_roots(roots: Iterable[PurePosixPath], shared: Iterable[PurePosixPath], workdir: PurePosixPath | None) -> None:
    _remove(map(Path, roots))
    for root in roots:
        Path(root).mkdir(parents=True)
    for root in shared:
        Path(root).mkdir(parents=True, exist_ok=True)
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)


def _landlock_fs_rights(abi: int) -> int:
    """Every filesystem right Landlock ABI ``abi`` handles: ABI 1 has 13; 2 adds REFER, 3 TRUNCATE, 5 IOCTL_DEV."""
    return (1 << {1: 13, 2: 14, 3: 15, 4: 15}.get(abi, 16)) - 1


def _interpreter_roots(bin_dirs: Iterable[Path]) -> tuple[str, ...]:
    """Directories to read for the programs in ``bin_dirs``: each venv and the Python it links to, or the directory."""
    roots: list[str] = []
    for directory in bin_dirs:
        venv = directory.parent
        if (venv / "pyvenv.cfg").exists():
            roots += [str(venv), str(Path(os.path.realpath(directory / "python3")).parent.parent)]
        else:
            roots.append(str(directory))
    return tuple(roots)


def _landlock_ruleset(
    abi: int, *, readable: Iterable[str], writable: Iterable[str], network: NetworkPolicy
) -> LandlockRuleset:
    fs_rights = _landlock_fs_rights(abi)
    return LandlockRuleset(
        handled_fs=fs_rights,
        handled_net=launch.LANDLOCK_NET_TCP if abi >= 4 and network == NetworkPolicy.DENY else 0,
        scoped=launch.LANDLOCK_SCOPES if abi >= 6 else 0,
        rules=(
            *((path, LANDLOCK_READ) for path in readable),
            *((path, LANDLOCK_DEVICE_WRITE & fs_rights) for path in WRITABLE_DEVICES),
            *((path, fs_rights) for path in writable),
        ),
    )


def _resource_limits(timeout: float | None) -> list[tuple[int, int]]:
    host_tasks = int(Path("/proc/loadavg").read_text().split()[3].split("/")[1])
    limits = [
        (resource.RLIMIT_NPROC, host_tasks + NPROC_HEADROOM),
        (resource.RLIMIT_FSIZE, FILE_SIZE_LIMIT),
        (resource.RLIMIT_CORE, 0),
    ]
    if timeout is not None:
        # A process's CPU time accrues on every CPU it runs on, so only a process that escapes the timeout's
        # kill, for example by leaving the process group, can reach this limit.
        limits.append((resource.RLIMIT_CPU, math.ceil(timeout * len(os.sched_getaffinity(0))) + CPU_GRACE))
    return limits


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
        shared_roots: tuple[PurePosixPath, ...],
        environment: dict[str, str],
        home: Path,
        lock: int,
        no_new_privs: bool,
        landlock: LandlockRuleset | None,
    ):
        self.spec = spec
        self.owned_roots = owned_roots
        self.shared_roots = shared_roots
        self.environment = environment
        self.home = home
        self._lock = lock
        self._no_new_privs = no_new_privs
        self._landlock = landlock
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
        launch_config = {
            "rlimits": _resource_limits(command.timeout),
            "no_new_privs": self._no_new_privs,
            "landlock": None if self._landlock is None else dataclasses.asdict(self._landlock),
            # The launcher switches users after confining itself, so the command's user needs no access to
            # this process's interpreter, which may lie in a private home directory.
            "account": None if account is None else (account.pw_uid, account.pw_gid),
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-c",
            LAUNCH_SOURCE,
            json.dumps(launch_config),
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
            if path in (*self.shared_roots, SCRATCH_ROOT) or not _within(path, (*self.shared_roots, SCRATCH_ROOT)):
                raise UnsupportedMachineSpec(
                    f"The local backend uploads only into its owned or shared roots or {SCRATCH_ROOT}, not {target}"
                )
            # Shared roots and scratch outlive the machine, so only what this upload adds is removed at close.
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

    Use this backend only for trusted commands: they share the host's kernel, processes, and
    network. Each command runs under resource limits and whatever confinement ``lockdown``
    reports this host's kernel allows; a command may escape where the kernel does not.
    ``memory_mb`` is ignored. Each machine owns ``owned_roots``: ``create`` empties them and
    ``close`` removes them with the machine's uploads, so no owned root may hold this process's
    interpreter or working directory. ``shared_roots`` are directories commands may also write and
    uploads may target, such as a workspace the host process itself runs from; ``create`` makes them
    without emptying them and ``close`` removes only the machine's uploads there, so files a command
    writes to a shared root remain for the next machine. An exclusive ``flock`` on ``lock_path``
    lets only one machine exist at a time, across processes that share the path. Commands
    never inherit the host's environment: they get ``bin_dirs`` ahead of a standard ``PATH``,
    a private ``HOME``, ``LANG``, the host's ``PYTHONHASHSEED`` if set, and the spec's and
    command's variables.
    """

    backend: Backend = Backend.LOCAL

    def __init__(
        self,
        owned_roots: tuple[str, ...],
        *,
        shared_roots: tuple[str, ...] = (),
        bin_dirs: tuple[Path, ...] = (),
        lock_path: Path = DEFAULT_LOCK_PATH,
    ):
        roots = tuple(_absolute_path(root) for root in owned_roots)
        shared = tuple(_absolute_path(root) for root in shared_roots)
        if PurePosixPath("/") in (*roots, *shared):
            raise ValueError("The filesystem root cannot be an owned or shared root")
        if any(_within(root, roots) or any(_within(owned, (root,)) for owned in roots) for root in shared):
            raise ValueError("A shared root cannot overlap an owned root")
        # create() deletes the owned roots, so a root holding this process's interpreter or working directory,
        # as /app holds an Iris task's bundle and venv, would delete the running worker.
        for path in (os.getcwd(), sys.prefix, sys.executable):
            if _within(_absolute_path(path), roots):
                raise ValueError(f"An owned root holds {path}, which this process runs from")
        self.owned_roots = roots
        self.shared_roots = shared
        self.bin_dirs = tuple(directory.absolute() for directory in bin_dirs)
        self.lock_path = lock_path
        no_new_privs = launch.no_new_privs_available()
        # Landlock needs no_new_privs to confine an unprivileged process.
        self._landlock_abi = launch.landlock_abi() if no_new_privs else 0
        self.lockdown = Lockdown(
            no_new_privs=no_new_privs,
            filesystem=self._landlock_abi >= 1,
            tcp=self._landlock_abi >= 4,
            scopes=self._landlock_abi >= 6,
        )
        logger.info("Local backend lockdown on this host: %s", self.lockdown)

    async def create(self, spec: MachineSpec) -> LocalMachine:
        if not isinstance(spec.source, HostImage):
            raise UnsupportedMachineSpec(f"The local backend requires HostImage, not {type(spec.source).__name__}")
        if spec.cpus is not None or spec.storage_mb is not None or spec.gpus:
            raise UnsupportedMachineSpec("The local backend does not provide CPU, storage, or GPU allocations")
        # Commands run in / without a workdir; / always exists, so it needs no owned root.
        workdir = _absolute_path(spec.workdir) if spec.workdir not in ("", "/") else None
        if workdir is not None and not _within(workdir, (*self.owned_roots, *self.shared_roots)):
            raise UnsupportedMachineSpec(
                f"The local backend's workdir {spec.workdir} must lie in an owned or shared root"
            )
        lock = await _exclusive_lock(self.lock_path)
        try:
            await asyncio.to_thread(_reset_roots, self.owned_roots, self.shared_roots, workdir)
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
        landlock = None
        if self.lockdown.filesystem:
            landlock = _landlock_ruleset(
                self._landlock_abi,
                readable=(*SYSTEM_READ_ROOTS, *_interpreter_roots(self.bin_dirs)),
                writable=(
                    *map(str, self.owned_roots),
                    *map(str, self.shared_roots),
                    str(home),
                    str(SCRATCH_ROOT),
                    SHARED_MEMORY,
                ),
                network=spec.network,
            )
        return LocalMachine(
            spec,
            owned_roots=self.owned_roots,
            shared_roots=self.shared_roots,
            environment=environment,
            home=home,
            lock=lock,
            no_new_privs=self.lockdown.no_new_privs,
            landlock=landlock,
        )
