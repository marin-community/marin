# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run trusted commands on the host in bubblewrap sandboxes, without a container image.

Each machine's commands see the host's system directories read-only on top of a private root
directory, which holds every other path they write, from ``/app`` and ``/tmp`` to ``HOME``.
"""

import asyncio
import contextlib
import json
import logging
import math
import os
import pwd
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from bubblewrap_bin import bwrap_path

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

DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
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
# Host directories that commands read and execute but cannot write.
SYSTEM_DIRECTORIES = (
    "/bin",
    "/etc",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/opt",
    "/run/systemd/resolve",  # The target of /etc/resolv.conf on hosts that run systemd-resolved.
    "/sbin",
    "/sys",
    "/usr",
)
SHARED_MEMORY = "/dev/shm"
"""Python's multiprocessing creates its semaphores here."""
SCRATCH = PurePosixPath("/tmp")
HOME = PurePosixPath("/home/shellbox")
ROOT_MODE = 0o755
"""Lets a command that runs as another user traverse the machine's root directory."""
STICKY_WORLD_WRITABLE = 0o1777
PROBE_TIMEOUT = 30


class SandboxUnavailable(RuntimeError):
    """No bwrap executable can build a command sandbox on this host."""


def _absolute_path(path: str) -> PurePosixPath:
    if not path.startswith("/"):
        raise ValueError(f"Path must be absolute: {path}")
    return PurePosixPath(os.path.normpath(path))


def _within(path: PurePosixPath, roots: Iterable[str]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _copy(source: Path, target: Path) -> None:
    """Copy a file, or a directory's contents, to ``target``, creating its parents."""
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _sandbox_argv(
    bwrap: Path,
    *,
    root: Path,
    read_only: Iterable[str],
    network: NetworkPolicy,
    account: pwd.struct_passwd | None,
) -> list[str]:
    """The bwrap invocation that runs a program over ``root`` with ``read_only`` and the system directories.

    The program runs without capabilities; as root, bwrap needs no user namespace for its own.
    """
    argv = [str(bwrap), "--unshare-ipc", "--unshare-pid", "--unshare-uts", "--unshare-cgroup-try"]
    if network == NetworkPolicy.DENY:
        argv.append("--unshare-net")
    argv += ["--new-session", "--die-with-parent", "--cap-drop", "ALL"]
    if account is not None:
        # setpriv uses these to switch to the account, and the switch clears them with every other capability.
        argv += ["--cap-add", "CAP_SETUID", "--cap-add", "CAP_SETGID"]
    argv += ["--bind", str(root), "/"]
    for path in SYSTEM_DIRECTORIES:
        if os.path.islink(path):
            # Merged-/usr hosts link /bin and /lib into /usr; the link keeps paths such as the ELF loader valid.
            argv += ["--symlink", os.readlink(path), path]
        else:
            argv += ["--ro-bind-try", path, path]
    argv += ["--dev", "/dev", "--tmpfs", SHARED_MEMORY, "--proc", "/proc"]
    for path in read_only:
        argv += ["--ro-bind", path, path]
    if account is not None:
        argv += ["setpriv", f"--reuid={account.pw_uid}", f"--regid={account.pw_gid}", "--clear-groups", "--"]
    return argv


def _bwrap_candidates(bwrap: Path | None) -> tuple[Path, ...]:
    """``bwrap`` alone, or the bundled bwrap followed by any bwrap on ``PATH``; the caller probes each in turn."""
    if bwrap is not None:
        return (bwrap,)
    system = shutil.which("bwrap")
    return (bwrap_path(), *(() if system is None else (Path(system),)))


def _sandbox_error(bwrap: Path) -> str | None:
    """Why ``bwrap`` cannot build a command sandbox here, or None when it can."""
    with tempfile.TemporaryDirectory(prefix="shellbox-bwrap-probe-") as root:
        Path(root).chmod(ROOT_MODE)
        argv = _sandbox_argv(bwrap, root=Path(root), read_only=(), network=NetworkPolicy.DENY, account=None)
        try:
            probe = subprocess.run(
                [*argv, "/bin/sh", "-c", "exit 0"], capture_output=True, text=True, timeout=PROBE_TIMEOUT, check=False
            )
        except OSError as error:
            return str(error)
    return None if probe.returncode == 0 else probe.stderr.strip() or f"exit status {probe.returncode}"


def _working_bwrap(candidates: Iterable[Path]) -> Path:
    errors = []
    for candidate in candidates:
        error = _sandbox_error(candidate)
        if error is None:
            return candidate
        errors.append(f"{candidate}: {error}")
    raise SandboxUnavailable(
        "No bwrap can sandbox commands on this host. A root process needs CAP_SYS_ADMIN, CAP_NET_ADMIN, no "
        "AppArmor confinement and a seccomp filter that allows pivot_root; another user needs unprivileged "
        "user namespaces. " + "; ".join(errors)
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
        # kill can reach this limit.
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


def _prepare_root(root: Path, workdir: PurePosixPath) -> None:
    root.chmod(ROOT_MODE)
    # Any user may write /tmp and HOME, as in a container image.
    for directory in (SCRATCH, HOME):
        path = root / directory.relative_to("/")
        path.mkdir(parents=True)
        path.chmod(STICKY_WORLD_WRITABLE)
    (root / workdir.relative_to("/")).mkdir(parents=True, exist_ok=True)


class LocalMachine:
    """Commands in bubblewrap sandboxes over ``root``, a host directory that only this machine uses.

    Each command gets new PID, IPC and UTS namespaces, and a network namespace with only loopback under
    ``NetworkPolicy.DENY``. Files persist in ``root`` between commands; processes end with their command.
    """

    def __init__(
        self,
        spec: MachineSpec,
        *,
        bwrap: Path,
        root: Path,
        read_only: tuple[str, ...],
        environment: dict[str, str],
    ):
        self.spec = spec
        self.root = root
        self.read_only = read_only
        self._bwrap = bwrap
        self._environment = environment
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")

    def _host_path(self, path: PurePosixPath) -> Path:
        """Where the host keeps the file that commands see at ``path``.

        A command may plant a symlink in its root that points outside it; a transfer must not follow one.
        """
        if _within(path, (*SYSTEM_DIRECTORIES, *self.read_only)):
            return Path(path)
        host = self.root / path.relative_to("/")
        if not Path(os.path.realpath(host)).is_relative_to(self.root):
            raise RuntimeError(f"{path} leaves the machine root through a symlink")
        return host

    async def run(self, command: Command) -> Result:
        self._check_open()
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        sandbox = _sandbox_argv(
            self._bwrap,
            root=self.root,
            read_only=self.read_only,
            network=self.spec.network,
            account=_command_account(command.user),
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-c",
            LAUNCH_SOURCE,
            json.dumps(_resource_limits(command.timeout)),
            *sandbox,
            "/bin/sh",
            "-c",
            RUN_SCRIPT,
            "shellbox-local",
            command.cwd or self.spec.workdir or "/",
            *command.argv,
            env={**self._environment, **self.spec.env, **command.env},
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
            # Killing bwrap ends the sandbox: --die-with-parent kills its PID namespace and every process in it.
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
        if _within(path, (*SYSTEM_DIRECTORIES, *self.read_only)):
            raise UnsupportedMachineSpec(f"The local backend's {target} is a read-only host directory")
        await asyncio.to_thread(_copy, source, self._host_path(path))

    async def download(self, source: str, target: Path) -> None:
        self._check_open()
        path = self._host_path(_absolute_path(source))
        if not path.exists():
            raise RuntimeError(f"No such file or directory: {source}")
        await asyncio.to_thread(_copy, path, target)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(shutil.rmtree, self.root)


class LocalMachineFactory:
    """Run trusted commands on this host in bubblewrap sandboxes, any number of machines at a time.

    A machine's commands see the host's system directories (``/usr``, ``/etc``, ``/opt`` and the like)
    and each directory in ``read_only`` at its own path, read-only. Every other path, including ``/tmp``
    and ``HOME``, lies in a root directory of the machine's own that starts empty and is removed by
    ``close``; no other host file is visible. The caller keeps each ``read_only`` directory self-contained:
    a symlink that leads out of the mounted directories does not resolve in the sandbox.
    ``build_python_environment`` builds such a directory for a Python interpreter and its packages.

    Commands never inherit the host's environment: they get ``bin_dirs`` ahead of a standard ``PATH``,
    ``HOME``, ``LANG``, ``PYTHONHASHSEED`` when the factory has a ``hash_seed``, and the spec's and
    command's variables. ``bin_dirs`` only sets ``PATH``; a directory outside the mounted ones is not
    visible to commands. A spec with ``memory_mb`` is rejected, since the backend enforces no memory limit.

    ``bwrap`` names the executable to use. By default the factory takes the first of the bundled bwrap
    and any bwrap on ``PATH`` that can build a sandbox on this host, and raises ``SandboxUnavailable``
    when none can.
    """

    backend: Backend = Backend.LOCAL

    def __init__(
        self,
        *,
        read_only: tuple[Path, ...] = (),
        bin_dirs: tuple[Path, ...] = (),
        bwrap: Path | None = None,
        hash_seed: str | None = None,
    ):
        self.read_only = tuple(directory.absolute() for directory in read_only)
        missing = [str(directory) for directory in self.read_only if not directory.is_dir()]
        if missing:
            raise ValueError(f"Read-only directories do not exist: {', '.join(missing)}")
        self.bin_dirs = tuple(directory.absolute() for directory in bin_dirs)
        self.hash_seed = hash_seed
        self.bwrap = _working_bwrap(_bwrap_candidates(bwrap))
        logger.info("Local backend sandboxes commands with %s", self.bwrap)

    async def create(self, spec: MachineSpec) -> LocalMachine:
        if not isinstance(spec.source, HostImage):
            raise UnsupportedMachineSpec(f"The local backend requires HostImage, not {type(spec.source).__name__}")
        if spec.cpus is not None or spec.storage_mb is not None or spec.gpus or spec.memory_mb is not None:
            raise UnsupportedMachineSpec("The local backend does not provide CPU, memory, storage, or GPU allocations")
        read_only = (*self.read_only, *(directory.absolute() for directory in spec.source.read_only))
        missing = [str(directory) for directory in read_only if not directory.is_dir()]
        if missing:
            raise ValueError(f"Read-only directories do not exist: {', '.join(missing)}")
        read_only_paths = tuple(map(str, read_only))
        workdir = _absolute_path(spec.workdir or "/")
        if _within(workdir, (*SYSTEM_DIRECTORIES, *read_only_paths)):
            raise UnsupportedMachineSpec(f"The local backend's workdir {spec.workdir} is a read-only host directory")
        root = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="shellbox-local-"))
        try:
            await asyncio.to_thread(_prepare_root, root, workdir)
        except BaseException:
            shutil.rmtree(root)
            raise
        environment = {
            "PATH": os.pathsep.join(
                [
                    *(str(directory.absolute()) for directory in spec.source.bin_dirs),
                    *map(str, self.bin_dirs),
                    DEFAULT_PATH,
                ]
            ),
            "HOME": str(HOME),
            "LANG": "C.UTF-8",
        }
        if self.hash_seed is not None:
            environment["PYTHONHASHSEED"] = self.hash_seed
        return LocalMachine(spec, bwrap=self.bwrap, root=root, read_only=read_only_paths, environment=environment)
