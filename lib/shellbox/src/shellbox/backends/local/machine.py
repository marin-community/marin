# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run trusted commands on the host in bubblewrap sandboxes, without a container image.

Each machine's commands see the host's programs, read-only or copy-on-write, and private runtime
configuration on top of a private root directory, which holds paths from ``/app`` and ``/tmp`` to ``HOME``.
"""

import asyncio
import contextlib
import io
import json
import logging
import math
import os
import pwd
import re
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path, PurePosixPath

from shellbox.backends.local import launch
from shellbox.backends.local.python_environment import DEFAULT_PYTHON_VERSION, resolve_python_environment
from shellbox.machine import (
    DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    Command,
    CommandSemantics,
    ExecutionRequirements,
    ExitReason,
    HostImage,
    MachineSpec,
    Mount,
    NetworkPolicy,
    PackageSpec,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
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
# Reserve program directories and private configuration paths from caller-supplied mounts.
SYSTEM_DIRECTORIES = (
    "/bin",
    "/etc",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/sbin",
    "/sys",
    "/usr",
)
OVERLAID_DIRECTORIES = ("/bin", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/sbin", "/usr")
"""The system directories that a copy-on-write machine can write; ``/sys`` stays read-only."""
PRIVATE_SYSTEM_DIRECTORIES = ("/etc", "/sys")
# Copy only runtime configuration. Host credentials, shell startup files, /opt and sysfs are not inputs.
RUNTIME_CONFIGURATION = (
    "/etc/alternatives",
    "/etc/group",
    "/etc/hosts",
    "/etc/ld.so.cache",
    "/etc/localtime",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/resolv.conf",
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs",
)
MOUNTINFO = Path("/proc/self/mountinfo")
# Transfers that reach an overlaid directory pass through the sandbox as a tar stream, so they see the machine's
# layer. A file that is a symlink is archived as the file it leads to inside the sandbox, not as the link.
EXTRACT_SCRIPT = 'mkdir -p -- "$1" && exec tar -xof - -C "$1"'
ARCHIVE_SCRIPT = (
    '[ -e "$1" ] || { echo "No such file or directory: $1" >&2; exit 2; }; '
    'if [ -d "$1" ]; then exec tar -cf - -C "$1" .; fi; '
    'exec tar -chf - -C "$(dirname -- "$1")" -- "$(basename -- "$1")"'
)
ARCHIVE_LIMIT_BYTES = 1 << 30
"""The largest download from an overlaid directory, as a tar stream."""
SHARED_MEMORY = "/dev/shm"
"""Python's multiprocessing creates its semaphores here."""
SCRATCH = PurePosixPath("/tmp")
HOME = PurePosixPath("/home/shellbox")
CONTROL_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "HOME": str(HOME), "LANG": "C.UTF-8"}
"""The environment of the backend's own commands, such as transfers; ``MachineSpec.env`` does not reach them."""
ROOT_MODE = 0o755
"""Lets a command that runs as another user traverse the machine's root directory."""
STICKY_WORLD_WRITABLE = 0o1777
PROBE_TIMEOUT = 30


class SandboxUnavailable(RuntimeError):
    """No bwrap executable can build a command sandbox on this host."""


class SystemDirectories(StrEnum):
    """How a local machine's commands see the host's system directories."""

    READ_ONLY = "read_only"
    """Bound read-only: a write fails."""
    COPY_ON_WRITE = "copy_on_write"
    """Writable per machine on every directory in ``OVERLAID_DIRECTORIES``.

    Commands see the host's programs, private runtime configuration and their own machine's writes,
    which persist between its commands and which ``close`` discards; the host's files never change.
    Files the host mounts inside overlaid directories stay read-only. Needs overlayfs and a temporary
    directory that is not itself on overlayfs.
    """


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


def _remove(path: Path) -> None:
    """Remove a machine's directory, including the work directory that overlayfs leaves without permissions."""

    def unlock(_, failed: str, error: BaseException) -> None:
        if not isinstance(error, PermissionError) or not os.path.isdir(failed):
            raise error
        os.chmod(failed, 0o700)
        shutil.rmtree(failed)

    shutil.rmtree(path, onexc=unlock)


def _overlaid_directories() -> tuple[str, ...]:
    """The overlaid directories this host has; a link into ``/usr`` on a merged-/usr host is not one."""
    return tuple(
        path
        for path in OVERLAID_DIRECTORIES
        if path not in PRIVATE_SYSTEM_DIRECTORIES and os.path.isdir(path) and not os.path.islink(path)
    )


def _layer(layers: Path, directory: str) -> Path:
    return layers / directory.strip("/")


def _prepare_layers(layers: Path) -> None:
    for directory in _overlaid_directories():
        (_layer(layers, directory) / "upper").mkdir(parents=True)
        (_layer(layers, directory) / "work").mkdir()


def _submounts(mountinfo: str, directories: Iterable[str]) -> tuple[str, ...]:
    """The mount points strictly inside ``directories``, parents first, from the text of ``/proc/self/mountinfo``."""
    points = []
    for line in mountinfo.splitlines():
        # The kernel escapes space, tab, newline and backslash in mount points as three octal digits.
        point = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), line.split()[4])
        if any(PurePosixPath(point).is_relative_to(directory) and point != directory for directory in directories):
            points.append(point)
    return tuple(sorted(dict.fromkeys(points), key=lambda point: (point.count("/"), point)))


def read_only_system_mounts() -> tuple[str, ...]:
    """The paths the host mounts inside overlaid directories; copy-on-write machines keep them read-only."""
    return _submounts(MOUNTINFO.read_text(), _overlaid_directories())


def _sandbox_argv(
    bwrap: Path,
    *,
    root: Path,
    layers: Path | None,
    mounts: Iterable[Mount],
    network: NetworkPolicy,
    account: pwd.struct_passwd | None,
    environment: dict[str, str],
) -> list[str]:
    """The bwrap invocation that runs a program over ``root`` with the system directories and ``mounts``.

    The system directories are read-only, or, with ``layers``, overlaid by the machine's layer of each.
    The program runs without capabilities; as root, bwrap needs no user namespace for its own.
    """
    argv = [str(bwrap), "--unshare-ipc", "--unshare-pid", "--unshare-uts", "--unshare-cgroup-try", "--clearenv"]
    for key, value in environment.items():
        argv += ["--setenv", key, value]
    if network == NetworkPolicy.DENY:
        argv.append("--unshare-net")
    argv += ["--new-session", "--die-with-parent", "--cap-drop", "ALL"]
    if account is not None:
        # setpriv uses these to switch to the account, and the switch clears them with every other capability.
        argv += ["--cap-add", "CAP_SETUID", "--cap-add", "CAP_SETGID"]
    argv += ["--bind", str(root), "/"]
    overlaid = () if layers is None else _overlaid_directories()
    for path in SYSTEM_DIRECTORIES:
        if path in PRIVATE_SYSTEM_DIRECTORIES:
            if path == "/sys" or layers is None:
                argv += ["--ro-bind", str(root / path.lstrip("/")), path]
        elif os.path.islink(path):
            # Merged-/usr hosts link /bin and /lib into /usr; the link keeps paths such as the ELF loader valid.
            argv += ["--symlink", os.readlink(path), path]
        elif path in overlaid:
            assert layers is not None
            layer = _layer(layers, path)
            argv += ["--overlay-src", path, "--overlay", str(layer / "upper"), str(layer / "work"), path]
        else:
            argv += ["--ro-bind-try", path, path]
    # Overlays hide nested host mounts; preserve those inside shared program directories read-only.
    for point in read_only_system_mounts() if overlaid else ():
        argv += ["--ro-bind", point, point]
    argv += ["--dev", "/dev", "--tmpfs", SHARED_MEMORY, "--proc", "/proc"]
    for mount in mounts:
        argv += ["--ro-bind", str(mount.host), str(mount.guest)]
    if account is not None:
        argv += ["setpriv", f"--reuid={account.pw_uid}", f"--regid={account.pw_gid}", "--clear-groups", "--"]
    return argv


def _bwrap_candidates(bwrap: Path | None) -> tuple[Path, ...]:
    """``bwrap`` alone, or the bundled bwrap followed by any bwrap on ``PATH``; the caller probes each in turn."""
    if sys.platform != "linux":
        raise SandboxUnavailable("The local bubblewrap backend requires Linux")
    # This dependency is installed only on Linux.
    from bubblewrap_bin import bwrap_path  # noqa: PLC0415

    if bwrap is not None:
        return (bwrap,)
    bundled = ()
    if sys.platform == "linux":
        from bubblewrap_bin import bwrap_path  # noqa: PLC0415  (optional Linux-only dependency)

        bundled = (bwrap_path(),)
    system = shutil.which("bwrap")
    return (*bundled, *(() if system is None else (Path(system),)))


def _prepare_sandbox(bwrap: Path, system_directories: SystemDirectories) -> str | None:
    """Build a sandbox with ``bwrap`` and run a command in it; why it cannot, or None when it can."""
    root = Path(tempfile.mkdtemp(prefix="shellbox-bwrap-probe-"))
    layers = None
    try:
        _prepare_root(root, None)
        if system_directories == SystemDirectories.COPY_ON_WRITE:
            layers = Path(tempfile.mkdtemp(prefix="shellbox-bwrap-probe-layers-"))
            _prepare_layers(layers)
        argv = _sandbox_argv(
            bwrap,
            root=root,
            layers=layers,
            mounts=(),
            network=NetworkPolicy.DENY,
            account=None,
            environment=CONTROL_ENVIRONMENT,
        )
        try:
            probe = subprocess.run(
                [*argv, "/bin/sh", "-c", "exit 0"],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT,
                check=False,
                env=CONTROL_ENVIRONMENT,
            )
        except OSError as error:
            return str(error)
    finally:
        _remove(root)
        if layers is not None:
            _remove(layers)
    return None if probe.returncode == 0 else probe.stderr.strip() or f"exit status {probe.returncode}"


def _working_bwrap(candidates: Iterable[Path], system_directories: SystemDirectories) -> Path:
    errors = []
    for candidate in candidates:
        error = _prepare_sandbox(candidate, system_directories)
        if error is None:
            return candidate
        errors.append(f"{candidate}: {error}")
    overlay = (
        "Copy-on-write system directories also need overlayfs and a temporary directory that is not on overlayfs. "
        if system_directories == SystemDirectories.COPY_ON_WRITE
        else ""
    )
    raise SandboxUnavailable(
        "No bwrap can sandbox commands on this host. A root process needs CAP_SYS_ADMIN, CAP_NET_ADMIN, no "
        "AppArmor confinement and a seccomp filter that allows pivot_root; another user needs unprivileged "
        "user namespaces. " + overlay + "; ".join(errors)
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


def _prepare_root(root: Path, workdir: PurePosixPath | None) -> None:
    root.chmod(ROOT_MODE)
    for directory in (*PRIVATE_SYSTEM_DIRECTORIES, "/opt"):
        (root / directory.lstrip("/")).mkdir(parents=True)
    for configuration in RUNTIME_CONFIGURATION:
        source = Path(configuration)
        if source.exists():
            _copy(source, root / configuration.lstrip("/"))
    # Any user may write /tmp and HOME, as in a container image.
    for directory in (SCRATCH, HOME):
        path = root / directory.relative_to("/")
        path.mkdir(parents=True)
        path.chmod(STICKY_WORLD_WRITABLE)
    if workdir is not None:
        (root / workdir.relative_to("/")).mkdir(parents=True, exist_ok=True)


def _archive(source: Path, name: str) -> bytes:
    """A tar stream of ``source`` under ``name``; ``name`` "." takes a directory's contents.

    As ``_copy`` does, a ``source`` that is a symlink contributes what it leads to, and symlinks inside a
    directory stay symlinks.
    """
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        archive.add(os.path.realpath(source), arcname=name)
    return stream.getvalue()


def _guest_member(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
    """Extract a member of a stream that a machine wrote, keeping its symlinks but nothing that reads the host.

    A hard link must lead to a file of the stream, and a device would let a copy read the host's disk.
    """
    if member.islnk() or member.isdev():
        return tarfile.data_filter(member, path)
    return tarfile.tar_filter(member, path)


def _unpack(stream: bytes, target: Path) -> None:
    """Write a stream from ``ARCHIVE_SCRIPT`` to ``target``: a directory's contents or a single file.

    The machine may write any stream, so a first member that would lead the copy out of the staging
    directory, such as a symlink to a host path, is refused; symlinks inside a directory are copied as links.
    """
    with tempfile.TemporaryDirectory() as staging, tarfile.open(fileobj=io.BytesIO(stream)) as archive:
        members = archive.getmembers()
        if not members or not (members[0].isdir() if members[0].name == "." else members[0].isfile()):
            raise RuntimeError("The machine's archive does not start with a regular file or directory")
        archive.extractall(staging, filter=_guest_member)
        _copy(Path(staging) if members[0].name == "." else Path(staging, members[0].name), target)


class LocalMachine:
    """Commands in bubblewrap sandboxes over ``root``, a host directory that only this machine uses.

    Each command gets new PID, IPC and UTS namespaces, and a network namespace with only loopback under
    ``NetworkPolicy.DENY``. Files persist in ``root`` between commands, and, with ``layers``, so do writes
    under the overlaid system directories; processes end with their command.
    """

    def __init__(
        self,
        spec: MachineSpec,
        *,
        bwrap: Path,
        root: Path,
        layers: Path | None,
        mounts: tuple[Mount, ...],
        environment: dict[str, str],
    ):
        self.spec = spec
        self.root = root
        self.layers = layers
        self.mounts = mounts
        self._bwrap = bwrap
        self._environment = environment
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")

    def _mounted(self, path: PurePosixPath) -> Mount | None:
        return next((mount for mount in self.mounts if path.is_relative_to(mount.guest)), None)

    def _overlaid(self, path: PurePosixPath) -> bool:
        """Whether ``path`` lies in or contains a directory that this machine overlays."""
        return self.layers is not None and any(
            path.is_relative_to(directory) or PurePosixPath(directory).is_relative_to(path)
            for directory in OVERLAID_DIRECTORIES
        )

    def _host_path(self, path: PurePosixPath) -> Path:
        """Where the host keeps the file that commands see at ``path``.

        A command may plant a symlink in its root that points outside it; a transfer must not follow one.
        """
        mount = self._mounted(path)
        if mount is not None:
            return mount.host / path.relative_to(mount.guest)
        if _within(path, SYSTEM_DIRECTORIES) and not _within(path, PRIVATE_SYSTEM_DIRECTORIES):
            return Path(path)
        host = self.root / path.relative_to("/")
        if not Path(os.path.realpath(host)).is_relative_to(self.root):
            raise RuntimeError(f"{path} leaves the machine root through a symlink")
        return host

    async def run(self, command: Command) -> Result:
        return await self._execute(command, {**self._environment, **self.spec.env, **command.env})

    async def _control(
        self,
        script: str,
        path: PurePosixPath,
        *,
        stdin: bytes = b"",
        output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    ) -> Result:
        """Run one of the backend's own scripts on ``path``, in ``CONTROL_ENVIRONMENT``."""
        command = Command(
            ("/bin/sh", "-c", script, "sh", str(path)),
            cwd="/",
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        return await self._execute(command, CONTROL_ENVIRONMENT)

    async def _execute(self, command: Command, environment: dict[str, str]) -> Result:
        self._check_open()
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        sandbox = _sandbox_argv(
            self._bwrap,
            root=self.root,
            layers=self.layers,
            mounts=self.mounts,
            network=self.spec.network,
            account=_command_account(command.user),
            environment=environment,
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
            # Task variables, including loader options, take effect only after bwrap builds the sandbox.
            env=CONTROL_ENVIRONMENT,
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
        if self._overlaid(path):
            # Unpacking inside the sandbox writes through the overlay, as a command's own write would. A
            # directory uploaded to / may hold files for /opt, so it takes this path too.
            name = "." if source.is_dir() else path.name
            stream = await asyncio.to_thread(_archive, source, name)
            directory = path if source.is_dir() else path.parent
            result = await self._control(EXTRACT_SCRIPT, directory, stdin=stream)
            if result.exit_code != 0:
                raise RuntimeError(f"Cannot upload to {target}: {result.stderr.decode(errors='replace').strip()}")
            return
        if self._mounted(path) is not None or _within(path, SYSTEM_DIRECTORIES):
            raise UnsupportedMachineSpec(f"The local backend's {target} is a read-only host path")
        await asyncio.to_thread(_copy, source, self._host_path(path))

    async def download(self, source: str, target: Path) -> None:
        self._check_open()
        guest = _absolute_path(source)
        if self._overlaid(guest) and not _within(guest, OVERLAID_DIRECTORIES):
            # Through the sandbox, / holds the host's whole system as well as /proc and /dev.
            raise UnsupportedMachineSpec(
                f"A copy-on-write machine cannot download {source}, which contains its system directories"
            )
        if self._overlaid(guest) or _within(guest, PRIVATE_SYSTEM_DIRECTORIES):
            result = await self._control(ARCHIVE_SCRIPT, guest, output_limit_bytes=ARCHIVE_LIMIT_BYTES)
            if result.exit_code != 0:
                raise RuntimeError(result.stderr.decode(errors="replace").strip())
            if result.stdout_truncated:
                raise RuntimeError(f"{source} exceeds the {ARCHIVE_LIMIT_BYTES}-byte download limit")
            await asyncio.to_thread(_unpack, result.stdout, target)
            return
        path = self._host_path(guest)
        if not path.exists():
            raise RuntimeError(f"No such file or directory: {source}")
        await asyncio.to_thread(_copy, path, target)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(_remove, self.root)
        if self.layers is not None:
            await asyncio.to_thread(_remove, self.layers)


def _check_mounts(mounts: tuple[Mount, ...]) -> None:
    """Every mount names an existing host path and an absolute guest path outside system directories and other mounts.

    bwrap creates a mount point by mkdir, which fails inside a directory it has already bound read-only.
    """
    missing = [str(mount.host) for mount in mounts if not mount.host.exists()]
    if missing:
        raise ValueError(f"Mounted host paths do not exist: {', '.join(missing)}")
    for mount in mounts:
        if not mount.guest.is_absolute():
            raise UnsupportedMachineSpec(f"The local backend mounts only at absolute guest paths: {mount.guest}")
        if _within(mount.guest, SYSTEM_DIRECTORIES):
            raise UnsupportedMachineSpec(f"The local backend cannot mount inside a host system directory: {mount.guest}")
        if any(other is not mount and mount.guest.is_relative_to(other.guest) for other in mounts):
            raise UnsupportedMachineSpec(f"The local backend's mount {mount.guest} lies inside another mount")


def _check_spec(spec: MachineSpec) -> HostImage | PackageSpec:
    if not isinstance(spec.source, (HostImage, PackageSpec)):
        raise UnsupportedMachineSpec(
            f"The local backend requires HostImage or PackageSpec, not {type(spec.source).__name__}"
        )
    if spec.cpus is not None or spec.storage_mb is not None or spec.gpus or spec.memory_mb is not None:
        raise UnsupportedMachineSpec("The local backend does not provide CPU, memory, storage, or GPU allocations")
    return spec.source


def check_local_environment(requirements: ExecutionRequirements, spec: MachineSpec) -> None:
    """Check local execution support without acquiring a bubblewrap sandbox."""
    _check_spec(spec)
    check_execution_requirements(
        requirements, spec, semantics=CommandSemantics.LINUX_PROCESS, capabilities=LINUX_PROCESS_CAPABILITIES
    )


class LocalMachineFactory:
    """Run trusted commands on this host in bubblewrap sandboxes, any number of machines at a time.

    A machine's commands see the host's programs and private runtime configuration as
    ``system_directories`` says: read-only by default, or writable per machine. They see each directory
    in ``read_only`` at its own path, read-only, and each ``Mount`` of a ``HostImage`` at its guest path,
    read-only; a guest path inside a system directory is refused. Every other path, including ``/tmp``
    and ``HOME``, lies in a root directory of the machine's own that is removed by
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
        package_cache: Path | None = None,
        python_version: str = DEFAULT_PYTHON_VERSION,
        system_directories: SystemDirectories = SystemDirectories.READ_ONLY,
    ):
        self.read_only = tuple(directory.absolute() for directory in read_only)
        missing = [str(directory) for directory in self.read_only if not directory.is_dir()]
        if missing:
            raise ValueError(f"Read-only directories do not exist: {', '.join(missing)}")
        self.bin_dirs = tuple(directory.absolute() for directory in bin_dirs)
        self.hash_seed = hash_seed
        self.package_cache = package_cache if package_cache is not None else Path.home() / ".cache/shellbox/python"
        self.python_version = python_version
        self.system_directories = system_directories
        self.bwrap = _working_bwrap(_bwrap_candidates(bwrap), system_directories)
        logger.info("Local backend sandboxes commands with %s, system directories %s", self.bwrap, system_directories)

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> LocalMachine:
        source = _check_spec(spec)
        host = source if isinstance(source, HostImage) else HostImage()
        if isinstance(source, PackageSpec) and source.pypi_packages is not None:
            installed = await asyncio.to_thread(
                resolve_python_environment, self.package_cache, source.pypi_packages, self.python_version
            )
            host = HostImage(read_only=(installed.root,), bin_dirs=(installed.bin_dir,))
        # A read-only directory is a mount at its own path; the factory's and the image's may name the same one.
        read_only = dict.fromkeys((*self.read_only, *(directory.absolute() for directory in host.read_only)))
        mounts = tuple(
            dict.fromkeys(
                (
                    *(Mount(directory, PurePosixPath(directory)) for directory in read_only),
                    *(Mount(mount.host.absolute(), mount.guest) for mount in host.mounts),
                )
            )
        )
        _check_mounts(mounts)
        workdir = _absolute_path(spec.workdir or "/")
        copy_on_write = self.system_directories == SystemDirectories.COPY_ON_WRITE
        overlaid_workdir = copy_on_write and _within(workdir, OVERLAID_DIRECTORIES)
        if (_within(workdir, SYSTEM_DIRECTORIES) and not overlaid_workdir) or any(
            workdir.is_relative_to(mount.guest) for mount in mounts
        ):
            raise UnsupportedMachineSpec(f"The local backend's workdir {spec.workdir} is a read-only host path")
        # A workdir in an overlaid directory is made by a command once the machine runs, so it lands in the layer.
        root, layers = _machine_directories(None if overlaid_workdir else workdir, copy_on_write=copy_on_write)
        environment = {
            "PATH": os.pathsep.join(
                [
                    *(str(directory.absolute()) for directory in host.bin_dirs),
                    *map(str, self.bin_dirs),
                    DEFAULT_PATH,
                ]
            ),
            "HOME": str(HOME),
            "LANG": "C.UTF-8",
        }
        if self.hash_seed is not None:
            environment["PYTHONHASHSEED"] = self.hash_seed
        machine = LocalMachine(spec, bwrap=self.bwrap, root=root, layers=layers, mounts=mounts, environment=environment)
        tools = source.tools if isinstance(source, PackageSpec) else ()
        try:
            await _start(machine, workdir if overlaid_workdir else None, tools)
        except BaseException:
            await machine.close()
            raise
        return machine


def _machine_directories(workdir: PurePosixPath | None, *, copy_on_write: bool) -> tuple[Path, Path | None]:
    """A new machine's root, with ``workdir`` made in it, and its layers when its system directories are overlaid."""
    root = Path(tempfile.mkdtemp(prefix="shellbox-local-"))
    layers = None
    try:
        _prepare_root(root, workdir)
        if copy_on_write:
            layers = Path(tempfile.mkdtemp(prefix="shellbox-layers-"))
            _prepare_layers(layers)
    except BaseException:
        _remove(root)
        if layers is not None:
            _remove(layers)
        raise
    return root, layers


async def _start(machine: LocalMachine, overlaid_workdir: PurePosixPath | None, tools: tuple[str, ...]) -> None:
    """Make a workdir that lies in the machine's layer, then check that each requested tool runs."""
    if overlaid_workdir is not None:
        made = await machine._control('exec mkdir -p -- "$1"', overlaid_workdir)
        if made.exit_code != 0:
            raise RuntimeError(f"Cannot create workdir {overlaid_workdir}: {made.stderr.decode(errors='replace')}")
    if not tools:
        return
    result = await machine.run(
        Command(
            (
                "/bin/sh",
                "-c",
                'for tool do path=$(command -v "$tool") && test -x "$path" || exit; done',
                "sh",
                *tools,
            )
        )
    )
    if result.exit_code != 0:
        raise UnsupportedMachineSpec("Requested tools must be executable in the sandbox PATH")
