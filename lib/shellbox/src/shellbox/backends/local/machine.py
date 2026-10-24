# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent bubblewrap sandboxes over host programs or prepared OCI filesystems.

Each machine's commands see the host's programs, read-only or copy-on-write, and private runtime
configuration on top of a private root directory, which holds paths from ``/app`` and ``/tmp`` to ``HOME``.
"""

import asyncio
import fcntl
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
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace
from enum import StrEnum
from pathlib import Path, PurePosixPath

from shellbox.backends.local import launch
from shellbox.backends.local.images import PreparedLocalImage, prepare_local_image
from shellbox.backends.local.python_environment import DEFAULT_PYTHON_VERSION, resolve_python_environment
from shellbox.backends.local.transfers import unpack_local_archive
from shellbox.image import PreparedImage, RegistryImage, process_image_cache
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
    MachineTerminated,
    Mount,
    NetworkPolicy,
    PackageSpec,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
)
from shellbox.service import ServiceGroup, ServiceGroupSpec

logger = logging.getLogger(__name__)

DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
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
NS_GET_USERNS = 0xB701


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
) -> list[str]:
    """The bwrap invocation that runs a program over ``root`` with the system directories and ``mounts``.

    The system directories are read-only, or, with ``layers``, overlaid by the machine's layer of each.
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


def _pin_init(pid: int, namespace: int) -> int:
    """Pin bwrap's init, verifying its identity through an anchored proc directory."""
    directory = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    descriptor = None
    try:
        descriptor = os.pidfd_open(pid)
        process_namespace = os.open("ns/pid", os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory)
        try:
            if os.fstat(process_namespace).st_ino != namespace:
                raise SandboxUnavailable("Bubblewrap init namespace identity changed")
        finally:
            os.close(process_namespace)
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        os.close(directory)


class LocalMachine:
    """A persistent bubblewrap member, with commands entering pinned namespaces."""

    def __init__(
        self,
        spec: MachineSpec,
        *,
        bwrap: Path,
        root: Path,
        layers: Path | None,
        mounts: tuple[Mount, ...],
        environment: dict[str, str],
        image: PreparedLocalImage | None = None,
        bundle: Path | None = None,
    ):
        self.spec = spec
        self.root = root
        self.layers = layers
        self.mounts = mounts
        self.image = image
        self._bundle = bundle
        self._bwrap = bwrap
        self._environment = environment
        self._closed = False
        self._closing: asyncio.Task[None] | None = None
        self._group_close: Callable[[], Awaitable[None]] | None = None
        self._keeper: asyncio.subprocess.Process | None = None
        self._keeper_pidfd: int | None = None
        self._init_pidfd: int | None = None
        self._namespaces: tuple[int, ...] = ()
        self._owned_namespaces: tuple[int, ...] = ()
        self._network_namespaces: tuple[int, ...] = ()
        self._root_fd: int | None = None
        self._commands: set[asyncio.subprocess.Process] = set()
        self._watcher: asyncio.Task[None] | None = None
        self._expiry: asyncio.Task[None] | None = None

    def _check_open(self) -> None:
        if self._closed:
            raise MachineTerminated("Machine is closed")

    async def _start_keeper(self, peer: "LocalMachine | None" = None) -> None:
        if self.image is None:
            arguments = _sandbox_argv(
                self._bwrap,
                root=self.root,
                layers=self.layers,
                mounts=self.mounts,
                network=self.spec.network if peer is None else NetworkPolicy.ALLOW,
                account=None,
            )
        else:
            arguments = [
                str(self._bwrap),
                "--unshare-user",
                "--uid",
                "0",
                "--gid",
                "0",
                "--unshare-ipc",
                "--unshare-pid",
                "--unshare-uts",
                "--new-session",
                "--die-with-parent",
                "--cap-drop",
                "ALL",
                "--bind",
                str(self.root),
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/dev/shm",
            ]
            if peer is None and self.spec.network is NetworkPolicy.DENY:
                arguments.append("--unshare-net")
        # A new user namespace is needed for each independent mount namespace,
        # including members that inherit a peer's network namespace.
        if os.geteuid() != 0 and "--unshare-user" not in arguments:
            arguments.insert(1, "--unshare-user")
        if self.image is None and os.geteuid() != 0:
            arguments[1:1] = ["--uid", str(os.geteuid()), "--gid", str(os.getegid())]
        block_read, block_write = os.pipe()
        allocation = asyncio.create_task(self._allocate_keeper(arguments, peer, block_read, block_write))
        try:
            await asyncio.shield(allocation)
        except asyncio.CancelledError:
            # Namespace setup owns processes before create_subprocess_exec has
            # returned their handles. Finish bounded setup before reclaiming it.
            try:
                await asyncio.shield(allocation)
            finally:
                await asyncio.shield(self._close_local())
            raise
        finally:
            os.close(block_read)
            os.close(block_write)
        assert self._keeper_pidfd is not None
        self._watcher = asyncio.create_task(self._watch_keeper(os.dup(self._keeper_pidfd)))
        self._watcher.add_done_callback(_log_task_failure)

    async def _allocate_keeper(
        self, arguments: list[str], peer: "LocalMachine | None", block_read: int, block_write: int
    ) -> None:
        with tempfile.TemporaryFile() as information:
            arguments += [
                "--block-fd",
                str(block_read),
                "--info-fd",
                str(information.fileno()),
                "--clearenv",
                "--setenv",
                "PATH",
                DEFAULT_PATH,
                "/bin/sh",
                "-c",
                "exec sleep 2147483647",
            ]
            prefix = () if peer is None else peer._network_namespaces
            config = {"namespaces": prefix, "program": arguments}
            with tempfile.TemporaryFile() as configuration:
                configuration.write(json.dumps(config).encode())
                configuration.seek(0)
                self._keeper = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    LAUNCH_SOURCE,
                    str(configuration.fileno()),
                    pass_fds=(configuration.fileno(), information.fileno(), block_read, *prefix),
                    env=CONTROL_ENVIRONMENT,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                self._keeper_pidfd = os.pidfd_open(self._keeper.pid)
            async with asyncio.timeout(PROBE_TIMEOUT):
                while True:
                    information.seek(0)
                    payload = information.read()
                    if payload:
                        try:
                            status = json.loads(payload)
                            pid = status["child-pid"]
                            if self._init_pidfd is None:
                                self._init_pidfd = _pin_init(pid, status["pid-namespace"])
                            # info-fd is written before bwrap finishes mounting
                            # and pivoting. Pinning root earlier would capture
                            # the host root rather than the guest root.
                            expected = self.root.stat()
                            actual = os.stat(f"/proc/{pid}/root")
                            expected_uid = 0 if self.image is not None else os.geteuid()
                            mapped = Path(f"/proc/{pid}/uid_map").read_text().splitlines()
                            final_user = any(int(row.split()[0]) == expected_uid for row in mapped)
                            if final_user and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
                                break
                        except json.JSONDecodeError:
                            pass  # bwrap may still be writing its single status document.
                    if self._keeper.returncode is not None:
                        assert self._keeper.stderr is not None
                        error = (await self._keeper.stderr.read()).decode(errors="replace")
                        raise SandboxUnavailable(f"Persistent bubblewrap keeper failed: {error.strip()}")
                    await asyncio.sleep(0.01)
            # The keeper cannot be reused after exit: the opened descriptors pin
            # this namespace and root even if the kernel later reuses its PID.
            descriptors = []
            try:
                for namespace in ("user", "net", "ipc", "uts", "mnt", "pid"):
                    descriptors.append(os.open(f"/proc/{pid}/ns/{namespace}", os.O_RDONLY | os.O_CLOEXEC))
                    expected_inode = json.loads(payload).get(f"{namespace}-namespace")
                    if expected_inode is not None and os.fstat(descriptors[-1]).st_ino != expected_inode:
                        raise SandboxUnavailable("Bubblewrap startup namespace identity changed")
                self._root_fd = os.open(f"/proc/{pid}/root", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                owner = fcntl.ioctl(descriptors[4], NS_GET_USERNS)
                os.set_inheritable(owner, False)
            except BaseException:
                for descriptor in descriptors:
                    os.close(descriptor)
                raise
            self._owned_namespaces = (owner, *descriptors)
            # bwrap drops capabilities by entering a final child user namespace.
            # Join mount/network owners first, then that restricted namespace.
            self._namespaces = (*prefix, owner, *descriptors[1:], descriptors[0])
            self._network_namespaces = (*prefix, owner, descriptors[1])
            os.write(block_write, b"1")

    async def _watch_keeper(self, descriptor: int) -> None:
        # Process.wait also waits for stderr EOF. Watch process death directly:
        # an orphaned init can keep that pipe open until our cleanup kills it.
        loop = asyncio.get_running_loop()
        exited = loop.create_future()

        def ready() -> None:
            loop.remove_reader(descriptor)
            exited.set_result(None)

        loop.add_reader(descriptor, ready)
        try:
            await exited
        finally:
            loop.remove_reader(descriptor)
            os.close(descriptor)
        if not self._closed:
            await self.close()

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
        return await self._execute(
            Command(
                ("/bin/sh", "-c", script, "sh", str(path)), cwd="/", stdin=stdin, output_limit_bytes=output_limit_bytes
            ),
            CONTROL_ENVIRONMENT,
        )

    async def _execute(self, command: Command, environment: dict[str, str]) -> Result:
        self._check_open()
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        if self.image is not None:
            if command.user not in (None, "0", "0:0"):
                raise UnsupportedMachineSpec("Prepared local images currently support only root USER")
            account = None
        else:
            selected = _command_account(command.user)
            account = None if selected is None else (selected.pw_uid, selected.pw_gid)
        config = {
            "namespaces": self._namespaces,
            "root": self._root_fd,
            "limits": _resource_limits(command.timeout),
            "account": account,
            "script": RUN_SCRIPT,
            "cwd": command.cwd or self.spec.workdir or "/",
            "argv": command.argv,
            "environment": environment,
        }
        if any("\0" in argument for argument in (*command.argv, *environment, *environment.values())):
            raise ValueError("Sandbox arguments cannot contain NUL bytes")
        # Anonymous output files let a successful foreground command finish even
        # when its background services keep stdout open. RLIMIT_FSIZE bounds writers.
        with (
            tempfile.TemporaryFile() as configuration,
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
            tempfile.TemporaryFile() as stdin_file,
        ):
            configuration.write(json.dumps(config).encode())
            configuration.seek(0)
            stdin_file.write(command.stdin)
            stdin_file.seek(0)
            assert self._root_fd is not None
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                "-c",
                LAUNCH_SOURCE,
                str(configuration.fileno()),
                pass_fds=(configuration.fileno(), self._root_fd, *dict.fromkeys(self._namespaces)),
                env=CONTROL_ENVIRONMENT,
                stdin=stdin_file,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            self._commands.add(process)
            try:
                async with asyncio.timeout(command.timeout):
                    await process.wait()
            except BaseException as interruption:
                if not self._closed:
                    await asyncio.shield(self.close())
                await process.wait()
                if isinstance(interruption, TimeoutError):
                    return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
                raise
            finally:
                self._commands.discard(process)
            self._check_open()
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(command.output_limit_bytes + 1)
            stderr = stderr_file.read(command.output_limit_bytes + 1)
        assert process.returncode is not None
        code = process.returncode
        return Result(
            128 - code if code < 0 else code,
            stdout[: command.output_limit_bytes],
            stderr[: command.output_limit_bytes],
            len(stdout) > command.output_limit_bytes,
            len(stderr) > command.output_limit_bytes,
            ExitReason.EXITED,
        )

    async def upload(self, source: Path, target: str) -> None:
        self._check_open()
        path = _absolute_path(target)
        if self.image is None:
            writable_system = self.layers is not None and _within(path, OVERLAID_DIRECTORIES)
            if (_within(path, SYSTEM_DIRECTORIES) and not writable_system) or any(
                path.is_relative_to(mount.guest) for mount in self.mounts
            ):
                raise UnsupportedMachineSpec(f"The local backend's {target} is a read-only host path")
        name = "." if source.is_dir() else path.name
        stream = await asyncio.to_thread(_archive, source, name)
        result = await self._control(EXTRACT_SCRIPT, path if source.is_dir() else path.parent, stdin=stream)
        if result.exit_code != 0:
            raise RuntimeError(f"Cannot upload to {target}: {result.stderr.decode(errors='replace').strip()}")

    async def download(self, source: str, target: Path) -> None:
        self._check_open()
        path = _absolute_path(source)
        if self.image is None and path == PurePosixPath("/"):
            raise UnsupportedMachineSpec("A host-program machine cannot download its entire system root")
        result = await self._control(ARCHIVE_SCRIPT, path, output_limit_bytes=ARCHIVE_LIMIT_BYTES)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
        if result.stdout_truncated:
            raise RuntimeError(f"{source} exceeds the {ARCHIVE_LIMIT_BYTES}-byte download limit")
        await asyncio.to_thread(unpack_local_archive, result.stdout, target)

    async def close(self) -> None:
        if self._group_close is not None:
            await self._group_close()
            return
        await self._close_local()

    async def _close_local(self) -> None:
        if self._closing is None:
            self._closed = True
            self._closing = asyncio.create_task(self._destroy())
        await asyncio.shield(self._closing)

    async def _destroy(self) -> None:
        if self._expiry is not None:
            self._expiry.cancel()
        # Kill the verified, pinned guest init before its host launcher. An
        # immediate close can otherwise race bwrap's parent-death setup and
        # leave the init holding the launcher's stderr pipe open.
        for descriptor in (self._init_pidfd, self._keeper_pidfd):
            if descriptor is None:
                continue
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._keeper is not None:
            await self._keeper.wait()
        if self._commands:
            await asyncio.gather(*(process.wait() for process in tuple(self._commands)))
        if self._keeper_pidfd is not None:
            os.close(self._keeper_pidfd)
        if self._init_pidfd is not None:
            os.close(self._init_pidfd)
        # Prefix descriptors belong to the first member and are closed by it.
        for descriptor in self._owned_namespaces:
            os.close(descriptor)
        if self._root_fd is not None:
            os.close(self._root_fd)
        await asyncio.to_thread(_remove, self._bundle or self.root)
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


def _check_spec(spec: MachineSpec) -> HostImage | PackageSpec | PreparedImage | RegistryImage:
    if not isinstance(spec.source, (HostImage, PackageSpec, PreparedImage, RegistryImage)):
        raise UnsupportedMachineSpec(
            "The local backend requires HostImage, PackageSpec, or a prepared/pinned OCI image, "
            f"not {type(spec.source).__name__}"
        )
    if spec.cpus is not None or spec.storage_mb is not None or spec.gpus or spec.memory_mb is not None:
        raise UnsupportedMachineSpec("The local backend does not provide CPU, memory, storage, or GPU allocations")
    if (
        isinstance(spec.source, RegistryImage)
        and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", spec.source.reference) is None
    ):
        raise UnsupportedMachineSpec("Local registry images must be pinned by sha256 digest")
    return spec.source


def check_local_environment(requirements: ExecutionRequirements, spec: MachineSpec) -> None:
    """Check local execution support without acquiring a bubblewrap sandbox."""
    source = _check_spec(spec)
    origin = source.source if isinstance(source, PreparedImage) else source
    reference = origin.reference if isinstance(origin, RegistryImage) else None
    check_execution_requirements(
        requirements,
        spec,
        semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=LINUX_PROCESS_CAPABILITIES,
        image_reference=reference,
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

    Prepared images use private rootless umoci bundles, without host program mounts. Registry inputs
    must be digest pinned. Only numeric root image users are supported. ``ttl`` bounds the lifetime of
    a machine or group; a command timeout or cancellation also closes its entire machine or group.
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
        ttl: float | None = None,
        image_cache: Path | None = None,
        authfile: Path | None = None,
        policy: Path | None = None,
        umoci: Path = Path("umoci"),
    ):
        if ttl is not None and ttl <= 0:
            raise ValueError("Local machine TTL must be positive")
        self.ttl = ttl
        self.image_cache = process_image_cache(image_cache or Path.home() / ".cache/shellbox/images", authfile, policy)
        self.umoci = umoci
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
        machine = await self._create(spec)
        if self.ttl is not None:
            machine._expiry = asyncio.create_task(_expire(machine.close, self.ttl))
        return machine

    async def create_group(self, spec: ServiceGroupSpec) -> ServiceGroup:
        """Start separate members sharing only a private loopback network."""
        for member in (spec.task, *(service.machine for service in spec.services)):
            _check_spec(member)
        if any(re.fullmatch(r"[a-z][a-z0-9-]{0,62}", service.name) is None for service in spec.services):
            raise UnsupportedMachineSpec("Local service names must be lowercase DNS labels")
        members: list[LocalMachine] = []
        monitors: list[asyncio.Task[Result]] = []
        expiry: asyncio.Task[None] | None = None
        closing_tasks: set[asyncio.Task[None]] = set()

        async def close() -> None:
            if expiry is not None:
                expiry.cancel()
            await asyncio.gather(*(member._close_local() for member in members))
            for monitor in monitors:
                if not monitor.done():
                    monitor.cancel()
            await asyncio.gather(*monitors, return_exceptions=True)

        try:
            aliases = "127.0.0.1 localhost " + " ".join(service.name for service in spec.services) + "\n"
            task = await self._create(spec.task, aliases=aliases)
            members.append(task)
            services = {}
            for service in spec.services:
                member = await self._create(service.machine, peer=task, aliases=aliases)
                members.append(member)
                services[service.name] = member
            group = ServiceGroup(task, services, spec, close)
            for member in members:
                member._group_close = group.close
            for service in spec.services:
                member = services[service.name]
                image = member.image
                entrypoint = (
                    service.entrypoint if service.entrypoint is not None else (() if image is None else image.entrypoint)
                )
                arguments = (
                    service.arguments if service.arguments is not None else (() if image is None else image.arguments)
                )
                if not entrypoint and not arguments:
                    raise UnsupportedMachineSpec(f"Service {service.name} has no startup command")
                monitor = asyncio.create_task(member.run(Command((*entrypoint, *arguments))))
                monitor.add_done_callback(lambda finished: _service_finished(group, finished, closing_tasks))
                monitors.append(monitor)
            if self.ttl is not None:
                expiry = asyncio.create_task(_expire(group.close, self.ttl))
            return group
        except BaseException:
            await asyncio.shield(asyncio.gather(*(member._close_local() for member in members)))
            raise

    async def _create(
        self, spec: MachineSpec, peer: LocalMachine | None = None, aliases: str | None = None
    ) -> LocalMachine:
        source = _check_spec(spec)
        if isinstance(source, (PreparedImage, RegistryImage)):
            return await self._create_image(spec, source, peer, aliases)

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
        if aliases is not None:
            (root / "etc/hosts").write_text(aliases)
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
            await machine._start_keeper(peer)
            await _start(machine, workdir if overlaid_workdir else None, tools)
        except BaseException:
            await machine.close()
            raise
        return machine

    async def _create_image(
        self, spec: MachineSpec, source: PreparedImage | RegistryImage, peer: LocalMachine | None, aliases: str | None
    ) -> LocalMachine:
        if self.read_only or self.bin_dirs or self.system_directories is not SystemDirectories.READ_ONLY:
            raise UnsupportedMachineSpec("Prepared local images cannot use host program mounts or overlays")
        parent = Path(tempfile.mkdtemp(prefix="shellbox-image-"))
        bundle = parent / "bundle"
        machine = None

        async def prepare() -> PreparedLocalImage:
            prepared = (
                source
                if isinstance(source, PreparedImage)
                else await asyncio.to_thread(self.image_cache.prepare, source)
            )
            return await asyncio.to_thread(prepare_local_image, prepared, bundle, umoci=self.umoci)

        preparation = asyncio.create_task(prepare())
        try:
            image = await asyncio.shield(preparation)
            if image.user not in (None, "", "0", "0:0"):
                raise UnsupportedMachineSpec("Prepared local images currently support only root USER")
            effective = replace(spec, workdir=spec.workdir or image.workdir)
            machine = LocalMachine(
                effective,
                bwrap=self.bwrap,
                root=image.root,
                layers=None,
                mounts=(),
                environment={"PATH": DEFAULT_PATH, "HOME": "/root", **image.environment},
                image=image,
                bundle=parent,
            )
            await machine._start_keeper(peer)
            if aliases is not None:
                hosts = await machine._control("cat > /etc/hosts", PurePosixPath("/"), stdin=aliases.encode())
                if hosts.exit_code != 0:
                    raise RuntimeError(f"Cannot prepare group aliases: {hosts.stderr.decode(errors='replace')}")
            result = await machine._control('mkdir -p -- "$1"', _absolute_path(effective.workdir))
            if result.exit_code != 0:
                raise RuntimeError(f"Cannot create image workdir: {result.stderr.decode(errors='replace')}")
            return machine
        except BaseException:
            # A cancelled to_thread still writes its bundle. Reclaim it only
            # after the trusted unpacker has stopped.
            try:
                await asyncio.shield(preparation)
            finally:
                if machine is not None:
                    await machine._close_local()
                else:
                    await asyncio.to_thread(_remove, parent)
            raise


async def _expire(close: Callable[[], Awaitable[None]], ttl: float) -> None:
    await asyncio.sleep(ttl)
    await close()


def _log_task_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        if isinstance(error, MachineTerminated):
            return  # Closing a group interrupts its still-running service monitors.
        logger.error("Local sandbox lifecycle failed", exc_info=(type(error), error, error.__traceback__))


def _service_finished(group: ServiceGroup, task: asyncio.Task[Result], closing_tasks: set[asyncio.Task[None]]) -> None:
    _log_task_failure(task)
    reclaim = asyncio.create_task(group.close())
    closing_tasks.add(reclaim)
    reclaim.add_done_callback(closing_tasks.discard)
    reclaim.add_done_callback(_log_task_failure)


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
