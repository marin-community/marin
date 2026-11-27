# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build private host filesystem layouts and bubblewrap arguments."""

import io
import os
import pwd
import re
import resource
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from shellbox.backends.local import launch
from shellbox.backends.local.disk import LocalDisk
from shellbox.machine import (
    Mount,
    NetworkPolicy,
    UnsupportedMachineSpec,
)

DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Resolve the program through sh, as the container backends do: a missing or unexecutable program
# or working directory becomes a failed command (127 or 126 for the program) rather than an exception.
RUN_SCRIPT = 'cd "$1" || exit; shift; exec "$@"'
LAUNCH_SOURCE = Path(launch.__file__).read_text()
KEEPER_PAYLOAD = """\
read -r ready && [ "$ready" = shellbox-ready ] || exit 1
command -v sleep >/dev/null || { echo "The machine's keeper needs sleep" >&2; exit 1; }
while :
do
    sleep 2147483647 &
    keeper=$!
    wait "$keeper" 2>/dev/null
    while kill -0 "$keeper" 2>/dev/null
    do
        sleep 1
        wait "$keeper" 2>/dev/null
    done
done
"""
"""The guest PID namespace's init, which ``--as-pid-1`` makes PID 1.

The init owns the machine's lifecycle and reaps its orphans. Guest processes
whose parents exit reparent to PID 1, so only the init can collect them, and
guest code observes a dead daemon through ``kill -0`` and ``/proc`` just as it
does on the Docker backend's ``--init``. The init also keeps the sleeper
``sleep 2147483647`` running: grading quiesces spare the machine's PID 1 and
spare exactly that argv, and an init that would exit when the sleeper dies
would close the machine mid-grading.
"""
NPROC_HEADROOM = 256
"""Tasks a command may add beyond the host's task count when it starts; RLIMIT_NPROC counts all of a user's tasks."""
FILE_SIZE_LIMIT = 1 << 30
PROCESS_STOP_TIMEOUT = 1
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
CONTROL_ENVIRONMENT = MappingProxyType({"PATH": "/usr/bin:/bin", "HOME": str(HOME), "LANG": "C.UTF-8"})
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
    """Remove a private tree after its guest processes have been reclaimed."""
    try:
        shutil.rmtree(path)
    except PermissionError:
        descriptor = os.open(path, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            _unlock_directories(descriptor)
        finally:
            os.close(descriptor)
        shutil.rmtree(path)


def _unlock_directories(descriptor: int) -> None:
    # O_PATH opens even mode000 directories. The proc link refers to this pinned
    # directory, so neither chmod nor traversal can follow a guest symlink out.
    pinned = f"/proc/self/fd/{descriptor}"
    os.chmod(pinned, 0o700)
    with os.scandir(pinned) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            child = os.open(entry.name, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor)
            try:
                _unlock_directories(child)
            finally:
                os.close(child)


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
) -> list[str]:
    """The bwrap invocation that runs the machine's keeper program over ``root``.

    The system directories are read-only, or, with ``layers``, overlaid by the machine's layer of each.
    The program runs without capabilities; as root, bwrap needs no user namespace for its own.
    ``--as-pid-1`` makes the payload the guest PID namespace's PID 1: the machine's lifecycle
    keeper must be PID 1, because grading quiesces kill every process outside their ancestry
    except PID 1, and a keeper below PID 1 dies with them, closing the machine mid-grading.
    The payload (``KEEPER_PAYLOAD``) is an init that keeps the sleeper ``sleep 2147483647``
    running as its child, for graders that spare machines by that exact argv, and reaps the
    guest's orphaned processes, which reparent to PID 1.
    """
    argv = [str(bwrap), "--unshare-ipc", "--unshare-pid", "--unshare-uts", "--unshare-cgroup-try"]
    if network == NetworkPolicy.DENY:
        argv.append("--unshare-net")
    argv += ["--as-pid-1", "--new-session", "--die-with-parent", "--cap-drop", "ALL"]
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
    return argv


def _bwrap_candidates(bwrap: Path | None) -> tuple[Path, ...]:
    """``bwrap`` alone, or the bundled bwrap followed by any bwrap on ``PATH``; the caller probes each in turn."""
    if sys.platform != "linux":
        raise SandboxUnavailable("The local bubblewrap backend requires Linux")
    # This dependency is installed only on Linux.
    from bubblewrap_bin import bwrap_path  # noqa: PLC0415

    if bwrap is not None:
        return (bwrap,)
    system = shutil.which("bwrap")
    return (bwrap_path(), *(() if system is None else (Path(system),)))


def _resource_limits(file_size_limit: int = FILE_SIZE_LIMIT) -> list[tuple[int, int]]:
    host_tasks = int(Path("/proc/loadavg").read_text().split()[3].split("/")[1])
    return [
        (resource.RLIMIT_NPROC, host_tasks + NPROC_HEADROOM),
        (resource.RLIMIT_FSIZE, file_size_limit),
        (resource.RLIMIT_CORE, 0),
    ]


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


def _machine_directories(
    workdir: PurePosixPath | None, *, system_directories: SystemDirectories, disk: LocalDisk | None = None
) -> tuple[Path, Path | None]:
    """A new machine's root, with ``workdir`` made in it, and its layers when its system directories are overlaid."""
    root = disk.data / "root" if disk is not None else Path(tempfile.mkdtemp(prefix="shellbox-local-"))
    root.mkdir(exist_ok=True)
    layers = None
    try:
        _prepare_root(root, workdir)
        if system_directories is SystemDirectories.COPY_ON_WRITE:
            layers = disk.data / "layers" if disk is not None else Path(tempfile.mkdtemp(prefix="shellbox-layers-"))
            _prepare_layers(layers)
    except BaseException:
        _remove(root)
        if layers is not None:
            _remove(layers)
        raise
    return root, layers
