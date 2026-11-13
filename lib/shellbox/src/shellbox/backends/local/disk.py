# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified per-sandbox ext4 filesystems for privileged LOCAL workers."""

import asyncio
import errno
import fcntl
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from shellbox.backends.local import disk_lifecycle, launch
from shellbox.machine import DiskLimitExceeded, UnsupportedMachineSpec

PROBE_STORAGE_MB = 64
GUARD_SOURCE = Path(disk_lifecycle.__file__).read_text()
POOL_LOCK = threading.Lock()
VERIFIED_POOLS: dict[tuple[int, int, Path], "LocalDiskPool"] = {}


async def verified_local_disk_pool(root: Path) -> "LocalDiskPool":
    """Reuse one verified pool per worker process, mount namespace and disk root."""
    return await asyncio.to_thread(_verified_local_disk_pool, root)


def _verified_local_disk_pool(root: Path) -> "LocalDiskPool":
    try:
        mount_namespace = os.stat("/proc/self/ns/mnt").st_ino
    except FileNotFoundError as error:
        raise UnsupportedMachineSpec("LOCAL disk enforcement requires a privileged Linux mount namespace") from error
    key = (os.getpid(), mount_namespace, root.resolve())
    with POOL_LOCK:
        if key not in VERIFIED_POOLS:
            VERIFIED_POOLS[key] = LocalDiskPool(root)
        return VERIFIED_POOLS[key]


@dataclass
class LocalDisk:
    """A bounded filesystem whose guardian owns mount and loop-device cleanup."""

    directory: Path
    guardian: subprocess.Popen[bytes]
    errors: BinaryIO
    resources: ExitStack
    _exhausted: bool = field(default=False, init=False)

    @property
    def data(self) -> Path:
        return self.directory / "data"

    def overlay(self, lower: Path) -> Path:
        """Mount a writable root with all copy-up and metadata charged to this disk."""
        if any(character in str(lower) for character in (",", ":", "\n", "\\")):
            raise UnsupportedMachineSpec("LOCAL image cache cannot contain OverlayFS option separators")
        upper, work = self.data / "upper", self.data / "work"
        upper.mkdir()
        work.mkdir()
        root = self.directory / "root"
        disk_lifecycle.run(
            "mount",
            "-t",
            "overlay",
            "overlay",
            "-o",
            f"lowerdir={lower},upperdir={upper},workdir={work},nodev,nosuid",
            str(root),
        )
        return root

    def check_limit(self, *, allocation_errno: int | None = None) -> None:
        """Raise a sticky disk limit after an observed failed allocation."""
        # Consuming the final block or inode is within the budget. Latch only a
        # failed allocation reported by the kernel or a quota-scoped host syscall.
        self._exhausted = self._exhausted or allocation_errno in (errno.ENOSPC, errno.EDQUOT)
        if self._exhausted:
            raise DiskLimitExceeded("RESOURCE_LIMIT_DISK: LOCAL sandbox filesystem allocation failed (ENOSPC/EDQUOT)")

    def close(self) -> None:
        """Release the filesystem and surface any guardian cleanup failure."""
        assert self.guardian.stdin is not None
        self.guardian.stdin.close()
        code = self.guardian.wait(timeout=disk_lifecycle.CLEANUP_TIMEOUT + disk_lifecycle.COMMAND_TIMEOUT)
        self.errors.seek(0)
        detail = self.errors.read().decode(errors="replace")
        self.resources.close()
        if code:
            raise RuntimeError(f"Local disk guardian failed; allocation retained for startup recovery: {detail}")


class LocalDiskPool:
    """A host capability: construction verifies mount, overlay, ENOSPC and cleanup.

    Use a private directory on the worker's disk, shared by its LOCAL factories.
    Only allocations whose lease is released are reclaimed on startup.
    """

    def __init__(self, root: Path):
        if sys.platform != "linux" or os.geteuid() != 0:
            raise UnsupportedMachineSpec("LOCAL disk enforcement requires a privileged Linux worker")
        if not Path("/dev/loop-control").exists():
            raise UnsupportedMachineSpec("LOCAL disk enforcement requires /dev/loop-control in the privileged worker")
        for program in ("mkfs.ext4", "mount", "umount", "losetup", "findmnt"):
            if shutil.which(program) is None:
                raise UnsupportedMachineSpec(f"LOCAL disk enforcement requires host executable {program}")
        self.root = root.resolve()
        if any(character in str(self.root) for character in (",", ":", "\n", "\\")):
            raise UnsupportedMachineSpec("LOCAL disk root cannot contain OverlayFS option separators")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.stat().st_uid != 0 or self.root.stat().st_mode & 0o077:
            raise UnsupportedMachineSpec("LOCAL disk root must be root-owned and mode 0700")
        propagation = disk_lifecycle.run(
            "findmnt", "--noheadings", "--output", "PROPAGATION", "--target", str(self.root)
        )
        if "shared" in propagation:
            raise UnsupportedMachineSpec("LOCAL disk root must be in a private/slave pod mount namespace")
        with (self.root / "pool.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._reclaim()
        try:
            self._probe()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            raise UnsupportedMachineSpec(f"LOCAL disk enforcement host probe failed: {error}") from error

    def _reclaim(self) -> None:
        for directory in self.root.glob("sandbox-*"):
            try:
                lease_file = (directory / "lease").open("a")
            except FileNotFoundError:
                continue  # A live guardian finished removing its allocation.
            with lease_file as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if directory.exists():
                    disk_lifecycle.reclaim(directory)

    def create(self, storage_mb: int) -> LocalDisk:
        """Allocate exactly storage_mb MiB and wait for the ext4 mount."""
        with (self.root / "pool.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._reclaim()
            directory = Path(tempfile.mkdtemp(prefix="sandbox-", dir=self.root))
            with ExitStack() as resources:
                errors = resources.enter_context(tempfile.TemporaryFile())
                guardian = subprocess.Popen(
                    [sys.executable, "-I", "-S", "-c", GUARD_SOURCE, str(directory), str(storage_mb)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=errors,
                )
                disk = LocalDisk(directory, guardian, errors, resources.pop_all())
                assert guardian.stdout is not None
                try:
                    if not select.select([guardian.stdout], [], [], disk_lifecycle.COMMAND_TIMEOUT)[0]:
                        raise TimeoutError("LOCAL disk guardian did not finish filesystem creation")
                    if guardian.stdout.readline() != b'{"ready": true}\n':
                        raise RuntimeError("LOCAL disk guardian failed to create filesystem")
                except BaseException:
                    disk.close()
                    raise
                finally:
                    guardian.stdout.close()
                return disk

    def _probe(self) -> None:
        disk = self.create(PROBE_STORAGE_MB)
        try:
            lower = disk.directory / "lower"
            lower.mkdir()
            (lower / "image-file").write_text("read-only base")
            root = disk.overlay(lower)
            assert (root / "image-file").read_text() == "read-only base"
            with (root / "overflow").open("wb", buffering=0) as output:
                try:
                    for _ in range(PROBE_STORAGE_MB + 1):
                        output.write(bytes(1024 * 1024))
                except OSError as error:
                    if error.errno != errno.ENOSPC:
                        raise
                else:
                    raise RuntimeError("Fixed-size ext4 filesystem did not return ENOSPC")
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    Path(launch.__file__).read_text(),
                    "--probe-allocations",
                    str(root / "observed-overflow"),
                ],
                check=True,
                capture_output=True,
                timeout=disk_lifecycle.COMMAND_TIMEOUT,
            )
        finally:
            disk.close()
