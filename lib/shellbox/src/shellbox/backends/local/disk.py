# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified per-sandbox ext4 filesystems for privileged LOCAL workers."""

import asyncio
import errno
import fcntl
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
from _thread import LockType
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from shellbox.backends.local import disk_lifecycle
from shellbox.machine import MachineDiskFull, UnsupportedMachineSpec

PROBE_STORAGE_MB = 64
POOL_CACHE_SIZE = 16
GUARD_SOURCE = Path(disk_lifecycle.__file__).read_text()
POOL_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PoolIdentity:
    pid: int
    mount_namespace: int
    root: Path
    device: int
    inode: int


@dataclass
class _PoolEntry:
    identity: PoolIdentity
    lock: LockType = field(default_factory=threading.Lock)
    pool: "LocalDiskPool | None" = None


async def verified_local_disk_pool(root: Path) -> "LocalDiskPool":
    """Reuse one verified pool per worker process, mount namespace and disk root."""
    return await asyncio.to_thread(_verified_local_disk_pool, root)


def _verified_local_disk_pool(root: Path) -> "LocalDiskPool":
    try:
        mount_namespace = os.stat("/proc/self/ns/mnt").st_ino
    except FileNotFoundError as error:
        raise UnsupportedMachineSpec("LOCAL disk enforcement requires a privileged Linux mount namespace") from error
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity = root.stat()
    key = PoolIdentity(os.getpid(), mount_namespace, root, identity.st_dev, identity.st_ino)
    with POOL_CACHE_LOCK:
        entry = _pool_entry(key)
    with entry.lock:
        if entry.pool is None:
            entry.pool = LocalDiskPool(entry.identity.root)
        return entry.pool


@lru_cache(maxsize=POOL_CACHE_SIZE)
def _pool_entry(key: PoolIdentity) -> _PoolEntry:
    return _PoolEntry(key)


@dataclass
class LocalDisk:
    """A bounded filesystem whose guardian owns mount and loop-device cleanup."""

    directory: Path
    guardian: subprocess.Popen[bytes]
    errors: BinaryIO
    resources: ExitStack

    @property
    def data(self) -> Path:
        return self.directory / disk_lifecycle.DATA_DIRECTORY

    def overlay(self, lower: Path) -> Path:
        """Mount a writable root with all copy-up and metadata charged to this disk."""
        if any(character in str(lower) for character in (",", ":", "\n", "\\")):
            raise UnsupportedMachineSpec("LOCAL image cache cannot contain OverlayFS option separators")
        upper, work = self.data / "upper", self.data / "work"
        upper.mkdir()
        work.mkdir()
        root = self.directory / disk_lifecycle.ROOT_DIRECTORY
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

    @contextmanager
    def spool(self) -> Iterator[BinaryIO]:
        """Allocate a command spool; raise MachineDiskFull when its setup exhausts storage."""
        with ExitStack() as resources:
            try:
                output = resources.enter_context(tempfile.TemporaryFile(dir=self.data, buffering=0))
            except OSError as error:
                if error.errno in (errno.ENOSPC, errno.EDQUOT):
                    raise MachineDiskFull("LOCAL cannot allocate a command spool") from error
                raise
            yield output

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


def _validated_root(root: Path) -> Path:
    if sys.platform != "linux" or os.geteuid() != 0:
        raise UnsupportedMachineSpec("LOCAL disk enforcement requires a privileged Linux worker")
    if not Path("/dev/loop-control").exists():
        raise UnsupportedMachineSpec("LOCAL disk enforcement requires /dev/loop-control in the privileged worker")
    for program in ("mkfs.ext4", "mount", "umount", "losetup", "findmnt"):
        if shutil.which(program) is None:
            raise UnsupportedMachineSpec(f"LOCAL disk enforcement requires host executable {program}")
    root = root.resolve()
    if any(character in str(root) for character in (",", ":", "\n", "\\")):
        raise UnsupportedMachineSpec("LOCAL disk root cannot contain OverlayFS option separators")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != 0 or root.stat().st_mode & 0o077:
        raise UnsupportedMachineSpec("LOCAL disk root must be root-owned and mode 0700")
    propagation = disk_lifecycle.run("findmnt", "--noheadings", "--output", "PROPAGATION", "--target", str(root))
    if "shared" in propagation:
        raise UnsupportedMachineSpec("LOCAL disk root must be in a private/slave pod mount namespace")
    return root


class LocalDiskPool:
    """A host capability: construction verifies mount, overlay, ENOSPC and cleanup.

    Use a private directory on the worker's disk, shared by its LOCAL factories.
    Only allocations whose lease is released are reclaimed on startup.
    """

    def __init__(self, root: Path):
        self.root = _validated_root(root)
        with (self.root / disk_lifecycle.POOL_LOCK_FILE).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._reclaim()
        self._verify_filesystem()

    def _verify_filesystem(self) -> None:
        try:
            self._probe()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            raise UnsupportedMachineSpec(f"LOCAL disk enforcement host probe failed: {error}") from error

    def _reclaim(self) -> None:
        for directory in self.root.glob(f"{disk_lifecycle.ALLOCATION_PREFIX}*"):
            try:
                lease_file = (directory / disk_lifecycle.LEASE_FILE).open("a")
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
        with (self.root / disk_lifecycle.POOL_LOCK_FILE).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._reclaim()
            directory = Path(tempfile.mkdtemp(prefix=disk_lifecycle.ALLOCATION_PREFIX, dir=self.root))
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
                    if json.loads(guardian.stdout.readline()) != json.loads(disk_lifecycle.READY_MESSAGE):
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
                        output.write(bytes(disk_lifecycle.MEBIBYTE))
                except OSError as error:
                    if error.errno != errno.ENOSPC:
                        raise
                else:
                    raise RuntimeError("Fixed-size ext4 filesystem did not return ENOSPC")
        finally:
            disk.close()
