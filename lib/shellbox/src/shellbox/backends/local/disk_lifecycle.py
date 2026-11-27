# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only disk guardian, also executed in an isolated subprocess.

The guardian holds the allocation lease until it has unmounted everything. EOF
on its control pipe means the owner closed or crashed. Loop attachment atomically enables
autoclear, so destroying the pod's mount namespace also releases them.
"""

import ctypes
import errno
import fcntl
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOOP_CONTROL = Path("/dev/loop-control")
SHARED_MEMORY_DIRECTORY = "shm"
STICKY_WORLD_WRITABLE = 0o1777
ALLOCATION_PREFIX = "sandbox-"
BACKING_IMAGE = "disk.img"
POOL_LOCK_FILE = "pool.lock"
LEASE_FILE = "lease"
DATA_DIRECTORY = "data"
ROOT_DIRECTORY = "root"
READY_MESSAGE = '{"ready": true}'
MEBIBYTE = 1024 * 1024
COMMAND_TIMEOUT = 30
CLEANUP_TIMEOUT = 30
LOOP_CTL_GET_FREE = 0x4C82
LOOP_CONFIGURE = 0x4C0A
LO_FLAGS_AUTOCLEAR = 4
LOOP_MAJOR = 7
LOOP_ALLOCATION_ATTEMPTS = 16


# Linux UAPI layouts: include/uapi/linux/loop.h. LOOP_CONFIGURE has been
# available since Linux 5.8 and atomically enables autoclear with attachment.
class LoopInfo64(ctypes.Structure):
    _fields_ = (
        ("device", ctypes.c_uint64),
        ("inode", ctypes.c_uint64),
        ("rdevice", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("sizelimit", ctypes.c_uint64),
        ("number", ctypes.c_uint32),
        ("encrypt_type", ctypes.c_uint32),
        ("encrypt_key_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("file_name", ctypes.c_uint8 * 64),
        ("crypt_name", ctypes.c_uint8 * 64),
        ("encrypt_key", ctypes.c_uint8 * 32),
        ("init", ctypes.c_uint64 * 2),
    )


class LoopConfig(ctypes.Structure):
    _fields_ = (
        ("fd", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("info", LoopInfo64),
        ("reserved", ctypes.c_uint64 * 8),
    )


@contextmanager
def attached_loop(image: Path) -> Iterator[Path]:
    """Attach an owned image, holding autoclear until its mount takes a reference."""
    with LOOP_CONTROL.open("rb") as control, image.open("r+b") as backing:
        for _ in range(LOOP_ALLOCATION_ATTEMPTS):
            minor = fcntl.ioctl(control, LOOP_CTL_GET_FREE)
            device = Path(f"/dev/loop{minor}")
            expected = os.makedev(LOOP_MAJOR, minor)
            try:
                os.mknod(device, stat.S_IFBLK | 0o600, expected)
            except FileExistsError:
                pass
            descriptor = os.open(device, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                node = os.fstat(descriptor)
                if not stat.S_ISBLK(node.st_mode) or node.st_rdev != expected:
                    raise RuntimeError(f"Invalid loop device node: {device}")
                config = LoopConfig(fd=backing.fileno(), info=LoopInfo64(flags=LO_FLAGS_AUTOCLEAR))
                try:
                    fcntl.ioctl(descriptor, LOOP_CONFIGURE, bytes(config))
                except OSError as error:
                    if error.errno == errno.EBUSY:
                        continue  # Another worker attached this free minor first.
                    raise
                yield device
                return
            finally:
                os.close(descriptor)
    raise RuntimeError("Cannot acquire a free loop device after concurrent allocation attempts")


def run(*argv: str) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def reclaim(directory: Path) -> None:
    """Unmount only this allocation; never use lazy unmount or detach other loops."""
    deadline = time.monotonic() + CLEANUP_TIMEOUT
    for path in (directory / ROOT_DIRECTORY, directory / DATA_DIRECTORY):
        while os.path.ismount(path):
            result = subprocess.run(["umount", str(path)], capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
            if result.returncode == 0:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Cannot reclaim local disk mount {path}: {result.stderr.strip()}")
            # bwrap's parent-death signal and the kernel's final namespace release
            # can race the control pipe closing after owner death.
            time.sleep(0.05)
    image = directory / BACKING_IMAGE
    if image.exists():
        # Normally umount releases the autoclear loop already. Scope recovery to
        # this exact backing file, including a crash between loop setup and mount.
        for row in run("losetup", "--associated", str(image), "--noheadings", "--output", "NAME").splitlines():
            run("losetup", "--detach", row.strip())
        # LOOP_CLR_FD may return before the kernel releases its last reference.
        # Keep the backing file identifiable until detachment actually finishes;
        # otherwise a successful close can leave a deleted image consuming disk.
        while run("losetup", "--associated", str(image), "--noheadings", "--output", "NAME"):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Cannot reclaim local disk loop for {image}")
            time.sleep(0.05)
    shutil.rmtree(directory)


def guard(directory: Path, storage_mb: int) -> None:
    with (directory / LEASE_FILE).open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        try:
            image = directory / BACKING_IMAGE
            with image.open("xb") as output:
                output.truncate(storage_mb * MEBIBYTE)
            run("mkfs.ext4", "-q", "-F", "-m", "0", "-E", "lazy_itable_init=0,lazy_journal_init=0", str(image))
            data = directory / DATA_DIRECTORY
            data.mkdir()
            (directory / ROOT_DIRECTORY).mkdir()
            with attached_loop(image) as device:
                run("mount", "-t", "ext4", "-o", "nodev,nosuid,noatime", str(device), str(data))
            (data / SHARED_MEMORY_DIRECTORY).mkdir(mode=STICKY_WORLD_WRITABLE)
            (data / SHARED_MEMORY_DIRECTORY).chmod(STICKY_WORLD_WRITABLE)
            print(READY_MESSAGE, flush=True)
            sys.stdin.buffer.read()
        finally:
            reclaim(directory)


if __name__ == "__main__":
    guard(Path(sys.argv[1]), int(sys.argv[2]))
