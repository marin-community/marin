# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only disk guardian, also executed in an isolated subprocess.

The guardian holds the allocation lease until it has unmounted everything. EOF
on its control pipe means the owner closed or crashed. Loop mounts use kernel
autoclear devices, so destroying the pod's mount namespace also releases them.
"""

import errno
import fcntl
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

COMMAND_TIMEOUT = 30
CLEANUP_TIMEOUT = 30
LOOP_CTL_GET_FREE = 0x4C82
LOOP_CONFIGURE = 0x4C0A
LO_FLAGS_AUTOCLEAR = 4
LOOP_DEVICE_MAJOR = 7
LOOP_CONFIG_BYTES = 304
LOOP_CONFIG_FLAGS_OFFSET = 60
LOOP_ALLOCATION_ATTEMPTS = 64


@contextmanager
def loop_device(image: Path) -> Iterator[Path]:
    """Atomically attach an autoclearing loop, creating its node in this pod."""
    # linux/loop.h: loop_config embeds loop_info64 after two u32 fields.
    with Path("/dev/loop-control").open("rb") as control, image.open("r+b") as backing:
        config = bytearray(LOOP_CONFIG_BYTES)
        struct.pack_into("=I", config, 0, backing.fileno())
        struct.pack_into("=I", config, LOOP_CONFIG_FLAGS_OFFSET, LO_FLAGS_AUTOCLEAR)
        for _ in range(LOOP_ALLOCATION_ATTEMPTS):
            index = fcntl.ioctl(control, LOOP_CTL_GET_FREE)
            device = Path(f"/dev/loop{index}")
            number = os.makedev(LOOP_DEVICE_MAJOR, index)
            try:
                os.mknod(device, stat.S_IFBLK | 0o600, number)
            except FileExistsError:
                if device.stat().st_rdev != number:
                    raise
            with device.open("r+b", buffering=0) as attached:
                try:
                    fcntl.ioctl(attached, LOOP_CONFIGURE, config)
                except OSError as error:
                    if error.errno == errno.EBUSY:
                        continue
                    raise
                yield device
                return
    raise RuntimeError("Loop devices remained busy during atomic allocation")


def run(*argv: str) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def reclaim(directory: Path) -> None:
    """Unmount only this allocation; never use lazy unmount or detach other loops."""
    deadline = time.monotonic() + CLEANUP_TIMEOUT
    for path in (directory / "root", directory / "data"):
        while os.path.ismount(path):
            result = subprocess.run(["umount", str(path)], capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
            if result.returncode == 0:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Cannot reclaim local disk mount {path}: {result.stderr.strip()}")
            # bwrap's parent-death signal and the kernel's final namespace release
            # can race the control pipe closing after owner death.
            time.sleep(0.05)
    image = directory / "disk.img"
    if image.exists():
        # Normally umount releases the autoclear loop already. Scope recovery to
        # this exact backing file, including a crash between loop setup and mount.
        for row in run("losetup", "--associated", str(image), "--noheadings", "--output", "NAME").splitlines():
            run("losetup", "--detach", row.strip())
    shutil.rmtree(directory)


def guard(directory: Path, storage_mb: int) -> None:
    with (directory / "lease").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        try:
            image = directory / "disk.img"
            with image.open("xb") as output:
                output.truncate(storage_mb * 1024 * 1024)
            run("mkfs.ext4", "-q", "-F", "-m", "0", "-E", "lazy_itable_init=0,lazy_journal_init=0", str(image))
            data = directory / "data"
            data.mkdir()
            (directory / "root").mkdir()
            with loop_device(image) as device:
                run("mount", "-t", "ext4", "-o", "nodev,nosuid,noatime", str(device), str(data))
            (data / "shm").mkdir(mode=0o1777)
            (data / "shm").chmod(0o1777)
            print(json.dumps({"ready": True}), flush=True)
            sys.stdin.buffer.read()
        finally:
            reclaim(directory)


if __name__ == "__main__":
    guard(Path(sys.argv[1]), int(sys.argv[2]))
