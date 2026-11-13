# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only disk guardian, also executed in an isolated subprocess.

The guardian holds the allocation lease until it has unmounted everything. EOF
on its control pipe means the owner closed or crashed. Loop mounts use util-linux's
autoclear devices, so destroying the pod's mount namespace also releases them.
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMMAND_TIMEOUT = 30
CLEANUP_TIMEOUT = 30


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
            run("mount", "-t", "ext4", "-o", "loop,nodev,nosuid,noatime", str(image), str(data))
            (data / "shm").mkdir(mode=0o1777)
            (data / "shm").chmod(0o1777)
            print(json.dumps({"ready": True}), flush=True)
            sys.stdin.buffer.read()
        finally:
            reclaim(directory)


if __name__ == "__main__":
    guard(Path(sys.argv[1]), int(sys.argv[2]))
