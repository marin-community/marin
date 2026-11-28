# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only disk guardian, also executed in an isolated subprocess.

The guardian locks the allocation directory inode through removal. EOF
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
from contextlib import ExitStack
from pathlib import Path

COMMAND_TIMEOUT = 30
CLEANUP_TIMEOUT = 30
LOOP_CONTROL = Path("/dev/loop-control")


def node_name() -> str:
    return os.environ.get("IRIS_NODE_NAME", os.uname().nodename)


def run(*argv: str) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed on node {node_name()}: {result.stderr.strip()}")
    return result.stdout.strip()


def attach(image: Path) -> str:
    """Allocate one loop device, serializing cooperating pools across mount namespaces."""
    with LOOP_CONTROL.open("rb") as control:
        fcntl.flock(control, fcntl.LOCK_EX)
        return run("losetup", "--find", "--show", "--nooverlap", str(image))


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
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        # Concurrent recovery may finish removal while this guardian unmounts.
        # Missing children in a still-present allocation are not successful cleanup.
        if directory.exists():
            raise


def guard(directory: Path, storage_mb: int) -> None:
    with ExitStack() as resources:
        # A child lease file can be unlinked and recreated during rmtree, giving
        # startup recovery a different unlocked inode while this guard is live.
        lease = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        resources.callback(os.close, lease)
        fcntl.flock(lease, fcntl.LOCK_EX)
        try:
            image = directory / "disk.img"
            with image.open("xb") as output:
                output.truncate(storage_mb * 1024 * 1024)
            run("mkfs.ext4", "-q", "-F", "-m", "0", "-E", "lazy_itable_init=0,lazy_journal_init=0", str(image))
            data = directory / "data"
            data.mkdir()
            (directory / "root").mkdir()
            device = attach(image)
            run("mount", "-t", "ext4", "-o", "nodev,nosuid,noatime", device, str(data))
            # On a mounted loop, detach sets AUTOCLEAR; the final unmount releases it.
            # This also preserves namespace-destruction cleanup after guardian death.
            run("losetup", "--detach", device)
            (data / "shm").mkdir(mode=0o1777)
            (data / "shm").chmod(0o1777)
            print(json.dumps({"ready": True}), flush=True)
            sys.stdin.buffer.read()
        except Exception as error:
            error.add_note(f"LOCAL disk guardian node: {node_name()}")
            raise
        finally:
            reclaim(directory)


if __name__ == "__main__":
    guard(Path(sys.argv[1]), int(sys.argv[2]))
