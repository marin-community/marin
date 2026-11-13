# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real loop/OverlayFS enforcement; run explicitly in a privileged Linux worker."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from shellbox.backends.local.disk import LocalDiskPool
from shellbox.backends.local.machine import LocalMachineFactory, check_local_environment
from shellbox.machine import (
    Command,
    DiskLimitExceeded,
    MachineSpec,
    MachineTerminated,
    ResourceLimit,
    UnsupportedMachineSpec,
)
from shellbox.survey import ProbeRequirements

pytestmark = [pytest.mark.manual, pytest.mark.asyncio]


def disk_state(root: Path) -> dict:
    loops = json.loads(subprocess.check_output(["losetup", "--json", "--output", "NAME,BACK-FILE"], text=True))
    return {
        "loops": [row for row in loops["loopdevices"] if str(root) in row["back-file"]],
        "mounts": [row for row in Path("/proc/self/mountinfo").read_text().splitlines() if str(root) in row],
        "allocations": [str(path) for path in root.glob("sandbox-*")],
    }


@pytest.mark.parametrize("path", ["/app/fill", "/tmp/fill", "/dev/shm/fill", "/usr/fill"])
async def test_local_disk_overflow_is_typed_and_neighbor_survives(tmp_path, executable_oci_image, path):
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_pool=pool, image_cache=tmp_path / "images")
    spec = MachineSpec(executable_oci_image, workdir="/app", storage_mb=64)
    with pytest.raises(UnsupportedMachineSpec):
        check_local_environment(ProbeRequirements(), spec)
    factory.check_environment(ProbeRequirements(), spec)
    machine = await factory.create(spec)
    neighbor = await factory.create(replace(spec, storage_mb=96))
    try:
        assert (await machine.run(Command(("/bin/mkdir", "-p", "/tmp", "/usr")))).exit_code == 0
        # Pipes expose the actual guest ENOSPC before the host reports its typed
        # outcome; a full filesystem cannot store another diagnostic spool file.
        session = await machine.open_process(
            Command(("/bin/dd", "if=/dev/zero", f"of={path}", "bs=1048576", "count=128"))
        )
        try:
            await session.drain_output()
            with pytest.raises(DiskLimitExceeded) as failure:
                await session.wait()
            assert b"No space left on device" in session.stderr
        finally:
            await session.close()
        with pytest.raises(DiskLimitExceeded):
            await machine.run(Command(("/bin/sh", "-c", "true")))
        assert failure.value.resource is ResourceLimit.DISK
        result = await neighbor.run(Command(("/bin/sh", "-c", "echo retained > /app/answer; cat /app/answer")))
        assert (result.exit_code, result.stdout) == (0, b"retained\n")
        assert (await neighbor.run(Command(("/bin/cat", "/etc/hosts")))).exit_code == 0
        print(
            json.dumps(
                {"path": path, "guest_errno": "ENOSPC", "resource": failure.value.resource, "neighbor": "retained"}
            ),
            flush=True,
        )
    finally:
        await asyncio.gather(machine.close(), neighbor.close())
    state = disk_state(pool.root)
    print(json.dumps({"post_teardown": state}), flush=True)
    assert state == {"loops": [], "mounts": [], "allocations": []}


async def test_local_disk_run_overflow_raises_disk_limit_and_preserves_image(tmp_path, executable_oci_image):
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_pool=pool, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    try:
        with pytest.raises(DiskLimitExceeded) as failure:
            await machine.run(Command(("/bin/dd", "if=/dev/zero", "of=/app/fill", "bs=1048576", "count=128")))
        assert failure.value.resource is ResourceLimit.DISK
    finally:
        await machine.close()
    fresh = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    try:
        result = await fresh.run(Command(("/bin/sh", "-c", "test ! -e /app/fill && /bin/busybox echo immutable")))
        assert (result.exit_code, result.stdout) == (0, b"immutable\n")
    finally:
        await fresh.close()
    assert disk_state(pool.root) == {"loops": [], "mounts": [], "allocations": []}


async def test_local_disk_final_inode_succeeds_until_an_allocation_fails(tmp_path, executable_oci_image):
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_pool=pool, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    fill_inodes = (
        "free=$(/bin/busybox stat -f -c %d /app); i=0; "
        'while [ "$i" -lt "$free" ]; do : > "/app/inode$i" || exit; i=$((i+1)); done; '
        'test "$(/bin/busybox stat -f -c %d /app)" = 0 || exit 99; '
        "printf full-success"
    )
    try:
        session = await machine.open_process(Command(("/bin/sh", "-c", fill_inodes)))
        try:
            assert await session.read() == b"full-success"
            assert await session.wait() == 0
        finally:
            await session.close()
        # A command using pipes needs no new inode on the task disk. Fullness
        # must not terminate it or retroactively fail the successful allocation.
        reader = await machine.open_process(Command(("/bin/busybox", "stat", "-f", "-c", "%d", "/app")))
        try:
            assert await reader.read() == b"0\n"
            assert await reader.wait() == 0
        finally:
            await reader.close()
        overflow = await machine.open_process(Command(("/bin/busybox", "touch", "/app/overflow")))
        try:
            await overflow.drain_output()
            assert b"No space left on device" in overflow.stderr
            with pytest.raises(DiskLimitExceeded) as failure:
                await overflow.wait()
            assert failure.value.resource is ResourceLimit.DISK
        finally:
            await overflow.close()
        with pytest.raises(DiskLimitExceeded):
            await machine.run(Command(("/bin/sh", "-c", "true")))
        print(json.dumps({"final_inode": "success", "subsequent_allocation": "typed_disk_failure"}), flush=True)
    finally:
        await machine.close()
    assert disk_state(pool.root) == {"loops": [], "mounts": [], "allocations": []}


async def test_local_disk_startup_reclaims_crashed_guardian_and_keeps_live_neighbor(tmp_path):
    pool = LocalDiskPool(tmp_path / "disks")
    crashed = pool.create(64)
    neighbor = pool.create(64)
    try:
        (neighbor.data / "answer").write_text("preserved")
        crashed.guardian.kill()
        crashed.guardian.wait(timeout=5)
        assert os.path.ismount(crashed.data)
        LocalDiskPool(pool.root)
        assert not crashed.directory.exists()
        assert (neighbor.data / "answer").read_text() == "preserved"
    finally:
        assert crashed.guardian.stdin is not None
        crashed.guardian.stdin.close()
        crashed.errors.close()
        neighbor.close()
    assert disk_state(pool.root) == {"loops": [], "mounts": [], "allocations": []}


async def test_local_disk_expired_sandbox_releases_mounts_and_loop(tmp_path, executable_oci_image):
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_pool=pool, image_cache=tmp_path / "images", ttl=0.1)
    machine = await factory.create(MachineSpec(executable_oci_image, storage_mb=64))
    try:
        with pytest.raises(MachineTerminated):
            await machine.run(Command(("/bin/sleep", "3600")))
    finally:
        await machine.close()
    assert disk_state(pool.root) == {"loops": [], "mounts": [], "allocations": []}


async def test_local_disk_owner_crash_triggers_guardian_cleanup(tmp_path):
    root = tmp_path / "disks"
    program = (
        "import os\nfrom pathlib import Path\n"
        "from shellbox.backends.local.disk import LocalDiskPool\n"
        f"disk=LocalDiskPool(Path({str(root)!r})).create(64)\n"
        "print(disk.guardian.pid, flush=True)\nos._exit(0)\n"
    )
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", program, stdout=asyncio.subprocess.PIPE)
    stdout, _ = await process.communicate()
    assert process.returncode == 0
    try:
        descriptor = os.pidfd_open(int(stdout))
    except ProcessLookupError:
        pass  # The guardian already completed cleanup and was reaped.
    else:
        loop = asyncio.get_running_loop()
        exited = loop.create_future()
        loop.add_reader(descriptor, lambda: not exited.done() and exited.set_result(None))
        try:
            await asyncio.wait_for(exited, 10)
        finally:
            loop.remove_reader(descriptor)
            os.close(descriptor)
    assert disk_state(root) == {"loops": [], "mounts": [], "allocations": []}
