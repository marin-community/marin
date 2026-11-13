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
from shellbox.backends.local.disk import LocalDiskPool, verified_local_disk_pool
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


async def test_local_disk_verified_pool_is_shared_by_concurrent_host_requests(tmp_path):
    first, second = await asyncio.gather(
        verified_local_disk_pool(tmp_path / "disks"), verified_local_disk_pool(tmp_path / "disks")
    )
    assert first is second
    assert disk_state(first.root) == {"loops": [], "mounts": [], "allocations": []}


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
        # Both overlays still expose their pinned image's read-only contents.
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


@pytest.mark.parametrize("output_limit", [0, 16])
@pytest.mark.parametrize("process_session", [False, True], ids=["run", "session"])
async def test_local_disk_kernel_evidence_ignores_text_and_output_limits(
    tmp_path, executable_oci_image, output_limit, process_session
):
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_pool=pool, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    diagnostic = Command(
        ("/bin/sh", "-c", "printf 'ENOSPC EDQUOT No space left on device Disk quota exceeded' >&2; exit 1"),
        output_limit_bytes=output_limit,
    )
    overflow = Command(
        (
            "/bin/sh",
            "-c",
            "printf 'unrelated diagnostic before allocation failure' >&2; "
            "/bin/dd if=/dev/zero of=/app/fill bs=1048576 count=128 2>/dev/null; exit 0",
        ),
        output_limit_bytes=output_limit,
    )
    try:
        if process_session:
            session = await machine.open_process(diagnostic)
            try:
                await session.drain_output()
                assert await session.wait() == 1
            finally:
                await session.close()
        else:
            assert (await machine.run(diagnostic)).exit_code == 1
        # ENOSPC from a device is not an allocation failure on the task disk.
        device = Command(
            ("/bin/dd", "if=/dev/zero", "of=/dev/full", "bs=1024", "count=1"),
            output_limit_bytes=output_limit,
        )
        assert (await machine.run(device)).exit_code != 0
        assert (await machine.run(Command(("/bin/sh", "-c", "true")))).exit_code == 0
        if process_session:
            session = await machine.open_process(overflow)
            try:
                await session.drain_output()
                with pytest.raises(DiskLimitExceeded):
                    await session.wait()
            finally:
                await session.close()
        else:
            with pytest.raises(DiskLimitExceeded):
                await machine.run(overflow)
    finally:
        await machine.close()
    assert disk_state(pool.root) == {"loops": [], "mounts": [], "allocations": []}


async def test_local_disk_run_final_inode_succeeds_with_charged_spools(tmp_path, executable_oci_image):
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_pool=pool, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    fill_inodes = (
        "free=$(/bin/busybox stat -f -c %d /app); i=0; "
        'while [ "$i" -lt "$free" ]; do : > "/app/inode$i" || exit; i=$((i+1)); done; '
        'test "$(/bin/busybox stat -f -c %d /app)" = 0'
    )
    try:
        # run() already holds its three anonymous spool inodes. The final guest
        # allocation succeeds; closing those spools frees them for the next run.
        assert (await machine.run(Command(("/bin/sh", "-c", fill_inodes), output_limit_bytes=0))).exit_code == 0
        result = await machine.run(Command(("/bin/busybox", "stat", "-f", "-c", "%d", "/app")))
        assert (result.exit_code, result.stdout) == (0, b"0\n")
        # Those same spool allocations are legitimate quota usage. No inode
        # remains for another guest file while the next run holds its spools.
        with pytest.raises(DiskLimitExceeded):
            await machine.run(Command(("/bin/busybox", "touch", "/app/overflow")))
    finally:
        await machine.close()
    assert disk_state(pool.root) == {"loops": [], "mounts": [], "allocations": []}


@pytest.mark.parametrize("allocation", ["guest", "run-spool"])
async def test_local_disk_final_inode_succeeds_until_an_allocation_fails(tmp_path, executable_oci_image, allocation):
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
        if allocation == "run-spool":
            with pytest.raises(DiskLimitExceeded) as failure:
                await machine.run(Command(("/bin/sh", "-c", "true")))
            assert failure.value.resource is ResourceLimit.DISK
        else:
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
        print(json.dumps({"final_inode": "success", "failed_allocation": allocation, "resource": "disk"}), flush=True)
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
