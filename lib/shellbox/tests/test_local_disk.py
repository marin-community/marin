# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real loop/OverlayFS enforcement; run explicitly in a privileged Linux worker."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from shellbox.backends.local.cgroup import delegated_cgroup_parent
from shellbox.backends.local.disk import LocalDiskPool, verified_local_disk_pool
from shellbox.backends.local.machine import LocalMachineFactory, check_local_environment
from shellbox.machine import (
    Command,
    CommandSemantics,
    ExitReason,
    MachineDiskFull,
    MachineSpec,
    MachineTerminated,
    UnsupportedMachineSpec,
)


@dataclass(frozen=True)
class ProbeRequirements:
    command_semantics: CommandSemantics = CommandSemantics.LINUX_PROCESS
    capabilities: tuple[str, ...] = ()
    docker_image: str | None = None


pytestmark = [pytest.mark.manual, pytest.mark.asyncio]


# Large ext4 writes can fail with a usable tail block remaining. Exhaust that
# tail in the guest before asserting a failed command at the actual block limit.
# Keep diagnostics off quota-backed spools: closing a nonempty spool would free
# its blocks before the host observes the final filesystem state.
OVERFLOW_SCRIPT = (
    '/bin/dd if=/dev/zero of="$1" bs=1048576 count=128 2>/dev/null; '
    'while :; do /bin/dd if=/dev/zero bs=1 count=1024 >> "$1" 2>/dev/null || exit; done'
)


@dataclass(frozen=True)
class DiskState:
    loops: tuple[str, ...] = ()
    mounts: tuple[str, ...] = ()
    allocations: tuple[str, ...] = ()


def disk_state(root: Path) -> DiskState:
    loops = json.loads(subprocess.check_output(["losetup", "--json", "--output", "NAME,BACK-FILE"], text=True))
    return DiskState(
        loops=tuple(row["name"] for row in loops["loopdevices"] if str(root) in row["back-file"]),
        mounts=tuple(row for row in Path("/proc/self/mountinfo").read_text().splitlines() if str(root) in row),
        allocations=tuple(str(path) for path in root.glob("sandbox-*")),
    )


async def test_local_disk_verified_pool_is_shared_by_concurrent_host_requests(tmp_path):
    first, second = await asyncio.gather(
        verified_local_disk_pool(tmp_path / "disks"), verified_local_disk_pool(tmp_path / "disks")
    )
    assert first is second
    assert disk_state(first.root) == DiskState()


async def test_local_disk_command_timeout_releases_spools_before_unmount(tmp_path, executable_oci_image, monkeypatch):
    monkeypatch.setattr("shellbox.backends.local.machine.delegated_cgroup_parent", lambda: None)
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_root=pool.root, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    try:
        result = await machine.run(Command(("/bin/sleep", "60"), timeout=1))
        assert result.reason is ExitReason.TIMED_OUT
    finally:
        await machine.close()
    assert disk_state(pool.root) == DiskState()


async def test_local_disk_exhaustion_does_not_prevent_command_quiescence(tmp_path, executable_oci_image):
    parent = delegated_cgroup_parent()
    if parent is None:
        pytest.skip("Command quiescence requires a delegated worker cgroup")
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_root=pool.root, image_cache=tmp_path / "images", cgroup_parent=parent)
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    try:
        result = await machine.run(
            Command(("/bin/sh", "-c", "sleep 60 & echo $! > /app/pid; " + OVERFLOW_SCRIPT, "sh", "/app/fill"))
        )
        assert result.exit_code != 0
        await machine.quiesce_commands()
        session = await machine.open_process(Command(("/bin/sh", "-c", 'kill -0 "$(cat /app/pid)"')))
        try:
            assert await session.wait() != 0
        finally:
            await session.close()
    finally:
        await machine.close()
    assert disk_state(pool.root) == DiskState()


@pytest.mark.parametrize("path", ["/app/fill", "/tmp/fill", "/dev/shm/fill", "/usr/fill"])
async def test_local_disk_overflow_returns_failure_and_neighbor_survives(tmp_path, executable_oci_image, path):
    root = tmp_path / "disks"
    factory = LocalMachineFactory(disk_root=root, image_cache=tmp_path / "images")
    spec = MachineSpec(executable_oci_image, workdir="/app", storage_mb=64)
    with pytest.raises(UnsupportedMachineSpec):
        check_local_environment(ProbeRequirements(), spec)
    factory.check_environment(ProbeRequirements(), spec)
    machine = await factory.create(spec)
    neighbor = await factory.create(replace(spec, storage_mb=96))
    try:
        assert (await machine.run(Command(("/bin/mkdir", "-p", str(Path(path).parent))))).exit_code == 0
        result = await machine.run(Command(("/bin/sh", "-c", OVERFLOW_SCRIPT, "sh", path)))
        assert result.exit_code != 0
        assert os.statvfs(machine.root).f_bavail == 0
        session = await machine.open_process(Command(("/bin/rm", path)))
        try:
            assert await session.wait() == 0
        finally:
            await session.close()
        assert (await machine.run(Command(("/bin/true",)))).exit_code == 0
        result = await neighbor.run(Command(("/bin/sh", "-c", "echo retained > /app/answer; cat /app/answer")))
        assert (result.exit_code, result.stdout) == (0, b"retained\n")
    finally:
        await asyncio.gather(machine.close(), neighbor.close())
    assert disk_state(root) == DiskState()


@pytest.mark.parametrize("storage_mb", [64, 512])
async def test_local_disk_low_headroom_preserves_command_exit_status(tmp_path, executable_oci_image, storage_mb):
    root = tmp_path / "disks"
    factory = LocalMachineFactory(disk_root=root, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=storage_mb))
    initial = os.statvfs(machine.root)
    allocation = initial.f_bavail * initial.f_bsize - 2 * 1024 * 1024
    try:
        await asyncio.to_thread(
            subprocess.run, ["fallocate", "-l", str(allocation), str(machine.root / "app/fill")], check=True
        )
        assert (await machine.run(Command(("/bin/sh", "-c", "exit 7"), output_limit_bytes=0))).exit_code == 7
        assert (await machine.run(Command(("/bin/true",)))).exit_code == 0
    finally:
        await machine.close()
    assert disk_state(root) == DiskState()


FILL_INODES = (
    "free=$(/bin/busybox stat -f -c %d /app); i=0; "
    'while [ "$i" -lt "$free" ]; do : > "/app/inode$i" || exit; i=$((i+1)); done; '
    'test "$(/bin/busybox stat -f -c %d /app)" = 0'
)


async def test_local_disk_full_inodes_block_spools_but_allow_recovery(tmp_path, executable_oci_image):
    root = tmp_path / "disks"
    factory = LocalMachineFactory(disk_root=root, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    try:
        session = await machine.open_process(Command(("/bin/sh", "-c", FILL_INODES)))
        try:
            assert await session.wait() == 0
        finally:
            await session.close()
        with pytest.raises(MachineDiskFull):
            await machine.run(Command(("/bin/true",)))
        recovery = await machine.open_process(Command(("/bin/sh", "-c", "rm /app/inode*")))
        try:
            assert await recovery.wait() == 0
        finally:
            await recovery.close()
        assert (await machine.run(Command(("/bin/true",)))).exit_code == 0
    finally:
        await machine.close()
    assert disk_state(root) == DiskState()


async def test_local_disk_stdin_exhaustion_preserves_error_and_allows_next_command(tmp_path, executable_oci_image):
    root = tmp_path / "disks"
    factory = LocalMachineFactory(disk_root=root, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    try:
        with pytest.raises(MachineDiskFull):
            await machine.run(Command(("/bin/cat",), stdin=bytes(96 * 1024 * 1024)))
        result = await machine.run(Command(("/bin/sh", "-c", "echo recovered")))
        assert (result.exit_code, result.stdout) == (0, b"recovered\n")
    finally:
        await machine.close()
    assert disk_state(root) == DiskState()


async def test_local_disk_failed_command_with_reclaimed_space_is_an_ordinary_failure(tmp_path, executable_oci_image):
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_root=pool.root, image_cache=tmp_path / "images")
    machine = await factory.create(MachineSpec(executable_oci_image, workdir="/app", storage_mb=64))
    try:
        result = await machine.run(
            Command(
                (
                    "/bin/sh",
                    "-c",
                    "/bin/dd if=/dev/zero of=/app/fill bs=1048576 count=128 2>/dev/null; "
                    "/bin/busybox rm /app/fill; exit 7",
                ),
                output_limit_bytes=0,
            )
        )
        assert result.exit_code == 7
        assert (await machine.run(Command(("/bin/sh", "-c", "true")))).exit_code == 0
    finally:
        await machine.close()
    assert disk_state(pool.root) == DiskState()


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
    assert disk_state(pool.root) == DiskState()


async def test_local_disk_expired_sandbox_releases_mounts_and_loop(tmp_path, executable_oci_image):
    pool = LocalDiskPool(tmp_path / "disks")
    factory = LocalMachineFactory(disk_root=pool.root, image_cache=tmp_path / "images", ttl=0.1)
    machine = await factory.create(MachineSpec(executable_oci_image, storage_mb=64))
    try:
        with pytest.raises(MachineTerminated):
            await machine.run(Command(("/bin/sleep", "3600")))
    finally:
        await machine.close()
    assert disk_state(pool.root) == DiskState()


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
    assert disk_state(root) == DiskState()
