# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pool verification must not stall concurrent trials or repeat the host probe."""

import asyncio
import io
import shutil
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from types import SimpleNamespace

import pytest
from shellbox.backends.local import disk, disk_lifecycle


@pytest.mark.asyncio
async def test_pool_verification_keeps_loop_responsive_and_reuses_concurrent_result(tmp_path, monkeypatch):
    started = threading.Event()
    released = threading.Event()
    real_stat = disk.os.stat

    def host_stat(path, *args, **kwargs):
        if str(path) == "/proc/self/ns/mnt":
            return SimpleNamespace(st_ino=123)
        return real_stat(path, *args, **kwargs)

    def probe(root):
        started.set()
        if not released.wait(timeout=5):
            raise TimeoutError("Host probe blocked the event loop")
        return SimpleNamespace(root=root)

    # Substitute the privileged host filesystem/probe boundaries, keeping the
    # public asynchronous acquisition and concurrent cache behavior real.
    monkeypatch.setattr(disk.os, "stat", host_stat)
    monkeypatch.setattr(disk, "LocalDiskPool", probe)
    first = asyncio.create_task(disk.verified_local_disk_pool(tmp_path))
    second = asyncio.create_task(disk.verified_local_disk_pool(tmp_path))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        heartbeat = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(heartbeat.set_result, None)
        await heartbeat
        assert not first.done() and not second.done()
    finally:
        released.set()
    first_pool, second_pool = await asyncio.gather(first, second)
    assert first_pool is second_pool
    assert await disk.verified_local_disk_pool(tmp_path / ".") is first_pool


def test_concurrent_guardian_cleanup_of_same_allocation_is_idempotent(tmp_path, monkeypatch):
    directory = tmp_path / "sandbox-concurrent"
    directory.mkdir()
    entered = threading.Barrier(2)
    removed = threading.Event()
    order = count()
    remove = shutil.rmtree

    def concurrent_remove(path):
        # Synchronize at the filesystem boundary so both guardians reach the
        # same allocation before either removes it. No mounts need substitution.
        position = next(order)
        entered.wait(timeout=5)
        if position:
            assert removed.wait(timeout=5)
            remove(path)
        else:
            try:
                remove(path)
            finally:
                removed.set()

    monkeypatch.setattr(disk_lifecycle.shutil, "rmtree", concurrent_remove)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(disk_lifecycle.reclaim, directory) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert not directory.exists()


@pytest.mark.parametrize("error", [PermissionError("denied"), FileNotFoundError("missing child")])
def test_guardian_cleanup_propagates_errors_while_allocation_remains(tmp_path, monkeypatch, error):
    directory = tmp_path / "sandbox-incomplete"
    directory.mkdir()

    def failed_remove(path):
        raise error

    monkeypatch.setattr(disk_lifecycle.shutil, "rmtree", failed_remove)
    with pytest.raises(type(error)):
        disk_lifecycle.reclaim(directory)
    assert directory.is_dir()


def test_startup_recovery_does_not_reclaim_guardian_during_directory_removal(tmp_path, monkeypatch):
    directory = tmp_path / "sandbox-guarded"
    directory.mkdir()
    remove = shutil.rmtree
    cleaning = False

    def command(*args, **kwargs):
        # Exercise real guardian/recovery ownership without privileged mount I/O.
        output = "/dev/loop999" if args[0][:3] == ("losetup", "--find", "--show") else ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    def remove_during_recovery(path):
        nonlocal cleaning
        if path == directory and not cleaning:
            cleaning = True
            # Recursive removal may unlink the lease before the other children.
            (directory / "lease").unlink(missing_ok=True)
            disk.LocalDiskPool._reclaim(SimpleNamespace(root=tmp_path))
            assert (directory / "data" / "shm").is_dir()
        remove(path)

    monkeypatch.setattr(disk_lifecycle.subprocess, "run", command)
    control = tmp_path / "loop-control"
    control.touch()
    monkeypatch.setattr(disk_lifecycle, "LOOP_CONTROL", control)
    monkeypatch.setattr(disk_lifecycle, "expose_free_loop_device", lambda control: None)
    monkeypatch.setattr(disk_lifecycle.shutil, "rmtree", remove_during_recovery)
    monkeypatch.setattr(disk_lifecycle.sys, "stdin", SimpleNamespace(buffer=io.BytesIO()))
    disk_lifecycle.guard(directory, storage_mb=1)
    assert cleaning and not directory.exists()


@pytest.mark.parametrize("mount_fails", [False, True])
def test_guardian_owns_explicit_loop_through_mount_failure_and_cleanup(tmp_path, monkeypatch, mount_fails):
    directory = tmp_path / "sandbox-explicit-loop"
    directory.mkdir()
    control = tmp_path / "loop-control"
    control.touch()
    devices = {"/dev/loop8": "/another-owner/disk.img"}
    mounts = set()
    autoclear = set()
    device = "/dev/loop19"
    nodes = {}
    sysfs = tmp_path / "sysfs" / "loop19"
    sysfs.mkdir(parents=True)
    (sysfs / "dev").write_text("7:19\n")
    real_lstat = disk_lifecycle.Path.lstat

    def create_node(path, mode, identity):
        nodes[str(path)] = SimpleNamespace(st_mode=mode, st_rdev=identity)

    def node_stat(path, *args, **kwargs):
        if str(path) == device:
            return nodes[device]
        return real_lstat(path, *args, **kwargs)

    def command(argv, **kwargs):
        output = ""
        error = ""
        if argv[0] == "mkfs.ext4":
            pass
        elif argv[:3] == ("losetup", "--find", "--show"):
            if device not in nodes:
                error = "failed to set up loop device: No such file or directory"
            else:
                devices[device] = str(directory / "disk.img")
                output = device
        elif argv[0] == "mount":
            # The affected worker cannot allocate a loop implicitly through mount.
            if argv[-2] != device or mount_fails:
                error = "failed to setup loop device"
            else:
                mounts.add(argv[-1])
        elif argv[:2] == ("losetup", "--detach"):
            if mounts:
                autoclear.add(argv[-1])
            else:
                devices.pop(argv[-1])
        elif argv[:2] == ("losetup", "--associated"):
            output = "\n".join(name for name, image in devices.items() if image == argv[2])
        elif argv[0] == "umount":
            mounts.remove(argv[1])
            for name in autoclear:
                devices.pop(name)
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=int(bool(error)), stdout=output, stderr=error)

    monkeypatch.setattr(disk_lifecycle, "LOOP_CONTROL", control, raising=False)
    monkeypatch.setattr(disk_lifecycle, "LOOP_SYSFS", sysfs.parent, raising=False)
    monkeypatch.setattr(disk_lifecycle.fcntl, "ioctl", lambda control, operation: 19)
    monkeypatch.setattr(disk_lifecycle.os, "mknod", create_node)
    monkeypatch.setattr(disk_lifecycle.Path, "lstat", node_stat)
    monkeypatch.setattr(disk_lifecycle.subprocess, "run", command)
    monkeypatch.setattr(disk_lifecycle.os.path, "ismount", lambda path: str(path) in mounts)
    monkeypatch.setattr(disk_lifecycle.sys, "stdin", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setenv("IRIS_NODE_NAME", "gpu-node-proof")
    if mount_fails:
        with pytest.raises(RuntimeError, match="node gpu-node-proof: failed to setup loop device"):
            disk_lifecycle.guard(directory, storage_mb=1)
    else:
        disk_lifecycle.guard(directory, storage_mb=1)
        assert device in autoclear
    assert devices == {"/dev/loop8": "/another-owner/disk.img"}
    assert not mounts and not directory.exists()


def test_loop_exposure_rejects_an_existing_path_with_the_wrong_identity(tmp_path, monkeypatch):
    sysfs = tmp_path / "loop19"
    sysfs.mkdir()
    (sysfs / "dev").write_text("7:19\n")
    mutations = []

    def existing_node(path, mode, identity):
        raise FileExistsError(path)

    monkeypatch.setattr(disk_lifecycle, "LOOP_SYSFS", tmp_path)
    monkeypatch.setattr(disk_lifecycle.fcntl, "ioctl", lambda control, operation: 19)
    monkeypatch.setattr(disk_lifecycle.os, "mknod", existing_node)
    monkeypatch.setattr(
        disk_lifecycle.Path,
        "lstat",
        lambda path: SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=disk_lifecycle.os.makedev(7, 8)),
    )
    monkeypatch.setattr(disk_lifecycle.os, "unlink", lambda path: mutations.append(path))
    with pytest.raises(RuntimeError, match="wrong device identity"):
        disk_lifecycle.expose_free_loop_device(io.BytesIO())
    assert mutations == []
