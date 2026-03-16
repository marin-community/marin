# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DockerRuntime tmpfs management via resolve_mounts and release_tmpfs."""

import concurrent.futures
import logging
import subprocess
import threading
import time
from unittest.mock import Mock

import pytest

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DEFAULT_WORKDIR_DISK_BYTES, DockerRuntime
from iris.cluster.runtime.types import MountKind, MountSpec


@pytest.fixture
def runtime(tmp_path):
    return DockerRuntime(cache_dir=tmp_path / "cache")


@pytest.fixture
def mock_bundle_store():
    store = Mock(spec=BundleStore)
    store.extract_bundle_to = Mock()
    store.write_workdir_files = Mock()
    return store


def test_resolve_mounts_workdir_mounts_tmpfs(monkeypatch, tmp_path, runtime):
    """resolve_mounts mounts tmpfs on the workdir host path for WORKDIR mounts."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024 * 1024 * 512)]
    resolved = runtime.resolve_mounts(mounts, workdir_host_path=workdir)

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "mount"
    assert "-t" in cmd and "tmpfs" in cmd
    assert f"size={1024 * 1024 * 512}" in cmd[cmd.index("-o") + 1]
    assert str(workdir) in cmd

    assert len(resolved) == 1
    assert resolved[0].host_path == str(workdir)
    assert resolved[0].container_path == "/app"
    assert resolved[0].kind == MountKind.WORKDIR


def test_resolve_mounts_workdir_uses_default_size_when_zero(monkeypatch, tmp_path, runtime):
    """size_bytes=0 mounts tmpfs with the default 10GB size."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "w"
    workdir.mkdir()
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=0)]
    runtime.resolve_mounts(mounts, workdir_host_path=workdir)

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "mount"
    assert f"size={DEFAULT_WORKDIR_DISK_BYTES}" in cmd[cmd.index("-o") + 1]


def test_resolve_mounts_workdir_rejects_non_linux(monkeypatch, tmp_path, runtime):
    """tmpfs mounts require Linux; other platforms should raise RuntimeError."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "darwin")

    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)]
    with pytest.raises(RuntimeError, match="Linux"):
        runtime.resolve_mounts(mounts, workdir_host_path=tmp_path / "w")


def test_resolve_mounts_workdir_skips_already_mounted(monkeypatch, tmp_path, runtime):
    """If workdir is already a mountpoint, no mount call is made."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: True)

    calls: list = []
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)]
    runtime.resolve_mounts(mounts, workdir_host_path=workdir)
    assert len(calls) == 0


def test_resolve_mounts_workdir_raises_on_mount_failure(monkeypatch, tmp_path, runtime):
    """Failed mount command propagates as RuntimeError."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="mount: permission denied"),
    )

    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)]
    with pytest.raises(RuntimeError, match="permission denied"):
        runtime.resolve_mounts(mounts, workdir_host_path=tmp_path / "w")


def test_resolve_mounts_workdir_requires_host_path(tmp_path):
    """WORKDIR mount without workdir_host_path raises RuntimeError."""
    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR)]
    with pytest.raises(RuntimeError, match="workdir_host_path"):
        runtime.resolve_mounts(mounts)


def test_stage_bundle_stages_without_tmpfs(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """stage_bundle extracts bundle and writes workdir files without mounting tmpfs."""
    calls: list = []
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    workdir = tmp_path / "w"
    workdir.mkdir()
    runtime.stage_bundle(
        bundle_id="abc",
        workdir=workdir,
        workdir_files={},
        bundle_store=mock_bundle_store,
    )
    assert len(calls) == 0
    mock_bundle_store.extract_bundle_to.assert_called_once_with("abc", workdir)
    mock_bundle_store.write_workdir_files.assert_called_once_with(workdir, {})


def test_release_tmpfs_unmounts(monkeypatch, tmp_path, runtime):
    """release_tmpfs unmounts tracked tmpfs workdirs."""
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "task-workdir"
    runtime._tmpfs_mounts.add(workdir)
    runtime.release_tmpfs(workdir)

    assert len(calls) == 1
    assert calls[0] == ["umount", str(workdir)]


def test_release_tmpfs_noop_when_not_tracked(monkeypatch, tmp_path, runtime):
    """release_tmpfs does nothing for workdirs it didn't mount."""
    calls: list = []
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    runtime.release_tmpfs(tmp_path / "task-workdir")
    assert len(calls) == 0


def test_release_tmpfs_keeps_tracking_on_umount_failure(monkeypatch, tmp_path, runtime, caplog):
    """Failed umount logs a warning and keeps the mount tracked for retry."""
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: True)
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="device busy"),
    )

    workdir = tmp_path / "task-workdir"
    runtime._tmpfs_mounts.add(workdir)
    with caplog.at_level(logging.WARNING, logger="iris.cluster.runtime.docker"):
        runtime.release_tmpfs(workdir)

    assert "device busy" in caplog.text
    # Mount stays tracked so a later cleanup pass can retry
    assert workdir in runtime._tmpfs_mounts


def test_mount_tmpfs_recovers_on_concurrent_race(monkeypatch, tmp_path, runtime):
    """When mount fails but the path is now a mountpoint, treat as success (concurrent race)."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")

    # First ismount check returns False (pre-mount), second returns True (post-failure recheck)
    ismount_calls = iter([False, True])
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: next(ismount_calls))

    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="tmpfs already mounted on /dev"),
    )

    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)]
    # Should NOT raise — the post-failure ismount check sees it's mounted
    runtime.resolve_mounts(mounts, workdir_host_path=workdir)
    assert workdir in runtime._tmpfs_mounts


def test_mount_tmpfs_fails_when_not_mounted_after_error(monkeypatch, tmp_path, runtime):
    """When mount fails and the path is still not a mountpoint, raise RuntimeError."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="permission denied"),
    )

    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)]
    with pytest.raises(RuntimeError, match="permission denied"):
        runtime.resolve_mounts(mounts, workdir_host_path=workdir)


def test_concurrent_resolve_mounts_serialized(monkeypatch, tmp_path):
    """Multiple threads resolving WORKDIR mounts are serialized by _mount_lock."""
    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)

    active = threading.Event()
    overlap_detected = []

    def fake_run(cmd, **kwargs):
        if active.is_set():
            overlap_detected.append(True)
        active.set()
        time.sleep(0.01)
        active.clear()
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)]

    def do_resolve(i):
        workdir = tmp_path / f"w{i}"
        workdir.mkdir()
        runtime.resolve_mounts(mounts, workdir_host_path=workdir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(do_resolve, range(8)))

    assert not overlap_detected, "mount calls overlapped — _mount_lock is not working"
    assert len(runtime._tmpfs_mounts) == 8
