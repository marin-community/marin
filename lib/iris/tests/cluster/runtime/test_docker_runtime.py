# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DockerRuntime prepare_workdir / cleanup_workdir tmpfs management."""

import subprocess

import pytest

from iris.cluster.runtime.docker import DockerRuntime


@pytest.fixture
def runtime():
    return DockerRuntime()


def test_prepare_workdir_mounts_tmpfs(monkeypatch, tmp_path, runtime):
    """On Linux with disk_bytes > 0 and no existing mount, subprocess.run is called to mount tmpfs."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "task-workdir"
    runtime.prepare_workdir(workdir=workdir, disk_bytes=1024 * 1024 * 512)

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "mount"
    assert "-t" in cmd and "tmpfs" in cmd
    assert f"size={1024 * 1024 * 512}" in cmd[cmd.index("-o") + 1]
    assert str(workdir) in cmd


def test_prepare_workdir_skips_when_no_disk(monkeypatch, tmp_path, runtime):
    """disk_bytes=0 means no limit, so no subprocess call."""
    calls: list = []
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    runtime.prepare_workdir(workdir=tmp_path / "w", disk_bytes=0)
    assert len(calls) == 0


def test_prepare_workdir_rejects_non_linux(monkeypatch, tmp_path, runtime):
    """tmpfs mounts require Linux; other platforms should raise RuntimeError."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "darwin")

    with pytest.raises(RuntimeError, match="Linux"):
        runtime.prepare_workdir(workdir=tmp_path / "w", disk_bytes=1024)


def test_prepare_workdir_skips_already_mounted(monkeypatch, tmp_path, runtime):
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
    runtime.prepare_workdir(workdir=workdir, disk_bytes=1024)
    assert len(calls) == 0


def test_cleanup_workdir_unmounts(monkeypatch, tmp_path, runtime):
    """When workdir is a mountpoint, cleanup_workdir calls umount."""
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "task-workdir"
    runtime.cleanup_workdir(workdir)

    assert len(calls) == 1
    assert calls[0] == ["umount", str(workdir)]


def test_cleanup_workdir_noop_when_not_mounted(monkeypatch, tmp_path, runtime):
    """When workdir is not a mountpoint, cleanup_workdir does nothing."""
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)

    calls: list = []
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    runtime.cleanup_workdir(tmp_path / "task-workdir")
    assert len(calls) == 0


def test_prepare_workdir_raises_on_mount_failure(monkeypatch, tmp_path, runtime):
    """Failed mount command propagates as RuntimeError."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="mount: permission denied"),
    )

    with pytest.raises(RuntimeError, match="permission denied"):
        runtime.prepare_workdir(workdir=tmp_path / "w", disk_bytes=1024)
