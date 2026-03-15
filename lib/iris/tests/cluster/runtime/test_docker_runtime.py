# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DockerRuntime tmpfs management via stage_bundle and release_tmpfs."""

import logging
import subprocess
from unittest.mock import Mock

import pytest

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import WorkdirSpec


@pytest.fixture
def runtime():
    return DockerRuntime()


@pytest.fixture
def mock_bundle_store():
    store = Mock(spec=BundleStore)
    store.extract_bundle_to = Mock()
    store.write_workdir_files = Mock()
    return store


def test_stage_bundle_mounts_tmpfs(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """On Linux with disk_bytes > 0, stage_bundle mounts tmpfs before extracting."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    spec = WorkdirSpec(disk_bytes=1024 * 1024 * 512, tmpfs=True)
    runtime.stage_bundle(
        bundle_id="abc",
        workdir=workdir,
        workdir_files={},
        bundle_store=mock_bundle_store,
        workdir_spec=spec,
    )

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "mount"
    assert "-t" in cmd and "tmpfs" in cmd
    assert f"size={1024 * 1024 * 512}" in cmd[cmd.index("-o") + 1]
    assert str(workdir) in cmd

    mock_bundle_store.extract_bundle_to.assert_called_once_with("abc", workdir)
    assert workdir in runtime._tmpfs_mounts


def test_stage_bundle_no_tmpfs_without_spec(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """Without workdir_spec, no subprocess call for mounting."""
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
    mock_bundle_store.extract_bundle_to.assert_called_once()


def test_stage_bundle_no_tmpfs_when_zero_disk(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """disk_bytes=0 means no limit, so no tmpfs mount."""
    calls: list = []
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    workdir = tmp_path / "w"
    workdir.mkdir()
    spec = WorkdirSpec(disk_bytes=0)
    runtime.stage_bundle(
        bundle_id="",
        workdir=workdir,
        workdir_files={},
        bundle_store=mock_bundle_store,
        workdir_spec=spec,
    )
    assert len(calls) == 0


def test_stage_bundle_rejects_non_linux(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """tmpfs mounts require Linux; other platforms should raise RuntimeError."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "darwin")

    spec = WorkdirSpec(disk_bytes=1024, tmpfs=True)
    with pytest.raises(RuntimeError, match="Linux"):
        runtime.stage_bundle(
            bundle_id="",
            workdir=tmp_path / "w",
            workdir_files={},
            bundle_store=mock_bundle_store,
            workdir_spec=spec,
        )


def test_stage_bundle_skips_already_mounted(monkeypatch, tmp_path, runtime, mock_bundle_store):
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
    spec = WorkdirSpec(disk_bytes=1024, tmpfs=True)
    runtime.stage_bundle(
        bundle_id="",
        workdir=workdir,
        workdir_files={},
        bundle_store=mock_bundle_store,
        workdir_spec=spec,
    )
    assert len(calls) == 0


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
    assert workdir not in runtime._tmpfs_mounts


def test_release_tmpfs_noop_when_not_tracked(monkeypatch, tmp_path, runtime):
    """release_tmpfs does nothing for workdirs it didn't mount."""
    calls: list = []
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    runtime.release_tmpfs(tmp_path / "task-workdir")
    assert len(calls) == 0


def test_stage_bundle_raises_on_mount_failure(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """Failed mount command propagates as RuntimeError."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="mount: permission denied"),
    )

    spec = WorkdirSpec(disk_bytes=1024, tmpfs=True)
    with pytest.raises(RuntimeError, match="permission denied"):
        runtime.stage_bundle(
            bundle_id="",
            workdir=tmp_path / "w",
            workdir_files={},
            bundle_store=mock_bundle_store,
            workdir_spec=spec,
        )


def test_release_tmpfs_warns_on_umount_failure(monkeypatch, tmp_path, runtime, caplog):
    """Failed umount logs a warning instead of raising."""
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
    assert workdir not in runtime._tmpfs_mounts
