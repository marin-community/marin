# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DockerRuntime tmpfs management via stage_bundle and release_tmpfs."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerRuntime, ResolvedMount
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


def test_stage_bundle_mounts_tmpfs(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """On Linux with size_bytes > 0, stage_bundle mounts tmpfs before extracting."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda p: False)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    mount = MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024 * 1024 * 512)
    runtime.stage_bundle(
        bundle_id="abc",
        workdir=workdir,
        workdir_files={},
        bundle_store=mock_bundle_store,
        workdir_mount=mount,
    )

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "mount"
    assert "-t" in cmd and "tmpfs" in cmd
    assert f"size={1024 * 1024 * 512}" in cmd[cmd.index("-o") + 1]
    assert str(workdir) in cmd

    mock_bundle_store.extract_bundle_to.assert_called_once_with("abc", workdir)
    assert workdir in runtime._tmpfs_mounts


def test_stage_bundle_no_tmpfs_without_mount(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """Without workdir_mount, no subprocess call for mounting."""
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


def test_stage_bundle_no_tmpfs_when_zero_size(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """size_bytes=0 means no limit, so no tmpfs mount."""
    calls: list = []
    monkeypatch.setattr(
        "iris.cluster.runtime.docker.subprocess.run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    workdir = tmp_path / "w"
    workdir.mkdir()
    mount = MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=0)
    runtime.stage_bundle(
        bundle_id="",
        workdir=workdir,
        workdir_files={},
        bundle_store=mock_bundle_store,
        workdir_mount=mount,
    )
    assert len(calls) == 0


def test_stage_bundle_rejects_non_linux(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """tmpfs mounts require Linux; other platforms should raise RuntimeError."""
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "darwin")

    mount = MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)
    with pytest.raises(RuntimeError, match="Linux"):
        runtime.stage_bundle(
            bundle_id="",
            workdir=tmp_path / "w",
            workdir_files={},
            bundle_store=mock_bundle_store,
            workdir_mount=mount,
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
    mount = MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)
    runtime.stage_bundle(
        bundle_id="",
        workdir=workdir,
        workdir_files={},
        bundle_store=mock_bundle_store,
        workdir_mount=mount,
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

    mount = MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024)
    with pytest.raises(RuntimeError, match="permission denied"):
        runtime.stage_bundle(
            bundle_id="",
            workdir=tmp_path / "w",
            workdir_files={},
            bundle_store=mock_bundle_store,
            workdir_mount=mount,
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


def test_resolve_mounts_workdir(tmp_path):
    """WORKDIR mount resolves to the provided workdir_host_path."""
    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR)]
    resolved = runtime.resolve_mounts(mounts, workdir_host_path=workdir)
    assert resolved == [ResolvedMount(str(workdir), "/app", "rw", MountKind.WORKDIR)]


def test_resolve_mounts_cache(tmp_path):
    """CACHE mount creates a host directory under fast_io_dir."""
    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    mounts = [MountSpec(container_path="/uv/cache", kind=MountKind.CACHE)]
    resolved = runtime.resolve_mounts(mounts)
    assert len(resolved) == 1
    rm = resolved[0]
    assert rm.container_path == "/uv/cache"
    assert rm.mode == "rw"
    assert rm.kind == MountKind.CACHE
    assert Path(rm.host_path).exists()


def test_resolve_mounts_workdir_requires_host_path(tmp_path):
    """WORKDIR mount without workdir_host_path raises RuntimeError."""
    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR)]
    with pytest.raises(RuntimeError, match="workdir_host_path"):
        runtime.resolve_mounts(mounts)


def test_resolve_mounts_read_only(tmp_path):
    """read_only=True produces 'ro' mode."""
    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    mounts = [MountSpec(container_path="/data", kind=MountKind.CACHE, read_only=True)]
    resolved = runtime.resolve_mounts(mounts)
    assert resolved[0].mode == "ro"


def test_create_container_resolves_mounts(tmp_path):
    """create_container builds resolved mounts on the handle from MountSpecs."""
    from iris.cluster.runtime.types import ContainerConfig
    from iris.rpc import cluster_pb2

    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    ep = cluster_pb2.RuntimeEntrypoint()
    ep.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=["echo", "hi"]))

    config = ContainerConfig(
        image="busybox",
        entrypoint=ep,
        env={},
        mounts=[
            MountSpec("/app", kind=MountKind.WORKDIR, size_bytes=0),
            MountSpec("/uv/cache", kind=MountKind.CACHE),
        ],
        workdir_host_path=workdir,
    )

    handle = runtime.create_container(config)
    assert len(handle._resolved_mounts) == 2
    assert handle._resolved_mounts[0].kind == MountKind.WORKDIR
    assert handle._resolved_mounts[0].host_path == str(workdir)
    assert handle._resolved_mounts[1].kind == MountKind.CACHE
    assert Path(handle._resolved_mounts[1].host_path).is_dir()


def test_cleanup_releases_workdir_and_tmpfs_mounts(tmp_path):
    """cleanup() calls release_tmpfs for WORKDIR and TMPFS resolved mounts."""
    from iris.cluster.runtime.types import ContainerConfig
    from iris.rpc import cluster_pb2

    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    ep = cluster_pb2.RuntimeEntrypoint()
    ep.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=["true"]))

    config = ContainerConfig(
        image="busybox",
        entrypoint=ep,
        env={},
        mounts=[
            MountSpec("/app", kind=MountKind.WORKDIR),
            MountSpec("/scratch", kind=MountKind.TMPFS),
            MountSpec("/uv/cache", kind=MountKind.CACHE),
        ],
        workdir_host_path=workdir,
    )

    handle = runtime.create_container(config)
    # Manually track which paths release_tmpfs was called on
    released: list[Path] = []
    original_release = runtime.release_tmpfs
    runtime.release_tmpfs = lambda p: (released.append(p), original_release(p))

    handle.cleanup()

    # Should release WORKDIR and TMPFS but not CACHE
    assert len(released) == 2
    assert released[0] == workdir
    assert released[1].name == "scratch"
