# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DockerRuntime mount resolution and TPU docker flags."""

from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerContainerHandle, DockerRuntime
from iris.cluster.runtime.types import ContainerConfig, MountKind, MountSpec
from iris.rpc import cluster_pb2


@pytest.fixture
def runtime(tmp_path):
    return DockerRuntime(cache_dir=tmp_path / "cache")


@pytest.fixture
def mock_bundle_store():
    store = Mock(spec=BundleStore)
    store.extract_bundle_to = Mock()
    store.write_workdir_files = Mock()
    return store


def _entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    ep = cluster_pb2.RuntimeEntrypoint()
    ep.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=["python", "-c", "print('ok')"]))
    return ep


def _tpu_resources() -> cluster_pb2.ResourceSpecProto:
    device = cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v4-8", count=4))
    return cluster_pb2.ResourceSpecProto(device=device, cpu_millicores=1000, memory_bytes=1024 * 1024 * 1024)


def _cpu_resources() -> cluster_pb2.ResourceSpecProto:
    return cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024 * 1024 * 1024)


def _make_handle(
    runtime: DockerRuntime,
    tmp_path,
    resources: cluster_pb2.ResourceSpecProto,
) -> DockerContainerHandle:
    workdir = tmp_path / "task-workdir"
    workdir.mkdir(exist_ok=True)
    config = ContainerConfig(
        image="ghcr.io/example/task:latest",
        entrypoint=_entrypoint(),
        env={},
        resources=resources,
        mounts=[MountSpec(container_path="/app", kind=MountKind.WORKDIR)],
        workdir_host_path=workdir,
    )
    return runtime.create_container(config)


def test_resolve_mounts_workdir(monkeypatch, tmp_path, runtime):
    """resolve_mounts resolves WORKDIR to the given host path."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024 * 1024 * 512)]
    resolved = runtime.resolve_mounts(mounts, workdir_host_path=workdir)

    assert len(calls) == 0
    assert len(resolved) == 1
    assert resolved[0].host_path == str(workdir)
    assert resolved[0].container_path == "/app"
    assert resolved[0].kind == MountKind.WORKDIR


def test_resolve_mounts_cache_uses_cache_dir(tmp_path, runtime):
    """CACHE mounts resolve to subdirectories under cache_dir."""
    mounts = [MountSpec(container_path="/root/.cache/uv", kind=MountKind.CACHE)]
    resolved = runtime.resolve_mounts(mounts)

    assert len(resolved) == 1
    assert resolved[0].host_path.startswith(str(tmp_path / "cache"))
    assert resolved[0].container_path == "/root/.cache/uv"
    assert resolved[0].kind == MountKind.CACHE


def test_resolve_mounts_workdir_requires_host_path(tmp_path):
    """WORKDIR mount without workdir_host_path raises RuntimeError."""
    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR)]
    with pytest.raises(RuntimeError, match="workdir_host_path"):
        runtime.resolve_mounts(mounts)


def test_prepare_workdir_is_noop(tmp_path, runtime):
    """prepare_workdir is a no-op since cache_dir is already on /dev/shm."""
    workdir = tmp_path / "task-workdir"
    workdir.mkdir()
    runtime.prepare_workdir(workdir, disk_bytes=1024 * 1024 * 512)


def test_stage_bundle(monkeypatch, tmp_path, runtime, mock_bundle_store):
    """stage_bundle extracts bundle and writes workdir files."""
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


def test_tpu_create_adds_discovered_tpu_devices(monkeypatch, runtime, tmp_path):
    handle = _make_handle(runtime, tmp_path, _tpu_resources())
    monkeypatch.setattr(runtime, "ensure_image", lambda _image: None)
    discovered: list[str] = []

    def fake_discover() -> list[str]:
        discovered.append("called")
        return ["/dev/vfio:/dev/vfio", "/dev/accel0:/dev/accel0", "/dev/accel1:/dev/accel1"]

    monkeypatch.setattr(
        "iris.cluster.runtime.docker._discover_tpu_device_mappings",
        fake_discover,
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    container_id = handle._docker_create(command=["python", "-c", "print('ok')"], include_resources=True)

    assert container_id == "container-id"
    assert calls
    assert discovered == ["called"]
    cmd = calls[0]
    assert cmd.count("--device") == 3
    assert "/dev/vfio:/dev/vfio" in cmd
    assert "/dev/accel0:/dev/accel0" in cmd
    assert "/dev/accel1:/dev/accel1" in cmd
    assert "memlock=68719476736:68719476736" in cmd


def test_non_tpu_create_skips_tpu_device_discovery(monkeypatch, runtime, tmp_path):
    handle = _make_handle(runtime, tmp_path, _cpu_resources())
    monkeypatch.setattr(runtime, "ensure_image", lambda _image: None)
    monkeypatch.setattr(
        "iris.cluster.runtime.docker._discover_tpu_device_mappings",
        lambda: (_ for _ in ()).throw(AssertionError("non-TPU should not discover TPU devices")),
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    container_id = handle._docker_create(command=["python", "-c", "print('ok')"], include_resources=True)

    assert container_id == "container-id"
    assert calls
    cmd = calls[0]
    assert "--device" not in cmd
    assert "--cap-add=SYS_RESOURCE" not in cmd
