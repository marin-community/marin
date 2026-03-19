# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DockerRuntime mount resolution, staging, and container creation."""

from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerContainerHandle, DockerRuntime
from iris.cluster.runtime.types import ContainerConfig, MountKind, MountSpec


@pytest.fixture
def runtime(tmp_path):
    return DockerRuntime(cache_dir=tmp_path / "cache")


@pytest.fixture
def mock_bundle_store():
    store = Mock(spec=BundleStore)
    store.extract_bundle_to = Mock()
    return store


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


def test_resolve_mounts_tmpfs_has_no_host_path(tmp_path, runtime):
    """TMPFS mounts get empty host_path (Docker --tmpfs provides per-container isolation)."""
    mounts = [MountSpec(container_path="/tmp", kind=MountKind.TMPFS)]
    resolved = runtime.resolve_mounts(mounts)

    assert len(resolved) == 1
    assert resolved[0].host_path == ""
    assert resolved[0].container_path == "/tmp"
    assert resolved[0].kind == MountKind.TMPFS


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


def test_build_container_gets_memory_limit(monkeypatch, tmp_path, runtime):
    """BUILD containers should have --memory cgroup limits to prevent worker OOM."""
    from iris.rpc import cluster_pb2

    docker_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        docker_cmds.append(cmd)
        # docker image inspect (ensure_image) and docker create both go through here
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="fake-container-id\n", stderr="")
        if cmd[:2] == ["docker", "start"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout='{"Running": false, "ExitCode": 0}', stderr=""
            )
        if cmd[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    workdir = tmp_path / "task-workdir"
    workdir.mkdir()

    resources = cluster_pb2.ResourceSpecProto(
        cpu_millicores=4000,
        memory_bytes=32 * 1024**3,  # 32 GB
    )
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    entrypoint.setup_commands[:] = ["echo setup"]

    config = ContainerConfig(
        image="test-image:latest",
        entrypoint=entrypoint,
        env={},
        resources=resources,
        mounts=[MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=1024 * 1024 * 512)],
        workdir_host_path=workdir,
    )

    resolved = runtime.resolve_mounts(config.mounts, workdir_host_path=workdir)
    handle = DockerContainerHandle(config=config, runtime=runtime, _resolved_mounts=resolved)
    handle.build()

    # Find the docker create command
    create_cmds = [cmd for cmd in docker_cmds if cmd[:2] == ["docker", "create"]]
    assert len(create_cmds) == 1
    create_cmd = create_cmds[0]

    # Verify --memory flag is present on the build container
    assert "--memory" in create_cmd, "BUILD container must have --memory cgroup limit"
    memory_idx = create_cmd.index("--memory")
    memory_value = create_cmd[memory_idx + 1]
    expected_mb = (32 * 1024**3) // (1024 * 1024)
    assert memory_value == f"{expected_mb}m"
