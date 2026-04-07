# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProcessRuntime mount resolution and TMPFS cleanup."""

from pathlib import Path

from iris.cluster.runtime.process import _resolve_mount_map
from iris.cluster.runtime.types import ContainerConfig, MountKind, MountSpec
from iris.rpc import job_pb2


def _make_config(mounts: list[MountSpec], workdir_host_path: Path | None = None) -> ContainerConfig:
    """Build a minimal ContainerConfig for mount resolution tests."""
    return ContainerConfig(
        image="test:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(),
        env={},
        mounts=mounts,
        workdir_host_path=workdir_host_path,
        task_id="test-task",
    )


def test_tmpfs_mount_creates_unique_dirs(tmp_path):
    """Each _resolve_mount_map call creates a unique TMPFS directory."""
    mounts = [MountSpec("/tmp", kind=MountKind.TMPFS)]
    config = _make_config(mounts)

    map1 = _resolve_mount_map(config, cache_dir=tmp_path)
    map2 = _resolve_mount_map(config, cache_dir=tmp_path)

    assert map1["/tmp"] != map2["/tmp"], "TMPFS dirs must be unique per call"
    assert Path(map1["/tmp"]).exists()
    assert Path(map2["/tmp"]).exists()


def test_tmpfs_mount_cache_mount_independence(tmp_path):
    """TMPFS and CACHE mounts resolve independently."""
    mounts = [
        MountSpec("/tmp", kind=MountKind.TMPFS),
        MountSpec("/root/.cache/uv", kind=MountKind.CACHE),
    ]
    config = _make_config(mounts)

    mount_map = _resolve_mount_map(config, cache_dir=tmp_path)

    assert "/tmp" in mount_map
    assert "/root/.cache/uv" in mount_map
    assert mount_map["/tmp"] != mount_map["/root/.cache/uv"]


def test_process_handle_cleanup_removes_tmpfs(tmp_path):
    """ProcessContainerHandle.cleanup() removes TMPFS directories."""
    from iris.cluster.runtime.process import ProcessRuntime

    runtime = ProcessRuntime(cache_dir=tmp_path)
    mounts = [MountSpec("/tmp", kind=MountKind.TMPFS)]
    config = _make_config(mounts)
    handle = runtime.create_container(config)

    # Simulate what run() does: resolve mounts and track tmpfs dirs
    mount_map = _resolve_mount_map(config, cache_dir=tmp_path)
    tmpfs_path = Path(mount_map["/tmp"])
    handle._tmpfs_dirs.append(tmpfs_path)

    assert tmpfs_path.exists()
    handle.cleanup()
    assert not tmpfs_path.exists(), "cleanup() must remove TMPFS directories"
