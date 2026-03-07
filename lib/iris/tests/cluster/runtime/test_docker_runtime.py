# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DockerRuntime host-side workdir preparation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from iris.cluster.runtime.docker import DockerRuntime
from iris.rpc import cluster_pb2


def test_prepare_workdir_mounts_tmpfs_for_disk_bound_tasks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = DockerRuntime()
    workdir = tmp_path / "workdir"
    commands: list[list[str]] = []

    def fake_run(cmd, capture_output=False, text=False, check=False):
        del capture_output, text, check
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "linux")
    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda _: False)
    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    runtime.prepare_workdir(
        workdir=workdir,
        resources=cluster_pb2.ResourceSpecProto(disk_bytes=8 * 1024**3),
    )

    assert workdir.is_dir()
    assert commands == [
        [
            "mount",
            "-t",
            "tmpfs",
            "-o",
            f"size={8 * 1024**3},nodev,nosuid",
            "tmpfs",
            str(workdir),
        ]
    ]


def test_prepare_workdir_skips_mount_without_disk_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = DockerRuntime()
    workdir = tmp_path / "workdir"
    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", lambda *args, **kwargs: pytest.fail("mount"))

    runtime.prepare_workdir(workdir=workdir, resources=cluster_pb2.ResourceSpecProto())


def test_prepare_workdir_rejects_non_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = DockerRuntime()
    workdir = tmp_path / "workdir"
    monkeypatch.setattr("iris.cluster.runtime.docker.sys.platform", "darwin")

    with pytest.raises(RuntimeError, match="Linux tmpfs mounts"):
        runtime.prepare_workdir(
            workdir=workdir,
            resources=cluster_pb2.ResourceSpecProto(disk_bytes=1024),
        )


def test_cleanup_workdir_unmounts_mountpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = DockerRuntime()
    workdir = tmp_path / "workdir"
    commands: list[list[str]] = []

    def fake_run(cmd, capture_output=False, text=False, check=False):
        del capture_output, text, check
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("iris.cluster.runtime.docker.os.path.ismount", lambda _: True)
    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    runtime.cleanup_workdir(workdir)

    assert commands == [["umount", str(workdir)]]
