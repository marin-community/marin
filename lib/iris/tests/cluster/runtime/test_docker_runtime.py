# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DockerRuntime mount resolution, staging, and container creation."""

from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest
from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerRuntime, _DockerProfileDispatch
from iris.cluster.runtime.types import MountKind, MountSpec


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


# ---------------------------------------------------------------------------
# _DockerProfileDispatch: in-container `timeout --signal=KILL` reaps a hung
# profiler and a SIGCONT sweep clears any group-stop it leaves, so an orphaned
# `docker exec` can never hold the target ptrace-stopped (py-spy#390).
# ---------------------------------------------------------------------------


def _docker_exec_payload(cmd: list[str]) -> list[str]:
    """Strip the leading ``docker exec <cid>`` wrapper from a captured argv."""
    assert cmd[:2] == ["docker", "exec"]
    return cmd[3:]


def test_exec_profiler_wraps_in_kill_watchdog_with_longer_host_backstop(monkeypatch):
    """The profiler runs under `timeout --signal=KILL`; the host timeout is longer."""
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    dispatch = _DockerProfileDispatch("cid123")
    dispatch.exec_profiler(["/app/.venv/bin/py-spy", "record", "--pid", "1"], sample_timeout=35)

    cmd, kwargs = calls[0]
    payload = _docker_exec_payload(cmd)
    assert payload[:3] == ["timeout", "--signal=KILL", "35"]
    assert payload[3].endswith("py-spy")
    # Host backstop outlives the in-container watchdog so the watchdog wins.
    assert kwargs["timeout"] > 35


def test_exec_profiler_sends_sigcont_sweep_after_profiler_exits(monkeypatch):
    """After the profiler exits, a SIGCONT sweep clears any lingering group-stop."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    _DockerProfileDispatch("cid123").exec_profiler(["/app/.venv/bin/py-spy", "dump"], sample_timeout=30)

    payloads = [_docker_exec_payload(c) for c in calls]
    assert payloads[0][0] == "timeout"  # profiler first, under the watchdog
    sweep = payloads[-1]
    assert sweep[0] == "sh" and "kill -CONT" in sweep[-1]


def test_exec_profiler_sigcont_runs_even_when_profiler_raises(monkeypatch):
    """A host-side timeout must not skip the SIGCONT recovery."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if _docker_exec_payload(cmd)[0] == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("iris.cluster.runtime.docker.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired):
        _DockerProfileDispatch("cid123").exec_profiler(["py-spy", "dump"], sample_timeout=30)

    assert any(_docker_exec_payload(c)[0] == "sh" and "kill -CONT" in c[-1] for c in calls)
