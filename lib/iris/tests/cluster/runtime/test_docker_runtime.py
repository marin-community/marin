# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DockerRuntime TPU command construction."""

from __future__ import annotations

import subprocess

from iris.cluster.runtime.docker import DockerContainerHandle, DockerRuntime
from iris.cluster.runtime.types import ContainerConfig
from iris.rpc import cluster_pb2


def _entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    ep = cluster_pb2.RuntimeEntrypoint()
    ep.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=["python", "-c", "print('ok')"]))
    return ep


def _tpu_resources() -> cluster_pb2.ResourceSpecProto:
    device = cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v4-8", count=4))
    return cluster_pb2.ResourceSpecProto(device=device, cpu_millicores=1000, memory_bytes=1024 * 1024 * 1024)


def _cpu_resources() -> cluster_pb2.ResourceSpecProto:
    return cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024 * 1024 * 1024)


def _make_handle(resources: cluster_pb2.ResourceSpecProto) -> DockerContainerHandle:
    config = ContainerConfig(
        image="ghcr.io/example/task:latest",
        entrypoint=_entrypoint(),
        env={},
        resources=resources,
    )
    runtime = DockerRuntime()
    return DockerContainerHandle(config=config, runtime=runtime)


def test_tpu_create_uses_privileged_without_hardening_flags(monkeypatch):
    handle = _make_handle(_tpu_resources())
    monkeypatch.setattr(handle.runtime, "ensure_image", lambda _image: None)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    container_id = handle._docker_create(command=["python", "-c", "print('ok')"], include_resources=True)

    assert container_id == "container-id"
    assert calls
    cmd = calls[0]
    assert "--privileged" in cmd
    assert "--cap-drop" not in cmd
    assert "--security-opt" not in cmd
    assert "--cap-add=SYS_RESOURCE" in cmd


def test_non_tpu_create_keeps_hardening_flags(monkeypatch):
    handle = _make_handle(_cpu_resources())
    monkeypatch.setattr(handle.runtime, "ensure_image", lambda _image: None)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    container_id = handle._docker_create(command=["python", "-c", "print('ok')"], include_resources=True)

    assert container_id == "container-id"
    assert calls
    cmd = calls[0]
    assert "--privileged" not in cmd
    assert "--security-opt" in cmd
    assert "no-new-privileges" in cmd
    assert "--cap-drop" in cmd
    assert "ALL" in cmd
    assert "--cap-add" in cmd
    assert "SYS_PTRACE" in cmd
