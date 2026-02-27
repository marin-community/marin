# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KubernetesRuntime.

These tests mock kubectl subprocess calls and validate manifest generation,
including GPU resource requests, hostNetwork, tolerations, and S3 secret refs.
"""

from __future__ import annotations

import json
import subprocess


from iris.cluster.runtime.kubernetes import KubernetesRuntime
from iris.cluster.runtime.types import ContainerConfig, ContainerErrorKind, ContainerPhase
from iris.rpc import cluster_pb2


def _make_entrypoint(argv: list[str], setup_commands: list[str] | None = None) -> cluster_pb2.RuntimeEntrypoint:
    ep = cluster_pb2.RuntimeEntrypoint()
    ep.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=argv))
    if setup_commands:
        ep.setup_commands.extend(setup_commands)
    return ep


def _make_config(
    *,
    gpu_count: int = 0,
    gpu_name: str = "",
    network_mode: str = "host",
) -> ContainerConfig:
    resources = None
    if gpu_count > 0:
        device = cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(count=gpu_count, variant=gpu_name))
        resources = cluster_pb2.ResourceSpecProto(device=device)
    return ContainerConfig(
        image="ghcr.io/example/task:latest",
        entrypoint=_make_entrypoint(["python", "-c", "print('ok')"], setup_commands=["echo setup"]),
        env={"FOO": "bar"},
        workdir="/app",
        task_id="job/task/0",
        network_mode=network_mode,
        resources=resources,
    )


def _capture_manifest(monkeypatch) -> list[dict]:
    """Patch subprocess.run to capture the manifest JSON from kubectl apply."""
    manifests: list[dict] = []

    def fake_run(cmd, input_data=None, capture_output=False, text=False, check=False, timeout=None, **kwargs):
        del capture_output, text, check, timeout
        if "input" in kwargs:
            input_data = kwargs["input"]
        if input_data:
            manifests.append(json.loads(input_data))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return manifests


def test_run_builds_pod_manifest(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()

    assert manifests, "expected kubectl invocation"
    manifest = manifests[0]
    assert manifest["metadata"]["namespace"] == "iris"
    assert manifest["metadata"]["labels"]["iris.managed"] == "true"
    assert manifest["spec"]["containers"][0]["name"] == "task"
    assert manifest["spec"]["containers"][0]["imagePullPolicy"] == "Always"


def test_host_network_enabled_by_default(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config(network_mode="host"))
    handle.run()

    spec = manifests[0]["spec"]
    assert spec["hostNetwork"] is True
    assert spec["dnsPolicy"] == "ClusterFirstWithHostNet"


def test_host_network_disabled_when_not_host(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config(network_mode="bridge"))
    handle.run()

    spec = manifests[0]["spec"]
    assert "hostNetwork" not in spec
    assert "dnsPolicy" not in spec


def test_gpu_resources_and_tolerations(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config(gpu_count=8, gpu_name="H100"))
    handle.run()

    manifest = manifests[0]
    container = manifest["spec"]["containers"][0]
    limits = container["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "8"

    tolerations = manifest["spec"]["tolerations"]
    assert any(t["key"] == "nvidia.com/gpu" and t["operator"] == "Exists" for t in tolerations)


def test_no_gpu_resources_when_zero_gpus(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config(gpu_count=0))
    handle.run()

    spec = manifests[0]["spec"]
    # No GPU resources requested
    container = spec["containers"][0]
    if "resources" in container:
        assert "nvidia.com/gpu" not in container["resources"].get("limits", {})
    # No tolerations
    assert "tolerations" not in spec


def test_service_account_name(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(
        namespace="iris",
        service_account_name="iris-controller",
    )
    handle = runtime.create_container(_make_config())
    handle.run()

    assert manifests[0]["spec"]["serviceAccountName"] == "iris-controller"


def test_no_service_account_when_unset(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()

    assert "serviceAccountName" not in manifests[0]["spec"]


def test_s3_secret_env_vars(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(
        namespace="iris",
        s3_secret_name="iris-s3-credentials",
    )
    handle = runtime.create_container(_make_config())
    handle.run()

    env = manifests[0]["spec"]["containers"][0]["env"]
    secret_envs = [e for e in env if "valueFrom" in e and "secretKeyRef" in e.get("valueFrom", {})]
    secret_keys = {e["name"] for e in secret_envs}
    assert "AWS_ACCESS_KEY_ID" in secret_keys
    assert "AWS_SECRET_ACCESS_KEY" in secret_keys
    for e in secret_envs:
        assert e["valueFrom"]["secretKeyRef"]["name"] == "iris-s3-credentials"


def test_no_s3_env_when_unset(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()

    env = manifests[0]["spec"]["containers"][0]["env"]
    secret_envs = [e for e in env if "valueFrom" in e and "secretKeyRef" in e.get("valueFrom", {})]
    assert len(secret_envs) == 0


def test_advertise_host_uses_downward_api(monkeypatch):
    """Task pods must use the K8s downward API for IRIS_ADVERTISE_HOST
    instead of inheriting the worker's static value, since the task pod
    may be scheduled on a different node."""
    manifests = _capture_manifest(monkeypatch)

    config = _make_config()
    config.env["IRIS_ADVERTISE_HOST"] = "10.0.0.1"
    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(config)
    handle.run()

    env = manifests[0]["spec"]["containers"][0]["env"]
    advertise = [e for e in env if e["name"] == "IRIS_ADVERTISE_HOST"]
    assert len(advertise) == 1
    assert advertise[0] == {"name": "IRIS_ADVERTISE_HOST", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}}
    # The static value from config.env must NOT appear
    assert not any(e.get("value") == "10.0.0.1" for e in env)


def test_status_retries_transient_pod_not_found(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()
    assert manifests  # Ensure pod is initialized before status checks.

    responses = [
        None,
        None,
        {"status": {"phase": "Running"}},
    ]

    def fake_get_json(resource: str, name: str):
        assert resource == "pod"
        assert name
        return responses.pop(0)

    monkeypatch.setattr(handle.kubectl, "get_json", fake_get_json)

    first = handle.status()
    second = handle.status()
    third = handle.status()

    assert first.phase == ContainerPhase.PENDING
    assert first.error is None
    assert second.phase == ContainerPhase.PENDING
    assert second.error is None
    assert third.phase == ContainerPhase.RUNNING


def test_status_reflects_pod_phase_progression(monkeypatch):
    """status() tracks pod phase: Pending → PENDING, Running → RUNNING."""
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()
    assert manifests

    responses = [
        {"status": {"phase": "Pending"}},
        {"status": {"phase": "Pending"}},
        {"status": {"phase": "Running"}},
    ]

    def fake_get_json(resource: str, name: str):
        del name
        assert resource == "pod"
        return responses.pop(0)

    monkeypatch.setattr(handle.kubectl, "get_json", fake_get_json)

    first = handle.status()
    assert first.phase == ContainerPhase.PENDING

    second = handle.status()
    assert second.phase == ContainerPhase.PENDING

    third = handle.status()
    assert third.phase == ContainerPhase.RUNNING


def test_status_returns_structured_error_after_persistent_pod_not_found(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()
    assert manifests

    def fake_get_json(resource: str, name: str):
        assert resource == "pod"
        assert name
        return None

    monkeypatch.setattr(handle.kubectl, "get_json", fake_get_json)

    first = handle.status()
    second = handle.status()
    third = handle.status()

    assert first.phase == ContainerPhase.PENDING
    assert second.phase == ContainerPhase.PENDING
    assert third.phase == ContainerPhase.STOPPED
    assert third.error_kind == ContainerErrorKind.INFRA_NOT_FOUND
    assert third.error is not None
    assert "Task pod not found after retry window" in third.error
