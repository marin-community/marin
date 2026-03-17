# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KubernetesRuntime.

These tests mock kubectl subprocess calls and validate manifest generation,
including GPU resource requests, hostNetwork, tolerations, and S3 secret refs.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile


from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.kubernetes import KubernetesRuntime
from iris.cluster.runtime.types import ContainerConfig, ContainerErrorKind, ContainerPhase, MountKind, MountSpec
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


def _pod_manifest(manifests: list[dict]) -> dict:
    return next(m for m in manifests if m.get("kind") == "Pod")


def _configmap_manifest(manifests: list[dict]) -> dict:
    return next(m for m in manifests if m.get("kind") == "ConfigMap")


def test_run_builds_pod_manifest(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()

    assert manifests, "expected kubectl invocation"
    manifest = _pod_manifest(manifests)
    assert manifest["metadata"]["namespace"] == "iris"
    assert manifest["metadata"]["labels"]["iris.managed"] == "true"
    assert manifest["spec"]["containers"][0]["name"] == "task"
    assert manifest["spec"]["containers"][0]["imagePullPolicy"] == "Always"


def test_workdir_files_are_staged_via_configmap(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    config = _make_config()
    config.entrypoint.workdir_files["inline.txt"] = b"hello"
    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(config)
    handle.run()

    pod = _pod_manifest(manifests)
    task_script = pod["spec"]["containers"][0]["command"][2]
    assert "IRIS_WORKDIR_FILE_" not in task_script
    assert "base64 -d" not in task_script

    cfg = _configmap_manifest(manifests)
    assert cfg["kind"] == "ConfigMap"
    assert "binaryData" in cfg
    assert len(cfg["binaryData"]) == 1
    assert pod["spec"]["volumes"][-1]["name"] == "workdir-files"


def test_bundle_fetch_init_container_is_present(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    config = _make_config()
    config.env["IRIS_BUNDLE_ID"] = "a" * 64
    config.env["IRIS_CONTROLLER_URL"] = "http://controller.internal:10000"
    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(config)
    handle.run()

    spec = _pod_manifest(manifests)["spec"]
    assert "initContainers" in spec
    init = spec["initContainers"][0]
    assert init["name"] == "stage-workdir"
    assert init["command"][:2] == ["bash", "-lc"]
    script = init["command"][2]
    assert "urlopen" in script
    assert "IRIS_BUNDLE_ID" in script
    assert "IRIS_CONTROLLER_URL" in script
    env = init["env"]
    assert {"name": "IRIS_BUNDLE_ID", "value": "a" * 64} in env
    assert {"name": "IRIS_CONTROLLER_URL", "value": "http://controller.internal:10000"} in env


def test_host_network_enabled_by_default(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config(network_mode="host"))
    handle.run()

    spec = _pod_manifest(manifests)["spec"]
    assert spec["hostNetwork"] is True
    assert spec["dnsPolicy"] == "ClusterFirstWithHostNet"


def test_host_network_disabled_when_not_host(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config(network_mode="bridge"))
    handle.run()

    spec = _pod_manifest(manifests)["spec"]
    assert "hostNetwork" not in spec
    assert "dnsPolicy" not in spec


def test_gpu_resources_and_tolerations(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config(gpu_count=8, gpu_name="H100"))
    handle.run()

    manifest = _pod_manifest(manifests)
    container = manifest["spec"]["containers"][0]
    limits = container["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "8"

    tolerations = manifest["spec"]["tolerations"]
    assert any(t["key"] == "nvidia.com/gpu" and t["operator"] == "Exists" for t in tolerations)
    assert any(t["key"] == "qos.coreweave.cloud/interruptable" and t["effect"] == "NoExecute" for t in tolerations)


def test_no_gpu_resources_when_zero_gpus(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config(gpu_count=0))
    handle.run()

    spec = _pod_manifest(manifests)["spec"]
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

    assert _pod_manifest(manifests)["spec"]["serviceAccountName"] == "iris-controller"


def test_no_service_account_when_unset(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()

    assert "serviceAccountName" not in _pod_manifest(manifests)["spec"]


def test_s3_secret_env_vars(monkeypatch):
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(
        namespace="iris",
        s3_secret_name="iris-s3-credentials",
    )
    handle = runtime.create_container(_make_config())
    handle.run()

    env = _pod_manifest(manifests)["spec"]["containers"][0]["env"]
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

    env = _pod_manifest(manifests)["spec"]["containers"][0]["env"]
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

    env = _pod_manifest(manifests)["spec"]["containers"][0]["env"]
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


def test_status_surfaces_init_container_failure(monkeypatch):
    """Init-container failures (e.g. bundle fetch 404) should be surfaced
    with the init container's name and error details."""
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    config = _make_config()
    config.env["IRIS_BUNDLE_ID"] = "a" * 64
    config.env["IRIS_CONTROLLER_URL"] = "http://controller.internal:10000"
    handle = runtime.create_container(config)
    handle.run()
    assert manifests

    failed_pod = {
        "status": {
            "phase": "Failed",
            "initContainerStatuses": [
                {
                    "name": "stage-workdir",
                    "state": {
                        "terminated": {
                            "exitCode": 1,
                            "reason": "Error",
                            "message": "Bundle hash mismatch: expected aaa, got bbb",
                        }
                    },
                }
            ],
            "containerStatuses": [],
        }
    }

    def fake_get_json(resource: str, name: str):
        return failed_pod

    monkeypatch.setattr(handle.kubectl, "get_json", fake_get_json)

    status = handle.status()
    assert status.phase == ContainerPhase.STOPPED
    assert status.exit_code == 1
    assert "stage-workdir" in status.error
    assert "Bundle hash mismatch" in status.error


def test_status_ignores_successful_init_containers(monkeypatch):
    """Successfully completed init containers should not affect status."""
    manifests = _capture_manifest(monkeypatch)

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(_make_config())
    handle.run()
    assert manifests

    running_pod = {
        "status": {
            "phase": "Running",
            "initContainerStatuses": [
                {
                    "name": "stage-workdir",
                    "state": {
                        "terminated": {
                            "exitCode": 0,
                            "reason": "Completed",
                        }
                    },
                }
            ],
            "containerStatuses": [
                {
                    "name": "task",
                    "state": {"running": {}},
                }
            ],
        }
    }

    def fake_get_json(resource: str, name: str):
        return running_pod

    monkeypatch.setattr(handle.kubectl, "get_json", fake_get_json)

    status = handle.status()
    assert status.phase == ContainerPhase.RUNNING


def test_disk_bytes_sets_emptydir_sizelimit(monkeypatch):
    """When a WORKDIR MountSpec has size_bytes, the emptyDir volume should have a sizeLimit."""
    manifests = _capture_manifest(monkeypatch)

    device = cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(count=0))
    resources = cluster_pb2.ResourceSpecProto(device=device, disk_bytes=10 * 1024**3)
    config = ContainerConfig(
        image="ghcr.io/example/task:latest",
        entrypoint=_make_entrypoint(["python", "-c", "print('ok')"]),
        env={"FOO": "bar"},
        workdir="/app",
        task_id="job/task/0",
        network_mode="host",
        resources=resources,
        mounts=[MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=10 * 1024**3)],
    )

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(config)
    handle.run()

    pod = _pod_manifest(manifests)
    workdir_vol = next(v for v in pod["spec"]["volumes"] if "emptyDir" in v)
    assert "sizeLimit" in workdir_vol["emptyDir"]
    assert workdir_vol["emptyDir"]["sizeLimit"] == str(10 * 1024**3)


def test_cache_mounts_use_cache_dir_host_path(monkeypatch):
    """CACHE mounts must map hostPath under worker cache_dir, not use the container path directly."""
    manifests = _capture_manifest(monkeypatch)

    config = _make_config()
    config.mounts = [
        MountSpec(container_path="/uv/cache", kind=MountKind.CACHE),
        MountSpec(container_path="/root/.cargo/registry", kind=MountKind.CACHE),
    ]
    from pathlib import Path

    runtime = KubernetesRuntime(namespace="iris", cache_dir=Path("/mnt/nvme/iris"))
    handle = runtime.create_container(config)
    handle.run()

    pod = _pod_manifest(manifests)
    volumes = pod["spec"]["volumes"]
    cache_volumes = [v for v in volumes if "hostPath" in v]
    assert len(cache_volumes) == 2
    host_paths = sorted(v["hostPath"]["path"] for v in cache_volumes)
    assert host_paths == ["/mnt/nvme/iris/root-.cargo-registry", "/mnt/nvme/iris/uv-cache"]


def test_cache_mounts_fallback_without_cache_dir(monkeypatch):
    """Without cache_dir, CACHE mounts fall back to container_path as hostPath."""
    manifests = _capture_manifest(monkeypatch)

    config = _make_config()
    config.mounts = [MountSpec(container_path="/uv/cache", kind=MountKind.CACHE)]
    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(config)
    handle.run()

    pod = _pod_manifest(manifests)
    cache_vol = next(v for v in pod["spec"]["volumes"] if "hostPath" in v)
    assert cache_vol["hostPath"]["path"] == "/uv/cache"


def test_no_disk_bytes_emptydir_has_no_sizelimit(monkeypatch):
    """When a WORKDIR MountSpec has size_bytes=0, the emptyDir volume should have no sizeLimit."""
    manifests = _capture_manifest(monkeypatch)

    config = _make_config()
    config.mounts = [MountSpec(container_path="/app", kind=MountKind.WORKDIR, size_bytes=0)]
    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(config)
    handle.run()

    pod = _pod_manifest(manifests)
    workdir_vol = next(v for v in pod["spec"]["volumes"] if "emptyDir" in v)
    assert workdir_vol["emptyDir"] == {}


def _make_zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return output.getvalue()


def test_k8s_init_script_fetches_and_extracts_bundle(tmp_path):
    """Exercise the kubernetes_bundle_fetch.py init script against a real
    BundleStore acting as controller, verifying end-to-end bundle staging."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading

    bundle_zip = _make_zip({"main.py": b"print('hello')", "lib/util.py": b"x = 1\n"})
    bundle_id = hashlib.sha256(bundle_zip).hexdigest()

    # Set up a controller-side BundleStore and write the bundle
    controller_store = BundleStore(storage_dir=str(tmp_path / "controller_bundles"))
    stored_id = controller_store.write_zip(bundle_zip)
    assert stored_id == bundle_id

    # Start a minimal HTTP server that serves bundles from the store
    class BundleHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Expect /bundles/<bundle_id>.zip
            parts = self.path.strip("/").split("/")
            if len(parts) == 2 and parts[0] == "bundles" and parts[1].endswith(".zip"):
                bid = parts[1][:-4]
                try:
                    data = controller_store.get_zip(bid)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.end_headers()
                    self.wfile.write(data)
                except FileNotFoundError:
                    self.send_response(404)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            pass  # Suppress log output in tests

    server = HTTPServer(("127.0.0.1", 0), BundleHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        # Run the init script in a subprocess, simulating the K8s init container
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "src",
            "iris",
            "cluster",
            "runtime",
            "kubernetes_bundle_fetch.py",
        )
        env = {
            "IRIS_BUNDLE_ID": bundle_id,
            "IRIS_CONTROLLER_URL": f"http://127.0.0.1:{port}",
            "IRIS_WORKDIR": str(workdir),
        }
        result = subprocess.run(
            [sys.executable, "-I", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Init script failed: {result.stderr}"
        assert (workdir / "main.py").exists()
        assert (workdir / "main.py").read_text() == "print('hello')"
        assert (workdir / "lib/util.py").exists()
    finally:
        server.shutdown()
