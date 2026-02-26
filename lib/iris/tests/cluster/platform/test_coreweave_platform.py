# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for CoreweavePlatform.

Tests exercise the public interface of CoreweavePlatform using mocked kubectl
subprocess calls. We test lifecycle state transitions, failure paths, listing,
termination, controller discovery, and the shared NodePool model.
"""

from __future__ import annotations

import io
import json
import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    Labels,
    PlatformError,
)
from iris.cluster.platform.coreweave import CoreweavePlatform
from iris.cluster.platform.factory import create_platform
from iris.rpc import config_pb2


def _make_platform(
    region: str = "LGA1",
    namespace: str = "iris",
    kubeconfig_path: str = "",
    label_prefix: str = "iris",
) -> CoreweavePlatform:
    config = config_pb2.CoreweavePlatformConfig(
        region=region,
        namespace=namespace,
        kubeconfig_path=kubeconfig_path,
    )
    return CoreweavePlatform(config=config, label_prefix=label_prefix, poll_interval=0.05)


def _make_slice_config(
    name_prefix: str = "h100-8x",
    instance_type: str = "gd-8xh100ib-i128",
    gpu_count: int = 8,
) -> config_pb2.SliceConfig:
    return config_pb2.SliceConfig(
        name_prefix=name_prefix,
        num_vms=1,
        gpu_count=gpu_count,
        accelerator_variant="H100",
        coreweave=config_pb2.CoreweaveSliceConfig(
            region="LGA1",
            instance_type=instance_type,
            gpu_class="H100",
        ),
    )


def _make_worker_config() -> config_pb2.WorkerConfig:
    return config_pb2.WorkerConfig(
        docker_image="ghcr.io/marin-community/iris-worker:latest",
        port=10001,
        cache_dir="/var/cache/iris",
        runtime="kubernetes",
    )


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeKubectl:
    """Stateful fake that intercepts subprocess.run calls for kubectl.

    Tracks NodePool and Pod state. Supports injecting failures and simulating
    lifecycle transitions.
    """

    def __init__(self):
        self._nodepools: dict[str, dict] = {}
        self._pods: dict[str, dict] = {}
        self._deployments: dict[str, dict] = {}
        self._services: dict[str, dict] = {}
        self._configmaps: dict[str, dict] = {}
        self._secrets: dict[str, dict] = {}
        self._failures: dict[str, str] = {}
        self._pod_logs: dict[str, str] = {}
        self._events: list[dict] = []

    def set_failure(self, operation: str, error: str) -> None:
        self._failures[operation] = error

    def make_nodepool_ready(self, name: str | None = None) -> None:
        """Mark a NodePool as having readyNodes >= 1."""
        if name:
            if name in self._nodepools:
                self._nodepools[name]["status"]["readyNodes"] = 1
        else:
            for np in self._nodepools.values():
                np["status"]["readyNodes"] = 1

    def make_pod_ready(self, name: str | None = None) -> None:
        """Mark a Pod as ready with an IP."""
        if name:
            if name in self._pods:
                self._pods[name]["status"]["phase"] = "Running"
                self._pods[name]["status"]["podIP"] = "10.0.0.42"
                self._pods[name]["status"]["conditions"] = [
                    {"type": "Ready", "status": "True"},
                ]
        else:
            for pod in self._pods.values():
                pod["status"]["phase"] = "Running"
                pod["status"]["podIP"] = "10.0.0.42"
                pod["status"]["conditions"] = [
                    {"type": "Ready", "status": "True"},
                ]

    def make_deployment_available(self, name: str | None = None) -> None:
        """Mark a Deployment as having availableReplicas >= 1."""
        if name:
            if name in self._deployments:
                self._deployments[name]["status"]["availableReplicas"] = 1
        else:
            for dep in self._deployments.values():
                dep["status"]["availableReplicas"] = 1

    def set_nodepool_failed(self, name: str, message: str) -> None:
        if name in self._nodepools:
            self._nodepools[name]["status"]["conditions"] = [
                {"type": "Failed", "status": "True", "message": message},
            ]

    def __call__(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        if not cmd or cmd[0] != "kubectl":
            raise ValueError(f"FakeKubectl: unexpected command: {cmd}")

        args = list(cmd[1:])
        namespace = "iris"
        i = 0
        clean_args: list[str] = []
        while i < len(args):
            if args[i] == "--kubeconfig":
                i += 2
                continue
            if args[i] == "-n":
                namespace = args[i + 1]
                i += 2
                continue
            clean_args.append(args[i])
            i += 1

        if "apply" in clean_args and "-f" in clean_args and "-" in clean_args:
            return self._handle_apply(kwargs.get("input", ""), namespace)
        if "get" in clean_args and "nodepool" in clean_args and "nodepools" not in clean_args:
            return self._handle_get_nodepool(clean_args)
        if "get" in clean_args and "nodepools" in clean_args:
            return self._handle_get_nodepools(clean_args)
        if "get" in clean_args and "pod" in clean_args and "pods" not in clean_args:
            return self._handle_get_pod(clean_args, namespace)
        if "get" in clean_args and "pods" in clean_args:
            return self._handle_get_pods(clean_args, namespace)
        if "get" in clean_args and "deployment" in clean_args:
            return self._handle_get_deployment(clean_args)
        if "get" in clean_args and "service" in clean_args:
            return self._handle_get_service(clean_args)
        if "get" in clean_args and "configmap" in clean_args:
            return self._handle_get_configmap(clean_args)
        if "get" in clean_args and "events" in clean_args:
            return self._handle_get_events(clean_args)
        if "get" in clean_args and "secret" in clean_args:
            return self._handle_get_secret(clean_args)
        if "delete" in clean_args and "pod" in clean_args:
            return self._handle_delete_pod(clean_args, namespace)
        if "delete" in clean_args and "nodepool" in clean_args:
            return self._handle_delete_nodepool(clean_args)
        if "delete" in clean_args and "deployment" in clean_args:
            return self._handle_delete_generic(clean_args, "deployment", self._deployments)
        if "delete" in clean_args and "service" in clean_args:
            return self._handle_delete_generic(clean_args, "service", self._services)
        if "delete" in clean_args and "configmap" in clean_args:
            return self._handle_delete_generic(clean_args, "configmap", self._configmaps)
        if "delete" in clean_args and "secret" in clean_args:
            return self._handle_delete_generic(clean_args, "secret", self._secrets)
        if "set" in clean_args and "image" in clean_args:
            return self._handle_set_image(clean_args)
        if "rollout" in clean_args and "restart" in clean_args:
            return _completed()
        if "rollout" in clean_args and "status" in clean_args:
            return self._handle_rollout_status(clean_args)
        if "logs" in clean_args:
            return self._handle_logs(clean_args, namespace)
        if "exec" in clean_args:
            return self._handle_exec(clean_args, namespace)

        return _completed(returncode=1, stderr=f"FakeKubectl: unrecognized: {clean_args}")

    def _handle_apply(self, input_data: str, namespace: str) -> subprocess.CompletedProcess:
        if "apply" in self._failures:
            return _completed(returncode=1, stderr=self._failures.pop("apply"))

        data = json.loads(input_data)
        kind = data.get("kind", "")
        name = data.get("metadata", {}).get("name", "")

        if kind == "NodePool":
            self._nodepools[name] = {
                "metadata": {
                    "name": name,
                    "labels": data.get("metadata", {}).get("labels", {}),
                    "creationTimestamp": "2026-02-18T00:00:00Z",
                },
                "spec": data.get("spec", {}),
                "status": {"readyNodes": 0, "conditions": []},
            }
            return _completed()
        elif kind == "Pod":
            self._pods[name] = {
                "metadata": {
                    "name": name,
                    "namespace": namespace,
                    "labels": data.get("metadata", {}).get("labels", {}),
                },
                "spec": data.get("spec", {}),
                "status": {
                    "phase": "Pending",
                    "podIP": "",
                    "conditions": [],
                },
            }
            return _completed()
        elif kind == "Deployment":
            existing_status = self._deployments.get(name, {}).get("status", {"availableReplicas": 0})
            self._deployments[name] = {
                "metadata": data.get("metadata", {}),
                "spec": data.get("spec", {}),
                "status": existing_status,
            }
            return _completed()
        elif kind == "Service":
            self._services[name] = {
                "metadata": data.get("metadata", {}),
                "spec": data.get("spec", {}),
            }
            return _completed()
        elif kind == "ConfigMap":
            self._configmaps[name] = {
                "metadata": data.get("metadata", {}),
                "data": data.get("data", {}),
            }
            return _completed()
        elif kind == "Secret":
            self._secrets[name] = {
                "metadata": data.get("metadata", {}),
                "data": data.get("data", {}),
            }
            return _completed()

        return _completed()

    def _handle_get_nodepool(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = args.index("nodepool") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing nodepool name")
        name = args[idx]

        if name not in self._nodepools:
            return _completed(returncode=1, stderr=f"nodepool {name} not found")

        return _completed(stdout=json.dumps(self._nodepools[name]))

    def _handle_get_nodepools(self, args: list[str]) -> subprocess.CompletedProcess:
        items = list(self._nodepools.values())
        if "-l" in args:
            selector_idx = args.index("-l") + 1
            if selector_idx < len(args):
                selector = args[selector_idx]
                items = self._filter_by_selector(items, selector)

        return _completed(stdout=json.dumps({"items": items}))

    def _handle_get_pod(self, args: list[str], namespace: str) -> subprocess.CompletedProcess:
        idx = args.index("pod") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing pod name")
        name = args[idx]

        if name not in self._pods:
            return _completed(returncode=1, stderr=f"pod {name} not found")

        pod = self._pods[name]

        # Check for jsonpath output: -o jsonpath={...}
        jsonpath = ""
        for a in args:
            if a.startswith("jsonpath="):
                jsonpath = a[len("jsonpath=") :]
                break

        if jsonpath:
            if ".status.conditions" in jsonpath:
                conditions = pod.get("status", {}).get("conditions", [])
                for c in conditions:
                    if c.get("type") == "Ready":
                        return _completed(stdout=c.get("status", "False"))
                return _completed(stdout="False")
            if ".status.podIP" in jsonpath:
                return _completed(stdout=pod.get("status", {}).get("podIP", ""))
            if ".status.phase" in jsonpath:
                return _completed(stdout=pod.get("status", {}).get("phase", ""))

        return _completed(stdout=json.dumps(pod))

    def _handle_get_pods(self, args: list[str], namespace: str) -> subprocess.CompletedProcess:
        items = list(self._pods.values())
        if "-l" in args:
            selector_idx = args.index("-l") + 1
            if selector_idx < len(args):
                selector = args[selector_idx]
                items = self._filter_by_selector(items, selector)

        return _completed(stdout=json.dumps({"items": items}))

    def _handle_delete_pod(self, args: list[str], namespace: str) -> subprocess.CompletedProcess:
        idx = args.index("pod") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing pod name")
        name = args[idx]
        self._pods.pop(name, None)
        return _completed()

    def _handle_delete_nodepool(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = args.index("nodepool") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing nodepool name")
        name = args[idx]
        self._nodepools.pop(name, None)
        return _completed()

    def _handle_get_deployment(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = args.index("deployment") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing deployment name")
        name = args[idx]
        if name not in self._deployments:
            return _completed(returncode=1, stderr=f'deployments.apps "{name}" not found')
        return _completed(stdout=json.dumps(self._deployments[name]))

    def _handle_get_service(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = args.index("service") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing service name")
        name = args[idx]
        if name not in self._services:
            return _completed(returncode=1, stderr=f'services "{name}" not found')
        return _completed(stdout=json.dumps(self._services[name]))

    def _handle_get_configmap(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = args.index("configmap") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing configmap name")
        name = args[idx]
        if name not in self._configmaps:
            return _completed(returncode=1, stderr=f'configmaps "{name}" not found')
        return _completed(stdout=json.dumps(self._configmaps[name]))

    def _handle_get_secret(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = args.index("secret") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing secret name")
        name = args[idx]
        if name not in self._secrets:
            return _completed(returncode=1, stderr=f'secrets "{name}" not found')
        return _completed(stdout=json.dumps(self._secrets[name]))

    def _handle_delete_generic(
        self, args: list[str], resource_type: str, store: dict[str, dict]
    ) -> subprocess.CompletedProcess:
        idx = args.index(resource_type) + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr=f"missing {resource_type} name")
        name = args[idx]
        store.pop(name, None)
        return _completed()

    def _handle_set_image(self, args: list[str]) -> subprocess.CompletedProcess:
        if "set_image" in self._failures:
            return _completed(returncode=1, stderr=self._failures.pop("set_image"))
        # Parse: set image deployment/NAME CONTAINER=IMAGE
        resource_name = None
        container_image = None
        for arg in args:
            if arg in ("set", "image"):
                continue
            if "=" in arg:
                container_image = arg
            elif "/" in arg:
                resource_name = arg

        if resource_name and container_image:
            parts = resource_name.split("/")
            if len(parts) == 2 and parts[0] == "deployment":
                dep_name = parts[1]
                container_name, new_image = container_image.split("=", 1)
                if dep_name in self._deployments:
                    containers = (
                        self._deployments[dep_name]
                        .get("spec", {})
                        .get("template", {})
                        .get("spec", {})
                        .get("containers", [])
                    )
                    for c in containers:
                        if c.get("name") == container_name:
                            c["image"] = new_image
        return _completed()

    def _handle_rollout_status(self, args: list[str]) -> subprocess.CompletedProcess:
        if "rollout_status" in self._failures:
            return _completed(returncode=1, stderr=self._failures.pop("rollout_status"))
        return _completed()

    def _handle_get_events(self, args: list[str]) -> subprocess.CompletedProcess:
        items = list(self._events)
        if "--field-selector" in args:
            idx = args.index("--field-selector") + 1
            if idx < len(args):
                selector = args[idx]
                # Simple field selector filtering for involvedObject.name=X
                for part in selector.split(","):
                    if "=" in part:
                        key, val = part.split("=", 1)
                        if key == "involvedObject.name":
                            items = [e for e in items if e.get("involvedObject", {}).get("name") == val]
                        elif key == "type":
                            items = [e for e in items if e.get("type") == val]
        return _completed(stdout=json.dumps({"items": items}))

    def _handle_logs(self, args: list[str], namespace: str) -> subprocess.CompletedProcess:
        idx = args.index("logs") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing pod name")
        pod_name = args[idx]
        if pod_name in self._pod_logs:
            return _completed(stdout=self._pod_logs[pod_name])
        return _completed(returncode=1, stderr=f"pod {pod_name} not found")

    def _handle_exec(self, args: list[str], namespace: str) -> subprocess.CompletedProcess:
        return _completed(stdout="exec output")

    def _filter_by_selector(self, items: list[dict], selector: str) -> list[dict]:
        required = {}
        for pair in selector.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                required[k] = v

        return [
            item
            for item in items
            if all(item.get("metadata", {}).get("labels", {}).get(k) == v for k, v in required.items())
        ]


@pytest.fixture
def fake_kubectl() -> FakeKubectl:
    fake = FakeKubectl()
    with patch("iris.cluster.k8s.kubectl.subprocess.run", fake):
        yield fake


@pytest.fixture(autouse=True)
def _s3_env_vars(monkeypatch):
    """Set R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY so _ensure_s3_credentials_secret() succeeds."""
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-key-secret")


# ============================================================================
# Tests
# ============================================================================


def test_create_slice_returns_handle_in_creating_state(fake_kubectl: FakeKubectl):
    """create_slice() returns a handle immediately in CREATING state."""
    platform = _make_platform()
    config = _make_slice_config()
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)
    status = handle.describe()

    # Handle starts CREATING; background thread transitions to BOOTSTRAPPING
    assert status.state in (CloudSliceState.CREATING, CloudSliceState.BOOTSTRAPPING)
    assert handle.scale_group == "h100-8x"
    assert handle.slice_id.startswith("iris-h100-8x-")
    platform.shutdown()


def test_slice_lifecycle_happy_path(fake_kubectl: FakeKubectl):
    """Simulate the full lifecycle: CREATING -> BOOTSTRAPPING -> READY."""
    platform = _make_platform()
    config = _make_slice_config()
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)

    # Wait for Pod creation (monitor thread creates it immediately)
    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)
    fake_kubectl.make_pod_ready()

    # Wait for handle to transition to READY
    _wait_for_condition(lambda: handle.describe().state == CloudSliceState.READY, timeout=5)

    status = handle.describe()
    assert status.state == CloudSliceState.READY
    assert status.worker_count == 1
    assert len(status.workers) == 1
    assert status.workers[0].internal_address == "10.0.0.42"
    platform.shutdown()


def test_slice_transitions_through_bootstrapping(fake_kubectl: FakeKubectl):
    """Verify the BOOTSTRAPPING state is observed between CREATING and READY."""
    platform = _make_platform()
    config = _make_slice_config()
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)

    # Wait for Pod creation which happens in BOOTSTRAPPING phase
    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)

    # Before marking pod ready, should be BOOTSTRAPPING
    _wait_for_condition(
        lambda: handle.describe().state == CloudSliceState.BOOTSTRAPPING,
        timeout=5,
    )

    fake_kubectl.make_pod_ready()
    _wait_for_condition(lambda: handle.describe().state == CloudSliceState.READY, timeout=5)
    platform.shutdown()


def test_slice_failure_on_pod_apply_error(fake_kubectl: FakeKubectl):
    """When Pod apply fails, handle transitions to FAILED."""
    platform = _make_platform()
    config = _make_slice_config()
    bootstrap = _make_worker_config()

    # Fail on the first apply (which will be the Pod apply)
    fake_kubectl.set_failure("apply", "node provisioning failed")

    handle = platform.create_slice(config, bootstrap)

    _wait_for_condition(lambda: handle.describe().state == CloudSliceState.FAILED, timeout=5)
    assert handle.describe().state == CloudSliceState.FAILED
    platform.shutdown()


def test_list_slices_by_label(fake_kubectl: FakeKubectl):
    """list_slices returns handles for worker Pods matching labels."""
    platform = _make_platform()

    _seed_worker_pod(fake_kubectl, "iris-h100-8x-1000", "h100-8x")
    _seed_worker_pod(fake_kubectl, "iris-a100-4x-2000", "a100-4x")

    handles = platform.list_slices(
        zones=["LGA1"],
        labels={Labels("iris").iris_scale_group: "h100-8x"},
    )
    assert len(handles) == 1
    assert handles[0].slice_id == "iris-h100-8x-1000"
    assert handles[0].scale_group == "h100-8x"
    platform.shutdown()


def test_list_all_slices(fake_kubectl: FakeKubectl):
    """list_all_slices returns all iris-managed worker Pods."""
    platform = _make_platform()

    _seed_worker_pod(fake_kubectl, "iris-h100-8x-1000", "h100-8x")
    _seed_worker_pod(fake_kubectl, "iris-a100-4x-2000", "a100-4x")

    # Add a non-worker Pod that should be ignored
    iris_labels = Labels("iris")
    fake_kubectl._pods["some-other-pod"] = {
        "metadata": {
            "name": "some-other-pod",
            "namespace": "iris",
            "labels": {iris_labels.iris_managed: "true"},
        },
        "spec": {},
        "status": {"phase": "Running", "podIP": "10.0.0.99", "conditions": []},
    }

    handles = platform.list_all_slices()
    assert len(handles) == 2
    platform.shutdown()


def test_discover_controller_dns():
    """discover_controller returns correct K8s Service DNS name."""
    config = config_pb2.CoreweavePlatformConfig(region="LGA1", namespace="iris")
    platform = CoreweavePlatform(config=config, label_prefix="iris")

    controller_config = config_pb2.ControllerVmConfig(
        coreweave=config_pb2.CoreweaveControllerConfig(
            port=10000,
            service_name="iris-controller-svc",
        )
    )

    address = platform.discover_controller(controller_config)
    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    platform.shutdown()


def test_discover_controller_defaults():
    """discover_controller uses default port and service name when not set."""
    config = config_pb2.CoreweavePlatformConfig(region="LGA1", namespace="my-ns")
    platform = CoreweavePlatform(config=config, label_prefix="iris")

    controller_config = config_pb2.ControllerVmConfig(coreweave=config_pb2.CoreweaveControllerConfig())

    address = platform.discover_controller(controller_config)
    assert address == "iris-controller-svc.my-ns.svc.cluster.local:10000"
    platform.shutdown()


def test_create_slice_requires_worker_config(fake_kubectl: FakeKubectl):
    """create_slice() raises ValueError when worker_config is not provided."""
    platform = _make_platform()
    config = _make_slice_config()

    with pytest.raises(ValueError, match="worker_config is required"):
        platform.create_slice(config)
    platform.shutdown()


def test_terminate_deletes_pod_only(fake_kubectl: FakeKubectl):
    """terminate() deletes the Pod but does not touch NodePools."""
    platform = _make_platform()
    config = _make_slice_config()
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)
    slice_id = handle.slice_id

    pod_name = f"iris-worker-{slice_id}"
    # Wait for Pod to be created by monitor thread
    _wait_for_condition(lambda: pod_name in fake_kubectl._pods, timeout=5)

    handle.terminate()

    assert pod_name not in fake_kubectl._pods
    # No NodePool should have been deleted (there is no per-slice NodePool)
    platform.shutdown()


def test_create_vm_raises_not_implemented():
    """CoreWeave does not support standalone VMs."""
    platform = _make_platform()
    with pytest.raises(NotImplementedError, match="standalone VMs"):
        platform.create_vm(config_pb2.VmConfig())
    platform.shutdown()


def test_worker_handle_status(fake_kubectl: FakeKubectl):
    """CoreweaveWorkerHandle.status() maps Pod phase to CloudWorkerState."""
    platform = _make_platform()
    config = _make_slice_config()
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)
    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)
    fake_kubectl.make_pod_ready()
    _wait_for_condition(lambda: handle.describe().state == CloudSliceState.READY, timeout=5)

    worker = handle.describe().workers[0]
    status = worker.status()
    assert status.state == CloudWorkerState.RUNNING
    platform.shutdown()


def test_factory_creates_coreweave_platform():
    """create_platform returns CoreweavePlatform for coreweave config."""
    platform_config = config_pb2.PlatformConfig(
        label_prefix="iris",
        coreweave=config_pb2.CoreweavePlatformConfig(
            region="LGA1",
            namespace="iris",
        ),
    )

    platform = create_platform(platform_config)
    assert isinstance(platform, CoreweavePlatform)
    platform.shutdown()


def test_run_command_on_line_streams_output(fake_kubectl: FakeKubectl):
    """run_command() with on_line streams output line-by-line via Popen."""
    platform = _make_platform()
    config = _make_slice_config()
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)
    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)
    fake_kubectl.make_pod_ready()
    _wait_for_condition(lambda: handle.describe().state == CloudSliceState.READY, timeout=5)

    worker = handle.describe().workers[0]
    lines_received: list[str] = []

    fake_output = "line1\nline2\nline3\n"

    with patch("iris.cluster.k8s.kubectl.subprocess.Popen") as mock_popen:
        mock_proc = mock_popen.return_value
        mock_proc.stdout = io.StringIO(fake_output)
        mock_proc.stderr = io.StringIO("")
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

        result = worker.run_command("echo hello", on_line=lines_received.append)

    assert lines_received == ["line1", "line2", "line3"]
    assert result.stdout == "line1\nline2\nline3"
    assert result.returncode == 0
    platform.shutdown()


def test_node_selector_uses_label_override_not_name_prefix(fake_kubectl: FakeKubectl):
    """When prepare_slice_config sets a different scale-group label than name_prefix,
    the nodeSelector must use the label value so Pods schedule on the correct NodePool."""
    platform = _make_platform()
    # Simulate what prepare_slice_config does: name_prefix = "iris-h100_8x"
    # but labels override iris-scale-group = "h100_8x" (the bare scale group name).
    iris_labels = Labels("iris")
    config = _make_slice_config(name_prefix="iris-h100_8x")
    config.labels[iris_labels.iris_scale_group] = "h100_8x"
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)
    assert handle.scale_group == "h100_8x"

    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)

    pod_name = f"iris-worker-{handle.slice_id}"
    node_selector = fake_kubectl._pods[pod_name]["spec"]["nodeSelector"]
    assert node_selector == {iris_labels.iris_scale_group: "h100_8x"}
    platform.shutdown()


def test_create_slice_rejects_multi_node(fake_kubectl: FakeKubectl):
    """create_slice() raises ValueError when num_vms > 1."""
    platform = _make_platform()
    config = config_pb2.SliceConfig(
        name_prefix="h100-8x",
        num_vms=2,
        gpu_count=8,
        accelerator_variant="H100",
        coreweave=config_pb2.CoreweaveSliceConfig(
            region="LGA1",
            instance_type="gd-8xh100ib-i128",
        ),
    )

    with pytest.raises(ValueError, match="does not support multi-node"):
        platform.create_slice(config)
    platform.shutdown()


def test_list_slices_only_returns_worker_pods(fake_kubectl: FakeKubectl):
    """list_slices() only returns iris-worker-* Pods, ignoring other managed Pods."""
    platform = _make_platform()

    _seed_worker_pod(fake_kubectl, "iris-h100-8x-1000", "h100-8x")

    # Non-worker Pod (e.g. controller or auxiliary)
    iris_labels = Labels("iris")
    fake_kubectl._pods["iris-controller-abc"] = {
        "metadata": {
            "name": "iris-controller-abc",
            "namespace": "iris",
            "labels": {iris_labels.iris_managed: "true"},
        },
        "spec": {},
        "status": {"phase": "Running", "podIP": "10.0.0.99", "conditions": []},
    }

    handles = platform.list_slices(zones=["LGA1"])
    assert len(handles) == 1
    assert handles[0].slice_id == "iris-h100-8x-1000"
    platform.shutdown()


# ============================================================================
# ensure_nodepools tests
# ============================================================================


def test_ensure_nodepools_creates_all_pools(fake_kubectl: FakeKubectl):
    """ensure_nodepools creates one NodePool per scale group plus the controller pool."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()
    sg = cluster_config.scale_groups["h100-8x"]
    sg.min_slices = 2
    sg.max_slices = 5
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix="h100-8x",
            num_vms=1,
            coreweave=config_pb2.CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128"),
        )
    )

    platform.ensure_nodepools(cluster_config)

    assert "iris-h100-8x" in fake_kubectl._nodepools
    assert "iris-cpu-erapids" in fake_kubectl._nodepools

    # min_slices/max_slices from config flow through to NodePool minNodes/maxNodes
    h100_pool = fake_kubectl._nodepools["iris-h100-8x"]
    assert h100_pool["spec"]["minNodes"] == 2
    assert h100_pool["spec"]["maxNodes"] == 5
    platform.shutdown()


def test_ensure_nodepools_errors_without_min_max_slices(fake_kubectl: FakeKubectl):
    """ensure_nodepools raises when a scale group is missing min_slices or max_slices."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()
    sg = cluster_config.scale_groups["h100-8x"]
    # Don't set min_slices or max_slices
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix="h100-8x",
            num_vms=1,
            coreweave=config_pb2.CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128"),
        )
    )

    with pytest.raises(PlatformError, match="must set min_slices"):
        platform.ensure_nodepools(cluster_config)
    platform.shutdown()


def test_ensure_nodepools_reconciles_existing(fake_kubectl: FakeKubectl):
    """ensure_nodepools reconciles spec on existing NodePools (e.g. minNodes/maxNodes changes)."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    # Pre-create the cpu-erapids pool with stale spec and 3 current nodes
    fake_kubectl._nodepools["iris-cpu-erapids"] = {
        "metadata": {"name": "iris-cpu-erapids", "labels": {}},
        "spec": {"instanceType": "cd-gp-i64-erapids", "minNodes": 5, "maxNodes": 5, "targetNodes": 3},
        "status": {"readyNodes": 0, "currentNodes": 3, "conditions": []},
    }

    platform.ensure_nodepools(cluster_config)

    # Pool should be reconciled with config values
    pool = fake_kubectl._nodepools["iris-cpu-erapids"]
    assert pool["spec"]["minNodes"] == 0
    assert pool["spec"]["maxNodes"] == 10
    # targetNodes clamped to min(currentNodes=3, 1) = 1
    assert pool["spec"]["targetNodes"] == 1
    platform.shutdown()


def test_ensure_nodepools_new_pool_starts_at_zero(fake_kubectl: FakeKubectl):
    """New NodePools start with targetNodes=0."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    platform.ensure_nodepools(cluster_config)

    pool = fake_kubectl._nodepools["iris-cpu-erapids"]
    assert pool["spec"]["targetNodes"] == 0
    platform.shutdown()


def test_ensure_nodepools_deletes_stale_pools(fake_kubectl: FakeKubectl):
    """ensure_nodepools deletes managed NodePools not in the current config."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    # Pre-create a stale NodePool from a previous config (e.g. renamed scale group)
    iris_labels = Labels("iris")
    fake_kubectl._nodepools["iris-old-gpu-pool"] = {
        "metadata": {
            "name": "iris-old-gpu-pool",
            "labels": {iris_labels.iris_managed: "true", iris_labels.iris_scale_group: "old-gpu-pool"},
        },
        "spec": {"instanceType": "gd-8xh100ib-i128"},
        "status": {"readyNodes": 0, "conditions": []},
    }

    platform.ensure_nodepools(cluster_config)

    # Stale pool should be deleted
    assert "iris-old-gpu-pool" not in fake_kubectl._nodepools
    # Expected pool should exist
    assert "iris-cpu-erapids" in fake_kubectl._nodepools
    platform.shutdown()


def test_ensure_nodepools_preserves_unmanaged_pools(fake_kubectl: FakeKubectl):
    """ensure_nodepools does not delete NodePools without the managed label."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    # Pre-create an unmanaged NodePool (no iris-managed label)
    fake_kubectl._nodepools["user-custom-pool"] = {
        "metadata": {
            "name": "user-custom-pool",
            "labels": {},
        },
        "spec": {"instanceType": "some-type"},
        "status": {"readyNodes": 0, "conditions": []},
    }

    platform.ensure_nodepools(cluster_config)

    # Unmanaged pool should be untouched
    assert "user-custom-pool" in fake_kubectl._nodepools
    platform.shutdown()


# ============================================================================
# start_controller / stop_controller / tunnel tests
# ============================================================================


def _make_cluster_config(
    port: int = 10000,
    service_name: str = "iris-controller-svc",
    image: str = "ghcr.io/marin-community/iris-controller:latest",
    bundle_prefix: str = "gs://test-bucket/bundles",
    controller_scale_group: str = "cpu-erapids",
) -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig(
        platform=config_pb2.PlatformConfig(
            label_prefix="iris",
            coreweave=config_pb2.CoreweavePlatformConfig(
                region="LGA1",
                namespace="iris",
            ),
        ),
        controller=config_pb2.ControllerVmConfig(
            image=image,
            coreweave=config_pb2.CoreweaveControllerConfig(
                port=port,
                service_name=service_name,
                scale_group=controller_scale_group,
            ),
        ),
        storage=config_pb2.StorageConfig(
            bundle_prefix=bundle_prefix,
        ),
    )
    # Add the controller's scale group so start_controller can validate it
    sg = config.scale_groups[controller_scale_group]
    sg.min_slices = 0
    sg.max_slices = 10
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix=controller_scale_group,
            num_vms=1,
            coreweave=config_pb2.CoreweaveSliceConfig(instance_type="cd-gp-i64-erapids"),
        )
    )
    return config


def test_start_controller_creates_all_resources(fake_kubectl: FakeKubectl):
    """start_controller creates ConfigMap, shared NodePools, Deployment, and Service."""
    platform = _make_platform()
    cluster_config = _make_cluster_config(bundle_prefix="s3://test-bucket/bundles")

    # Make Deployment available once it exists
    def auto_ready_deployment():
        _wait_for_condition(lambda: "iris-controller" in fake_kubectl._deployments, timeout=10)
        fake_kubectl.make_deployment_available("iris-controller")

    t = threading.Thread(target=auto_ready_deployment, daemon=True)
    t.start()

    address = platform.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    assert "iris-s3-credentials" in fake_kubectl._secrets
    assert "iris-cluster-config" in fake_kubectl._configmaps
    assert "iris-cpu-erapids" in fake_kubectl._nodepools
    assert "iris-controller" in fake_kubectl._deployments
    assert "iris-controller-svc" in fake_kubectl._services

    # Verify Deployment nodeSelector targets the configured scale group
    iris_labels = Labels("iris")
    deploy_spec = fake_kubectl._deployments["iris-controller"]["spec"]
    node_selector = deploy_spec["template"]["spec"]["nodeSelector"]
    assert node_selector == {iris_labels.iris_scale_group: "cpu-erapids"}

    # Verify controller uses S3 env vars (no GCS credentials)
    container = deploy_spec["template"]["spec"]["containers"][0]
    env_names = [e["name"] for e in container["env"]]
    assert "AWS_ACCESS_KEY_ID" in env_names
    assert "AWS_SECRET_ACCESS_KEY" in env_names
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env_names

    t.join(timeout=5)
    platform.shutdown()


def test_start_controller_reconciles_when_already_available(fake_kubectl: FakeKubectl):
    """start_controller reconciles all resources even if Deployment is already available."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    # Pre-create an available Deployment
    fake_kubectl._deployments["iris-controller"] = {
        "metadata": {"name": "iris-controller"},
        "spec": {},
        "status": {"availableReplicas": 1},
    }

    address = platform.start_controller(cluster_config)
    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"

    # ConfigMap, NodePools, and Service should all be reconciled
    assert "iris-cluster-config" in fake_kubectl._configmaps
    assert "iris-cpu-erapids" in fake_kubectl._nodepools
    assert "iris-controller-svc" in fake_kubectl._services
    platform.shutdown()


def test_stop_controller_deletes_resources_except_nodepool(fake_kubectl: FakeKubectl):
    """stop_controller deletes Deployment, Service, ConfigMap, and S3 secret but not NodePool."""
    platform = _make_platform()
    cluster_config = _make_cluster_config(bundle_prefix="s3://test-bucket/bundles")

    # Pre-populate resources
    fake_kubectl._deployments["iris-controller"] = {
        "metadata": {},
        "spec": {},
        "status": {},
    }
    fake_kubectl._services["iris-controller-svc"] = {"metadata": {}, "spec": {}}
    fake_kubectl._nodepools["iris-cpu-erapids"] = {
        "metadata": {"name": "iris-cpu-erapids", "labels": {}},
        "spec": {},
        "status": {"readyNodes": 1, "conditions": []},
    }
    fake_kubectl._configmaps["iris-cluster-config"] = {"metadata": {}, "data": {}}
    fake_kubectl._secrets["iris-s3-credentials"] = {"metadata": {}, "data": {}}

    platform.stop_controller(cluster_config)

    assert "iris-controller" not in fake_kubectl._deployments
    assert "iris-controller-svc" not in fake_kubectl._services
    assert "iris-cluster-config" not in fake_kubectl._configmaps
    assert "iris-s3-credentials" not in fake_kubectl._secrets
    # NodePool should remain (shared, scales to zero)
    assert "iris-cpu-erapids" in fake_kubectl._nodepools
    platform.shutdown()


def test_stop_controller_idempotent(fake_kubectl: FakeKubectl):
    """stop_controller succeeds even if resources don't exist."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    # No resources exist -- should not raise
    platform.stop_controller(cluster_config)
    platform.shutdown()


def test_tunnel_parses_address():
    """tunnel() extracts service name and port from the address string."""
    config = config_pb2.CoreweavePlatformConfig(region="LGA1", namespace="iris")
    platform = CoreweavePlatform(config=config, label_prefix="iris")

    controller_config = config_pb2.ControllerVmConfig(
        coreweave=config_pb2.CoreweaveControllerConfig(port=9999, service_name="my-svc"),
    )
    address = platform.discover_controller(controller_config)
    assert address == "my-svc.iris.svc.cluster.local:9999"

    # Verify parsing logic directly
    host, port_str = address.rsplit(":", 1)
    service_name = host.split(".")[0]
    remote_port = int(port_str)
    assert service_name == "my-svc"
    assert remote_port == 9999
    platform.shutdown()


def test_start_controller_deployment_command_references_config_json(
    fake_kubectl: FakeKubectl,
):
    """The controller Deployment command uses config.json (not config.yaml)."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    def auto_ready():
        _wait_for_condition(lambda: "iris-controller" in fake_kubectl._deployments, timeout=10)
        fake_kubectl.make_deployment_available("iris-controller")

    t = threading.Thread(target=auto_ready, daemon=True)
    t.start()

    platform.start_controller(cluster_config)

    container = fake_kubectl._deployments["iris-controller"]["spec"]["template"]["spec"]["containers"][0]
    # Must reference config.json, not config.yaml
    config_args = [arg for arg in container["command"] if "config" in arg and arg.startswith("--config")]
    assert len(config_args) == 1
    assert config_args[0] == "--config=/etc/iris/config.json"

    t.join(timeout=5)
    platform.shutdown()


def test_configmap_strips_kubeconfig_path(fake_kubectl: FakeKubectl):
    """ConfigMap must not contain kubeconfig_path so pods use in-cluster auth."""
    platform = _make_platform(kubeconfig_path="/home/user/.kube/coreweave-iris")
    cluster_config = _make_cluster_config()
    cluster_config.platform.coreweave.kubeconfig_path = "/home/user/.kube/coreweave-iris"

    def auto_ready():
        _wait_for_condition(lambda: "iris-controller" in fake_kubectl._deployments, timeout=10)
        fake_kubectl.make_deployment_available("iris-controller")

    t = threading.Thread(target=auto_ready, daemon=True)
    t.start()

    platform.start_controller(cluster_config)

    cm_data = json.loads(fake_kubectl._configmaps["iris-cluster-config"]["data"]["config.json"])
    cw_config = cm_data.get("platform", {}).get("coreweave", {})
    assert "kubeconfig_path" not in cw_config

    t.join(timeout=5)
    platform.shutdown()


def test_controller_deployment_includes_endpoint_url(fake_kubectl: FakeKubectl):
    """When object_storage_endpoint is set, the controller Deployment includes AWS_ENDPOINT_URL."""
    cw_config = config_pb2.CoreweavePlatformConfig(
        region="LGA1",
        namespace="iris",
        object_storage_endpoint="https://object.lga1.coreweave.com",
    )
    platform = CoreweavePlatform(config=cw_config, label_prefix="iris", poll_interval=0.05)

    cluster_config = _make_cluster_config(bundle_prefix="s3://test-bucket/bundles")
    cluster_config.platform.coreweave.object_storage_endpoint = "https://object.lga1.coreweave.com"

    def auto_ready():
        _wait_for_condition(lambda: "iris-controller" in fake_kubectl._deployments, timeout=10)
        fake_kubectl.make_deployment_available("iris-controller")

    t = threading.Thread(target=auto_ready, daemon=True)
    t.start()

    platform.start_controller(cluster_config)

    container = fake_kubectl._deployments["iris-controller"]["spec"]["template"]["spec"]["containers"][0]
    env_by_name = {e["name"]: e for e in container["env"]}
    assert "AWS_ENDPOINT_URL" in env_by_name
    assert env_by_name["AWS_ENDPOINT_URL"]["value"] == "https://object.lga1.coreweave.com"

    t.join(timeout=5)
    platform.shutdown()


def test_worker_pod_has_gpu_resource_limits_with_docker_runtime(fake_kubectl: FakeKubectl):
    """Worker Pods request nvidia.com/gpu resource limits when runtime is docker."""
    platform = _make_platform()
    config = _make_slice_config(gpu_count=8)
    wc = config_pb2.WorkerConfig(
        docker_image="ghcr.io/marin-community/iris-worker:latest",
        port=10001,
        cache_dir="/var/cache/iris",
        runtime="docker",
    )

    handle = platform.create_slice(config, wc)
    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)

    pod_name = f"iris-worker-{handle.slice_id}"
    container = fake_kubectl._pods[pod_name]["spec"]["containers"][0]
    limits = container["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "8"
    platform.shutdown()


def test_worker_pod_no_gpu_limits_with_kubernetes_runtime(fake_kubectl: FakeKubectl):
    """With kubernetes runtime, worker Pods must not request GPU resources (task Pods claim them)."""
    platform = _make_platform()
    config = _make_slice_config(gpu_count=8)
    bootstrap = _make_worker_config()  # runtime="kubernetes"

    handle = platform.create_slice(config, bootstrap)
    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)

    pod_name = f"iris-worker-{handle.slice_id}"
    container = fake_kubectl._pods[pod_name]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in container.get("resources", {}).get("limits", {})
    platform.shutdown()


def test_worker_pod_no_gpu_limits_when_zero(fake_kubectl: FakeKubectl):
    """Worker Pods omit nvidia.com/gpu when gpu_count is 0 (CPU-only)."""
    platform = _make_platform()
    config = _make_slice_config(gpu_count=0)
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)
    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)

    pod_name = f"iris-worker-{handle.slice_id}"
    container = fake_kubectl._pods[pod_name]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in container.get("resources", {}).get("limits", {})
    platform.shutdown()


def test_worker_pod_has_s3_env_vars(fake_kubectl: FakeKubectl):
    """Worker Pods include S3 credential env vars (secretKeyRef) when S3 storage is enabled."""
    platform = _make_platform()
    platform._s3_enabled = True
    config = _make_slice_config()
    bootstrap = _make_worker_config()

    handle = platform.create_slice(config, bootstrap)

    _wait_for_condition(lambda: len(fake_kubectl._pods) > 0, timeout=5)

    pod_name = f"iris-worker-{handle.slice_id}"
    container = fake_kubectl._pods[pod_name]["spec"]["containers"][0]
    env_by_name = {e["name"]: e for e in container["env"]}

    assert "AWS_ACCESS_KEY_ID" in env_by_name
    key_id_ref = env_by_name["AWS_ACCESS_KEY_ID"]["valueFrom"]["secretKeyRef"]
    assert key_id_ref["name"] == "iris-s3-credentials"
    assert key_id_ref["key"] == "AWS_ACCESS_KEY_ID"

    assert "AWS_SECRET_ACCESS_KEY" in env_by_name
    secret_ref = env_by_name["AWS_SECRET_ACCESS_KEY"]["valueFrom"]["secretKeyRef"]
    assert secret_ref["name"] == "iris-s3-credentials"
    assert secret_ref["key"] == "AWS_SECRET_ACCESS_KEY"

    platform.shutdown()


def test_start_controller_errors_without_scale_group(fake_kubectl: FakeKubectl):
    """start_controller raises when scale_group is not set."""
    platform = _make_platform()
    config = config_pb2.IrisClusterConfig(
        platform=config_pb2.PlatformConfig(
            label_prefix="iris",
            coreweave=config_pb2.CoreweavePlatformConfig(region="LGA1", namespace="iris"),
        ),
        controller=config_pb2.ControllerVmConfig(
            image="ghcr.io/marin-community/iris-controller:latest",
            coreweave=config_pb2.CoreweaveControllerConfig(port=10000),
        ),
    )
    with pytest.raises(PlatformError, match="must set scale_group"):
        platform.start_controller(config)
    platform.shutdown()


def test_start_controller_errors_with_invalid_scale_group(fake_kubectl: FakeKubectl):
    """start_controller raises when scale_group references a nonexistent group."""
    platform = _make_platform()
    config = config_pb2.IrisClusterConfig(
        platform=config_pb2.PlatformConfig(
            label_prefix="iris",
            coreweave=config_pb2.CoreweavePlatformConfig(region="LGA1", namespace="iris"),
        ),
        controller=config_pb2.ControllerVmConfig(
            image="ghcr.io/marin-community/iris-controller:latest",
            coreweave=config_pb2.CoreweaveControllerConfig(port=10000, scale_group="nonexistent"),
        ),
    )
    with pytest.raises(PlatformError, match="not found in scale_groups"):
        platform.start_controller(config)
    platform.shutdown()


def test_start_controller_errors_without_s3_credentials(fake_kubectl: FakeKubectl, monkeypatch):
    """start_controller raises when S3 storage is configured but R2 credentials are not set."""
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    platform = _make_platform()
    cluster_config = _make_cluster_config(bundle_prefix="s3://my-bucket/bundles")

    with pytest.raises(PlatformError, match="R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY"):
        platform.start_controller(cluster_config)
    platform.shutdown()


def test_start_controller_detects_crash_loop_backoff(fake_kubectl: FakeKubectl):
    """start_controller fails fast when the controller Pod enters CrashLoopBackOff."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    crash_logs = (
        "ValueError: scale_groups.cpu-erapids.resources has unknown keys: memory_bytes\n"
        "Error: Failed to load cluster config\n"
    )

    def simulate_crash_loop():
        _wait_for_condition(lambda: "iris-controller" in fake_kubectl._deployments, timeout=10)
        # Simulate the controller Pod in CrashLoopBackOff: Deployment exists
        # but the Pod keeps crashing.
        fake_kubectl._pods["iris-controller-abc123"] = {
            "metadata": {
                "name": "iris-controller-abc123",
                "namespace": "iris",
                "labels": {"app": "iris-controller"},
            },
            "spec": {},
            "status": {
                "phase": "Running",
                "podIP": "10.0.0.1",
                "conditions": [{"type": "ContainersReady", "status": "False"}],
                "containerStatuses": [
                    {
                        "name": "iris-controller",
                        "restartCount": 3,
                        "state": {
                            "waiting": {
                                "reason": "CrashLoopBackOff",
                                "message": "back-off 40s restarting failed container",
                            }
                        },
                    }
                ],
            },
        }
        fake_kubectl._pod_logs["iris-controller-abc123"] = crash_logs

    t = threading.Thread(target=simulate_crash_loop, daemon=True)
    t.start()

    with pytest.raises(PlatformError, match="CrashLoopBackOff"):
        platform.start_controller(cluster_config)

    t.join(timeout=5)
    platform.shutdown()


def test_start_controller_detects_image_pull_failure(fake_kubectl: FakeKubectl):
    """start_controller fails fast on ImagePullBackOff."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    def simulate_image_pull_failure():
        _wait_for_condition(lambda: "iris-controller" in fake_kubectl._deployments, timeout=10)
        fake_kubectl._pods["iris-controller-abc123"] = {
            "metadata": {
                "name": "iris-controller-abc123",
                "namespace": "iris",
                "labels": {"app": "iris-controller"},
            },
            "spec": {},
            "status": {
                "phase": "Pending",
                "podIP": "",
                "conditions": [],
                "containerStatuses": [
                    {
                        "name": "iris-controller",
                        "restartCount": 0,
                        "state": {
                            "waiting": {
                                "reason": "ImagePullBackOff",
                                "message": "failed to pull image: 403 Forbidden",
                            }
                        },
                    }
                ],
            },
        }

    t = threading.Thread(target=simulate_image_pull_failure, daemon=True)
    t.start()

    with pytest.raises(PlatformError, match="ImagePullBackOff"):
        platform.start_controller(cluster_config)

    t.join(timeout=5)
    platform.shutdown()


def test_start_controller_crash_loop_includes_logs(fake_kubectl: FakeKubectl):
    """When CrashLoopBackOff is detected, the error includes container logs."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    crash_logs = "ValueError: bad config key\nTraceback ...\n"

    def simulate_crash_loop():
        _wait_for_condition(lambda: "iris-controller" in fake_kubectl._deployments, timeout=10)
        fake_kubectl._pods["iris-controller-xyz"] = {
            "metadata": {
                "name": "iris-controller-xyz",
                "namespace": "iris",
                "labels": {"app": "iris-controller"},
            },
            "spec": {},
            "status": {
                "phase": "Running",
                "podIP": "10.0.0.1",
                "conditions": [],
                "containerStatuses": [
                    {
                        "name": "iris-controller",
                        "restartCount": 5,
                        "state": {
                            "waiting": {
                                "reason": "CrashLoopBackOff",
                                "message": "back-off restarting",
                            }
                        },
                    }
                ],
            },
        }
        fake_kubectl._pod_logs["iris-controller-xyz"] = crash_logs

    t = threading.Thread(target=simulate_crash_loop, daemon=True)
    t.start()

    with pytest.raises(PlatformError, match="bad config key"):
        platform.start_controller(cluster_config)

    t.join(timeout=5)
    platform.shutdown()


def test_start_controller_skips_s3_for_gs_storage(fake_kubectl: FakeKubectl, monkeypatch):
    """start_controller succeeds without S3 credentials when using gs:// storage."""
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    platform = _make_platform()
    cluster_config = _make_cluster_config(bundle_prefix="gs://test-bucket/bundles")

    def auto_ready():
        _wait_for_condition(lambda: "iris-controller" in fake_kubectl._deployments, timeout=10)
        fake_kubectl.make_deployment_available("iris-controller")

    t = threading.Thread(target=auto_ready, daemon=True)
    t.start()

    address = platform.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    # No S3 secret should be created
    assert "iris-s3-credentials" not in fake_kubectl._secrets
    # Controller Deployment should have no S3 env vars
    container = fake_kubectl._deployments["iris-controller"]["spec"]["template"]["spec"]["containers"][0]
    env_names = [e["name"] for e in container["env"]]
    assert "AWS_ACCESS_KEY_ID" not in env_names
    assert "AWS_SECRET_ACCESS_KEY" not in env_names

    t.join(timeout=5)
    platform.shutdown()


# ============================================================================
# stop_all tests
# ============================================================================


def test_stop_all_deletes_pods_not_nodepools(fake_kubectl: FakeKubectl):
    """stop_all deletes managed Pods and controller resources, leaving NodePools."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    _seed_worker_pod(fake_kubectl, "iris-h100-8x-1000", "h100-8x")

    # Pre-populate controller resources
    fake_kubectl._deployments["iris-controller"] = {"metadata": {}, "spec": {}, "status": {}}
    fake_kubectl._services["iris-controller-svc"] = {"metadata": {}, "spec": {}}
    fake_kubectl._configmaps["iris-cluster-config"] = {"metadata": {}, "data": {}}
    fake_kubectl._nodepools["iris-h100-8x"] = {
        "metadata": {"name": "iris-h100-8x", "labels": {}},
        "spec": {},
        "status": {"readyNodes": 0, "conditions": []},
    }

    targets = platform.stop_all(cluster_config)

    # Worker Pod should be deleted
    assert "iris-worker-iris-h100-8x-1000" not in fake_kubectl._pods
    # NodePool should remain
    assert "iris-h100-8x" in fake_kubectl._nodepools
    # Controller resources deleted
    assert "iris-controller" not in fake_kubectl._deployments
    assert "controller" in targets
    platform.shutdown()


def test_stop_all_dry_run(fake_kubectl: FakeKubectl):
    """stop_all with dry_run=True returns target names without deleting."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    _seed_worker_pod(fake_kubectl, "iris-h100-8x-1000", "h100-8x")

    targets = platform.stop_all(cluster_config, dry_run=True)

    assert "pod:iris-worker-iris-h100-8x-1000" in targets
    assert "controller" in targets
    # Pod should still exist
    assert "iris-worker-iris-h100-8x-1000" in fake_kubectl._pods
    platform.shutdown()


# ============================================================================
# reload tests
# ============================================================================


def _make_cluster_config_with_workers(
    scale_group_name: str = "h100-8x",
    worker_image: str = "ghcr.io/marin-community/iris-worker:v2",
    controller_image: str = "ghcr.io/marin-community/iris-controller:v2",
    port: int = 10000,
    service_name: str = "iris-controller-svc",
    bundle_prefix: str = "gs://test-bucket/bundles",
) -> config_pb2.IrisClusterConfig:
    """Build a cluster config with scale_groups and worker config for reload tests."""
    config = _make_cluster_config(
        port=port,
        service_name=service_name,
        image=controller_image,
        bundle_prefix=bundle_prefix,
    )
    config.defaults.worker.CopyFrom(
        config_pb2.WorkerConfig(
            docker_image=worker_image,
            port=10001,
            cache_dir="/var/cache/iris",
            runtime="kubernetes",
        )
    )
    sg = config.scale_groups[scale_group_name]
    sg.min_slices = 0
    sg.max_slices = 10
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix=scale_group_name,
            num_vms=1,
            accelerator_variant="H100",
            coreweave=config_pb2.CoreweaveSliceConfig(
                region="LGA1",
                instance_type="gd-8xh100ib-i128",
                gpu_class="H100",
            ),
        )
    )
    sg.resources.CopyFrom(config_pb2.ScaleGroupResources(gpu_count=8))
    return config


def _seed_worker_pod(
    fake_kubectl: FakeKubectl,
    slice_id: str,
    scale_group: str,
    label_prefix: str = "iris",
    ready: bool = True,
) -> None:
    """Inject a managed worker Pod into FakeKubectl state."""
    pod_labels = Labels(label_prefix)
    pod_name = f"iris-worker-{slice_id}"
    fake_kubectl._pods[pod_name] = {
        "metadata": {
            "name": pod_name,
            "namespace": "iris",
            "labels": {
                pod_labels.iris_managed: "true",
                pod_labels.iris_scale_group: scale_group,
                pod_labels.iris_slice_id: slice_id,
            },
            "creationTimestamp": "2026-02-18T00:00:00Z",
        },
        "spec": {},
        "status": {
            "phase": "Running" if ready else "Pending",
            "podIP": "10.0.0.42" if ready else "",
            "conditions": [{"type": "Ready", "status": "True"}] if ready else [],
        },
    }


def _seed_controller_deployment(fake_kubectl: FakeKubectl, image: str = "ghcr.io/marin-community/iris-controller:v1"):
    """Pre-create an available controller Deployment in FakeKubectl state."""
    fake_kubectl._deployments["iris-controller"] = {
        "metadata": {"name": "iris-controller", "labels": {"app": "iris-controller"}},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "iris-controller"}},
            "template": {
                "metadata": {"labels": {"app": "iris-controller"}},
                "spec": {
                    "containers": [
                        {
                            "name": "iris-controller",
                            "image": image,
                        }
                    ],
                },
            },
        },
        "status": {"availableReplicas": 1},
    }


def test_reload_updates_configmap_and_controller(fake_kubectl: FakeKubectl):
    """reload() updates ConfigMap with new config and performs rolling update on controller Deployment."""
    platform = _make_platform()
    _seed_controller_deployment(fake_kubectl)

    cluster_config = _make_cluster_config_with_workers(controller_image="ghcr.io/marin-community/iris-controller:v2")

    address = platform.reload(cluster_config)

    # ConfigMap should be updated
    assert "iris-cluster-config" in fake_kubectl._configmaps
    cm_data = fake_kubectl._configmaps["iris-cluster-config"]["data"]
    config_dict = json.loads(cm_data["config.json"])
    assert config_dict["controller"]["image"] == "ghcr.io/marin-community/iris-controller:v2"

    # Controller Deployment image should be updated via set_image
    container = fake_kubectl._deployments["iris-controller"]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "ghcr.io/marin-community/iris-controller:v2"

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    platform.shutdown()


def test_reload_replaces_worker_pods(fake_kubectl: FakeKubectl):
    """reload() deletes old worker Pod and recreates it with updated image."""
    platform = _make_platform()
    _seed_controller_deployment(fake_kubectl)
    _seed_worker_pod(fake_kubectl, slice_id="iris-h100-8x-1000", scale_group="h100-8x")

    pod_name = "iris-worker-iris-h100-8x-1000"
    assert pod_name in fake_kubectl._pods

    cluster_config = _make_cluster_config_with_workers(
        scale_group_name="h100-8x",
        worker_image="ghcr.io/marin-community/iris-worker:v2",
    )

    # The reload will: delete old Pod, create new Pod via apply, then wait for readiness.
    deleted = threading.Event()
    original_delete = fake_kubectl._handle_delete_pod

    def tracking_delete(args, namespace):
        result = original_delete(args, namespace)
        deleted.set()
        return result

    fake_kubectl._handle_delete_pod = tracking_delete

    def auto_ready_new_pod():
        deleted.wait(timeout=10)
        # Wait for the Pod to be re-created via apply
        _wait_for_condition(lambda: pod_name in fake_kubectl._pods, timeout=10)
        fake_kubectl.make_pod_ready(pod_name)

    t = threading.Thread(target=auto_ready_new_pod, daemon=True)
    t.start()

    platform.reload(cluster_config)

    # The recreated worker Pod should have the updated image
    assert pod_name in fake_kubectl._pods
    container = fake_kubectl._pods[pod_name]["spec"]["containers"][0]
    assert container["image"] == "ghcr.io/marin-community/iris-worker:v2"

    t.join(timeout=5)
    platform.shutdown()


def test_reload_skips_missing_scale_group(fake_kubectl: FakeKubectl):
    """reload() warns and skips Pod recreation when scale group is not in config."""
    platform = _make_platform()
    _seed_controller_deployment(fake_kubectl)
    # Seed a worker with scale_group "unknown-group" which won't be in config
    _seed_worker_pod(fake_kubectl, slice_id="iris-unknown-group-2000", scale_group="unknown-group")

    old_pod_name = "iris-worker-iris-unknown-group-2000"
    assert old_pod_name in fake_kubectl._pods

    # Config only has "h100-8x" scale group, not "unknown-group"
    cluster_config = _make_cluster_config_with_workers(scale_group_name="h100-8x")

    platform.reload(cluster_config)

    # Old Pod should be deleted (always happens)
    assert old_pod_name not in fake_kubectl._pods
    # No new Pod should be created (scale group not in config)
    worker_pods = [k for k in fake_kubectl._pods if k.startswith("iris-worker-")]
    assert len(worker_pods) == 0
    platform.shutdown()


# ============================================================================
# Helpers
# ============================================================================


def _wait_for_condition(condition, timeout: float = 5.0, poll: float = 0.05):
    """Poll until condition() is truthy, or raise on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(poll)
    raise TimeoutError(f"Condition not met within {timeout}s")
