# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for CoreweavePlatform.

Tests exercise the public interface of CoreweavePlatform using mocked kubectl
subprocess calls. We test controller lifecycle, discovery, tunnel, RBAC, and
configuration. Worker/slice management is handled by KubernetesProvider (not
CoreweavePlatform).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from iris.cluster.platform.base import (
    Labels,
    PlatformError,
)
from iris.cluster.platform.coreweave import (
    _CONTROLLER_CPU_REQUEST,
    _CONTROLLER_MEMORY_REQUEST,
    CoreweavePlatform,
)
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


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FakeKubectl:
    """Stateful fake that intercepts subprocess.run calls for kubectl.

    Tracks Deployment, Service, ConfigMap, Secret, and RBAC state.
    Supports injecting failures and simulating lifecycle transitions.
    """

    def __init__(self):
        self._deployments: dict[str, dict] = {}
        self._services: dict[str, dict] = {}
        self._configmaps: dict[str, dict] = {}
        self._secrets: dict[str, dict] = {}
        self._cluster_roles: dict[str, dict] = {}
        self._cluster_role_bindings: dict[str, dict] = {}
        self._nodepools: dict[str, dict] = {}
        self._nodes: dict[str, dict] = {}
        self._pods: dict[str, dict] = {}
        self._failures: dict[str, str] = {}
        self._pod_logs: dict[str, str] = {}
        self._events: list[dict] = []

    def set_failure(self, operation: str, error: str) -> None:
        self._failures[operation] = error

    def make_deployment_available(self, name: str | None = None) -> None:
        """Mark a Deployment as having availableReplicas >= 1."""
        if name:
            if name in self._deployments:
                self._deployments[name]["status"]["availableReplicas"] = 1
        else:
            for dep in self._deployments.values():
                dep["status"]["availableReplicas"] = 1

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
        if "get" in clean_args and "nodepool" in clean_args:
            return self._handle_get_nodepool(clean_args)
        if "get" in clean_args and "nodepools" in clean_args:
            return self._handle_list_nodepools(clean_args)
        if "get" in clean_args and "node" in clean_args and "nodes" not in clean_args:
            return self._handle_get_node(clean_args)
        if "get" in clean_args and "pods" in clean_args:
            return self._handle_get_pods(clean_args, namespace)
        if "delete" in clean_args and "nodepool" in clean_args:
            return self._handle_delete_generic(clean_args, "nodepool", self._nodepools)
        if "delete" in clean_args and "deployment" in clean_args:
            return self._handle_delete_generic(clean_args, "deployment", self._deployments)
        if "delete" in clean_args and "service" in clean_args:
            return self._handle_delete_generic(clean_args, "service", self._services)
        if "delete" in clean_args and "configmap" in clean_args:
            return self._handle_delete_generic(clean_args, "configmap", self._configmaps)
        if "delete" in clean_args and "secret" in clean_args:
            return self._handle_delete_generic(clean_args, "secret", self._secrets)
        if "delete" in clean_args and "clusterrolebinding" in clean_args:
            return self._handle_delete_generic(clean_args, "clusterrolebinding", self._cluster_role_bindings)
        if "delete" in clean_args and "clusterrole" in clean_args:
            return self._handle_delete_generic(clean_args, "clusterrole", self._cluster_roles)
        if "rollout" in clean_args and "restart" in clean_args:
            return _completed()
        if "rollout" in clean_args and "status" in clean_args:
            return self._handle_rollout_status(clean_args)
        if "logs" in clean_args:
            return self._handle_logs(clean_args, namespace)

        return _completed(returncode=1, stderr=f"FakeKubectl: unrecognized: {clean_args}")

    def _handle_apply(self, input_data: str, namespace: str) -> subprocess.CompletedProcess:
        if "apply" in self._failures:
            return _completed(returncode=1, stderr=self._failures.pop("apply"))

        data = json.loads(input_data)
        kind = data.get("kind", "")
        name = data.get("metadata", {}).get("name", "")

        if kind == "Deployment":
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
        elif kind == "ClusterRole":
            self._cluster_roles[name] = {
                "metadata": data.get("metadata", {}),
                "rules": data.get("rules", []),
            }
            return _completed()
        elif kind == "ClusterRoleBinding":
            self._cluster_role_bindings[name] = {
                "metadata": data.get("metadata", {}),
                "subjects": data.get("subjects", []),
                "roleRef": data.get("roleRef", {}),
            }
            return _completed()
        elif kind == "NodePool":
            existing_status = self._nodepools.get(name, {}).get("status", {})
            self._nodepools[name] = {
                "metadata": data.get("metadata", {}),
                "spec": data.get("spec", {}),
                "status": existing_status,
            }
            return _completed()
        elif kind == "Namespace":
            return _completed()
        elif kind == "ServiceAccount":
            return _completed()

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

    def _handle_get_nodepool(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = args.index("nodepool") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing nodepool name")
        name = args[idx]
        if name == "-o":
            return _completed(returncode=1, stderr="missing nodepool name")
        if name not in self._nodepools:
            return _completed(returncode=1, stderr=f'nodepools.compute.coreweave.com "{name}" not found')
        return _completed(stdout=json.dumps(self._nodepools[name]))

    def _handle_list_nodepools(self, args: list[str]) -> subprocess.CompletedProcess:
        items = list(self._nodepools.values())
        if "-l" in args:
            selector_idx = args.index("-l") + 1
            if selector_idx < len(args):
                selector = args[selector_idx]
                items = self._filter_by_selector(items, selector)
        return _completed(stdout=json.dumps({"items": items}))

    def _handle_get_pods(self, args: list[str], namespace: str) -> subprocess.CompletedProcess:
        items = list(self._pods.values())
        if "-l" in args:
            selector_idx = args.index("-l") + 1
            if selector_idx < len(args):
                selector = args[selector_idx]
                items = self._filter_by_selector(items, selector)

        return _completed(stdout=json.dumps({"items": items}))

    def _handle_get_node(self, args: list[str]) -> subprocess.CompletedProcess:
        idx = args.index("node") + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr="missing node name")
        name = args[idx]
        if name not in self._nodes:
            return _completed(returncode=1, stderr=f"node {name} not found")
        return _completed(stdout=json.dumps(self._nodes[name]))

    def _handle_delete_generic(
        self, args: list[str], resource_type: str, store: dict[str, dict]
    ) -> subprocess.CompletedProcess:
        idx = args.index(resource_type) + 1
        if idx >= len(args):
            return _completed(returncode=1, stderr=f"missing {resource_type} name")
        name = args[idx]
        store.pop(name, None)
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
# Tests: unsupported operations
# ============================================================================


def test_create_vm_raises_platform_error():
    """CoreWeave does not support standalone VMs."""
    platform = _make_platform()
    with pytest.raises(PlatformError, match="standalone VMs"):
        platform.create_vm(config_pb2.VmConfig())
    platform.shutdown()


def test_create_slice_raises_platform_error():
    """CoreWeave does not manage slices (KubernetesProvider does)."""
    platform = _make_platform()
    config = config_pb2.SliceConfig(name_prefix="h100-8x")
    with pytest.raises(PlatformError, match="does not manage slices"):
        platform.create_slice(config)
    platform.shutdown()


def test_list_slices_returns_empty():
    """list_slices returns empty list since CoreWeave doesn't manage slices."""
    platform = _make_platform()
    assert platform.list_slices(zones=["LGA1"]) == []
    platform.shutdown()


def test_list_all_slices_returns_empty():
    """list_all_slices returns empty list since CoreWeave doesn't manage slices."""
    platform = _make_platform()
    assert platform.list_all_slices() == []
    platform.shutdown()


def test_list_vms_returns_empty():
    """list_vms returns empty list since CoreWeave doesn't manage VMs."""
    platform = _make_platform()
    assert platform.list_vms(zones=["LGA1"]) == []
    platform.shutdown()


# ============================================================================
# Tests: discover_controller
# ============================================================================


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


# ============================================================================
# Tests: factory
# ============================================================================


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


# ============================================================================
# Tests: start_controller / stop_controller / tunnel
# ============================================================================


def _make_cluster_config(
    port: int = 10000,
    service_name: str = "iris-controller-svc",
    image: str = "ghcr.io/marin-community/iris-controller:latest",
    remote_state_dir: str = "gs://test-bucket/bundles",
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
            remote_state_dir=remote_state_dir,
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
    """start_controller creates ConfigMap, Deployment, and Service."""
    platform = _make_platform()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")

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
    assert container["resources"]["requests"] == {"cpu": _CONTROLLER_CPU_REQUEST, "memory": _CONTROLLER_MEMORY_REQUEST}
    assert container["resources"]["limits"] == {"cpu": _CONTROLLER_CPU_REQUEST, "memory": _CONTROLLER_MEMORY_REQUEST}

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

    # ConfigMap and Service should all be reconciled
    assert "iris-cluster-config" in fake_kubectl._configmaps
    assert "iris-controller-svc" in fake_kubectl._services
    platform.shutdown()


def test_stop_controller_deletes_resources(fake_kubectl: FakeKubectl):
    """stop_controller deletes Deployment, Service, ConfigMap, S3 secret, and RBAC."""
    platform = _make_platform()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")

    # Pre-populate resources
    fake_kubectl._deployments["iris-controller"] = {
        "metadata": {},
        "spec": {},
        "status": {},
    }
    fake_kubectl._services["iris-controller-svc"] = {"metadata": {}, "spec": {}}
    fake_kubectl._configmaps["iris-cluster-config"] = {"metadata": {}, "data": {}}
    fake_kubectl._secrets["iris-s3-credentials"] = {"metadata": {}, "data": {}}

    platform.stop_controller(cluster_config)

    assert "iris-controller" not in fake_kubectl._deployments
    assert "iris-controller-svc" not in fake_kubectl._services
    assert "iris-cluster-config" not in fake_kubectl._configmaps
    assert "iris-s3-credentials" not in fake_kubectl._secrets
    platform.shutdown()


def test_stop_controller_idempotent(fake_kubectl: FakeKubectl):
    """stop_controller succeeds even if resources don't exist."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    # No resources exist -- should not raise
    platform.stop_controller(cluster_config)
    platform.shutdown()


def test_stop_all_only_stops_controller(fake_kubectl: FakeKubectl):
    """stop_all only stops the controller (no worker slices to enumerate)."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    fake_kubectl._deployments["iris-controller"] = {"metadata": {}, "spec": {}, "status": {}}
    fake_kubectl._services["iris-controller-svc"] = {"metadata": {}, "spec": {}}
    fake_kubectl._configmaps["iris-cluster-config"] = {"metadata": {}, "data": {}}

    targets = platform.stop_all(cluster_config)

    assert targets == ["controller"]
    assert "iris-controller" not in fake_kubectl._deployments
    platform.shutdown()


def test_stop_all_dry_run(fake_kubectl: FakeKubectl):
    """stop_all with dry_run=True returns target names without deleting."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()

    fake_kubectl._deployments["iris-controller"] = {"metadata": {}, "spec": {}, "status": {}}

    targets = platform.stop_all(cluster_config, dry_run=True)

    assert targets == ["controller"]
    # Deployment should still exist
    assert "iris-controller" in fake_kubectl._deployments
    platform.shutdown()


# ============================================================================
# Tests: RBAC
# ============================================================================


def test_rbac_isolation_across_namespaces(fake_kubectl: FakeKubectl):
    """Two Iris instances with different namespaces get isolated RBAC; teardown of one doesn't affect the other."""
    platform_a = _make_platform(namespace="alpha")
    platform_b = _make_platform(namespace="beta")

    platform_a.ensure_rbac()
    platform_b.ensure_rbac()

    # Each gets a namespace-qualified ClusterRole and ClusterRoleBinding
    assert "iris-controller-alpha" in fake_kubectl._cluster_roles
    assert "iris-controller-beta" in fake_kubectl._cluster_roles

    # Binding references the correct ClusterRole and namespace
    binding_a = fake_kubectl._cluster_role_bindings["iris-controller-alpha"]
    assert binding_a["roleRef"]["name"] == "iris-controller-alpha"
    assert binding_a["subjects"][0]["namespace"] == "alpha"

    # Stopping alpha cleans up its RBAC without affecting beta
    platform_a.stop_controller(_make_cluster_config())

    assert "iris-controller-alpha" not in fake_kubectl._cluster_roles
    assert "iris-controller-alpha" not in fake_kubectl._cluster_role_bindings
    assert "iris-controller-beta" in fake_kubectl._cluster_roles
    assert "iris-controller-beta" in fake_kubectl._cluster_role_bindings

    platform_a.shutdown()
    platform_b.shutdown()


# ============================================================================
# Tests: tunnel
# ============================================================================


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


# ============================================================================
# Tests: controller deployment details
# ============================================================================


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

    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
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


# ============================================================================
# Tests: controller error handling
# ============================================================================


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
    cluster_config = _make_cluster_config(remote_state_dir="s3://my-bucket/bundles")

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
    cluster_config = _make_cluster_config(remote_state_dir="gs://test-bucket/bundles")

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


def test_ensure_nodepools_scales_multihost_groups_by_num_vms(fake_kubectl: FakeKubectl):
    """NodePool capacity is counted in nodes, so multihost groups scale by num_vms per slice."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()
    sg = cluster_config.scale_groups["h100-16x"]
    sg.min_slices = 0
    sg.max_slices = 1
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix="h100-16x",
            num_vms=2,
            coreweave=config_pb2.CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128"),
        )
    )

    platform.ensure_nodepools(cluster_config)

    h100_pool = fake_kubectl._nodepools["iris-h100-16x"]
    assert h100_pool["spec"]["minNodes"] == 0
    assert h100_pool["spec"]["maxNodes"] == 2
    platform.shutdown()


def test_ensure_nodepools_keeps_one_multihost_slice_warm(fake_kubectl: FakeKubectl):
    """Existing multihost pools keep one full slice worth of desired nodes."""
    platform = _make_platform()
    cluster_config = _make_cluster_config()
    sg = cluster_config.scale_groups["h100-16x"]
    sg.min_slices = 0
    sg.max_slices = 1
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix="h100-16x",
            num_vms=2,
            coreweave=config_pb2.CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128"),
        )
    )

    fake_kubectl._nodepools["iris-h100-16x"] = {
        "metadata": {"name": "iris-h100-16x", "labels": {}},
        "spec": {"instanceType": "gd-8xh100ib-i128", "minNodes": 0, "maxNodes": 1, "targetNodes": 1},
        "status": {"readyNodes": 1, "currentNodes": 1, "conditions": []},
    }

    platform.ensure_nodepools(cluster_config)

    assert fake_kubectl._nodepools["iris-h100-16x"]["spec"]["targetNodes"] == 2
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
