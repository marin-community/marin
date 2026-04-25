# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for K8sControllerProvider (CoreWeave CKS).

Tests exercise the public interface using InMemoryK8sService (in-memory K8s
service). We test controller lifecycle, discovery, tunnel, RBAC, and
configuration. Worker/slice management is handled by K8sTaskProvider (not
K8sControllerProvider).
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from iris.cluster.providers.k8s.controller import (
    K8sControllerProvider,
    _CONTROLLER_CPU_REQUEST,
    _CONTROLLER_MEMORY_REQUEST,
)
from iris.cluster.dashboard_common import DASHBOARD_TITLE_ENV_VAR
from iris.cluster.providers.k8s.fake import InMemoryK8sService
from iris.cluster.providers.k8s.types import K8sResource
from iris.cluster.providers.types import (
    Labels,
    InfraError,
)
from iris.rpc import config_pb2


def _make_provider(
    region: str = "LGA1",
    namespace: str = "iris",
    label_prefix: str = "iris",
    k8s: InMemoryK8sService | None = None,
) -> tuple[K8sControllerProvider, InMemoryK8sService]:
    k8s = k8s or InMemoryK8sService(namespace=namespace)
    config = config_pb2.CoreweavePlatformConfig(
        region=region,
        namespace=namespace,
    )
    provider = K8sControllerProvider(config=config, label_prefix=label_prefix, poll_interval=0.05, kubectl=k8s)
    return provider, k8s


def _auto_ready_deployment(k8s: InMemoryK8sService, name: str, timeout: float = 10):
    """Wait for deployment to appear in the in-memory store, then mark it available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        dep = k8s.get_json(K8sResource.DEPLOYMENTS, name)
        if dep is not None:
            dep.setdefault("status", {})["availableReplicas"] = dep.get("spec", {}).get("replicas", 1)
            return
        time.sleep(0.05)


@pytest.fixture(autouse=True)
def _s3_env_vars(monkeypatch):
    """Set R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY so ensure_s3_credentials_secret() succeeds."""
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-key-secret")


# ============================================================================
# Tests: discover_controller
# ============================================================================


def test_discover_controller_dns():
    """discover_controller returns correct K8s Service DNS name."""
    provider, _ = _make_provider()
    controller_config = config_pb2.ControllerVmConfig(
        coreweave=config_pb2.CoreweaveControllerConfig(
            port=10000,
            service_name="iris-controller-svc",
        )
    )
    address = provider.discover_controller(controller_config)
    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    provider.shutdown()


def test_discover_controller_defaults():
    """discover_controller uses default port and service name when not set."""
    provider, _ = _make_provider(namespace="my-ns")
    controller_config = config_pb2.ControllerVmConfig(coreweave=config_pb2.CoreweaveControllerConfig())
    address = provider.discover_controller(controller_config)
    assert address == "iris-controller-svc.my-ns.svc.cluster.local:10000"
    provider.shutdown()


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
    sg.buffer_slices = 0
    sg.max_slices = 10
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix=controller_scale_group,
            num_vms=1,
            coreweave=config_pb2.CoreweaveSliceConfig(instance_type="cd-gp-i64-erapids"),
        )
    )
    return config


def test_start_controller_creates_all_resources():
    """start_controller creates ConfigMap, Deployment, and Service."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    address = provider.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    assert k8s.get_json(K8sResource.SECRETS, "iris-s3-credentials") is not None
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config") is not None
    assert k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is not None
    assert k8s.get_json(K8sResource.SERVICES, "iris-controller-svc") is not None

    # Verify Deployment nodeSelector targets the configured scale group
    iris_labels = Labels("iris")
    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    deploy_spec = dep["spec"]
    node_selector = deploy_spec["template"]["spec"]["nodeSelector"]
    assert node_selector == {iris_labels.iris_scale_group: "cpu-erapids"}

    # Verify controller uses S3 env vars (no GCS credentials)
    container = deploy_spec["template"]["spec"]["containers"][0]
    env_by_name = {e["name"]: e for e in container["env"]}
    assert "AWS_ACCESS_KEY_ID" in env_by_name
    assert "AWS_SECRET_ACCESS_KEY" in env_by_name
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env_by_name
    assert env_by_name[DASHBOARD_TITLE_ENV_VAR]["value"] == "iris"
    assert container["resources"]["requests"] == {"cpu": _CONTROLLER_CPU_REQUEST, "memory": _CONTROLLER_MEMORY_REQUEST}
    assert container["resources"]["limits"] == {"cpu": _CONTROLLER_CPU_REQUEST, "memory": _CONTROLLER_MEMORY_REQUEST}

    t.join(timeout=5)
    provider.shutdown()


def test_start_controller_reconciles_when_already_available():
    """start_controller reconciles all resources even if Deployment is already available."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    # InMemoryK8sService replaces the full manifest on re-apply, so use auto_ready thread
    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    address = provider.start_controller(cluster_config)
    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"

    # ConfigMap and Service should all be reconciled
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config") is not None
    assert k8s.get_json(K8sResource.SERVICES, "iris-controller-svc") is not None

    t.join(timeout=5)
    provider.shutdown()


def test_stop_controller_deletes_resources():
    """stop_controller deletes Deployment, Service, ConfigMap, S3 secret, and RBAC."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")

    # Pre-populate resources
    _apply_stub(k8s, "Deployment", "iris-controller")
    _apply_stub(k8s, "Service", "iris-controller-svc")
    _apply_stub(k8s, "ConfigMap", "iris-cluster-config")
    _apply_stub(k8s, "Secret", "iris-s3-credentials")

    provider.stop_controller(cluster_config)

    assert k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is None
    assert k8s.get_json(K8sResource.SERVICES, "iris-controller-svc") is None
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config") is None
    assert k8s.get_json(K8sResource.SECRETS, "iris-s3-credentials") is None
    provider.shutdown()


def test_stop_controller_idempotent():
    """stop_controller succeeds even if resources don't exist."""
    provider, _ = _make_provider()
    cluster_config = _make_cluster_config()

    # No resources exist -- should not raise
    provider.stop_controller(cluster_config)
    provider.shutdown()


def test_stop_all_only_stops_controller():
    """stop_all only stops the controller (no worker slices to enumerate)."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    _apply_stub(k8s, "Deployment", "iris-controller")
    _apply_stub(k8s, "Service", "iris-controller-svc")
    _apply_stub(k8s, "ConfigMap", "iris-cluster-config")

    targets = provider.stop_all(cluster_config)

    assert targets == ["controller"]
    assert k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is None
    provider.shutdown()


def test_stop_all_dry_run():
    """stop_all with dry_run=True returns target names without deleting."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    _apply_stub(k8s, "Deployment", "iris-controller")

    targets = provider.stop_all(cluster_config, dry_run=True)

    assert targets == ["controller"]
    # Deployment should still exist
    assert k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is not None
    provider.shutdown()


# ============================================================================
# Tests: RBAC
# ============================================================================


def test_rbac_isolation_across_namespaces():
    """Two Iris instances with different namespaces get isolated RBAC; teardown of one doesn't affect the other."""
    k8s = InMemoryK8sService(namespace="alpha")
    provider_a, _ = _make_provider(namespace="alpha", k8s=k8s)
    provider_b, _ = _make_provider(namespace="beta", k8s=k8s)

    provider_a.ensure_rbac()
    provider_b.ensure_rbac()

    # Each gets a namespace-qualified ClusterRole and ClusterRoleBinding
    assert k8s.get_json(K8sResource.CLUSTER_ROLES, "iris-controller-alpha") is not None
    assert k8s.get_json(K8sResource.CLUSTER_ROLES, "iris-controller-beta") is not None

    # Binding references the correct ClusterRole and namespace
    binding_a = k8s.get_json(K8sResource.CLUSTER_ROLE_BINDINGS, "iris-controller-alpha")
    assert binding_a["roleRef"]["name"] == "iris-controller-alpha"
    assert binding_a["subjects"][0]["namespace"] == "alpha"

    # Stopping alpha cleans up its RBAC without affecting beta
    provider_a.stop_controller(_make_cluster_config())

    assert k8s.get_json(K8sResource.CLUSTER_ROLES, "iris-controller-alpha") is None
    assert k8s.get_json(K8sResource.CLUSTER_ROLE_BINDINGS, "iris-controller-alpha") is None
    assert k8s.get_json(K8sResource.CLUSTER_ROLES, "iris-controller-beta") is not None
    assert k8s.get_json(K8sResource.CLUSTER_ROLE_BINDINGS, "iris-controller-beta") is not None

    provider_a.shutdown()
    provider_b.shutdown()


# ============================================================================
# Tests: tunnel
# ============================================================================


def test_tunnel_parses_address_and_forwards():
    """tunnel() parses address and delegates to K8sService.port_forward()."""
    provider, _ = _make_provider()
    controller_config = config_pb2.ControllerVmConfig(
        coreweave=config_pb2.CoreweaveControllerConfig(port=9999, service_name="my-svc"),
    )
    address = provider.discover_controller(controller_config)
    assert address == "my-svc.iris.svc.cluster.local:9999"

    with provider.tunnel(address) as url:
        assert url.startswith("http://127.0.0.1:")
    provider.shutdown()


# ============================================================================
# Tests: controller deployment details
# ============================================================================


def test_start_controller_deployment_command_references_config_json():
    """The controller Deployment command uses config.json (not config.yaml)."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    provider.start_controller(cluster_config)

    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    # Must reference config.json, not config.yaml
    config_args = [arg for arg in container["command"] if "config" in arg and arg.startswith("--config")]
    assert len(config_args) == 1
    assert config_args[0] == "--config=/etc/iris/config.json"

    t.join(timeout=5)
    provider.shutdown()


def test_configmap_strips_kubeconfig_path():
    """ConfigMap must not contain kubeconfig_path so pods use in-cluster auth."""
    k8s = InMemoryK8sService(namespace="iris")
    cw_config = config_pb2.CoreweavePlatformConfig(
        region="LGA1",
        namespace="iris",
        kubeconfig_path="/home/user/.kube/coreweave-iris",
    )
    provider = K8sControllerProvider(config=cw_config, label_prefix="iris", poll_interval=0.05, kubectl=k8s)

    cluster_config = _make_cluster_config()
    cluster_config.platform.coreweave.kubeconfig_path = "/home/user/.kube/coreweave-iris"

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    provider.start_controller(cluster_config)

    cm = k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config")
    cm_data = json.loads(cm["data"]["config.json"])
    cw_config_data = cm_data.get("platform", {}).get("coreweave", {})
    assert "kubeconfig_path" not in cw_config_data

    t.join(timeout=5)
    provider.shutdown()


def test_controller_deployment_includes_endpoint_url():
    """When object_storage_endpoint is set, the controller Deployment includes AWS_ENDPOINT_URL."""
    k8s = InMemoryK8sService(namespace="iris")
    cw_config = config_pb2.CoreweavePlatformConfig(
        region="LGA1",
        namespace="iris",
        object_storage_endpoint="https://object.lga1.coreweave.com",
    )
    provider = K8sControllerProvider(config=cw_config, label_prefix="iris", poll_interval=0.05, kubectl=k8s)

    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.platform.coreweave.object_storage_endpoint = "https://object.lga1.coreweave.com"

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    provider.start_controller(cluster_config)

    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    env_by_name = {e["name"]: e for e in container["env"]}
    assert "AWS_ENDPOINT_URL" in env_by_name
    assert env_by_name["AWS_ENDPOINT_URL"]["value"] == "https://object.lga1.coreweave.com"

    t.join(timeout=5)
    provider.shutdown()


# ============================================================================
# Tests: controller error handling
# ============================================================================


def test_start_controller_errors_without_scale_group():
    """start_controller raises when scale_group is not set."""
    provider, _ = _make_provider()
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
    with pytest.raises(InfraError, match="must set scale_group"):
        provider.start_controller(config)
    provider.shutdown()


def test_start_controller_errors_with_invalid_scale_group():
    """start_controller raises when scale_group references a nonexistent group."""
    provider, _ = _make_provider()
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
    with pytest.raises(InfraError, match="not found in scale_groups"):
        provider.start_controller(config)
    provider.shutdown()


def test_start_controller_errors_without_s3_credentials(monkeypatch):
    """start_controller raises when S3 storage is configured but R2 credentials are not set."""
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    provider, _ = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://my-bucket/bundles")

    with pytest.raises(InfraError, match="R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY"):
        provider.start_controller(cluster_config)
    provider.shutdown()


def test_start_controller_detects_crash_loop_backoff():
    """start_controller fails fast when the controller Pod enters CrashLoopBackOff."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    crash_logs = (
        "ValueError: scale_groups.cpu-erapids.resources has unknown keys: memory_bytes\n"
        "Error: Failed to load cluster config\n"
    )

    def simulate_crash_loop():
        _wait_for_condition(lambda: k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is not None, timeout=10)
        k8s.apply_json(
            {
                "kind": "Pod",
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
        )
        k8s.set_logs("iris-controller-abc123", crash_logs)

    t = threading.Thread(target=simulate_crash_loop, daemon=True)
    t.start()

    with pytest.raises(InfraError, match="CrashLoopBackOff"):
        provider.start_controller(cluster_config)

    t.join(timeout=5)
    provider.shutdown()


def test_start_controller_detects_image_pull_failure():
    """start_controller fails fast on ImagePullBackOff."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    def simulate_image_pull_failure():
        _wait_for_condition(lambda: k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is not None, timeout=10)
        k8s.apply_json(
            {
                "kind": "Pod",
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
        )

    t = threading.Thread(target=simulate_image_pull_failure, daemon=True)
    t.start()

    with pytest.raises(InfraError, match="ImagePullBackOff"):
        provider.start_controller(cluster_config)

    t.join(timeout=5)
    provider.shutdown()


def test_start_controller_crash_loop_includes_logs():
    """When CrashLoopBackOff is detected, the error includes container logs."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    crash_logs = "ValueError: bad config key\nTraceback ...\n"

    def simulate_crash_loop():
        _wait_for_condition(lambda: k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is not None, timeout=10)
        k8s.apply_json(
            {
                "kind": "Pod",
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
        )
        k8s.set_logs("iris-controller-xyz", crash_logs)

    t = threading.Thread(target=simulate_crash_loop, daemon=True)
    t.start()

    with pytest.raises(InfraError, match="bad config key"):
        provider.start_controller(cluster_config)

    t.join(timeout=5)
    provider.shutdown()


def test_start_controller_skips_s3_for_gs_storage(monkeypatch):
    """start_controller succeeds without S3 credentials when using gs:// storage."""
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="gs://test-bucket/bundles")

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    address = provider.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    # No S3 secret should be created
    assert k8s.get_json(K8sResource.SECRETS, "iris-s3-credentials") is None
    # Controller Deployment should have no S3 env vars
    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    env_names = [e["name"] for e in container["env"]]
    assert "AWS_ACCESS_KEY_ID" not in env_names
    assert "AWS_SECRET_ACCESS_KEY" not in env_names

    t.join(timeout=5)
    provider.shutdown()


def test_ensure_nodepools_scales_multihost_groups_by_num_vms():
    """NodePool capacity is counted in nodes, so multihost groups scale by num_vms per slice."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    sg = cluster_config.scale_groups["h100-16x"]
    sg.buffer_slices = 0
    sg.max_slices = 1
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix="h100-16x",
            num_vms=2,
            coreweave=config_pb2.CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128"),
        )
    )

    provider.ensure_nodepools(cluster_config)

    h100_pool = k8s.get_json(K8sResource.NODE_POOLS, "iris-h100-16x")
    assert h100_pool is not None
    assert h100_pool["spec"]["minNodes"] == 0
    assert h100_pool["spec"]["maxNodes"] == 2
    provider.shutdown()


def test_ensure_nodepools_keeps_one_multihost_slice_warm():
    """Existing multihost pools keep one full slice worth of desired nodes."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    sg = cluster_config.scale_groups["h100-16x"]
    sg.buffer_slices = 0
    sg.max_slices = 1
    sg.slice_template.CopyFrom(
        config_pb2.SliceConfig(
            name_prefix="h100-16x",
            num_vms=2,
            coreweave=config_pb2.CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128"),
        )
    )

    # Pre-create nodepool so _ensure_one_nodepool detects it as existing
    k8s.apply_json(
        {
            "apiVersion": "compute.coreweave.com/v1alpha1",
            "kind": "NodePool",
            "metadata": {"name": "iris-h100-16x", "labels": {}},
            "spec": {
                "instanceType": "gd-8xh100ib-i128",
                "minNodes": 0,
                "maxNodes": 1,
                "targetNodes": 1,
            },
            "status": {"readyNodes": 1, "currentNodes": 1, "conditions": []},
        }
    )

    provider.ensure_nodepools(cluster_config)

    h100_pool = k8s.get_json(K8sResource.NODE_POOLS, "iris-h100-16x")
    assert h100_pool["spec"]["targetNodes"] == 2
    provider.shutdown()


# ============================================================================
# Helpers
# ============================================================================


def _apply_stub(k8s: InMemoryK8sService, kind: str, name: str, namespace: str = "iris") -> None:
    """Apply a minimal stub resource into the in-memory K8s store."""
    k8s.apply_json({"kind": kind, "metadata": {"name": name, "namespace": namespace}, "spec": {}})


def _wait_for_condition(condition, timeout: float = 5.0, poll: float = 0.05):
    """Poll until condition() is truthy, or raise on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(poll)
    raise TimeoutError(f"Condition not met within {timeout}s")
