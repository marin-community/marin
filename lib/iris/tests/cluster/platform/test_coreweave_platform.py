# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for CoreweavePlatform.

Tests exercise the public interface of CoreweavePlatform using K8sServiceImpl
(in-memory K8s service). We test controller lifecycle, discovery, tunnel, RBAC,
and configuration. Worker/slice management is handled by KubernetesProvider (not
CoreweavePlatform).
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from iris.cluster.k8s.k8s_service_impl import K8sServiceImpl
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
    label_prefix: str = "iris",
    k8s: K8sServiceImpl | None = None,
) -> tuple[CoreweavePlatform, K8sServiceImpl]:
    k8s = k8s or K8sServiceImpl(namespace=namespace)
    config = config_pb2.CoreweavePlatformConfig(
        region=region,
        namespace=namespace,
    )
    platform = CoreweavePlatform(config=config, label_prefix=label_prefix, poll_interval=0.05, kubectl=k8s)
    return platform, k8s


def _auto_ready_deployment(k8s: K8sServiceImpl, name: str, timeout: float = 10):
    """Wait for deployment to appear in the in-memory store, then mark it available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        dep = k8s.get_json("deployment", name)
        if dep is not None:
            dep.setdefault("status", {})["availableReplicas"] = dep.get("spec", {}).get("replicas", 1)
            return
        time.sleep(0.05)


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
    platform, _ = _make_platform()
    with pytest.raises(PlatformError, match="standalone VMs"):
        platform.create_vm(config_pb2.VmConfig())
    platform.shutdown()


def test_create_slice_raises_platform_error():
    """CoreWeave does not manage slices (KubernetesProvider does)."""
    platform, _ = _make_platform()
    config = config_pb2.SliceConfig(name_prefix="h100-8x")
    with pytest.raises(PlatformError, match="does not manage slices"):
        platform.create_slice(config)
    platform.shutdown()


def test_list_slices_returns_empty():
    """list_slices returns empty list since CoreWeave doesn't manage slices."""
    platform, _ = _make_platform()
    assert platform.list_slices(zones=["LGA1"]) == []
    platform.shutdown()


def test_list_all_slices_returns_empty():
    """list_all_slices returns empty list since CoreWeave doesn't manage slices."""
    platform, _ = _make_platform()
    assert platform.list_all_slices() == []
    platform.shutdown()


def test_list_vms_returns_empty():
    """list_vms returns empty list since CoreWeave doesn't manage VMs."""
    platform, _ = _make_platform()
    assert platform.list_vms(zones=["LGA1"]) == []
    platform.shutdown()


# ============================================================================
# Tests: discover_controller
# ============================================================================


def test_discover_controller_dns():
    """discover_controller returns correct K8s Service DNS name."""
    k8s = K8sServiceImpl(namespace="iris")
    config = config_pb2.CoreweavePlatformConfig(region="LGA1", namespace="iris")
    platform = CoreweavePlatform(config=config, label_prefix="iris", kubectl=k8s)

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
    k8s = K8sServiceImpl(namespace="my-ns")
    config = config_pb2.CoreweavePlatformConfig(region="LGA1", namespace="my-ns")
    platform = CoreweavePlatform(config=config, label_prefix="iris", kubectl=k8s)

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


def test_start_controller_creates_all_resources():
    """start_controller creates ConfigMap, Deployment, and Service."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    address = platform.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    assert k8s.get_json("secret", "iris-s3-credentials") is not None
    assert k8s.get_json("configmap", "iris-cluster-config") is not None
    assert k8s.get_json("deployment", "iris-controller") is not None
    assert k8s.get_json("service", "iris-controller-svc") is not None

    # Verify Deployment nodeSelector targets the configured scale group
    iris_labels = Labels("iris")
    dep = k8s.get_json("deployment", "iris-controller")
    deploy_spec = dep["spec"]
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


def test_start_controller_reconciles_when_already_available():
    """start_controller reconciles all resources even if Deployment is already available."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config()

    # K8sServiceImpl replaces the full manifest on re-apply, so use auto_ready thread
    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    address = platform.start_controller(cluster_config)
    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"

    # ConfigMap and Service should all be reconciled
    assert k8s.get_json("configmap", "iris-cluster-config") is not None
    assert k8s.get_json("service", "iris-controller-svc") is not None

    t.join(timeout=5)
    platform.shutdown()


def test_stop_controller_deletes_resources():
    """stop_controller deletes Deployment, Service, ConfigMap, S3 secret, and RBAC."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")

    # Pre-populate resources
    _apply_stub(k8s, "Deployment", "iris-controller")
    _apply_stub(k8s, "Service", "iris-controller-svc")
    _apply_stub(k8s, "ConfigMap", "iris-cluster-config")
    _apply_stub(k8s, "Secret", "iris-s3-credentials")

    platform.stop_controller(cluster_config)

    assert k8s.get_json("deployment", "iris-controller") is None
    assert k8s.get_json("service", "iris-controller-svc") is None
    assert k8s.get_json("configmap", "iris-cluster-config") is None
    assert k8s.get_json("secret", "iris-s3-credentials") is None
    platform.shutdown()


def test_stop_controller_idempotent():
    """stop_controller succeeds even if resources don't exist."""
    platform, _ = _make_platform()
    cluster_config = _make_cluster_config()

    # No resources exist -- should not raise
    platform.stop_controller(cluster_config)
    platform.shutdown()


def test_stop_all_only_stops_controller():
    """stop_all only stops the controller (no worker slices to enumerate)."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config()

    _apply_stub(k8s, "Deployment", "iris-controller")
    _apply_stub(k8s, "Service", "iris-controller-svc")
    _apply_stub(k8s, "ConfigMap", "iris-cluster-config")

    targets = platform.stop_all(cluster_config)

    assert targets == ["controller"]
    assert k8s.get_json("deployment", "iris-controller") is None
    platform.shutdown()


def test_stop_all_dry_run():
    """stop_all with dry_run=True returns target names without deleting."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config()

    _apply_stub(k8s, "Deployment", "iris-controller")

    targets = platform.stop_all(cluster_config, dry_run=True)

    assert targets == ["controller"]
    # Deployment should still exist
    assert k8s.get_json("deployment", "iris-controller") is not None
    platform.shutdown()


# ============================================================================
# Tests: RBAC
# ============================================================================


def test_rbac_isolation_across_namespaces():
    """Two Iris instances with different namespaces get isolated RBAC; teardown of one doesn't affect the other."""
    k8s = K8sServiceImpl(namespace="alpha")
    platform_a, _ = _make_platform(namespace="alpha", k8s=k8s)
    platform_b, _ = _make_platform(namespace="beta", k8s=k8s)

    platform_a.ensure_rbac()
    platform_b.ensure_rbac()

    # Each gets a namespace-qualified ClusterRole and ClusterRoleBinding
    assert k8s.get_json("clusterrole", "iris-controller-alpha") is not None
    assert k8s.get_json("clusterrole", "iris-controller-beta") is not None

    # Binding references the correct ClusterRole and namespace
    binding_a = k8s.get_json("clusterrolebinding", "iris-controller-alpha")
    assert binding_a["roleRef"]["name"] == "iris-controller-alpha"
    assert binding_a["subjects"][0]["namespace"] == "alpha"

    # Stopping alpha cleans up its RBAC without affecting beta
    platform_a.stop_controller(_make_cluster_config())

    assert k8s.get_json("clusterrole", "iris-controller-alpha") is None
    assert k8s.get_json("clusterrolebinding", "iris-controller-alpha") is None
    assert k8s.get_json("clusterrole", "iris-controller-beta") is not None
    assert k8s.get_json("clusterrolebinding", "iris-controller-beta") is not None

    platform_a.shutdown()
    platform_b.shutdown()


# ============================================================================
# Tests: tunnel
# ============================================================================


def test_tunnel_parses_address_and_forwards():
    """tunnel() parses address and delegates to K8sService.port_forward()."""
    k8s = K8sServiceImpl(namespace="iris")
    config = config_pb2.CoreweavePlatformConfig(region="LGA1", namespace="iris")
    platform = CoreweavePlatform(config=config, label_prefix="iris", kubectl=k8s)

    controller_config = config_pb2.ControllerVmConfig(
        coreweave=config_pb2.CoreweaveControllerConfig(port=9999, service_name="my-svc"),
    )
    address = platform.discover_controller(controller_config)
    assert address == "my-svc.iris.svc.cluster.local:9999"

    with platform.tunnel(address) as url:
        assert url.startswith("http://127.0.0.1:")
    platform.shutdown()


# ============================================================================
# Tests: controller deployment details
# ============================================================================


def test_start_controller_deployment_command_references_config_json():
    """The controller Deployment command uses config.json (not config.yaml)."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config()

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    platform.start_controller(cluster_config)

    dep = k8s.get_json("deployment", "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    # Must reference config.json, not config.yaml
    config_args = [arg for arg in container["command"] if "config" in arg and arg.startswith("--config")]
    assert len(config_args) == 1
    assert config_args[0] == "--config=/etc/iris/config.json"

    t.join(timeout=5)
    platform.shutdown()


def test_configmap_strips_kubeconfig_path():
    """ConfigMap must not contain kubeconfig_path so pods use in-cluster auth."""
    k8s = K8sServiceImpl(namespace="iris")
    cw_config = config_pb2.CoreweavePlatformConfig(
        region="LGA1",
        namespace="iris",
        kubeconfig_path="/home/user/.kube/coreweave-iris",
    )
    platform = CoreweavePlatform(config=cw_config, label_prefix="iris", poll_interval=0.05, kubectl=k8s)

    cluster_config = _make_cluster_config()
    cluster_config.platform.coreweave.kubeconfig_path = "/home/user/.kube/coreweave-iris"

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    platform.start_controller(cluster_config)

    cm = k8s.get_json("configmap", "iris-cluster-config")
    cm_data = json.loads(cm["data"]["config.json"])
    cw_config_data = cm_data.get("platform", {}).get("coreweave", {})
    assert "kubeconfig_path" not in cw_config_data

    t.join(timeout=5)
    platform.shutdown()


def test_controller_deployment_includes_endpoint_url():
    """When object_storage_endpoint is set, the controller Deployment includes AWS_ENDPOINT_URL."""
    k8s = K8sServiceImpl(namespace="iris")
    cw_config = config_pb2.CoreweavePlatformConfig(
        region="LGA1",
        namespace="iris",
        object_storage_endpoint="https://object.lga1.coreweave.com",
    )
    platform = CoreweavePlatform(config=cw_config, label_prefix="iris", poll_interval=0.05, kubectl=k8s)

    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.platform.coreweave.object_storage_endpoint = "https://object.lga1.coreweave.com"

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    platform.start_controller(cluster_config)

    dep = k8s.get_json("deployment", "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    env_by_name = {e["name"]: e for e in container["env"]}
    assert "AWS_ENDPOINT_URL" in env_by_name
    assert env_by_name["AWS_ENDPOINT_URL"]["value"] == "https://object.lga1.coreweave.com"

    t.join(timeout=5)
    platform.shutdown()


# ============================================================================
# Tests: controller error handling
# ============================================================================


def test_start_controller_errors_without_scale_group():
    """start_controller raises when scale_group is not set."""
    platform, _ = _make_platform()
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


def test_start_controller_errors_with_invalid_scale_group():
    """start_controller raises when scale_group references a nonexistent group."""
    platform, _ = _make_platform()
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


def test_start_controller_errors_without_s3_credentials(monkeypatch):
    """start_controller raises when S3 storage is configured but R2 credentials are not set."""
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    platform, _ = _make_platform()
    cluster_config = _make_cluster_config(remote_state_dir="s3://my-bucket/bundles")

    with pytest.raises(PlatformError, match="R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY"):
        platform.start_controller(cluster_config)
    platform.shutdown()


def test_start_controller_detects_crash_loop_backoff():
    """start_controller fails fast when the controller Pod enters CrashLoopBackOff."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config()

    crash_logs = (
        "ValueError: scale_groups.cpu-erapids.resources has unknown keys: memory_bytes\n"
        "Error: Failed to load cluster config\n"
    )

    def simulate_crash_loop():
        _wait_for_condition(lambda: k8s.get_json("deployment", "iris-controller") is not None, timeout=10)
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

    with pytest.raises(PlatformError, match="CrashLoopBackOff"):
        platform.start_controller(cluster_config)

    t.join(timeout=5)
    platform.shutdown()


def test_start_controller_detects_image_pull_failure():
    """start_controller fails fast on ImagePullBackOff."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config()

    def simulate_image_pull_failure():
        _wait_for_condition(lambda: k8s.get_json("deployment", "iris-controller") is not None, timeout=10)
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

    with pytest.raises(PlatformError, match="ImagePullBackOff"):
        platform.start_controller(cluster_config)

    t.join(timeout=5)
    platform.shutdown()


def test_start_controller_crash_loop_includes_logs():
    """When CrashLoopBackOff is detected, the error includes container logs."""
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config()

    crash_logs = "ValueError: bad config key\nTraceback ...\n"

    def simulate_crash_loop():
        _wait_for_condition(lambda: k8s.get_json("deployment", "iris-controller") is not None, timeout=10)
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

    with pytest.raises(PlatformError, match="bad config key"):
        platform.start_controller(cluster_config)

    t.join(timeout=5)
    platform.shutdown()


def test_start_controller_skips_s3_for_gs_storage(monkeypatch):
    """start_controller succeeds without S3 credentials when using gs:// storage."""
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    platform, k8s = _make_platform()
    cluster_config = _make_cluster_config(remote_state_dir="gs://test-bucket/bundles")

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    address = platform.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    # No S3 secret should be created
    assert k8s.get_json("secret", "iris-s3-credentials") is None
    # Controller Deployment should have no S3 env vars
    dep = k8s.get_json("deployment", "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    env_names = [e["name"] for e in container["env"]]
    assert "AWS_ACCESS_KEY_ID" not in env_names
    assert "AWS_SECRET_ACCESS_KEY" not in env_names

    t.join(timeout=5)
    platform.shutdown()


def test_ensure_nodepools_scales_multihost_groups_by_num_vms():
    """NodePool capacity is counted in nodes, so multihost groups scale by num_vms per slice."""
    platform, k8s = _make_platform()
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

    h100_pool = k8s.get_json("nodepool", "iris-h100-16x")
    assert h100_pool is not None
    assert h100_pool["spec"]["minNodes"] == 0
    assert h100_pool["spec"]["maxNodes"] == 2
    platform.shutdown()


def test_ensure_nodepools_keeps_one_multihost_slice_warm():
    """Existing multihost pools keep one full slice worth of desired nodes."""
    platform, k8s = _make_platform()
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

    platform.ensure_nodepools(cluster_config)

    h100_pool = k8s.get_json("nodepool", "iris-h100-16x")
    assert h100_pool["spec"]["targetNodes"] == 2
    platform.shutdown()


# ============================================================================
# Helpers
# ============================================================================


def _apply_stub(k8s: K8sServiceImpl, kind: str, name: str, namespace: str = "iris") -> None:
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
