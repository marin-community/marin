# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for K8sControllerProvider (CoreWeave CKS).

Tests exercise the public interface using InMemoryK8sService (in-memory K8s
service). We test controller lifecycle, discovery, tunnel, RBAC, and
configuration. Worker/slice management is handled by K8sTaskProvider (not
K8sControllerProvider).
"""

import base64
import json
import os
import threading
import time

import pytest
from iris.cluster.config import (
    ControllerVmConfig,
    CoreweaveControllerConfig,
    CoreweavePlatformConfig,
    CoreweaveSliceConfig,
    IrisClusterConfig,
    KubernetesProviderConfig,
    PlatformConfig,
    ScaleGroupConfig,
    SliceConfig,
    StorageConfig,
)
from iris.cluster.platforms.k8s.controller import (
    _CONTROLLER_CPU_REQUEST,
    _CONTROLLER_MEMORY_REQUEST,
    _CONTROLLER_STATE_PVC_NAME,
    _CONTROLLER_STATE_PVC_SIZE,
    K8sControllerProvider,
    configure_client_s3,
)
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import (
    K8sResource,
    iris_priority_class_manifest,
)
from iris.cluster.platforms.types import (
    InfraError,
    Labels,
)


def _make_provider(
    region: str = "LGA1",
    namespace: str = "iris",
    label_prefix: str = "iris",
    k8s: InMemoryK8sService | None = None,
) -> tuple[K8sControllerProvider, InMemoryK8sService]:
    k8s = k8s or InMemoryK8sService(namespace=namespace)
    config = CoreweavePlatformConfig(
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
    """Set CW_KEY_ID / CW_KEY_SECRET so the S3 task-env build succeeds."""
    monkeypatch.setenv("CW_KEY_ID", "test-key-id")
    monkeypatch.setenv("CW_KEY_SECRET", "test-key-secret")


# ============================================================================
# Tests: discover_controller
# ============================================================================


def test_discover_controller_dns():
    """discover_controller returns correct K8s Service DNS name."""
    provider, _ = _make_provider()
    controller_config = ControllerVmConfig(
        coreweave=CoreweaveControllerConfig(
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
    controller_config = ControllerVmConfig(coreweave=CoreweaveControllerConfig())
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
) -> IrisClusterConfig:
    config = IrisClusterConfig(
        platform=PlatformConfig(
            label_prefix="iris",
            coreweave=CoreweavePlatformConfig(
                region="LGA1",
                namespace="iris",
            ),
        ),
        controller=ControllerVmConfig(
            image=image,
            coreweave=CoreweaveControllerConfig(
                port=port,
                service_name=service_name,
                scale_group=controller_scale_group,
            ),
        ),
        storage=StorageConfig(
            remote_state_dir=remote_state_dir,
        ),
        kubernetes_provider=KubernetesProviderConfig(),
        # The controller's scale group so start_controller can validate it.
        scale_groups={
            controller_scale_group: ScaleGroupConfig(
                buffer_slices=0,
                max_slices=10,
                slice_template=SliceConfig(
                    name_prefix=controller_scale_group,
                    num_vms=1,
                    coreweave=CoreweaveSliceConfig(instance_type="cd-gp-i64-erapids"),
                ),
            )
        },
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
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config") is not None
    assert k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is not None
    assert k8s.get_json(K8sResource.PERSISTENT_VOLUME_CLAIMS, _CONTROLLER_STATE_PVC_NAME) is not None
    assert k8s.get_json(K8sResource.SERVICES, "iris-controller-svc") is not None

    # S3 storage auth lives in the iris-task-env Secret, not the ConfigMap.
    secret = k8s.get_json(K8sResource.SECRETS, "iris-task-env")
    assert secret is not None
    assert "AWS_ACCESS_KEY_ID" in secret["data"]
    assert "AWS_SECRET_ACCESS_KEY" in secret["data"]

    # Verify Deployment nodeSelector targets the configured scale group
    iris_labels = Labels("iris")
    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    deploy_spec = dep["spec"]
    node_selector = deploy_spec["template"]["spec"]["nodeSelector"]
    assert node_selector == {iris_labels.iris_scale_group: "cpu-erapids"}

    # Controller is stamped with the control-plane PriorityClass, and that class
    # is provisioned so the reference resolves at admission.
    assert deploy_spec["template"]["spec"]["priorityClassName"] == "iris-system"
    assert k8s.get_json(K8sResource.PRIORITY_CLASSES, "iris-system") is not None

    # Controller consumes that env via envFrom (S3 + injected, one flow).
    container = deploy_spec["template"]["spec"]["containers"][0]
    assert container["envFrom"] == [{"secretRef": {"name": "iris-task-env", "optional": True}}]
    # CPU is requested but unlimited (Burstable QoS, not Guaranteed) so the
    # controller can burst onto spare node cores during reconcile spikes; memory
    # is capped to protect the node (issue #6944).
    assert container["resources"]["requests"] == {"cpu": _CONTROLLER_CPU_REQUEST, "memory": _CONTROLLER_MEMORY_REQUEST}
    assert container["resources"]["limits"] == {"memory": _CONTROLLER_MEMORY_REQUEST}
    # Recreate (not RollingUpdate): the old controller pod must be gone before
    # the new one mounts the ReadWriteOnce SQLite state PVC.
    assert deploy_spec["strategy"] == {"type": "Recreate"}
    assert {"name": "local-state", "mountPath": "/var/cache/iris/controller"} in container["volumeMounts"]
    assert {
        "name": "local-state",
        "persistentVolumeClaim": {"claimName": _CONTROLLER_STATE_PVC_NAME},
    } in deploy_spec[
        "template"
    ]["spec"]["volumes"]

    pvc = k8s.get_json(K8sResource.PERSISTENT_VOLUME_CLAIMS, _CONTROLLER_STATE_PVC_NAME)
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert pvc["spec"]["resources"]["requests"]["storage"] == _CONTROLLER_STATE_PVC_SIZE

    t.join(timeout=5)
    provider.shutdown()


def test_start_controller_local_state_dir_uses_hostpath_not_pvc():
    """Setting storage.local_state_dir mounts it via hostPath and creates no PVC."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.storage.local_state_dir = "/mnt/local/iris-controller-state"

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    provider.start_controller(cluster_config)

    assert k8s.get_json(K8sResource.PERSISTENT_VOLUME_CLAIMS, _CONTROLLER_STATE_PVC_NAME) is None
    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    spec = dep["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert {"name": "local-state", "mountPath": "/mnt/local/iris-controller-state"} in container["volumeMounts"]
    assert {
        "name": "local-state",
        "hostPath": {"path": "/mnt/local/iris-controller-state", "type": "DirectoryOrCreate"},
    } in spec["volumes"]

    t.join(timeout=5)
    provider.shutdown()


def test_start_controller_injects_operator_env(monkeypatch):
    """inject_env writes the iris-task-env Secret and wires the controller envFrom."""
    monkeypatch.setenv("WANDB_API_KEY", "wb-secret")
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.defaults.inject_env.append("WANDB_API_KEY")

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()
    provider.start_controller(cluster_config)

    secret = k8s.get_json(K8sResource.SECRETS, "iris-task-env")
    assert secret is not None
    # Values are base64-encoded in the Secret, never in the ConfigMap.
    assert base64.b64decode(secret["data"]["WANDB_API_KEY"]).decode() == "wb-secret"
    configmap = k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config")
    assert "wb-secret" not in json.dumps(configmap)

    container = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")["spec"]["template"]["spec"]["containers"][0]
    assert container["envFrom"] == [{"secretRef": {"name": "iris-task-env", "optional": True}}]

    t.join(timeout=5)
    provider.shutdown()


def test_start_controller_s3_storage_creates_task_env_secret():
    """S3 storage alone (no inject_env) still populates the iris-task-env Secret."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()
    provider.start_controller(cluster_config)

    secret = k8s.get_json(K8sResource.SECRETS, "iris-task-env")
    assert secret is not None
    assert "AWS_ACCESS_KEY_ID" in secret["data"]
    container = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")["spec"]["template"]["spec"]["containers"][0]
    assert container["envFrom"] == [{"secretRef": {"name": "iris-task-env", "optional": True}}]

    t.join(timeout=5)
    provider.shutdown()


def _s3_cluster_config(endpoint: str, external: str = "") -> IrisClusterConfig:
    return IrisClusterConfig(
        platform=PlatformConfig(
            coreweave=CoreweavePlatformConfig(
                object_storage_endpoint=endpoint,
                external_object_storage_endpoint=external,
            )
        )
    )


@pytest.mark.parametrize(
    "endpoint, external, expected",
    [
        # The operator's machine cannot resolve LOTA's private address, so the public
        # endpoint wins whenever the config supplies one.
        ("http://cwlota.com", "https://cwobject.com", "https://cwobject.com"),
        # An endpoint reachable from both sides (R2) needs no second address.
        ("https://acct.r2.cloudflarestorage.com", "", "https://acct.r2.cloudflarestorage.com"),
        # Either field alone is enough to configure the operator's client.
        ("", "https://cwobject.com", "https://cwobject.com"),
    ],
)
def test_operator_client_signs_against_the_externally_reachable_endpoint(endpoint, external, expected, monkeypatch):
    for var in ("AWS_ENDPOINT_URL", "AWS_REGION", "AWS_DEFAULT_REGION", "FSSPEC_S3"):
        monkeypatch.delenv(var, raising=False)

    configure_client_s3(_s3_cluster_config(endpoint, external))

    assert os.environ["AWS_ENDPOINT_URL"] == expected
    assert json.loads(os.environ["FSSPEC_S3"])["endpoint_url"] == expected


def test_operator_client_leaves_a_caller_supplied_endpoint_alone(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.delenv("FSSPEC_S3", raising=False)

    configure_client_s3(_s3_cluster_config("http://cwlota.com", "https://cwobject.com"))

    assert os.environ["AWS_ENDPOINT_URL"] == "http://localhost:9000"


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


def _keep_deployment_ready(k8s: InMemoryK8sService, name: str, stop: threading.Event):
    """Continuously mark the deployment available until stopped.

    Unlike _auto_ready_deployment (one-shot), this keeps re-marking the
    deployment available across re-applies, since apply_json replaces the
    manifest and drops its status.
    """
    while not stop.is_set():
        dep = k8s.get_json(K8sResource.DEPLOYMENTS, name)
        if dep is not None:
            dep.setdefault("status", {})["availableReplicas"] = dep.get("spec", {}).get("replicas", 1)
        time.sleep(0.02)


def _controller_command(k8s: InMemoryK8sService) -> list[str]:
    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    return dep["spec"]["template"]["spec"]["containers"][0]["command"]


def test_fresh_start_strips_fresh_from_persisted_deployment():
    """A --fresh start must not leave --fresh baked into the persisted pod command.

    Otherwise an involuntary pod respawn comes back --fresh and wipes live job
    state (issue #6808).
    """
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    stop = threading.Event()
    t = threading.Thread(target=_keep_deployment_ready, args=(k8s, "iris-controller", stop), daemon=True)
    t.start()
    try:
        provider.start_controller(cluster_config, fresh=True)
    finally:
        stop.set()
        t.join(timeout=5)

    assert "--fresh" not in _controller_command(k8s)
    provider.shutdown()


def test_non_fresh_start_never_includes_fresh():
    """A normal (non-fresh) start deploys a pod command without --fresh."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    provider.start_controller(cluster_config)

    assert "--fresh" not in _controller_command(k8s)
    t.join(timeout=5)
    provider.shutdown()


def test_start_controller_stops_old_controller_before_reapply():
    """start_controller tears the old Deployment down before applying the new one.

    The controller SQLite state lives on a ReadWriteOnce PVC, so two controller
    pods must never run at once. start_controller must delete the existing
    Deployment (and wait for it to disappear) before re-applying, rather than
    letting a rolling update briefly run two pods.
    """
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()

    events: list[tuple[str, str]] = []
    real_delete = k8s.delete
    real_apply = k8s.apply_json

    def recording_delete(resource, name, **kwargs):
        if resource is K8sResource.DEPLOYMENTS and name == "iris-controller":
            events.append(("delete", name))
        return real_delete(resource, name, **kwargs)

    def recording_apply(manifest):
        if manifest.get("kind") == "Deployment" and manifest["metadata"]["name"] == "iris-controller":
            events.append(("apply", "iris-controller"))
        return real_apply(manifest)

    k8s.delete = recording_delete
    k8s.apply_json = recording_apply

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    provider.start_controller(cluster_config)

    assert events == [("delete", "iris-controller"), ("apply", "iris-controller")]

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
    _apply_stub(k8s, "Secret", "iris-task-env")
    _apply_stub(k8s, "PersistentVolumeClaim", _CONTROLLER_STATE_PVC_NAME)

    provider.stop_controller(cluster_config)

    assert k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is None
    assert k8s.get_json(K8sResource.SERVICES, "iris-controller-svc") is None
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config") is None
    assert k8s.get_json(K8sResource.SECRETS, "iris-task-env") is None
    assert k8s.get_json(K8sResource.PERSISTENT_VOLUME_CLAIMS, _CONTROLLER_STATE_PVC_NAME) is None
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
    controller_config = ControllerVmConfig(
        coreweave=CoreweaveControllerConfig(port=9999, service_name="my-svc"),
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
    """ConfigMap must not contain kubeconfig_path/kube_context so pods use in-cluster auth."""
    k8s = InMemoryK8sService(namespace="iris")
    cw_config = CoreweavePlatformConfig(
        region="LGA1",
        namespace="iris",
        kubeconfig_path="/home/user/.kube/coreweave-iris",
        kube_context="marin_LGA1",
    )
    provider = K8sControllerProvider(config=cw_config, label_prefix="iris", poll_interval=0.05, kubectl=k8s)

    cluster_config = _make_cluster_config()
    cluster_config.platform.coreweave.kubeconfig_path = "/home/user/.kube/coreweave-iris"
    cluster_config.platform.coreweave.kube_context = "marin_LGA1"

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    provider.start_controller(cluster_config)

    cm = k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config")
    cm_data = json.loads(cm["data"]["config.json"])
    cw_config_data = cm_data.get("platform", {}).get("coreweave", {})
    assert "kubeconfig_path" not in cw_config_data
    assert "kube_context" not in cw_config_data

    t.join(timeout=5)
    provider.shutdown()


def test_controller_endpoint_url_in_task_env_secret():
    """When object_storage_endpoint is set, AWS_ENDPOINT_URL lands in the iris-task-env Secret."""
    k8s = InMemoryK8sService(namespace="iris")
    cw_config = CoreweavePlatformConfig(
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

    secret = k8s.get_json(K8sResource.SECRETS, "iris-task-env")
    assert base64.b64decode(secret["data"]["AWS_ENDPOINT_URL"]).decode() == "https://object.lga1.coreweave.com"

    t.join(timeout=5)
    provider.shutdown()


# ============================================================================
# Tests: controller error handling
# ============================================================================


def test_start_controller_errors_without_scale_group():
    """start_controller raises when scale_group is not set."""
    provider, _ = _make_provider()
    config = IrisClusterConfig(
        platform=PlatformConfig(
            label_prefix="iris",
            coreweave=CoreweavePlatformConfig(region="LGA1", namespace="iris"),
        ),
        controller=ControllerVmConfig(
            image="ghcr.io/marin-community/iris-controller:latest",
            coreweave=CoreweaveControllerConfig(port=10000),
        ),
    )
    with pytest.raises(InfraError, match="must set scale_group"):
        provider.start_controller(config)
    provider.shutdown()


def test_start_controller_errors_with_invalid_scale_group():
    """start_controller raises when scale_group references a nonexistent group."""
    provider, _ = _make_provider()
    config = IrisClusterConfig(
        platform=PlatformConfig(
            label_prefix="iris",
            coreweave=CoreweavePlatformConfig(region="LGA1", namespace="iris"),
        ),
        controller=ControllerVmConfig(
            image="ghcr.io/marin-community/iris-controller:latest",
            coreweave=CoreweaveControllerConfig(port=10000, scale_group="nonexistent"),
        ),
    )
    with pytest.raises(InfraError, match="not found in scale_groups"):
        provider.start_controller(config)
    provider.shutdown()


def test_start_controller_errors_without_s3_credentials(monkeypatch):
    """start_controller raises when S3 storage is configured but R2 credentials are not set."""
    monkeypatch.delenv("CW_KEY_ID", raising=False)
    monkeypatch.delenv("CW_KEY_SECRET", raising=False)
    provider, _ = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://my-bucket/bundles")

    with pytest.raises(InfraError, match="CW_KEY_ID and CW_KEY_SECRET"):
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
    monkeypatch.delenv("CW_KEY_ID", raising=False)
    monkeypatch.delenv("CW_KEY_SECRET", raising=False)
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="gs://test-bucket/bundles")

    t = threading.Thread(target=_auto_ready_deployment, args=(k8s, "iris-controller"), daemon=True)
    t.start()

    address = provider.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    # GCS storage with no inject_env: no task-env Secret, no envFrom.
    assert k8s.get_json(K8sResource.SECRETS, "iris-task-env") is None
    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert "envFrom" not in container

    t.join(timeout=5)
    provider.shutdown()


def test_ensure_nodepools_scales_multihost_groups_by_num_vms():
    """NodePool capacity is counted in nodes, so multihost groups scale by num_vms per slice."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    cluster_config.scale_groups["h100-16x"] = ScaleGroupConfig(
        buffer_slices=0,
        max_slices=1,
        slice_template=SliceConfig(
            name_prefix="h100-16x",
            num_vms=2,
            coreweave=CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128"),
        ),
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
    cluster_config.scale_groups["h100-16x"] = ScaleGroupConfig(
        buffer_slices=0,
        max_slices=1,
        slice_template=SliceConfig(
            name_prefix="h100-16x",
            num_vms=2,
            coreweave=CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128"),
        ),
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
# Tests: ensure_kueue_queues
# ============================================================================


def test_ensure_kueue_queues_reconciles_local_queue():
    """A configured cluster_queue creates the derived LocalQueue ({label_prefix}-lq)
    in the cluster namespace, bound to the admin-provisioned ClusterQueue."""
    provider, k8s = _make_provider(namespace="iris", label_prefix="iris")
    cluster_config = _make_cluster_config()
    cluster_config.kubernetes_provider.kueue.cluster_queue = "iris-cq"

    provider.ensure_kueue_queues(cluster_config)

    lq = k8s.get_json(K8sResource.LOCAL_QUEUES, "iris-lq")
    assert lq is not None
    assert lq["metadata"]["namespace"] == "iris"
    assert lq["spec"]["clusterQueue"] == "iris-cq"
    provider.shutdown()


def test_ensure_kueue_queues_noop_without_cluster_queue():
    """No cluster_queue configured -> Kueue not in use -> nothing applied."""
    provider, k8s = _make_provider(label_prefix="iris")
    provider.ensure_kueue_queues(_make_cluster_config())
    assert k8s.get_json(K8sResource.LOCAL_QUEUES, "iris-lq") is None
    provider.shutdown()


def test_iris_priority_class_manifest_rejects_unknown_band():
    """The Kueue installer pins its manager via this helper; an unknown band must
    raise rather than silently return a wrong-priority (or empty) manifest."""
    with pytest.raises(ValueError, match="unknown Iris priority class"):
        iris_priority_class_manifest("iris-nonexistent")


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
