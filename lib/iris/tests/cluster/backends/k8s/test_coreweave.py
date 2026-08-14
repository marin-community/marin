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

import pytest
from iris.cluster.config import (
    AuthConfig,
    ControllerVmConfig,
    CoreweaveControllerConfig,
    CoreweavePlatformConfig,
    CoreweaveSliceConfig,
    EndpointSpec,
    IrisClusterConfig,
    KubernetesProviderConfig,
    KueueConfig,
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
    PrerequisitesNotProvisionedError,
    configure_client_s3,
)
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.kueue_manifests import RESOURCE_FLAVOR_NAME, build_cluster_queue
from iris.cluster.platforms.k8s.nodepool_manifests import (
    CPU_TOPOLOGY_NODE_LABELS,
    compute_target_racks,
    nodepool_manifest,
    nodepool_name,
    nodepool_node_labels,
)
from iris.cluster.platforms.k8s.rbac_manifests import (
    cluster_role_binding_manifest,
    cluster_role_manifest,
    cluster_role_name,
    namespace_manifest,
    service_account_manifest,
)
from iris.cluster.platforms.k8s.types import (
    K8sResource,
    iris_priority_class_manifest,
)
from iris.cluster.platforms.types import (
    InfraError,
    Labels,
)
from iris.cluster.types import AcceleratorType
from rigging.timing import Duration


class _ControllerDeploymentK8sService(InMemoryK8sService):
    """Kubernetes boundary fake with a deterministic controller admission result."""

    def __init__(self, *, namespace: str = "iris", controller_pod: dict | None = None, logs: str = "") -> None:
        super().__init__(namespace=namespace)
        self.controller_pod = controller_pod
        self.controller_logs = logs

    def apply_json(self, manifest: dict) -> None:
        super().apply_json(manifest)
        if manifest.get("kind") != "Deployment" or manifest["metadata"]["name"] != "iris-controller":
            return
        if self.controller_pod is not None:
            super().apply_json(self.controller_pod)
            if self.controller_logs:
                self.set_logs(self.controller_pod["metadata"]["name"], self.controller_logs)
            return
        manifest.setdefault("status", {})["availableReplicas"] = manifest.get("spec", {}).get("replicas", 1)


def _controller_failure_pod(
    name: str,
    *,
    reason: str,
    message: str,
    phase: str,
    restart_count: int,
) -> dict:
    return {
        "kind": "Pod",
        "metadata": {"name": name, "namespace": "iris", "labels": {"app": "iris-controller"}},
        "spec": {},
        "status": {
            "phase": phase,
            "podIP": "10.0.0.1" if phase == "Running" else "",
            "conditions": [],
            "containerStatuses": [
                {
                    "name": "iris-controller",
                    "restartCount": restart_count,
                    "state": {"waiting": {"reason": reason, "message": message}},
                }
            ],
        },
    }


def _make_provider(
    region: str = "LGA1",
    namespace: str = "iris",
    label_prefix: str = "iris",
    k8s: InMemoryK8sService | None = None,
) -> tuple[K8sControllerProvider, InMemoryK8sService]:
    k8s = k8s or _ControllerDeploymentK8sService(namespace=namespace)
    config = CoreweavePlatformConfig(
        region=region,
        namespace=namespace,
    )
    provider = K8sControllerProvider(config=config, label_prefix=label_prefix, poll_interval=0.05, kubectl=k8s)
    return provider, k8s


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
        # Kueue is mandatory on the K8s backend.
        kubernetes_provider=KubernetesProviderConfig(kueue=KueueConfig(cluster_queue="iris-cq")),
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


def _seed_prerequisites(k8s: InMemoryK8sService, cluster_config: IrisClusterConfig) -> None:
    """Populate the prerequisites that Pulumi owns in production."""
    namespace = cluster_config.platform.coreweave.namespace or "iris"
    label_prefix = cluster_config.platform.label_prefix
    service_account = "iris-controller"

    k8s.apply_json(namespace_manifest(namespace))
    k8s.apply_json(service_account_manifest(namespace, service_account))
    role_name = cluster_role_name(namespace)
    k8s.apply_json(cluster_role_manifest(role_name))
    k8s.apply_json(cluster_role_binding_manifest(role_name, namespace, service_account))

    for name, sg in cluster_config.scale_groups.items():
        cw = sg.slice_template.coreweave if sg.slice_template is not None else None
        if cw is None or not cw.instance_type:
            continue
        num_vms = max(1, sg.slice_template.num_vms)
        min_nodes = sg.buffer_slices * num_vms
        max_nodes = sg.max_slices * num_vms
        pool_name = nodepool_name(label_prefix, name)
        k8s.apply_json(
            nodepool_manifest(
                pool_name,
                cw.instance_type,
                node_labels=nodepool_node_labels(
                    label_prefix,
                    name,
                    min_nodes=min_nodes,
                    topology_node_labels=(
                        CPU_TOPOLOGY_NODE_LABELS
                        if sg.resources is not None and sg.resources.device_type == AcceleratorType.CPU
                        else ()
                    ),
                ),
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                target_nodes=min_nodes,
                target_racks=compute_target_racks(cw.instance_type, max_nodes, name),
            )
        )

    cluster_queue = cluster_config.kubernetes_provider.kueue.cluster_queue
    if cluster_queue:
        k8s.apply_json(build_cluster_queue(cluster_queue))
    k8s.apply_json(
        {"apiVersion": "kueue.x-k8s.io/v1beta1", "kind": "ResourceFlavor", "metadata": {"name": RESOURCE_FLAVOR_NAME}}
    )


def test_start_controller_creates_controller_resources():
    """start_controller creates the runtime resources that remain Iris-owned."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.endpoints["/system/log-server"] = EndpointSpec(uri="http://finelog:10001")
    _seed_prerequisites(k8s, cluster_config)

    address = provider.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config") is not None
    assert k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is not None
    node_agent = k8s.get_json(K8sResource.DAEMONSETS, "iris-node-agent")
    assert node_agent is not None
    assert k8s.get_json(K8sResource.PERSISTENT_VOLUME_CLAIMS, _CONTROLLER_STATE_PVC_NAME) is not None
    assert k8s.get_json(K8sResource.SERVICES, "iris-controller-svc") is not None

    # S3 storage auth lives in the iris-task-env Secret, not the ConfigMap.
    secret = k8s.get_json(K8sResource.SECRETS, "iris-task-env")
    assert secret is not None
    assert "AWS_ACCESS_KEY_ID" in secret["data"]
    assert "AWS_SECRET_ACCESS_KEY" in secret["data"]

    # Verify the controller targets the configured scale group's amd64 nodes.
    iris_labels = Labels("iris")
    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    deploy_spec = dep["spec"]
    node_selector = deploy_spec["template"]["spec"]["nodeSelector"]
    assert node_selector == {
        "kubernetes.io/arch": "amd64",
        iris_labels.iris_scale_group: "cpu-erapids",
    }

    # Controller is stamped with the control-plane PriorityClass, and that class
    # is provisioned so the reference resolves at admission.
    assert deploy_spec["template"]["spec"]["priorityClassName"] == "iris-system"
    assert k8s.get_json(K8sResource.PRIORITY_CLASSES, "iris-system") is not None
    assert {
        name: k8s.get_json(K8sResource.WORKLOAD_PRIORITY_CLASSES, name)["value"]
        for name in (
            "iris-cpu-batch",
            "iris-accelerator-batch",
            "iris-coscheduled-batch",
            "iris-cpu-interactive",
            "iris-accelerator-interactive",
            "iris-coscheduled-interactive",
            "iris-cpu-production",
            "iris-accelerator-production",
            "iris-coscheduled-production",
        )
    } == {
        "iris-cpu-batch": 0,
        "iris-accelerator-batch": 1,
        "iris-coscheduled-batch": 2,
        "iris-cpu-interactive": 10,
        "iris-accelerator-interactive": 11,
        "iris-coscheduled-interactive": 12,
        "iris-cpu-production": 1000,
        "iris-accelerator-production": 1001,
        "iris-coscheduled-production": 1002,
    }

    agent_spec = node_agent["spec"]["template"]["spec"]
    assert agent_spec["hostNetwork"] is True
    assert agent_spec["containers"][0]["command"] == [
        ".venv/bin/python",
        "-m",
        "iris.cluster.node_agent",
        "k8s",
        "--config=/etc/iris/config.json",
    ]

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

    provider.shutdown()


def test_start_controller_mounts_task_cache_in_node_agent_when_reclaim_enabled():
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.kubernetes_provider.cache_dir = "/mnt/local/iris-cache"
    cluster_config.kubernetes_provider.cache_max_age = Duration.from_hours(24)
    _seed_prerequisites(k8s, cluster_config)

    provider.start_controller(cluster_config)

    node_agent = k8s.get_json(K8sResource.DAEMONSETS, "iris-node-agent")
    agent_spec = node_agent["spec"]["template"]["spec"]
    assert node_agent["spec"]["template"]["metadata"]["annotations"] == {
        "iris.marin.community/cache-max-age-ms": "86400000"
    }
    assert {"name": "task-cache", "mountPath": "/mnt/local/iris-cache"} in agent_spec["containers"][0]["volumeMounts"]
    assert {
        "name": "task-cache",
        "hostPath": {"path": "/mnt/local/iris-cache", "type": "DirectoryOrCreate"},
    } in agent_spec["volumes"]
    provider.shutdown()


def test_start_controller_leaves_node_agent_absent_without_external_finelog():
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)
    provider.start_controller(cluster_config)

    assert k8s.get_json(K8sResource.DAEMONSETS, "iris-node-agent") is None
    provider.shutdown()


def test_start_controller_local_state_dir_uses_hostpath_not_pvc():
    """Setting storage.local_state_dir mounts it via hostPath and creates no PVC."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.storage.local_state_dir = "/mnt/local/iris-controller-state"
    _seed_prerequisites(k8s, cluster_config)

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

    provider.shutdown()


def test_start_controller_injects_operator_env(monkeypatch):
    """inject_env writes the iris-task-env Secret and wires the controller envFrom."""
    monkeypatch.setenv("WANDB_API_KEY", "wb-secret")
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.defaults.inject_env.append("WANDB_API_KEY")
    _seed_prerequisites(k8s, cluster_config)

    provider.start_controller(cluster_config)

    secret = k8s.get_json(K8sResource.SECRETS, "iris-task-env")
    assert secret is not None
    # Values are base64-encoded in the Secret, never in the ConfigMap.
    assert base64.b64decode(secret["data"]["WANDB_API_KEY"]).decode() == "wb-secret"
    configmap = k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config")
    assert "wb-secret" not in json.dumps(configmap)

    container = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")["spec"]["template"]["spec"]["containers"][0]
    assert container["envFrom"] == [{"secretRef": {"name": "iris-task-env", "optional": True}}]

    provider.shutdown()


def test_preflight_controller_caches_signing_key_without_mutating_kubernetes(monkeypatch):
    """The key validated before a rollout is the key projected after the image build."""
    monkeypatch.delenv("IRIS_SIGNING_KEY", raising=False)
    monkeypatch.setenv("TEST_OPERATOR_SIGNING_KEY", "key-before-build")
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    cluster_config.auth = AuthConfig(
        signing_key=["env:IRIS_SIGNING_KEY", "env:TEST_OPERATOR_SIGNING_KEY"],
    )
    _seed_prerequisites(k8s, cluster_config)

    provider.preflight_controller(cluster_config)

    assert k8s.get_json(K8sResource.SECRETS, "iris-controller-env") is None
    monkeypatch.setenv("TEST_OPERATOR_SIGNING_KEY", "key-after-build")
    provider.start_controller(cluster_config)

    secret = k8s.get_json(K8sResource.SECRETS, "iris-controller-env")
    assert base64.b64decode(secret["data"]["IRIS_SIGNING_KEY"]).decode() == "key-before-build"

    provider.shutdown()


def test_start_controller_s3_storage_creates_task_env_secret():
    """S3 storage alone (no inject_env) still populates the iris-task-env Secret."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    _seed_prerequisites(k8s, cluster_config)

    provider.start_controller(cluster_config)

    secret = k8s.get_json(K8sResource.SECRETS, "iris-task-env")
    assert secret is not None
    assert "AWS_ACCESS_KEY_ID" in secret["data"]
    container = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")["spec"]["template"]["spec"]["containers"][0]
    assert container["envFrom"] == [{"secretRef": {"name": "iris-task-env", "optional": True}}]

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
    _seed_prerequisites(k8s, cluster_config)

    address = provider.start_controller(cluster_config)
    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"

    # ConfigMap and Service should all be reconciled
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config") is not None
    assert k8s.get_json(K8sResource.SERVICES, "iris-controller-svc") is not None

    provider.shutdown()


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
    _seed_prerequisites(k8s, cluster_config)

    provider.start_controller(cluster_config, fresh=True)

    assert "--fresh" not in _controller_command(k8s)
    provider.shutdown()


def test_non_fresh_start_never_includes_fresh():
    """A normal (non-fresh) start deploys a pod command without --fresh."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)

    provider.start_controller(cluster_config)

    assert "--fresh" not in _controller_command(k8s)
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
    _seed_prerequisites(k8s, cluster_config)

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

    provider.start_controller(cluster_config)

    assert events == [("delete", "iris-controller"), ("apply", "iris-controller")]

    provider.shutdown()


def test_stop_controller_deletes_resources():
    """stop_controller deletes every Iris-owned runtime resource."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")

    # Pre-populate resources
    _apply_stub(k8s, "Deployment", "iris-controller")
    _apply_stub(k8s, "DaemonSet", "iris-node-agent")
    _apply_stub(k8s, "Service", "iris-controller-svc")
    _apply_stub(k8s, "ConfigMap", "iris-cluster-config")
    _apply_stub(k8s, "Secret", "iris-task-env")
    _apply_stub(k8s, "PersistentVolumeClaim", _CONTROLLER_STATE_PVC_NAME)

    provider.stop_controller(cluster_config)

    assert k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is None
    assert k8s.get_json(K8sResource.DAEMONSETS, "iris-node-agent") is None
    assert k8s.get_json(K8sResource.SERVICES, "iris-controller-svc") is None
    assert k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config") is None
    assert k8s.get_json(K8sResource.SECRETS, "iris-task-env") is None
    assert k8s.get_json(K8sResource.PERSISTENT_VOLUME_CLAIMS, _CONTROLLER_STATE_PVC_NAME) is None
    provider.shutdown()


def test_stop_controller_leaves_iac_provisioned_rbac_alone():
    """stop_controller must not delete the ClusterRole/ClusterRoleBinding.

    These are IaC-provisioned (spec.md §4), not created by start_controller anymore;
    verify_prerequisites hard-fails the next start_controller if they're missing, so
    deleting them on stop would break a plain `iris cluster stop && iris cluster start`.
    """
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)
    role_name = provider.rbac_cluster_role_name()

    provider.stop_controller(cluster_config)

    assert k8s.get_json(K8sResource.CLUSTER_ROLES, role_name) is not None
    assert k8s.get_json(K8sResource.CLUSTER_ROLE_BINDINGS, role_name) is not None
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
# Tests: verify_prerequisites
# ============================================================================


def test_verify_prerequisites_raises_when_nothing_provisioned():
    """A fresh cluster with no IaC-provisioned prerequisites fails fast, enumerating them."""
    provider, _ = _make_provider()
    cluster_config = _make_cluster_config()

    with pytest.raises(PrerequisitesNotProvisionedError) as exc_info:
        provider.verify_prerequisites(cluster_config)

    message = str(exc_info.value)
    assert "Namespace/iris" in message
    assert "ServiceAccount/iris/iris-controller" in message
    assert "ClusterRole/iris-controller-iris" in message
    assert "ClusterRoleBinding/iris-controller-iris" in message
    assert "NodePool/iris-cpu-erapids" in message
    assert "ClusterQueue/iris-cq" in message
    assert "pulumi up" in message
    provider.shutdown()


def test_verify_prerequisites_passes_once_seeded():
    """Once every IaC-provisioned prerequisite exists, verify_prerequisites is a no-op."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)
    pools_before = {p["metadata"]["name"] for p in k8s.list_json(K8sResource.NODE_POOLS)}

    provider.verify_prerequisites(cluster_config)  # must not raise

    # Presence-only: verifying must not create, delete, or otherwise touch anything.
    assert {p["metadata"]["name"] for p in k8s.list_json(K8sResource.NODE_POOLS)} == pools_before
    assert k8s.get_json(K8sResource.NAMESPACES, "iris") is not None
    provider.shutdown()


def test_verify_prerequisites_uses_cluster_queue_resource_flavors():
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)
    k8s.delete(K8sResource.RESOURCE_FLAVORS, RESOURCE_FLAVOR_NAME)
    k8s.apply_json(
        {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "ClusterQueue",
            "metadata": {"name": "iris-cq"},
            "spec": {
                "resourceGroups": [
                    {
                        "flavors": [
                            {"name": "existing-cpu", "resources": []},
                            {"name": "existing-gpu", "resources": []},
                        ]
                    }
                ]
            },
        }
    )
    for flavor in ("existing-cpu", "existing-gpu"):
        k8s.apply_json({"apiVersion": "kueue.x-k8s.io/v1beta1", "kind": "ResourceFlavor", "metadata": {"name": flavor}})

    # Presence validation follows the live ClusterQueue, independent of the
    # flavor name rendered by the currently checked-out IaC code.
    provider.verify_prerequisites(cluster_config)
    provider.shutdown()


def test_verify_prerequisites_reports_missing_cluster_queue_flavor():
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)
    k8s.delete(K8sResource.RESOURCE_FLAVORS, RESOURCE_FLAVOR_NAME)

    with pytest.raises(PrerequisitesNotProvisionedError) as exc_info:
        provider.verify_prerequisites(cluster_config)

    message = str(exc_info.value)
    assert f"ResourceFlavor/{RESOURCE_FLAVOR_NAME}" in message
    assert "ClusterQueue/iris-cq" not in message
    provider.shutdown()


def test_verify_prerequisites_reports_only_the_missing_pieces():
    """The error names exactly what's absent, not everything."""
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)
    k8s.delete(K8sResource.NODE_POOLS, "iris-cpu-erapids")

    with pytest.raises(PrerequisitesNotProvisionedError) as exc_info:
        provider.verify_prerequisites(cluster_config)

    message = str(exc_info.value)
    assert "NodePool/iris-cpu-erapids" in message
    assert "Namespace/iris" not in message
    assert "ClusterRole/iris-controller-iris" not in message
    provider.shutdown()


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
    _seed_prerequisites(k8s, cluster_config)

    provider.start_controller(cluster_config)

    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    # Must reference config.json, not config.yaml
    config_args = [arg for arg in container["command"] if "config" in arg and arg.startswith("--config")]
    assert len(config_args) == 1
    assert config_args[0] == "--config=/etc/iris/config.json"

    provider.shutdown()


def test_configmap_strips_kubeconfig_path():
    """ConfigMap must not contain kubeconfig_path/kube_context so pods use in-cluster auth."""
    k8s = _ControllerDeploymentK8sService(namespace="iris")
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
    _seed_prerequisites(k8s, cluster_config)

    provider.start_controller(cluster_config)

    cm = k8s.get_json(K8sResource.CONFIGMAPS, "iris-cluster-config")
    cm_data = json.loads(cm["data"]["config.json"])
    cw_config_data = cm_data.get("platform", {}).get("coreweave", {})
    assert "kubeconfig_path" not in cw_config_data
    assert "kube_context" not in cw_config_data

    provider.shutdown()


def test_controller_endpoint_url_in_task_env_secret():
    """When object_storage_endpoint is set, AWS_ENDPOINT_URL lands in the iris-task-env Secret."""
    k8s = _ControllerDeploymentK8sService(namespace="iris")
    cw_config = CoreweavePlatformConfig(
        region="LGA1",
        namespace="iris",
        object_storage_endpoint="https://object.lga1.coreweave.com",
    )
    provider = K8sControllerProvider(config=cw_config, label_prefix="iris", poll_interval=0.05, kubectl=k8s)

    cluster_config = _make_cluster_config(remote_state_dir="s3://test-bucket/bundles")
    cluster_config.platform.coreweave.object_storage_endpoint = "https://object.lga1.coreweave.com"
    _seed_prerequisites(k8s, cluster_config)

    provider.start_controller(cluster_config)

    secret = k8s.get_json(K8sResource.SECRETS, "iris-task-env")
    assert base64.b64decode(secret["data"]["AWS_ENDPOINT_URL"]).decode() == "https://object.lga1.coreweave.com"

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
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="s3://my-bucket/bundles")
    _seed_prerequisites(k8s, cluster_config)

    with pytest.raises(InfraError, match="CW_KEY_ID and CW_KEY_SECRET"):
        provider.start_controller(cluster_config)
    provider.shutdown()


def test_start_controller_detects_crash_loop_backoff():
    """start_controller fails fast when the controller Pod enters CrashLoopBackOff."""
    crash_logs = (
        "ValueError: scale_groups.cpu-erapids.resources has unknown keys: memory_bytes\n"
        "Error: Failed to load cluster config\n"
    )
    k8s = _ControllerDeploymentK8sService(
        controller_pod=_controller_failure_pod(
            "iris-controller-abc123",
            reason="CrashLoopBackOff",
            message="back-off 40s restarting failed container",
            phase="Running",
            restart_count=3,
        ),
        logs=crash_logs,
    )
    provider, k8s = _make_provider(k8s=k8s)
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)

    with pytest.raises(InfraError, match="CrashLoopBackOff"):
        provider.start_controller(cluster_config)

    provider.shutdown()


def test_start_controller_detects_image_pull_failure():
    """start_controller fails fast on ImagePullBackOff."""
    k8s = _ControllerDeploymentK8sService(
        controller_pod=_controller_failure_pod(
            "iris-controller-abc123",
            reason="ImagePullBackOff",
            message="failed to pull image: 403 Forbidden",
            phase="Pending",
            restart_count=0,
        )
    )
    provider, k8s = _make_provider(k8s=k8s)
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)

    with pytest.raises(InfraError, match="ImagePullBackOff"):
        provider.start_controller(cluster_config)

    provider.shutdown()


def test_start_controller_crash_loop_includes_logs():
    """When CrashLoopBackOff is detected, the error includes container logs."""
    crash_logs = "ValueError: bad config key\nTraceback ...\n"
    k8s = _ControllerDeploymentK8sService(
        controller_pod=_controller_failure_pod(
            "iris-controller-xyz",
            reason="CrashLoopBackOff",
            message="back-off restarting",
            phase="Running",
            restart_count=5,
        ),
        logs=crash_logs,
    )
    provider, k8s = _make_provider(k8s=k8s)
    cluster_config = _make_cluster_config()
    _seed_prerequisites(k8s, cluster_config)

    with pytest.raises(InfraError, match="bad config key"):
        provider.start_controller(cluster_config)

    provider.shutdown()


def test_start_controller_skips_s3_for_gs_storage(monkeypatch):
    """start_controller succeeds without S3 credentials when using gs:// storage."""
    monkeypatch.delenv("CW_KEY_ID", raising=False)
    monkeypatch.delenv("CW_KEY_SECRET", raising=False)
    provider, k8s = _make_provider()
    cluster_config = _make_cluster_config(remote_state_dir="gs://test-bucket/bundles")
    _seed_prerequisites(k8s, cluster_config)

    address = provider.start_controller(cluster_config)

    assert address == "iris-controller-svc.iris.svc.cluster.local:10000"
    # GCS storage with no inject_env: no task-env Secret, no envFrom.
    assert k8s.get_json(K8sResource.SECRETS, "iris-task-env") is None
    dep = k8s.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert "envFrom" not in container

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


def test_ensure_kueue_queues_raises_without_cluster_queue():
    """Kueue is mandatory on the K8s backend, so a missing cluster_queue is a fail-fast
    misconfiguration rather than a silent skip."""
    provider, _ = _make_provider(label_prefix="iris")
    cfg = _make_cluster_config()
    cfg.kubernetes_provider.kueue.cluster_queue = ""
    with pytest.raises(ValueError, match=r"kueue\.cluster_queue is required"):
        provider.ensure_kueue_queues(cfg)
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
