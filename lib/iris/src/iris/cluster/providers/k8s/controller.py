# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8sControllerProvider: controller lifecycle for Kubernetes (CoreWeave CKS) clusters.

Manages the controller Deployment, Service, ConfigMap, RBAC, NodePools, and
S3 credential Secrets. Worker pods and node scaling are handled by K8sTaskProvider.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import datetime
from urllib.parse import urlparse

from iris.cluster.config import config_to_dict
from iris.cluster.providers.k8s.service import K8sService
from iris.cluster.providers.k8s.types import K8sResource
from iris.cluster.providers.types import InfraError, Labels
from iris.rpc import config_pb2
from rigging.timing import Deadline

logger = logging.getLogger(__name__)

# How often to poll Deployment status during bootstrap (seconds)
_DEFAULT_POLL_INTERVAL = 10.0

# Maximum time to wait for the controller Deployment to become available (seconds).
# Includes time for the autoscaler to provision a node.
_DEPLOYMENT_READY_TIMEOUT = 2400.0

# Default kubectl timeout for CoreWeave operations (seconds).
# CoreWeave bare-metal provisioning/deprovisioning is slow; 60s is not enough.
_KUBECTL_TIMEOUT = 1800.0

_S3_SECRET_NAME = "iris-s3-credentials"
_CONTROLLER_CPU_REQUEST = "4"
_CONTROLLER_MEMORY_REQUEST = "16Gi"


# S3-compatible endpoints that require virtual-hosted-style addressing where the
# bucket name is a subdomain (https://<bucket>.cwobject.com). Path-style
# requests are rejected with PathStyleRequestNotAllowed.
_VIRTUAL_HOST_ONLY_S3_DOMAINS = ("cwobject.com", "cwlota.com")


def _needs_virtual_host_addressing(endpoint_url: str) -> bool:
    hostname = urlparse(endpoint_url).hostname or ""
    return any(hostname == domain or hostname.endswith("." + domain) for domain in _VIRTUAL_HOST_ONLY_S3_DOMAINS)


def configure_client_s3(config: config_pb2.IrisClusterConfig) -> None:
    """Configure S3 env vars for fsspec access on CoreWeave (R2 → AWS mapping).

    Maps R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY to their AWS equivalents and
    sets FSSPEC_S3 with the correct endpoint and addressing style. No-op if the
    config has no CoreWeave object storage endpoint.
    """
    endpoint = config.platform.coreweave.object_storage_endpoint
    if not endpoint:
        return

    r2_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if r2_key and r2_secret:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", r2_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", r2_secret)

    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)

    if "FSSPEC_S3" not in os.environ:
        fsspec_conf: dict = {"endpoint_url": endpoint}
        if _needs_virtual_host_addressing(endpoint):
            fsspec_conf["config_kwargs"] = {"s3": {"addressing_style": "virtual"}}
        if ".r2.cloudflarestorage.com" in endpoint:
            fsspec_conf.setdefault("client_kwargs", {})["region_name"] = "auto"
        os.environ["FSSPEC_S3"] = json.dumps(fsspec_conf)

    # Flush fsspec/s3fs cached instances so they pick up the new config.
    import fsspec.config

    fsspec.config.set_conf_env(fsspec.config.conf)
    try:
        import s3fs

        s3fs.S3FileSystem.clear_instance_cache()
    except ImportError:
        pass


_COREWEAVE_TOPOLOGY_LABEL_PREFIXES = (
    "backend.coreweave.cloud/",
    "ib.coreweave.cloud/",
    "node.coreweave.cloud/",
)
_COREWEAVE_TOPOLOGY_DISCOVERY_VALUE = "same-slice"


def split_coreweave_topology_selector(worker_attributes: dict[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Split worker attributes into static topology selectors and keys needing discovery.

    Keys with known CoreWeave topology prefixes and a concrete value become static
    nodeSelector entries. Keys with the sentinel value "same-slice" are returned as
    discovered_keys -- the provider should read the leader pod's node labels to fill
    these in at runtime.
    """
    static_selector: dict[str, str] = {}
    discovered_keys: list[str] = []
    for key, value in worker_attributes.items():
        if not any(key.startswith(prefix) for prefix in _COREWEAVE_TOPOLOGY_LABEL_PREFIXES):
            continue
        if value == _COREWEAVE_TOPOLOGY_DISCOVERY_VALUE:
            discovered_keys.append(key)
        else:
            static_selector[key] = value
    return static_selector, tuple(discovered_keys)


# ============================================================================
# Controller Deployment manifest builder
# ============================================================================


def _build_controller_deployment(
    *,
    namespace: str,
    image: str,
    port: int,
    node_selector: dict[str, str],
    s3_env_vars: list[dict],
) -> dict:
    """Build the controller Deployment manifest as a dict."""
    # Reserve controller CPU/memory so Kubernetes doesn't classify this Pod
    # as BestEffort. Matching limits keep the controller in Guaranteed QoS.
    controller_resources = {
        "requests": {"cpu": _CONTROLLER_CPU_REQUEST, "memory": _CONTROLLER_MEMORY_REQUEST},
        "limits": {"cpu": _CONTROLLER_CPU_REQUEST, "memory": _CONTROLLER_MEMORY_REQUEST},
    }
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "iris-controller",
            "namespace": namespace,
            "labels": {"app": "iris-controller"},
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "iris-controller"}},
            "template": {
                "metadata": {"labels": {"app": "iris-controller"}},
                "spec": {
                    "serviceAccountName": "iris-controller",
                    "nodeSelector": node_selector,
                    "containers": [
                        {
                            "name": "iris-controller",
                            "image": image,
                            "imagePullPolicy": "Always",
                            "command": [
                                ".venv/bin/python",
                                "-m",
                                "iris.cluster.controller.main",
                                "serve",
                                "--host=0.0.0.0",
                                f"--port={port}",
                                "--config=/etc/iris/config.json",
                            ],
                            "ports": [{"containerPort": port}],
                            "env": s3_env_vars,
                            "securityContext": {"capabilities": {"add": ["SYS_PTRACE"]}},
                            "resources": controller_resources,
                            "volumeMounts": [
                                {"name": "config", "mountPath": "/etc/iris", "readOnly": True},
                                {"name": "local-state", "mountPath": "/var/cache/iris/controller"},
                            ],
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": port},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 10,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": port},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 30,
                            },
                        },
                    ],
                    "volumes": [
                        {"name": "config", "configMap": {"name": "iris-cluster-config"}},
                        {"name": "local-state", "emptyDir": {}},
                    ],
                },
            },
        },
    }


# ============================================================================
# K8sControllerProvider
# ============================================================================


class K8sControllerProvider:
    """Controller lifecycle + connectivity for Kubernetes (CoreWeave CKS) clusters.

    Manages RBAC, ConfigMap, Deployment, Service, NodePools, and S3 credential
    Secrets. Implements the ControllerProvider protocol.
    """

    def __init__(
        self,
        config: config_pb2.CoreweavePlatformConfig,
        label_prefix: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        kubectl: K8sService | None = None,
    ):
        self._config = config
        self._namespace = config.namespace or "iris"
        self._region = config.region
        self._label_prefix = label_prefix
        self._iris_labels = Labels(label_prefix)
        if kubectl is not None:
            self._kubectl: K8sService = kubectl
        else:
            from iris.cluster.providers.k8s.service import CloudK8sService

            self._kubectl = CloudK8sService(
                namespace=self._namespace,
                kubeconfig_path=None if os.environ.get("KUBECONFIG") else (config.kubeconfig_path or None),
                timeout=_KUBECTL_TIMEOUT,
            )
        self._poll_interval = poll_interval
        self._shutdown_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="coreweave")
        self._s3_enabled = False

    @property
    def kubectl(self) -> K8sService:
        return self._kubectl

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def iris_labels(self) -> Labels:
        return self._iris_labels

    @property
    def s3_enabled(self) -> bool:
        return self._s3_enabled

    # -- ControllerProvider protocol methods -----------------------------------

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        cw = controller_config.coreweave
        service_name = cw.service_name or "iris-controller-svc"
        port = cw.port or 10000
        return f"{service_name}.{self._namespace}.svc.cluster.local:{port}"

    def start_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        """Start the controller, reconciling all resources. Returns address (host:port).

        Fully idempotent: always applies ConfigMap, Deployment, and Service
        (all no-ops if unchanged). wait_for_deployment_ready returns
        immediately if the Deployment is already available.
        """
        cw = config.controller.coreweave
        port = cw.port or 10000
        service_name = cw.service_name or "iris-controller-svc"
        if not cw.scale_group:
            raise InfraError("CoreWeave controller config must set scale_group")
        if cw.scale_group not in config.scale_groups:
            raise InfraError(
                f"Controller scale_group {cw.scale_group!r} not found in scale_groups: "
                f"{list(config.scale_groups.keys())}"
            )

        self.ensure_rbac()

        self._s3_enabled = self.uses_s3_storage(config)
        if self._s3_enabled:
            self.ensure_s3_credentials_secret()

        config_json = self._config_json_for_configmap(config)
        configmap_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "iris-cluster-config", "namespace": self._namespace},
            "data": {"config.json": config_json},
        }
        self._kubectl.apply_json(configmap_manifest)
        logger.info("ConfigMap iris-cluster-config applied")

        self.ensure_nodepools(config)

        s3_env = self.s3_env_vars() if self._s3_enabled else []
        deploy_manifest = _build_controller_deployment(
            namespace=self._namespace,
            image=config.controller.image,
            port=port,
            node_selector={self._iris_labels.iris_scale_group: cw.scale_group},
            s3_env_vars=s3_env,
        )
        self._kubectl.apply_json(deploy_manifest)
        self._kubectl.rollout_restart(K8sResource.DEPLOYMENTS, "iris-controller")
        logger.info("Controller Deployment iris-controller applied (rollout restarted)")

        svc_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": service_name, "namespace": self._namespace},
            "spec": {
                "selector": {"app": "iris-controller"},
                "ports": [{"port": port, "targetPort": port}],
                "type": "ClusterIP",
            },
        }
        self._kubectl.apply_json(svc_manifest)
        logger.info("Controller Service %s applied", service_name)

        pdb_manifest = {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {"name": "iris-controller-pdb", "namespace": self._namespace},
            "spec": {
                "minAvailable": 1,
                "selector": {"matchLabels": {"app": "iris-controller"}},
            },
        }
        self._kubectl.apply_json(pdb_manifest)
        logger.info("PodDisruptionBudget iris-controller-pdb applied")

        self.wait_for_deployment_ready()
        self._kubectl.rollout_status(K8sResource.DEPLOYMENTS, "iris-controller")

        return self.discover_controller(config.controller)

    def restart_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        return self.start_controller(config)

    def stop_controller(self, config: config_pb2.IrisClusterConfig) -> None:
        cw = config.controller.coreweave
        service_name = cw.service_name or "iris-controller-svc"

        self._kubectl.delete(K8sResource.DEPLOYMENTS, "iris-controller")
        self._kubectl.delete(K8sResource.SERVICES, service_name)
        self._kubectl.delete(K8sResource.PDBS, "iris-controller-pdb")
        self._kubectl.delete(K8sResource.CONFIGMAPS, "iris-cluster-config")
        if self.uses_s3_storage(config):
            self._kubectl.delete(K8sResource.SECRETS, _S3_SECRET_NAME)

        cluster_role_name = self.rbac_cluster_role_name()
        self._kubectl.delete(K8sResource.CLUSTER_ROLE_BINDINGS, cluster_role_name)
        self._kubectl.delete(K8sResource.CLUSTER_ROLES, cluster_role_name)
        logger.info("Controller resources deleted (including RBAC %s)", cluster_role_name)

    def stop_all(
        self,
        config: config_pb2.IrisClusterConfig,
        dry_run: bool = False,
        label_prefix: str | None = None,
    ) -> list[str]:
        target_names = ["controller"]
        if dry_run:
            return target_names
        self.stop_controller(config)
        return target_names

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        # address is like "iris-controller-svc.iris.svc.cluster.local:10000"
        host, port_str = address.rsplit(":", 1)
        service_name = host.split(".")[0]
        remote_port = int(port_str)
        return self._kubectl.port_forward(
            service_name=service_name,
            remote_port=remote_port,
            local_port=local_port,
        )

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return image

    def debug_report(self) -> None:
        """Log controller pod termination reason and previous container logs."""
        pods = self._kubectl.list_json(K8sResource.PODS, labels={"app": "iris-controller"})
        if not pods:
            logger.warning("Post-mortem: no controller pods found")
            return

        for pod in pods:
            name = pod.get("metadata", {}).get("name", "unknown")
            phase = pod.get("status", {}).get("phase", "Unknown")

            for cs in pod.get("status", {}).get("containerStatuses", []):
                restarts = cs.get("restartCount", 0)
                terminated = cs.get("lastState", {}).get("terminated", {})
                if terminated:
                    logger.warning(
                        "Post-mortem %s: phase=%s reason=%s exitCode=%s restarts=%d",
                        name,
                        phase,
                        terminated.get("reason"),
                        terminated.get("exitCode"),
                        restarts,
                    )
                else:
                    logger.warning("Post-mortem %s: phase=%s restarts=%d", name, phase, restarts)

            prev_logs = self._kubectl.logs(name, tail=50, previous=True)
            if prev_logs:
                logger.warning("Post-mortem %s previous logs:\n%s", name, prev_logs)

    def shutdown(self) -> None:
        self._shutdown_event.set()

    # -- RBAC / Namespace Prerequisites ----------------------------------------

    def rbac_cluster_role_name(self) -> str:
        """Namespace-qualified ClusterRole name to avoid collisions across Iris instances."""
        return f"iris-controller-{self._namespace}"

    def ensure_rbac(self) -> None:
        """Create the namespace, ServiceAccount, ClusterRole, and ClusterRoleBinding.

        Idempotent (kubectl apply). These were previously manual operator
        prerequisites; now they're auto-applied at cluster start so a single
        ``iris cluster start`` is sufficient.

        ClusterRole and ClusterRoleBinding names are qualified with the namespace
        (e.g. ``iris-controller-iris``) so multiple Iris instances on the same
        CKS cluster don't collide on these cluster-scoped resources.
        """
        cluster_role_name = self.rbac_cluster_role_name()

        namespace_manifest = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": self._namespace}}

        sa_manifest = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": "iris-controller", "namespace": self._namespace},
        }

        role_manifest = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": cluster_role_name},
            "rules": [
                {
                    "apiGroups": ["compute.coreweave.com"],
                    "resources": ["nodepools"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["pods", "pods/exec", "pods/log"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["nodes"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["events"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["configmaps"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": ["metrics.k8s.io"],
                    "resources": ["pods"],
                    "verbs": ["get", "list"],
                },
                {
                    "apiGroups": ["policy"],
                    "resources": ["poddisruptionbudgets"],
                    "verbs": ["get", "list", "create", "update", "patch", "delete"],
                },
            ],
        }

        binding_manifest = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": cluster_role_name},
            "subjects": [
                {"kind": "ServiceAccount", "name": "iris-controller", "namespace": self._namespace},
            ],
            "roleRef": {
                "kind": "ClusterRole",
                "name": cluster_role_name,
                "apiGroup": "rbac.authorization.k8s.io",
            },
        }

        for manifest in [namespace_manifest, sa_manifest, role_manifest, binding_manifest]:
            self._kubectl.apply_json(manifest)

        logger.info("RBAC prerequisites applied (namespace=%s, clusterRole=%s)", self._namespace, cluster_role_name)

    # -- NodePool Management ---------------------------------------------------

    def _resource_labels(self, scale_group: str) -> dict[str, str]:
        return {
            self._iris_labels.iris_managed: "true",
            self._iris_labels.iris_scale_group: scale_group,
        }

    def _nodepool_name(self, scale_group: str) -> str:
        # NodePool metadata.name must be a valid RFC 1123 subdomain (lowercase alphanumeric, '-', '.')
        return f"{self._label_prefix}-{scale_group}".replace("_", "-").lower()

    def ensure_nodepools(self, config: config_pb2.IrisClusterConfig) -> None:
        """Create shared NodePools for all scale groups and delete stale ones.

        Idempotent. After creating/verifying expected NodePools, any managed
        NodePool not in the current config is deleted (e.g. renamed or removed
        scale groups).
        """
        expected_names: set[str] = set()
        futures = []
        for name, sg in config.scale_groups.items():
            cw = sg.slice_template.coreweave
            if not cw.instance_type:
                logger.info("Scale group %r has no coreweave.instance_type; skipping NodePool", name)
                continue
            pool_name = self._nodepool_name(name)
            expected_names.add(pool_name)
            num_vms = max(1, sg.slice_template.num_vms)
            min_nodes = sg.buffer_slices * num_vms
            max_nodes = sg.max_slices * num_vms
            futures.append(
                self._executor.submit(
                    self._ensure_one_nodepool,
                    pool_name,
                    cw.instance_type,
                    name,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                    warm_nodes=num_vms,
                )
            )
        for f in futures:
            f.result()

        self._delete_stale_nodepools(expected_names)

    def _delete_stale_nodepools(self, expected_names: set[str]) -> None:
        """Delete managed NodePools that are not in the expected set.

        Uses wait=False because CoreWeave NodePool deletion involves bare-metal
        deprovisioning that can take many minutes. We don't need to block on it.
        """
        existing = self._kubectl.list_json(
            K8sResource.NODE_POOLS,
            labels={self._iris_labels.iris_managed: "true"},
        )
        for item in existing:
            pool_name = item.get("metadata", {}).get("name", "")
            if pool_name and pool_name not in expected_names:
                logger.info("Deleting stale NodePool %s (async)", pool_name)
                self._kubectl.delete(K8sResource.NODE_POOLS, pool_name, wait=False)

    def _ensure_one_nodepool(
        self,
        pool_name: str,
        instance_type: str,
        scale_group_name: str,
        *,
        min_nodes: int,
        max_nodes: int,
        warm_nodes: int,
    ) -> None:
        """Create or reconcile a single NodePool.

        Always applies the manifest so that spec fields (minNodes, maxNodes)
        are reconciled on every cluster start. Existing pools keep one full
        slice worth of nodes warm so a transient pod deletion does not collapse
        multihost desired capacity back to a single node.
        """
        target_nodes = 0
        existing = self._kubectl.get_json(K8sResource.NODE_POOLS, pool_name)
        if existing is not None:
            target_nodes = max(min_nodes, min(max_nodes, warm_nodes))

        manifest = {
            "apiVersion": "compute.coreweave.com/v1alpha1",
            "kind": "NodePool",
            "metadata": {
                "name": pool_name,
                "labels": self._resource_labels(scale_group_name),
            },
            "spec": {
                "computeClass": "default",
                "instanceType": instance_type,
                "autoscaling": True,
                "minNodes": min_nodes,
                "maxNodes": max_nodes,
                "targetNodes": target_nodes,
                "nodeLabels": {
                    self._iris_labels.iris_managed: "true",
                    self._iris_labels.iris_scale_group: scale_group_name,
                    # Pin Konnectivity agents and monitoring pods to always-on nodes
                    # so GPU NodePools can safely scale to zero.
                    **({"cks.coreweave.cloud/system-critical": "true"} if min_nodes > 0 else {}),
                },
            },
        }
        self._kubectl.apply_json(manifest)
        logger.info("NodePool %s applied (instance_type=%s, targetNodes=%d)", pool_name, instance_type, target_nodes)

    # -- Storage Detection ----------------------------------------------------

    def uses_s3_storage(self, config: config_pb2.IrisClusterConfig) -> bool:
        """Check if any storage URI uses S3."""
        return config.storage.remote_state_dir.startswith("s3://")

    # -- S3 Credentials -------------------------------------------------------

    def ensure_s3_credentials_secret(self) -> None:
        """Create K8s Secret from operator's R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY env vars.

        Called during start_controller(). Pods reference the secret via
        secretKeyRef so boto3/s3fs picks up credentials automatically.
        """
        key_id = os.environ.get("R2_ACCESS_KEY_ID")
        key_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
        if not key_id or not key_secret:
            raise InfraError(
                "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables are required "
                "for S3-compatible object storage"
            )
        manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": _S3_SECRET_NAME, "namespace": self._namespace},
            "type": "Opaque",
            "data": {
                "AWS_ACCESS_KEY_ID": base64.b64encode(key_id.encode()).decode(),
                "AWS_SECRET_ACCESS_KEY": base64.b64encode(key_secret.encode()).decode(),
            },
        }
        self._kubectl.apply_json(manifest)

    def s3_env_vars(self) -> list[dict]:
        """K8s env var specs for S3 auth (secretKeyRef + endpoint).

        Used by _build_controller_deployment() so fsspec/s3fs can authenticate
        to S3-compatible object storage.

        Sets FSSPEC_S3 so that all fsspec operations (in zephyr, marin,
        levanter, etc.) automatically use the correct endpoint without
        per-call-site configuration.
        """
        env = [
            {
                "name": "AWS_ACCESS_KEY_ID",
                "valueFrom": {"secretKeyRef": {"name": _S3_SECRET_NAME, "key": "AWS_ACCESS_KEY_ID"}},
            },
            {
                "name": "AWS_SECRET_ACCESS_KEY",
                "valueFrom": {"secretKeyRef": {"name": _S3_SECRET_NAME, "key": "AWS_SECRET_ACCESS_KEY"}},
            },
        ]
        endpoint = self._config.object_storage_endpoint
        if endpoint:
            env.append({"name": "AWS_ENDPOINT_URL", "value": endpoint})
            fsspec_conf: dict = {"endpoint_url": endpoint}
            if _needs_virtual_host_addressing(endpoint):
                fsspec_conf["config_kwargs"] = {"s3": {"addressing_style": "virtual"}}
            env.append({"name": "FSSPEC_S3", "value": json.dumps(fsspec_conf)})
        return env

    # -- Deployment readiness --------------------------------------------------

    def wait_for_deployment_ready(self) -> None:
        """Poll controller Deployment until availableReplicas >= 1.

        Checks Pod-level events and conditions to detect fatal errors early
        (e.g. missing Secrets, image pull failures) rather than silently
        waiting for the full timeout.
        """
        deadline = Deadline.from_seconds(_DEPLOYMENT_READY_TIMEOUT)
        last_status_log = 0.0
        status_log_interval = 30.0
        prev_pod_state: tuple[str, str] | None = None

        while not self._shutdown_event.is_set():
            if deadline.expired():
                raise InfraError(
                    f"Controller Deployment iris-controller did not become available within {_DEPLOYMENT_READY_TIMEOUT}s"
                )
            data = self._kubectl.get_json(K8sResource.DEPLOYMENTS, "iris-controller")
            if data is not None:
                available = data.get("status", {}).get("availableReplicas", 0)
                if available >= 1:
                    logger.info("Controller Deployment iris-controller is available")
                    return

                now = time.monotonic()
                if now - last_status_log >= status_log_interval:
                    last_status_log = now
                    remaining = int(deadline.remaining_seconds())
                    logger.info(
                        "Waiting for controller Deployment (availableReplicas=%d, %ds remaining)",
                        available,
                        remaining,
                    )

                pods = self._check_controller_pods_health()

                if pods:
                    pod = pods[0]
                    phase = pod.get("status", {}).get("phase", "Unknown")
                    node = pod.get("spec", {}).get("nodeName") or "<none>"
                    pod_state = (phase, node)
                    if pod_state != prev_pod_state:
                        logger.info("Controller pod: phase=%s node=%s", phase, node)
                        prev_pod_state = pod_state

            self._shutdown_event.wait(self._poll_interval)
        raise InfraError("K8sControllerProvider shutting down while waiting for controller Deployment")

    # Waiting reasons that indicate the container image cannot be pulled.
    _IMAGE_PULL_REASONS = frozenset({"ImagePullBackOff", "ErrImagePull", "InvalidImageName"})

    # Waiting reasons that indicate the container keeps crashing after start.
    _CRASH_LOOP_REASONS = frozenset({"CrashLoopBackOff"})

    # Minimum restarts before treating CrashLoopBackOff as fatal.
    _CRASH_LOOP_MIN_RESTARTS = 2

    def _check_controller_pods_health(self) -> list[dict]:
        """Check controller Pods for fatal conditions and fail fast.

        Detects three categories of unrecoverable failure:
        1. Image pull errors (ErrImagePull, ImagePullBackOff)
        2. Container crash loops (CrashLoopBackOff)
        3. Volume/secret mount failures (via events)
        """
        pods = self._kubectl.list_json(K8sResource.PODS, labels={"app": "iris-controller"})
        for pod in pods:
            pod_name = pod.get("metadata", {}).get("name", "")
            status = pod.get("status", {})

            for cs in status.get("containerStatuses", []) + status.get("initContainerStatuses", []):
                waiting = cs.get("state", {}).get("waiting", {})
                reason = waiting.get("reason", "")
                message = waiting.get("message", "")
                container_name = cs.get("name", "")
                restart_count = cs.get("restartCount", 0)

                if reason in self._IMAGE_PULL_REASONS:
                    raise InfraError(f"Controller Pod {pod_name} has fatal error: {reason}: {message}")

                if reason == "CreateContainerConfigError":
                    raise InfraError(f"Controller Pod {pod_name} has fatal error: {reason}: {message}")

                if reason in self._CRASH_LOOP_REASONS and restart_count >= self._CRASH_LOOP_MIN_RESTARTS:
                    log_tail = self._kubectl.logs(pod_name, container=container_name, tail=30, previous=True)
                    raise InfraError(
                        f"Controller Pod {pod_name} is in {reason} "
                        f"(restarts={restart_count}).\n"
                        f"Last logs:\n{log_tail}"
                    )

            conditions = status.get("conditions", [])
            for cond in conditions:
                if cond.get("type") == "ContainersReady" and cond.get("status") == "False":
                    cond_reason = cond.get("reason", "")
                    cond_message = cond.get("message", "")
                    if cond_reason:
                        logger.info("Controller Pod %s not ready: %s: %s", pod_name, cond_reason, cond_message)

        self._check_controller_pod_events()
        return pods

    # Known-fatal event reasons that will never self-resolve
    _FATAL_EVENT_REASONS = frozenset(
        {
            "FailedMount",
            "FailedAttachVolume",
        }
    )

    # Grace period before treating FailedMount as fatal.
    _MOUNT_FAILURE_GRACE_SECONDS = 90.0

    def _check_controller_pod_events(self) -> None:
        """Query events for controller Pods and fail fast on fatal warnings.

        Uses ``kubectl get events --field-selector`` to find Warning events
        for controller Pods. After a grace period, known-fatal events
        (e.g. FailedMount for a missing Secret) cause an immediate failure
        instead of waiting for the full deployment timeout.
        """
        pods = self._kubectl.list_json(K8sResource.PODS, labels={"app": "iris-controller"})
        for pod in pods:
            pod_name = pod.get("metadata", {}).get("name", "")
            if not pod_name:
                continue

            events = self._kubectl.get_events(
                field_selector=f"involvedObject.name={pod_name},type=Warning",
            )
            for event in events:
                reason = event.get("reason", "")
                message = event.get("message", "")
                count = event.get("count", 1)

                if reason in self._FATAL_EVENT_REASONS and count >= 3:
                    first_ts = event.get("firstTimestamp", "")
                    last_ts = event.get("lastTimestamp", "")
                    if first_ts and last_ts:
                        try:
                            first = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                            last = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                            span = (last - first).total_seconds()
                            if span >= self._MOUNT_FAILURE_GRACE_SECONDS:
                                raise InfraError(
                                    f"Controller Pod {pod_name} has fatal event: "
                                    f"{reason}: {message} (repeated {count}x over {span:.0f}s)"
                                )
                        except (ValueError, AttributeError):
                            pass

                    if count >= 10:
                        raise InfraError(
                            f"Controller Pod {pod_name} has fatal event: " f"{reason}: {message} (repeated {count}x)"
                        )

                if reason in self._FATAL_EVENT_REASONS:
                    logger.warning("Controller Pod %s: %s: %s (count=%d)", pod_name, reason, message, count)

    # -- Internal helpers ------------------------------------------------------

    def _config_json_for_configmap(self, config: config_pb2.IrisClusterConfig) -> str:
        """Serialize cluster config to JSON for the in-cluster ConfigMap."""
        config_dict = config_to_dict(config)
        cw_dict = config_dict.get("platform", {}).get("coreweave", {})
        cw_dict.pop("kubeconfig_path", None)
        return json.dumps(config_dict)
