# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CoreWeave platform implementation.

Implements the Platform protocol for CoreWeave CKS clusters. The only
infrastructure Iris manages is the controller Deployment, Service, and
ConfigMap. Worker pods and node scaling are handled by KubernetesProvider,
which creates pods directly against the K8s API.

All kubectl commands use in-cluster auth by default. If kubeconfig_path is
set in config, --kubeconfig is passed to kubectl.

Controller lifecycle (start_controller / stop_controller) creates and tears
down the ConfigMap, Deployment, and Service via kubectl.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from urllib.parse import urlparse

from iris.cluster.config import config_to_dict
from iris.cluster.k8s.constants import CW_INTERRUPTABLE_TOLERATION
from iris.cluster.k8s.kubectl import Kubectl
from iris.cluster.platform.base import (
    Labels,
    PlatformError,
    QuotaExhaustedError,
    SliceHandle,
    StandaloneWorkerHandle,
    find_free_port,
)
from iris.rpc import config_pb2
from iris.time_utils import Deadline, ExponentialBackoff

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
_CONTROLLER_CPU_REQUEST = "2"
_CONTROLLER_MEMORY_REQUEST = "4Gi"


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


def _classify_kubectl_error(stderr: str) -> PlatformError:
    """Classify a kubectl error into a specific PlatformError subclass."""
    lower = stderr.lower()
    if "quota" in lower or "insufficient" in lower or "capacity" in lower:
        return QuotaExhaustedError(stderr)
    return PlatformError(stderr)


def split_coreweave_topology_selector(worker_attributes: dict[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Split worker attributes into static topology selectors and keys needing discovery.

    Keys with known CoreWeave topology prefixes and a concrete value become static
    nodeSelector entries. Keys with the sentinel value "same-slice" are returned as
    discovered_keys — the provider should read the leader pod's node labels to fill
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
# CoreweavePlatform
# ============================================================================


class CoreweavePlatform:
    """Platform implementation for CoreWeave CKS clusters.

    Manages only the controller Deployment, Service, and ConfigMap.
    Worker pods and node scaling are handled by KubernetesProvider.
    """

    def __init__(
        self,
        config: config_pb2.CoreweavePlatformConfig,
        label_prefix: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ):
        self._config = config
        self._namespace = config.namespace or "iris"
        self._region = config.region
        self._label_prefix = label_prefix
        self._iris_labels = Labels(label_prefix)
        self._kubectl = Kubectl(
            namespace=self._namespace,
            # Let kubectl handle KUBECONFIG natively; only pass --kubeconfig when no env override is present.
            kubeconfig_path=None if os.environ.get("KUBECONFIG") else (config.kubeconfig_path or None),
            timeout=_KUBECTL_TIMEOUT,
        )
        self._poll_interval = poll_interval
        self._shutdown_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="coreweave")
        self._s3_enabled = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        if self._s3_enabled:
            logger.info("S3 credentials detected in environment; enabling S3 env injection")

    # -- RBAC / Namespace Prerequisites ----------------------------------------

    def _rbac_cluster_role_name(self) -> str:
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
        cluster_role_name = self._rbac_cluster_role_name()

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
            min_nodes = sg.min_slices * num_vms
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
            "nodepools",
            labels={self._iris_labels.iris_managed: "true"},
            cluster_scoped=True,
        )
        for item in existing:
            pool_name = item.get("metadata", {}).get("name", "")
            if pool_name and pool_name not in expected_names:
                logger.info("Deleting stale NodePool %s (async)", pool_name)
                self._kubectl.delete("nodepool", pool_name, cluster_scoped=True, wait=False)

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
        existing = self._kubectl.get_json("nodepool", pool_name, cluster_scoped=True)
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

    def _uses_s3_storage(self, config: config_pb2.IrisClusterConfig) -> bool:
        """Check if any storage URI uses S3."""
        return config.storage.remote_state_dir.startswith("s3://")

    # -- S3 Credentials -------------------------------------------------------

    def _ensure_s3_credentials_secret(self) -> None:
        """Create K8s Secret from operator's R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY env vars.

        Called during start_controller(). Pods reference the secret via
        secretKeyRef so boto3/s3fs picks up credentials automatically.
        """
        key_id = os.environ.get("R2_ACCESS_KEY_ID")
        key_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
        if not key_id or not key_secret:
            raise PlatformError(
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

    def _s3_env_vars(self) -> list[dict]:
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

    # -- Platform Protocol: unsupported operations -----------------------------

    def create_vm(self, config: config_pb2.VmConfig) -> StandaloneWorkerHandle:
        raise PlatformError("CoreweavePlatform does not create standalone VMs (use KubernetesProvider)")

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> SliceHandle:
        raise PlatformError("CoreweavePlatform does not manage slices (use KubernetesProvider)")

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[SliceHandle]:
        return []

    def list_all_slices(self) -> list[SliceHandle]:
        return []

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list:
        return []

    # -- Platform Protocol: supported operations -------------------------------

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return image

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        # address is like "iris-controller-svc.iris.svc.cluster.local:10000"
        host, port_str = address.rsplit(":", 1)
        service_name = host.split(".")[0]
        remote_port = int(port_str)
        return _coreweave_tunnel(
            kubectl=self._kubectl,
            service_name=service_name,
            remote_port=remote_port,
            local_port=local_port,
        )

    def shutdown(self) -> None:
        self._shutdown_event.set()

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Return the K8s Service DNS name for the controller.

        Format: {service_name}.{namespace}.svc.cluster.local:{port}
        """
        cw = controller_config.coreweave
        service_name = cw.service_name or "iris-controller-svc"
        port = cw.port or 10000
        return f"{service_name}.{self._namespace}.svc.cluster.local:{port}"

    def _config_json_for_configmap(self, config: config_pb2.IrisClusterConfig) -> str:
        """Serialize cluster config to JSON for the in-cluster ConfigMap.

        Uses config_to_dict() to normalize resource field names (memory_bytes
        -> ram, disk_bytes -> disk) so the controller's load_config() can parse
        them.  Strips kubeconfig_path since pods use in-cluster auth.
        """
        config_dict = config_to_dict(config)
        cw_dict = config_dict.get("platform", {}).get("coreweave", {})
        cw_dict.pop("kubeconfig_path", None)
        return json.dumps(config_dict)

    def start_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        """Start the controller, reconciling all resources. Returns address (host:port).

        Fully idempotent: always applies ConfigMap, Deployment, and Service
        (all no-ops if unchanged). _wait_for_deployment_ready returns
        immediately if the Deployment is already available.
        """
        cw = config.controller.coreweave
        port = cw.port or 10000
        service_name = cw.service_name or "iris-controller-svc"
        if not cw.scale_group:
            raise PlatformError("CoreWeave controller config must set scale_group")
        if cw.scale_group not in config.scale_groups:
            raise PlatformError(
                f"Controller scale_group {cw.scale_group!r} not found in scale_groups: "
                f"{list(config.scale_groups.keys())}"
            )

        # Ensure namespace, ServiceAccount, ClusterRole, ClusterRoleBinding exist
        self.ensure_rbac()

        # Create S3 credentials secret if S3 storage is configured
        self._s3_enabled = self._uses_s3_storage(config)
        if self._s3_enabled:
            self._ensure_s3_credentials_secret()

        # Create/update ConfigMap with JSON-serialized cluster config
        config_json = self._config_json_for_configmap(config)
        configmap_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "iris-cluster-config", "namespace": self._namespace},
            "data": {"config.json": config_json},
        }
        self._kubectl.apply_json(configmap_manifest)
        logger.info("ConfigMap iris-cluster-config applied")

        # Create all shared NodePools for each scale group
        self.ensure_nodepools(config)

        # Apply controller Deployment (scheduled onto the configured scale group's NodePool)
        s3_env = self._s3_env_vars() if self._s3_enabled else []
        deploy_manifest = _build_controller_deployment(
            namespace=self._namespace,
            image=config.controller.image,
            port=port,
            node_selector={self._iris_labels.iris_scale_group: cw.scale_group},
            s3_env_vars=s3_env,
        )
        self._kubectl.apply_json(deploy_manifest)
        self._kubectl.rollout_restart("deployment", "iris-controller", namespaced=True)
        logger.info("Controller Deployment iris-controller applied (rollout restarted)")

        # Apply controller Service
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

        # Wait for Deployment to be available (returns immediately if already up)
        self._wait_for_deployment_ready()

        # Wait for the rollout to fully complete (all old pods terminated).
        # Without this, a port-forward through the Service can land on a
        # dying pod from the previous ReplicaSet and get connection-refused.
        self._kubectl.rollout_status("deployment", "iris-controller", namespaced=True)

        return self.discover_controller(config.controller)

    def restart_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        """Restart controller by re-applying manifests and rolling restart."""
        return self.start_controller(config)

    def stop_controller(self, config: config_pb2.IrisClusterConfig) -> None:
        """Stop the controller and clean up its RBAC resources."""
        cw = config.controller.coreweave
        service_name = cw.service_name or "iris-controller-svc"

        self._kubectl.delete("deployment", "iris-controller")
        self._kubectl.delete("service", service_name)
        self._kubectl.delete("configmap", "iris-cluster-config")
        if self._uses_s3_storage(config):
            self._kubectl.delete("secret", _S3_SECRET_NAME)

        # Clean up cluster-scoped RBAC resources created by ensure_rbac().
        cluster_role_name = self._rbac_cluster_role_name()
        self._kubectl.delete("clusterrolebinding", cluster_role_name, cluster_scoped=True)
        self._kubectl.delete("clusterrole", cluster_role_name, cluster_scoped=True)
        logger.info("Controller resources deleted (including RBAC %s)", cluster_role_name)

    def stop_all(
        self,
        config: config_pb2.IrisClusterConfig,
        dry_run: bool = False,
        label_prefix: str | None = None,
    ) -> list[str]:
        """Stop all CoreWeave cluster resources (controller only -- no worker slices)."""
        target_names = ["controller"]
        if dry_run:
            return target_names
        self.stop_controller(config)
        return target_names

    def _wait_for_deployment_ready(self) -> None:
        """Poll controller Deployment until availableReplicas >= 1.

        Checks Pod-level events and conditions to detect fatal errors early
        (e.g. missing Secrets, image pull failures) rather than silently
        waiting for the full timeout.
        """
        deadline = Deadline.from_seconds(_DEPLOYMENT_READY_TIMEOUT)
        last_status_log = 0.0
        status_log_interval = 30.0  # log progress every 30s
        prev_pod_state: tuple[str, str] | None = None  # (phase, node)

        while not self._shutdown_event.is_set():
            if deadline.expired():
                raise PlatformError(
                    f"Controller Deployment iris-controller did not become available within {_DEPLOYMENT_READY_TIMEOUT}s"
                )
            data = self._kubectl.get_json("deployment", "iris-controller")
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

                # Check Pods owned by this Deployment for fatal errors
                pods = self._check_controller_pods_health()

                # Log pod phase/node on transitions only -- distinguishes
                # node-provisioning vs image-pull vs readiness-probe time.
                if pods:
                    pod = pods[0]
                    phase = pod.get("status", {}).get("phase", "Unknown")
                    node = pod.get("spec", {}).get("nodeName") or "<none>"
                    pod_state = (phase, node)
                    if pod_state != prev_pod_state:
                        logger.info("Controller pod: phase=%s node=%s", phase, node)
                        prev_pod_state = pod_state

            self._shutdown_event.wait(self._poll_interval)
        raise PlatformError("Platform shutting down while waiting for controller Deployment")

    # Waiting reasons that indicate the container image cannot be pulled.
    _IMAGE_PULL_REASONS = frozenset({"ImagePullBackOff", "ErrImagePull", "InvalidImageName"})

    # Waiting reasons that indicate the container keeps crashing after start.
    _CRASH_LOOP_REASONS = frozenset({"CrashLoopBackOff"})

    # Minimum restarts before treating CrashLoopBackOff as fatal. Gives K8s
    # a couple of attempts in case the first crash was transient.
    _CRASH_LOOP_MIN_RESTARTS = 2

    def debug_report(self) -> None:
        """Log controller pod termination reason and previous container logs."""
        pods = self._kubectl.list_json("pods", labels={"app": "iris-controller"})
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

    def _check_controller_pods_health(self) -> list[dict]:
        """Check controller Pods for fatal conditions and fail fast.

        Detects three categories of unrecoverable failure:
        1. Image pull errors (ErrImagePull, ImagePullBackOff)
        2. Container crash loops (CrashLoopBackOff) -- fetches logs to surface
           the actual error (e.g. bad config, missing dependency)
        3. Volume/secret mount failures (via events)
        """
        pods = self._kubectl.list_json("pods", labels={"app": "iris-controller"})
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
                    raise PlatformError(f"Controller Pod {pod_name} has fatal error: {reason}: {message}")

                if reason == "CreateContainerConfigError":
                    raise PlatformError(f"Controller Pod {pod_name} has fatal error: {reason}: {message}")

                if reason in self._CRASH_LOOP_REASONS and restart_count >= self._CRASH_LOOP_MIN_RESTARTS:
                    log_tail = self._kubectl.logs(pod_name, container=container_name, tail=30, previous=True)
                    raise PlatformError(
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

    # Grace period before treating FailedMount as fatal. Kubelet retries
    # volume mounts for a short window; only bail after this period.
    _MOUNT_FAILURE_GRACE_SECONDS = 90.0

    def _check_controller_pod_events(self) -> None:
        """Query events for controller Pods and fail fast on fatal warnings.

        Uses ``kubectl get events --field-selector`` to find Warning events
        for controller Pods. After a grace period, known-fatal events
        (e.g. FailedMount for a missing Secret) cause an immediate failure
        instead of waiting for the full deployment timeout.
        """
        pods = self._kubectl.list_json("pods", labels={"app": "iris-controller"})
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
                    # Check if enough time has passed for retries to be exhausted
                    first_ts = event.get("firstTimestamp", "")
                    last_ts = event.get("lastTimestamp", "")
                    if first_ts and last_ts:
                        try:
                            first = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                            last = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                            span = (last - first).total_seconds()
                            if span >= self._MOUNT_FAILURE_GRACE_SECONDS:
                                raise PlatformError(
                                    f"Controller Pod {pod_name} has fatal event: "
                                    f"{reason}: {message} (repeated {count}x over {span:.0f}s)"
                                )
                        except (ValueError, AttributeError):
                            pass

                    # Even without timestamps, many repetitions indicate a persistent failure
                    if count >= 10:
                        raise PlatformError(
                            f"Controller Pod {pod_name} has fatal event: " f"{reason}: {message} (repeated {count}x)"
                        )

                if reason in self._FATAL_EVENT_REASONS:
                    logger.warning("Controller Pod %s: %s: %s (count=%d)", pod_name, reason, message, count)


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
                    "tolerations": [CW_INTERRUPTABLE_TOLERATION],
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
# Tunnel
# ============================================================================


@contextmanager
def _coreweave_tunnel(
    kubectl: Kubectl,
    service_name: str,
    remote_port: int,
    local_port: int | None = None,
    timeout: float = 90.0,
) -> Iterator[str]:
    """kubectl port-forward to a K8s Service, yielding the local URL.

    Uses a single deadline with exponential backoff to handle freshly
    provisioned nodes whose konnectivity agent may not be ready when
    the pod first passes its readiness probe.  If the kubectl process
    exits (e.g. konnectivity timeout), it is relaunched automatically.
    """
    if local_port is None:
        local_port = find_free_port(start=10000)

    proc: subprocess.Popen | None = None

    def _stop() -> None:
        nonlocal proc
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc = None

    deadline = Deadline.from_seconds(timeout)
    backoff = ExponentialBackoff(initial=1.0, maximum=5.0, factor=2.0)

    while not deadline.expired():
        # (Re-)launch kubectl port-forward when needed.
        if proc is None:
            proc = kubectl.popen(
                ["port-forward", f"svc/{service_name}", f"{local_port}:{remote_port}"],
                namespaced=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )

        # Process died -- log, back off, and relaunch on next iteration.
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            logger.warning("Port-forward exited (retrying): %s", stderr.strip())
            proc = None
            time.sleep(min(backoff.next_interval(), max(0, deadline.remaining_seconds())))
            continue

        # Try to connect to the forwarded port.
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    else:
        _stop()
        # Capture konnectivity-agent state -- it lives in kube-system and is
        # invisible to normal pod-scoped queries. Without this, diagnosing
        # the tunnel race requires manual kubectl before events TTL (~1h).
        try:
            result = kubectl.run(["get", "pods", "-n", "kube-system", "-o", "wide"], timeout=10)
            if result.returncode == 0:
                logger.warning("kube-system pods at tunnel failure:\n%s", result.stdout.strip())
        except subprocess.TimeoutExpired:
            pass
        raise RuntimeError(f"kubectl port-forward to {service_name}:{remote_port} failed after {timeout}s")

    logger.info("Tunnel ready: 127.0.0.1:%d -> %s:%d", local_port, service_name, remote_port)
    try:
        yield f"http://127.0.0.1:{local_port}"
    finally:
        _stop()
