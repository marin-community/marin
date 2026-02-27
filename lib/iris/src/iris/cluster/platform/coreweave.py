# Copyright 2025 The Marin Authors
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

Implements the Platform protocol for CoreWeave CKS clusters. Compute is
provisioned through pre-defined NodePools (one per scale group) with
CoreWeave autoscaling enabled. Iris manages Pods only; CoreWeave manages
node lifecycle. NodePool names are derived from config:
``{label_prefix}-{scale_group_name}``.

All kubectl commands use in-cluster auth by default. If kubeconfig_path is
set in config, --kubeconfig is passed to kubectl.

Controller lifecycle (start_controller / stop_controller) creates and tears
down the ConfigMap, Deployment, and Service via kubectl. Shared NodePools
(including the controller pool) are created once by ``ensure_nodepools()``.
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
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from urllib.parse import urlparse

from google.protobuf.json_format import MessageToDict

from iris.cluster.config import config_to_dict
from iris.cluster.controller.scaling_group import prepare_slice_config
from iris.cluster.k8s.kubectl import Kubectl
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    CommandResult,
    Labels,
    PlatformError,
    QuotaExhaustedError,
    SliceStatus,
    StandaloneWorkerHandle,
    WorkerStatus,
    find_free_port,
)
from iris.rpc import config_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)

# How often to poll Pod/Deployment status during bootstrap (seconds)
_DEFAULT_POLL_INTERVAL = 10.0

# Maximum time to wait for a worker Pod to become ready (seconds).
# Pod may pend while CoreWeave autoscaler provisions a bare-metal node.
_POD_READY_TIMEOUT = 2400.0

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


def _worker_pod_name(slice_id: str) -> str:
    return f"iris-worker-{slice_id}"


def _worker_config_cm_name(slice_id: str) -> str:
    return f"iris-worker-{slice_id}-wc"


def _classify_kubectl_error(stderr: str) -> PlatformError:
    """Classify a kubectl error into a specific PlatformError subclass."""
    lower = stderr.lower()
    if "quota" in lower or "insufficient" in lower or "capacity" in lower:
        return QuotaExhaustedError(stderr)
    return PlatformError(stderr)


# ============================================================================
# Handle Implementations
# ============================================================================


class CoreweaveWorkerHandle:
    """Handle to a single worker Pod on CoreWeave.

    Implements the RemoteWorkerHandle protocol. The worker is a Kubernetes Pod
    scheduled onto a dedicated NodePool node. There is no SSH -- commands run
    via kubectl exec, and reboot means deleting the Pod (the platform or
    Kubernetes will recreate it).
    """

    def __init__(
        self,
        *,
        pod_name: str,
        internal_address: str,
        kubectl: Kubectl,
    ):
        self._pod_name = pod_name
        self._internal_address = internal_address
        self._kubectl = kubectl

    @property
    def worker_id(self) -> str:
        return self._pod_name

    @property
    def vm_id(self) -> str:
        return self._pod_name

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    @property
    def bootstrap_log(self) -> str:
        return ""

    def status(self) -> WorkerStatus:
        data = self._kubectl.get_json("pod", self._pod_name)
        if data is None:
            return WorkerStatus(state=CloudWorkerState.UNKNOWN)
        phase = data.get("status", {}).get("phase", "")
        state_map = {
            "Running": CloudWorkerState.RUNNING,
            "Succeeded": CloudWorkerState.TERMINATED,
            "Failed": CloudWorkerState.TERMINATED,
            "Pending": CloudWorkerState.STOPPED,
        }
        return WorkerStatus(state=state_map.get(phase, CloudWorkerState.UNKNOWN))

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        timeout_s = int(timeout.seconds) if timeout else int(self._kubectl.timeout)
        exec_args = ["exec", self._pod_name, "--", "bash", "-c", command]

        if on_line is None:
            proc = self._kubectl.popen(
                exec_args,
                namespaced=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            return CommandResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )

        proc = self._kubectl.popen(
            exec_args,
            namespaced=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_lines: list[str] = []
        try:
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                on_line(stripped)
                stdout_lines.append(stripped)
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return CommandResult(
            returncode=proc.returncode,
            stdout="\n".join(stdout_lines),
            stderr=proc.stderr.read() if proc.stderr else "",
        )

    def reboot(self) -> None:
        """Delete the Pod. The platform's monitoring thread or a higher-level
        controller is expected to recreate it if needed."""
        logger.info("Rebooting (deleting) Pod: %s", self._pod_name)
        self._kubectl.delete("pod", self._pod_name, force=True)


class CoreweaveSliceHandle:
    """Handle to a CoreWeave worker Pod on a shared NodePool.

    Tracks lifecycle state internally: CREATING -> BOOTSTRAPPING -> READY | FAILED.
    The monitoring thread (spawned by CoreweavePlatform.create_slice) drives
    transitions. describe() returns the current state.
    """

    def __init__(
        self,
        *,
        slice_id: str,
        region: str,
        scale_group: str,
        labels: dict[str, str],
        kubectl: Kubectl,
        created_at: Timestamp | None = None,
    ):
        self._slice_id = slice_id
        self._region = region
        self._scale_group = scale_group
        self._labels = labels
        self._kubectl = kubectl
        self._created_at = created_at or Timestamp.now()
        self._lock = threading.Lock()
        self._state = CloudSliceState.CREATING
        self._workers: list[CoreweaveWorkerHandle] = []
        self._error_message: str = ""

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return self._region

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def describe(self) -> SliceStatus:
        with self._lock:
            return SliceStatus(
                state=self._state,
                worker_count=len(self._workers),
                workers=list(self._workers),
                error_message=self._error_message,
            )

    def terminate(self) -> None:
        """Delete the worker Pod and its per-worker ConfigMap."""
        pod_name = _worker_pod_name(self._slice_id)
        cm_name = _worker_config_cm_name(self._slice_id)
        logger.info("Deleting worker Pod: %s", pod_name)
        self._kubectl.delete("pod", pod_name, force=True)
        self._kubectl.delete("configmap", cm_name)
        with self._lock:
            self._state = CloudSliceState.DELETING

    def _set_state(
        self,
        state: CloudSliceState,
        workers: list[CoreweaveWorkerHandle] | None = None,
        error_message: str = "",
    ) -> None:
        with self._lock:
            self._state = state
            if workers is not None:
                self._workers = workers
            if error_message:
                self._error_message = error_message


# ============================================================================
# CoreweavePlatform
# ============================================================================


class CoreweavePlatform:
    """Platform implementation for CoreWeave CKS clusters.

    Uses shared NodePools (one per scale group) with CoreWeave autoscaling
    enabled. Iris manages Pods only; CoreWeave manages node provisioning.
    NodePools are created idempotently by ``ensure_nodepools()`` and left
    running across slice lifecycles (they scale to zero when idle).
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

    def ensure_rbac(self) -> None:
        """Create the namespace, ServiceAccount, ClusterRole, and ClusterRoleBinding.

        Idempotent (kubectl apply). These were previously manual operator
        prerequisites; now they're auto-applied at cluster start so a single
        ``iris cluster start`` is sufficient.
        """
        namespace_manifest = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": self._namespace}}

        sa_manifest = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": "iris-controller", "namespace": self._namespace},
        }

        role_manifest = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": "iris-controller"},
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
                    "resources": ["configmaps"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
            ],
        }

        binding_manifest = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": "iris-controller"},
            "subjects": [
                {"kind": "ServiceAccount", "name": "iris-controller", "namespace": self._namespace},
            ],
            "roleRef": {
                "kind": "ClusterRole",
                "name": "iris-controller",
                "apiGroup": "rbac.authorization.k8s.io",
            },
        }

        for manifest in [namespace_manifest, sa_manifest, role_manifest, binding_manifest]:
            self._kubectl.apply_json(manifest)

        logger.info("RBAC prerequisites applied (namespace=%s)", self._namespace)

    # -- Storage Detection ----------------------------------------------------

    def _uses_s3_storage(self, config: config_pb2.IrisClusterConfig) -> bool:
        """Check if any storage URI uses S3."""
        return config.storage.bundle_prefix.startswith("s3://") or config.storage.log_prefix.startswith("s3://")

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

        Used by both _build_controller_deployment() and _create_worker_pod()
        so fsspec/s3fs can authenticate to S3-compatible object storage.

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

    # -- Labels ---------------------------------------------------------------

    def _resource_labels(self, scale_group: str, slice_id: str) -> dict[str, str]:
        return {
            self._iris_labels.iris_managed: "true",
            self._iris_labels.iris_scale_group: scale_group,
            self._iris_labels.iris_slice_id: slice_id,
        }

    # -- NodePool Management ----------------------------------------------------

    def _nodepool_name(self, scale_group: str) -> str:
        # NodePool metadata.name must be a valid RFC 1123 subdomain (lowercase alphanumeric, '-', '.')
        return f"{self._label_prefix}-{scale_group}".replace("_", "-").lower()

    def ensure_nodepools(self, config: config_pb2.IrisClusterConfig) -> None:
        """Create shared NodePools for all scale groups and delete stale ones.

        Idempotent. The controller runs on the NodePool specified by its
        scale_group config field, so no dedicated controller NodePool is needed.

        After creating/verifying expected NodePools, any managed NodePool not in
        the current config is deleted (e.g. renamed or removed scale groups).
        """
        expected_names: set[str] = set()
        futures = []
        for name, sg in config.scale_groups.items():
            cw = sg.slice_template.coreweave
            pool_name = self._nodepool_name(name)
            expected_names.add(pool_name)
            if not sg.HasField("min_slices"):
                raise PlatformError(f"Scale group {name!r} must set min_slices for CoreWeave NodePool")
            if not sg.HasField("max_slices"):
                raise PlatformError(f"Scale group {name!r} must set max_slices for CoreWeave NodePool")
            min_nodes = sg.min_slices
            max_nodes = sg.max_slices
            futures.append(
                self._executor.submit(
                    self._ensure_one_nodepool,
                    pool_name,
                    cw.instance_type,
                    name,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
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
    ) -> None:
        """Create or reconcile a single NodePool.

        Always applies the manifest so that spec fields (minNodes, maxNodes)
        are reconciled on every cluster start. Clamps targetNodes to
        min(currentNodes, 1) to prevent runaway scaling from system pods
        (e.g. konnectivity-agent) while keeping an existing node warm.
        """
        # For existing pools, clamp targetNodes to avoid runaway autoscaling.
        # New pools start at 0.
        target_nodes = 0
        existing = self._kubectl.get_json("nodepool", pool_name, cluster_scoped=True)
        if existing is not None:
            current_nodes = existing.get("status", {}).get("currentNodes", 0)
            target_nodes = min(current_nodes, 1)

        manifest = {
            "apiVersion": "compute.coreweave.com/v1alpha1",
            "kind": "NodePool",
            "metadata": {
                "name": pool_name,
                "labels": {
                    self._iris_labels.iris_managed: "true",
                    self._iris_labels.iris_scale_group: scale_group_name,
                },
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
                },
            },
        }
        self._kubectl.apply_json(manifest)
        logger.info("NodePool %s applied (instance_type=%s, targetNodes=%d)", pool_name, instance_type, target_nodes)

    # -- Platform Protocol Methods --------------------------------------------

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return image

    def create_vm(self, config: config_pb2.VmConfig) -> StandaloneWorkerHandle:
        raise NotImplementedError("CoreWeave does not use standalone VMs. Controller is an operator-managed Deployment.")

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> CoreweaveSliceHandle:
        """Create a worker Pod on a shared NodePool and return a handle in CREATING state.

        A background thread creates the Pod and waits for it to become ready.
        The handle transitions through CREATING -> BOOTSTRAPPING -> READY (or FAILED).
        """
        if config.num_vms > 1:
            raise ValueError(
                f"CoreWeave platform does not support multi-node slices (num_vms={config.num_vms}). "
                "Only num_vms=1 is supported."
            )
        if not worker_config:
            raise ValueError(
                "worker_config is required for CoreWeave slices (need docker_image for worker Pod creation)"
            )

        # prepare_slice_config() sets name_prefix to "{label_prefix}-{sg_name}" but
        # the scale-group label to just sg_name. The nodeSelector must match the
        # NodePool's nodeLabels, which use the bare scale-group name.
        scale_group_name = config.name_prefix
        slice_id = f"{self._label_prefix}-{scale_group_name}-{Timestamp.now().epoch_ms()}"
        labels = self._resource_labels(scale_group_name, slice_id)

        if config.labels:
            labels.update(dict(config.labels))

        # Resolve the actual scale-group value from merged labels (may differ from
        # name_prefix when prepare_slice_config overrides the label).
        resolved_scale_group = labels.get(self._iris_labels.iris_scale_group, scale_group_name)

        handle = CoreweaveSliceHandle(
            slice_id=slice_id,
            region=self._region or config.coreweave.region,
            scale_group=resolved_scale_group,
            labels=labels,
            kubectl=self._kubectl,
        )

        self._executor.submit(self._monitor_slice, handle, config, worker_config)
        return handle

    def _monitor_slice(
        self,
        handle: CoreweaveSliceHandle,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig,
    ) -> None:
        """Background thread: create Pod, wait for Pod ready."""
        try:
            handle._set_state(CloudSliceState.BOOTSTRAPPING)
            self._create_worker_pod(handle, config, worker_config)
            self._wait_for_pod_ready(handle)

            pod_name = _worker_pod_name(handle.slice_id)
            pod_ip = self._get_pod_ip(pod_name)
            worker = CoreweaveWorkerHandle(
                pod_name=pod_name,
                internal_address=pod_ip,
                kubectl=self._kubectl,
            )
            handle._set_state(CloudSliceState.READY, workers=[worker])
            logger.info("Slice %s is READY (pod=%s, ip=%s)", handle.slice_id, pod_name, pod_ip)

        except Exception as e:
            # With CoreWeave-managed autoscaling, quota errors surface as K8s events on the
            # NodePool rather than errors we observe directly. We won't see QuotaExhaustedError
            # here because CoreWeave's autoscaler handles node provisioning. Failures reaching
            # this path are typically Pod scheduling or readiness timeouts.
            logger.error("Slice %s monitoring failed: %s", handle.slice_id, e)
            try:
                pod_name = _worker_pod_name(handle.slice_id)
                cm_name = _worker_config_cm_name(handle.slice_id)
                self._kubectl.delete("pod", pod_name, force=True)
                self._kubectl.delete("configmap", cm_name)
            except Exception as cleanup_err:
                logger.warning(
                    "Cleanup after failure also failed for slice %s: %s",
                    handle.slice_id,
                    cleanup_err,
                )
            handle._set_state(CloudSliceState.FAILED, error_message=str(e))

    def _create_worker_pod(
        self,
        handle: CoreweaveSliceHandle,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig,
    ) -> None:
        """Create the worker Pod on the NodePool's node."""
        cw = config.coreweave
        pod_name = _worker_pod_name(handle.slice_id)
        labels = dict(handle.labels)

        env_vars = [
            {"name": "IRIS_WORKER_NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
            {"name": "IRIS_POD_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
            {"name": "IRIS_POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
            {"name": "IRIS_POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
        ]
        if worker_config.default_task_env:
            for k, v in worker_config.default_task_env.items():
                env_vars.append({"name": k, "value": v})
        if self._s3_enabled:
            env_vars.extend(self._s3_env_vars())

        worker_port = worker_config.port
        if worker_port <= 0:
            raise PlatformError(f"Invalid worker_config.port={worker_port}; must be > 0")

        cache_dir = worker_config.cache_dir
        if not cache_dir:
            raise PlatformError("worker_config.cache_dir must be non-empty")

        runtime = worker_config.runtime
        if not runtime:
            raise PlatformError("worker_config.runtime must be set (docker/kubernetes)")

        worker_image = worker_config.docker_image.strip()
        if not worker_image:
            raise PlatformError("worker_config.docker_image must be non-empty")

        # When using the kubernetes runtime, task containers are separate Pods
        # that claim GPU/RDMA resources directly from the device plugin. Pass
        # the service account name and S3 secret name so the KubernetesRuntime
        # can include them in the task Pod spec.
        if runtime == "kubernetes":
            env_vars.append({"name": "IRIS_SERVICE_ACCOUNT_NAME", "value": "iris-controller"})
            if self._s3_enabled:
                env_vars.append({"name": "IRIS_S3_SECRET_NAME", "value": _S3_SECRET_NAME})

        # Serialize WorkerConfig proto as JSON and store in a per-worker ConfigMap
        # so the worker process can load it via --worker-config.
        wc_cm_name = _worker_config_cm_name(handle.slice_id)
        worker_config_json = json.dumps(
            MessageToDict(worker_config, preserving_proto_field_name=True),
            indent=2,
        )
        wc_cm_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": wc_cm_name,
                "namespace": self._namespace,
                "labels": dict(handle.labels),
            },
            "data": {"worker_config.json": worker_config_json},
        }
        self._kubectl.apply_json(wc_cm_manifest)

        container_spec: dict = {
            "name": "iris-worker",
            "image": worker_image,
            "imagePullPolicy": "Always",
            "command": [
                ".venv/bin/python",
                "-m",
                "iris.cluster.worker.main",
                "serve",
                "--worker-config",
                "/etc/iris/worker_config.json",
            ],
            "ports": [{"containerPort": worker_port}],
            "env": env_vars,
            "volumeMounts": [
                {
                    "name": "worker-config",
                    "mountPath": "/etc/iris/worker_config.json",
                    "subPath": "worker_config.json",
                    "readOnly": True,
                },
                {"name": "cache", "mountPath": cache_dir},
            ],
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": worker_port},
                "initialDelaySeconds": 5,
                "periodSeconds": 5,
            },
        }

        container_spec["securityContext"] = {
            "allowPrivilegeEscalation": False,
            "privileged": False,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"},
        }

        # When runtime is "kubernetes", task Pods claim GPU/RDMA resources
        # directly. The worker Pod must not also request them (the device
        # plugin would double-count and the task Pod would get nothing).
        # CPU/memory requests are always set so the worker gets Burstable QoS
        # and isn't starved by task Pods on the same node.
        resource_limits: dict = {}
        if runtime != "kubernetes":
            if config.gpu_count > 0:
                resource_limits["nvidia.com/gpu"] = str(config.gpu_count)
            if cw.infiniband:
                resource_limits["rdma/ib"] = "1"
        resources: dict = {"requests": {"cpu": "2", "memory": "4Gi"}}
        if resource_limits:
            resources["limits"] = resource_limits
        container_spec["resources"] = resources
        logger.info(
            "CoreWeave pod %s: resource_limits=%s node_selector=%s",
            pod_name,
            resource_limits or "none",
            {self._iris_labels.iris_scale_group: handle.scale_group},
        )

        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": self._namespace,
                "labels": labels,
            },
            "spec": {
                "serviceAccountName": "iris-controller",
                "hostNetwork": True,
                "dnsPolicy": "ClusterFirstWithHostNet",
                "nodeSelector": {
                    self._iris_labels.iris_scale_group: handle.scale_group,
                },
                "containers": [container_spec],
                "volumes": [
                    {"name": "worker-config", "configMap": {"name": wc_cm_name}},
                    {"name": "cache", "hostPath": {"path": cache_dir, "type": "DirectoryOrCreate"}},
                ],
                "restartPolicy": "Always",
            },
        }

        self._kubectl.apply_json(pod_manifest)

    def _wait_for_pod_ready(self, handle: CoreweaveSliceHandle) -> None:
        """Wait for the worker Pod to pass its readiness probe.

        Also inspects container waiting reasons each poll cycle to detect
        fatal errors (ImagePullBackOff, CrashLoopBackOff, etc.) early,
        using the same constants as _check_controller_pods_health().
        """
        pod_name = _worker_pod_name(handle.slice_id)
        deadline = Deadline.from_seconds(_POD_READY_TIMEOUT)
        while not self._shutdown_event.is_set():
            if deadline.expired():
                raise PlatformError(f"Pod {pod_name} did not become ready within {_POD_READY_TIMEOUT}s")
            data = self._kubectl.get_json("pod", pod_name)
            if data is not None:
                conditions = data.get("status", {}).get("conditions", [])
                is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
                if is_ready:
                    logger.info("Pod %s is ready", pod_name)
                    return

                status = data.get("status", {})
                for cs in status.get("containerStatuses", []) + status.get("initContainerStatuses", []):
                    waiting = cs.get("state", {}).get("waiting", {})
                    reason = waiting.get("reason", "")
                    message = waiting.get("message", "")
                    restart_count = cs.get("restartCount", 0)

                    if reason in self._IMAGE_PULL_REASONS:
                        raise PlatformError(f"Pod {pod_name}: {reason}: {message}")

                    if reason == "CreateContainerConfigError":
                        raise PlatformError(f"Pod {pod_name}: {reason}: {message}")

                    if reason in self._CRASH_LOOP_REASONS and restart_count >= self._CRASH_LOOP_MIN_RESTARTS:
                        container_name = cs.get("name", "")
                        log_tail = self._kubectl.logs(pod_name, container=container_name, tail=30, previous=True)
                        raise PlatformError(
                            f"Pod {pod_name}: {reason} (restarts={restart_count}).\n" f"Last logs:\n{log_tail}"
                        )

            self._shutdown_event.wait(self._poll_interval)
        raise PlatformError(f"Platform shutting down while waiting for Pod {pod_name}")

    def _get_pod_ip(self, pod_name: str) -> str:
        """Get the Pod's IP address."""
        data = self._kubectl.get_json("pod", pod_name)
        if data is None:
            raise PlatformError(f"Pod {pod_name} not found")
        ip = data.get("status", {}).get("podIP", "")
        if not ip:
            raise PlatformError(f"Pod {pod_name} has no IP address")
        return ip

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[CoreweaveSliceHandle]:
        """List iris-managed worker Pods, optionally filtered by additional labels."""
        merged = {self._iris_labels.iris_managed: "true"}
        if labels:
            merged.update(labels)
        return self._list_slices_by_labels(merged)

    def list_all_slices(self, labels: dict[str, str] | None = None) -> list[CoreweaveSliceHandle]:
        """List all iris-managed worker Pods, optionally filtered by labels."""
        merged = {self._iris_labels.iris_managed: "true"}
        if labels:
            merged.update(labels)
        return self._list_slices_by_labels(merged)

    def _list_slices_by_labels(self, labels: dict[str, str] | None) -> list[CoreweaveSliceHandle]:
        """Query worker Pods by label selector and return slice handles."""
        items = self._kubectl.list_json("pods", labels=labels)
        handles: list[CoreweaveSliceHandle] = []

        for item in items:
            metadata = item.get("metadata", {})
            item_labels = metadata.get("labels", {})
            pod_name = metadata.get("name", "")

            if not pod_name.startswith("iris-worker-"):
                continue

            slice_id = pod_name.removeprefix("iris-worker-")
            scale_group = item_labels.get(self._iris_labels.iris_scale_group, "")

            creation_ts = metadata.get("creationTimestamp", "")
            created_at = Timestamp.now()
            if creation_ts:
                try:
                    dt = datetime.fromisoformat(creation_ts.replace("Z", "+00:00"))
                    created_at = Timestamp.from_epoch_ms(int(dt.timestamp() * 1000))
                except (ValueError, AttributeError):
                    pass

            handle = CoreweaveSliceHandle(
                slice_id=slice_id,
                region=self._region,
                scale_group=scale_group,
                labels=item_labels,
                kubectl=self._kubectl,
                created_at=created_at,
            )

            pod_phase = item.get("status", {}).get("phase", "")
            conditions = item.get("status", {}).get("conditions", [])
            is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
            pod_ip = item.get("status", {}).get("podIP", "")

            worker = CoreweaveWorkerHandle(pod_name=pod_name, internal_address=pod_ip, kubectl=self._kubectl)
            if is_ready:
                handle._set_state(CloudSliceState.READY, workers=[worker])
            elif pod_phase in ("Pending", "Running"):
                handle._set_state(CloudSliceState.BOOTSTRAPPING, workers=[worker])

            handles.append(handle)

        return handles

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[CoreweaveWorkerHandle]:
        """List worker Pods filtered by labels."""
        merged = {self._iris_labels.iris_managed: "true"}
        if labels:
            merged.update(labels)
        items = self._kubectl.list_json("pods", labels=merged)
        workers: list[CoreweaveWorkerHandle] = []
        for item in items:
            pod_name = item.get("metadata", {}).get("name", "")
            pod_ip = item.get("status", {}).get("podIP", "")
            workers.append(
                CoreweaveWorkerHandle(
                    pod_name=pod_name,
                    internal_address=pod_ip,
                    kubectl=self._kubectl,
                )
            )
        return workers

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
        self._executor.shutdown(wait=True)

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
        → ram, disk_bytes → disk) so the controller's load_config() can parse
        them.  Strips kubeconfig_path since pods use in-cluster auth.
        """
        config_dict = config_to_dict(config)
        cw_dict = config_dict.get("platform", {}).get("coreweave", {})
        cw_dict.pop("kubeconfig_path", None)
        return json.dumps(config_dict)

    def start_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        """Start the controller, reconciling all resources. Returns address (host:port).

        Fully idempotent: always applies ConfigMap, NodePools, Deployment, and
        Service (all no-ops if unchanged). _wait_for_deployment_ready returns
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

        # Create all shared NodePools in parallel
        self.ensure_nodepools(config)

        # Apply controller Deployment (scheduled onto the configured scale group's NodePool)
        s3_env = self._s3_env_vars() if self._s3_enabled else []
        deploy_manifest = _build_controller_deployment(
            namespace=self._namespace,
            image=config.controller.image,
            port=port,
            bundle_prefix=config.storage.bundle_prefix,
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

        return self.discover_controller(config.controller)

    def stop_controller(self, config: config_pb2.IrisClusterConfig) -> None:
        """Stop the controller by deleting its K8s resources."""
        cw = config.controller.coreweave
        service_name = cw.service_name or "iris-controller-svc"

        self._kubectl.delete("deployment", "iris-controller")
        self._kubectl.delete("service", service_name)
        self._kubectl.delete("configmap", "iris-cluster-config")
        if self._uses_s3_storage(config):
            self._kubectl.delete("secret", _S3_SECRET_NAME)
        logger.info("Controller resources deleted")

    def stop_all(
        self,
        config: config_pb2.IrisClusterConfig,
        dry_run: bool = False,
        label_prefix: str | None = None,
    ) -> list[str]:
        """Stop all managed Pods and controller. NodePools are left to scale to zero."""
        prefix = label_prefix or config.platform.label_prefix or "iris"

        pods = self._kubectl.list_json("pods", labels={Labels(prefix).iris_managed: "true"})
        target_names: list[str] = []
        for pod in pods:
            name = pod.get("metadata", {}).get("name", "")
            if name:
                target_names.append(f"pod:{name}")
        target_names.append("controller")

        if dry_run:
            return target_names

        for pod in pods:
            name = pod.get("metadata", {}).get("name", "")
            if name:
                self._kubectl.delete("pod", name, force=True)
                # Clean up per-worker ConfigMap (name derived from pod name convention)
                cm_name = name + "-wc"
                self._kubectl.delete("configmap", cm_name)

        self.stop_controller(config)
        return target_names

    def reload(self, config: config_pb2.IrisClusterConfig) -> str:
        """Reload workers and controller with updated images/config.

        Workers are reloaded first (in parallel) to minimize downtime.
        Then the controller Deployment image is updated and rolled out.
        """
        # Phase 1: Update ConfigMap
        config_json = self._config_json_for_configmap(config)
        configmap_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "iris-cluster-config", "namespace": self._namespace},
            "data": {"config.json": config_json},
        }
        self._kubectl.apply_json(configmap_manifest)
        logger.info("ConfigMap iris-cluster-config updated for reload")

        # Phase 2: Reload worker Pods in parallel
        self._reload_worker_pods(config)

        # Phase 3: Rolling update controller Deployment
        controller_image = config.controller.image
        if controller_image:
            self._kubectl.set_image(
                "deployment",
                "iris-controller",
                "iris-controller",
                controller_image,
                namespaced=True,
            )
            self._kubectl.rollout_status(
                "deployment",
                "iris-controller",
                timeout=_DEPLOYMENT_READY_TIMEOUT,
                namespaced=True,
            )
            logger.info("Controller Deployment updated to image %s", controller_image)

        return self.discover_controller(config.controller)

    def _reload_worker_pods(self, config: config_pb2.IrisClusterConfig) -> None:
        """Delete and recreate all managed worker Pods in parallel with updated images."""
        worker_config = config.defaults.worker
        slices = self.list_all_slices()
        if not slices:
            logger.info("No worker slices to reload")
            return

        def _reload_one(slice_handle: CoreweaveSliceHandle) -> None:
            pod_name = _worker_pod_name(slice_handle.slice_id)
            cm_name = _worker_config_cm_name(slice_handle.slice_id)
            logger.info("Reloading worker Pod %s", pod_name)
            self._kubectl.delete("pod", pod_name, force=True)
            self._kubectl.delete("configmap", cm_name)

            sg_name = slice_handle.scale_group
            sg_config = config.scale_groups.get(sg_name)
            if sg_config is None:
                logger.warning(
                    "Scale group %s not in config, skipping Pod recreation for slice %s",
                    sg_name,
                    slice_handle.slice_id,
                )
                return

            slice_config = prepare_slice_config(sg_config.slice_template, sg_config, self._label_prefix)
            self._create_worker_pod(slice_handle, slice_config, worker_config)
            self._wait_for_pod_ready(slice_handle)
            logger.info("Worker Pod %s reloaded", pod_name)

        futures = [self._executor.submit(_reload_one, s) for s in slices]
        errors: list[Exception] = []
        for future in futures:
            try:
                future.result(timeout=_POD_READY_TIMEOUT + 60)
            except Exception as e:
                errors.append(e)

        if errors:
            raise PlatformError(f"Failed to reload {len(errors)}/{len(slices)} worker Pods: {errors[0]}")

    def _wait_for_deployment_ready(self) -> None:
        """Poll controller Deployment until availableReplicas >= 1.

        Checks Pod-level events and conditions to detect fatal errors early
        (e.g. missing Secrets, image pull failures) rather than silently
        waiting for the full timeout.
        """
        deadline = Deadline.from_seconds(_DEPLOYMENT_READY_TIMEOUT)
        last_status_log = 0.0
        status_log_interval = 30.0  # log progress every 30s

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
                self._check_controller_pods_health()

            self._shutdown_event.wait(self._poll_interval)
        raise PlatformError("Platform shutting down while waiting for controller Deployment")

    # Waiting reasons that indicate the container image cannot be pulled.
    _IMAGE_PULL_REASONS = frozenset({"ImagePullBackOff", "ErrImagePull", "InvalidImageName"})

    # Waiting reasons that indicate the container keeps crashing after start.
    _CRASH_LOOP_REASONS = frozenset({"CrashLoopBackOff"})

    # Minimum restarts before treating CrashLoopBackOff as fatal. Gives K8s
    # a couple of attempts in case the first crash was transient.
    _CRASH_LOOP_MIN_RESTARTS = 2

    def _check_controller_pods_health(self) -> None:
        """Check controller Pods for fatal conditions and fail fast.

        Detects three categories of unrecoverable failure:
        1. Image pull errors (ErrImagePull, ImagePullBackOff)
        2. Container crash loops (CrashLoopBackOff) — fetches logs to surface
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
    bundle_prefix: str,
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
                                f"--bundle-prefix={bundle_prefix}",
                            ],
                            "ports": [{"containerPort": port}],
                            "env": s3_env_vars,
                            "resources": controller_resources,
                            "volumeMounts": [
                                {"name": "config", "mountPath": "/etc/iris", "readOnly": True},
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
    timeout: float = 30.0,
) -> Iterator[str]:
    """kubectl port-forward to a K8s Service, yielding the local URL."""
    if local_port is None:
        local_port = find_free_port(start=10000)

    proc = kubectl.popen(
        ["port-forward", f"svc/{service_name}", f"{local_port}:{remote_port}"],
        namespaced=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        deadline = Deadline.from_seconds(timeout)
        while not deadline.expired():
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            stderr = proc.stderr.read() if proc.stderr else ""
            proc.terminate()
            proc.wait()
            raise RuntimeError(f"kubectl port-forward failed to establish: {stderr}")

        logger.info("Tunnel ready: 127.0.0.1:%d -> %s:%d", local_port, service_name, remote_port)
        yield f"http://127.0.0.1:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
