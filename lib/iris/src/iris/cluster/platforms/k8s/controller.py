# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8sControllerProvider: controller lifecycle for Kubernetes (CoreWeave CKS) clusters.

Manages the controller Deployment, Service, ConfigMap, RBAC, NodePools, and
S3 credential Secrets. Worker pods and node scaling are handled by K8sTaskProvider.
"""

import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from urllib.parse import urlparse

import fsspec.config
import s3fs
from rigging.timing import Deadline

from iris.cluster.config import (
    ControllerVmConfig,
    CoreweaveControllerConfig,
    CoreweavePlatformConfig,
    IrisClusterConfig,
    config_to_dict,
)
from iris.cluster.inject_env import TASK_ENV_SECRET_NAME, collect_inject_env, projects_task_env_secret
from iris.cluster.platforms.k8s.constants import COREWEAVE_INTERRUPTABLE_TOLERATION, NVIDIA_GPU_TOLERATION
from iris.cluster.platforms.k8s.service import CloudK8sService, K8sService
from iris.cluster.platforms.k8s.types import (
    IRIS_PRIORITY_CLASS_SYSTEM,
    IRIS_PRIORITY_CLASSES,
    K8sResource,
    build_priority_class_manifest,
    parse_k8s_timestamp,
)
from iris.cluster.platforms.types import InfraError, Labels, local_queue_name

logger = logging.getLogger(__name__)

# How often to poll Deployment status during bootstrap (seconds)
_DEFAULT_POLL_INTERVAL = 10.0

# Maximum time to wait for the controller Deployment to become available (seconds).
# Includes time for the autoscaler to provision a node.
_DEPLOYMENT_READY_TIMEOUT = 2400.0

# Maximum time to wait for the controller Deployment to fully delete on `--fresh`
# restarts. Should exceed the controller pod's terminationGracePeriodSeconds.
_DEPLOYMENT_DELETE_TIMEOUT = 120.0

# Default kubectl timeout for CoreWeave operations (seconds).
# CoreWeave bare-metal provisioning/deprovisioning is slow; 60s is not enough.
_KUBECTL_TIMEOUT = 1800.0

# The controller tracks every task pod in memory and runs a CPU-heavy reconcile
# loop, so it needs generous headroom: a large fan-out (~1500 tokenize pods) OOMs
# it at less memory, and a CPU-starved event loop misses the /health probe and
# gets liveness-killed mid-reconcile (issue #6944). CPU is requested but left
# without a limit so the Pod is Burstable, not Guaranteed — the controller is
# never CFS-throttled at its guaranteed share and can burst onto spare node
# cores during reconcile spikes. Memory is capped to protect the node.
_CONTROLLER_CPU_REQUEST = "16"
_CONTROLLER_MEMORY_REQUEST = "64Gi"
# Relax the liveness/readiness deadline off the k8s defaults (1s / 3): a
# busy-but-alive controller must not be SIGKILLed just because a /health request
# queued behind a reconcile tick under heavy load.
_CONTROLLER_PROBE_TIMEOUT_SECONDS = 10
_CONTROLLER_PROBE_FAILURE_THRESHOLD = 6
_CONTROLLER_STATE_PVC_NAME = "iris-controller-state"
# Must match main.py's LOCAL_STATE_DIR_DEFAULT — the path the controller
# process falls back to when storage.local_state_dir is unset.
_DEFAULT_STATE_MOUNT_PATH = "/var/cache/iris/controller"
_CONTROLLER_STATE_PVC_SIZE = "50Gi"


# S3-compatible endpoints that require virtual-hosted-style addressing where the
# bucket name is a subdomain (https://<bucket>.cwobject.com). Path-style
# requests are rejected with PathStyleRequestNotAllowed.
_VIRTUAL_HOST_ONLY_S3_DOMAINS = ("cwobject.com", "cwlota.com")


def _needs_virtual_host_addressing(endpoint_url: str) -> bool:
    hostname = urlparse(endpoint_url).hostname or ""
    return any(hostname == domain or hostname.endswith("." + domain) for domain in _VIRTUAL_HOST_ONLY_S3_DOMAINS)


def configure_client_s3(config: IrisClusterConfig) -> None:
    """Configure S3 env vars for fsspec access on CoreWeave (CW_KEY_* → AWS mapping).

    Maps CW_KEY_ID/CW_KEY_SECRET to their AWS equivalents and sets FSSPEC_S3
    with the correct endpoint and addressing style. No-op if the config has no
    CoreWeave object storage endpoint.
    """
    coreweave = config.platform.coreweave
    if coreweave is None or not coreweave.object_storage_endpoint:
        return
    endpoint = coreweave.object_storage_endpoint

    cw_key = os.environ.get("CW_KEY_ID", "")
    cw_secret = os.environ.get("CW_KEY_SECRET", "")
    if cw_key and cw_secret:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", cw_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", cw_secret)

    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)
    os.environ.setdefault("AWS_REGION", "auto")
    os.environ.setdefault("AWS_DEFAULT_REGION", "auto")

    if "FSSPEC_S3" not in os.environ:
        fsspec_conf: dict = {"endpoint_url": endpoint}
        if _needs_virtual_host_addressing(endpoint):
            fsspec_conf["config_kwargs"] = {"s3": {"addressing_style": "virtual"}}
        # Non-AWS S3-compatible endpoints (R2, CoreWeave Object Storage, etc.)
        # don't honor the AWS region scheme; signing with the wrong region
        # surfaces as 400 Bad Request. "auto" tells boto3 to skip region
        # validation and let the endpoint route the request itself.
        fsspec_conf.setdefault("client_kwargs", {})["region_name"] = "auto"
        os.environ["FSSPEC_S3"] = json.dumps(fsspec_conf)

    # Flush fsspec/s3fs cached instances so they pick up the new config.
    fsspec.config.set_conf_env(fsspec.config.conf)
    s3fs.S3FileSystem.clear_instance_cache()


# ============================================================================
# Controller Deployment manifest builder
# ============================================================================


def _build_controller_deployment(
    *,
    namespace: str,
    image: str,
    port: int,
    node_selector: dict[str, str],
    task_env_secret: bool = False,
    fresh: bool = False,
    state_mount_path: str = _DEFAULT_STATE_MOUNT_PATH,
    local_state_hostpath: bool = False,
    checkpoint_interval_seconds: float = 0,
) -> dict:
    """Build the controller Deployment manifest as a dict."""
    # Reserve controller CPU/memory so Kubernetes doesn't classify this Pod as
    # BestEffort. Only memory has a limit, so the Pod is Burstable (not
    # Guaranteed): the controller can burst onto spare node CPU during reconcile
    # spikes instead of being throttled at its request. See the constants above.
    controller_resources = {
        "requests": {"cpu": _CONTROLLER_CPU_REQUEST, "memory": _CONTROLLER_MEMORY_REQUEST},
        "limits": {"memory": _CONTROLLER_MEMORY_REQUEST},
    }
    # The controller SQLite DB lives on a PersistentVolumeClaim (or, with
    # local_state_hostpath, a node-local directory), so two controller pods
    # must never mount the same local state dir at once. We guarantee that by
    # tearing the old Deployment down and waiting for it to fully disappear
    # before applying the new one (see start_controller); the Recreate
    # strategy is belt-and-suspenders for any in-place apply path.
    deploy_spec: dict = {
        "replicas": 1,
        "selector": {"matchLabels": {"app": "iris-controller"}},
        "strategy": {"type": "Recreate"},
        "template": {
            "metadata": {"labels": {"app": "iris-controller"}},
            "spec": {
                "serviceAccountName": "iris-controller",
                # Pin the controller above every user band so a user pod can never
                # preempt it off the shared control node (see IRIS_PRIORITY_CLASSES).
                "priorityClassName": IRIS_PRIORITY_CLASS_SYSTEM,
                "nodeSelector": node_selector,
                # Tolerate the NVIDIA GPU taint (so the controller can run on
                # GPU-only clusters with no CPU NodePool) and CoreWeave's
                # interruptable-capacity taint: freshly provisioned nodes carry
                # qos.coreweave.cloud/interruptable:NoExecute, which would
                # otherwise leave the controller Pending forever. Task pods
                # already tolerate both (see tasks.py). Harmless on untainted nodes.
                "tolerations": [NVIDIA_GPU_TOLERATION, COREWEAVE_INTERRUPTABLE_TOLERATION],
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
                            *(["--fresh"] if fresh else []),
                            *(
                                [f"--checkpoint-interval={checkpoint_interval_seconds}"]
                                if checkpoint_interval_seconds
                                else []
                            ),
                        ],
                        "ports": [{"containerPort": port}],
                        # The cluster default env (S3 storage auth + operator-injected
                        # vars) arrives via the iris-task-env Secret. optional=true so a
                        # controller-only restart that predates the Secret does not crash-loop.
                        **(
                            {"envFrom": [{"secretRef": {"name": TASK_ENV_SECRET_NAME, "optional": True}}]}
                            if task_env_secret
                            else {}
                        ),
                        "securityContext": {"capabilities": {"add": ["SYS_PTRACE"]}},
                        "resources": controller_resources,
                        "volumeMounts": [
                            {"name": "config", "mountPath": "/etc/iris", "readOnly": True},
                            {"name": "local-state", "mountPath": state_mount_path},
                        ],
                        "readinessProbe": {
                            "httpGet": {"path": "/health", "port": port},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 10,
                            "timeoutSeconds": _CONTROLLER_PROBE_TIMEOUT_SECONDS,
                            "failureThreshold": _CONTROLLER_PROBE_FAILURE_THRESHOLD,
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": port},
                            "initialDelaySeconds": 30,
                            "periodSeconds": 30,
                            "timeoutSeconds": _CONTROLLER_PROBE_TIMEOUT_SECONDS,
                            "failureThreshold": _CONTROLLER_PROBE_FAILURE_THRESHOLD,
                        },
                    },
                ],
                "volumes": [
                    {"name": "config", "configMap": {"name": "iris-cluster-config"}},
                    (
                        {"name": "local-state", "hostPath": {"path": state_mount_path, "type": "DirectoryOrCreate"}}
                        if local_state_hostpath
                        else {
                            "name": "local-state",
                            "persistentVolumeClaim": {"claimName": _CONTROLLER_STATE_PVC_NAME},
                        }
                    ),
                ],
            },
        },
    }
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "iris-controller",
            "namespace": namespace,
            "labels": {"app": "iris-controller"},
        },
        "spec": deploy_spec,
    }


def _build_controller_state_pvc(*, namespace: str) -> dict:
    """Build the PVC that stores the controller SQLite state."""
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": _CONTROLLER_STATE_PVC_NAME,
            "namespace": namespace,
            "labels": {"app": "iris-controller"},
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": _CONTROLLER_STATE_PVC_SIZE}},
        },
    }


# Name of the Ingress that publishes only the controller's /proxy path.
_CONTROLLER_PROXY_INGRESS_NAME = "iris-controller-proxy"


def _build_controller_proxy_ingress(
    *,
    namespace: str,
    service_name: str,
    port: int,
    host: str,
    ingress_class: str,
    tls_secret: str,
    cluster_issuer: str,
) -> dict:
    """Build the Ingress that publishes ONLY the controller's ``/proxy`` path.

    The dashboard and RPC surface stay ClusterIP-internal — only ``/proxy`` is
    routed in. CoreWeave has no IAP layer, so the controller's own per-endpoint
    auth (PRIVATE/PUBLIC/BEARER) is the sole gate for that path.

    Ingress-controller-agnostic: no controller-specific annotations. Traefik (the
    default class) streams responses without buffering, so vLLM SSE works out of
    the box; raise long-request timeouts at the Traefik entrypoint if needed.
    When ``cluster_issuer`` is set, cert-manager auto-issues the TLS cert into
    ``tls_secret``.
    """
    metadata: dict = {"name": _CONTROLLER_PROXY_INGRESS_NAME, "namespace": namespace}
    if cluster_issuer:
        metadata["annotations"] = {"cert-manager.io/cluster-issuer": cluster_issuer}
    spec: dict = {
        "ingressClassName": ingress_class,
        "rules": [
            {
                "host": host,
                "http": {
                    "paths": [
                        {
                            "path": "/proxy",
                            "pathType": "Prefix",
                            "backend": {"service": {"name": service_name, "port": {"number": port}}},
                        }
                    ]
                },
            }
        ],
    }
    if tls_secret:
        spec["tls"] = [{"hosts": [host], "secretName": tls_secret}]
    return {"apiVersion": "networking.k8s.io/v1", "kind": "Ingress", "metadata": metadata, "spec": spec}


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
        config: CoreweavePlatformConfig,
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

    def discover_controller(self, controller_config: ControllerVmConfig) -> str:
        cw = controller_config.coreweave
        service_name = cw.service_name or "iris-controller-svc"
        port = cw.port or 10000
        return f"{service_name}.{self._namespace}.svc.cluster.local:{port}"

    def start_controller(self, config: IrisClusterConfig, *, fresh: bool = False) -> str:
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

        # storage.local_state_dir set => mount it from node-local hostPath rather
        # than the default network-attached PVC.
        local_state_hostpath = bool(config.storage.local_state_dir)
        state_mount_path = config.storage.local_state_dir or _DEFAULT_STATE_MOUNT_PATH

        self.ensure_rbac()

        # Build the cluster default env and project it into the controller and
        # every task via the iris-task-env Secret + envFrom. Resolution happens
        # here, in the operator's shell -- the controller never has these secrets.
        # S3 storage auth and operator-injected vars share one flow.
        self._s3_enabled = self.uses_s3_storage(config)
        default_env: dict[str, str] = {}
        if self._s3_enabled:
            default_env.update(self._s3_task_env())
        default_env.update(collect_inject_env(config.defaults.inject_env))
        if default_env:
            self.ensure_task_env_secret(default_env)

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
        self.ensure_kueue_queues(config)
        self.ensure_priority_classes()
        if local_state_hostpath:
            logger.info("controller local state uses node-local hostPath %s (no PVC)", state_mount_path)
        else:
            self._kubectl.apply_json(_build_controller_state_pvc(namespace=self._namespace))
            logger.info("PersistentVolumeClaim %s applied", _CONTROLLER_STATE_PVC_NAME)

        deploy_kwargs = dict(
            namespace=self._namespace,
            image=config.controller.image,
            port=port,
            node_selector={self._iris_labels.iris_scale_group: cw.scale_group},
            task_env_secret=projects_task_env_secret(config),
            state_mount_path=state_mount_path,
            local_state_hostpath=local_state_hostpath,
            checkpoint_interval_seconds=config.controller.checkpoint_interval_seconds,
        )
        deploy_manifest = _build_controller_deployment(**deploy_kwargs, fresh=fresh)
        # Always stop the old controller before starting the new one. The
        # SQLite state PVC is ReadWriteOnce and a rolling update could briefly
        # mount it from two pods on the same node, corrupting the DB. Iris
        # restarts are fast and clients tolerate a short controller outage, so a
        # clean stop/start is simpler and safer than any overlap-avoidance hack.
        # The PVC is intentionally retained across the restart so the new
        # controller reuses the local DB (a --fresh controller wipes its DB
        # directory itself after starting).
        self._delete_controller_deployment_and_wait()
        self._kubectl.apply_json(deploy_manifest)
        logger.info("Controller Deployment iris-controller applied")

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

        # Publish only /proxy off-cluster when a host is configured. The rest of
        # the controller stays ClusterIP-internal; the controller's per-endpoint
        # auth gates /proxy (CoreWeave has no IAP layer). Idempotent apply.
        if cw.public_proxy_host:
            self._apply_proxy_ingress(cw, service_name=service_name, port=port)

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

        if fresh:
            self._persist_deployment_without_fresh(deploy_kwargs)

        return self.discover_controller(config.controller)

    def _apply_proxy_ingress(self, cw: CoreweaveControllerConfig, *, service_name: str, port: int) -> None:
        """Apply the /proxy Ingress and validate its prerequisites.

        The external address is served by the ingress controller's own
        LoadBalancer, not the controller Pod, so this warns (rather than fails)
        when the configured ``IngressClass`` is absent: the Ingress is still
        applied and starts serving once a controller providing that class exists.
        A missing ingress controller must not block the controller itself, which
        is fine on its ClusterIP Service.
        """
        if self._kubectl.get_json(K8sResource.INGRESS_CLASSES, cw.ingress_class) is None:
            logger.warning(
                "IngressClass %r not found — the /proxy Ingress %s will stay pending (no external "
                "address) until an ingress controller providing that class is installed (e.g. "
                "ingress-nginx exposed as a LoadBalancer Service).",
                cw.ingress_class,
                _CONTROLLER_PROXY_INGRESS_NAME,
            )
        ingress_manifest = _build_controller_proxy_ingress(
            namespace=self._namespace,
            service_name=service_name,
            port=port,
            host=cw.public_proxy_host,
            ingress_class=cw.ingress_class,
            tls_secret=cw.tls_secret,
            cluster_issuer=cw.cluster_issuer,
        )
        self._kubectl.apply_json(ingress_manifest)
        logger.info(
            "Controller /proxy Ingress %s applied (host=%s). Point DNS for %s at the ingress "
            "controller's LoadBalancer external IP/hostname (kubectl get ingress %s -n %s -o wide); "
            "TLS terminates in-cluster via cert-manager.",
            _CONTROLLER_PROXY_INGRESS_NAME,
            cw.public_proxy_host,
            cw.public_proxy_host,
            _CONTROLLER_PROXY_INGRESS_NAME,
            self._namespace,
        )

    def _persist_deployment_without_fresh(self, deploy_kwargs: dict) -> None:
        """Re-apply the controller Deployment with ``--fresh`` stripped from the pod command.

        ``--fresh`` baked into the pod ``command`` persists in the template, so any
        involuntary respawn (eviction, node upgrade, OOMKill) comes back ``--fresh``
        and wipes live job state — the failure in issue #6808. The fresh controller
        has already emitted an (empty) checkpoint by now, so re-applying without the
        flag makes ``--fresh`` mean "ignore existing state on this deploy" rather
        than "wipe on every restart forever".
        """
        logger.info("Re-applying controller Deployment without --fresh so the wipe is one-shot")
        self._kubectl.apply_json(_build_controller_deployment(**deploy_kwargs, fresh=False))
        self.wait_for_deployment_ready()
        self._kubectl.rollout_status(K8sResource.DEPLOYMENTS, "iris-controller")

    def restart_controller(self, config: IrisClusterConfig) -> str:
        return self.start_controller(config)

    def _delete_controller_deployment_and_wait(self) -> None:
        """Wait for the old controller to completely be stopped so we can reuse the PV."""
        self._kubectl.delete(K8sResource.DEPLOYMENTS, "iris-controller")
        deadline = Deadline.from_seconds(_DEPLOYMENT_DELETE_TIMEOUT)
        while not self._shutdown_event.is_set():
            deployment_gone = self._kubectl.get_json(K8sResource.DEPLOYMENTS, "iris-controller") is None
            pods = self._kubectl.list_json(K8sResource.PODS, labels={"app": "iris-controller"})
            if deployment_gone and not pods:
                logger.info("Controller Deployment iris-controller and its Pods deleted")
                return
            if deadline.expired():
                pod_names = [p.get("metadata", {}).get("name", "") for p in pods]
                raise InfraError(
                    f"Controller Deployment iris-controller did not fully delete within "
                    f"{_DEPLOYMENT_DELETE_TIMEOUT}s (deployment_gone={deployment_gone}, "
                    f"pods still present: {pod_names})"
                )
            self._shutdown_event.wait(self._poll_interval)

    def stop_controller(self, config: IrisClusterConfig) -> None:
        cw = config.controller.coreweave
        service_name = cw.service_name or "iris-controller-svc"

        self._kubectl.delete(K8sResource.DEPLOYMENTS, "iris-controller")
        self._kubectl.delete(K8sResource.SERVICES, service_name)
        self._kubectl.delete(K8sResource.PDBS, "iris-controller-pdb")
        self._kubectl.delete(K8sResource.CONFIGMAPS, "iris-cluster-config")
        self._kubectl.delete(K8sResource.PERSISTENT_VOLUME_CLAIMS, _CONTROLLER_STATE_PVC_NAME)
        self._kubectl.delete(K8sResource.INGRESSES, _CONTROLLER_PROXY_INGRESS_NAME)
        if self.uses_s3_storage(config) or config.defaults.inject_env:
            self._kubectl.delete(K8sResource.SECRETS, TASK_ENV_SECRET_NAME)

        cluster_role_name = self.rbac_cluster_role_name()
        self._kubectl.delete(K8sResource.CLUSTER_ROLE_BINDINGS, cluster_role_name)
        self._kubectl.delete(K8sResource.CLUSTER_ROLES, cluster_role_name)
        logger.info("Controller resources deleted (including RBAC %s)", cluster_role_name)

    def stop_all(
        self,
        config: IrisClusterConfig,
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
                    # Bound via ClusterRoleBinding, so this grants pod access in
                    # ALL namespaces — required for blocker eviction in
                    # kubernetes_provider.preempt_namespaces, not just the Iris
                    # namespace.
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
                {
                    # Kueue gang admission: Iris deletes the per-pod-group
                    # Workload to release a torn-down gang's reserved quota
                    # (Kueue parks it in WaitingForReplacementPods otherwise).
                    "apiGroups": ["kueue.x-k8s.io"],
                    "resources": ["workloads"],
                    "verbs": ["get", "list", "watch", "delete"],
                },
                {
                    # Iris creates iris-{production,interactive,batch} PriorityClass
                    # objects at startup so pods can be stamped without manual setup.
                    "apiGroups": ["scheduling.k8s.io"],
                    "resources": ["priorityclasses"],
                    "verbs": ["get", "create", "update", "patch", "delete"],
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

    # -- Kueue ------------------------------------------------------------------

    def ensure_kueue_queues(self, config: IrisClusterConfig) -> None:
        """Reconcile the namespaced Kueue LocalQueue this cluster dispatches into.

        The Kueue operator, ClusterQueue, ResourceFlavor and Topology CRs are
        cluster-global and admin-provisioned out of band (the CKS cluster is
        shared across tenants); see scripts/install_kueue.py. Iris owns
        only its own LocalQueue, binding its namespace to the admin ClusterQueue.
        The LocalQueue name is derived from label_prefix, not configured. No-op
        when Kueue is not configured (cluster_queue unset).
        """
        cluster_queue = config.kubernetes_provider.kueue.cluster_queue
        if not cluster_queue:
            return
        name = local_queue_name(self._label_prefix)
        manifest = {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "LocalQueue",
            "metadata": {"name": name, "namespace": self._namespace},
            "spec": {"clusterQueue": cluster_queue},
        }
        self._kubectl.apply_json(manifest)
        logger.info("LocalQueue %s applied (clusterQueue=%s)", name, cluster_queue)

    def ensure_priority_classes(self) -> None:
        """Create or update the iris-{system,production,interactive,batch} PriorityClass objects.

        PriorityClass is cluster-scoped. Iris owns these names; any cluster
        running Iris gets them so pods are stamped without manual admin setup.

        Priority values (see IRIS_PRIORITY_CLASSES):
          iris-system     10000  — control plane (controller, finelog, Kueue); never preempted by user work
          iris-production  1000  — preempts interactive/batch; never preempted
          iris-interactive   10  — normal user work
          iris-batch          0  — opportunistic; below interactive, above CoreWeave NHC
        """
        for name, value, preemption_policy in IRIS_PRIORITY_CLASSES:
            manifest = build_priority_class_manifest(name, value, preemption_policy)
            existing = self._kubectl.get_json(K8sResource.PRIORITY_CLASSES, name)
            if existing and (existing.get("value") != value or existing.get("preemptionPolicy") != preemption_policy):
                # PriorityClass.value and preemptionPolicy are immutable. Existing
                # pods keep their admitted numeric priority after the class is
                # deleted, and new pods resolve the recreated class.
                logger.info("Replacing immutable PriorityClass %s", name)
                self._kubectl.delete(K8sResource.PRIORITY_CLASSES, name)
            self._kubectl.apply_json(manifest)
        logger.info(
            "PriorityClasses applied: %s",
            ", ".join(n for n, _, _ in IRIS_PRIORITY_CLASSES),
        )

    # -- NodePool Management ---------------------------------------------------

    def _resource_labels(self, scale_group: str) -> dict[str, str]:
        return {
            self._iris_labels.iris_managed: "true",
            self._iris_labels.iris_scale_group: scale_group,
        }

    def _nodepool_name(self, scale_group: str) -> str:
        # NodePool metadata.name must be a valid RFC 1123 subdomain (lowercase alphanumeric, '-', '.')
        return f"{self._label_prefix}-{scale_group}".replace("_", "-").lower()

    def ensure_nodepools(self, config: IrisClusterConfig) -> None:
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

    def uses_s3_storage(self, config: IrisClusterConfig) -> bool:
        """Check if any storage URI uses S3."""
        return config.storage.remote_state_dir.startswith("s3://")

    # -- Cluster default env (S3 storage auth + operator-injected vars) -------

    def _s3_task_env(self) -> dict[str, str]:
        """Compute S3 storage env (creds + endpoint + FSSPEC) from the operator's shell.

        Maps the operator's CoreWeave object-storage credentials to the AWS
        names boto3/s3fs expect and derives endpoint/region/FSSPEC_S3 from the
        configured object-storage endpoint. Folded into the iris-task-env
        Secret so the controller and every task authenticate to s3:// without
        per-call-site configuration.
        """
        key_id = os.environ.get("CW_KEY_ID")
        key_secret = os.environ.get("CW_KEY_SECRET")
        if not key_id or not key_secret:
            raise InfraError(
                "CW_KEY_ID and CW_KEY_SECRET environment variables are required for S3-compatible object storage"
            )
        env = {"AWS_ACCESS_KEY_ID": key_id, "AWS_SECRET_ACCESS_KEY": key_secret}
        endpoint = self._config.object_storage_endpoint
        if endpoint:
            env["AWS_ENDPOINT_URL"] = endpoint
            env["AWS_REGION"] = "auto"
            env["AWS_DEFAULT_REGION"] = "auto"
            fsspec_conf: dict = {"endpoint_url": endpoint}
            if _needs_virtual_host_addressing(endpoint):
                fsspec_conf["config_kwargs"] = {"s3": {"addressing_style": "virtual"}}
            # Non-AWS S3-compatible endpoints (R2, CoreWeave Object Storage)
            # reject the default us-east-1 region in the v4 signature with
            # 400 Bad Request. "auto" tells boto3 to skip region validation.
            fsspec_conf.setdefault("client_kwargs", {})["region_name"] = "auto"
            env["FSSPEC_S3"] = json.dumps(fsspec_conf)
        return env

    def ensure_task_env_secret(self, env: dict[str, str]) -> None:
        """Create the iris-task-env Secret holding the cluster default env.

        The controller and task pods reference this Secret via envFrom, so the
        values reach containers without ever passing through the ConfigMap.
        """
        self._kubectl.apply_json(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": TASK_ENV_SECRET_NAME, "namespace": self._namespace},
                "type": "Opaque",
                "data": {k: base64.b64encode(v.encode()).decode() for k, v in env.items()},
            }
        )

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
                            first = parse_k8s_timestamp(first_ts)
                            last = parse_k8s_timestamp(last_ts)
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

    def _config_json_for_configmap(self, config: IrisClusterConfig) -> str:
        """Serialize cluster config to JSON for the in-cluster ConfigMap."""
        config_dict = config_to_dict(config)
        cw_dict = config_dict.get("platform", {}).get("coreweave", {})
        cw_dict.pop("kubeconfig_path", None)
        return json.dumps(config_dict)
