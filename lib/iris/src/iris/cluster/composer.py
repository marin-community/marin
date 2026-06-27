# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose live cluster components from a parsed config.

:mod:`iris.cluster.config` parses and validates configuration into plain
pydantic models. This module is the inverse boundary: it imports the backends,
controller, and autoscaler and stitches a validated config into running
objects (task backends, provider bundles, worker configs, the autoscaler).

Keeping construction here means the config layer stays dependency-free and a
single place owns the wiring order.
"""

from __future__ import annotations

import logging

from finelog.client.log_client import Table

from iris.cluster.backends.k8s.tasks import _CW_DEFAULT_TOPOLOGIES, _DEFAULT_PRIORITY_CLASS_NAMES, K8sTaskProvider
from iris.cluster.backends.rpc.backend import RpcTaskBackend, RpcWorkerStubFactory
from iris.cluster.backends.types import local_queue_name
from iris.cluster.config import IrisClusterConfig, WorkerConfig
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.autoscaler.factory import create_autoscaler
from iris.cluster.controller.backend import BackendCapability, TaskBackend
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.log_stack import LogStack
from iris.cluster.inject_env import TASK_ENV_SECRET_NAME, projects_task_env_secret
from iris.cluster.platforms.factory import ProviderBundle, create_provider_bundle
from iris.cluster.platforms.k8s.service import CloudK8sService
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# Maps the band names used as keys in KueueConfig.priority_classes (and
# kubernetes_provider.priority_classes) to the PriorityBand enum stamped on pods.
_KUEUE_PRIORITY_BANDS = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}


def make_task_backend(
    config: IrisClusterConfig,
    *,
    task_stats_table: Table | None = None,
    profile_table: Table | None = None,
) -> TaskBackend:
    """Create a TaskBackend from cluster configuration.

    Returns a ``K8sTaskProvider`` when ``kubernetes_provider`` is configured,
    or an ``RpcTaskBackend`` when ``worker_provider`` is configured. The finelog
    tables are passed to the K8s backend (which writes per-pod resource/profile
    samples directly); the RPC backend ignores them — its worker daemons write
    their own rows.
    """
    which = config.provider_kind()
    if which == "kubernetes_provider":
        kp = config.kubernetes_provider
        namespace = kp.namespace or "iris"
        label_prefix = config.platform.label_prefix or "iris"
        managed_label = f"iris-{label_prefix}-managed" if label_prefix else ""

        priority_classes: dict[int, str] = {}
        for band_name, wpc in kp.kueue.priority_classes.items():
            band = _KUEUE_PRIORITY_BANDS.get(band_name)
            if band is None:
                raise ValueError(
                    f"Unknown Kueue priority band {band_name!r} in kueue.priority_classes; "
                    f"valid bands: {sorted(_KUEUE_PRIORITY_BANDS)}"
                )
            priority_classes[band] = wpc

        # Start from the iris-{band} defaults; override with any explicit config.
        pod_priority_classes: dict[int, str] = dict(_DEFAULT_PRIORITY_CLASS_NAMES)
        for band_name, pc_name in kp.priority_classes.items():
            band = _KUEUE_PRIORITY_BANDS.get(band_name)
            if band is None:
                raise ValueError(
                    f"Unknown priority band {band_name!r} in kubernetes_provider.priority_classes; "
                    f"valid bands: {sorted(_KUEUE_PRIORITY_BANDS)}"
                )
            pod_priority_classes[band] = pc_name

        # Empty topologies falls back to the CoreWeave-convention defaults.
        topologies = {group_by: (topo.node_label, topo.required) for group_by, topo in kp.kueue.topologies.items()}
        # Kueue is enabled by a configured cluster_queue; the LocalQueue name is
        # derived from label_prefix, not configured.
        local_queue = local_queue_name(label_prefix) if kp.kueue.cluster_queue else ""
        env_secret_name = TASK_ENV_SECRET_NAME if projects_task_env_secret(config) else ""
        return K8sTaskProvider(
            kubectl=CloudK8sService(namespace=namespace, kubeconfig_path=kp.kubeconfig or None),
            namespace=namespace,
            default_image=kp.default_image,
            logship_image=config.controller.image,
            service_account=kp.service_account or "",
            host_network=kp.host_network,
            cache_dir=kp.cache_dir or "/cache",
            controller_address=kp.controller_address or None,
            managed_label=managed_label,
            task_env=dict(config.defaults.task_env),
            env_secret_name=env_secret_name,
            local_queue=local_queue,
            kueue_priority_classes=priority_classes,
            kueue_topologies=topologies or dict(_CW_DEFAULT_TOPOLOGIES),
            preempt_namespaces=list(kp.preempt_namespaces),
            priority_class_names=pod_priority_classes,
            task_stats_table=task_stats_table,
            profile_table=profile_table,
        )
    if which == "worker_provider":
        return RpcTaskBackend(stub_factory=RpcWorkerStubFactory())
    raise ValueError(
        "IrisClusterConfig.provider must be set. Add either:\n"
        "  worker_provider: {}\n"
        "or:\n"
        "  kubernetes_provider:\n"
        "    namespace: iris\n"
        "    default_image: ...\n"
        "to your cluster config."
    )


def provider_bundle(config: IrisClusterConfig) -> ProviderBundle:
    """Create the ControllerProvider + WorkerInfraProvider bundle for *config*."""
    return create_provider_bundle(
        platform_config=config.platform,
        worker_port=config.defaults.worker.port,
        cluster_config=config,
        ssh_config=config.defaults.ssh,
    )


def build_base_worker_config(
    config: IrisClusterConfig,
    *,
    controller_address: str,
    storage_prefix: str,
    auth_token: str,
) -> WorkerConfig:
    """Build the base worker config the autoscaler ships to every worker.

    ``controller_address`` is pre-resolved by the caller (discovery runs only
    when the configured default is empty).
    """
    worker_config = config.defaults.worker.model_copy(deep=True)
    worker_config.controller_address = controller_address
    worker_config.platform = config.platform.model_copy(deep=True)
    worker_config.storage_prefix = storage_prefix
    if auth_token:
        worker_config.auth_token = auth_token
    return worker_config


def make_backend(
    config: IrisClusterConfig,
    *,
    db: ControllerDB,
    auth: ControllerAuth,
    remote_state_dir: str,
    dry_run: bool,
    log_stack: LogStack,
) -> TaskBackend:
    """Create the TaskBackend and, for Iris-provisioned backends, build, restore,
    and attach the autoscaler.

    The finelog tables from ``log_stack`` are threaded into the backend and
    autoscaler at construction. Capacity-managing backends (k8s) provision their
    own pods, so no autoscaler is attached. In dry-run both the autoscaler and the
    provider bundle are skipped (bundle creation needs platform credentials
    unavailable on a dev machine).
    """
    provider = make_task_backend(
        config,
        task_stats_table=log_stack.task_stats_table,
        profile_table=log_stack.profile_table,
    )
    logger.info("Backend created: %s", type(provider).__name__)

    if dry_run:
        logger.info("Dry-run mode: skipping autoscaler and provider bundle creation")
        return provider
    if BackendCapability.IRIS_AUTOSCALER not in provider.capabilities:
        return provider

    bundle = provider_bundle(config)
    workers = bundle.workers
    logger.info("Provider bundle created")

    base_worker_config = None
    if config.defaults.worker.docker_image:
        controller_address = config.defaults.worker.controller_address
        if not controller_address:
            controller_address = bundle.controller.discover_controller(config.controller)
        base_worker_config = build_base_worker_config(
            config,
            controller_address=controller_address,
            storage_prefix=remote_state_dir,
            auth_token=auth.worker_token or "",
        )

    autoscaler = create_autoscaler(
        platform=workers,
        autoscaler_config=config.defaults.autoscaler,
        scale_groups=config.scale_groups,
        label_prefix=config.platform.label_prefix or "iris",
        base_worker_config=base_worker_config,
        provisioning_table=log_stack.provisioning_table,
    )
    logger.info("Autoscaler created with %d scale groups", len(autoscaler.groups))

    autoscaler.restore_from_db(db, workers)
    logger.info("Autoscaler state restored from DB")

    provider.attach_autoscaler(autoscaler)
    return provider
