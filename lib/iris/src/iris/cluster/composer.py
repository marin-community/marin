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
from rigging.timing import Duration

from iris.cluster.backends.k8s.tasks import _CW_DEFAULT_TOPOLOGIES, _DEFAULT_PRIORITY_CLASS_NAMES, K8sTaskProvider
from iris.cluster.backends.rpc.backend import RpcTaskBackend, RpcWorkerStubFactory
from iris.cluster.config import (
    BackendConfig,
    IrisClusterConfig,
    WorkerConfig,
    backend_attribute_sets,
    resolve_backends,
)
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.factory import create_autoscaler
from iris.cluster.controller.backend import TaskBackend
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.log_stack import LogStack
from iris.cluster.controller.reconcile.loader import TransitionReader
from iris.cluster.controller.transition_reader import DbTransitionReader
from iris.cluster.inject_env import TASK_ENV_SECRET_NAME, projects_task_env_secret
from iris.cluster.platforms.factory import ProviderBundle, create_provider_bundle
from iris.cluster.platforms.k8s.service import CloudK8sService
from iris.cluster.platforms.types import local_queue_name
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
    unreachable_grace: Duration,
    task_stats_table: Table | None = None,
    profile_table: Table | None = None,
    autoscaler: Autoscaler | None = None,
    transition_reader: TransitionReader | None = None,
) -> TaskBackend:
    """Create a TaskBackend from cluster configuration.

    Returns a ``K8sTaskProvider`` when ``kubernetes_provider`` is configured,
    or an ``RpcTaskBackend`` when ``worker_provider`` is configured. The finelog
    tables are passed to the K8s backend (which writes per-pod resource/profile
    samples directly); the RPC backend ignores them — its worker daemons write
    their own rows. ``unreachable_grace`` sizes the liveness tracker the
    worker-daemon backend constructs and owns. ``transition_reader`` is the K8s
    backend's controller-DB read surface; ``autoscaler`` provisions capacity for
    the worker-daemon backend (None for clusters with no scale groups).
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
            kubectl=CloudK8sService(
                namespace=namespace,
                kubeconfig_path=kp.kubeconfig or None,
                context=kp.kube_context or None,
            ),
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
            transition_reader=transition_reader,
        )
    if which == "worker_provider":
        return RpcTaskBackend(
            stub_factory=RpcWorkerStubFactory(),
            unreachable_grace=unreachable_grace,
            autoscaler=autoscaler,
        )
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
    unreachable_grace: Duration,
) -> TaskBackend:
    """Create the TaskBackend and, for Iris-provisioned backends, build and restore
    the autoscaler that drives it.

    The finelog tables from ``log_stack`` are threaded into the backend and
    autoscaler at construction. Capacity-managing backends (k8s) provision their
    own pods, so no autoscaler is built; they read their dispatch drain through a
    controller-DB :class:`DbTransitionReader`. The autoscaler is built BEFORE the
    worker-daemon backend so it can be passed to its constructor. In dry-run both
    the autoscaler and the provider bundle are skipped (bundle creation needs
    platform credentials unavailable on a dev machine).
    """
    which = config.provider_kind()
    if which == "kubernetes_provider":
        provider = make_task_backend(
            config,
            unreachable_grace=unreachable_grace,
            task_stats_table=log_stack.task_stats_table,
            profile_table=log_stack.profile_table,
            transition_reader=DbTransitionReader(db),
        )
        logger.info("Backend created: %s", type(provider).__name__)
        return provider

    if which != "worker_provider":
        # Neither provider configured: defer to make_task_backend's guidance error.
        return make_task_backend(
            config,
            unreachable_grace=unreachable_grace,
            task_stats_table=log_stack.task_stats_table,
            profile_table=log_stack.profile_table,
        )

    autoscaler = None
    if dry_run:
        logger.info("Dry-run mode: skipping autoscaler and provider bundle creation")
    else:
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

    provider = make_task_backend(
        config,
        unreachable_grace=unreachable_grace,
        task_stats_table=log_stack.task_stats_table,
        profile_table=log_stack.profile_table,
        autoscaler=autoscaler,
    )
    logger.info("Backend created: %s", type(provider).__name__)
    return provider


def _backend_subconfig(config: IrisClusterConfig, backend: BackendConfig) -> IrisClusterConfig:
    """Project one ``BackendConfig`` back onto the top-level single-backend shape.

    ``make_backend`` reads the provider/scale-group/platform fields off the
    top-level config. A multi-backend cluster carries those per backend, so this
    rebuilds the implicit single-backend view for one entry while sharing the
    cluster-wide ``defaults``/``controller``/auth.
    """
    sub = config.model_copy(deep=True)
    sub.backends = None
    sub.worker_provider = backend.worker_provider
    sub.kubernetes_provider = backend.kubernetes_provider
    sub.scale_groups = dict(backend.scale_groups)
    sub.platform = backend.platform if backend.platform is not None else config.platform
    return sub


def make_backends(
    config: IrisClusterConfig,
    *,
    db: ControllerDB,
    auth: ControllerAuth,
    remote_state_dir: str,
    dry_run: bool,
    log_stack: LogStack,
    unreachable_grace: Duration,
) -> dict[str, TaskBackend]:
    """Build the controller's ``{backend_id: TaskBackend}`` collection.

    Iterates the resolved ``backends:`` map (or the implicit single entry keyed
    :data:`~iris.cluster.types.DEFAULT_BACKEND_ID`) and builds each through the
    single-backend path, tagging it with its backend id.
    """
    backends: dict[str, TaskBackend] = {}
    for backend_id, backend_cfg in resolve_backends(config).items():
        provider = make_backend(
            _backend_subconfig(config, backend_cfg),
            db=db,
            auth=auth,
            remote_state_dir=remote_state_dir,
            dry_run=dry_run,
            log_stack=log_stack,
            unreachable_grace=unreachable_grace,
        )
        provider.name = backend_id
        provider.configure_routing(backend_attribute_sets(backend_cfg), frozenset(backend_cfg.allow_policy.users))
        backends[backend_id] = provider
        logger.info("Backend %r ready: %s", backend_id, type(provider).__name__)
    return backends
