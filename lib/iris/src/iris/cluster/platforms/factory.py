# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Factory for creating provider bundles from cluster configuration."""

from dataclasses import dataclass

from iris.cluster.config import GcpPlatformConfig, IrisClusterConfig, PlatformConfig, SshConfig
from iris.cluster.platforms.gcp.controller import GcpControllerProvider
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.workers import GcpWorkerProvider
from iris.cluster.platforms.k8s.controller import K8sControllerProvider
from iris.cluster.platforms.manual.controller import ManualControllerProvider
from iris.cluster.platforms.manual.workers import ManualWorkerProvider
from iris.cluster.platforms.protocols import ControllerProvider, WorkerInfraProvider
from iris.cluster.service_mode import ServiceMode


@dataclass
class ProviderBundle:
    """ControllerProvider + optional WorkerInfraProvider returned by create_provider_bundle.

    workers is None for K8s deployments where the K8sTaskProvider manages pods
    directly rather than through an autoscaler.
    """

    controller: ControllerProvider
    workers: WorkerInfraProvider | None


def create_provider_bundle(
    platform_config: PlatformConfig,
    worker_port: int,
    cluster_config: IrisClusterConfig | None = None,
    ssh_config: SshConfig | None = None,
) -> ProviderBundle:
    """Create a ControllerProvider + WorkerInfraProvider bundle from configuration.

    Args:
        platform_config: Provider type and provider-specific settings.
        worker_port: RPC port workers serve on, used to build their reachable URLs.
        cluster_config: Full cluster configuration when provider-specific defaults
            need controller/worker settings beyond platform_config.
        ssh_config: SSH settings (used by GCP and manual providers).

    Raises:
        ValueError: If platform type is unspecified or unknown.
    """
    which = platform_config.platform_kind() if platform_config is not None else None
    if which is None:
        raise ValueError("platform is required")
    label_prefix = platform_config.label_prefix or "iris"

    if which == "gcp":
        if not platform_config.gcp.project_id:
            raise ValueError("platform.gcp.project_id is required")
        worker_provider = GcpWorkerProvider(
            gcp_config=platform_config.gcp,
            label_prefix=label_prefix,
            worker_port=worker_port,
            ssh_config=ssh_config,
        )
        return ProviderBundle(
            controller=GcpControllerProvider(
                worker_provider=worker_provider,
                controller_service_account=(
                    cluster_config.controller.gcp.service_account
                    if cluster_config and cluster_config.controller.controller_kind() == "gcp"
                    else None
                ),
            ),
            workers=worker_provider,
        )

    if which == "manual":
        worker_provider = ManualWorkerProvider(
            label_prefix=label_prefix,
            worker_port=worker_port,
            ssh_config=ssh_config,
        )
        return ProviderBundle(
            controller=ManualControllerProvider(worker_provider=worker_provider),
            workers=worker_provider,
        )

    if which == "local":
        local_gcp_config = GcpPlatformConfig(project_id="local")
        gcp_service = InMemoryGcpService(mode=ServiceMode.LOCAL, project_id="local", label_prefix=label_prefix)
        worker_provider = GcpWorkerProvider(
            gcp_config=local_gcp_config,
            label_prefix=label_prefix,
            worker_port=worker_port,
            gcp_service=gcp_service,
        )
        return ProviderBundle(
            controller=GcpControllerProvider(worker_provider=worker_provider),
            workers=worker_provider,
        )

    if which == "coreweave":
        controller = K8sControllerProvider(
            config=platform_config.coreweave,
            label_prefix=label_prefix,
        )
        return ProviderBundle(
            controller=controller,
            workers=None,
        )

    raise ValueError(f"Unknown platform: {which}")
