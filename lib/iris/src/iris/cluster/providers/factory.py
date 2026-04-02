# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Factory for creating provider bundles from cluster configuration."""

from dataclasses import dataclass

from iris.cluster.providers.gcp.controller import GcpControllerProvider
from iris.cluster.providers.gcp.fake import InMemoryGcpService
from iris.cluster.providers.gcp.workers import GcpWorkerProvider
from iris.cluster.providers.k8s.controller import K8sControllerProvider
from iris.cluster.providers.manual.provider import ManualControllerProvider, ManualWorkerProvider
from iris.cluster.providers.protocols import ControllerProvider, WorkerInfraProvider
from iris.cluster.service_mode import ServiceMode
from iris.rpc import config_pb2


@dataclass
class ProviderBundle:
    """ControllerProvider + optional WorkerInfraProvider returned by create_provider_bundle.

    workers is None for K8s deployments where the K8sTaskProvider manages pods
    directly rather than through an autoscaler.
    """

    controller: ControllerProvider
    workers: WorkerInfraProvider | None


def create_provider_bundle(
    platform_config: config_pb2.PlatformConfig,
    cluster_config: config_pb2.IrisClusterConfig | None = None,
    ssh_config: config_pb2.SshConfig | None = None,
) -> ProviderBundle:
    """Create a ControllerProvider + WorkerInfraProvider bundle from configuration.

    Args:
        platform_config: Provider type and provider-specific settings.
        cluster_config: Full cluster configuration when provider-specific defaults
            need controller/worker settings beyond platform_config.
        ssh_config: SSH settings (used by GCP and manual providers).

    Raises:
        ValueError: If platform type is unspecified or unknown.
    """
    if not platform_config.HasField("platform"):
        raise ValueError("platform is required")

    which = platform_config.WhichOneof("platform")
    label_prefix = platform_config.label_prefix or "iris"

    if which == "gcp":
        if not platform_config.gcp.project_id:
            raise ValueError("platform.gcp.project_id is required")
        worker_provider = GcpWorkerProvider(
            gcp_config=platform_config.gcp,
            label_prefix=label_prefix,
            ssh_config=ssh_config,
        )
        return ProviderBundle(
            controller=GcpControllerProvider(
                worker_provider=worker_provider,
                controller_service_account=(
                    cluster_config.controller.gcp.service_account
                    if cluster_config and cluster_config.controller.WhichOneof("controller") == "gcp"
                    else None
                ),
            ),
            workers=worker_provider,
        )

    if which == "manual":
        worker_provider = ManualWorkerProvider(
            label_prefix=label_prefix,
            ssh_config=ssh_config,
        )
        return ProviderBundle(
            controller=ManualControllerProvider(worker_provider=worker_provider),
            workers=worker_provider,
        )

    if which == "local":
        local_gcp_config = config_pb2.GcpPlatformConfig(project_id="local")
        gcp_service = InMemoryGcpService(mode=ServiceMode.LOCAL, project_id="local", label_prefix=label_prefix)
        worker_provider = GcpWorkerProvider(
            gcp_config=local_gcp_config,
            label_prefix=label_prefix,
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
