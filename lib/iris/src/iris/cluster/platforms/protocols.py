# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider protocols for Iris infrastructure management.

Two protocols define the boundary between Iris orchestration and infrastructure:

- ControllerProvider: controller lifecycle + connectivity (tunnel, image resolution).
- WorkerInfraProvider: worker/slice CRUD used by the Autoscaler and ScalingGroup.

Concrete implementations live in the sibling subpackages (gcp/, k8s/, manual/).
"""

from contextlib import AbstractContextManager
from typing import Protocol

from iris.cluster.config import ControllerVmConfig, IrisClusterConfig, SliceConfig, VmConfig, WorkerConfig
from iris.cluster.platforms.types import ListedSlice, SliceHandle, StandaloneWorkerHandle


class ControllerProvider(Protocol):
    """Controller lifecycle + connectivity.

    Covers controller discovery, start/restart/stop/stop_all, and
    connectivity methods (tunnel, resolve_image, debug_report).
    """

    def discover_controller(self, controller_config: ControllerVmConfig) -> str:
        """Discover controller address from platform-specific mechanism.

        Returns 'host:port' string. GCP queries VMs by label, Manual uses
        static config, CoreWeave returns K8s Service DNS.
        """
        ...

    def start_controller(self, config: IrisClusterConfig, *, fresh: bool = False) -> str:
        """Start or discover existing controller. Returns address (host:port).

        If fresh=True, the controller starts with an empty database instead
        of restoring from a remote checkpoint.
        """
        ...

    def restart_controller(self, config: IrisClusterConfig) -> str:
        """Restart controller in-place without destroying underlying compute."""
        ...

    def stop_controller(self, config: IrisClusterConfig) -> None:
        """Stop the controller and clean up its resources."""
        ...

    def stop_all(
        self,
        config: IrisClusterConfig,
        dry_run: bool = False,
        label_prefix: str | None = None,
    ) -> list[str]:
        """Stop controller and all managed slices.

        When dry_run=True, discovers resources but does not terminate them.
        Returns list of resource names that were (or would be) terminated.
        """
        ...

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        """Create a tunnel to a remote address if needed.

        GCP: SSH tunnel with port forwarding.
        Manual/Local: returns address directly (nullcontext).
        """
        ...

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        """Resolve a container image reference for this platform's registry.

        On GCP, rewrites public-registry images to the pull-through mirror the
        cluster's ``registry_mirrors`` config maps for the zone's continent.
        Other platforms return the image unchanged.
        """
        ...

    def debug_report(self) -> None:
        """Log diagnostic info about the controller after a failure."""
        ...

    def shutdown(self) -> None:
        """Release provider-owned resources (threads, connections, caches)."""
        ...


class WorkerInfraProvider(Protocol):
    """Worker infrastructure management for the Autoscaler and ScalingGroup.

    Handles creating and listing worker slices and standalone VMs.
    """

    def create_vm(self, config: VmConfig) -> StandaloneWorkerHandle:
        """Create a single standalone VM (e.g., for the controller)."""
        ...

    def create_slice(
        self,
        config: SliceConfig,
        worker_config: WorkerConfig | None = None,
    ) -> SliceHandle:
        """Create a slice of connected workers (e.g., TPU pod, IB GPU cluster).

        The slice is the atomic scaling unit. When worker_config is provided,
        the provider handles worker bootstrapping internally.
        """
        ...

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[SliceHandle]:
        """List existing slices, filtered by zone and optionally by labels."""
        ...

    def list_all_slices(self) -> list[ListedSlice]:
        """List every iris-managed slice across all zones, paired with its cloud state."""
        ...

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[StandaloneWorkerHandle]:
        """List existing standalone VMs, filtered by zone and optionally by labels."""
        ...

    def shutdown(self) -> None:
        """Release provider-owned resources (threads, connections, caches)."""
        ...
