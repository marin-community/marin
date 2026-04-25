# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider protocols for Iris infrastructure management.

Two protocols define the boundary between Iris orchestration and infrastructure:

- ControllerProvider: controller lifecycle + connectivity (tunnel, image resolution).
- WorkerInfraProvider: worker/slice CRUD used by the Autoscaler and ScalingGroup.

Concrete implementations live under providers/gcp/, providers/k8s/, etc.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from iris.cluster.providers.types import SliceHandle, StandaloneWorkerHandle
from iris.rpc import config_pb2


class ControllerProvider(Protocol):
    """Controller lifecycle + connectivity.

    Covers controller discovery, start/restart/stop/stop_all, and
    connectivity methods (tunnel, resolve_image, debug_report).
    """

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Discover controller address from platform-specific mechanism.

        Returns 'host:port' string. GCP queries VMs by label, Manual uses
        static config, CoreWeave returns K8s Service DNS.
        """
        ...

    def start_controller(self, config: config_pb2.IrisClusterConfig, *, fresh: bool = False) -> str:
        """Start or discover existing controller. Returns address (host:port).

        If fresh=True, the controller starts with an empty database instead
        of restoring from a remote checkpoint.
        """
        ...

    def restart_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        """Restart controller in-place without destroying underlying compute."""
        ...

    def stop_controller(self, config: config_pb2.IrisClusterConfig) -> None:
        """Stop the controller and clean up its resources."""
        ...

    def stop_all(
        self,
        config: config_pb2.IrisClusterConfig,
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

    def tunnel_to(
        self,
        host: str,
        port: int,
        local_port: int | None = None,
    ) -> AbstractContextManager[tuple[str, int]]:
        """Create a tunnel to an arbitrary internal ``host:port`` and yield
        the local ``(host, port)`` reachable from this process.

        Used by ``iris.client.maybe_proxy`` for off-cluster access. Unlike
        :meth:`tunnel`, the target is not the controller — it is any
        cluster-internal address (a system-service VM, a K8s Service).

        GCP: ``gcloud compute ssh`` to a cluster VM with ``-L``.
        K8s: ``kubectl port-forward`` to a Service.
        Manual/Local: nullcontext returning the input as-is.
        """
        ...

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        """Resolve a container image reference for this platform's registry.

        On GCP, rewrites ghcr.io/ images to the Artifact Registry remote repo
        for the given zone's continent. Other platforms return the image unchanged.
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

    def create_vm(self, config: config_pb2.VmConfig) -> StandaloneWorkerHandle:
        """Create a single standalone VM (e.g., for the controller)."""
        ...

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
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

    def list_all_slices(self) -> list[SliceHandle]:
        """List all slices managed by this cluster across all zones."""
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
