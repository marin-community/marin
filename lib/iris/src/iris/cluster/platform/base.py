# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
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

"""Protocol definitions, status types, and exception types for the Platform layer.

The Platform protocol hierarchy defines the interface that infrastructure providers
(GCP, CoreWeave, Manual, Local) must implement. The Platform itself handles resource
allocation (workers, slices) and infrastructure operations -- it does NOT manage lifecycle
state machines, which are the controller layer's responsibility.

Status types (CloudSliceState, CloudWorkerState, etc.) represent the infrastructure
provider's view of resources, distinct from the Iris lifecycle states in vm.proto.
"""

from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from iris.rpc import config_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)

# ============================================================================
# Label Keys
# ============================================================================


class Labels:
    """Label keys for Iris-managed cloud resources.

    All keys follow the format ``iris-{prefix}-<suffix>`` so resources are
    self-documenting and namespaced per cluster.
    """

    def __init__(self, prefix: str):
        self.iris_managed = f"iris-{prefix}-managed"
        self.iris_scale_group = f"iris-{prefix}-scale-group"
        self.iris_controller = f"iris-{prefix}-controller"
        self.iris_controller_address = f"iris-{prefix}-controller-address"
        self.iris_slice_id = f"iris-{prefix}-slice-id"


# ============================================================================
# Port Utilities
# ============================================================================


def find_free_port(start: int = -1) -> int:
    """Find an available port.

    Args:
        start: Starting port for sequential scan. Default of -1 lets the kernel
            pick a random ephemeral port, which avoids collisions when multiple
            processes search for ports concurrently (e.g. pytest-xdist).
    """
    if start == -1:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    for port in range(start, start + 1000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + 1000}")


# ============================================================================
# Exception Types
# ============================================================================


class PlatformError(Exception):
    """Base for platform operation failures."""


class QuotaExhaustedError(PlatformError):
    """No capacity in the requested zone. Try another zone or wait."""


class ResourceNotFoundError(PlatformError):
    """The requested resource type/variant doesn't exist."""


class PlatformUnavailableError(PlatformError):
    """Transient platform failure. Retry with backoff."""


# ============================================================================
# Status Types
# ============================================================================


class CloudSliceState(StrEnum):
    """Cloud-level slice states. Provider implementations map to these."""

    CREATING = "CREATING"
    BOOTSTRAPPING = "BOOTSTRAPPING"
    READY = "READY"
    FAILED = "FAILED"
    REPAIRING = "REPAIRING"
    DELETING = "DELETING"
    UNKNOWN = "UNKNOWN"


class CloudWorkerState(StrEnum):
    """Cloud-level worker states."""

    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"


@dataclass
class SliceStatus:
    """Cloud-level slice status, including worker handles from the same query."""

    state: CloudSliceState
    worker_count: int
    workers: list[RemoteWorkerHandle] = field(default_factory=list)
    error_message: str = ""


@dataclass
class WorkerStatus:
    """Cloud-level worker status."""

    state: CloudWorkerState


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


# ============================================================================
# Handle Protocols
# ============================================================================


class RemoteWorkerHandle(Protocol):
    """Handle to a single worker within a slice.

    Represents a remote worker process: a TPU VM on GCP, a Pod on CoreWeave,
    a thread on LocalPlatform. Provides infrastructure-level operations.
    Lifecycle state machines, retries, and health checking are the
    orchestration layer's responsibility.

    No terminate -- slices are the atomic unit. Individual slice members
    cannot be terminated independently.

    Thread safety: implementations must be safe for concurrent run_command() calls.
    """

    @property
    def worker_id(self) -> str: ...

    @property
    def vm_id(self) -> str: ...

    @property
    def internal_address(self) -> str:
        """Internal/private IP address of this worker.

        This is the primary address used for all intra-cluster communication.
        """
        ...

    @property
    def external_address(self) -> str | None:
        """External/public IP address, if available. Returns None when no external IP."""
        ...

    @property
    def bootstrap_log(self) -> str:
        """Most recent bootstrap output captured for this worker."""
        ...

    def status(self) -> WorkerStatus:
        """Cloud-level worker status."""
        ...

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        """Run a command on the worker. Optionally stream output lines."""
        ...

    def reboot(self) -> None:
        """Reboot the worker."""
        ...


class StandaloneWorkerHandle(RemoteWorkerHandle, Protocol):
    """Handle to a standalone worker (e.g., controller). Can be terminated and labeled.

    Returned by platform.create_vm(). Extends RemoteWorkerHandle with operations that
    only make sense for independently-managed workers:
    - terminate(): Destroy the worker
    - set_labels(): Tag for discovery (controller discovery uses labels)
    - set_metadata(): Pass data to the worker (controller address)

    Slice member workers (from SliceHandle.describe().workers) are plain
    RemoteWorkerHandle -- they can't be individually terminated, and their
    labels/metadata are set on the slice at creation time.
    """

    def terminate(self) -> None:
        """Destroy the worker."""
        ...

    def set_labels(self, labels: dict[str, str]) -> None:
        """Set labels on the worker (for discovery via list_vms).

        GCE label values: lowercase alphanumeric + hyphens, max 63 chars.
        """
        ...

    def set_metadata(self, metadata: dict[str, str]) -> None:
        """Set arbitrary key-value metadata on the worker.

        Unlike labels, metadata values have no character restrictions.
        On GCP, accessible via the metadata server from within the VM.
        """
        ...


class SliceHandle(Protocol):
    """Handle to an allocated slice of connected workers.

    A slice is the atomic scaling unit. For TPUs, it's a complete pod.
    For GPUs, it could be a set of IB-connected nodes.
    """

    @property
    def slice_id(self) -> str:
        """Unique identifier (e.g., 'iris-tpu_v5e_16-1738000000000')."""
        ...

    @property
    def zone(self) -> str:
        """Zone where this slice is allocated."""
        ...

    @property
    def scale_group(self) -> str:
        """Name of the scale group this slice belongs to.

        Extracted from labels (e.g., labels["iris-{prefix}-scale-group"]).
        """
        ...

    @property
    def labels(self) -> dict[str, str]:
        """Labels/tags set on this slice at creation time."""
        ...

    @property
    def created_at(self) -> Timestamp:
        """When this slice was created. Used for age-based scaling decisions."""
        ...

    def describe(self) -> SliceStatus:
        """Query cloud state, returning status and worker handles.

        Implementations may cache the result for a short TTL to avoid
        redundant cloud API calls within a single autoscaler cycle.
        """
        ...

    def terminate(self) -> None:
        """Destroy the slice and all its workers."""
        ...


# ============================================================================
# Platform Protocol
# ============================================================================


class Platform(Protocol):
    """Infrastructure provider abstraction.

    Handles resource allocation (workers, slices) and infrastructure operations.
    Does NOT manage lifecycle state machines -- that's the controller layer's job.
    """

    def create_vm(self, config: config_pb2.VmConfig) -> StandaloneWorkerHandle:
        """Create a single VM (e.g., for the controller)."""
        ...

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        bootstrap_config: config_pb2.BootstrapConfig | None = None,
    ) -> SliceHandle:
        """Create a slice of connected workers (e.g., TPU pod, IB GPU cluster).

        The slice is the atomic scaling unit -- it succeeds or fails as a whole.
        When bootstrap_config is provided, the platform handles worker bootstrapping
        internally (docker setup, worker container startup). describe() returns
        BOOTSTRAPPING while in progress, then READY or FAILED when complete.
        """
        ...

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[SliceHandle]:
        """List existing slices, filtered by zone and optionally by labels.

        Labels are exact key=value match. Zones are required because GCP TPU
        listing is per-zone. Non-cloud platforms (Manual, Local) return all
        slices regardless of the zones parameter -- their resources exist in
        a single synthetic zone ("manual" or "local").
        """
        ...

    def list_all_slices(
        self,
        labels: dict[str, str] | None = None,
    ) -> list[SliceHandle]:
        """List all slices across all configured zones, filtered by labels.

        Use list_slices() when the zone is already known for a more targeted query.
        """
        ...

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[RemoteWorkerHandle]:
        """List existing VMs, filtered by zone and optionally by labels.

        Non-cloud platforms (Manual, Local) return all VMs regardless of the
        zones parameter -- their resources exist in a single synthetic zone.
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

    def shutdown(self) -> None:
        """Release platform-owned resources (threads, connections, caches).

        Distinct from terminate() on handles -- shutdown() doesn't destroy
        cloud resources. It cleans up the Platform object itself.

        For LocalPlatform this stops worker threads managed by ThreadContainer.
        For GCP/Manual this is typically a no-op.
        """
        ...

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        """Resolve a container image reference for this platform's registry.

        On GCP, rewrites ``ghcr.io/`` images to the Artifact Registry remote
        repo for the given zone's continent.  Other platforms return the image
        unchanged.

        Args:
            image: Container image tag (e.g. ``ghcr.io/org/img:v1``).
            zone: Cloud zone used to select the regional mirror.  Required on
                GCP when the image starts with ``ghcr.io/``.

        Returns:
            Resolved image tag ready for ``docker pull``.
        """
        ...

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Discover controller address from platform-specific mechanism.

        Returns 'host:port' string. Each platform resolves this differently:
        - GCP: queries VMs by controller label
        - Manual: uses static host from config
        - Local: returns configured address or localhost
        """
        ...

    def start_controller(self, config: config_pb2.IrisClusterConfig) -> str:
        """Start or discover existing controller. Returns address (host:port).

        Each platform implements its own controller lifecycle:
        - GCP: creates GCE VM, SSHes in, bootstraps container
        - Manual: SSHes to configured host, bootstraps container
        - CoreWeave: kubectl apply ConfigMap + NodePool + Deployment + Service
        - Local: starts in-process LocalController
        """
        ...

    def stop_controller(self, config: config_pb2.IrisClusterConfig) -> None:
        """Stop the controller.

        Each platform tears down its own controller resources:
        - GCP: terminates GCE VM
        - Manual: terminates bootstrap on host
        - CoreWeave: kubectl delete Deployment + Service + NodePool
        - Local: stops in-process LocalController
        """
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

        Each platform implements its own teardown strategy:
        - GCP/Manual: list_all_slices + terminate each + stop_controller (parallel)
        - CoreWeave: kubectl delete NodePools + controller resources
        - Local: terminate slices + stop controller
        """
        ...

    def reload(self, config: config_pb2.IrisClusterConfig) -> str:
        """Reload controller and workers with updated images/config.

        Each platform implements its own reload strategy:
        - GCP/Manual: full stop + start (terminate all worker slices, then controller)
        - CoreWeave: update ConfigMap, reload worker Pods in parallel, then
          rolling update controller Deployment
        - Local: restart in-process controller

        Returns the controller address after reload.
        """
        ...


# ============================================================================
# Default stop_all implementation
# ============================================================================

TERMINATE_TIMEOUT_SECONDS = 60


def default_stop_all(
    platform: Platform,
    config: config_pb2.IrisClusterConfig,
    dry_run: bool = False,
    label_prefix: str | None = None,
) -> list[str]:
    """Default stop_all: discover via list_all_slices, terminate in parallel.

    Shared by GCP, Manual, and Local platforms. Enumerates all managed slices
    plus the controller, then runs all terminates concurrently via daemon threads
    with a hard timeout. Daemon threads are used instead of ThreadPoolExecutor so
    that timed-out threads don't block interpreter shutdown.
    """
    prefix = label_prefix or config.platform.label_prefix or "iris"
    labels = Labels(prefix)

    target_names: list[str] = ["controller"]
    all_slices = platform.list_all_slices(labels={labels.iris_managed: "true"})
    for s in all_slices:
        logger.info("Found managed slice %s", s.slice_id)
        target_names.append(f"slice:{s.slice_id}")

    if dry_run:
        return target_names

    targets: list[tuple[str, Callable[[], None]]] = [
        ("controller", lambda: platform.stop_controller(config)),
    ]
    for s in all_slices:
        targets.append((f"slice:{s.slice_id}", s.terminate))

    if not targets:
        logger.info("No resources to terminate")
        return target_names

    logger.info("Terminating %d resource(s) in parallel", len(targets))

    errors: list[str] = []
    results: dict[str, Exception | None] = {}
    lock = threading.Lock()

    def _run(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            with lock:
                results[name] = exc
            return
        with lock:
            results[name] = None

    threads: dict[str, threading.Thread] = {}
    for name, fn in targets:
        t = threading.Thread(target=_run, args=(name, fn), daemon=True)
        t.start()
        threads[name] = t

    dl = Deadline.from_seconds(TERMINATE_TIMEOUT_SECONDS)
    for _name, t in threads.items():
        t.join(timeout=dl.remaining_seconds())

    for name, t in threads.items():
        if t.is_alive():
            logger.warning(
                "Termination of %s still running after %ds, giving up",
                name,
                TERMINATE_TIMEOUT_SECONDS,
            )
            errors.append(f"timeout:{name}")
        else:
            exc = results.get(name)
            if exc is not None:
                logger.exception("Failed to terminate %s", name, exc_info=exc)
                errors.append(name)

    if errors:
        logger.error("Errors when stopping cluster: %s", errors)

    return target_names
