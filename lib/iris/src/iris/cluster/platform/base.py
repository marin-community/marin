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
allocation (VMs, slices) and infrastructure operations — it does NOT manage lifecycle
state machines, which are the controller layer's responsibility.

Status types (CloudSliceState, CloudVmState, etc.) represent the infrastructure
provider's view of resources, distinct from the Iris lifecycle states in vm.proto.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from iris.rpc import config_pb2
from iris.time_utils import Duration, Timestamp

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
    READY = "READY"
    REPAIRING = "REPAIRING"
    DELETING = "DELETING"
    UNKNOWN = "UNKNOWN"


class CloudVmState(StrEnum):
    """Cloud-level VM states."""

    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"


@dataclass
class SliceStatus:
    """Cloud-level slice status, including VM handles from the same query."""

    state: CloudSliceState
    vm_count: int
    vms: list[VmHandle] = field(default_factory=list)


@dataclass
class VmStatus:
    """Cloud-level VM status."""

    state: CloudVmState


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


# ============================================================================
# Handle Protocols
# ============================================================================


class VmHandle(Protocol):
    """Handle to a single VM within a slice. Provides raw infrastructure operations.

    The orchestration layer (WorkerVm) wraps this with lifecycle state machines,
    retries, and health checking.

    No terminate — slices are the atomic unit. Individual slice members cannot
    be terminated independently (e.g., TPU pod workers). Termination lives on
    SliceHandle.

    Thread safety: implementations must be safe for concurrent run_command() calls.
    For SSH-based handles, this means each run_command() creates a new SSH process.
    """

    @property
    def vm_id(self) -> str: ...

    @property
    def internal_address(self) -> str:
        """Internal/private IP address of this VM.

        This is the primary address used for all intra-cluster communication.
        """
        ...

    @property
    def external_address(self) -> str | None:
        """External/public IP address, if available. Returns None when no external IP."""
        ...

    def status(self) -> VmStatus:
        """Cloud-level VM status."""
        ...

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        """Poll until SSH/connection is available. Returns False on timeout."""
        ...

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        """Run a command on the VM via SSH. Optionally stream output lines."""
        ...

    def bootstrap(self, script: str) -> None:
        """Run a bootstrap script on the VM.

        Equivalent to run_command(f'bash -c {script}') with streaming.
        Raises on non-zero exit.
        """
        ...

    def reboot(self) -> None:
        """Reboot the VM."""
        ...


class StandaloneVmHandle(VmHandle, Protocol):
    """Handle to a standalone VM (e.g., controller). Can be terminated and labeled.

    Returned by platform.create_vm(). Extends VmHandle with operations that
    only make sense for independently-managed VMs:
    - terminate(): Destroy the VM
    - set_labels(): Tag for discovery (controller discovery uses labels)
    - set_metadata(): Pass data to the VM (controller address)

    Slice member VMs (from SliceHandle.describe().vms) are plain VmHandle —
    they can't be individually terminated, and their labels/metadata are
    set on the slice at creation time.
    """

    def terminate(self) -> None:
        """Destroy the VM."""
        ...

    def set_labels(self, labels: dict[str, str]) -> None:
        """Set labels on the VM (for discovery via list_vms).

        GCE label values: lowercase alphanumeric + hyphens, max 63 chars.
        """
        ...

    def set_metadata(self, metadata: dict[str, str]) -> None:
        """Set arbitrary key-value metadata on the VM.

        Unlike labels, metadata values have no character restrictions.
        On GCP, accessible via the metadata server from within the VM.
        """
        ...


class SliceHandle(Protocol):
    """Handle to an allocated slice of connected VMs.

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

        Extracted from labels (e.g., labels["{prefix}-scale-group"]).
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
        """Query cloud state, returning status and VM handles.

        Implementations may cache the result for a short TTL to avoid
        redundant cloud API calls within a single autoscaler cycle.
        """
        ...

    def terminate(self) -> None:
        """Destroy the slice and all its VMs."""
        ...


# ============================================================================
# Platform Protocol
# ============================================================================


class Platform(Protocol):
    """Infrastructure provider abstraction.

    Handles resource allocation (VMs, slices) and infrastructure operations.
    Does NOT manage lifecycle state machines — that's the controller layer's job.
    """

    def create_vm(self, config: config_pb2.VmConfig) -> StandaloneVmHandle:
        """Create a single VM (e.g., for the controller)."""
        ...

    def create_slice(self, config: config_pb2.SliceConfig) -> SliceHandle:
        """Create a slice of connected VMs (e.g., TPU pod, IB GPU cluster).

        The slice is the atomic scaling unit — it succeeds or fails as a whole.
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
        slices regardless of the zones parameter — their resources exist in
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
    ) -> list[VmHandle]:
        """List existing VMs, filtered by zone and optionally by labels.

        Non-cloud platforms (Manual, Local) return all VMs regardless of the
        zones parameter — their resources exist in a single synthetic zone.
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

        Distinct from terminate() on handles — shutdown() doesn't destroy
        cloud resources. It cleans up the Platform object itself.

        For LocalPlatform this stops worker threads managed by ThreadContainer.
        For GCP/Manual this is typically a no-op.
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
