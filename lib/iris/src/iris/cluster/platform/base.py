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

"""Platform-agnostic shared types for VM lifecycle.

These types define the contract for VM lifecycle helpers without leaking
provider-specific details into controller code.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import logging
from typing import Literal, Protocol
from collections.abc import Mapping

from iris.cluster.platform.ssh import SshConnection
from iris.rpc import config_pb2, vm_pb2


@dataclass(frozen=True)
class ContainerSpec:
    """Container launch specification."""

    image: str
    entrypoint: list[str]
    env: Mapping[str, str]
    ports: Mapping[str, int]
    health_port: int | None = None


@dataclass(frozen=True)
class VmBootstrapSpec:
    """Bootstrap specification for a VM role."""

    role: Literal["controller", "worker"]
    container: ContainerSpec
    labels: Mapping[str, str]
    bootstrap_script: str | None = None
    provider_overrides: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SliceVmTarget:
    """Connection target for a VM within a slice."""

    vm_id: str
    zone: str
    conn: SshConnection
    address: str | None = None


class SliceHandle(Protocol):
    """Handle for a provider slice (atomic group of VMs)."""

    @property
    def slice_id(self) -> str: ...

    @property
    def scale_group(self) -> str: ...

    @property
    def labels(self) -> Mapping[str, str]: ...

    @property
    def discovery_preamble(self) -> str: ...

    def vm_targets(self) -> list[SliceVmTarget]: ...

    def describe(self) -> vm_pb2.SliceInfo: ...

    def terminate(self) -> None: ...


class Platform(Protocol):
    """Factory for provider-specific VM managers and ops."""

    def tunnel(
        self,
        controller_address: str,
        local_port: int | None = None,
        timeout: float | None = None,
        tunnel_logger: logging.Logger | None = None,
    ) -> AbstractContextManager[str]:
        """Create tunnel to controller if needed."""
        ...

    def list_vms(self, *, tag: str | None = None, zone: str | None = None) -> list[vm_pb2.VmInfo]:
        """List VMs for the platform."""
        ...

    def start_vms(self, spec: VmBootstrapSpec, *, zone: str | None = None) -> list[vm_pb2.VmInfo]:
        """Start VMs using the bootstrap spec."""
        ...

    def stop_vms(self, ids: list[str], *, zone: str | None = None) -> None:
        """Stop VMs by id."""
        ...

    def list_slices(
        self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None
    ) -> list[vm_pb2.SliceInfo]:
        """List slices for the scale group."""
        ...

    def create_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        *,
        tags: dict[str, str] | None = None,
        zone: str | None = None,
    ) -> SliceHandle:
        """Create a new slice and return a handle."""
        ...

    def discover_slices(
        self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None
    ) -> list[SliceHandle]:
        """Discover existing slices and return handles."""
        ...

    def delete_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        slice_id: str,
        *,
        zone: str | None = None,
    ) -> None:
        """Delete a slice by id."""
        ...
