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

"""ManualPlatform implementation for pre-existing hosts.

Implements the full Platform interface for manually managed hosts. Hosts are
drawn from a configured pool and returned on slice termination. SSH connections
use DirectSshConnection (raw ssh, no gcloud).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

from iris.cluster.platform._vm_base import SshVmBase
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudVmState,
    SliceStatus,
    VmStatus,
)
from iris.cluster.platform.ssh import (
    DirectSshConnection,
)
from iris.rpc import config_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


# ============================================================================
# Handle Implementations
# ============================================================================


@dataclass
class ManualVmHandle(SshVmBase):
    """Handle to a VM on a manual (pre-existing) host.

    Uses DirectSshConnection for SSH. Thread-safe: each run_command() spawns
    a new SSH process.
    """

    def status(self) -> VmStatus:
        return VmStatus(state=CloudVmState.RUNNING)


@dataclass
class ManualStandaloneVmHandle(SshVmBase):
    """Handle to a standalone VM on a manual host (e.g., controller).

    Extends ManualVmHandle with terminate, set_labels, and set_metadata.
    Labels and metadata are tracked in-memory since manual hosts don't have
    a cloud metadata service.
    """

    _labels: dict[str, str] = field(default_factory=dict)
    _metadata: dict[str, str] = field(default_factory=dict)
    _on_terminate: Callable[[], None] | None = None

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    def status(self) -> VmStatus:
        return VmStatus(state=CloudVmState.RUNNING)

    def terminate(self) -> None:
        if self._on_terminate:
            self._on_terminate()

    def set_labels(self, labels: dict[str, str]) -> None:
        self._labels.update(labels)

    def set_metadata(self, metadata: dict[str, str]) -> None:
        self._metadata.update(metadata)


@dataclass
class ManualSliceHandle:
    """Handle to a slice of manual hosts.

    Hosts are pre-existing and not destroyed on terminate — they are returned
    to the pool instead.
    """

    _slice_id: str
    _hosts: list[str]
    _labels: dict[str, str]
    _created_at: Timestamp
    _label_prefix: str
    _ssh_connections: list[DirectSshConnection]
    _on_terminate: Callable[[list[str]], None] | None = None
    _terminated: bool = False

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return "manual"

    @property
    def scale_group(self) -> str:
        return self._labels.get(f"{self._label_prefix}-scale-group", "")

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def list_vms(self) -> list[ManualVmHandle]:
        return [
            ManualVmHandle(
                _vm_id=f"{self._slice_id}-{host.replace('.', '-').replace(':', '-')}",
                _internal_address=host,
                _ssh=ssh,
            )
            for host, ssh in zip(self._hosts, self._ssh_connections, strict=True)
        ]

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        if self._on_terminate:
            self._on_terminate(list(self._hosts))
        logger.info("Terminated manual slice %s (%d hosts)", self._slice_id, len(self._hosts))

    def status(self) -> SliceStatus:
        if self._terminated:
            return SliceStatus(state=CloudSliceState.DELETING, vm_count=0)
        return SliceStatus(state=CloudSliceState.READY, vm_count=len(self._hosts))


# ============================================================================
# ManualPlatform
# ============================================================================


class ManualPlatform:
    """Platform for pre-existing hosts managed without cloud provisioning.

    Hosts are drawn from a configured pool. On slice termination, hosts are
    returned to the pool for reuse. SSH uses DirectSshConnection.
    """

    def __init__(
        self,
        label_prefix: str,
        ssh_config: config_pb2.SshConfig | None = None,
        hosts: list[str] | None = None,
    ):
        self._label_prefix = label_prefix
        self._ssh_config = ssh_config
        self._all_hosts = list(hosts or [])
        self._available_hosts: set[str] = set(self._all_hosts)
        self._allocated_hosts: set[str] = set()
        self._slices: dict[str, ManualSliceHandle] = {}
        self._vms: dict[str, ManualStandaloneVmHandle] = {}

    def create_vm(self, config: config_pb2.VmConfig) -> ManualStandaloneVmHandle:
        """Allocate a host from the pool for a standalone VM (e.g., controller)."""
        manual = config.manual
        host = manual.host

        # If specific host is requested, use it; otherwise pop from pool
        if host:
            if host in self._allocated_hosts:
                raise RuntimeError(f"Host {host} is already allocated")
            self._available_hosts.discard(host)
        elif self._available_hosts:
            host = self._available_hosts.pop()
        else:
            raise RuntimeError("No hosts available in manual platform pool")

        self._allocated_hosts.add(host)
        ssh = self._create_ssh_connection(host, manual)

        def on_terminate() -> None:
            self._return_hosts([host])
            self._vms.pop(config.name, None)

        handle = ManualStandaloneVmHandle(
            _vm_id=config.name,
            _internal_address=host,
            _ssh=ssh,
            _labels=dict(config.labels),
            _metadata=dict(config.metadata),
            _on_terminate=on_terminate,
        )
        self._vms[config.name] = handle
        return handle

    def create_slice(self, config: config_pb2.SliceConfig) -> ManualSliceHandle:
        """Allocate hosts from the pool for a slice."""
        manual = config.manual
        slice_id = f"{config.name_prefix}-{Timestamp.now().epoch_ms()}"

        # Use explicitly listed hosts if provided, otherwise draw from pool
        if manual.hosts:
            hosts = list(manual.hosts)
            already_allocated = self._allocated_hosts & set(hosts)
            if already_allocated:
                raise RuntimeError(f"Hosts already allocated: {sorted(already_allocated)}")
            for h in hosts:
                self._available_hosts.discard(h)
        else:
            needed = config.slice_size or 1
            if len(self._available_hosts) < needed:
                raise RuntimeError(f"Need {needed} hosts but only {len(self._available_hosts)} available")
            hosts = [self._available_hosts.pop() for _ in range(needed)]

        self._allocated_hosts.update(hosts)
        ssh_connections = [self._create_ssh_connection(h, manual) for h in hosts]

        def on_terminate(terminated_hosts: list[str]) -> None:
            self._return_hosts(terminated_hosts)
            self._slices.pop(slice_id, None)

        handle = ManualSliceHandle(
            _slice_id=slice_id,
            _hosts=hosts,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _ssh_connections=ssh_connections,
            _on_terminate=on_terminate,
        )
        self._slices[slice_id] = handle
        return handle

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[ManualSliceHandle]:
        """List all manual slices, optionally filtered by labels.

        The zones parameter is accepted for interface compatibility but ignored —
        all manual slices report zone="manual".
        """
        results = list(self._slices.values())
        if labels:
            results = [s for s in results if all(s.labels.get(k) == v for k, v in labels.items())]
        return results

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[ManualStandaloneVmHandle]:
        """List all manual standalone VMs, optionally filtered by labels.

        The zones parameter is accepted for interface compatibility but ignored —
        manual VMs have no zone concept.
        """
        results = list(self._vms.values())
        if labels:
            results = [v for v in results if all(v.labels.get(k) == v_ for k, v_ in labels.items())]
        return results

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        return nullcontext(address)

    def shutdown(self) -> None:
        pass

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Return static controller address from config."""
        manual = controller_config.manual
        port = manual.port or 10000
        return f"{manual.host}:{port}"

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _create_ssh_connection(
        self, host: str, manual_config: config_pb2.ManualVmConfig | config_pb2.ManualSliceConfig | None = None
    ) -> DirectSshConnection:
        """Create an SSH connection for the given host.

        Uses SSH config from the manual_config if provided (per-VM/slice overrides),
        falling back to the platform-level ssh_config.
        """
        user = "root"
        key_file: str | None = None
        connect_timeout = Duration.from_seconds(30)

        # Platform-level defaults
        if self._ssh_config:
            if self._ssh_config.user:
                user = self._ssh_config.user
            if self._ssh_config.key_file:
                key_file = self._ssh_config.key_file
            if self._ssh_config.HasField("connect_timeout"):
                connect_timeout = Duration.from_proto(self._ssh_config.connect_timeout)

        # Per-VM/slice overrides
        if manual_config is not None:
            ssh_user = getattr(manual_config, "ssh_user", "")
            ssh_key = getattr(manual_config, "ssh_key_file", "")
            if ssh_user:
                user = ssh_user
            if ssh_key:
                key_file = ssh_key

        return DirectSshConnection(
            host=host,
            user=user,
            key_file=key_file,
            connect_timeout=connect_timeout,
        )

    def _return_hosts(self, hosts: list[str]) -> None:
        """Return hosts to the available pool."""
        for host in hosts:
            self._allocated_hosts.discard(host)
            if host in self._all_hosts:
                self._available_hosts.add(host)
                logger.debug("Host %s returned to pool", host)

    @property
    def available_host_count(self) -> int:
        return len(self._available_hosts)
