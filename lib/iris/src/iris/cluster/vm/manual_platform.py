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

"""Manual platform implementation for VM management with pre-existing hosts.

This module provides:
- ManualVmManager: Factory for creating VM groups from a pool of hosts
- ManualVmGroup: VM group implementation for manually managed hosts

Unlike cloud platforms, manual hosts are:
- Pre-existing (not provisioned by the manager)
- Managed as a fixed pool (returned when terminated)
- Accessed via direct SSH (not gcloud)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from iris.cluster.vm.managed_vm import (
    ManagedVm,
    PoolExhaustedError,
    SshConfig,
    VmFactory,
    VmRegistry,
)
from iris.cluster.vm.vm_platform import (
    MAX_RECONCILE_WORKERS,
    VmGroupProtocol,
    VmGroupStatus,
    VmSnapshot,
)
from iris.cluster.vm.ssh import DirectSshConnection
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


class ManualVmGroup:
    """A VM group of pre-existing hosts managed without cloud provisioning.

    Similar to TpuVmGroup but for manually managed hosts (not cloud-provisioned).
    On terminate(), this group:
    1. Stops worker containers on all hosts (via shutdown)
    2. Stops all VM lifecycle threads
    3. Unregisters VMs from the registry
    4. Calls on_terminate callback to return hosts to pool
    5. Does NOT delete cloud resources (there are none)
    """

    def __init__(
        self,
        group_id: str,
        scale_group: str,
        vms: list[ManagedVm],
        vm_registry: VmRegistry,
        created_at: Timestamp | None = None,
        on_terminate: Callable[[list[str]], None] | None = None,
    ):
        self._group_id = group_id
        self._scale_group = scale_group
        self._vms = vms
        self._vm_registry = vm_registry
        self._created_at = created_at if created_at is not None else Timestamp.now()
        self._on_terminate = on_terminate

    @property
    def group_id(self) -> str:
        return self._group_id

    @property
    def slice_id(self) -> str:
        return self._group_id

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def created_at_ms(self) -> int:
        """Timestamp when this VM group was created (milliseconds since epoch)."""
        return self._created_at.epoch_ms()

    def status(self) -> VmGroupStatus:
        """Compute status from current VM states."""
        snapshots = [
            VmSnapshot(
                vm_id=vm.info.vm_id,
                state=vm.info.state,
                address=vm.info.address,
                init_phase=vm.info.init_phase,
                init_error=vm.info.init_error,
            )
            for vm in self._vms
        ]
        return VmGroupStatus(vms=snapshots)

    def vms(self) -> list[ManagedVm]:
        return list(self._vms)

    def terminate(self) -> None:
        """Terminate this VM group by stopping workers and unregistering VMs.

        Unlike TpuVmGroup, this does not delete cloud resources since manual
        hosts are externally managed. It does:
        1. Gracefully shut down worker containers on each host
        2. Stop VM lifecycle threads
        3. Unregister VMs from the registry
        4. Call on_terminate callback to return hosts to pool
        """
        hosts = [vm.info.address for vm in self._vms]

        for vm in self._vms:
            vm.shutdown(graceful=True)
            vm.stop()
            self._vm_registry.unregister(vm.info.vm_id)

        if self._on_terminate:
            self._on_terminate(hosts)

        logger.info("Terminated manual VM group %s (%d hosts)", self._group_id, len(self._vms))

    def to_proto(self) -> vm_pb2.SliceInfo:
        """Convert to proto for RPC APIs."""

        return vm_pb2.SliceInfo(
            slice_id=self._group_id,
            scale_group=self._scale_group,
            created_at=self._created_at.to_proto(),
            vms=[vm.info for vm in self._vms],
        )


class ManualVmManager:
    """Creates VM groups from pre-existing hosts.

    One instance per scale group. This is a factory that creates ManualVmGroup
    objects from a pool of pre-configured host addresses.

    Unlike TpuVmManager:
    - Does not create/delete cloud resources
    - Uses DirectSshExecutor for host communication
    - Treats hosts as a fixed pool

    On create_vm_group(), assigns a host from the available pool.
    On vm_group.terminate(), the host is returned to the pool.
    """

    def __init__(
        self,
        hosts: list[str],
        config: config_pb2.ScaleGroupConfig,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeouts: config_pb2.TimeoutConfig,
        vm_factory: VmFactory,
        ssh_config: SshConfig | None = None,
        label_prefix: str = "iris",
        dry_run: bool = False,
    ):
        self._hosts = list(hosts)
        self._config = config
        self._bootstrap_config = bootstrap_config
        self._timeouts = timeouts
        self._vm_factory = vm_factory
        self._ssh_config = ssh_config or SshConfig()
        self._label_prefix = label_prefix
        self._dry_run = dry_run

        # Track available hosts (those not currently in use)
        self._available_hosts: set[str] = set(hosts)

    def _get_discovery_preamble(self) -> str:
        """Generate static controller discovery script for worker bootstrap."""
        addr = self._bootstrap_config.controller_address
        return f"""
# Use static controller address
CONTROLLER_ADDRESS="{addr}"
if [ -z "$CONTROLLER_ADDRESS" ]; then
    echo "[iris-init] ERROR: No controller address configured"
    exit 1
fi
echo "[iris-init] Using static controller at $CONTROLLER_ADDRESS"
"""

    def create_vm_group(self, tags: dict[str, str] | None = None) -> ManualVmGroup:
        """Create a new VM group by assigning a host from the pool.

        Raises PoolExhaustedError if no hosts are available.
        """
        if not self._available_hosts:
            raise PoolExhaustedError(f"No hosts available in pool (total: {len(self._hosts)}, all currently in use)")

        host = self._available_hosts.pop()
        group_id = f"{self._label_prefix}-{self._config.name}-{Timestamp.now().epoch_ms()}"

        if self._dry_run:
            logger.info("[DRY-RUN] Would create manual VM group %s (host=%s)", group_id, host)
            self._available_hosts.add(host)
            return ManualVmGroup(
                group_id=group_id,
                scale_group=self._config.name,
                vms=[],
                vm_registry=self._vm_factory.registry,
            )

        vm_id = f"{group_id}-{host.replace('.', '-').replace(':', '-')}"

        labels = {
            f"{self._label_prefix}-managed": "true",
            f"{self._label_prefix}-scale-group": self._config.name,
            f"{self._label_prefix}-slice-id": group_id,
        }
        labels.update(tags or {})

        logger.info(
            "Creating manual VM group %s (host=%s, scale_group=%s)",
            group_id,
            host,
            self._config.name,
        )

        # Manual workers use static controller address from config
        discovery_preamble = self._get_discovery_preamble()

        conn = self._create_ssh_connection(host)
        vm = self._vm_factory.create_vm(
            vm_id=vm_id,
            slice_id=group_id,
            scale_group=self._config.name,
            zone="manual",
            conn=conn,
            bootstrap_config=self._bootstrap_config,
            timeouts=self._timeouts,
            labels=labels,
            address=host,
            discovery_preamble=discovery_preamble,
        )

        return ManualVmGroup(
            group_id=group_id,
            scale_group=self._config.name,
            vms=[vm],
            vm_registry=self._vm_factory.registry,
            on_terminate=self._return_hosts,
        )

    def discover_vm_groups(self) -> list[VmGroupProtocol]:
        """Find existing VM groups by checking which hosts have running workers.

        Probes each host via SSH to check if a worker is already running.
        Returns ManualVmGroup objects for hosts with active workers.
        """
        groups: list[VmGroupProtocol] = []
        hosts_to_check = list(self._available_hosts)

        if not hosts_to_check:
            return groups

        logger.info("Discovering manual VM groups across %d hosts", len(hosts_to_check))

        workers = min(MAX_RECONCILE_WORKERS, len(hosts_to_check))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._check_worker_running, host): host for host in hosts_to_check}

            for future in as_completed(futures):
                host = futures[future]
                try:
                    is_running = future.result()
                    if is_running:
                        vm_group = self._adopt_running_host(host)
                        groups.append(vm_group)
                except Exception as e:
                    logger.warning("Failed to check host %s: %s", host, e)

        logger.info("Discovered %d manual VM groups", len(groups))
        return groups

    def _check_worker_running(self, host: str) -> bool:
        """Check if a worker is healthy on the given host."""
        conn = self._create_ssh_connection(host)
        port = self._bootstrap_config.worker_port or 10001
        try:
            result = conn.run(f"curl -sf http://localhost:{port}/health", timeout=Duration.from_seconds(10))
            return result.returncode == 0
        except Exception:
            return False

    def _adopt_running_host(self, host: str) -> ManualVmGroup:
        """Create a ManualVmGroup for a host that already has a running worker."""
        self._available_hosts.discard(host)

        group_id = f"{self._label_prefix}-{self._config.name}-recovered-{Timestamp.now().epoch_ms()}"
        vm_id = f"{group_id}-{host.replace('.', '-').replace(':', '-')}"

        labels = {
            f"{self._label_prefix}-managed": "true",
            f"{self._label_prefix}-scale-group": self._config.name,
            f"{self._label_prefix}-slice-id": group_id,
        }

        logger.info("Adopting running host %s as VM group %s", host, group_id)

        # Manual workers use static controller address from config
        discovery_preamble = self._get_discovery_preamble()

        conn = self._create_ssh_connection(host)
        vm = self._vm_factory.create_vm(
            vm_id=vm_id,
            slice_id=group_id,
            scale_group=self._config.name,
            zone="manual",
            conn=conn,
            bootstrap_config=self._bootstrap_config,
            timeouts=self._timeouts,
            labels=labels,
            address=host,
            discovery_preamble=discovery_preamble,
        )

        return ManualVmGroup(
            group_id=group_id,
            scale_group=self._config.name,
            vms=[vm],
            vm_registry=self._vm_factory.registry,
            on_terminate=self._return_hosts,
        )

    def _create_ssh_connection(self, host: str) -> DirectSshConnection:
        """Create an SSH connection for the given host."""
        return DirectSshConnection(
            host=host,
            user=self._ssh_config.user,
            port=self._ssh_config.port,
            key_file=self._ssh_config.key_file,
            connect_timeout=self._ssh_config.connect_timeout,
        )

    def _return_hosts(self, hosts: list[str]) -> None:
        """Return hosts to the available pool (callback for ManualVmGroup.terminate)."""
        for host in hosts:
            if host in self._hosts:
                self._available_hosts.add(host)
                logger.debug("Host %s returned to pool", host)

    def return_host(self, host: str) -> None:
        """Return a host to the available pool.

        Called when a VM group is terminated to make the host available again.
        Safe to call multiple times for the same host.
        """
        self._return_hosts([host])

    @property
    def available_host_count(self) -> int:
        """Number of hosts available for new VM groups."""
        return len(self._available_hosts)

    @property
    def total_host_count(self) -> int:
        """Total number of hosts in the pool."""
        return len(self._hosts)
