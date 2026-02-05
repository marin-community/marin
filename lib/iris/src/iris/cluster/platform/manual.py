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
import shlex
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, nullcontext

from iris.cluster.platform.base import VmBootstrapSpec, VmInfo, VmState
from iris.cluster.platform.bootstrap import (
    CONTROLLER_CONTAINER_NAME,
    CONTROLLER_STOP_SCRIPT,
    DEFAULT_CONTROLLER_PORT,
    wait_healthy_via_ssh,
)
from iris.cluster.platform.ssh import DirectSshConnection, run_streaming_with_retry
from iris.cluster.platform.worker_vm import (
    WorkerVm,
    PoolExhaustedError,
    SshConfig,
    VmFactory,
    VmRegistry,
    TrackedVmFactory,
)
from iris.cluster.platform.vm_platform import (
    MAX_RECONCILE_WORKERS,
    VmGroupProtocol,
    VmGroupStatus,
    VmSnapshot,
    VmManagerProtocol,
)
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
        vms: list[WorkerVm],
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

    def vms(self) -> list[WorkerVm]:
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

    def stop(self) -> None:
        """No-op: ManualVmManager has no background threads to stop."""
        pass


class _ManualPlatformOps:
    def __init__(self, ssh_config: SshConfig, bootstrap: config_pb2.BootstrapConfig):
        self._ssh_config = ssh_config
        self._bootstrap = bootstrap

    def list_slices(self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None) -> list[str]:
        hosts = list(group_config.manual.hosts)
        if not hosts:
            return []
        port = self._bootstrap.worker_port or 10001
        running: list[str] = []
        for host in hosts:
            conn = _direct_ssh(_apply_manual_overrides(self._ssh_config, group_config.manual), host)
            try:
                result = conn.run(
                    f"curl -sf http://localhost:{port}/health",
                    timeout=Duration.from_seconds(10),
                )
                if result.returncode == 0:
                    running.append(host)
            except Exception as exc:
                logger.warning("Manual slice probe failed for host %s: %s", host, exc)
                continue
        return running

    def delete_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        slice_id: str,
        *,
        zone: str | None = None,
    ) -> None:
        conn = _direct_ssh(_apply_manual_overrides(self._ssh_config, group_config.manual), slice_id)
        cmd = (
            "sudo docker stop iris-worker 2>/dev/null || "
            "sudo docker kill iris-worker 2>/dev/null || true; "
            "sudo docker rm -f iris-worker 2>/dev/null || true"
        )
        conn.run(cmd, timeout=Duration.from_seconds(60))


class ManualPlatform:
    def __init__(
        self,
        label_prefix: str,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeout_config: config_pb2.TimeoutConfig,
        ssh_config: config_pb2.SshConfig,
    ):
        """Create manual platform with explicit config sections."""
        self._bootstrap = bootstrap_config
        self._timeouts = timeout_config
        self._label_prefix = label_prefix
        self._ssh = _ssh_config_to_dataclass(ssh_config)

    def vm_ops(self) -> _ManualPlatformOps:
        return _ManualPlatformOps(self._ssh, self._bootstrap)

    def tunnel(
        self,
        controller_address: str,
        local_port: int | None = None,
        timeout: float | None = None,
        tunnel_logger: logging.Logger | None = None,
    ) -> AbstractContextManager[str]:
        return nullcontext(controller_address)

    def vm_manager(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        vm_factory: TrackedVmFactory,
        *,
        dry_run: bool = False,
    ) -> VmManagerProtocol:
        if group_config.vm_type != config_pb2.VM_TYPE_MANUAL_VM:
            raise ValueError(f"Unsupported vm_type for manual platform: {group_config.vm_type}")
        hosts = list(group_config.manual.hosts)
        if not hosts:
            raise ValueError(f"Manual scale group {group_config.name} missing hosts")
        ssh_config = _apply_manual_overrides(self._ssh, group_config.manual)
        return ManualVmManager(
            hosts=hosts,
            config=group_config,
            bootstrap_config=self._bootstrap,
            timeouts=self._timeouts,
            vm_factory=vm_factory,
            ssh_config=ssh_config,
            label_prefix=self._label_prefix,
            dry_run=dry_run,
        )

    def list_vms(self, *, tag: str | None = None, zone: str | None = None) -> list[VmInfo]:
        return []

    def start_vms(self, spec: VmBootstrapSpec, *, zone: str | None = None) -> list[VmInfo]:
        if spec.role != "controller":
            raise NotImplementedError("ManualPlatform.start_vms currently supports controller role only")

        host = spec.provider_overrides.get("host")
        if not host:
            raise ValueError("provider_overrides.host is required for manual controller bootstrap")

        port = spec.provider_overrides.get("port", DEFAULT_CONTROLLER_PORT)
        ssh_user = spec.provider_overrides.get("ssh_user", self._ssh.user)
        ssh_key_file = spec.provider_overrides.get("ssh_key_file", self._ssh.key_file)

        conn = DirectSshConnection(
            host=host,
            user=ssh_user,
            port=self._ssh.port,
            key_file=ssh_key_file or None,
            connect_timeout=self._ssh.connect_timeout,
        )

        if spec.bootstrap_script is None:
            raise ValueError("bootstrap_script is required to start controller VM")

        def on_line(line: str) -> None:
            logger.info("[%s] %s", host, line)

        run_streaming_with_retry(
            conn,
            f"bash -c {shlex.quote(spec.bootstrap_script)}",
            on_line=on_line,
        )

        if port and not wait_healthy_via_ssh(conn, port):
            raise RuntimeError(f"Controller at {host}:{port} failed health check after bootstrap")

        address = spec.provider_overrides.get("address", f"http://{host}:{port}")

        return [
            VmInfo(
                vm_id=host,
                address=address,
                zone=None,
                labels=dict(spec.labels),
                state=VmState.RUNNING,
                created_at_ms=Timestamp.now().epoch_ms(),
            )
        ]

    def stop_vms(self, ids: list[str], *, zone: str | None = None) -> None:
        for host in ids:
            conn = DirectSshConnection(
                host=host,
                user=self._ssh.user,
                port=self._ssh.port,
                key_file=self._ssh.key_file,
                connect_timeout=self._ssh.connect_timeout,
            )
            try:
                conn.run(
                    f"bash -c {shlex.quote(CONTROLLER_STOP_SCRIPT.format(container_name=CONTROLLER_CONTAINER_NAME))}",
                    timeout=Duration.from_seconds(60),
                )
            except Exception as exc:
                logger.warning("Failed to stop controller on %s: %s", host, exc)


def _direct_ssh(ssh_config: SshConfig, host: str) -> DirectSshConnection:
    return DirectSshConnection(
        host=host,
        user=ssh_config.user,
        port=ssh_config.port,
        key_file=ssh_config.key_file or None,
        connect_timeout=ssh_config.connect_timeout,
    )


def _ssh_config_to_dataclass(ssh: config_pb2.SshConfig) -> SshConfig:
    """Convert proto SshConfig to dataclass SshConfig."""
    from iris.config import DEFAULT_CONFIG, DEFAULT_SSH_PORT

    connect_timeout = (
        Duration.from_proto(ssh.connect_timeout)
        if ssh.HasField("connect_timeout") and ssh.connect_timeout.milliseconds > 0
        else Duration.from_proto(DEFAULT_CONFIG.ssh.connect_timeout)
    )
    return SshConfig(
        user=ssh.user or DEFAULT_CONFIG.ssh.user,
        key_file=ssh.key_file or None,
        port=DEFAULT_SSH_PORT,
        connect_timeout=connect_timeout,
    )


def _apply_manual_overrides(ssh_config: SshConfig, manual: config_pb2.ManualProvider) -> SshConfig:
    return SshConfig(
        user=manual.ssh_user or ssh_config.user,
        key_file=manual.ssh_key_file or ssh_config.key_file,
        connect_timeout=ssh_config.connect_timeout,
    )
