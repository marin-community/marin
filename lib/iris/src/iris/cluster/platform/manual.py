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
- ManualSliceHandle: Slice handle for manually managed hosts
- ManualPlatform: Manual slice management and controller VM bootstrap

Unlike cloud platforms, manual hosts are:
- Pre-existing (not provisioned by the manager)
- Managed as a fixed pool (returned when terminated)
- Accessed via direct SSH (not gcloud)
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

from iris.cluster.platform.base import SliceHandle, SliceVmTarget, VmBootstrapSpec
from iris.cluster.platform.bootstrap import (
    CONTROLLER_CONTAINER_NAME,
    CONTROLLER_STOP_SCRIPT,
    DEFAULT_CONTROLLER_PORT,
    wait_healthy_via_ssh,
)
from iris.cluster.platform.ssh import DirectSshConnection, run_streaming_with_retry
from iris.cluster.controller.worker_vm import PoolExhaustedError, SshConfig
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


class ManualSliceHandle:
    """Slice handle for a single manual host."""

    def __init__(
        self,
        host: str,
        scale_group: str,
        labels: dict[str, str],
        ssh_config: SshConfig,
        discovery_preamble: str,
        on_release: Callable[[str], None] | None = None,
        created_at: Timestamp | None = None,
    ):
        self._host = host
        self._scale_group = scale_group
        self._labels = labels
        self._ssh_config = ssh_config
        self._discovery_preamble = discovery_preamble
        self._on_release = on_release
        self._created_at = created_at if created_at is not None else Timestamp.now()

    @property
    def slice_id(self) -> str:
        return self._host

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def labels(self) -> dict[str, str]:
        return self._labels

    @property
    def discovery_preamble(self) -> str:
        return self._discovery_preamble

    def vm_targets(self) -> list[SliceVmTarget]:
        conn = _direct_ssh(self._ssh_config, self._host)
        return [
            SliceVmTarget(
                vm_id=self._host,
                zone="",
                conn=conn,
                address=self._host,
            )
        ]

    def describe(self) -> vm_pb2.SliceInfo:
        vm_info = vm_pb2.VmInfo(
            vm_id=self._host,
            slice_id=self._host,
            scale_group=self._scale_group,
            state=vm_pb2.VM_STATE_BOOTING,
            address=self._host,
            zone="",
            created_at=self._created_at.to_proto(),
            labels=dict(self._labels),
        )
        return vm_pb2.SliceInfo(
            slice_id=self._host,
            scale_group=self._scale_group,
            created_at=self._created_at.to_proto(),
            vms=[vm_info],
        )

    def terminate(self) -> None:
        if self._on_release:
            self._on_release(self._host)


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
        self._available_hosts: dict[str, set[str]] = {}

    def tunnel(
        self,
        controller_address: str,
        local_port: int | None = None,
        timeout: float | None = None,
        tunnel_logger: logging.Logger | None = None,
    ) -> AbstractContextManager[str]:
        return nullcontext(controller_address)

    def list_vms(self, *, tag: str | None = None, zone: str | None = None) -> list[vm_pb2.VmInfo]:
        return []

    def start_vms(self, spec: VmBootstrapSpec, *, zone: str | None = None) -> list[vm_pb2.VmInfo]:
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
            vm_pb2.VmInfo(
                vm_id=host,
                address=address,
                zone="",
                labels=dict(spec.labels),
                state=vm_pb2.VM_STATE_READY,
                created_at=Timestamp.now().to_proto(),
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

    def list_slices(
        self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None
    ) -> list[vm_pb2.SliceInfo]:
        hosts = _probe_manual_hosts(group_config, self._ssh, self._bootstrap)
        slices: list[vm_pb2.SliceInfo] = []
        for host in hosts:
            handle = self._build_handle(group_config, host, reserve=False)
            slices.append(handle.describe())
        return slices

    def create_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        *,
        tags: dict[str, str] | None = None,
        zone: str | None = None,
    ) -> SliceHandle:
        if group_config.vm_type != config_pb2.VM_TYPE_MANUAL_VM:
            raise ValueError(f"Unsupported vm_type for manual platform: {group_config.vm_type}")
        hosts = list(group_config.manual.hosts)
        if not hosts:
            raise ValueError(f"Manual scale group {group_config.name} missing hosts")

        pool = self._available_hosts.setdefault(group_config.name, set(hosts))
        if not pool:
            raise PoolExhaustedError(f"No hosts available in pool (total: {len(hosts)}, all currently in use)")
        host = pool.pop()
        handle = self._build_handle(group_config, host, reserve=True, tags=tags)
        return handle

    def discover_slices(
        self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None
    ) -> list[SliceHandle]:
        hosts = _probe_manual_hosts(group_config, self._ssh, self._bootstrap)
        pool = self._available_hosts.setdefault(group_config.name, set(group_config.manual.hosts))
        for host in hosts:
            if host in pool:
                pool.remove(host)
        return [self._build_handle(group_config, host, reserve=False) for host in hosts]

    def delete_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        slice_id: str,
        *,
        zone: str | None = None,
    ) -> None:
        conn = _direct_ssh(_apply_manual_overrides(self._ssh, group_config.manual), slice_id)
        cmd = (
            "sudo docker stop iris-worker 2>/dev/null || "
            "sudo docker kill iris-worker 2>/dev/null || true; "
            "sudo docker rm -f iris-worker 2>/dev/null || true"
        )
        conn.run(cmd, timeout=Duration.from_seconds(60))

    def _build_handle(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        host: str,
        *,
        reserve: bool,
        tags: dict[str, str] | None = None,
    ) -> ManualSliceHandle:
        labels = {
            f"{self._label_prefix}-managed": "true",
            f"{self._label_prefix}-scale-group": group_config.name,
            f"{self._label_prefix}-slice-id": host,
        }
        labels.update(tags or {})
        discovery_preamble = _manual_discovery_preamble(self._bootstrap)
        ssh_config = _apply_manual_overrides(self._ssh, group_config.manual)

        def on_release(released_host: str) -> None:
            if reserve:
                pool = self._available_hosts.setdefault(group_config.name, set(group_config.manual.hosts))
                pool.add(released_host)

        return ManualSliceHandle(
            host=host,
            scale_group=group_config.name,
            labels=labels,
            ssh_config=ssh_config,
            discovery_preamble=discovery_preamble,
            on_release=on_release,
        )


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


def _manual_discovery_preamble(bootstrap: config_pb2.BootstrapConfig) -> str:
    addr = bootstrap.controller_address
    return f"""
# Use static controller address
CONTROLLER_ADDRESS="{addr}"
if [ -z "$CONTROLLER_ADDRESS" ]; then
    echo "[iris-init] ERROR: No controller address configured"
    exit 1
fi
echo "[iris-init] Using static controller at $CONTROLLER_ADDRESS"
"""


def _probe_manual_hosts(
    group_config: config_pb2.ScaleGroupConfig,
    ssh_config: SshConfig,
    bootstrap: config_pb2.BootstrapConfig,
) -> list[str]:
    hosts = list(group_config.manual.hosts)
    if not hosts:
        return []
    port = bootstrap.worker_port or 10001
    running: list[str] = []
    for host in hosts:
        conn = _direct_ssh(_apply_manual_overrides(ssh_config, group_config.manual), host)
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
