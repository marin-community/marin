# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Platform abstraction for VM lifecycle and direct operations."""

from __future__ import annotations

import json
import logging
import subprocess
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from iris.cluster.vm.gcp_tpu_platform import TpuVmManager
from iris.cluster.vm.managed_vm import SshConfig, TrackedVmFactory
from iris.cluster.vm.manual_platform import ManualVmManager
from iris.cluster.vm.ssh import DirectSshConnection
from iris.cluster.vm.vm_platform import VmManagerProtocol
from iris.rpc import config_pb2
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


class PlatformOps(Protocol):
    """Direct, non-lifecycle VM operations used by CLI cleanup."""

    def list_slices(self, group_config: config_pb2.ScaleGroupConfig) -> list[str]: ...

    def delete_slice(self, group_config: config_pb2.ScaleGroupConfig, slice_id: str) -> None: ...


class Platform(Protocol):
    """Factory for provider-specific VM managers and ops."""

    def vm_ops(self) -> PlatformOps: ...

    def vm_manager(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        vm_factory: TrackedVmFactory,
        *,
        dry_run: bool = False,
    ) -> VmManagerProtocol: ...

    def tunnel(
        self,
        controller_address: str,
        local_port: int | None = None,
        timeout: float | None = None,
        tunnel_logger: logging.Logger | None = None,
    ) -> AbstractContextManager[str]:
        """Create tunnel to controller if needed.

        For GCP: Returns SSH tunnel context manager that discovers controller VM
        For local/manual: Returns nullcontext with direct address

        Args:
            controller_address: Controller address (used directly for local/manual)
            local_port: Optional local port for tunnel (GCP only)
            timeout: Optional connection timeout
            tunnel_logger: Optional logger for tunnel status

        Returns:
            Context manager yielding controller URL
        """
        ...


class _GcpPlatformOps:
    def __init__(self, platform: config_pb2.GcpPlatformConfig, label_prefix: str):
        self._platform = platform
        self._label_prefix = label_prefix

    def list_slices(self, group_config: config_pb2.ScaleGroupConfig) -> list[str]:
        zones = _resolve_zones(group_config, self._platform)
        label_filter = f"labels.{self._label_prefix}-scale-group={group_config.name}"
        slices: list[str] = []
        for zone in zones:
            for tpu in _gcloud_list_tpus(self._platform.project_id, zone, label_filter):
                state = tpu.get("state", "UNKNOWN")
                if state not in ("READY", "CREATING", "REPAIRING"):
                    continue
                slices.append(_extract_node_name(tpu.get("name", "")))
        return slices

    def delete_slice(self, group_config: config_pb2.ScaleGroupConfig, slice_id: str) -> None:
        zones = _resolve_zones(group_config, self._platform)
        for zone in zones:
            if _gcloud_delete_tpu(self._platform.project_id, zone, slice_id):
                return
        raise RuntimeError(f"Failed to delete TPU slice {slice_id} in any zone")


class _ManualPlatformOps:
    def __init__(self, ssh_config: SshConfig, bootstrap: config_pb2.BootstrapConfig):
        self._ssh_config = ssh_config
        self._bootstrap = bootstrap

    def list_slices(self, group_config: config_pb2.ScaleGroupConfig) -> list[str]:
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

    def delete_slice(self, group_config: config_pb2.ScaleGroupConfig, slice_id: str) -> None:
        conn = _direct_ssh(_apply_manual_overrides(self._ssh_config, group_config.manual), slice_id)
        cmd = (
            "sudo docker stop iris-worker 2>/dev/null || "
            "sudo docker kill iris-worker 2>/dev/null || true; "
            "sudo docker rm -f iris-worker 2>/dev/null || true"
        )
        conn.run(cmd, timeout=Duration.from_seconds(60))


class _LocalPlatformOps:
    def list_slices(self, group_config: config_pb2.ScaleGroupConfig) -> list[str]:
        return []

    def delete_slice(self, group_config: config_pb2.ScaleGroupConfig, slice_id: str) -> None:
        return None


class GcpPlatform:
    def __init__(
        self,
        gcp_config: config_pb2.GcpPlatformConfig,
        label_prefix: str,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeout_config: config_pb2.TimeoutConfig,
    ):
        """Create GCP platform with explicit config sections.

        No defaults here - all defaults resolved before this point.

        Args:
            gcp_config: GCP platform settings (project_id, region, zones, etc.)
            label_prefix: Prefix for GCP resource labels
            bootstrap_config: Worker bootstrap settings
            timeout_config: VM lifecycle timeouts
        """
        self._platform = gcp_config
        self._label_prefix = label_prefix
        self._bootstrap = bootstrap_config
        self._timeouts = timeout_config

    def vm_ops(self) -> PlatformOps:
        return _GcpPlatformOps(self._platform, self._label_prefix)

    def tunnel(
        self,
        controller_address: str,
        local_port: int | None = None,
        timeout: float | None = None,
        tunnel_logger: logging.Logger | None = None,
    ) -> AbstractContextManager[str]:
        """Create SSH tunnel to GCP controller VM.

        Discovers the controller VM via labels and establishes port forwarding.
        """
        from iris.cluster.vm.debug import controller_tunnel

        zone = self._platform.zone or (self._platform.default_zones[0] if self._platform.default_zones else "")
        kwargs: dict = {
            "zone": zone,
            "project": self._platform.project_id,
            "label_prefix": self._label_prefix,
        }
        if local_port is not None:
            kwargs["local_port"] = local_port
        if timeout is not None:
            kwargs["timeout"] = timeout
        if tunnel_logger is not None:
            kwargs["tunnel_logger"] = tunnel_logger
        return controller_tunnel(**kwargs)

    def vm_manager(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        vm_factory: TrackedVmFactory,
        *,
        dry_run: bool = False,
    ) -> VmManagerProtocol:
        if group_config.vm_type == config_pb2.VM_TYPE_TPU_VM:
            config_copy = config_pb2.ScaleGroupConfig()
            config_copy.CopyFrom(group_config)
            if not config_copy.zones:
                config_copy.zones.extend(_resolve_zones(group_config, self._platform))
            return TpuVmManager(  # type: ignore[return-value]
                project_id=self._platform.project_id,
                config=config_copy,
                bootstrap_config=self._bootstrap,
                timeouts=self._timeouts,
                vm_factory=vm_factory,
                label_prefix=self._label_prefix,
                dry_run=dry_run,
            )
        if group_config.vm_type == config_pb2.VM_TYPE_GCE_VM:
            raise NotImplementedError("VM_TYPE_GCE_VM is not implemented yet")
        raise ValueError(f"Unsupported vm_type for GCP platform: {group_config.vm_type}")


class ManualPlatform:
    def __init__(
        self,
        label_prefix: str,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeout_config: config_pb2.TimeoutConfig,
        ssh_config: config_pb2.SshConfig,
    ):
        """Create manual platform with explicit config sections.

        Args:
            label_prefix: Prefix for resource labels
            bootstrap_config: Worker bootstrap settings
            timeout_config: VM lifecycle timeouts
            ssh_config: SSH connection settings
        """
        self._bootstrap = bootstrap_config
        self._timeouts = timeout_config
        self._label_prefix = label_prefix
        self._ssh = _ssh_config_to_dataclass(ssh_config)

    def vm_ops(self) -> PlatformOps:
        return _ManualPlatformOps(self._ssh, self._bootstrap)

    def tunnel(
        self,
        controller_address: str,
        local_port: int | None = None,
        timeout: float | None = None,
        tunnel_logger: logging.Logger | None = None,
    ) -> AbstractContextManager[str]:
        """Return direct connection for manual platform (no tunnel needed)."""
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


class LocalPlatform:
    """Platform for local testing (no cloud resources)."""

    def vm_ops(self) -> PlatformOps:
        return _LocalPlatformOps()

    def vm_manager(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        vm_factory: TrackedVmFactory,
        *,
        dry_run: bool = False,
    ) -> VmManagerProtocol:
        raise NotImplementedError("Local platform uses LocalController autoscaler")

    def tunnel(
        self,
        controller_address: str,
        local_port: int | None = None,
        timeout: float | None = None,
        tunnel_logger: logging.Logger | None = None,
    ) -> AbstractContextManager[str]:
        """Return direct connection for local platform (no tunnel needed)."""
        return nullcontext(controller_address)


def create_platform(
    platform_config: config_pb2.PlatformConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
    ssh_config: config_pb2.SshConfig,
) -> Platform:
    """Create platform from explicit config sections.

    Args:
        platform_config: Platform type and settings (gcp/manual/local)
        bootstrap_config: Worker bootstrap settings
        timeout_config: VM lifecycle timeouts
        ssh_config: SSH connection settings

    Returns:
        Platform instance for the configured platform type

    Raises:
        ValueError: If platform type is unspecified or invalid
    """
    if not platform_config.HasField("platform"):
        raise ValueError("platform is required")

    which = platform_config.WhichOneof("platform")

    if which == "gcp":
        if not platform_config.gcp.project_id:
            raise ValueError("platform.gcp.project_id is required")
        return GcpPlatform(
            gcp_config=platform_config.gcp,
            label_prefix=platform_config.label_prefix or "iris",
            bootstrap_config=bootstrap_config,
            timeout_config=timeout_config,
        )

    if which == "manual":
        return ManualPlatform(
            label_prefix=platform_config.label_prefix or "iris",
            bootstrap_config=bootstrap_config,
            timeout_config=timeout_config,
            ssh_config=ssh_config,
        )

    if which == "local":
        return LocalPlatform()

    raise ValueError(f"Unknown platform: {which}")


def _resolve_zones(
    group_config: config_pb2.ScaleGroupConfig,
    platform: config_pb2.GcpPlatformConfig,
) -> list[str]:
    if group_config.zones:
        return list(group_config.zones)
    if platform.default_zones:
        return list(platform.default_zones)
    if platform.zone:
        return [platform.zone]
    raise ValueError(f"No zones configured for scale group {group_config.name}")


def _extract_node_name(resource_name: str) -> str:
    if "/" in resource_name:
        return resource_name.split("/")[-1]
    return resource_name


def _gcloud_list_tpus(project_id: str, zone: str, label_filter: str) -> list[dict]:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "list",
        f"--zone={zone}",
        f"--project={project_id}",
        "--format=json",
        f"--filter={label_filter}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to list TPUs in zone %s: %s", zone, result.stderr.strip())
        return []
    if not result.stdout.strip():
        return []
    tpus = json.loads(result.stdout)
    for tpu in tpus:
        tpu["name"] = _extract_node_name(tpu.get("name", ""))
    return tpus


def _gcloud_delete_tpu(project_id: str, zone: str, name: str) -> bool:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "delete",
        name,
        f"--project={project_id}",
        f"--zone={zone}",
        "--quiet",
        "--async",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to delete TPU %s in zone %s: %s", name, zone, result.stderr.strip())
        return False
    return True


def _direct_ssh(ssh_config: SshConfig, host: str) -> DirectSshConnection:
    return DirectSshConnection(
        host=host,
        user=ssh_config.user,
        port=ssh_config.port,
        key_file=ssh_config.key_file or None,
        connect_timeout=ssh_config.connect_timeout,
    )


def _ssh_config_to_dataclass(ssh: config_pb2.SshConfig) -> SshConfig:
    """Convert proto SshConfig to dataclass SshConfig.

    Args:
        ssh: Proto SSH configuration (should have defaults already applied)

    Returns:
        SshConfig dataclass for use by SSH connections
    """
    from iris.cluster.vm.config import DEFAULT_CONFIG, DEFAULT_SSH_PORT

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
