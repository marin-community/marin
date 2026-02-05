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

"""Platform abstraction for VM lifecycle and direct operations."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol
from collections.abc import Mapping

from iris.cluster.platform.base import VmBootstrapSpec, VmInfo, VmState
from iris.cluster.platform.gcp_tpu_platform import (
    TpuVmManager,
    controller_address_metadata_key,
    controller_metadata_key,
)
from iris.cluster.platform.worker_vm import SshConfig, TrackedVmFactory
from iris.cluster.platform.manual_platform import ManualVmManager
from iris.cluster.platform.ssh import DirectSshConnection, GceSshConnection, run_streaming_with_retry
from iris.cluster.platform.vm_platform import VmManagerProtocol
from iris.cluster.platform.controller_vm import (
    CONTROLLER_CONTAINER_NAME,
    CONTROLLER_STOP_SCRIPT,
    DEFAULT_BOOT_DISK_SIZE_GB,
    DEFAULT_CONTROLLER_PORT,
    DEFAULT_MACHINE_TYPE,
    wait_healthy_via_ssh,
)
from iris.rpc import config_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


class PlatformOps(Protocol):
    """Direct, non-lifecycle VM operations used by CLI cleanup."""

    def list_slices(self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None) -> list[str]: ...

    def delete_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        slice_id: str,
        *,
        zone: str | None = None,
    ) -> None: ...


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

    def list_vms(self, *, tag: str | None = None, zone: str | None = None) -> list[VmInfo]:
        """List VMs for the platform."""
        ...

    def start_vms(self, spec: VmBootstrapSpec, *, zone: str | None = None) -> list[VmInfo]:
        """Start VMs using the bootstrap spec."""
        ...

    def stop_vms(self, ids: list[str], *, zone: str | None = None) -> None:
        """Stop VMs by id."""
        ...


class _GcpPlatformOps:
    def __init__(self, platform: config_pb2.GcpPlatformConfig, label_prefix: str):
        self._platform = platform
        self._label_prefix = label_prefix

    def list_slices(self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None) -> list[str]:
        resolved_zone = _resolve_zone(group_config, self._platform, zone)
        label_filter = f"labels.{self._label_prefix}-scale-group={group_config.name}"
        slices: list[str] = []
        for tpu in _gcloud_list_tpus(self._platform.project_id, resolved_zone, label_filter):
            state = tpu.get("state", "UNKNOWN")
            if state not in ("READY", "CREATING", "REPAIRING"):
                continue
            slices.append(_extract_node_name(tpu.get("name", "")))
        return slices

    def delete_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        slice_id: str,
        *,
        zone: str | None = None,
    ) -> None:
        resolved_zone = _resolve_zone(group_config, self._platform, zone)
        if not _gcloud_delete_tpu(self._platform.project_id, resolved_zone, slice_id):
            raise RuntimeError(f"Failed to delete TPU slice {slice_id} in zone {resolved_zone}")


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


class _LocalPlatformOps:
    def list_slices(self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None) -> list[str]:
        return []

    def delete_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        slice_id: str,
        *,
        zone: str | None = None,
    ) -> None:
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
        from iris.cluster.platform.debug import controller_tunnel

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

    def list_vms(self, *, tag: str | None = None, zone: str | None = None) -> list[VmInfo]:
        resolved_zone = zone or _resolve_controller_zone(self._platform)
        if not resolved_zone:
            return []
        label_filter = f"labels.{self._label_prefix}-tag={tag}" if tag else ""
        instances = _gcloud_list_instances(self._platform.project_id, resolved_zone, label_filter)
        if not instances and tag == f"{self._label_prefix}-controller":
            meta_key = controller_metadata_key(self._label_prefix)
            instances = _gcloud_list_instances(
                self._platform.project_id,
                resolved_zone,
                f"metadata.items.{meta_key}=true",
            )
        addr_key = controller_address_metadata_key(self._label_prefix)
        return [_instance_to_vminfo(instance, addr_key=addr_key, default_port=None) for instance in instances]

    def start_vms(self, spec: VmBootstrapSpec, *, zone: str | None = None) -> list[VmInfo]:
        if spec.role != "controller":
            raise NotImplementedError("GcpPlatform.start_vms currently supports controller role only")

        resolved_zone = zone or _resolve_controller_zone(self._platform)
        if not resolved_zone:
            raise RuntimeError("platform.gcp.zone or platform.gcp.default_zones is required")

        machine_type = spec.provider_overrides.get("machine_type", DEFAULT_MACHINE_TYPE)
        boot_disk_size = spec.provider_overrides.get("boot_disk_size_gb", DEFAULT_BOOT_DISK_SIZE_GB)
        port = spec.provider_overrides.get("port", DEFAULT_CONTROLLER_PORT)
        vm_name = spec.provider_overrides.get("vm_name", f"iris-controller-{self._label_prefix}")

        meta_key = controller_metadata_key(self._label_prefix)
        labels_arg = _format_labels(spec.labels)

        address = _gcloud_create_instance(
            project_id=self._platform.project_id,
            zone=resolved_zone,
            vm_name=vm_name,
            machine_type=machine_type,
            boot_disk_size_gb=boot_disk_size,
            metadata={meta_key: "true"},
            labels=labels_arg,
            port=port,
        )

        if spec.bootstrap_script is None:
            raise ValueError("bootstrap_script is required to start controller VM")

        conn = GceSshConnection(
            project_id=self._platform.project_id,
            zone=resolved_zone,
            vm_name=vm_name,
        )

        def on_line(line: str) -> None:
            logger.info("[controller %s] %s", vm_name, line)

        run_streaming_with_retry(
            conn,
            f"bash -c {shlex.quote(spec.bootstrap_script)}",
            on_line=on_line,
        )

        if port and not wait_healthy_via_ssh(conn, port):
            raise RuntimeError(f"Controller at {address} failed health check after bootstrap")

        addr_key = controller_address_metadata_key(self._label_prefix)
        _gcloud_add_metadata(self._platform.project_id, resolved_zone, vm_name, {addr_key: address})

        return [
            VmInfo(
                vm_id=vm_name,
                address=address,
                zone=resolved_zone,
                labels=dict(spec.labels),
                state=VmState.RUNNING,
                created_at_ms=Timestamp.now().epoch_ms(),
            )
        ]

    def stop_vms(self, ids: list[str], *, zone: str | None = None) -> None:
        resolved_zone = zone or _resolve_controller_zone(self._platform)
        if not resolved_zone:
            raise RuntimeError("platform.gcp.zone or platform.gcp.default_zones is required")
        for vm_id in ids:
            _gcloud_delete_instance(self._platform.project_id, resolved_zone, vm_id)


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

    def list_vms(self, *, tag: str | None = None, zone: str | None = None) -> list[VmInfo]:
        return []

    def start_vms(self, spec: VmBootstrapSpec, *, zone: str | None = None) -> list[VmInfo]:
        raise NotImplementedError("Local platform uses in-process controller runtime")

    def stop_vms(self, ids: list[str], *, zone: str | None = None) -> None:
        return None


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


def _resolve_zone(
    group_config: config_pb2.ScaleGroupConfig,
    platform: config_pb2.GcpPlatformConfig,
    zone: str | None,
) -> str:
    if zone:
        return zone
    if group_config.zones:
        if len(group_config.zones) > 1:
            raise ValueError(f"Scale group {group_config.name} has multiple zones configured; pass zone explicitly.")
        return group_config.zones[0]
    if platform.default_zones:
        if len(platform.default_zones) > 1:
            raise ValueError(f"Multiple default zones configured; pass zone explicitly for {group_config.name}.")
        return platform.default_zones[0]
    if platform.zone:
        return platform.zone
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
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to delete TPU %s in zone %s: %s", name, zone, result.stderr.strip())
        return False
    return True


def _resolve_controller_zone(platform: config_pb2.GcpPlatformConfig) -> str:
    if platform.zone:
        return platform.zone
    if platform.default_zones:
        return platform.default_zones[0]
    return ""


def _format_labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    return ",".join(f"{key}={value}" for key, value in labels.items())


def _gcloud_list_instances(project_id: str, zone: str, label_filter: str) -> list[dict]:
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project_id}",
        "--format=json",
    ]
    filters: list[str] = []
    if zone:
        filters.append(f"zone:({zone})")
    if label_filter:
        filters.append(label_filter)
    if filters:
        cmd.append(f"--filter={' AND '.join(filters)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to list instances in zone %s: %s", zone, result.stderr.strip())
        return []
    if not result.stdout.strip():
        return []
    try:
        parsed = json.loads(result.stdout)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse instance list output: %s", exc)
        return []


def _gcloud_get_instance_ip(project_id: str, zone: str, vm_name: str) -> str:
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        vm_name,
        f"--project={project_id}",
        f"--zone={zone}",
        "--format=value(networkInterfaces[0].networkIP)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get VM address: {result.stderr.strip()}")
    ip = result.stdout.strip()
    if not ip:
        raise RuntimeError("VM has no internal IP")
    return ip


def _gcloud_create_instance(
    project_id: str,
    zone: str,
    vm_name: str,
    machine_type: str,
    boot_disk_size_gb: int,
    metadata: Mapping[str, str],
    labels: str,
    port: int,
) -> str:
    meta_flag = ",".join(f"{key}={value}" for key, value in metadata.items())
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        vm_name,
        f"--project={project_id}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--boot-disk-size={boot_disk_size_gb}GB",
        "--image-family=debian-12",
        "--image-project=debian-cloud",
        "--scopes=cloud-platform",
        f"--metadata={meta_flag}",
        "--format=json",
    ]
    if labels:
        cmd.append(f"--labels={labels}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        if "already exists" in error_msg.lower():
            logger.info("VM %s already exists (%.1fs), reusing", vm_name, elapsed)
            ip = _gcloud_get_instance_ip(project_id, zone, vm_name)
            return f"http://{ip}:{port}"
        raise RuntimeError(f"Failed to create VM: {error_msg}")

    try:
        parsed = json.loads(result.stdout)
        vm_data: dict = parsed[0] if isinstance(parsed, list) else parsed
        network_interfaces = vm_data.get("networkInterfaces", [])
        if network_interfaces:
            ip = network_interfaces[0].get("networkIP")
            if ip:
                return f"http://{ip}:{port}"
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse VM creation output: %s", exc)

    ip = _gcloud_get_instance_ip(project_id, zone, vm_name)
    return f"http://{ip}:{port}"


def _gcloud_add_metadata(project_id: str, zone: str, vm_name: str, metadata: Mapping[str, str]) -> None:
    meta_flag = ",".join(f"{key}={value}" for key, value in metadata.items())
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "add-metadata",
        vm_name,
        f"--project={project_id}",
        f"--zone={zone}",
        f"--metadata={meta_flag}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to tag VM metadata: %s", result.stderr.strip())


def _gcloud_delete_instance(project_id: str, zone: str, vm_name: str) -> None:
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "delete",
        vm_name,
        f"--project={project_id}",
        f"--zone={zone}",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error = result.stderr.strip()
        if "not found" not in error.lower():
            logger.warning("Failed to delete VM %s: %s", vm_name, error)


def _instance_to_vminfo(instance: dict, *, addr_key: str | None, default_port: int | None) -> VmInfo:
    address = ""
    if addr_key:
        for item in instance.get("metadata", {}).get("items", []):
            if item.get("key") == addr_key:
                address = item.get("value", "")
                break

    if not address:
        network_interfaces = instance.get("networkInterfaces", [])
        if network_interfaces:
            ip = network_interfaces[0].get("networkIP", "")
            if default_port is not None and ip:
                address = f"http://{ip}:{default_port}"
            else:
                address = ip

    status = instance.get("status", "")
    state = _map_instance_status(status)

    return VmInfo(
        vm_id=instance.get("name", ""),
        address=address,
        zone=instance.get("zone"),
        labels=instance.get("labels", {}),
        state=state,
        created_at_ms=Timestamp.now().epoch_ms(),
    )


def _map_instance_status(status: str) -> VmState:
    if status in ("PROVISIONING", "STAGING", "STARTING"):
        return VmState.BOOTING
    if status == "RUNNING":
        return VmState.RUNNING
    if status in ("STOPPING", "TERMINATED"):
        return VmState.TERMINATED
    return VmState.FAILED


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
