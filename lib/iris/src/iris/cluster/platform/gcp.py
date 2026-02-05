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

"""GCP platform implementation for controller VMs and TPU worker slices."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager

from iris.cluster.platform.base import SliceHandle, SliceVmTarget, VmBootstrapSpec
from iris.cluster.platform.bootstrap import (
    DEFAULT_BOOT_DISK_SIZE_GB,
    DEFAULT_CONTROLLER_PORT,
    DEFAULT_MACHINE_TYPE,
    wait_healthy_via_ssh,
)
from iris.cluster.platform.ssh import GceSshConnection, GcloudSshConnection, run_streaming_with_retry
from iris.cluster.types import get_tpu_topology
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)


def controller_metadata_key(label_prefix: str) -> str:
    """Metadata key to mark a VM as the controller for a given prefix."""
    return f"iris-controller-{label_prefix}"


def controller_address_metadata_key(label_prefix: str) -> str:
    """Metadata key for the controller address for a given prefix."""
    return f"iris-controller-address-{label_prefix}"


class TpuSliceHandle:
    """Provider slice handle for a TPU pod."""

    def __init__(
        self,
        slice_id: str,
        scale_group: str,
        zone: str,
        project_id: str,
        accelerator_variant: str,
        labels: dict[str, str],
        discovery_preamble: str,
        addresses: list[str] | None = None,
        created_at: Timestamp | None = None,
    ):
        self._slice_id = slice_id
        self._scale_group = scale_group
        self._zone = zone
        self._project_id = project_id
        self._accelerator_variant = accelerator_variant
        self._labels = labels
        self._discovery_preamble = discovery_preamble
        self._addresses = addresses or []
        self._created_at = created_at if created_at is not None else Timestamp.now()

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def labels(self) -> Mapping[str, str]:
        return self._labels

    @property
    def discovery_preamble(self) -> str:
        return self._discovery_preamble

    def vm_targets(self) -> list[SliceVmTarget]:
        vm_count = get_tpu_topology(self._accelerator_variant).vm_count
        targets: list[SliceVmTarget] = []
        for i in range(vm_count):
            conn = GcloudSshConnection(
                project_id=self._project_id,
                _zone=self._zone,
                vm_id=self._slice_id,
                worker_index=i,
            )
            address = self._addresses[i] if i < len(self._addresses) else None
            targets.append(
                SliceVmTarget(
                    vm_id=f"{self._slice_id}-worker-{i}",
                    zone=self._zone,
                    conn=conn,
                    address=address,
                )
            )
        return targets

    def describe(self) -> vm_pb2.SliceInfo:
        vms = []
        for i in range(get_tpu_topology(self._accelerator_variant).vm_count):
            address = self._addresses[i] if i < len(self._addresses) else ""
            vms.append(
                vm_pb2.VmInfo(
                    vm_id=f"{self._slice_id}-worker-{i}",
                    slice_id=self._slice_id,
                    scale_group=self._scale_group,
                    state=vm_pb2.VM_STATE_BOOTING,
                    address=address,
                    zone=self._zone,
                    created_at=self._created_at.to_proto(),
                    labels=dict(self._labels),
                )
            )
        return vm_pb2.SliceInfo(
            slice_id=self._slice_id,
            scale_group=self._scale_group,
            created_at=self._created_at.to_proto(),
            vms=vms,
        )

    def terminate(self) -> None:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            self._slice_id,
            f"--zone={self._zone}",
            f"--project={self._project_id}",
            "--quiet",
        ]
        logger.info("Terminating TPU: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to delete TPU %s: %s", self._slice_id, result.stderr.strip())


class GcpPlatform:
    def __init__(
        self,
        gcp_config: config_pb2.GcpPlatformConfig,
        label_prefix: str,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeout_config: config_pb2.TimeoutConfig,
    ):
        """Create GCP platform with explicit config sections."""
        self._platform = gcp_config
        self._label_prefix = label_prefix
        self._bootstrap = bootstrap_config
        self._timeouts = timeout_config

    def tunnel(
        self,
        controller_address: str,
        local_port: int | None = None,
        timeout: float | None = None,
        tunnel_logger: logging.Logger | None = None,
    ) -> AbstractContextManager[str]:
        """Create SSH tunnel to GCP controller VM."""
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

    def list_vms(self, *, tag: str | None = None, zone: str | None = None) -> list[vm_pb2.VmInfo]:
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

    def start_vms(self, spec: VmBootstrapSpec, *, zone: str | None = None) -> list[vm_pb2.VmInfo]:
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
            vm_pb2.VmInfo(
                vm_id=vm_name,
                address=address,
                zone=resolved_zone,
                labels=dict(spec.labels),
                state=vm_pb2.VM_STATE_READY,
                created_at=Timestamp.now().to_proto(),
            )
        ]

    def stop_vms(self, ids: list[str], *, zone: str | None = None) -> None:
        resolved_zone = zone or _resolve_controller_zone(self._platform)
        if not resolved_zone:
            raise RuntimeError("platform.gcp.zone or platform.gcp.default_zones is required")
        for vm_id in ids:
            _gcloud_delete_instance(self._platform.project_id, resolved_zone, vm_id)

    def list_slices(
        self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None
    ) -> list[vm_pb2.SliceInfo]:
        handles = self.discover_slices(group_config, zone=zone)
        return [handle.describe() for handle in handles]

    def create_slice(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        *,
        tags: dict[str, str] | None = None,
        zone: str | None = None,
    ) -> SliceHandle:
        if group_config.vm_type != config_pb2.VM_TYPE_TPU_VM:
            raise ValueError(f"Unsupported vm_type for GCP platform: {group_config.vm_type}")

        resolved_zone = _resolve_zone(group_config, self._platform, zone)
        slice_id = f"{self._label_prefix}-{group_config.name}-{Timestamp.now().epoch_ms()}"

        labels = {
            f"{self._label_prefix}-managed": "true",
            f"{self._label_prefix}-scale-group": group_config.name,
            f"{self._label_prefix}-slice-id": slice_id,
        }
        labels.update(tags or {})

        _gcloud_create_tpu(
            project_id=self._platform.project_id,
            zone=resolved_zone,
            name=slice_id,
            accelerator_type=group_config.accelerator_variant,
            runtime_version=group_config.runtime_version,
            labels=labels,
            preemptible=group_config.preemptible,
        )

        return TpuSliceHandle(
            slice_id=slice_id,
            scale_group=group_config.name,
            zone=resolved_zone,
            project_id=self._platform.project_id,
            accelerator_variant=group_config.accelerator_variant,
            labels=labels,
            discovery_preamble=self._discovery_preamble(),
            addresses=None,
        )

    def discover_slices(
        self, group_config: config_pb2.ScaleGroupConfig, *, zone: str | None = None
    ) -> list[SliceHandle]:
        if group_config.vm_type != config_pb2.VM_TYPE_TPU_VM:
            raise ValueError(f"Unsupported vm_type for GCP platform: {group_config.vm_type}")

        zones = [zone] if zone else _resolve_zones(group_config, self._platform)
        label_filter = f"labels.{self._label_prefix}-scale-group={group_config.name}"
        handles: list[SliceHandle] = []
        for zone_name in zones:
            for tpu_data in _gcloud_list_tpus(self._platform.project_id, zone_name, label_filter):
                state = tpu_data.get("state", "UNKNOWN")
                if state not in ("READY", "CREATING"):
                    continue
                addresses = [ep.get("ipAddress") for ep in tpu_data.get("networkEndpoints", [])]
                handles.append(
                    TpuSliceHandle(
                        slice_id=tpu_data["name"],
                        scale_group=group_config.name,
                        zone=zone_name,
                        project_id=self._platform.project_id,
                        accelerator_variant=group_config.accelerator_variant,
                        labels=tpu_data.get("labels", {}),
                        discovery_preamble=self._discovery_preamble(),
                        addresses=addresses,
                    )
                )
        return handles

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

    def _discovery_preamble(self) -> str:
        meta_key = controller_metadata_key(self._label_prefix)
        addr_key = controller_address_metadata_key(self._label_prefix)
        return f"""
# Discover controller from GCP instance metadata (prefix: {self._label_prefix})
CONTROLLER_ADDRESS=$(gcloud compute instances list \\
    --project={self._platform.project_id} \\
    --filter=\"metadata.items.{meta_key}=true AND status=RUNNING\" \\
    --format=\"value(metadata.items.filter(key:{addr_key}).firstof(value))\" \\
    --limit=1)

if [ -z \"$CONTROLLER_ADDRESS\" ]; then
    echo \"[iris-init] ERROR: Could not discover controller via GCP metadata (prefix: {self._label_prefix})\"
    exit 1
fi
echo \"[iris-init] Discovered controller at $CONTROLLER_ADDRESS\"
"""


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


def _gcloud_create_tpu(
    project_id: str,
    zone: str,
    name: str,
    accelerator_type: str,
    runtime_version: str,
    labels: Mapping[str, str],
    preemptible: bool,
) -> None:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "create",
        name,
        f"--zone={zone}",
        f"--project={project_id}",
        f"--accelerator-type={accelerator_type}",
        f"--version={runtime_version}",
        "--labels",
        ",".join(f"{k}={v}" for k, v in labels.items()),
    ]
    if preemptible:
        cmd.append("--preemptible")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create TPU: {result.stderr}")


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


def _instance_to_vminfo(instance: dict, *, addr_key: str | None, default_port: int | None) -> vm_pb2.VmInfo:
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

    return vm_pb2.VmInfo(
        vm_id=instance.get("name", ""),
        address=address,
        zone=instance.get("zone"),
        labels=instance.get("labels", {}),
        state=state,
        created_at=Timestamp.now().to_proto(),
    )


def _map_instance_status(status: str) -> vm_pb2.VmState:
    if status in ("PROVISIONING", "STAGING", "STARTING"):
        return vm_pb2.VM_STATE_BOOTING
    if status == "RUNNING":
        return vm_pb2.VM_STATE_READY
    if status in ("STOPPING", "TERMINATED"):
        return vm_pb2.VM_STATE_TERMINATED
    return vm_pb2.VM_STATE_FAILED
