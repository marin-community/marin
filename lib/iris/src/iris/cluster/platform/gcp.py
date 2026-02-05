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

from iris.cluster.platform.base import VmBootstrapSpec, VmInfo, VmState
from iris.cluster.platform.bootstrap import (
    DEFAULT_BOOT_DISK_SIZE_GB,
    DEFAULT_CONTROLLER_PORT,
    DEFAULT_MACHINE_TYPE,
    wait_healthy_via_ssh,
)
from iris.cluster.platform.ssh import GceSshConnection, GcloudSshConnection, run_streaming_with_retry
from iris.cluster.platform.vm_platform import VmGroupStatus, VmManagerProtocol, VmSnapshot
from iris.cluster.platform.worker_vm import TrackedVmFactory, WorkerVm, VmFactory, VmRegistry
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


class TpuVmGroup:
    """A TPU VM group with lifecycle management.

    Represents a TPU pod (potentially multi-host) that can be managed
    as an atomic unit. The group owns its WorkerVm instances and
    coordinates their lifecycle.
    """

    def __init__(
        self,
        group_id: str,
        scale_group: str,
        zone: str,
        project_id: str,
        vms: list[WorkerVm],
        vm_registry: VmRegistry,
        created_at: Timestamp | None = None,
    ):
        self._group_id = group_id
        self._scale_group = scale_group
        self._zone = zone
        self._project_id = project_id
        self._vms = vms
        self._vm_registry = vm_registry
        self._created_at = created_at if created_at is not None else Timestamp.now()

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
        """Terminate this VM group and unregister VMs.

        Performs three steps:
        1. Stop all VM lifecycle threads
        2. Unregister VMs from the registry
        3. Delete TPU via gcloud command
        """
        for vm in self._vms:
            vm.stop()
            self._vm_registry.unregister(vm.info.vm_id)

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            self._group_id,
            f"--zone={self._zone}",
            f"--project={self._project_id}",
            "--quiet",
        ]
        logger.info("Terminating TPU: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to delete TPU %s: %s", self._group_id, result.stderr.strip())

    def to_proto(self) -> vm_pb2.SliceInfo:
        """Convert to proto for RPC APIs."""

        return vm_pb2.SliceInfo(
            slice_id=self._group_id,
            scale_group=self._scale_group,
            created_at=self._created_at.to_proto(),
            vms=[vm.info for vm in self._vms],
        )


class TpuVmManager:
    """Creates TPU VM groups via gcloud compute tpus tpu-vm.

    One instance per scale group. This is a factory - it creates VM groups
    but doesn't track them (the ScalingGroup tracks groups).

    Multi-host TPUs (e.g., v5p-16, v5p-32) create a single TPU pod with
    multiple workers, each accessed via a different worker index.
    """

    def __init__(
        self,
        project_id: str,
        config: config_pb2.ScaleGroupConfig,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeouts: config_pb2.TimeoutConfig,
        vm_factory: VmFactory,
        label_prefix: str = "iris",
        dry_run: bool = False,
    ):
        self._project_id = project_id
        self._config = config
        self._zone = config.zones[0] if config.zones else "us-central1-a"
        self._bootstrap_config = bootstrap_config
        self._timeouts = timeouts
        self._vm_factory = vm_factory
        self._label_prefix = label_prefix
        self._dry_run = dry_run

    def create_vm_group(self, tags: dict[str, str] | None = None) -> TpuVmGroup:
        """Create a new TPU VM group.

        Creates the TPU pod via gcloud, then creates WorkerVm instances
        for each worker in the pod.
        """
        group_id = f"{self._label_prefix}-{self._config.name}-{Timestamp.now().epoch_ms()}"

        labels = {
            f"{self._label_prefix}-managed": "true",
            f"{self._label_prefix}-scale-group": self._config.name,
            f"{self._label_prefix}-slice-id": group_id,
        }
        labels.update(tags or {})

        logger.info(
            "Creating TPU VM group %s (type=%s, zone=%s, dry_run=%s)",
            group_id,
            self._config.accelerator_variant,
            self._zone,
            self._dry_run,
        )

        if self._dry_run:
            logger.info("[DRY-RUN] Would create TPU: %s", group_id)
        else:
            self._gcloud_create_tpu(group_id, labels)

        return self._make_vm_group(group_id, labels, addresses=None)

    def discover_vm_groups(self) -> list[TpuVmGroup]:
        """Find existing TPU VM groups for this scale group.

        Queries GCP for TPUs with the scale group label, then creates
        TpuVmGroup objects for each discovered TPU.
        """
        groups = []
        label_filter = f"labels.{self._label_prefix}-scale-group={self._config.name}"

        for zone in self._config.zones or [self._zone]:
            for tpu_data in self._gcloud_list_tpus(zone, label_filter):
                state = tpu_data.get("state", "UNKNOWN")
                if state not in ("READY", "CREATING"):
                    logger.info(
                        "Skipping TPU %s in state %s (not adoptable)",
                        tpu_data["name"],
                        state,
                    )
                    continue

                addresses = [ep.get("ipAddress") for ep in tpu_data.get("networkEndpoints", [])]
                vm_group = self._make_vm_group(
                    group_id=tpu_data["name"],
                    labels=tpu_data.get("labels", {}),
                    addresses=addresses,
                    zone=zone,
                )
                groups.append(vm_group)
                logger.info("Discovered TPU VM group %s in zone %s", tpu_data["name"], zone)

        return groups

    def _get_discovery_preamble(self) -> str:
        """Generate GCP metadata-based discovery script for worker bootstrap.

        Workers query GCP instance metadata to find a running controller.
        Uses prefix-scoped metadata keys to ensure workers connect to the correct
        controller when multiple controllers exist with different prefixes.
        """
        meta_key = controller_metadata_key(self._label_prefix)
        addr_key = controller_address_metadata_key(self._label_prefix)
        return f"""
# Discover controller from GCP instance metadata (prefix: {self._label_prefix})
CONTROLLER_ADDRESS=$(gcloud compute instances list \\
    --project={self._project_id} \\
    --filter="metadata.items.{meta_key}=true AND status=RUNNING" \\
    --format="value(metadata.items.filter(key:{addr_key}).firstof(value))" \\
    --limit=1)

if [ -z "$CONTROLLER_ADDRESS" ]; then
    echo "[iris-init] ERROR: Could not discover controller via GCP metadata (prefix: {self._label_prefix})"
    exit 1
fi
echo "[iris-init] Discovered controller at $CONTROLLER_ADDRESS"
"""

    def _make_vm_group(
        self,
        group_id: str,
        labels: dict[str, str],
        addresses: list[str] | None,
        zone: str | None = None,
    ) -> TpuVmGroup:
        """Create a TpuVmGroup with WorkerVm instances for each worker."""
        zone = zone or self._zone
        vm_count = get_tpu_topology(self._config.accelerator_variant).vm_count

        # GCP workers discover controller via GCP instance metadata
        discovery_preamble = self._get_discovery_preamble()

        vms: list[WorkerVm] = []
        for i in range(vm_count):
            conn = GcloudSshConnection(
                project_id=self._project_id,
                _zone=zone,
                vm_id=group_id,
                worker_index=i,
            )
            address = addresses[i] if addresses and i < len(addresses) else None

            vm = self._vm_factory.create_vm(
                vm_id=f"{group_id}-worker-{i}",
                slice_id=group_id,
                scale_group=self._config.name,
                zone=zone,
                conn=conn,
                bootstrap_config=self._bootstrap_config,
                timeouts=self._timeouts,
                labels=labels,
                address=address,
                discovery_preamble=discovery_preamble,
            )
            vms.append(vm)

        return TpuVmGroup(
            group_id=group_id,
            scale_group=self._config.name,
            zone=zone,
            project_id=self._project_id,
            vms=vms,
            vm_registry=self._vm_factory.registry,
        )

    def _gcloud_create_tpu(self, group_id: str, labels: dict[str, str]) -> None:
        """Create a TPU VM via gcloud."""
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            group_id,
            f"--zone={self._zone}",
            f"--project={self._project_id}",
            f"--accelerator-type={self._config.accelerator_variant}",
            f"--version={self._config.runtime_version}",
            "--labels",
            ",".join(f"{k}={v}" for k, v in labels.items()),
        ]
        if self._config.preemptible:
            cmd.append("--preemptible")

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create TPU: {result.stderr}")

    def _gcloud_list_tpus(self, zone: str, label_filter: str) -> list[dict]:
        """List TPU VMs in a zone matching a label filter."""
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--zone={zone}",
            f"--project={self._project_id}",
            "--format=json",
            f"--filter={label_filter}",
        ]
        logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to list TPUs in zone %s: %s", zone, result.stderr.strip())
            return []
        if not result.stdout.strip():
            return []
        tpus = json.loads(result.stdout)
        for tpu in tpus:
            tpu["name"] = self._extract_node_name(tpu.get("name", ""))
        return tpus

    def _extract_node_name(self, resource_name: str) -> str:
        """Extract node name from GCP resource path.

        GCP returns 'projects/proj/locations/zone/nodes/my-tpu'
        but gcloud delete expects just 'my-tpu'.
        """
        if "/" in resource_name:
            return resource_name.split("/")[-1]
        return resource_name

    def stop(self) -> None:
        """No-op: TpuVmManager has no background threads to stop."""
        pass


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

    def vm_ops(self) -> _GcpPlatformOps:
        return _GcpPlatformOps(self._platform, self._label_prefix)

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
