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

"""GCP Platform implementation.

Implements the Platform protocol for Google Cloud Platform, providing:
- GcpPlatform: Creates/lists VMs and TPU slices via Python API
- GcpSliceHandle: Manages a TPU pod (list workers, terminate, status)
- GcpVmHandle: SSH to a TPU worker VM via gcloud
- GcpStandaloneVmHandle: SSH to a GCE instance with terminate/label/metadata support

All GCP API operations use google-cloud-compute and google-cloud-tpu Python clients.
SSH operations still use gcloud CLI as there is no Python API equivalent.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol

from iris.cluster.platform._vm_base import SshVmBase
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudVmState,
    PlatformError,
    QuotaExhaustedError,
    SliceStatus,
    VmStatus,
)
from iris.cluster.platform.ssh import (
    GceSshConnection,
    GcloudSshConnection,
)
from iris.cluster.types import get_tpu_topology
from iris.rpc import config_pb2
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)

# GCP TPU state mapping
_TPU_STATE_MAP: dict[str, CloudSliceState] = {
    "CREATING": CloudSliceState.CREATING,
    "READY": CloudSliceState.READY,
    "REPAIRING": CloudSliceState.REPAIRING,
    "DELETING": CloudSliceState.DELETING,
}


def _extract_node_name(resource_name: str) -> str:
    """Extract node name from GCP resource path.

    GCP returns 'projects/proj/locations/zone/nodes/my-tpu'
    but we want just 'my-tpu'.
    """
    if "/" in resource_name:
        return resource_name.split("/")[-1]
    return resource_name


def _parse_tpu_created_at(create_time: str) -> Timestamp:
    """Parse createTime from GCP TPU into a Timestamp."""
    if not create_time:
        return Timestamp.now()
    # GCP returns ISO 8601 format like "2024-01-15T10:30:00.000Z"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
        epoch_ms = int(dt.timestamp() * 1000)
        return Timestamp.from_epoch_ms(epoch_ms)
    except (ValueError, AttributeError):
        return Timestamp.now()


def _validate_slice_config(config: config_pb2.SliceConfig) -> None:
    """Validate required fields on a SliceConfig before creating a TPU.

    Raises ValueError listing all missing fields so operators can fix config
    in one pass rather than discovering issues one-by-one.
    """
    missing: list[str] = []
    if not config.accelerator_variant:
        missing.append("accelerator_variant")
    if not config.gcp.zone:
        missing.append("gcp.zone")
    if not config.gcp.runtime_version:
        missing.append("gcp.runtime_version")
    if missing:
        raise ValueError(f"SliceConfig is missing required fields: {', '.join(missing)}")


def _validate_vm_config(config: config_pb2.VmConfig) -> None:
    """Validate required fields on a VmConfig before creating a GCE instance."""
    missing: list[str] = []
    if not config.name:
        missing.append("name")
    if not config.gcp.zone:
        missing.append("gcp.zone")
    if missing:
        raise ValueError(f"VmConfig is missing required fields: {', '.join(missing)}")


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Find a free port on the given host by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 30.0) -> bool:
    """Wait for a port to become available on the given host."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


# ============================================================================
# GCP API Protocols and Implementations
# ============================================================================


class TpuApi(Protocol):
    """Protocol for GCP TPU operations."""

    def get_node(self, project: str, zone: str, name: str) -> dict: ...

    def create_node(
        self,
        project: str,
        zone: str,
        node_id: str,
        accelerator_type: str,
        runtime_version: str,
        labels: dict[str, str] | None = None,
        preemptible: bool = False,
    ) -> None: ...

    def delete_node(self, project: str, zone: str, name: str) -> None: ...

    def list_nodes(self, project: str, zone: str) -> list[dict]: ...


class ComputeApi(Protocol):
    """Protocol for GCP Compute Engine operations."""

    def get_instance(self, project: str, zone: str, instance: str) -> dict: ...

    def create_instance(
        self,
        project: str,
        zone: str,
        instance_name: str,
        machine_type: str,
        boot_disk_size_gb: int,
        labels: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict: ...

    def delete_instance(self, project: str, zone: str, instance: str) -> None: ...

    def list_instances(self, project: str, zone: str) -> list[dict]: ...

    def reset_instance(self, project: str, zone: str, instance: str) -> None: ...

    def set_labels(self, project: str, zone: str, instance: str, labels: dict[str, str]) -> None: ...

    def set_metadata(self, project: str, zone: str, instance: str, metadata: dict[str, str]) -> None: ...


class RealTpuApi:
    """Real implementation of TpuApi using google-cloud-tpu."""

    def __init__(self):
        from google.cloud import tpu_v2

        self._client = tpu_v2.TpuClient()

    def get_node(self, project: str, zone: str, name: str) -> dict:
        """Get TPU node details."""
        from google.api_core import exceptions

        try:
            node_name = f"projects/{project}/locations/{zone}/nodes/{name}"
            node = self._client.get_node(name=node_name)
            return self._node_to_dict(node)
        except exceptions.NotFound:
            raise PlatformError(f"TPU node {name} not found in zone {zone}")

    def create_node(
        self,
        project: str,
        zone: str,
        node_id: str,
        accelerator_type: str,
        runtime_version: str,
        labels: dict[str, str] | None = None,
        preemptible: bool = False,
    ) -> None:
        """Create a TPU node."""
        from google.api_core import exceptions
        from google.cloud import tpu_v2

        parent = f"projects/{project}/locations/{zone}"

        node = tpu_v2.Node()
        node.accelerator_type = accelerator_type
        node.runtime_version = runtime_version
        if labels:
            node.labels.update(labels)

        scheduling_config = tpu_v2.SchedulingConfig()
        scheduling_config.preemptible = preemptible
        node.scheduling_config = scheduling_config

        request = tpu_v2.CreateNodeRequest(
            parent=parent,
            node_id=node_id,
            node=node,
        )

        try:
            operation = self._client.create_node(request=request)
            # Wait for operation to complete (this is a Long-Running Operation)
            operation.result(timeout=600)
        except exceptions.ResourceExhausted as e:
            raise QuotaExhaustedError(str(e))
        except Exception as e:
            raise PlatformError(f"Failed to create TPU node {node_id}: {e}")

    def delete_node(self, project: str, zone: str, name: str) -> None:
        """Delete a TPU node."""
        from google.api_core import exceptions

        try:
            node_name = f"projects/{project}/locations/{zone}/nodes/{name}"
            operation = self._client.delete_node(name=node_name)
            # Wait for operation to complete
            operation.result(timeout=600)
        except exceptions.NotFound:
            logger.info("TPU node %s not found (already deleted?)", name)
        except Exception as e:
            raise PlatformError(f"Failed to delete TPU node {name}: {e}")

    def list_nodes(self, project: str, zone: str) -> list[dict]:
        """List TPU nodes in a zone."""
        parent = f"projects/{project}/locations/{zone}"
        nodes = []
        try:
            for node in self._client.list_nodes(parent=parent):
                nodes.append(self._node_to_dict(node))
        except Exception as e:
            logger.warning("Failed to list TPU nodes in zone %s: %s", zone, e)
        return nodes

    def _node_to_dict(self, node) -> dict:
        """Convert TPU Node proto to dict matching gcloud JSON format."""
        from google.cloud import tpu_v2

        state_map = {
            tpu_v2.Node.State.CREATING: "CREATING",
            tpu_v2.Node.State.READY: "READY",
            tpu_v2.Node.State.REPAIRING: "REPAIRING",
            tpu_v2.Node.State.DELETING: "DELETING",
        }

        endpoints = []
        for endpoint in node.network_endpoints:
            ep_dict = {"ipAddress": endpoint.ip_address}
            if endpoint.access_config and endpoint.access_config.external_ip:
                ep_dict["accessConfig"] = {"externalIp": endpoint.access_config.external_ip}
            endpoints.append(ep_dict)

        return {
            "name": _extract_node_name(node.name),
            "state": state_map.get(node.state, "UNKNOWN"),
            "acceleratorType": node.accelerator_type,
            "labels": dict(node.labels),
            "networkEndpoints": endpoints,
            "createTime": node.create_time.isoformat() if node.create_time else "",
        }


class RealComputeApi:
    """Real implementation of ComputeApi using google-cloud-compute."""

    def __init__(self):
        from google.cloud import compute_v1

        self._client = compute_v1.InstancesClient()

    def get_instance(self, project: str, zone: str, instance: str) -> dict:
        """Get instance details."""
        from google.api_core import exceptions

        try:
            inst = self._client.get(project=project, zone=zone, instance=instance)
            return self._instance_to_dict(inst)
        except exceptions.NotFound:
            raise PlatformError(f"Instance {instance} not found in zone {zone}")

    def create_instance(
        self,
        project: str,
        zone: str,
        instance_name: str,
        machine_type: str,
        boot_disk_size_gb: int,
        labels: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict:
        """Create a GCE instance."""
        from google.api_core import exceptions
        from google.cloud import compute_v1

        instance_resource = compute_v1.Instance()
        instance_resource.name = instance_name
        instance_resource.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

        # Boot disk
        disk = compute_v1.AttachedDisk()
        disk.auto_delete = True
        disk.boot = True
        disk.initialize_params = compute_v1.AttachedDiskInitializeParams()
        disk.initialize_params.source_image = "projects/debian-cloud/global/images/family/debian-12"
        disk.initialize_params.disk_size_gb = boot_disk_size_gb
        instance_resource.disks = [disk]

        # Network interface
        network_interface = compute_v1.NetworkInterface()
        network_interface.name = "global/networks/default"
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        access_config.type_ = "ONE_TO_ONE_NAT"
        network_interface.access_configs = [access_config]
        instance_resource.network_interfaces = [network_interface]

        # Service account for cloud-platform scope
        service_account = compute_v1.ServiceAccount()
        service_account.email = "default"
        service_account.scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        instance_resource.service_accounts = [service_account]

        if labels:
            instance_resource.labels = labels

        if metadata:
            metadata_items = [
                compute_v1.Items(key=k, value=v)
                for k, v in metadata.items()
            ]
            instance_resource.metadata = compute_v1.Metadata(items=metadata_items)

        request = compute_v1.InsertInstanceRequest(
            project=project,
            zone=zone,
            instance_resource=instance_resource,
        )

        try:
            operation = self._client.insert(request=request)
            # Wait for operation to complete
            operation.result(timeout=600)
            # Return the created instance
            return self.get_instance(project, zone, instance_name)
        except exceptions.ResourceExhausted as e:
            raise QuotaExhaustedError(str(e))
        except exceptions.AlreadyExists:
            logger.info("Instance %s already exists, fetching details", instance_name)
            return self.get_instance(project, zone, instance_name)
        except Exception as e:
            raise PlatformError(f"Failed to create instance {instance_name}: {e}")

    def delete_instance(self, project: str, zone: str, instance: str) -> None:
        """Delete a GCE instance."""
        from google.api_core import exceptions

        try:
            operation = self._client.delete(project=project, zone=zone, instance=instance)
            operation.result(timeout=600)
        except exceptions.NotFound:
            logger.info("Instance %s not found (already deleted?)", instance)
        except Exception as e:
            logger.warning("Failed to delete instance %s: %s", instance, e)

    def list_instances(self, project: str, zone: str) -> list[dict]:
        """List instances in a zone."""
        instances = []
        try:
            for inst in self._client.list(project=project, zone=zone):
                instances.append(self._instance_to_dict(inst))
        except Exception as e:
            logger.warning("Failed to list instances in zone %s: %s", zone, e)
        return instances

    def reset_instance(self, project: str, zone: str, instance: str) -> None:
        """Reset (reboot) an instance."""
        try:
            operation = self._client.reset(project=project, zone=zone, instance=instance)
            operation.result(timeout=600)
        except Exception as e:
            raise PlatformError(f"Failed to reset instance {instance}: {e}")

    def set_labels(self, project: str, zone: str, instance: str, labels: dict[str, str]) -> None:
        """Set labels on an instance."""
        from google.cloud import compute_v1

        try:
            inst = self._client.get(project=project, zone=zone, instance=instance)
            request = compute_v1.SetLabelsInstanceRequest(
                project=project,
                zone=zone,
                instance=instance,
                instances_set_labels_request_resource=compute_v1.InstancesSetLabelsRequest(
                    labels=labels,
                    label_fingerprint=inst.label_fingerprint,
                ),
            )
            operation = self._client.set_labels(request=request)
            operation.result(timeout=600)
        except Exception as e:
            logger.warning("Failed to set labels on instance %s: %s", instance, e)

    def set_metadata(self, project: str, zone: str, instance: str, metadata: dict[str, str]) -> None:
        """Set metadata on an instance."""
        from google.cloud import compute_v1

        try:
            inst = self._client.get(project=project, zone=zone, instance=instance)
            existing_metadata = inst.metadata
            if not existing_metadata:
                existing_metadata = compute_v1.Metadata()

            # Merge with existing metadata
            metadata_dict = {item.key: item.value for item in (existing_metadata.items or [])}
            metadata_dict.update(metadata)

            metadata_items = [
                compute_v1.Items(key=k, value=v)
                for k, v in metadata_dict.items()
            ]

            request = compute_v1.SetMetadataInstanceRequest(
                project=project,
                zone=zone,
                instance=instance,
                metadata_resource=compute_v1.Metadata(
                    items=metadata_items,
                    fingerprint=existing_metadata.fingerprint,
                ),
            )
            operation = self._client.set_metadata(request=request)
            operation.result(timeout=600)
        except Exception as e:
            logger.warning("Failed to set metadata on instance %s: %s", instance, e)

    def _instance_to_dict(self, inst) -> dict:
        """Convert Instance proto to dict matching gcloud JSON format."""
        network_interfaces = []
        for ni in inst.network_interfaces:
            ni_dict = {"networkIP": ni.network_i_p}
            access_configs = []
            for ac in ni.access_configs:
                access_configs.append({"natIP": ac.nat_i_p})
            if access_configs:
                ni_dict["accessConfigs"] = access_configs
            network_interfaces.append(ni_dict)

        return {
            "name": inst.name,
            "status": inst.status,
            "networkInterfaces": network_interfaces,
            "labels": dict(inst.labels) if inst.labels else {},
            "metadata": {item.key: item.value for item in (inst.metadata.items or [])} if inst.metadata else {},
        }


# ============================================================================
# Handle Implementations
# ============================================================================


@dataclass
class GcpVmHandle(SshVmBase):
    """Handle to a TPU worker VM within a slice.

    Uses GcloudSshConnection for SSH via `gcloud compute tpus tpu-vm ssh`.
    Thread-safe: each run_command() spawns a new SSH process.
    """

    def status(self) -> VmStatus:
        # TPU worker VMs don't have independent status queries;
        # their status is derived from the slice status.
        return VmStatus(state=CloudVmState.RUNNING)


@dataclass
class GcpStandaloneVmHandle(SshVmBase):
    """Handle to a standalone GCE instance (e.g., controller VM).

    Uses GceSshConnection for SSH via `gcloud compute ssh`.
    Supports terminate, set_labels, and set_metadata operations.
    """

    _zone: str = ""
    _project_id: str = ""
    _compute_api: ComputeApi | None = None

    def status(self) -> VmStatus:
        if not self._compute_api:
            return VmStatus(state=CloudVmState.UNKNOWN)
        try:
            inst = self._compute_api.get_instance(self._project_id, self._zone, self._vm_id)
            status_str = inst.get("status", "UNKNOWN").upper()
            state_map = {
                "RUNNING": CloudVmState.RUNNING,
                "STOPPED": CloudVmState.STOPPED,
                "TERMINATED": CloudVmState.TERMINATED,
            }
            return VmStatus(state=state_map.get(status_str, CloudVmState.UNKNOWN))
        except Exception:
            return VmStatus(state=CloudVmState.UNKNOWN)

    def reboot(self) -> None:
        if not self._compute_api:
            raise PlatformError("No compute API configured for reboot")
        logger.info("Rebooting GCE instance: %s", self._vm_id)
        self._compute_api.reset_instance(self._project_id, self._zone, self._vm_id)

    def terminate(self) -> None:
        if not self._compute_api:
            raise PlatformError("No compute API configured for terminate")
        logger.info("Deleting GCE instance: %s", self._vm_id)
        self._compute_api.delete_instance(self._project_id, self._zone, self._vm_id)

    def set_labels(self, labels: dict[str, str]) -> None:
        if not self._compute_api:
            raise PlatformError("No compute API configured for set_labels")
        logger.info("Setting labels on GCE instance: %s", self._vm_id)
        self._compute_api.set_labels(self._project_id, self._zone, self._vm_id, labels)

    def set_metadata(self, metadata: dict[str, str]) -> None:
        if not self._compute_api:
            raise PlatformError("No compute API configured for set_metadata")
        logger.info("Setting metadata on GCE instance: %s", self._vm_id)
        self._compute_api.set_metadata(self._project_id, self._zone, self._vm_id, metadata)


class GcpSliceHandle:
    """Handle to a GCP TPU slice (pod).

    describe() queries TPU state and VM endpoints via TPU API.
    The slice is the atomic unit for termination.
    """

    def __init__(
        self,
        *,
        _slice_id: str,
        _zone: str,
        _project_id: str,
        _labels: dict[str, str],
        _created_at: Timestamp,
        _label_prefix: str,
        _accelerator_variant: str,
        _tpu_api: TpuApi,
        _ssh_config: config_pb2.SshConfig | None = None,
        _state: str = "READY",
    ):
        self._slice_id = _slice_id
        self._zone = _zone
        self._project_id = _project_id
        self._labels = _labels
        self._created_at = _created_at
        self._label_prefix = _label_prefix
        self._accelerator_variant = _accelerator_variant
        self._tpu_api = _tpu_api
        self._ssh_config = _ssh_config
        self._state = _state

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return self._zone

    @property
    def scale_group(self) -> str:
        return self._labels.get(f"{self._label_prefix}-scale-group", "")

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def describe(self) -> SliceStatus:
        """Query TPU state and VM endpoints."""
        try:
            tpu_data = self._tpu_api.get_node(self._project_id, self._zone, self._slice_id)
        except Exception as e:
            logger.warning("Failed to describe TPU %s: %s", self._slice_id, e)
            return SliceStatus(state=CloudSliceState.UNKNOWN, vm_count=0)

        state = _TPU_STATE_MAP.get(tpu_data.get("state", "UNKNOWN"), CloudSliceState.UNKNOWN)
        endpoints = tpu_data.get("networkEndpoints", [])

        try:
            vm_count = get_tpu_topology(self._accelerator_variant).vm_count
        except ValueError as e:
            raise PlatformError(
                f"Unknown TPU topology '{self._accelerator_variant}' for slice {self._slice_id}. "
                f"Cannot determine VM count without a known topology."
            ) from e

        vms: list[GcpVmHandle] = []
        for i in range(vm_count):
            ep = endpoints[i] if i < len(endpoints) else {}
            internal_ip = ep.get("ipAddress", "")
            external_ip = ep.get("accessConfig", {}).get("externalIp") if "accessConfig" in ep else None

            if not internal_ip and i < len(endpoints):
                logger.warning(
                    "TPU %s endpoint %d has no IP address; VM may still be provisioning",
                    self._slice_id,
                    i,
                )

            ssh = GcloudSshConnection(
                project_id=self._project_id,
                _zone=self._zone,
                vm_id=self._slice_id,
                worker_index=i,
                _address=internal_ip,
            )
            vms.append(
                GcpVmHandle(
                    _vm_id=f"{self._slice_id}-worker-{i}",
                    _internal_address=internal_ip,
                    _external_address=external_ip,
                    _ssh=ssh,
                )
            )

        return SliceStatus(state=state, vm_count=vm_count, vms=vms)

    def terminate(self) -> None:
        logger.info("Terminating TPU: %s", self._slice_id)
        self._tpu_api.delete_node(self._project_id, self._zone, self._slice_id)


# ============================================================================
# GcpPlatform
# ============================================================================

DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE_GB = 50


class GcpPlatform:
    """Platform implementation for Google Cloud Platform.

    Manages GCE instances (standalone VMs) and TPU slices via Python API.
    Zones are stored from GcpPlatformConfig for list_all_slices(); per-slice
    zones come from SliceConfig.
    """

    def __init__(
        self,
        gcp_config: config_pb2.GcpPlatformConfig,
        label_prefix: str,
        ssh_config: config_pb2.SshConfig | None = None,
        tpu_api: TpuApi | None = None,
        compute_api: ComputeApi | None = None,
    ):
        self._project_id = gcp_config.project_id
        self._label_prefix = label_prefix
        self._ssh_config = ssh_config
        self._zones = list(gcp_config.zones)
        self._tpu_api = tpu_api or RealTpuApi()
        self._compute_api = compute_api or RealComputeApi()

    def create_vm(self, config: config_pb2.VmConfig) -> GcpStandaloneVmHandle:
        """Create a GCE instance. Returns a handle with SSH and label/metadata support."""
        _validate_vm_config(config)
        gcp = config.gcp
        zone = gcp.zone
        machine_type = gcp.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = gcp.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB

        logger.info("Creating GCE instance: %s (zone=%s, type=%s)", config.name, zone, machine_type)

        inst_data = self._compute_api.create_instance(
            project=self._project_id,
            zone=zone,
            instance_name=config.name,
            machine_type=machine_type,
            boot_disk_size_gb=boot_disk_size,
            labels=dict(config.labels) if config.labels else None,
            metadata=dict(config.metadata) if config.metadata else None,
        )

        # Extract internal/external IP
        network_interfaces = inst_data.get("networkInterfaces", [])
        internal_ip = ""
        external_ip = None
        if network_interfaces:
            internal_ip = network_interfaces[0].get("networkIP", "")
            access_configs = network_interfaces[0].get("accessConfigs", [])
            if access_configs:
                external_ip = access_configs[0].get("natIP")

        ssh = GceSshConnection(
            project_id=self._project_id,
            zone=zone,
            vm_name=config.name,
        )

        return GcpStandaloneVmHandle(
            _vm_id=config.name,
            _internal_address=internal_ip,
            _external_address=external_ip,
            _zone=zone,
            _project_id=self._project_id,
            _compute_api=self._compute_api,
            _ssh=ssh,
        )

    def create_slice(self, config: config_pb2.SliceConfig) -> GcpSliceHandle:
        """Create a TPU slice via Python API."""
        _validate_slice_config(config)
        gcp = config.gcp
        slice_id = f"{config.name_prefix}-{Timestamp.now().epoch_ms()}"

        logger.info("Creating TPU slice: %s (type=%s, zone=%s)", slice_id, config.accelerator_variant, gcp.zone)

        self._tpu_api.create_node(
            project=self._project_id,
            zone=gcp.zone,
            node_id=slice_id,
            accelerator_type=config.accelerator_variant,
            runtime_version=gcp.runtime_version,
            labels=dict(config.labels) if config.labels else None,
            preemptible=config.preemptible,
        )

        return GcpSliceHandle(
            _slice_id=slice_id,
            _zone=gcp.zone,
            _project_id=self._project_id,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _accelerator_variant=config.accelerator_variant,
            _tpu_api=self._tpu_api,
            _ssh_config=self._ssh_config,
        )

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpSliceHandle]:
        """List TPU slices across zones, optionally filtered by labels."""
        results: list[GcpSliceHandle] = []
        for zone in zones:
            nodes = self._tpu_api.list_nodes(self._project_id, zone)
            for tpu_data in nodes:
                state = tpu_data.get("state", "UNKNOWN")
                if state not in ("READY", "CREATING"):
                    logger.info("Skipping TPU %s in state %s", tpu_data["name"], state)
                    continue

                tpu_labels = tpu_data.get("labels", {})
                # Filter by labels if provided
                if labels and not all(tpu_labels.get(k) == v for k, v in labels.items()):
                    continue

                accelerator_type = tpu_data.get("acceleratorType", "")
                # acceleratorType can be a full path or just the type name
                if "/" in accelerator_type:
                    accelerator_type = accelerator_type.split("/")[-1]

                results.append(
                    GcpSliceHandle(
                        _slice_id=tpu_data["name"],
                        _zone=zone,
                        _project_id=self._project_id,
                        _labels=tpu_labels,
                        _created_at=_parse_tpu_created_at(tpu_data.get("createTime", "")),
                        _label_prefix=self._label_prefix,
                        _accelerator_variant=accelerator_type,
                        _tpu_api=self._tpu_api,
                        _ssh_config=self._ssh_config,
                        _state=state,
                    )
                )

        return results

    def list_all_slices(self, labels: dict[str, str] | None = None) -> list[GcpSliceHandle]:
        if not self._zones:
            raise ValueError(
                "GcpPlatform.list_all_slices() called but no zones configured. "
                "Set platform.gcp.zones in your cluster config."
            )
        return self.list_slices(zones=self._zones, labels=labels)

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpStandaloneVmHandle]:
        """List GCE instances across zones, optionally filtered by labels."""
        results: list[GcpStandaloneVmHandle] = []
        for zone in zones:
            instances = self._compute_api.list_instances(self._project_id, zone)
            for instance in instances:
                name = instance.get("name", "")
                inst_labels = instance.get("labels", {})

                # Filter by labels if provided
                if labels and not all(inst_labels.get(k) == v for k, v in labels.items()):
                    continue

                network_interfaces = instance.get("networkInterfaces", [])
                internal_ip = ""
                external_ip = None
                if network_interfaces:
                    internal_ip = network_interfaces[0].get("networkIP", "")
                    access_configs = network_interfaces[0].get("accessConfigs", [])
                    if access_configs:
                        external_ip = access_configs[0].get("natIP")

                ssh = GceSshConnection(
                    project_id=self._project_id,
                    zone=zone,
                    vm_name=name,
                )
                results.append(
                    GcpStandaloneVmHandle(
                        _vm_id=name,
                        _internal_address=internal_ip,
                        _external_address=external_ip,
                        _zone=zone,
                        _project_id=self._project_id,
                        _compute_api=self._compute_api,
                        _ssh=ssh,
                    )
                )

        return results

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        return _gcp_tunnel(
            project=self._project_id,
            label_prefix=self._label_prefix,
            compute_api=self._compute_api,
            local_port=local_port,
        )

    def shutdown(self) -> None:
        pass

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Discover controller by querying GCP for labeled controller VM."""
        gcp = controller_config.gcp
        port = gcp.port or 10000
        label_key = f"{self._label_prefix}-controller"

        vms = self.list_vms(
            zones=[gcp.zone],
            labels={label_key: "true"},
        )
        if not vms:
            raise RuntimeError(f"No controller VM found (label={label_key}=true, project={self._project_id})")
        return f"{vms[0].internal_address}:{port}"


# ============================================================================
# Tunnel
# ============================================================================


@contextmanager
def _gcp_tunnel(
    project: str,
    label_prefix: str,
    compute_api: ComputeApi,
    local_port: int | None = None,
    timeout: float = 60.0,
) -> Iterator[str]:
    """SSH tunnel to the controller VM, yielding the local URL.

    Binds explicitly to 127.0.0.1 to avoid conflicts with other processes
    that may be listening on the same port on a different address family (IPv6).
    Picks a free port automatically if none is specified.

    Note: Still uses gcloud CLI for the actual SSH tunnel since there's no
    Python API equivalent for SSH tunneling.
    """
    if local_port is None:
        local_port = _find_free_port()

    # Find controller VM using Python API
    label_filter = f"labels.{label_prefix}-controller=true AND status=RUNNING"
    # For tunnel discovery, we need to use gcloud since we don't know the zone
    # and aggregated_list is complex. Keep this part as gcloud for now.
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter={label_filter}",
        "--format=value(name,zone)",
        "--limit=1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"No controller VM found (label={label_prefix}-controller=true, project={project})")

    parts = result.stdout.strip().split()
    vm_name = parts[0]
    # Zone comes as a full path like us-central2-b
    zone = parts[1] if len(parts) > 1 else ""

    logger.info("Establishing SSH tunnel to %s (zone=%s)...", vm_name, zone)

    proc = subprocess.Popen(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--",
            "-L",
            f"127.0.0.1:{local_port}:localhost:10000",
            "-N",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    try:
        if not _wait_for_port(local_port, host="127.0.0.1", timeout=timeout):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait()
            raise RuntimeError(f"SSH tunnel failed to establish: {stderr}")

        logger.info("Tunnel ready: 127.0.0.1:%d -> %s:10000", local_port, vm_name)
        yield f"http://127.0.0.1:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
