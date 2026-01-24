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

"""GCP TPU platform implementation for VM management.

This module provides:
- TpuVmManager: Factory for creating TPU VM groups via gcloud
- TpuVmGroup: VM group implementation for TPU pods

Multi-host TPUs (e.g., v5p-16, v5p-32) create a single TPU pod with
multiple workers, each accessed via a different worker index.
"""

from __future__ import annotations

import json
import logging
import subprocess

from iris.cluster.types import get_tpu_topology
from iris.cluster.vm.managed_vm import ManagedVm, VmFactory, VmRegistry
from iris.cluster.vm.vm_platform import VmGroupStatus, VmSnapshot
from iris.cluster.vm.ssh import GcloudSshConnection
from iris.rpc import vm_pb2
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)

# Metadata keys for controller discovery (used by workers to find controller)
CONTROLLER_METADATA_KEY = "iris-controller"
CONTROLLER_ADDRESS_METADATA_KEY = "iris-controller-address"


class TpuVmGroup:
    """A TPU VM group with lifecycle management.

    Represents a TPU pod (potentially multi-host) that can be managed
    as an atomic unit. The group owns its ManagedVm instances and
    coordinates their lifecycle.
    """

    def __init__(
        self,
        group_id: str,
        scale_group: str,
        zone: str,
        project_id: str,
        vms: list[ManagedVm],
        vm_registry: VmRegistry,
        created_at_ms: int | None = None,
    ):
        self._group_id = group_id
        self._scale_group = scale_group
        self._zone = zone
        self._project_id = project_id
        self._vms = vms
        self._vm_registry = vm_registry
        self._created_at_ms = created_at_ms if created_at_ms is not None else now_ms()

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
        return self._created_at_ms

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
            created_at_ms=self._created_at_ms,
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
        config: vm_pb2.ScaleGroupConfig,
        bootstrap_config: vm_pb2.BootstrapConfig,
        timeouts: vm_pb2.TimeoutConfig,
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

        Creates the TPU pod via gcloud, then creates ManagedVm instances
        for each worker in the pod.
        """
        group_id = f"{self._label_prefix}-{self._config.name}-{now_ms()}"

        labels = {
            f"{self._label_prefix}-managed": "true",
            f"{self._label_prefix}-scale-group": self._config.name,
            f"{self._label_prefix}-slice-id": group_id,
        }
        labels.update(tags or {})

        logger.info(
            "Creating TPU VM group %s (type=%s, zone=%s, dry_run=%s)",
            group_id,
            self._config.accelerator_type,
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
        """
        return f"""
# Discover controller from GCP instance metadata
CONTROLLER_ADDRESS=$(gcloud compute instances list \\
    --project={self._project_id} \\
    --filter="metadata.items.{CONTROLLER_METADATA_KEY}=true AND status=RUNNING" \\
    --format="value(metadata.items.filter(key:{CONTROLLER_ADDRESS_METADATA_KEY}).firstof(value))" \\
    --limit=1)

if [ -z "$CONTROLLER_ADDRESS" ]; then
    echo "[iris-init] ERROR: Could not discover controller via GCP metadata"
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
        """Create a TpuVmGroup with ManagedVm instances for each worker."""
        zone = zone or self._zone
        vm_count = get_tpu_topology(self._config.accelerator_type).vm_count

        # GCP workers discover controller via GCP instance metadata
        discovery_preamble = self._get_discovery_preamble()

        vms: list[ManagedVm] = []
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
            f"--accelerator-type={self._config.accelerator_type}",
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
