# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLOUD-mode implementation of GcpService.

Extracts all gcloud subprocess calls from GcpPlatform into typed service methods
that return TpuInfo/VmInfo dataclasses. DRY_RUN and LOCAL modes are stubs for now.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime

from iris.cluster.platform.base import PlatformError, QuotaExhaustedError
from iris.cluster.platform.gcp_service import (
    TpuCreateRequest,
    TpuInfo,
    VmCreateRequest,
    VmInfo,
)
from iris.cluster.platform.service_mode import ServiceMode
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)


def _format_labels(labels: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in labels.items())


def _build_label_filter(labels: dict[str, str]) -> str:
    parts = [f"labels.{k}={v}" for k, v in labels.items()]
    return " AND ".join(parts)


def _classify_gcloud_error(stderr: str) -> PlatformError:
    lower = stderr.lower()
    if "quota" in lower or "insufficient" in lower or "resource_exhausted" in lower:
        return QuotaExhaustedError(stderr)
    return PlatformError(stderr)


def _extract_node_name(resource_name: str) -> str:
    if "/" in resource_name:
        return resource_name.split("/")[-1]
    return resource_name


def _parse_tpu_created_at(tpu_data: dict) -> Timestamp:
    create_time = tpu_data.get("createTime", "")
    if not create_time:
        return Timestamp.now()
    try:
        dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
        epoch_ms = int(dt.timestamp() * 1000)
        return Timestamp.from_ms(epoch_ms)
    except (ValueError, AttributeError):
        return Timestamp.now()


def _parse_vm_created_at(vm_data: dict) -> Timestamp:
    create_time = vm_data.get("creationTimestamp", "")
    if not create_time:
        return Timestamp.now()
    try:
        dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
        epoch_ms = int(dt.timestamp() * 1000)
        return Timestamp.from_ms(epoch_ms)
    except (ValueError, AttributeError):
        return Timestamp.now()


def _parse_tpu_info(tpu_data: dict, zone: str) -> TpuInfo:
    """Parse raw GCP TPU JSON into a TpuInfo dataclass."""
    name = _extract_node_name(tpu_data.get("name", ""))

    accelerator_type = tpu_data.get("acceleratorType", "")
    if "/" in accelerator_type:
        accelerator_type = accelerator_type.split("/")[-1]

    endpoints = tpu_data.get("networkEndpoints", [])
    ips = [ep.get("ipAddress", "") for ep in endpoints if ep.get("ipAddress")]

    return TpuInfo(
        name=name,
        state=tpu_data.get("state", "UNKNOWN"),
        accelerator_type=accelerator_type,
        zone=zone,
        labels=tpu_data.get("labels", {}),
        metadata=tpu_data.get("metadata", {}),
        network_endpoints=ips,
        created_at=_parse_tpu_created_at(tpu_data),
    )


def _parse_vm_info(vm_data: dict, fallback_zone: str = "") -> VmInfo:
    """Parse raw GCP VM JSON into a VmInfo dataclass."""
    zone_url = vm_data.get("zone", "")
    zone = zone_url.split("/")[-1] if zone_url else fallback_zone

    network_interfaces = vm_data.get("networkInterfaces", [])
    internal_ip = ""
    external_ip = None
    if network_interfaces:
        internal_ip = network_interfaces[0].get("networkIP", "")
        access_configs = network_interfaces[0].get("accessConfigs", [])
        if access_configs:
            external_ip = access_configs[0].get("natIP")

    # Metadata in GCP JSON is {"items": [{"key": ..., "value": ...}]}
    raw_metadata = vm_data.get("metadata", {})
    metadata: dict[str, str] = {}
    if isinstance(raw_metadata, dict):
        for item in raw_metadata.get("items", []):
            metadata[item["key"]] = item.get("value", "")

    return VmInfo(
        name=vm_data.get("name", ""),
        status=vm_data.get("status", "UNKNOWN"),
        zone=zone,
        internal_ip=internal_ip,
        external_ip=external_ip,
        labels=vm_data.get("labels", {}),
        metadata=metadata,
    )


class GcpServiceImpl:
    """GcpService implementation driven by ServiceMode.

    CLOUD mode shells out to gcloud CLI, faithfully extracted from GcpPlatform.
    DRY_RUN and LOCAL modes raise NotImplementedError (added in later tasks).
    """

    def __init__(self, mode: ServiceMode, project_id: str) -> None:
        self._mode = mode
        self._project_id = project_id

    @property
    def mode(self) -> ServiceMode:
        return self._mode

    @property
    def project_id(self) -> str:
        return self._project_id

    def _require_cloud(self) -> None:
        if self._mode != ServiceMode.CLOUD:
            raise NotImplementedError(f"GcpServiceImpl does not yet support mode={self._mode}")

    # ========================================================================
    # TPU operations
    # ========================================================================

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo:
        self._require_cloud()

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            request.name,
            f"--zone={request.zone}",
            f"--project={self._project_id}",
            f"--accelerator-type={request.accelerator_type}",
            f"--version={request.runtime_version}",
            "--format=json",
        ]

        if request.labels:
            cmd.extend(["--labels", _format_labels(request.labels)])
        if request.preemptible:
            cmd.append("--preemptible")
        if request.service_account:
            cmd.append(f"--service-account={request.service_account}")
        if request.network:
            cmd.append(f"--network={request.network}")
        if request.subnetwork:
            cmd.append(f"--subnetwork={request.subnetwork}")

        logger.info("Creating TPU: %s (type=%s, zone=%s)", request.name, request.accelerator_type, request.zone)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise _classify_gcloud_error(result.stderr.strip())

        # gcloud tpu create returns JSON for the created resource
        if result.stdout.strip():
            tpu_data = json.loads(result.stdout)
            return _parse_tpu_info(tpu_data, request.zone)

        # Fallback: describe the TPU we just created
        info = self.tpu_describe(request.name, request.zone)
        if info is None:
            raise PlatformError(f"TPU {request.name} created but could not be described")
        return info

    def tpu_delete(self, name: str, zone: str) -> None:
        self._require_cloud()

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            name,
            f"--zone={zone}",
            f"--project={self._project_id}",
            "--quiet",
            "--async",
        ]
        logger.info("Deleting TPU (async): %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip()
            if "not found" not in error.lower():
                raise _classify_gcloud_error(error)

    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None:
        self._require_cloud()

        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            name,
            f"--zone={zone}",
            f"--project={self._project_id}",
            "--format=json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip().lower()
            if "not found" in error or "could not be found" in error:
                return None
            logger.warning("Failed to describe TPU %s: %s", name, result.stderr.strip())
            return None

        tpu_data = json.loads(result.stdout)
        return _parse_tpu_info(tpu_data, zone)

    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]:
        self._require_cloud()

        results: list[TpuInfo] = []
        for zone in zones:
            cmd = [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "list",
                f"--zone={zone}",
                f"--project={self._project_id}",
                "--format=json",
            ]
            if labels:
                cmd.append(f"--filter={_build_label_filter(labels)}")

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("Failed to list TPUs in zone %s: %s", zone, result.stderr.strip())
                continue
            if not result.stdout.strip():
                continue

            for tpu_data in json.loads(result.stdout):
                results.append(_parse_tpu_info(tpu_data, zone))

        return results

    # ========================================================================
    # VM operations
    # ========================================================================

    def vm_create(self, request: VmCreateRequest) -> VmInfo:
        self._require_cloud()

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "create",
            request.name,
            f"--project={self._project_id}",
            f"--zone={request.zone}",
            f"--machine-type={request.machine_type}",
            f"--boot-disk-size={request.disk_size_gb}GB",
            f"--image-family={request.image_family}",
            f"--image-project={request.image_project}",
            "--scopes=cloud-platform",
            "--format=json",
        ]

        if request.labels:
            cmd.append(f"--labels={_format_labels(request.labels)}")
        if request.metadata:
            metadata_str = ",".join(f"{k}={v}" for k, v in request.metadata.items())
            cmd.append(f"--metadata={metadata_str}")
        if request.startup_script:
            cmd.append(f"--metadata=startup-script={request.startup_script}")
        if request.service_account:
            cmd.append(f"--service-account={request.service_account}")

        logger.info("Creating VM: %s (zone=%s, type=%s)", request.name, request.zone, request.machine_type)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "already exists" not in error_msg.lower():
                raise _classify_gcloud_error(error_msg)

        # Describe to get full info (create output may be incomplete for "already exists")
        info = self.vm_describe(request.name, request.zone)
        if info is None:
            raise PlatformError(f"VM {request.name} created but could not be described")
        return info

    def vm_delete(self, name: str, zone: str) -> None:
        self._require_cloud()

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "delete",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            "--quiet",
        ]
        logger.info("Deleting VM: %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip()
            if "not found" not in error.lower():
                raise _classify_gcloud_error(error)

    def vm_describe(self, name: str, zone: str) -> VmInfo | None:
        self._require_cloud()

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "describe",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            "--format=json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error = result.stderr.strip().lower()
            if "not found" in error or "could not be found" in error:
                return None
            logger.warning("Failed to describe VM %s: %s", name, result.stderr.strip())
            return None

        data = json.loads(result.stdout)
        return _parse_vm_info(data, fallback_zone=zone)

    def vm_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[VmInfo]:
        self._require_cloud()

        results: list[VmInfo] = []
        for zone in zones:
            cmd = [
                "gcloud",
                "compute",
                "instances",
                "list",
                f"--project={self._project_id}",
                f"--zones={zone}",
                "--format=json",
            ]
            if labels:
                cmd.append(f"--filter={_build_label_filter(labels)}")

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("Failed to list instances in zone %s: %s", zone, result.stderr.strip())
                continue
            if not result.stdout.strip():
                continue

            for vm_data in json.loads(result.stdout):
                results.append(_parse_vm_info(vm_data, fallback_zone=zone))

        return results

    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None:
        self._require_cloud()

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "update",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            f"--update-labels={_format_labels(labels)}",
        ]
        logger.info("Updating labels on VM %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise _classify_gcloud_error(result.stderr.strip())

    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None:
        self._require_cloud()

        metadata_str = ",".join(f"{k}={v}" for k, v in metadata.items())
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "add-metadata",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            f"--metadata={metadata_str}",
        ]
        logger.info("Setting metadata on VM %s", name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise _classify_gcloud_error(result.stderr.strip())

    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str:
        self._require_cloud()

        cmd = [
            "gcloud",
            "compute",
            "instances",
            "get-serial-port-output",
            name,
            f"--project={self._project_id}",
            f"--zone={zone}",
            f"--start={start}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to get serial port output for %s: %s", name, result.stderr.strip())
            return ""
        return result.stdout
