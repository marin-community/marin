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

"""Environment probing for worker registration."""

import logging
import os
import socket
import subprocess
import urllib.request
from pathlib import Path
from typing import Protocol

from iris.cluster.types import get_tpu_topology
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)


def _probe_gpu_info() -> tuple[int, str, int]:
    """Probe GPU info via nvidia-smi.

    Returns (0, "", 0) if no GPU.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            return 0, "", 0

        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return 0, "", 0

        # Parse first GPU for name/memory, count all GPUs
        first_line = lines[0].split(", ")
        gpu_name = first_line[0].strip() if first_line else ""
        gpu_memory_mb = int(first_line[1].strip()) if len(first_line) > 1 else 0
        return len(lines), gpu_name, gpu_memory_mb
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        logger.debug("GPU probe failed (nvidia-smi not available or error): %s", type(e).__name__)
        return 0, "", 0


def _probe_gce_metadata() -> tuple[str, str]:
    """Query GCE metadata server.

    Returns ("", "") if not on GCE.
    """
    try:
        headers = {"Metadata-Flavor": "Google"}

        def fetch(path: str) -> str:
            req = urllib.request.Request(
                f"http://169.254.169.254/computeMetadata/v1/{path}",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.read().decode().strip()

        instance_name = fetch("instance/name")
        zone_full = fetch("instance/zone")
        zone = zone_full.split("/")[-1] if zone_full else ""
        return instance_name, zone
    except Exception:
        logger.debug("GCE metadata probe failed (not running on GCE or metadata server unavailable)")
        return "", ""


def _get_memory_total_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # kB to bytes
    except FileNotFoundError:
        pass
    # Fallback for non-Linux
    return 8 * 1024**3  # Default 8GB


def _get_ip_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _get_disk_bytes() -> int:
    """Get available disk space in bytes."""
    try:
        stat = os.statvfs("/")
        return stat.f_bavail * stat.f_frsize
    except Exception:
        return 100 * 1024**3  # Default 100GB


def collect_workdir_size_mb(workdir: Path) -> int:
    """Calculate workdir size in MB using du -sm."""
    if not workdir.exists():
        return 0

    result = subprocess.run(
        ["du", "-sm", str(workdir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return 0

    # du -sm output format: "SIZE\tPATH"
    output = result.stdout.strip()
    size_str = output.split("\t")[0]

    return int(size_str)


def _get_extra_attributes() -> dict[str, str]:
    """Get extra worker attributes from IRIS_WORKER_ATTRIBUTES env var.

    Format: key1=value1,key2=value2,...
    Example: taint:maintenance=true,pool=large-jobs

    Values are always strings; the caller is responsible for type conversion if needed.
    """
    attrs_env = os.environ.get("IRIS_WORKER_ATTRIBUTES", "")
    if not attrs_env:
        return {}
    result: dict[str, str] = {}
    for pair in attrs_env.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            logger.warning("Skipping malformed attribute (no '='): %s", pair)
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result


def _build_worker_attributes(
    tpu_name: str,
    tpu_worker_id: str,
    device: cluster_pb2.DeviceConfig,
    extra_attributes: dict[str, str],
) -> dict[str, cluster_pb2.AttributeValue]:
    """Build worker attributes for constraint-based scheduling.

    Populates standard attributes from the TPU environment:
    - tpu-name: TPU slice name
    - tpu-worker-id: Worker ID within the slice (0-indexed)
    - tpu-topology: TPU topology variant (e.g., "v5litepod-16")
    - tpu-vm-count: Number of VMs in the TPU slice

    Also merges in extra_attributes from IRIS_WORKER_ATTRIBUTES env var.
    Extra attributes are treated as strings.
    """
    attributes: dict[str, cluster_pb2.AttributeValue] = {}

    if tpu_name:
        attributes["tpu-name"] = cluster_pb2.AttributeValue(string_value=tpu_name)
        attributes["tpu-worker-id"] = cluster_pb2.AttributeValue(int_value=int(tpu_worker_id) if tpu_worker_id else 0)

        # Extract topology from device config if available
        if device.HasField("tpu") and device.tpu.variant:
            tpu_variant = device.tpu.variant
            attributes["tpu-topology"] = cluster_pb2.AttributeValue(string_value=tpu_variant)

            # Look up VM count from topology
            try:
                topo = get_tpu_topology(tpu_variant)
                attributes["tpu-vm-count"] = cluster_pb2.AttributeValue(int_value=topo.vm_count)
            except ValueError:
                # Unknown topology - don't add vm-count attribute
                logger.warning("Unknown TPU topology: %s", tpu_variant)

    # Add extra attributes from environment
    for key, value in extra_attributes.items():
        attributes[key] = cluster_pb2.AttributeValue(string_value=value)

    return attributes


class EnvironmentProvider(Protocol):
    """Protocol for worker environment probing."""

    def probe(self) -> cluster_pb2.WorkerMetadata: ...


class DefaultEnvironmentProvider:
    """Default implementation that probes real system resources."""

    def probe(self) -> cluster_pb2.WorkerMetadata:
        hostname = socket.gethostname()
        ip_address = _get_ip_address()
        cpu_count = os.cpu_count() or 1
        memory_bytes = _get_memory_total_bytes()
        disk_bytes = _get_disk_bytes()

        # TPU environment variables
        tpu_name = os.environ.get("TPU_NAME", "")
        tpu_worker_hostnames = os.environ.get("TPU_WORKER_HOSTNAMES", "")
        tpu_worker_id = os.environ.get("TPU_WORKER_ID", "")
        tpu_chips_per_host_bounds = os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS", "")

        # GPU info via nvidia-smi
        gpu_count, gpu_name, gpu_memory_mb = _probe_gpu_info()

        # GCE metadata
        gce_instance_name, gce_zone = _probe_gce_metadata()

        # Build device config
        device = cluster_pb2.DeviceConfig()
        if tpu_name:
            tpu_chip_count = 0
            try:
                topo = get_tpu_topology(tpu_name)
                tpu_chip_count = topo.chips_per_vm
            except ValueError:
                logger.warning("Unknown TPU topology: %s", tpu_name)

            device.tpu.CopyFrom(
                cluster_pb2.TpuDevice(
                    variant=tpu_name,
                    count=tpu_chip_count,
                )
            )
        elif gpu_count > 0:
            device.gpu.CopyFrom(
                cluster_pb2.GpuDevice(
                    variant=gpu_name or "auto",
                    count=gpu_count,
                )
            )
        else:
            device.cpu.CopyFrom(cluster_pb2.CpuDevice(variant="cpu"))

        # Get extra worker attributes from environment
        extra_attributes = _get_extra_attributes()

        memory_gb = memory_bytes // (1024**3)
        logger.info(
            "Worker environment: hostname=%s ip=%s cpu=%d memory=%dGB gpu=%d tpu=%s extra_attributes=%s",
            hostname,
            ip_address,
            cpu_count,
            memory_gb,
            gpu_count,
            tpu_name or "none",
            extra_attributes or "none",
        )

        # Build worker attributes for constraint-based scheduling
        attributes = _build_worker_attributes(tpu_name, tpu_worker_id, device, extra_attributes)

        # VM address from environment (injected by ManagedVm bootstrap)
        vm_address = os.environ.get("IRIS_VM_ADDRESS", "")

        return cluster_pb2.WorkerMetadata(
            hostname=hostname,
            ip_address=ip_address,
            cpu_count=cpu_count,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            tpu_name=tpu_name,
            tpu_worker_hostnames=tpu_worker_hostnames,
            tpu_worker_id=tpu_worker_id,
            tpu_chips_per_host_bounds=tpu_chips_per_host_bounds,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory_mb,
            gce_instance_name=gce_instance_name,
            gce_zone=gce_zone,
            device=device,
            attributes=attributes,
            vm_address=vm_address,
        )
