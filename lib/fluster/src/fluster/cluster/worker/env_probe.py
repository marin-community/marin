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

"""Environment probing for worker registration.

Probes the worker environment to collect metadata for registration with the controller:
- Hostname and IP address
- TPU environment variables (TPU_NAME, TPU_WORKER_HOSTNAMES, etc.)
- GPU info via nvidia-smi
- GCE instance metadata (instance name, zone)
- System resources (CPU count, memory)
"""

import logging
import os
import socket
import subprocess
import urllib.request

from fluster import cluster_pb2

logger = logging.getLogger(__name__)


def _probe_gpu_info() -> tuple[int, str, int]:
    """Probe GPU info via nvidia-smi.

    Returns:
        Tuple of (gpu_count, gpu_name, gpu_memory_mb). Returns (0, "", 0) if no GPU.
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

    Returns:
        Tuple of (instance_name, zone). Returns ("", "") if not on GCE.
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
    """Get total system memory in bytes."""
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
    """Get the worker's IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def probe_worker_environment() -> cluster_pb2.WorkerMetadata:
    """Probe the current environment for worker registration.

    Returns:
        WorkerMetadata proto with all detected environment information.
    """
    # Basic info
    hostname = socket.gethostname()
    ip_address = _get_ip_address()
    cpu_count = os.cpu_count() or 1
    memory_bytes = _get_memory_total_bytes()

    # TPU environment variables
    tpu_name = os.environ.get("TPU_NAME", "")
    tpu_worker_hostnames = os.environ.get("TPU_WORKER_HOSTNAMES", "")
    tpu_worker_id = os.environ.get("TPU_WORKER_ID", "")
    tpu_chips_per_host_bounds = os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS", "")

    # GPU info via nvidia-smi
    gpu_count, gpu_name, gpu_memory_mb = _probe_gpu_info()

    # GCE metadata
    gce_instance_name, gce_zone = _probe_gce_metadata()

    memory_gb = memory_bytes // (1024**3)
    logger.info(
        "Worker environment: hostname=%s ip=%s cpu=%d memory=%dGB gpu=%d tpu=%s",
        hostname,
        ip_address,
        cpu_count,
        memory_gb,
        gpu_count,
        tpu_name or "none",
    )

    return cluster_pb2.WorkerMetadata(
        hostname=hostname,
        ip_address=ip_address,
        cpu_count=cpu_count,
        memory_bytes=memory_bytes,
        tpu_name=tpu_name,
        tpu_worker_hostnames=tpu_worker_hostnames,
        tpu_worker_id=tpu_worker_id,
        tpu_chips_per_host_bounds=tpu_chips_per_host_bounds,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        gce_instance_name=gce_instance_name,
        gce_zone=gce_zone,
    )


def build_resource_spec(metadata: cluster_pb2.WorkerMetadata) -> cluster_pb2.ResourceSpec:
    """Build a ResourceSpec proto from worker metadata.

    Args:
        metadata: WorkerMetadata from probe_worker_environment()

    Returns:
        A ResourceSpec with detected resources.
    """
    memory_gb = metadata.memory_bytes // (1024**3)

    resources = cluster_pb2.ResourceSpec(
        cpu=metadata.cpu_count,
        memory=f"{memory_gb}g",
    )

    # Add TPU device if detected
    if metadata.tpu_name:
        resources.device.CopyFrom(cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant=metadata.tpu_name)))
    # Add GPU device if detected
    elif metadata.gpu_count > 0:
        resources.device.CopyFrom(
            cluster_pb2.DeviceConfig(
                gpu=cluster_pb2.GpuDevice(
                    variant=metadata.gpu_name or "auto",
                    count=metadata.gpu_count,
                )
            )
        )

    return resources
