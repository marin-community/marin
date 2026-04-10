# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment probing for worker registration."""

import logging
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from iris.cluster.constraints import WellKnownAttribute, accelerator_type_to_string
from iris.cluster.types import get_tpu_topology
from iris.rpc import config_pb2
from iris.rpc import job_pb2
from iris.time_proto import timestamp_to_proto
from rigging.timing import Timestamp

logger = logging.getLogger(__name__)

_GCP_METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1/instance"
_GCP_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


@lru_cache(maxsize=1)
def _is_gcp_vm() -> bool:
    """Return True when running on a GCP VM."""
    dmi_paths = (
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/sys_vendor",
    )
    for path in dmi_paths:
        try:
            value = Path(path).read_text().strip().lower()
        except OSError:
            continue
        if "google" in value or "google compute engine" in value:
            return True
    return False


def _get_gcp_metadata(path: str) -> str | None:
    """Read GCP instance metadata path and return stripped text."""
    try:
        req = urllib.request.Request(
            f"{_GCP_METADATA_ROOT}/{path}",
            headers=_GCP_METADATA_HEADERS,
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            value = resp.read().decode().strip()
            return value or None
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None


def detect_gcp_zone() -> str | None:
    """Return the GCP zone name (e.g. 'us-central1-a') or None if not on GCP."""
    if not _is_gcp_vm():
        return None
    zone = _get_gcp_metadata("zone")
    if not zone:
        return None
    # zone format: projects/<project>/zones/us-central2-b
    zone_name = zone.split("/")[-1]
    if "-" not in zone_name:
        return None
    return zone_name


def _extract_tpu_name(instance_name: str) -> str:
    """Derive TPU slice name by stripping worker suffix from instance name."""
    return re.sub(r"-w-*[0-9]*$", "", instance_name.strip())


def _extract_tpu_worker_hostnames(worker_endpoints: str) -> str:
    """Extract comma-separated IP list from TPU worker-network-endpoints metadata."""
    hostnames: list[str] = []
    for entry in worker_endpoints.split(","):
        for field in entry.split(":"):
            candidate = field.strip()
            if "." in candidate:
                hostnames.append(candidate)
                break
    return ",".join(hostnames)


def _extract_tpu_chips_per_host_bounds(tpu_env_raw: str) -> str:
    """Extract CHIPS_PER_HOST_BOUNDS value from tpu-env metadata blob."""
    match = re.search(r"^CHIPS_PER_HOST_BOUNDS:\s*'([^']+)'", tpu_env_raw, flags=re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip()


def _probe_tpu_metadata() -> tuple[str, str, str, str, str]:
    """Probe TPU metadata from GCP metadata service.

    Returns:
        Tuple of (tpu_name, tpu_type, tpu_worker_hostnames, tpu_worker_id, tpu_chips_per_host_bounds).
    """
    if not _is_gcp_vm():
        return "", "", "", "", ""

    tpu_name = ""
    tpu_worker_hostnames = ""
    tpu_worker_id = ""
    tpu_chips_per_host_bounds = ""
    tpu_type = _get_gcp_metadata("attributes/accelerator-type") or ""

    # Only collect TPU-specific metadata once we know this is a TPU VM.
    has_tpu_signal = bool(tpu_type)
    if has_tpu_signal:
        if instance_name := _get_gcp_metadata("name"):
            tpu_name = _extract_tpu_name(instance_name)
        tpu_worker_id = _get_gcp_metadata("attributes/agent-worker-number") or ""
        if worker_endpoints := _get_gcp_metadata("attributes/worker-network-endpoints"):
            tpu_worker_hostnames = _extract_tpu_worker_hostnames(worker_endpoints)
        if tpu_env_raw := _get_gcp_metadata("attributes/tpu-env"):
            tpu_chips_per_host_bounds = _extract_tpu_chips_per_host_bounds(tpu_env_raw)

    return tpu_name, tpu_type, tpu_worker_hostnames, tpu_worker_id, tpu_chips_per_host_bounds


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


_MEMORY_HEADROOM_FRACTION = 0.10
"""Reserve up to 10% of physical RAM for OS, Docker daemon, and worker agent."""

_MEMORY_HEADROOM_MAX_BYTES = 8 * 1024**3
"""Cap memory headroom at 8 GB."""

_CPU_HEADROOM_FRACTION = 0.10
"""Reserve up to 10% of CPUs for OS and worker agent."""

_CPU_HEADROOM_MAX = 1
"""Cap CPU headroom at 1 core."""


def _compute_memory_headroom(physical_bytes: int) -> int:
    """Compute memory headroom: min(10% of physical RAM, 8 GB)."""
    return min(int(physical_bytes * _MEMORY_HEADROOM_FRACTION), _MEMORY_HEADROOM_MAX_BYTES)


def _compute_cpu_headroom(physical_cpus: int) -> int:
    """Compute CPU headroom: min(10% of CPUs, 1 core), as whole cores."""
    fractional = min(physical_cpus * _CPU_HEADROOM_FRACTION, _CPU_HEADROOM_MAX)
    return int(fractional)


def _get_memory_total_bytes() -> int:
    """Return schedulable memory in bytes, with headroom subtracted.

    Reserves min(10%, 8 GB) for the OS, Docker daemon, and worker agent so the
    scheduler cannot commit 100% of a machine's memory.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    physical = int(line.split()[1]) * 1024  # kB to bytes
                    headroom = _compute_memory_headroom(physical)
                    return physical - headroom
    except FileNotFoundError:
        pass
    # Fallback for non-Linux
    return 8 * 1024**3  # Default 8GB


def _get_cpu_count() -> int:
    """Return schedulable CPU count, with headroom subtracted.

    Reserves min(10%, 1 core) for the OS and worker agent.
    """
    physical = os.cpu_count() or 1
    headroom = _compute_cpu_headroom(physical)
    return max(1, physical - headroom)


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


def _build_worker_attributes(
    *,
    accelerator_type: int,
    accelerator_variant: str,
    capacity_type: int,
    tpu_name: str,
    tpu_worker_id: str,
    device: job_pb2.DeviceConfig,
    extra_attributes: dict[str, str],
) -> dict[str, job_pb2.AttributeValue]:
    """Build worker attributes for constraint-based scheduling.

    Scheduling-relevant attributes (device-type, device-variant, preemptible)
    come from WorkerConfig fields, which are populated by the autoscaler from
    ScaleGroupResources. Probed hardware data populates diagnostic fields on
    WorkerMetadata but does NOT influence the attributes map.

    TPU multi-host identity (tpu-name, tpu-worker-id) still comes from GCP
    metadata probes since these identify which specific VM in a TPU slice this
    worker is -- the config cannot know this.

    TPU topology and VM count are derived from device_variant when device_type
    is TPU.
    """
    attributes: dict[str, job_pb2.AttributeValue] = {}

    # Scheduling attributes from config
    device_type_str = accelerator_type_to_string(accelerator_type)
    attributes[WellKnownAttribute.DEVICE_TYPE] = job_pb2.AttributeValue(string_value=device_type_str)

    if accelerator_variant:
        attributes[WellKnownAttribute.DEVICE_VARIANT] = job_pb2.AttributeValue(string_value=accelerator_variant.lower())

    is_preemptible = capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE
    attributes[WellKnownAttribute.PREEMPTIBLE] = job_pb2.AttributeValue(string_value=str(is_preemptible).lower())

    # TPU multi-host identity from GCP metadata probes
    if tpu_name:
        attributes[WellKnownAttribute.TPU_NAME] = job_pb2.AttributeValue(string_value=tpu_name)
        attributes[WellKnownAttribute.TPU_WORKER_ID] = job_pb2.AttributeValue(
            int_value=int(tpu_worker_id) if tpu_worker_id else 0
        )

    # TPU topology attributes derived from variant
    if accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU and accelerator_variant:
        attributes[WellKnownAttribute.TPU_TOPOLOGY] = job_pb2.AttributeValue(string_value=accelerator_variant)
        try:
            topo = get_tpu_topology(accelerator_variant)
            attributes[WellKnownAttribute.TPU_VM_COUNT] = job_pb2.AttributeValue(int_value=topo.vm_count)
        except ValueError:
            logger.warning("Unknown TPU topology: %s", accelerator_variant)

    # GPU diagnostic attributes from device config (populated by build_worker_metadata)
    if device.HasField("gpu"):
        attributes[WellKnownAttribute.GPU_VARIANT] = job_pb2.AttributeValue(string_value=device.gpu.variant)
        attributes[WellKnownAttribute.GPU_COUNT] = job_pb2.AttributeValue(int_value=device.gpu.count)

    # Custom user attributes from YAML worker.attributes (merged last so they can override)
    for key, value in extra_attributes.items():
        attributes[key] = job_pb2.AttributeValue(string_value=value)

    return attributes


@dataclass
class HardwareProbe:
    """Result of probing local machine hardware. No config input needed."""

    hostname: str
    ip_address: str
    cpu_count: int
    memory_bytes: int
    disk_bytes: int
    gpu_count: int
    gpu_name: str
    gpu_memory_mb: int
    tpu_name: str
    tpu_type: str
    tpu_worker_hostnames: str
    tpu_worker_id: str
    tpu_chips_per_host_bounds: str
    gce_instance_name: str = ""


def _probe_gce_instance_name() -> str:
    """Read the GCE instance name from metadata. Empty string if not on GCP."""
    if not _is_gcp_vm():
        return ""
    return _get_gcp_metadata("name") or ""


def construct_worker_id(slice_id: str, worker_index: int) -> str:
    """Build a deterministic worker ID from slice identity and within-slice index."""
    return f"{slice_id}-worker-{worker_index}"


IRIS_WORKER_ID_ENV = "IRIS_WORKER_ID"


def infer_worker_id(hardware: HardwareProbe) -> str | None:
    """Infer worker_id from environment or GCP metadata probes.

    Priority:
    1. IRIS_WORKER_ID env var (set by all platforms via WorkerConfig or pod env).
    2. TPU metadata: combines tpu_name (the slice name) with the TPU worker index.
    3. GCE instance name: uses the instance name as slice_id with worker index 0.

    Returns None when not running on a recognized cloud VM and no env var is set.
    """
    env_worker_id = os.environ.get(IRIS_WORKER_ID_ENV)
    if env_worker_id:
        return env_worker_id
    if hardware.tpu_name:
        worker_index = int(hardware.tpu_worker_id) if hardware.tpu_worker_id else 0
        return construct_worker_id(hardware.tpu_name, worker_index)
    if hardware.gce_instance_name:
        return construct_worker_id(hardware.gce_instance_name, 0)
    return None


def probe_hardware() -> HardwareProbe:
    """Probe local machine hardware. Pure function, no config needed."""
    hostname = socket.gethostname()
    ip_address = _get_ip_address()
    tpu_name, tpu_type, tpu_worker_hostnames, tpu_worker_id, tpu_chips_per_host_bounds = _probe_tpu_metadata()
    gpu_count, gpu_name, gpu_memory_mb = _probe_gpu_info()
    gce_instance_name = _probe_gce_instance_name()
    return HardwareProbe(
        hostname=hostname,
        ip_address=ip_address,
        cpu_count=_get_cpu_count(),
        memory_bytes=_get_memory_total_bytes(),
        disk_bytes=_get_disk_bytes(),
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        tpu_name=tpu_name,
        tpu_type=tpu_type,
        tpu_worker_hostnames=tpu_worker_hostnames,
        tpu_worker_id=tpu_worker_id,
        tpu_chips_per_host_bounds=tpu_chips_per_host_bounds,
        gce_instance_name=gce_instance_name,
    )


def build_worker_metadata(
    hardware: HardwareProbe,
    accelerator_type: int = 0,
    accelerator_variant: str = "",
    gpu_count_override: int = 0,
    capacity_type: int = 0,
    worker_attributes: dict[str, str] | None = None,
) -> job_pb2.WorkerMetadata:
    """Combine hardware probe results with platform-provided config.

    Scheduling-relevant attributes (device-type, device-variant, preemptible) are
    derived from WorkerConfig fields (accelerator_type, accelerator_variant,
    capacity_type). Hardware probes populate diagnostic fields on WorkerMetadata
    (gpu_name, tpu_worker_hostnames, etc.) but do not influence the attributes map.

    The DeviceConfig oneof on WorkerMetadata is still built from config + probe
    data for capacity accounting (device count).
    """
    extra_attributes = worker_attributes or {}

    device = job_pb2.DeviceConfig()

    if accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU or hardware.tpu_type:
        tpu_type = hardware.tpu_type
        tpu_chip_count = 0
        if tpu_type:
            try:
                topo = get_tpu_topology(tpu_type)
                tpu_chip_count = topo.chips_per_vm
            except ValueError:
                logger.warning("Unknown TPU topology: %s", tpu_type)
        variant = accelerator_variant or tpu_type
        device.tpu.CopyFrom(job_pb2.TpuDevice(variant=variant, count=tpu_chip_count))
        gpu_count = 0
        gpu_name = ""
        gpu_memory_mb = 0
    elif accelerator_type == config_pb2.ACCELERATOR_TYPE_GPU or hardware.gpu_count > 0:
        gpu_count = gpu_count_override or hardware.gpu_count
        gpu_name = accelerator_variant or hardware.gpu_name or "auto"
        gpu_memory_mb = hardware.gpu_memory_mb
        device.gpu.CopyFrom(job_pb2.GpuDevice(variant=gpu_name, count=gpu_count))
    elif accelerator_type == config_pb2.ACCELERATOR_TYPE_CPU:
        device.cpu.CopyFrom(job_pb2.CpuDevice(variant=accelerator_variant or "cpu"))
        gpu_count = 0
        gpu_name = ""
        gpu_memory_mb = 0
    else:
        device.cpu.CopyFrom(job_pb2.CpuDevice(variant="cpu"))
        gpu_count = 0
        gpu_name = ""
        gpu_memory_mb = 0

    attributes = _build_worker_attributes(
        accelerator_type=accelerator_type,
        accelerator_variant=accelerator_variant,
        capacity_type=capacity_type,
        tpu_name=hardware.tpu_name,
        tpu_worker_id=hardware.tpu_worker_id,
        device=device,
        extra_attributes=extra_attributes,
    )

    return job_pb2.WorkerMetadata(
        hostname=hardware.hostname,
        ip_address=hardware.ip_address,
        cpu_count=hardware.cpu_count,
        memory_bytes=hardware.memory_bytes,
        disk_bytes=hardware.disk_bytes,
        tpu_name=hardware.tpu_name,
        tpu_worker_hostnames=hardware.tpu_worker_hostnames,
        tpu_worker_id=hardware.tpu_worker_id,
        tpu_chips_per_host_bounds=hardware.tpu_chips_per_host_bounds,
        gpu_count=gpu_count if not hardware.tpu_type else 0,
        gpu_name=gpu_name if not hardware.tpu_type else "",
        gpu_memory_mb=gpu_memory_mb,
        device=device,
        attributes=attributes,
        gce_instance_name=hardware.gce_instance_name,
        git_hash=os.environ.get("IRIS_GIT_HASH", "unknown"),
    )


class EnvironmentProvider(Protocol):
    """Protocol for worker environment probing."""

    def probe(self) -> job_pb2.WorkerMetadata: ...


class FixedEnvironmentProvider:
    """Returns pre-built worker metadata. Used by LOCAL mode and tests."""

    def __init__(self, metadata: job_pb2.WorkerMetadata):
        self._metadata = metadata

    def probe(self) -> job_pb2.WorkerMetadata:
        return self._metadata


class DefaultEnvironmentProvider:
    """Default implementation that probes real system resources."""

    def probe(self) -> job_pb2.WorkerMetadata:
        hardware = probe_hardware()
        return build_worker_metadata(hardware)


def _read_net_dev_bytes() -> tuple[int, int]:
    """Read cumulative network bytes from /proc/net/dev, summing all non-loopback interfaces.

    Returns (recv_bytes, sent_bytes). Works in Docker/K8s containers since
    /proc/net/dev reflects the container's network namespace.
    """
    recv_total = 0
    sent_total = 0
    with open("/proc/net/dev") as f:
        for line in f:
            # Skip header lines (contain "|")
            if "|" in line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            iface = parts[0].rstrip(":")
            if iface == "lo":
                continue
            recv_total += int(parts[1])  # bytes received
            sent_total += int(parts[9])  # bytes sent
    return recv_total, sent_total


MIN_DISK_FREE_FRACTION = 0.05
"""Worker is unhealthy if the work volume has less than 5% free space."""


@dataclass(frozen=True)
class HealthCheckResult:
    """Result of worker health checks run during heartbeat."""

    healthy: bool
    error: str = ""


def check_worker_health(disk_path: str = "/") -> HealthCheckResult:
    """Run basic health probes and return a combined result.

    Checks performed:
    - Can write and remove a tempfile in the work directory
    - Root/work volume has >= 5% free space

    Docker probing is implicit: if the worker is processing heartbeats
    and fetching task status, Docker is operational.

    If disk_path is not an existing directory (e.g. during teardown, or on
    platforms where the path does not exist), the probe is skipped and the
    worker is considered healthy.
    """
    dp = Path(disk_path)
    if not dp.is_dir():
        return HealthCheckResult(healthy=True)

    errors: list[str] = []

    # Check tempfile write
    try:
        probe_path = dp / ".iris_health_probe"
        probe_path.write_text("ok")
        probe_path.unlink()
    except FileNotFoundError:
        # TOCTOU: directory vanished between is_dir() check and write
        pass
    except OSError as e:
        errors.append(f"tempfile write failed: {e}")

    # Check disk free space
    try:
        usage = shutil.disk_usage(disk_path)
        if usage.total > 0:
            free_fraction = (usage.total - usage.used) / usage.total
            if free_fraction < MIN_DISK_FREE_FRACTION:
                pct = free_fraction * 100
                errors.append(f"disk free space {pct:.1f}% below threshold {MIN_DISK_FREE_FRACTION * 100:.0f}%")
    except OSError as e:
        errors.append(f"disk usage check failed: {e}")

    if errors:
        return HealthCheckResult(healthy=False, error="; ".join(errors))
    return HealthCheckResult(healthy=True)


class HostMetricsCollector:
    """Collects host-level resource metrics using /proc and standard library.

    CPU utilization and network bandwidth are computed as deltas between
    consecutive calls, so the first call always reports 0% CPU and 0 B/s
    network. Memory and disk are instantaneous snapshots.
    Gracefully returns partial data on non-Linux systems (macOS, etc.).
    """

    def __init__(self, disk_path: str = "/"):
        self._disk_path = disk_path
        self._prev_cpu_total = 0
        self._prev_cpu_idle = 0
        self._prev_net_recv = 0
        self._prev_net_sent = 0
        self._prev_net_time: float = 0.0

    def collect(self) -> job_pb2.WorkerResourceSnapshot:
        snapshot = job_pb2.WorkerResourceSnapshot()
        snapshot.timestamp.CopyFrom(timestamp_to_proto(Timestamp.now()))

        self._collect_memory(snapshot)
        self._collect_disk(snapshot)
        self._collect_cpu(snapshot)
        self._collect_network(snapshot)

        return snapshot

    def _collect_memory(self, snapshot: job_pb2.WorkerResourceSnapshot) -> None:
        try:
            with open("/proc/meminfo") as f:
                meminfo: dict[str, int] = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
                snapshot.memory_total_bytes = meminfo.get("MemTotal", 0)
                available = meminfo.get("MemAvailable", 0)
                snapshot.memory_used_bytes = snapshot.memory_total_bytes - available
        except (OSError, ValueError):
            pass

    def _collect_disk(self, snapshot: job_pb2.WorkerResourceSnapshot) -> None:
        try:
            usage = shutil.disk_usage(self._disk_path)
            snapshot.disk_total_bytes = usage.total
            snapshot.disk_used_bytes = usage.used
        except OSError:
            pass

    def _collect_cpu(self, snapshot: job_pb2.WorkerResourceSnapshot) -> None:
        """Compute CPU utilization as a delta between consecutive /proc/stat reads."""
        try:
            with open("/proc/stat") as f:
                line = f.readline()
                parts = line.split()
                if parts[0] != "cpu":
                    return
                values = [int(v) for v in parts[1:8]]
                total = sum(values)
                idle = values[3]

                delta_total = total - self._prev_cpu_total
                delta_idle = idle - self._prev_cpu_idle

                if delta_total > 0 and self._prev_cpu_total > 0:
                    snapshot.host_cpu_percent = max(0, min(100, 100 - int(delta_idle * 100 / delta_total)))

                self._prev_cpu_total = total
                self._prev_cpu_idle = idle
        except (OSError, ValueError, IndexError):
            pass

    def _collect_network(self, snapshot: job_pb2.WorkerResourceSnapshot) -> None:
        """Compute network bandwidth as bytes/sec delta from /proc/net/dev.

        Sums all non-loopback interfaces. Works inside Docker/K8s containers
        since /proc/net/dev reflects the container's network namespace.
        The first call establishes a baseline and reports 0 B/s.
        """
        try:
            recv, sent = _read_net_dev_bytes()
            now = time.monotonic()

            dt = now - self._prev_net_time
            if self._prev_net_time > 0 and dt > 0:
                snapshot.net_recv_bps = max(0, int((recv - self._prev_net_recv) / dt))
                snapshot.net_sent_bps = max(0, int((sent - self._prev_net_sent) / dt))

            self._prev_net_recv = recv
            self._prev_net_sent = sent
            self._prev_net_time = now
        except (OSError, ValueError, IndexError):
            pass
