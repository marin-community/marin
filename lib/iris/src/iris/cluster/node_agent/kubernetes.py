# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect node-exporter and DCGM telemetry beside a Kubernetes node."""

from __future__ import annotations

import logging
import math
import os
import shutil
import threading
import time
import urllib.request
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from finelog.client import LogClient
from finelog.client.log_client import Table
from finelog.deploy.config import derive_endpoint_uri, load_finelog_config
from rigging import telemetry
from rigging.timing import Duration

from iris.cluster.config import IrisClusterConfig, load_config
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME, TELEMETRY_ENDPOINT_PATH, resolve_endpoint_uri
from iris.cluster.node_agent import SERVICE_NAME
from iris.cluster.node_agent.metrics import DeviceMetric, NodeMetrics, NodeTarget, publish_node_telemetry
from iris.cluster.platforms.k8s.service import CloudK8sService, K8sService
from iris.cluster.platforms.k8s.types import (
    IRIS_ATTEMPT_ID_LABEL,
    IRIS_KUBERNETES_RUNTIME,
    IRIS_MANAGED_LABEL,
    IRIS_RUNTIME_LABEL,
    IRIS_TASK_CONTAINER_NAME,
    IRIS_TASK_ID_ANNOTATION,
    K8sResource,
    KubectlError,
)
from iris.cluster.stats.tables import TASK_STATS_NAMESPACE, IrisTaskStat, build_task_stat
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_INTERVAL = 30.0
CACHE_RECLAIM_INTERVAL = Duration.from_minutes(5)
K8S_API_TIMEOUT = 2.0
NODE_EXPORTER_ADDRESS = "127.0.0.1"
_K8S_GPU_MODEL_LABELS = ("nvidia.com/gpu.product", "gpu.nvidia.com/model")

# CoreWeave runs both exporters as DaemonSets in this namespace. node-exporter
# binds a host port (reachable at the node IP); dcgm-exporter is pod-network
# only (reachable at the pod IP), so it is discovered by listing the pods.
CW_EXPORTERS_NAMESPACE = "cw-exporters"
NODE_EXPORTER_PORT = 9100
DCGM_EXPORTER_PORT = 9400
_DCGM_NAME_LABEL = "app.kubernetes.io/name"
_DCGM_NAME_VALUE = "dcgm-exporter"

# CoreWeave bare-metal nodes mount the NVMe RAID here; the RAM-disk root ("/")
# is the fallback for a cluster laid out differently.
_PREFERRED_DISK_MOUNTS = ("/mnt/local", "/")

# Interfaces whose byte counters are not real host traffic (loopback, CNI
# plumbing, container veths). Everything else — physical NICs (enp*, eth*),
# InfiniBand (ibs*) — is summed, mirroring the worker daemon's /proc/net/dev sum.
_VIRTUAL_IFACE_PREFIXES = (
    "lo",
    "cilium",
    "lxc",
    "veth",
    "docker",
    "cni",
    "cali",
    "flannel",
    "tunl",
    "dummy",
    "kube",
    "nodelocaldns",
    "br-",
    "tap",
)

_MIB = 1024 * 1024
_SCRAPE_TIMEOUT = 2.0
_MAX_SCRAPE_WORKERS = 16
_MAX_SCRAPE_BYTES = 16 << 20
_MAX_DCGM_SAMPLES = 32_768
_MAX_DCGM_DEVICES = 256
_MAX_DCGM_EXPORTERS = 512
_TERMINAL_POD_PHASES = {"Failed", "Succeeded"}
_CACHE_RECLAIM_PREFIX = ".iris-reclaim-"

# One injectable seam so tests exercise the parsing/aggregation without a network.
Fetch = Callable[[str], str | None]


def _http_get(url: str, timeout: float = _SCRAPE_TIMEOUT) -> str | None:
    """GET a Prometheus ``/metrics`` endpoint; return its body or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            body = resp.read(_MAX_SCRAPE_BYTES + 1)
            if len(body) > _MAX_SCRAPE_BYTES:
                logger.warning("node-metrics scrape exceeded %d bytes for %s", _MAX_SCRAPE_BYTES, url)
                return None
            return body.decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("node-metrics scrape failed for %s: %s", url, e)
        return None


def _parse_labels(body: str) -> dict[str, str]:
    """Parse a Prometheus label set body (the text between ``{`` and ``}``).

    Handles quoted values that contain spaces or commas (e.g. dcgm's
    ``modelName="NVIDIA H100 80GB HBM3"``) and backslash escapes.
    """
    labels: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        eq = body.find("=", i)
        if eq == -1:
            break
        key = body[i:eq].strip().strip(",").strip()
        j = eq + 1
        if j >= n or body[j] != '"':
            break
        j += 1
        chars: list[str] = []
        while j < n and body[j] != '"':
            if body[j] == "\\" and j + 1 < n:
                nxt = body[j + 1]
                chars.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, nxt))
                j += 2
            else:
                chars.append(body[j])
                j += 1
        if key:
            labels[key] = "".join(chars)
        j += 1  # closing quote
        while j < n and body[j] in ", ":
            j += 1
        i = j
    return labels


class Sample(NamedTuple):
    """One parsed exporter sample line: ``metric_name``, its labels, and value."""

    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus(text: str) -> Iterator[Sample]:
    """Yield a :class:`Sample` for each sample line in exporter text.

    Comment/HELP/TYPE lines are skipped, as are samples whose value is not a
    finite float (``NaN`` / ``+Inf`` gauges some exporters emit for absent data).
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        brace = line.find("{")
        if brace == -1:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, labels, value_str = parts[0], {}, parts[1]
        else:
            end = line.rfind("}")
            if end == -1:
                continue
            name = line[:brace]
            labels = _parse_labels(line[brace + 1 : end])
            tail = line[end + 1 :].split()
            if not tail:
                continue
            value_str = tail[0]
        try:
            value = float(value_str)
        except ValueError:
            continue
        # Skip NaN / ±Inf gauges some exporters emit for absent data.
        if not math.isfinite(value):
            continue
        yield Sample(name, labels, value)


@dataclass
class ContainerResourceSample:
    """Kubelet resource counters for one container."""

    cpu_seconds: float | None = None
    memory_bytes: int | None = None


def parse_kubelet_resource_metrics(text: str) -> dict[tuple[str, str], ContainerResourceSample]:
    """Parse kubelet ``metrics/resource`` text by ``(pod, container)``."""
    resources: dict[tuple[str, str], ContainerResourceSample] = {}
    for name, labels, value in parse_prometheus(text):
        pod = labels.get("pod", "")
        container = labels.get("container", "")
        if not pod or not container:
            continue
        key = (pod, container)
        sample = resources.setdefault(key, ContainerResourceSample())
        if name == "container_cpu_usage_seconds_total":
            sample.cpu_seconds = value
        elif name == "container_memory_working_set_bytes":
            sample.memory_bytes = int(value)
    return resources


class TaskStatsCollector:
    """Write same-node Iris pod CPU and memory samples to ``iris.task``."""

    def __init__(
        self,
        kubectl: K8sService,
        node_name: str,
        table: Table,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._kubectl = kubectl
        self._node_name = node_name
        self._table = table
        self._clock = clock
        self._previous_cpu: dict[str, tuple[float, float]] = {}
        self._memory_peak: dict[str, int] = {}

    def collect_once(self) -> None:
        """Collect one task-resource sample for each running pod on this node."""
        labels = {IRIS_MANAGED_LABEL: "true", IRIS_RUNTIME_LABEL: IRIS_KUBERNETES_RUNTIME}
        try:
            pods = self._kubectl.list_json(
                K8sResource.PODS,
                labels=labels,
                field_selector=f"spec.nodeName={self._node_name}",
            )
            resources = parse_kubelet_resource_metrics(self._kubectl.node_resource_metrics(self._node_name))
        except KubectlError as error:
            logger.warning("task-resource collection failed for node %s: %s", self._node_name, error)
            return

        sampled_at = self._clock()
        live_pods = {
            pod.get("metadata", {}).get("name", "") for pod in pods if pod.get("status", {}).get("phase") == "Running"
        }
        rows: list[IrisTaskStat] = []
        for pod in pods:
            if pod.get("status", {}).get("phase") != "Running":
                continue
            metadata = pod.get("metadata", {})
            pod_name = metadata.get("name", "")
            task_id = metadata.get("annotations", {}).get(IRIS_TASK_ID_ANNOTATION, "")
            attempt = metadata.get("labels", {}).get(IRIS_ATTEMPT_ID_LABEL, "")
            sample = resources.get((pod_name, IRIS_TASK_CONTAINER_NAME))
            if not pod_name or not task_id or not attempt or sample is None:
                continue

            cpu_millicores = self._cpu_millicores(pod_name, sample.cpu_seconds, sampled_at)
            memory_bytes = sample.memory_bytes or 0
            memory_peak = max(memory_bytes, self._memory_peak.get(pod_name, 0))
            self._memory_peak[pod_name] = memory_peak
            rows.append(
                build_task_stat(
                    task_id=task_id,
                    attempt_id=int(attempt),
                    worker_id=pod_name,
                    usage=job_pb2.ResourceUsage(
                        cpu_millicores=cpu_millicores,
                        memory_mb=memory_bytes // _MIB,
                        memory_peak_mb=memory_peak // _MIB,
                    ),
                )
            )

        self._prune(live_pods)
        if rows:
            self._table.write(rows)

    def _cpu_millicores(self, pod_name: str, cpu_seconds: float | None, sampled_at: float) -> int:
        if cpu_seconds is None:
            return 0
        previous = self._previous_cpu.get(pod_name)
        self._previous_cpu[pod_name] = (cpu_seconds, sampled_at)
        if previous is None:
            return 0
        cpu_delta = cpu_seconds - previous[0]
        time_delta = sampled_at - previous[1]
        if cpu_delta < 0 or time_delta <= 0:
            return 0
        return max(0, round(cpu_delta / time_delta * 1000))

    def _prune(self, live_pods: set[str]) -> None:
        for pod_name in self._previous_cpu.keys() - live_pods:
            del self._previous_cpu[pod_name]
        for pod_name in self._memory_peak.keys() - live_pods:
            del self._memory_peak[pod_name]


def _entry_last_modified(entry: Path) -> float:
    latest = entry.lstat().st_mtime
    if entry.is_symlink() or not entry.is_dir():
        return latest

    errors: list[OSError] = []
    for root, directories, files in os.walk(entry, followlinks=False, onerror=errors.append):
        for name in directories + files:
            try:
                latest = max(latest, (Path(root) / name).lstat().st_mtime)
            except FileNotFoundError:
                continue
    if errors:
        raise errors[0]
    return latest


def _remove_cache_entry(entry: Path) -> None:
    if entry.name.startswith(_CACHE_RECLAIM_PREFIX):
        tombstone = entry
    else:
        # A task scheduled after the idle-node check can refill the original path
        # without racing the slower recursive deletion.
        tombstone = entry.with_name(f"{_CACHE_RECLAIM_PREFIX}{uuid.uuid4().hex}")
        entry.rename(tombstone)
    if tombstone.is_symlink() or not tombstone.is_dir():
        tombstone.unlink()
    else:
        shutil.rmtree(tombstone)


def reclaim_cache(
    cache_dir: Path,
    *,
    max_age: Duration,
    kubectl: K8sService,
    node_name: str,
    now: float | None = None,
) -> int:
    """Remove stale top-level cache entries while this node has no Iris tasks."""
    labels = {IRIS_MANAGED_LABEL: "true", IRIS_RUNTIME_LABEL: IRIS_KUBERNETES_RUNTIME}
    try:
        pods = kubectl.list_json(
            K8sResource.PODS,
            labels=labels,
            field_selector=f"spec.nodeName={node_name}",
        )
    except KubectlError as error:
        logger.warning("cache reclamation could not inspect tasks on node %s: %s", node_name, error)
        return 0
    if any(pod.get("status", {}).get("phase", "") not in _TERMINAL_POD_PHASES for pod in pods):
        return 0
    if not cache_dir.exists():
        return 0

    cutoff = (time.time() if now is None else now) - max_age.to_seconds()
    reclaimed = 0
    for namespace in cache_dir.iterdir():
        if namespace.is_symlink() or not namespace.is_dir():
            continue
        for entry in namespace.iterdir():
            try:
                if not entry.name.startswith(_CACHE_RECLAIM_PREFIX) and _entry_last_modified(entry) > cutoff:
                    continue
                _remove_cache_entry(entry)
                reclaimed += 1
            except FileNotFoundError:
                continue
            except OSError as error:
                logger.warning("could not reclaim cache entry %s: %s", entry, error)
    if reclaimed:
        logger.info("reclaimed %d stale cache entries from %s", reclaimed, cache_dir)
    return reclaimed


def _reclaim_cache_until_stopped(
    cache_dir: Path,
    max_age: Duration,
    kubectl: K8sService,
    node_name: str,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            reclaim_cache(cache_dir, max_age=max_age, kubectl=kubectl, node_name=node_name)
        except OSError:
            logger.exception("cache reclamation failed for %s", cache_dir)
        stop.wait(CACHE_RECLAIM_INTERVAL.to_seconds())


def _is_physical_iface(device: str) -> bool:
    return bool(device) and not device.startswith(_VIRTUAL_IFACE_PREFIXES)


@dataclass
class HostSample:
    """Host readings parsed from one node's ``node-exporter``.

    CPU seconds are cumulative counters; the collector differences consecutive
    samples into a busy percentage. Everything else is a point-in-time gauge.
    """

    cpu_idle_seconds: float
    cpu_total_seconds: float
    mem_total_bytes: int
    mem_used_bytes: int
    disk_total_bytes: int
    disk_used_bytes: int
    net_recv_bytes: int
    net_sent_bytes: int
    boot_time_seconds: int | None


@dataclass(frozen=True)
class _DcgmMetricSpec:
    name: str
    unit: str = ""
    scale: float = 1.0
    temporality: str = telemetry.CURRENT_SNAPSHOT
    attributes: tuple[tuple[str, str], ...] = ()
    source_labels: tuple[str, ...] = ()


_DCGM_METRICS = {
    "DCGM_FI_DEV_FB_USED": _DcgmMetricSpec("gpu_memory_used_bytes", "By", _MIB),
    "DCGM_FI_DEV_FB_TOTAL": _DcgmMetricSpec("gpu_memory_total_bytes", "By", _MIB),
    "DCGM_FI_DEV_FB_FREE": _DcgmMetricSpec("gpu_memory_free_bytes", "By", _MIB),
    "DCGM_FI_DEV_FB_RESERVED": _DcgmMetricSpec("gpu_memory_reserved_bytes", "By", _MIB),
    "DCGM_FI_DEV_GPU_UTIL": _DcgmMetricSpec("gpu_utilization_percent", "%"),
    "DCGM_FI_DEV_MEM_COPY_UTIL": _DcgmMetricSpec("gpu_memory_copy_utilization_percent", "%"),
    "DCGM_FI_DEV_GPU_TEMP": _DcgmMetricSpec("gpu_temperature_celsius", "Cel"),
    "DCGM_FI_DEV_MEMORY_TEMP": _DcgmMetricSpec("gpu_memory_temperature_celsius", "Cel"),
    "DCGM_FI_DEV_POWER_USAGE": _DcgmMetricSpec("gpu_power_watts", "W"),
    "DCGM_FI_DEV_POWER_MGMT_LIMIT": _DcgmMetricSpec("gpu_power_limit_watts", "W"),
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": _DcgmMetricSpec(
        "gpu_energy_millijoules", "mJ", temporality=telemetry.CUMULATIVE_SNAPSHOT
    ),
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": _DcgmMetricSpec(
        "gpu_pcie_replay_errors", "{error}", temporality=telemetry.CUMULATIVE_SNAPSHOT
    ),
    "DCGM_FI_DEV_XID_ERRORS": _DcgmMetricSpec("gpu_xid_error_code", source_labels=("err_code",)),
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS": _DcgmMetricSpec(
        "gpu_row_remapped_rows",
        "{row}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
        attributes=(("error_kind", "correctable"),),
    ),
    "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS": _DcgmMetricSpec(
        "gpu_row_remapped_rows",
        "{row}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
        attributes=(("error_kind", "uncorrectable"),),
    ),
    "DCGM_FI_DEV_ROW_REMAP_FAILURE": _DcgmMetricSpec("gpu_row_remap_failures", "{failure}"),
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE": _DcgmMetricSpec("gpu_graphics_engine_active_ratio", "1"),
    "DCGM_FI_PROF_SM_ACTIVE": _DcgmMetricSpec("gpu_sm_active_ratio", "1"),
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": _DcgmMetricSpec("gpu_tensor_active_ratio", "1"),
    "DCGM_FI_PROF_DRAM_ACTIVE": _DcgmMetricSpec("gpu_dram_active_ratio", "1"),
    "DCGM_FI_PROF_NVLINK_RX_BYTES": _DcgmMetricSpec("gpu_nvlink_receive_bytes_per_second", "By/s"),
    "DCGM_FI_PROF_NVLINK_TX_BYTES": _DcgmMetricSpec("gpu_nvlink_transmit_bytes_per_second", "By/s"),
    "DCGM_FI_PROF_PCIE_RX_BYTES": _DcgmMetricSpec("gpu_pcie_receive_bytes_per_second", "By/s"),
    "DCGM_FI_PROF_PCIE_TX_BYTES": _DcgmMetricSpec("gpu_pcie_transmit_bytes_per_second", "By/s"),
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL": _DcgmMetricSpec(
        "gpu_nvlink_errors",
        "{error}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
        attributes=(("error_kind", "crc_flit"),),
    ),
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL": _DcgmMetricSpec(
        "gpu_nvlink_errors",
        "{error}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
        attributes=(("error_kind", "crc_data"),),
    ),
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL": _DcgmMetricSpec(
        "gpu_nvlink_errors",
        "{error}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
        attributes=(("error_kind", "replay"),),
    ),
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL": _DcgmMetricSpec(
        "gpu_nvlink_errors",
        "{error}",
        temporality=telemetry.CUMULATIVE_SNAPSHOT,
        attributes=(("error_kind", "recovery"),),
    ),
    "DCGM_EXP_GPU_HEALTH_STATUS": _DcgmMetricSpec(
        "gpu_health_status",
        source_labels=("health_watch", "health_error_code", "health_error_severity", "health_error_category"),
    ),
}


@dataclass
class GpuSample:
    """GPU hardware readings aggregated across one node's GPUs from ``dcgm-exporter``.

    HBM and power sum across the node's GPUs (whole-node totals); utilization is
    the mean and temperature the max (the node's hottest GPU) — the readings an
    operator scans a fleet for.
    """

    gpu_count: int
    gpu_model: str
    hbm_used_bytes: int
    hbm_total_bytes: int
    util_pct: float
    temp_c: float
    power_w: float
    power_limit_w: float
    exporter_uid: str = ""
    device_metrics: tuple[DeviceMetric, ...] = ()


def parse_node_exporter(text: str, *, disk_mounts: tuple[str, ...] = _PREFERRED_DISK_MOUNTS) -> HostSample:
    """Parse the host readings the collector consumes from ``node-exporter`` text."""
    cpu_idle = cpu_total = 0.0
    mem_total = mem_avail = 0
    fs_size: dict[str, int] = {}
    fs_avail: dict[str, int] = {}
    net_recv = net_sent = 0
    boot_time: int | None = None
    for name, labels, value in parse_prometheus(text):
        if name == "node_cpu_seconds_total":
            cpu_total += value
            if labels.get("mode") == "idle":
                cpu_idle += value
        elif name == "node_memory_MemTotal_bytes":
            mem_total = int(value)
        elif name == "node_memory_MemAvailable_bytes":
            mem_avail = int(value)
        elif name == "node_filesystem_size_bytes":
            fs_size[labels.get("mountpoint", "")] = int(value)
        elif name == "node_filesystem_avail_bytes":
            fs_avail[labels.get("mountpoint", "")] = int(value)
        elif name == "node_network_receive_bytes_total" and _is_physical_iface(labels.get("device", "")):
            net_recv += int(value)
        elif name == "node_network_transmit_bytes_total" and _is_physical_iface(labels.get("device", "")):
            net_sent += int(value)
        elif name == "node_boot_time_seconds":
            boot_time = int(value)

    mount = next((m for m in disk_mounts if m in fs_size), "")
    disk_total = fs_size.get(mount, 0)
    disk_used = max(0, disk_total - fs_avail.get(mount, 0))
    return HostSample(
        cpu_idle_seconds=cpu_idle,
        cpu_total_seconds=cpu_total,
        mem_total_bytes=mem_total,
        mem_used_bytes=max(0, mem_total - mem_avail),
        disk_total_bytes=disk_total,
        disk_used_bytes=disk_used,
        net_recv_bytes=net_recv,
        net_sent_bytes=net_sent,
        boot_time_seconds=boot_time,
    )


@dataclass
class _DcgmDevice:
    host: str
    gpu_index: str
    gpu_uuid: str = ""
    pci_bus_id: str = ""
    gpu_model: str = ""
    driver_version: str = ""
    metrics: dict[tuple[str, tuple[tuple[str, str], ...]], Sample] = field(default_factory=dict)


def parse_dcgm(text: str, *, exporter_uid: str = "", node_name: str = "") -> dict[str, GpuSample]:
    """Parse ``dcgm-exporter`` text into one aggregate ``GpuSample`` per node.

    The node name normally comes from DCGM's hostname label and falls back to
    the exporter pod's assigned node. Per-device telemetry requires the stable
    UUID, PCI bus id, and exporter pod UID; aggregate dashboard readings remain
    available when an older exporter omits those labels.
    """
    parsed = list(parse_prometheus(text))
    if len(parsed) > _MAX_DCGM_SAMPLES:
        raise ValueError("dcgm sample limit exceeded")

    used: dict[str, dict[str, int]] = defaultdict(dict)
    total: dict[str, dict[str, int]] = defaultdict(dict)
    util: dict[str, dict[str, float]] = defaultdict(dict)
    temp: dict[str, dict[str, float]] = defaultdict(dict)
    power: dict[str, dict[str, float]] = defaultdict(dict)
    power_limit: dict[str, dict[str, float]] = defaultdict(dict)
    model: dict[str, str] = {}
    gpus: dict[str, set[str]] = defaultdict(set)
    devices: dict[tuple[str, str], _DcgmDevice] = {}

    for sample in parsed:
        name, labels, value = sample
        host = labels.get("hostname") or labels.get("Hostname") or node_name
        gpu = labels.get("gpu")
        if not host or gpu is None:
            continue
        gpus[host].add(gpu)
        key = (host, gpu)
        device = devices.setdefault(key, _DcgmDevice(host=host, gpu_index=gpu))
        device.gpu_uuid = labels.get("UUID") or labels.get("uuid") or device.gpu_uuid
        device.pci_bus_id = labels.get("pci_bus_id") or device.pci_bus_id
        device.gpu_model = labels.get("modelName") or device.gpu_model
        device.driver_version = labels.get("DCGM_FI_DRIVER_VERSION") or device.driver_version
        if labels.get("modelName"):
            model[host] = labels["modelName"]
        if spec := _DCGM_METRICS.get(name):
            metric_labels = tuple((label, labels[label]) for label in spec.source_labels if labels.get(label))
            device.metrics[(name, metric_labels)] = sample
        if name == "DCGM_FI_DEV_FB_USED":
            used[host][gpu] = int(value) * _MIB
        elif name == "DCGM_FI_DEV_FB_TOTAL":
            total[host][gpu] = int(value) * _MIB
        elif name == "DCGM_FI_DEV_GPU_UTIL":
            util[host][gpu] = value
        elif name == "DCGM_FI_DEV_GPU_TEMP":
            temp[host][gpu] = value
        elif name == "DCGM_FI_DEV_POWER_USAGE":
            power[host][gpu] = value
        elif name == "DCGM_FI_DEV_POWER_MGMT_LIMIT":
            power_limit[host][gpu] = value

    device_metrics: dict[str, list[DeviceMetric]] = defaultdict(list)
    if len(devices) > _MAX_DCGM_DEVICES:
        raise ValueError("dcgm device limit exceeded")
    for device in devices.values():
        if not (device.gpu_uuid and device.pci_bus_id and exporter_uid):
            continue
        inventory_attributes = {"device_kind": "gpu"}
        if device.gpu_model:
            inventory_attributes["gpu_model"] = device.gpu_model
        if device.driver_version:
            inventory_attributes["driver_version"] = device.driver_version
        device_metrics[device.host].append(
            DeviceMetric(
                "hardware_inventory",
                1.0,
                "",
                telemetry.CURRENT_SNAPSHOT,
                device.gpu_index,
                device.gpu_uuid,
                device.pci_bus_id,
                inventory_attributes,
            )
        )
        for (source_name, metric_labels), sample in device.metrics.items():
            spec = _DCGM_METRICS[source_name]
            device_metrics[device.host].append(
                DeviceMetric(
                    spec.name,
                    sample.value * spec.scale,
                    spec.unit,
                    spec.temporality,
                    device.gpu_index,
                    device.gpu_uuid,
                    device.pci_bus_id,
                    {**dict(spec.attributes), **dict(metric_labels)},
                )
            )

    samples: dict[str, GpuSample] = {}
    for host, gpu_ids in gpus.items():
        utils = list(util[host].values())
        temps = list(temp[host].values())
        samples[host] = GpuSample(
            gpu_count=len(gpu_ids),
            gpu_model=model.get(host, ""),
            hbm_used_bytes=sum(used[host].values()),
            hbm_total_bytes=sum(total[host].values()),
            util_pct=sum(utils) / len(utils) if utils else 0.0,
            temp_c=max(temps) if temps else 0.0,
            power_w=sum(power[host].values()),
            power_limit_w=sum(power_limit[host].values()),
            exporter_uid=exporter_uid,
            device_metrics=tuple(device_metrics[host]),
        )
    return samples


class NodeStatsScraper:
    """Scrapes node-exporter + dcgm-exporter and computes per-node ``NodeMetrics``.

    Holds the previous CPU counters per node so a busy percentage can be
    differenced from consecutive scrapes; a node's first scrape (or one after a
    counter reset) reports ``cpu_pct=None``.
    """

    def __init__(
        self,
        kubectl: K8sService,
        *,
        exporters_namespace: str = CW_EXPORTERS_NAMESPACE,
        node_exporter_port: int = NODE_EXPORTER_PORT,
        dcgm_port: int = DCGM_EXPORTER_PORT,
        fetch: Fetch = _http_get,
        max_workers: int = _MAX_SCRAPE_WORKERS,
    ) -> None:
        self._kubectl = kubectl
        self._ns = exporters_namespace
        self._node_port = node_exporter_port
        self._dcgm_port = dcgm_port
        self._fetch = fetch
        self._max_workers = max_workers
        self._prev_cpu: dict[str, tuple[float, float]] = {}

    def scrape(self, targets: list[NodeTarget]) -> dict[str, NodeMetrics]:
        """Return per-node metrics for ``targets`` (best-effort; missing nodes omitted)."""
        if not targets:
            return {}
        host_samples = self._scrape_hosts(targets)
        gpu_samples = self._scrape_gpus({target.name for target in targets})
        names = {t.name for t in targets}
        self._prune_prev(names)

        out: dict[str, NodeMetrics] = {}
        for target in targets:
            host = host_samples.get(target.name)
            gpu = gpu_samples.get(target.name)
            metrics = NodeMetrics()
            if host is not None:
                metrics.host_available = True
                metrics.host_source_replica_uid = target.node_uid
                if host.boot_time_seconds is not None:
                    metrics.host_source_replica_uid += f":{host.boot_time_seconds}"
                metrics.cpu_pct = self._cpu_pct(target.name, host)
                metrics.mem_used_bytes = host.mem_used_bytes
                metrics.mem_total_bytes = host.mem_total_bytes
                metrics.disk_used_bytes = host.disk_used_bytes
                metrics.disk_total_bytes = host.disk_total_bytes
                metrics.net_recv_bytes = host.net_recv_bytes
                metrics.net_sent_bytes = host.net_sent_bytes
            if gpu is not None:
                metrics.dcgm_available = True
                metrics.dcgm_exporter_uid = gpu.exporter_uid
                metrics.device_metrics = gpu.device_metrics
                metrics.gpu_count = gpu.gpu_count
                metrics.gpu_model = gpu.gpu_model
                metrics.hbm_used_bytes = gpu.hbm_used_bytes
                metrics.hbm_total_bytes = gpu.hbm_total_bytes
                metrics.gpu_util_pct = gpu.util_pct
                metrics.gpu_temp_c = gpu.temp_c
                metrics.gpu_power_w = gpu.power_w
                metrics.gpu_power_limit_w = gpu.power_limit_w
            out[target.name] = metrics
        return out

    def _cpu_pct(self, node: str, host: HostSample) -> float | None:
        prev = self._prev_cpu.get(node)
        self._prev_cpu[node] = (host.cpu_idle_seconds, host.cpu_total_seconds)
        if prev is None:
            return None
        d_idle = host.cpu_idle_seconds - prev[0]
        d_total = host.cpu_total_seconds - prev[1]
        if d_total <= 0 or d_idle < 0:  # counter reset / no elapsed time
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_total)))

    def _prune_prev(self, live: set[str]) -> None:
        for stale in self._prev_cpu.keys() - live:
            del self._prev_cpu[stale]

    def _scrape_hosts(self, targets: list[NodeTarget]) -> dict[str, HostSample]:
        urls = {t.name: f"http://{t.internal_ip}:{self._node_port}/metrics" for t in targets if t.internal_ip}
        texts = self._fetch_all(urls)
        return {name: parse_node_exporter(text) for name, text in texts.items()}

    def _scrape_gpus(self, node_names: set[str]) -> dict[str, GpuSample]:
        try:
            pods = self._kubectl.list_pods_in_namespace(self._ns)
        except Exception as e:
            logger.debug("node-metrics: listing dcgm exporters in %s failed: %s", self._ns, e)
            return {}
        dcgm_pods = [
            pod
            for pod in pods
            if pod.get("metadata", {}).get("labels", {}).get(_DCGM_NAME_LABEL) == _DCGM_NAME_VALUE
            and pod.get("spec", {}).get("nodeName", "") in node_names
        ]
        if len(dcgm_pods) > _MAX_DCGM_EXPORTERS:
            logger.warning("dcgm exporter count exceeded %d", _MAX_DCGM_EXPORTERS)
        exporters: dict[str, tuple[str, str, str]] = {}
        for pod in dcgm_pods[:_MAX_DCGM_EXPORTERS]:
            pod_ip = pod.get("status", {}).get("podIP")
            name = pod.get("metadata", {}).get("name", "")
            uid = pod.get("metadata", {}).get("uid", "")
            node_name = pod.get("spec", {}).get("nodeName", "")
            if pod_ip and name:
                exporters[name] = (f"http://{pod_ip}:{self._dcgm_port}/metrics", uid, node_name)
        merged: dict[str, GpuSample] = {}
        urls = {name: endpoint[0] for name, endpoint in exporters.items()}
        for name, text in self._fetch_all(urls).items():
            _, uid, node_name = exporters[name]
            try:
                merged.update(parse_dcgm(text, exporter_uid=uid, node_name=node_name))
            except ValueError as error:
                logger.warning("could not parse dcgm exporter %s: %s", name, error)
        return merged

    def _fetch_all(self, urls: dict[str, str]) -> dict[str, str]:
        if not urls:
            return {}
        keys = list(urls)
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(keys)), thread_name_prefix="node-scrape") as ex:
            bodies = list(ex.map(self._fetch, [urls[k] for k in keys]))
        return {k: body for k, body in zip(keys, bodies, strict=True) if body is not None}


def _log_service_endpoint(cluster_config: IrisClusterConfig) -> str:
    spec = cluster_config.endpoints.get(LOG_SERVER_ENDPOINT_NAME)
    if cluster_config.finelog.config:
        if spec is not None:
            raise ValueError("cluster config cannot set both finelog.config and /system/log-server")
        uri, metadata = derive_endpoint_uri(load_finelog_config(cluster_config.finelog.config))
    elif spec is not None:
        uri, metadata = spec.uri, dict(spec.metadata)
    else:
        raise ValueError("node telemetry requires an external /system/log-server endpoint")
    return resolve_endpoint_uri(uri, metadata).rstrip("/")


def _node_target(k8s: CloudK8sService, node_name: str) -> NodeTarget:
    node = k8s.get_json(K8sResource.NODES, node_name)
    if node is None:
        raise ConnectionError(f"Kubernetes node {node_name!r} is not visible")
    metadata = node.get("metadata", {})
    node_uid = metadata.get("uid", "")
    if not node_uid:
        raise ValueError(f"Kubernetes node {node_name!r} has no metadata.uid")
    labels = metadata.get("labels", {})
    allocatable = node.get("status", {}).get("allocatable", {})
    gpu_count = int(allocatable.get("nvidia.com/gpu", 0))
    gpu_model = next((labels[name] for name in _K8S_GPU_MODEL_LABELS if labels.get(name)), "")
    return NodeTarget(
        name=node_name,
        node_uid=node_uid,
        internal_ip=NODE_EXPORTER_ADDRESS,
        device_type="gpu" if gpu_count or gpu_model else "cpu",
        device_variant=gpu_model,
    )


def collect_once(
    scraper: NodeStatsScraper,
    target: NodeTarget,
    task_stats_collector: TaskStatsCollector | None = None,
) -> None:
    """Publish one same-node exporter and task-resource collection pass."""
    metrics = scraper.scrape([target])[target.name]
    publish_node_telemetry(target, metrics, time.time())
    if task_stats_collector is not None:
        task_stats_collector.collect_once()
    telemetry.record_runtime_health()


def run(config_path: Path, node_name: str, namespace: str, stop: threading.Event) -> None:
    """Collect telemetry until the process receives a shutdown signal."""
    config = load_config(config_path)
    log_service_endpoint = _log_service_endpoint(config)
    endpoint = log_service_endpoint + TELEMETRY_ENDPOINT_PATH
    k8s = CloudK8sService(namespace=namespace, timeout=K8S_API_TIMEOUT)
    target = _node_target(k8s, node_name)
    log_client = LogClient.connect(log_service_endpoint)
    task_stats_collector = TaskStatsCollector(
        k8s,
        node_name,
        log_client.get_table(TASK_STATS_NAMESPACE, IrisTaskStat),
    )
    telemetry.configure(
        endpoint=endpoint,
        service=SERVICE_NAME,
        attributes={"node_name": target.name, "node_uid": target.node_uid, "role": str(telemetry.TelemetryRole.WORKER)},
    )
    scraper = NodeStatsScraper(k8s)
    cache_reclaimer: threading.Thread | None = None
    if config.kubernetes_provider.cache_max_age is not None:
        cache_dir = Path(config.kubernetes_provider.cache_dir or "/cache")
        cache_reclaimer = threading.Thread(
            target=_reclaim_cache_until_stopped,
            args=(cache_dir, config.kubernetes_provider.cache_max_age, k8s, node_name, stop),
            name="cache-reclaimer",
            daemon=True,
        )
        cache_reclaimer.start()
    try:
        while not stop.is_set():
            collect_once(scraper, target, task_stats_collector)
            stop.wait(DEFAULT_COLLECTION_INTERVAL)
    finally:
        stop.set()
        if cache_reclaimer is not None:
            cache_reclaimer.join(timeout=10.0)
        log_client.close()
        telemetry.shutdown(5.0)
