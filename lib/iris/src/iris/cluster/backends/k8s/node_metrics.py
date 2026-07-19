# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scrape k8s node host + GPU hardware readings from the cluster's exporters.

A Kubernetes cluster has no per-node Iris worker daemon, so nothing emits the
host/device heartbeats the GCE/TPU worker daemon writes to ``iris.worker``.
CoreWeave CKS instead runs the standard Prometheus exporters on every node — a
``node-exporter`` DaemonSet (host CPU/memory/disk/network) and a
``dcgm-exporter`` DaemonSet (per-GPU HBM, temperature, utilization, power). The
controller scrapes those exporters directly and the cluster backend writes one
``iris.worker`` row per node, giving k8s clusters the host/device time-series the
worker-daemon clusters already get.

There is no in-cluster Prometheus/VictoriaMetrics query endpoint to aggregate
against — CoreWeave's is a scrape-only ``vmagent`` that forwards to a managed
store — so the controller reads the raw ``/metrics`` endpoints itself:
``node-exporter`` at the node IP's host port and ``dcgm-exporter`` at each
exporter pod's IP. Scraping is best-effort: a node whose exporter does not
answer simply has null utilization for that tick; its liveness and allocatable
capacity still surface through the cluster status RPC.
"""

from __future__ import annotations

import logging
import math
import threading
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple

from finelog.client.log_client import Table
from rigging.timing import Timestamp

from iris.cluster.platforms.k8s.service import K8sService
from iris.cluster.worker.stats import IrisWorkerStat, stats_timestamp

logger = logging.getLogger(__name__)

# The controller ticks reconcile at ~1s, but exporter scrapes are coarser: DCGM
# and node-exporter update on their own scrape cadence, and the cpu-busy delta
# wants a meaningful interval to difference over.
DEFAULT_NODE_STATS_POLL_INTERVAL = 30.0

# on_snapshot receives the freshly-scraped per-node metrics and the scrape time
# (epoch ms), which the cluster status RPC folds into its live NodeStatus rows.
SnapshotSink = Callable[[dict[str, "NodeMetrics"], int], None]

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

# One injectable seam so tests exercise the parsing/aggregation without a network.
Fetch = Callable[[str], str | None]


def _http_get(url: str, timeout: float = _SCRAPE_TIMEOUT) -> str | None:
    """GET a Prometheus ``/metrics`` endpoint; return its body or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8", errors="replace")
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


def parse_node_exporter(text: str, *, disk_mounts: tuple[str, ...] = _PREFERRED_DISK_MOUNTS) -> HostSample:
    """Parse the host readings the collector consumes from ``node-exporter`` text."""
    cpu_idle = cpu_total = 0.0
    mem_total = mem_avail = 0
    fs_size: dict[str, int] = {}
    fs_avail: dict[str, int] = {}
    net_recv = net_sent = 0
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
    )


def parse_dcgm(text: str) -> dict[str, GpuSample]:
    """Parse ``dcgm-exporter`` text into one aggregate ``GpuSample`` per node.

    Keys are the ``hostname`` label DCGM stamps on every series (the node name).
    One dcgm pod reports only its own node, but merging by hostname keeps a
    fleet-wide scrape correct regardless of which pod produced which series.
    """
    used: dict[str, dict[str, int]] = defaultdict(dict)
    total: dict[str, dict[str, int]] = defaultdict(dict)
    util: dict[str, dict[str, float]] = defaultdict(dict)
    temp: dict[str, dict[str, float]] = defaultdict(dict)
    power: dict[str, dict[str, float]] = defaultdict(dict)
    model: dict[str, str] = {}
    gpus: dict[str, set[str]] = defaultdict(set)

    for name, labels, value in parse_prometheus(text):
        host = labels.get("hostname") or labels.get("Hostname")
        gpu = labels.get("gpu")
        if not host or gpu is None:
            continue
        gpus[host].add(gpu)
        if labels.get("modelName"):
            model[host] = labels["modelName"]
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
        )
    return samples


@dataclass
class NodeTarget:
    """A node to scrape, plus the identity its ``iris.worker`` row carries.

    ``internal_ip`` is where the node-exporter is reached; the rest is static
    node metadata (allocatable capacity, accelerator type, region) replicated
    into every heartbeat row so the ``iris.worker`` table stays self-contained.
    """

    name: str
    internal_ip: str
    status: str = ""
    device_type: str = ""
    device_variant: str = ""
    zone: str = ""
    cpu_count: int = 0
    memory_bytes: int = 0
    running_pod_count: int = 0


@dataclass
class NodeMetrics:
    """Computed per-node snapshot: host utilization + aggregate GPU hardware.

    All fields are optional. Host fields are None when the node-exporter did not
    answer (or, for ``cpu_pct``, when there is no prior sample to difference
    against); GPU fields are None on a CPU node or when dcgm did not answer.
    """

    cpu_pct: float | None = None
    mem_used_bytes: int | None = None
    mem_total_bytes: int | None = None
    disk_used_bytes: int | None = None
    disk_total_bytes: int | None = None
    net_recv_bytes: int | None = None
    net_sent_bytes: int | None = None
    gpu_count: int | None = None
    gpu_model: str = ""
    hbm_used_bytes: int | None = None
    hbm_total_bytes: int | None = None
    gpu_util_pct: float | None = None
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None


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
        gpu_samples = self._scrape_gpus()
        names = {t.name for t in targets}
        self._prune_prev(names)

        out: dict[str, NodeMetrics] = {}
        for target in targets:
            host = host_samples.get(target.name)
            gpu = gpu_samples.get(target.name)
            metrics = NodeMetrics()
            if host is not None:
                metrics.cpu_pct = self._cpu_pct(target.name, host)
                metrics.mem_used_bytes = host.mem_used_bytes
                metrics.mem_total_bytes = host.mem_total_bytes
                metrics.disk_used_bytes = host.disk_used_bytes
                metrics.disk_total_bytes = host.disk_total_bytes
                metrics.net_recv_bytes = host.net_recv_bytes
                metrics.net_sent_bytes = host.net_sent_bytes
            if gpu is not None:
                metrics.gpu_count = gpu.gpu_count
                metrics.gpu_model = gpu.gpu_model
                metrics.hbm_used_bytes = gpu.hbm_used_bytes
                metrics.hbm_total_bytes = gpu.hbm_total_bytes
                metrics.gpu_util_pct = gpu.util_pct
                metrics.gpu_temp_c = gpu.temp_c
                metrics.gpu_power_w = gpu.power_w
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

    def _scrape_gpus(self) -> dict[str, GpuSample]:
        try:
            pods = self._kubectl.list_pods_in_namespace(self._ns)
        except Exception as e:
            logger.debug("node-metrics: listing dcgm exporters in %s failed: %s", self._ns, e)
            return {}
        urls: dict[str, str] = {}
        for pod in pods:
            labels = pod.get("metadata", {}).get("labels", {})
            if labels.get(_DCGM_NAME_LABEL) != _DCGM_NAME_VALUE:
                continue
            pod_ip = pod.get("status", {}).get("podIP")
            name = pod.get("metadata", {}).get("name", "")
            if pod_ip and name:
                urls[name] = f"http://{pod_ip}:{self._dcgm_port}/metrics"
        merged: dict[str, GpuSample] = {}
        for text in self._fetch_all(urls).values():
            merged.update(parse_dcgm(text))
        return merged

    def _fetch_all(self, urls: dict[str, str]) -> dict[str, str]:
        if not urls:
            return {}
        keys = list(urls)
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(keys)), thread_name_prefix="node-scrape") as ex:
            bodies = list(ex.map(self._fetch, [urls[k] for k in keys]))
        return {k: body for k, body in zip(keys, bodies, strict=True) if body is not None}


def build_node_stat(target: NodeTarget, metrics: NodeMetrics | None) -> IrisWorkerStat:
    """Build one ``iris.worker`` row for a node from its identity and scraped metrics.

    A k8s node has no worker daemon to emit its heartbeat, so the controller
    writes the row instead — worker_id is the node name, host utilization comes
    from node-exporter, and the GPU-hardware columns from dcgm-exporter. Missing
    metrics (``metrics is None`` on a tick with no successful scrape, or a null
    field) leave those columns null; identity and allocatable capacity always
    populate, so the node still shows in the fleet.
    """
    m = metrics or NodeMetrics()
    # Node GPU labels are unreliable on CoreWeave (some GPU nodes report an empty
    # nvidia.com/gpu allocatable), so dcgm-exporter is the authoritative signal:
    # fall back to it for the device type/variant when the node metadata is blank.
    device_type = target.device_type or ("gpu" if (m.gpu_count or 0) > 0 else "cpu")
    device_variant = target.device_variant or m.gpu_model
    # Utilization ints are non-nullable on IrisWorkerStat (worker daemons always
    # fill them); a node with no scrape reports 0 for the tick.
    return IrisWorkerStat(
        worker_id=target.name,
        ts=stats_timestamp(),
        status=target.status,
        address=target.internal_ip,
        cpu_pct=m.cpu_pct if m.cpu_pct is not None else 0.0,
        mem_bytes=m.mem_used_bytes or 0,
        mem_total_bytes=m.mem_total_bytes or 0,
        disk_used_bytes=m.disk_used_bytes or 0,
        disk_total_bytes=m.disk_total_bytes or 0,
        running_task_count=target.running_pod_count,
        total_process_count=0,
        net_recv_bytes=m.net_recv_bytes or 0,
        net_sent_bytes=m.net_sent_bytes or 0,
        device_type=device_type,
        device_variant=device_variant,
        cpu_count=target.cpu_count,
        memory_bytes=target.memory_bytes,
        tpu_name="",
        gce_instance_name="",
        zone=target.zone,
        gpu_count=m.gpu_count,
        hbm_used_bytes=m.hbm_used_bytes,
        hbm_total_bytes=m.hbm_total_bytes,
        gpu_util_pct=m.gpu_util_pct,
        gpu_temp_c=m.gpu_temp_c,
        gpu_power_w=m.gpu_power_w,
    )


class NodeStatsCollector:
    """Background thread that scrapes node metrics and writes ``iris.worker`` rows.

    The reconcile loop declares the current node set via :meth:`set_nodes` once
    per cluster-scan cycle. Each ``poll_interval`` the collector scrapes those
    nodes' exporters, appends one ``IrisWorkerStat`` per node to the
    ``iris.worker`` table, and hands the latest snapshot to ``on_snapshot`` so
    the cluster status RPC can serve current utilization without re-scraping.
    """

    def __init__(
        self,
        kubectl: K8sService,
        node_stats_table: Table,
        *,
        exporters_namespace: str = CW_EXPORTERS_NAMESPACE,
        poll_interval: float = DEFAULT_NODE_STATS_POLL_INTERVAL,
        on_snapshot: SnapshotSink | None = None,
    ) -> None:
        self._scraper = NodeStatsScraper(kubectl, exporters_namespace=exporters_namespace)
        self._table = node_stats_table
        self._poll_interval = poll_interval
        self._on_snapshot = on_snapshot
        self._targets: list[NodeTarget] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="node-stats-collector")
        self._thread.start()

    def set_nodes(self, targets: list[NodeTarget]) -> None:
        """Declare the authoritative node set to scrape (called once per sync cycle)."""
        with self._lock:
            self._targets = list(targets)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.collect_once()
            except Exception:
                logger.debug("node-stats collect cycle failed", exc_info=True)
            self._stop.wait(timeout=self._poll_interval)

    def collect_once(self) -> None:
        with self._lock:
            targets = list(self._targets)
        if not targets:
            return
        metrics = self._scraper.scrape(targets)
        rows = [build_node_stat(t, metrics.get(t.name)) for t in targets]
        try:
            self._table.write(rows)
        except Exception:
            logger.debug("node-stats table write failed", exc_info=True)
        if self._on_snapshot is not None:
            self._on_snapshot(metrics, Timestamp.now().epoch_ms())

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
