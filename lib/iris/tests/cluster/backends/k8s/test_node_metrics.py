# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Node-metrics scrape/parse/emit tests.

The exporter samples below mirror the real ``node-exporter`` and
``dcgm-exporter`` output on CoreWeave H100 nodes (scientific-notation values,
per-cpu/per-mode CPU counters, ``/mnt/local`` NVMe filesystem, multi-interface
network, and DCGM's ``hostname``/``gpu``/``modelName`` labels).
"""

import pytest
from iris.cluster.backends.k8s.node_metrics import (
    NodeMetrics,
    NodeStatsCollector,
    NodeStatsScraper,
    NodeTarget,
    build_node_stat,
    parse_dcgm,
    parse_node_exporter,
    parse_prometheus,
)
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.stats.tables import IrisWorkerStat, WorkerStatus
from iris.test_util import FakeStatsTable

NODE_EXPORTER_TEXT = """
# HELP node_memory_MemTotal_bytes Memory information field MemTotal_bytes.
# TYPE node_memory_MemTotal_bytes gauge
node_memory_MemAvailable_bytes 2.077648867328e+12
node_memory_MemTotal_bytes 2.162529861632e+12
node_cpu_seconds_total{cpu="0",mode="idle"} 2.45937328e+06
node_cpu_seconds_total{cpu="0",mode="system"} 66990.58
node_cpu_seconds_total{cpu="0",mode="user"} 84582.76
node_cpu_seconds_total{cpu="1",mode="idle"} 2.37498481e+06
node_cpu_seconds_total{cpu="1",mode="user"} 239660.76
node_filesystem_size_bytes{device="/dev/sda",fstype="ext4",mountpoint="/"} 5.0e+10
node_filesystem_avail_bytes{device="/dev/sda",fstype="ext4",mountpoint="/"} 1.0e+10
node_filesystem_size_bytes{device="/dev/md127",fstype="xfs",mountpoint="/mnt/local"} 3.0723257925632e+13
node_filesystem_avail_bytes{device="/dev/md127",fstype="xfs",mountpoint="/mnt/local"} 2.788129107968e+13
node_network_receive_bytes_total{device="enp157s0np0"} 1.0e+15
node_network_receive_bytes_total{device="ibs0"} 4.0e+09
node_network_receive_bytes_total{device="lo"} 9.9e+12
node_network_transmit_bytes_total{device="enp157s0np0"} 8.0e+14
node_network_transmit_bytes_total{device="cilium_host"} 5.0e+11
"""

# Two GPUs on one node. modelName carries spaces; values are integers (MiB / C /
# % / W). FB_USED differs per GPU so the sum is non-trivial.
DCGM_TEXT = """
# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
DCGM_FI_DEV_FB_USED{gpu="0",modelName="NVIDIA H100 80GB HBM3",hostname="g83d142"} 200
DCGM_FI_DEV_FB_TOTAL{gpu="0",modelName="NVIDIA H100 80GB HBM3",hostname="g83d142"} 81281
DCGM_FI_DEV_GPU_TEMP{gpu="0",hostname="g83d142"} 26
DCGM_FI_DEV_GPU_UTIL{gpu="0",hostname="g83d142"} 40
DCGM_FI_DEV_POWER_USAGE{gpu="0",hostname="g83d142"} 300
DCGM_FI_DEV_FB_USED{gpu="1",modelName="NVIDIA H100 80GB HBM3",hostname="g83d142"} 400
DCGM_FI_DEV_FB_TOTAL{gpu="1",modelName="NVIDIA H100 80GB HBM3",hostname="g83d142"} 81281
DCGM_FI_DEV_GPU_TEMP{gpu="1",hostname="g83d142"} 30
DCGM_FI_DEV_GPU_UTIL{gpu="1",hostname="g83d142"} 60
DCGM_FI_DEV_POWER_USAGE{gpu="1",hostname="g83d142"} 350
"""

_MIB = 1024 * 1024


def test_parse_prometheus_handles_labels_values_and_comments():
    samples = list(parse_prometheus('# HELP x\nfoo{a="1",b="two words"} 3.5e+02\nbar 7\n'))
    assert (
        "foo",
        {"a": "1", "b": "two words"},
        350.0,
    ) in samples
    assert ("bar", {}, 7.0) in samples
    # HELP/TYPE comment lines are skipped.
    assert all(name not in ("# HELP", "#") for name, _, _ in samples)


def test_parse_prometheus_skips_non_finite():
    assert list(parse_prometheus("g 1\nh NaN\ni +Inf\n")) == [("g", {}, 1.0)]


def test_parse_node_exporter_extracts_host_readings():
    host = parse_node_exporter(NODE_EXPORTER_TEXT)
    assert host.mem_total_bytes == 2162529861632
    assert host.mem_used_bytes == 2162529861632 - 2077648867328
    # /mnt/local is preferred over "/".
    assert host.disk_total_bytes == 30723257925632
    assert host.disk_used_bytes == 30723257925632 - 27881291079680
    # idle sums across cpus; total sums every mode.
    assert host.cpu_idle_seconds == pytest.approx(2459373.28 + 2374984.81)
    assert host.cpu_total_seconds == pytest.approx(2459373.28 + 66990.58 + 84582.76 + 2374984.81 + 239660.76)
    # Physical NIC + InfiniBand summed; loopback and cilium excluded.
    assert host.net_recv_bytes == int(1.0e15) + int(4.0e09)
    assert host.net_sent_bytes == int(8.0e14)


def test_parse_dcgm_aggregates_across_gpus_by_host():
    samples = parse_dcgm(DCGM_TEXT)
    s = samples["g83d142"]
    assert s.gpu_count == 2
    assert s.gpu_model == "NVIDIA H100 80GB HBM3"
    assert s.hbm_used_bytes == (200 + 400) * _MIB  # summed
    assert s.hbm_total_bytes == (81281 + 81281) * _MIB
    assert s.util_pct == pytest.approx(50.0)  # mean
    assert s.temp_c == pytest.approx(30.0)  # hottest GPU
    assert s.power_w == pytest.approx(650.0)  # summed


def _fetch_from(mapping: dict[str, str]):
    return lambda url: mapping.get(url)


def test_scraper_cpu_pct_needs_two_samples():
    k8s = InMemoryK8sService(namespace="iris")
    mapping = {"http://10.0.0.1:9100/metrics": NODE_EXPORTER_TEXT}
    scraper = NodeStatsScraper(k8s, fetch=_fetch_from(mapping))
    targets = [NodeTarget(name="n1", internal_ip="10.0.0.1")]

    first = scraper.scrape(targets)["n1"]
    assert first.cpu_pct is None  # no prior sample to difference
    assert first.mem_total_bytes == 2162529861632

    # Advance the counters: +100s total, of which +40s idle -> 60% busy.
    busier = NODE_EXPORTER_TEXT.replace(
        'node_cpu_seconds_total{cpu="0",mode="idle"} 2.45937328e+06',
        'node_cpu_seconds_total{cpu="0",mode="idle"} 2459413.28',
    ).replace(
        'node_cpu_seconds_total{cpu="0",mode="user"} 84582.76',
        'node_cpu_seconds_total{cpu="0",mode="user"} 84642.76',
    )
    mapping["http://10.0.0.1:9100/metrics"] = busier
    second = scraper.scrape(targets)["n1"]
    assert second.cpu_pct == pytest.approx(60.0)


def test_scraper_discovers_dcgm_pods_and_merges_gpu_readings():
    k8s = InMemoryK8sService(namespace="iris")
    k8s.seed_namespaced_pod(
        "cw-exporters",
        "dcgm-exporter-abc",
        {
            "metadata": {"name": "dcgm-exporter-abc", "labels": {"app.kubernetes.io/name": "dcgm-exporter"}},
            "status": {"podIP": "10.9.9.9"},
        },
    )
    # A non-dcgm pod in the same namespace must be ignored.
    k8s.seed_namespaced_pod(
        "cw-exporters",
        "node-exporter-xyz",
        {
            "metadata": {"name": "node-exporter-xyz", "labels": {"app.kubernetes.io/name": "node-exporter"}},
            "status": {"podIP": "10.8.8.8"},
        },
    )
    mapping = {
        "http://g83d142:9100/metrics": NODE_EXPORTER_TEXT,
        "http://10.9.9.9:9400/metrics": DCGM_TEXT,
    }
    scraper = NodeStatsScraper(k8s, fetch=_fetch_from(mapping))
    metrics = scraper.scrape([NodeTarget(name="g83d142", internal_ip="g83d142")])["g83d142"]
    assert metrics.gpu_count == 2
    assert metrics.hbm_used_bytes == (200 + 400) * _MIB
    assert metrics.gpu_temp_c == pytest.approx(30.0)
    assert metrics.mem_total_bytes == 2162529861632


def test_scraper_missing_exporter_yields_empty_metrics():
    k8s = InMemoryK8sService(namespace="iris")
    scraper = NodeStatsScraper(k8s, fetch=_fetch_from({}))  # nothing answers
    metrics = scraper.scrape([NodeTarget(name="n1", internal_ip="10.0.0.1")])
    assert metrics["n1"] == NodeMetrics()  # present but all-null, not dropped


def test_build_node_stat_maps_identity_and_metrics():
    target = NodeTarget(
        name="g83d142",
        internal_ip="10.0.0.5",
        status=WorkerStatus.RUNNING,
        device_type="gpu",
        device_variant="H100",
        zone="US-EAST-02",
        cpu_count=192,
        memory_bytes=1_583_533_196_000,
        running_pod_count=3,
    )
    m = NodeMetrics(cpu_pct=12.5, mem_used_bytes=100, mem_total_bytes=200, gpu_count=8, gpu_temp_c=55.0)
    row = build_node_stat(target, m)
    assert isinstance(row, IrisWorkerStat)
    assert row.worker_id == "g83d142"
    assert row.address == "10.0.0.5"
    assert row.status == WorkerStatus.RUNNING
    assert row.cpu_pct == 12.5
    assert row.mem_bytes == 100
    assert row.running_task_count == 3
    assert row.device_variant == "H100"
    assert row.gpu_count == 8
    assert row.gpu_temp_c == 55.0


def test_build_node_stat_falls_back_to_dcgm_for_device_type():
    # Node metadata says nothing about GPUs (unreliable CoreWeave labels), but
    # dcgm reported 8 devices -> the row is classified as a gpu node.
    target = NodeTarget(name="g1", internal_ip="10.0.0.6")
    m = NodeMetrics(gpu_count=8, gpu_model="NVIDIA H100 80GB HBM3")
    row = build_node_stat(target, m)
    assert row.device_type == "gpu"
    assert row.device_variant == "NVIDIA H100 80GB HBM3"


def test_build_node_stat_without_metrics_records_liveness_only():
    target = NodeTarget(name="cpu1", internal_ip="10.0.0.7", status=WorkerStatus.IDLE, cpu_count=64)
    row = build_node_stat(target, None)
    assert row.status == WorkerStatus.IDLE
    assert row.cpu_pct == 0.0
    assert row.gpu_count is None  # nullable device columns stay unset
    assert row.device_type == "cpu"


def test_collector_writes_worker_rows_and_reports_snapshot():
    k8s = InMemoryK8sService(namespace="iris")
    k8s.seed_namespaced_pod(
        "cw-exporters",
        "dcgm-exporter-abc",
        {
            "metadata": {"name": "dcgm-exporter-abc", "labels": {"app.kubernetes.io/name": "dcgm-exporter"}},
            "status": {"podIP": "10.9.9.9"},
        },
    )

    table = FakeStatsTable()
    snapshots: list[dict] = []
    collector = NodeStatsCollector(
        k8s,
        table,
        poll_interval=3600,  # never fires on its own during the test
        on_snapshot=lambda metrics, ts: snapshots.append(metrics),
    )
    try:
        collector.set_nodes([NodeTarget(name="g83d142", internal_ip="g83d142", device_type="gpu")])
        # Patch the scraper's fetch to serve our samples.
        collector._scraper._fetch = _fetch_from({"http://10.9.9.9:9400/metrics": DCGM_TEXT})
        collector.collect_once()
    finally:
        collector.close()

    rows = [r for batch in table.writes for r in batch]
    assert len(rows) == 1
    assert rows[0].worker_id == "g83d142"
    assert rows[0].gpu_count == 2
    assert snapshots and snapshots[0]["g83d142"].gpu_count == 2


def test_collector_no_targets_writes_nothing():
    k8s = InMemoryK8sService(namespace="iris")

    table = FakeStatsTable()
    collector = NodeStatsCollector(k8s, table, poll_interval=3600)
    try:
        collector.collect_once()  # set_nodes never called
    finally:
        collector.close()
    assert table.writes == []
