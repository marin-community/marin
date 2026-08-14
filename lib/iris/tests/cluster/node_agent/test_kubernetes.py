# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes node-agent scrape and parse tests.

The exporter samples below mirror the real ``node-exporter`` and
``dcgm-exporter`` output on CoreWeave H100 nodes (scientific-notation values,
per-cpu/per-mode CPU counters, ``/mnt/local`` NVMe filesystem, multi-interface
network, and DCGM's ``hostname``/``gpu``/``modelName`` labels).
"""

import os

import pytest
from iris.cluster.node_agent.kubernetes import (
    NodeStatsScraper,
    TaskStatsCollector,
    parse_dcgm,
    parse_kubelet_resource_metrics,
    parse_node_exporter,
    parse_prometheus,
    reclaim_cache,
)
from iris.cluster.node_agent.metrics import NodeMetrics, NodeTarget
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.test_util import FakeStatsTable
from rigging.timing import Duration

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
node_boot_time_seconds 1.752e+09
"""

# Two GPUs on one node. modelName carries spaces; values are integers (MiB / C /
# % / W). FB_USED differs per GPU so the sum is non-trivial.
_DCGM_GPU0 = (
    'gpu="0",UUID="GPU-aaa",pci_bus_id="00000000:1A:00.0",device="nvidia0",'
    'modelName="NVIDIA H100 80GB HBM3",hostname="g83d142"'
)
_DCGM_GPU1 = (
    'gpu="1",UUID="GPU-bbb",pci_bus_id="00000000:1B:00.0",device="nvidia1",'
    'modelName="NVIDIA H100 80GB HBM3",hostname="g83d142"'
)
DCGM_TEXT = f"""
# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
DCGM_FI_DEV_FB_USED{{{_DCGM_GPU0},DCGM_FI_DRIVER_VERSION="570.86.15"}} 200
DCGM_FI_DEV_FB_TOTAL{{{_DCGM_GPU0}}} 81281
DCGM_FI_DEV_GPU_TEMP{{{_DCGM_GPU0}}} 26
DCGM_FI_DEV_GPU_UTIL{{{_DCGM_GPU0}}} 40
DCGM_FI_DEV_POWER_USAGE{{{_DCGM_GPU0}}} 300
DCGM_FI_DEV_POWER_MGMT_LIMIT{{{_DCGM_GPU0}}} 700
DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{_DCGM_GPU0}}} 3
DCGM_FI_DEV_FB_USED{{{_DCGM_GPU1}}} 400
DCGM_FI_DEV_FB_TOTAL{{{_DCGM_GPU1}}} 81281
DCGM_FI_DEV_GPU_TEMP{{{_DCGM_GPU1}}} 30
DCGM_FI_DEV_GPU_UTIL{{{_DCGM_GPU1}}} 60
DCGM_FI_DEV_POWER_USAGE{{{_DCGM_GPU1}}} 350
DCGM_FI_DEV_POWER_MGMT_LIMIT{{{_DCGM_GPU1}}} 700
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


def test_parse_kubelet_resource_metrics_selects_container_resources():
    resources = parse_kubelet_resource_metrics(
        'container_cpu_usage_seconds_total{container="task",pod="pod-a"} 12.5 123\n'
        'container_memory_working_set_bytes{container="task",pod="pod-a"} 1073741824 123\n'
    )

    assert resources[("pod-a", "task")].cpu_seconds == 12.5
    assert resources[("pod-a", "task")].memory_bytes == 1024**3


def test_task_stats_collector_writes_cpu_memory_and_peak():
    k8s = InMemoryK8sService(namespace="iris")
    k8s.seed_resource(
        K8sResource.PODS,
        "pod-a",
        {
            "metadata": {
                "name": "pod-a",
                "labels": {
                    "iris.managed": "true",
                    "iris.runtime": "iris-kubernetes",
                    "iris.attempt_id": "3",
                },
                "annotations": {"iris.task_id": "/long/job/path/workers/0"},
            },
            "spec": {"nodeName": "node-a"},
            "status": {"phase": "Running"},
        },
    )
    table = FakeStatsTable()
    sampled_at = iter((100.0, 110.0))
    collector = TaskStatsCollector(k8s, "node-a", table, clock=lambda: next(sampled_at))
    k8s.set_node_resource_metrics(
        "node-a",
        'container_cpu_usage_seconds_total{container="task",pod="pod-a"} 10\n'
        'container_memory_working_set_bytes{container="task",pod="pod-a"} 1073741824\n',
    )

    collector.collect_once()
    first = table.writes[-1][0]
    assert first.task_id == "/long/job/path/workers/0"
    assert first.attempt_id == 3
    assert first.cpu_millicores == 0
    assert first.memory_mb == 1024
    assert first.memory_peak_mb == 1024

    k8s.set_node_resource_metrics(
        "node-a",
        'container_cpu_usage_seconds_total{container="task",pod="pod-a"} 12\n'
        'container_memory_working_set_bytes{container="task",pod="pod-a"} 536870912\n',
    )
    collector.collect_once()
    second = table.writes[-1][0]
    assert second.cpu_millicores == 200
    assert second.memory_mb == 512
    assert second.memory_peak_mb == 1024


def test_reclaim_cache_removes_stale_entries_after_node_becomes_idle(tmp_path):
    cache_dir = tmp_path / "iris-cache"
    cache_namespace = cache_dir / "cache"
    stale = cache_namespace / "old-model"
    fresh = cache_namespace / "current-model"
    stale.mkdir(parents=True)
    fresh.mkdir()
    (stale / "weights").write_bytes(b"stale")
    (fresh / "weights").write_bytes(b"fresh")
    os.utime(stale / "weights", (100.0, 100.0))
    os.utime(stale, (100.0, 100.0))
    os.utime(fresh, (100.0, 100.0))
    os.utime(fresh / "weights", (950.0, 950.0))

    reclaimed = reclaim_cache(
        cache_dir,
        max_age=Duration.from_seconds(500),
        kubectl=InMemoryK8sService(namespace="iris"),
        node_name="node-a",
        now=1_000.0,
    )

    assert reclaimed == 1
    assert cache_namespace.is_dir()
    assert not stale.exists()
    assert (fresh / "weights").read_bytes() == b"fresh"


@pytest.mark.parametrize("phase", ["Pending", "Running"])
def test_reclaim_cache_with_active_task_preserves_stale_entries(tmp_path, phase):
    cache_dir = tmp_path / "iris-cache"
    stale = cache_dir / "uv-cache" / "old-wheel"
    stale.mkdir(parents=True)
    os.utime(stale, (100.0, 100.0))
    k8s = InMemoryK8sService(namespace="iris")
    k8s.seed_resource(
        K8sResource.PODS,
        "task-pod",
        {
            "metadata": {
                "name": "task-pod",
                "labels": {"iris.managed": "true", "iris.runtime": "iris-kubernetes"},
            },
            "spec": {"nodeName": "node-a"},
            "status": {"phase": phase},
        },
    )

    reclaimed = reclaim_cache(
        cache_dir,
        max_age=Duration.from_seconds(500),
        kubectl=k8s,
        node_name="node-a",
        now=1_000.0,
    )

    assert reclaimed == 0
    assert stale.is_dir()


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
    assert s.power_limit_w == pytest.approx(1400.0)  # summed


def _fetch_from(mapping: dict[str, str]):
    return lambda url: mapping.get(url)


def test_scraper_cpu_pct_needs_two_samples():
    k8s = InMemoryK8sService(namespace="iris")
    mapping = {"http://10.0.0.1:9100/metrics": NODE_EXPORTER_TEXT}
    scraper = NodeStatsScraper(k8s, fetch=_fetch_from(mapping))
    targets = [NodeTarget(name="n1", node_uid="node-uid-1", internal_ip="10.0.0.1")]

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
            "spec": {"nodeName": "g83d142"},
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
    metrics = scraper.scrape([NodeTarget(name="g83d142", node_uid="node-uid-1", internal_ip="g83d142")])["g83d142"]
    assert metrics.gpu_count == 2
    assert metrics.hbm_used_bytes == (200 + 400) * _MIB
    assert metrics.gpu_temp_c == pytest.approx(30.0)
    assert metrics.gpu_power_limit_w == pytest.approx(1400.0)
    assert metrics.mem_total_bytes == 2162529861632


def test_scraper_missing_exporter_yields_empty_metrics():
    k8s = InMemoryK8sService(namespace="iris")
    scraper = NodeStatsScraper(k8s, fetch=_fetch_from({}))  # nothing answers
    metrics = scraper.scrape([NodeTarget(name="n1", node_uid="node-uid-1", internal_ip="10.0.0.1")])
    assert metrics["n1"] == NodeMetrics()  # present but all-null, not dropped
