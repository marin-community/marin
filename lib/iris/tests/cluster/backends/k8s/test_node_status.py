# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8s backend surfaces nodes as workers: iris.worker rows + NodeStatus in the cluster RPC."""

from iris.cluster.backends.k8s.tasks import _LABEL_MANAGED, _LABEL_RUNTIME, _RUNTIME_LABEL_VALUE
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.stats.tables import IrisWorkerStat, WorkerStatus
from iris.test_util import FakeStatsTable

from .conftest import make_batch, make_kueue_provider

NODE_EXPORTER_TEXT = """
node_memory_MemAvailable_bytes 2.077648867328e+12
node_memory_MemTotal_bytes 2.162529861632e+12
node_cpu_seconds_total{cpu="0",mode="idle"} 1000
node_cpu_seconds_total{cpu="0",mode="user"} 200
node_filesystem_size_bytes{mountpoint="/mnt/local"} 3.0723257925632e+13
node_filesystem_avail_bytes{mountpoint="/mnt/local"} 2.788129107968e+13
node_network_receive_bytes_total{device="enp0"} 1000
node_network_transmit_bytes_total{device="enp0"} 2000
"""

DCGM_TEXT = """
DCGM_FI_DEV_FB_USED{gpu="0",modelName="NVIDIA H100 80GB HBM3",hostname="g83d142"} 200
DCGM_FI_DEV_FB_TOTAL{gpu="0",modelName="NVIDIA H100 80GB HBM3",hostname="g83d142"} 81281
DCGM_FI_DEV_GPU_TEMP{gpu="0",hostname="g83d142"} 45
DCGM_FI_DEV_GPU_UTIL{gpu="0",hostname="g83d142"} 70
DCGM_FI_DEV_POWER_USAGE{gpu="0",hostname="g83d142"} 500
"""

_NODE_NAME = "g83d142"


def _seed_gpu_node(k8s, name=_NODE_NAME, ip="10.0.0.9", *, unschedulable=False, ready=True):
    node = {
        "kind": "Node",
        "metadata": {
            "name": name,
            "labels": {
                "node.kubernetes.io/instance-type": "gd-8xh100ib-i128",
                "topology.kubernetes.io/region": "US-EAST-02",
                "gpu.nvidia.com/model": "H100",
            },
            "creationTimestamp": "2026-06-09T12:45:32Z",
        },
        "spec": {"unschedulable": unschedulable},
        "status": {
            "allocatable": {
                "cpu": "191960m",
                "memory": "1583533196Ki",
                "ephemeral-storage": "7294177710093",
                "nvidia.com/gpu": "8",
            },
            "addresses": [{"type": "InternalIP", "address": ip}],
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }
    k8s.seed_resource(K8sResource.NODES, name, node)


def _seed_running_pod_on(k8s, node_name, pod_name="iris-job-0-0"):
    k8s.seed_resource(
        K8sResource.PODS,
        pod_name,
        {
            "kind": "Pod",
            "metadata": {"name": pod_name, "labels": {_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE}},
            "spec": {"nodeName": node_name},
            "status": {"phase": "Running", "containerStatuses": []},
        },
    )


def test_status_response_lists_nodes_with_identity(k8s):
    provider = make_kueue_provider(
        k8s, worker_stats_table=FakeStatsTable(), cluster_scan_interval=0.0, node_stats_poll_interval=3600
    )
    _seed_gpu_node(k8s)
    _seed_running_pod_on(k8s, _NODE_NAME)
    try:
        provider.sync(make_batch())
        resp = provider.get_cluster_status()
    finally:
        provider.close()

    assert resp.total_nodes == 1
    node = next(n for n in resp.nodes if n.name == _NODE_NAME)
    assert node.ready is True
    assert node.schedulable is True
    assert node.status_summary == "Ready"
    assert node.instance_type == "gd-8xh100ib-i128"
    assert node.region == "US-EAST-02"
    assert node.gpu_count == 8
    assert node.cpu_millicores == 191960
    assert node.running_pods == 1
    assert node.created == "2026-06-09T12:45:32Z"


def test_cordoned_node_reported_unschedulable(k8s):
    provider = make_kueue_provider(
        k8s, worker_stats_table=FakeStatsTable(), cluster_scan_interval=0.0, node_stats_poll_interval=3600
    )
    _seed_gpu_node(k8s, unschedulable=True)
    try:
        provider.sync(make_batch())
        resp = provider.get_cluster_status()
    finally:
        provider.close()
    node = resp.nodes[0]
    assert node.schedulable is False
    assert node.status_summary == "Ready,SchedulingDisabled"


def test_scrape_populates_worker_rows_and_live_snapshot(k8s):
    table = FakeStatsTable()
    provider = make_kueue_provider(
        k8s, worker_stats_table=table, cluster_scan_interval=0.0, node_stats_poll_interval=3600
    )
    _seed_gpu_node(k8s)
    _seed_running_pod_on(k8s, _NODE_NAME)
    k8s.seed_namespaced_pod(
        "cw-exporters",
        "dcgm-exporter-abc",
        {
            "metadata": {"name": "dcgm-exporter-abc", "labels": {"app.kubernetes.io/name": "dcgm-exporter"}},
            "status": {"podIP": "10.9.9.9"},
        },
    )
    try:
        # First sync creates + feeds the collector; drive one scrape deterministically.
        provider.sync(make_batch())
        collector = provider._node_stats_collector
        assert collector is not None
        collector._scraper._fetch = lambda url: {
            "http://10.0.0.9:9100/metrics": NODE_EXPORTER_TEXT,
            "http://10.9.9.9:9400/metrics": DCGM_TEXT,
        }.get(url)
        collector.collect_once()

        # A node row landed in the iris.worker table.
        rows = [r for batch in table.writes for r in batch]
        node_row = next(r for r in rows if r.worker_id == _NODE_NAME)
        assert isinstance(node_row, IrisWorkerStat)
        assert node_row.status == WorkerStatus.RUNNING  # a pod runs on it
        assert node_row.gpu_count == 1
        assert node_row.gpu_temp_c == 45.0
        assert node_row.hbm_total_bytes == 81281 * 1024 * 1024

        # The live snapshot now shows in the cluster status RPC.
        resp = provider.get_cluster_status()
        node = next(n for n in resp.nodes if n.name == _NODE_NAME)
        assert node.gpu_util_pct == 70.0
        assert node.gpu_power_w == 500.0
        assert node.mem_total_bytes == 2162529861632
        assert node.metrics_ts > 0
    finally:
        provider.close()
