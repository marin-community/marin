# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes node identity and capacity in the cluster status RPC."""

from iris.cluster.backends.k8s.tasks import _LABEL_MANAGED, _LABEL_RUNTIME, _RUNTIME_LABEL_VALUE
from iris.cluster.platforms.k8s.types import K8sResource

from .conftest import make_batch, make_kueue_provider

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
    provider = make_kueue_provider(k8s, cluster_scan_interval=0.0)
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
    provider = make_kueue_provider(k8s, cluster_scan_interval=0.0)
    _seed_gpu_node(k8s, unschedulable=True)
    try:
        provider.sync(make_batch())
        resp = provider.get_cluster_status()
    finally:
        provider.close()
    node = resp.nodes[0]
    assert node.schedulable is False
    assert node.status_summary == "Ready,SchedulingDisabled"
