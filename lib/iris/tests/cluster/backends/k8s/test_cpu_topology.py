# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair the observed 08a CPU topology drift without changing accelerator Nodes."""

from copy import deepcopy
from pathlib import Path

import pytest
from click.testing import CliRunner
from iris.cluster.config import load_config
from iris.cluster.platforms.k8s import cpu_topology
from iris.cluster.platforms.k8s.coreweave_topology import CW_MULTINODE_TOPOLOGY_LABELS
from iris.cluster.platforms.k8s.cpu_topology import apply_cpu_topology_repairs, plan_cpu_topology_repairs
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.nodepool_manifests import CPU_TOPOLOGY_NODE_LABELS, nodepool_manifest
from iris.cluster.platforms.k8s.types import K8sResource, KubectlError

CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "cw-us-east-08a.yaml"


@pytest.fixture
def topology_cluster():
    k8s = InMemoryK8sService(namespace="iris")
    pool = nodepool_manifest("cw-use08a-cpu-erapids", "cd-gp-i64-erapids", node_labels=dict(CPU_TOPOLOGY_NODE_LABELS))
    k8s.seed_resource(K8sResource.NODE_POOLS, pool["metadata"]["name"], pool)
    for name, group, instance, gpu_count in (
        ("cpu", "cpu-erapids", "cd-gp-i64-erapids", "0"),
        ("gpu", "gb200", "gb200-4x", "4"),
        ("foreign", "cpu-erapids", "cd-gp-i64-erapids", "0"),
    ):
        labels = {
            "iris-cw-use08a-managed": "false" if name == "foreign" else "true",
            "iris-cw-use08a-scale-group": group,
            "node.kubernetes.io/instance-type": instance,
            "kubernetes.io/hostname": name,
            "node.coreweave.cloud/gvisor": "true" if name == "cpu" else "false",
            "topology.kubernetes.io/region": "US-EAST-08",
            "backend.coreweave.cloud/fabric": "",
            "backend.coreweave.cloud/superpod": "",
            "backend.coreweave.cloud/leafgroup": "",
        }
        if name == "gpu":
            labels.update({key: "physical-rack" for key in CW_MULTINODE_TOPOLOGY_LABELS})
        k8s.seed_resource(
            K8sResource.NODES,
            name,
            {
                "metadata": {"name": name, "uid": f"uid-{name}", "resourceVersion": "7", "labels": labels},
                "spec": {"taints": [], "unschedulable": False},
                "status": {"allocatable": {"nvidia.com/gpu": gpu_count}},
            },
        )
    return k8s


def test_cpu_topology_repair_restores_gvisor_tas_eligibility_and_preserves_other_nodes(topology_cluster):
    k8s = topology_cluster
    before = deepcopy(k8s.list_json(K8sResource.NODES))
    cpu = k8s.get_json(K8sResource.NODES, "cpu")
    assert "ds.coreweave.com/nvlink.domain" not in cpu["metadata"]["labels"]
    repairs = plan_cpu_topology_repairs(load_config(CONFIG_PATH), before, k8s.list_json(K8sResource.NODE_POOLS))
    assert [r.name for r in repairs] == ["cpu"]
    apply_cpu_topology_repairs(k8s, repairs)
    cpu = k8s.get_json(K8sResource.NODES, "cpu")
    assert all(cpu["metadata"]["labels"][key] == "iris-cpu-only" for key in CW_MULTINODE_TOPOLOGY_LABELS)
    assert cpu["metadata"]["labels"]["node.coreweave.cloud/gvisor"] == "true"
    assert cpu["metadata"]["labels"]["topology.kubernetes.io/region"] == "US-EAST-08"
    assert cpu["spec"] == before[0]["spec"]
    assert k8s.get_json(K8sResource.NODES, "gpu") == before[1]
    assert k8s.get_json(K8sResource.NODES, "foreign") == before[2]
    assert (
        plan_cpu_topology_repairs(
            load_config(CONFIG_PATH), k8s.list_json(K8sResource.NODES), k8s.list_json(K8sResource.NODE_POOLS)
        )
        == []
    )


def test_cpu_topology_repair_refuses_accelerator_capacity_in_cpu_group(topology_cluster):
    k8s = topology_cluster
    cpu = k8s.get_json(K8sResource.NODES, "cpu")
    cpu["status"]["allocatable"]["nvidia.com/gpu"] = "4"
    with pytest.raises(ValueError, match="does not match CPU NodePool"):
        plan_cpu_topology_repairs(
            load_config(CONFIG_PATH), k8s.list_json(K8sResource.NODES), k8s.list_json(K8sResource.NODE_POOLS)
        )


def test_cpu_topology_repair_refuses_drifted_pool_declaration(topology_cluster):
    k8s = topology_cluster
    pool = k8s.get_json(K8sResource.NODE_POOLS, "cw-use08a-cpu-erapids")
    del pool["spec"]["nodeLabels"]["ds.coreweave.com/nvlink.domain"]
    with pytest.raises(ValueError, match="repair IaC first"):
        plan_cpu_topology_repairs(
            load_config(CONFIG_PATH), k8s.list_json(K8sResource.NODES), k8s.list_json(K8sResource.NODE_POOLS)
        )


def test_cpu_topology_repair_refuses_concurrent_node_changes(topology_cluster):
    k8s = topology_cluster
    repairs = plan_cpu_topology_repairs(
        load_config(CONFIG_PATH), k8s.list_json(K8sResource.NODES), k8s.list_json(K8sResource.NODE_POOLS)
    )
    k8s.patch_node("cpu", {"spec": {"unschedulable": True}})
    with pytest.raises(KubectlError, match="resourceVersion conflict"):
        apply_cpu_topology_repairs(k8s, repairs)
    assert "ds.coreweave.com/nvlink.domain" not in k8s.get_json(K8sResource.NODES, "cpu")["metadata"]["labels"]


def test_cpu_topology_command_previews_before_explicit_apply(topology_cluster, monkeypatch):
    k8s = topology_cluster
    monkeypatch.setattr(cpu_topology, "CloudK8sService", lambda **kwargs: k8s)
    runner = CliRunner()
    preview = runner.invoke(cpu_topology.main, ["--config", str(CONFIG_PATH)])
    assert preview.exit_code == 0, preview.output
    assert '"name": "cpu"' in preview.output
    assert "ds.coreweave.com/nvlink.domain" not in k8s.get_json(K8sResource.NODES, "cpu")["metadata"]["labels"]
    applied = runner.invoke(cpu_topology.main, ["--config", str(CONFIG_PATH), "--apply"])
    assert applied.exit_code == 0, applied.output
    assert (
        k8s.get_json(K8sResource.NODES, "cpu")["metadata"]["labels"]["ds.coreweave.com/nvlink.domain"] == "iris-cpu-only"
    )
