# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NodePool derivation + manifest rendering.

Guards the CoreWeave create contract: a NodePool must declare exactly one of spec.targetNodes
(node-based, autoscaling) or spec.targetRacks (NVL72 rack-based, fixed) — the API server rejects
a manifest that provides neither.
"""

import pytest
from iac.config import load_iris_config, load_provisioning
from iac.coreweave.cluster import _nodepool_manifest
from iac.coreweave.kueue import cpu_flavor_node_labels
from iac.nodepools import NodePoolSpec, derive_nodepools
from iris.cluster.platforms.k8s.coreweave_topology import (
    CW_LABEL_FABRIC,
    CW_LABEL_LEAFGROUP,
    CW_LABEL_NVLINK_DOMAIN,
    CW_LABEL_SUPERPOD,
    RACK_SIZE,
)
from iris.cluster.platforms.k8s.kueue_manifests import (
    CPU_TOPOLOGY_LEVELS,
    CPU_TOPOLOGY_NAME,
    TOPOLOGIES,
    build_cpu_resource_flavor,
    build_resource_flavor,
    build_topology_cr,
)


def _node_based_spec() -> NodePoolSpec:
    return NodePoolSpec(
        name="cw-x-h100",
        instance_type="gd-8xh100ib-i128",
        min_nodes=32,
        max_nodes=32,
        node_labels={"iris-x-managed": "true"},
    )


def _rack_based_spec() -> NodePoolSpec:
    return NodePoolSpec(
        name="cw-x-gb200",
        instance_type="gb200-4x",
        min_nodes=72,
        max_nodes=72,
        node_labels={"iris-x-managed": "true"},
        autoscaling=False,
        target_racks=4,
    )


def test_node_based_manifest_declares_target_nodes():
    spec = _nodepool_manifest(_node_based_spec())["spec"]
    assert spec["targetNodes"] == 32  # seeded to min; required at create
    assert spec["minNodes"] == 32 and spec["maxNodes"] == 32
    assert spec["autoscaling"] is True
    assert "targetRacks" not in spec


def test_rack_based_manifest_declares_target_racks_and_no_autoscaler():
    spec = _nodepool_manifest(_rack_based_spec())["spec"]
    assert spec["targetRacks"] == 4
    # NVL72 pools don't autoscale — the node-count/autoscaler fields must be absent so the
    # manifest declares exactly one capacity knob.
    for absent in ("targetNodes", "autoscaling", "minNodes", "maxNodes"):
        assert absent not in spec


def test_derive_gb200_pool_is_rack_based():
    pools = {p.name: p for p in derive_nodepools(load_iris_config("cw-us-east-08a"))}
    gb200 = pools["cw-use08a-gb200"]
    # The rack derivation is the invariant; the absolute fleet size lives in the
    # cluster config and grows without this test's involvement.
    assert gb200.target_racks == gb200.max_nodes // RACK_SIZE > 0
    assert gb200.autoscaling is False
    cpu = pools["cw-use08a-cpu-erapids"]
    assert cpu.target_racks is None and cpu.autoscaling is True


def test_derive_nodepools_assigns_unified_tas_labels():
    config = load_iris_config("cw-us-east-08a")
    provisioning = load_provisioning("cw-us-east-08a").coreweave
    assert provisioning is not None
    pools = {p.name: p for p in derive_nodepools(config, cpu_scale_group=provisioning.kueue.cpu_scale_group)}
    gb200 = pools["cw-use08a-gb200"]
    cpu = pools["cw-use08a-cpu-erapids"]

    assert all(pool.node_labels["iris.kueue"] == "true" for pool in pools.values())
    for topology_label in (CW_LABEL_FABRIC, CW_LABEL_SUPERPOD, CW_LABEL_LEAFGROUP, CW_LABEL_NVLINK_DOMAIN):
        assert topology_label not in cpu.node_labels
        assert topology_label not in gb200.node_labels


def test_cpu_flavor_admits_gvisor_nodes_without_any_provider_topology_labels():
    config = load_iris_config("cw-us-east-08a")
    provisioning = load_provisioning("cw-us-east-08a").coreweave
    assert provisioning is not None
    assert provisioning.kueue.cpu_scale_group is not None
    cpu_flavor = build_cpu_resource_flavor(cpu_flavor_node_labels(config, provisioning.kueue.cpu_scale_group))
    cpu_topology = build_topology_cr(CPU_TOPOLOGY_NAME, CPU_TOPOLOGY_LEVELS, "kueue.x-k8s.io/v1beta1")
    gpu_flavor = build_resource_flavor(provisioning.kueue.flavor_topology)
    gpu_levels = TOPOLOGIES[gpu_flavor["spec"]["topologyName"]]
    cpu_labels = {
        "iris.kueue": "true",
        "iris-cw-use08a-managed": "true",
        "iris-cw-use08a-scale-group": "cpu-erapids",
        "node.coreweave.cloud/gvisor": "true",
        "kubernetes.io/hostname": "cpu-host",
    }
    gpu_labels = {
        "iris.kueue": "true",
        "iris-cw-use08a-managed": "true",
        "iris-cw-use08a-scale-group": "gb200",
        **{key: "physical-domain" for key in gpu_levels},
    }
    selector = cpu_flavor["spec"]["nodeLabels"]
    levels = [level["nodeLabel"] for level in cpu_topology["spec"]["levels"]]
    # Kueue v0.18.3 NodeMatchesFlavor requires selector equality and key presence.
    assert all(cpu_labels.get(key) == value for key, value in selector.items())
    assert all(key in cpu_labels for key in levels)
    assert not all(gpu_labels.get(key) == value for key, value in selector.items())
    assert not all(key in cpu_labels for key in gpu_levels)
    # The managed selector on every Iris Pod must not become a flavor key:
    # cw-tas is immutable and lacks it, so that would reject all GPU Pods.
    flavor_keys = selector.keys() | gpu_flavor["spec"]["nodeLabels"].keys()
    assert "iris-cw-use08a-managed" not in flavor_keys


def test_derive_rejects_rack_pool_with_partial_rack():
    config = load_iris_config("cw-us-east-08a")
    # 70 is not a whole number of 18-node racks.
    config.scale_groups["gb200"].max_slices = 70
    with pytest.raises(ValueError, match="whole number of 18-node racks"):
        derive_nodepools(config)
