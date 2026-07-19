# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NodePool derivation + manifest rendering.

Guards the CoreWeave create contract: a NodePool must declare exactly one of spec.targetNodes
(node-based, autoscaling) or spec.targetRacks (NVL72 rack-based, fixed) — the API server rejects
a manifest that provides neither.
"""

import pytest
from iac.config import load_iris_config
from iac.coreweave.cluster import _nodepool_manifest
from iac.nodepools import RACK_SIZE, NodePoolSpec, derive_nodepools


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


def test_derive_rejects_rack_pool_with_partial_rack():
    config = load_iris_config("cw-us-east-08a")
    # 70 is not a whole number of 18-node racks.
    config.scale_groups["gb200"].max_slices = 70
    with pytest.raises(ValueError, match="whole number of 18-node racks"):
        derive_nodepools(config)
