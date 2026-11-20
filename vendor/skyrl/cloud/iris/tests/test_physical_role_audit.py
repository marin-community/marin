# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify distinct physical GPUs and NVLink domains for every model role."""

from collections import Counter
import json
import sys
from types import SimpleNamespace

import pytest
from skyrlrecipe.locality import LocalityPlan

from cloud.iris import task_runtime
from cloud.iris.role_topology import (
    CLIQUE_LABEL,
    GPU_LABEL_PREFIX,
    LOCALITY_GROUP_LABEL,
    RoleTopology,
    node_physical_labels,
    validate_role_nodes,
)


GPUS_PER_NODE = 4


def three_rack_nodes(topology=None):
    if topology is None:
        topology = RoleTopology(("policy", "inference_0", "inference_1"), LocalityPlan((16, 16, 16), (0, 1, 2)))
    nodes = []
    for rank in range(topology.num_nodes):
        devices = "".join(
            f"<gpu><uuid>GPU-{rank}-{index}</uuid><fabric><state>Completed</state><status>Success</status>"
            f"<clusterUuid>cluster</clusterUuid><cliqueId>{topology.locality.group_for_rank(rank)}</cliqueId></fabric></gpu>"
            for index in range(GPUS_PER_NODE)
        )
        labels = {
            **topology.node_labels(rank),
            **node_physical_labels(
                rank=rank,
                gpus_per_node=GPUS_PER_NODE,
                host=f"host-{rank}",
                fabric_xml=f"<nvidia_smi_log>{devices}</nvidia_smi_log>",
            ),
        }
        nodes.append({"NodeID": f"ray-{rank}", "Alive": True, "Resources": {"GPU": GPUS_PER_NODE}, "Labels": labels})
    return topology, nodes


def test_policy_and_inference_replicas_cover_distinct_physical_racks():
    topology, nodes = three_rack_nodes()
    rows = validate_role_nodes(nodes, topology=topology, gpus_per_node=GPUS_PER_NODE)
    assert Counter(row.roles[0] for row in rows) == {"policy": 16, "inference_0": 16, "inference_1": 16}
    assert {row.roles[0]: row.fabric for row in rows} == {
        "policy": "cluster:0",
        "inference_0": "cluster:1",
        "inference_1": "cluster:2",
    }
    assert {gpu for row in rows for gpu in row.gpu_uuids} == {
        f"GPU-{rank}-{index}" for rank in range(48) for index in range(GPUS_PER_NODE)
    }


@pytest.mark.parametrize("wrong_layout", ["split_policy", "duplicate_gpu", "wrong_label"])
def test_same_gpu_counts_reject_unsafe_physical_role_placement(wrong_layout):
    topology, nodes = three_rack_nodes()
    if wrong_layout == "split_policy":
        nodes[0]["Labels"][CLIQUE_LABEL] = "1"
    elif wrong_layout == "wrong_label":
        nodes[0]["Labels"][LOCALITY_GROUP_LABEL] = "1"
    else:
        nodes[1]["Labels"][GPU_LABEL_PREFIX + "0"] = "GPU-0-0"
    with pytest.raises(ValueError):
        validate_role_nodes(nodes, topology=topology, gpus_per_node=GPUS_PER_NODE)


def test_shared_domain_roles_use_the_planned_group_boundaries():
    topology, nodes = three_rack_nodes(
        RoleTopology(
            ("policy", *(f"inference_{index}" for index in range(7))),
            LocalityPlan((15, 14, 5), (0, 0, 0, 0, 1, 1, 1, 2)),
        )
    )
    rows = validate_role_nodes(nodes, topology=topology, gpus_per_node=GPUS_PER_NODE)
    assert rows[14].roles == ("policy", "inference_0", "inference_1", "inference_2")
    assert rows[15].roles == ("inference_3", "inference_4", "inference_5")
    assert rows[29].roles == ("inference_6",)


@pytest.mark.parametrize("split", [False, True])
def test_head_join_gate_writes_only_a_verified_physical_report(tmp_path, monkeypatch, split):
    topology, nodes = three_rack_nodes()
    if split:
        nodes[0]["Labels"][CLIQUE_LABEL] = "1"
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            init=lambda **_kwargs: None,
            shutdown=lambda: None,
            nodes=lambda: nodes,
            cluster_resources=lambda: {"GPU": 192},
        ),
    )

    def join():
        task_runtime.verify_role_placement("fixture", topology, tmp_path)

    output = tmp_path / "role-topology.json"
    if split:
        with pytest.raises(ValueError):
            join()
        assert not output.exists()
    else:
        join()
        report = json.loads(output.read_text())
        assert len(report["nodes"]) == 48
        assert len({gpu for row in report["nodes"] for gpu in row["gpu_uuids"]}) == 192


def test_separate_model_groups_may_share_one_physical_rack():
    topology, nodes = three_rack_nodes(RoleTopology(("policy", "inference_0"), LocalityPlan((16, 2), (0, 1))))
    for node in nodes:
        node["Labels"][CLIQUE_LABEL] = "0"
    rows = validate_role_nodes(nodes, topology=topology, gpus_per_node=GPUS_PER_NODE)
    assert len(rows) == 18
    assert {row.fabric for row in rows} == {"cluster:0"}


@pytest.mark.parametrize("gpu_capacity", [4, 3])
def test_single_node_audit_checks_ray_capacity_and_writes_report(tmp_path, monkeypatch, gpu_capacity):
    topology, nodes = three_rack_nodes(RoleTopology(("policy",), LocalityPlan((1,), (0,))))
    nodes[0]["Resources"]["GPU"] = gpu_capacity
    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(init=lambda **kwargs: None, shutdown=lambda: None, nodes=lambda: nodes)
    )
    if gpu_capacity != 4:
        with pytest.raises(ValueError):
            task_runtime.verify_role_placement("fixture", topology, tmp_path)
        assert not (tmp_path / "role-topology.json").exists()
        return
    task_runtime.verify_role_placement("fixture", topology, tmp_path)
    report = json.loads((tmp_path / "role-topology.json").read_text())
    assert report["nodes"][0]["gpu_uuids"] == [f"GPU-0-{i}" for i in range(4)]
