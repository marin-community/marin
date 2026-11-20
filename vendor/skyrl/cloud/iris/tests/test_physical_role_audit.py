# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check physical locality before model startup."""

import json
import sys
from types import SimpleNamespace

import pytest
from skyrlrecipe.locality import LocalityPlan

from cloud.iris import task_runtime
from cloud.iris.role_topology import FABRIC_LABEL, HOST_LABEL, RoleTopology, node_physical_labels


def fabric_xml(state="Completed"):
    device = (
        f"<gpu><fabric><state>{state}</state><status>Success</status>"
        "<clusterUuid>cluster</clusterUuid><cliqueId>1</cliqueId></fabric></gpu>"
    )
    return f"<nvidia_smi_log>{device * 4}</nvidia_smi_log>"


def role_nodes(topology):
    return [
        {
            "Alive": True,
            "Labels": {
                **topology.node_labels(rank),
                **node_physical_labels(rank=rank, gpus_per_node=4, host=f"host-{rank}", fabric_xml=fabric_xml()),
            },
        }
        for rank in range(topology.num_nodes)
    ]


@pytest.fixture
def audit(tmp_path, monkeypatch):
    def run(topology, nodes):
        monkeypatch.setitem(
            sys.modules,
            "ray",
            SimpleNamespace(
                init=lambda **kwargs: None,
                shutdown=lambda: None,
                nodes=lambda: nodes,
            ),
        )
        task_runtime.verify_role_placement("fixture", topology, tmp_path)
        return json.loads((tmp_path / "role-topology.json").read_text())

    return run


def test_healthy_single_node_writes_a_verified_report(audit):
    topology = RoleTopology(("policy",), LocalityPlan((1,), (0,)))
    report = audit(topology, role_nodes(topology))
    assert len(report["nodes"]) == 1
    assert report["nodes"][0]["fabric"] == "cluster:1"


def test_healthy_multi_node_groups_may_share_a_fabric(audit):
    topology = RoleTopology(("policy", "inference_0"), LocalityPlan((2, 1), (0, 1)))
    report = audit(topology, role_nodes(topology))
    assert len(report["nodes"]) == 3
    assert len({row["host"] for row in report["nodes"]}) == 3
    assert {row["fabric"] for row in report["nodes"]} == {"cluster:1"}


def test_a_group_split_across_fabrics_is_rejected(audit, tmp_path):
    topology = RoleTopology(("policy",), LocalityPlan((2,), (0,)))
    nodes = role_nodes(topology)
    nodes[1]["Labels"][FABRIC_LABEL] = "cluster:2"
    with pytest.raises(ValueError):
        audit(topology, nodes)
    assert not (tmp_path / "role-topology.json").exists()


def test_duplicate_hosts_are_rejected(audit, tmp_path):
    topology = RoleTopology(("policy",), LocalityPlan((2,), (0,)))
    nodes = role_nodes(topology)
    nodes[1]["Labels"][HOST_LABEL] = "host-0"
    with pytest.raises(ValueError):
        audit(topology, nodes)
    assert not (tmp_path / "role-topology.json").exists()


def test_unhealthy_node_fabric_is_rejected():
    with pytest.raises(ValueError):
        node_physical_labels(rank=0, gpus_per_node=4, host="host-0", fabric_xml=fabric_xml("In Progress"))
