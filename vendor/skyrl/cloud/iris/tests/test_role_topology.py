from collections import Counter

import pytest

from cloud.iris.role_topology import (
    FABRIC_RESOURCE_PREFIX,
    GPU_RESOURCE_PREFIX,
    node_role_resources,
    role_topology_groups,
    validate_role_nodes,
)
from cloud.iris.tests.launch_config_fixtures import raw_launch_config


def three_rack_nodes():
    config = raw_launch_config()
    config["iris"]["role_topology"] = "nvlink.domain"
    config["iris"]["allocation"].update(num_nodes=48, gpus_per_node=4, gpu_variant="GB200")
    config["skyrl"]["trainer"]["placement"].update(
        colocate_all=False, policy_num_nodes=16, policy_num_gpus_per_node=4,
    )
    config["skyrl"]["generator"].update(
        num_inference_engines=2, inference_engine_data_parallel_size=64,
        inference_engine_expert_parallel_size=64,
    )
    groups = role_topology_groups(config)
    nodes = []
    for rank in range(48):
        devices = "".join(
            f"<gpu><uuid>GPU-{rank}-{index}</uuid><fabric><state>Completed</state><status>Success</status>"
            f"<clusterUuid>cluster</clusterUuid><cliqueId>{rank // 16}</cliqueId></fabric></gpu>"
            for index in range(4)
        )
        resources = node_role_resources(
            groups=groups, rank=rank, num_nodes=48, gpus_per_node=4, host=f"host-{rank}",
            fabric_xml=f"<nvidia_smi_log>{devices}</nvidia_smi_log>",
        )
        nodes.append({"NodeID": f"ray-{rank}", "Alive": True, "Resources": {"GPU": 4, **resources}})
    return groups, nodes


def test_policy_and_inference_replicas_cover_distinct_physical_racks():
    groups, nodes = three_rack_nodes()
    rows = validate_role_nodes(nodes, groups=groups, num_nodes=48, gpus_per_node=4)
    assert Counter(row["role"] for row in rows) == {"policy": 16, "inference_0": 16, "inference_1": 16}
    assert {row["role"]: row["fabric"] for row in rows} == {
        "policy": "cluster:0", "inference_0": "cluster:1", "inference_1": "cluster:2",
    }
    assert {gpu for row in rows for gpu in row["gpu_uuids"]} == {
        f"GPU-{rank}-{index}" for rank in range(48) for index in range(4)
    }


@pytest.mark.parametrize("wrong_layout", ["split_policy", "overlapping_roles", "duplicate_gpu"])
def test_same_gpu_counts_reject_unsafe_physical_role_placement(wrong_layout):
    groups, nodes = three_rack_nodes()
    if wrong_layout == "split_policy":
        resources = nodes[0]["Resources"]
        del resources[FABRIC_RESOURCE_PREFIX + "cluster:0"]
        resources[FABRIC_RESOURCE_PREFIX + "cluster:1"] = 4
    elif wrong_layout == "overlapping_roles":
        for node in nodes[16:32]:
            resources = node["Resources"]
            del resources[FABRIC_RESOURCE_PREFIX + "cluster:1"]
            resources[FABRIC_RESOURCE_PREFIX + "cluster:0"] = 4
    else:
        resources = nodes[1]["Resources"]
        del resources[GPU_RESOURCE_PREFIX + "GPU-1-0"]
        resources[GPU_RESOURCE_PREFIX + "GPU-0-0"] = 1
    with pytest.raises(ValueError, match="physical NVLink domain|duplicate physical GPUs"):
        validate_role_nodes(nodes, groups=groups, num_nodes=48, gpus_per_node=4)
