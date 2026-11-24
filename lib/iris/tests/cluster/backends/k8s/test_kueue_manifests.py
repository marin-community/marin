# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for the Kueue manifests Iris renders."""

from iris.cluster.platforms.k8s.kueue_manifests import (
    CPU_NOMINAL_QUOTAS,
    build_cks_values,
    build_cluster_queue,
    build_resource_flavor,
    build_upstream_values,
)


def _gate(gates: list[dict], name: str) -> bool | None:
    for gate in gates:
        if gate["name"] == name:
            return gate["enabled"]
    return None


def test_cks_values_disable_tas_balanced_placement():
    # The cks-kueue chart default enables the Alpha TASBalancedPlacement gate, whose
    # balanced-placement scheduler divides by the selected-domain count and panics
    # (integer divide by zero) at zero domains, crashing the controller-manager and
    # dropping the admission-webhook endpoints — which fail-closes every pod CREATE in
    # the Iris namespace. Iris pins the gate OFF.
    gates = build_cks_values(["iris"])["kueue"]["controllerManager"]["featureGates"]
    assert _gate(gates, "TASBalancedPlacement") is False
    # TAS itself, and the multi-layer topology the sliced multi-rack placement rides on,
    # stay ON.
    assert _gate(gates, "TopologyAwareScheduling") is True
    assert _gate(gates, "TASMultiLayerTopology") is True
    assert _gate(gates, "TASRecomputeAssignmentWithinSchedulingCycle") is True


def test_upstream_values_never_enable_tas_balanced_placement():
    # Upstream Kueue defaults TASBalancedPlacement off; the upstream variant must not turn
    # it on (leaving it unset keeps the safe default).
    gates = build_upstream_values(["iris"])["controllerManager"]["featureGates"]
    assert _gate(gates, "TASBalancedPlacement") in (None, False)
    assert _gate(gates, "TopologyAwareScheduling") is True
    assert _gate(gates, "TASRecomputeAssignmentWithinSchedulingCycle") is True


def test_resource_flavor_and_cluster_queue_use_one_all_node_tas_flavor():
    flavor = build_resource_flavor()
    queue = build_cluster_queue(
        "iris-cq",
        nominal_quotas={"nvidia.com/gpu": "512", "rdma/ib": "512"},
    )

    assert flavor["metadata"]["name"] == "cw-tas"
    assert flavor["spec"]["nodeLabels"] == {"iris.kueue": "true"}
    assert [entry["name"] for entry in queue["spec"]["resourceGroups"][0]["flavors"]] == ["cw-tas"]
    assert queue["spec"]["preemption"] == {"withinClusterQueue": "LowerPriority"}
    resources = {
        resource["name"]: resource["nominalQuota"]
        for resource in queue["spec"]["resourceGroups"][0]["flavors"][0]["resources"]
    }
    assert resources["nvidia.com/gpu"] == "512"
    assert resources["rdma/ib"] == "512"


def test_cpu_flavor_is_preferred_without_splitting_accelerator_resources_or_changing_gpu_quota():
    gpu_quotas = {"nvidia.com/gpu": "864", "rdma/ib": "864"}
    original = build_cluster_queue("iris-cq", nominal_quotas=gpu_quotas)
    queue = build_cluster_queue("iris-cq", nominal_quotas=gpu_quotas, cpu_nominal_quotas=CPU_NOMINAL_QUOTAS)
    group = queue["spec"]["resourceGroups"][0]
    assert len(queue["spec"]["resourceGroups"]) == 1
    assert [flavor["name"] for flavor in group["flavors"]] == ["iris-cpu", "cw-tas"]
    assert group["flavors"][1] == original["spec"]["resourceGroups"][0]["flavors"][0]
    quotas = {entry["name"]: entry["nominalQuota"] for entry in group["flavors"][0]["resources"]}
    assert set(quotas) == set(group["coveredResources"])
    assert quotas["nvidia.com/gpu"] == "0" and quotas["rdma/ib"] == "0"
    assert queue["spec"]["preemption"] == original["spec"]["preemption"]


def test_cpu_flavor_lists_covered_resources_in_queue_order_regardless_of_mapping_order():
    """Kueue validates every flavor's resources against one coveredResources order."""
    scrambled = {"rdma/ib": "0", "memory": "1Pi", "nvidia.com/gpu": "0", "cpu": "1G", "ephemeral-storage": "1Pi"}
    queue = build_cluster_queue("iris-cq", cpu_nominal_quotas=scrambled)
    group = queue["spec"]["resourceGroups"][0]
    covered = group["coveredResources"]
    for flavor in group["flavors"]:
        assert [entry["name"] for entry in flavor["resources"]] == covered
