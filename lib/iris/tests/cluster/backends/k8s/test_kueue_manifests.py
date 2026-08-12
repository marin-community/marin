# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for the Kueue manifests Iris renders."""

from iris.cluster.platforms.k8s.kueue_manifests import (
    IRIS_WORKLOAD_PRIORITY_CLASSES,
    WorkloadPriorityKind,
    build_cks_values,
    build_cluster_queue,
    build_resource_flavor,
    build_upstream_values,
)
from iris.cluster.platforms.k8s.types import IRIS_PRIORITY_CLASS_SYSTEM, IRIS_PRIORITY_CLASSES


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


def test_upstream_values_never_enable_tas_balanced_placement():
    # Upstream Kueue defaults TASBalancedPlacement off; the upstream variant must not turn
    # it on (leaving it unset keeps the safe default).
    gates = build_upstream_values(["iris"])["controllerManager"]["featureGates"]
    assert _gate(gates, "TASBalancedPlacement") in (None, False)
    assert _gate(gates, "TopologyAwareScheduling") is True


def test_resource_flavor_and_cluster_queue_use_one_all_node_tas_flavor():
    flavor = build_resource_flavor()
    queue = build_cluster_queue("iris-cq")

    assert flavor["metadata"]["name"] == "cw-tas"
    assert flavor["spec"]["nodeLabels"] == {"iris.kueue": "true"}
    assert [entry["name"] for entry in queue["spec"]["resourceGroups"][0]["flavors"]] == ["cw-tas"]
    assert queue["spec"]["preemption"] == {"withinClusterQueue": "LowerPriority"}


def test_workload_priority_classes_order_cpu_accelerator_and_gang_within_each_band():
    native_values = {
        class_name.removeprefix("iris-"): value
        for class_name, value, _ in IRIS_PRIORITY_CLASSES
        if class_name != IRIS_PRIORITY_CLASS_SYSTEM
    }
    values = {
        (priority_class.band, priority_class.kind): priority_class.value
        for priority_class in IRIS_WORKLOAD_PRIORITY_CLASSES
    }

    for band, gang_value in native_values.items():
        assert values[band, WorkloadPriorityKind.CPU] == gang_value - 2
        assert values[band, WorkloadPriorityKind.ACCELERATOR] == gang_value - 1
