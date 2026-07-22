# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for the Kueue helm values Iris renders."""

from iris.cluster.platforms.k8s.kueue_manifests import build_cks_values, build_upstream_values


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


def test_cks_values_default_does_not_touch_manager_resources():
    # Most clusters run fine on the chart's own resource default. build_cks_values must not
    # emit a resources override unless a caller explicitly opts in — bumping every cluster
    # would force an unnecessary controller-manager restart on ones that don't need it.
    controller_manager = build_cks_values(["iris"])["kueue"]["controllerManager"]
    assert "manager" not in controller_manager


def test_cks_values_manager_memory_limit_overrides_only_memory():
    # The chart's own default (2 CPU / 512Mi memory) OOM-crash-loops once a cluster's
    # Kueue-managed Workload count grows large (cw-rno2a hit CrashLoopBackOff at ~2000
    # Workloads on this limit, 2026-07-22) — an explicit per-cluster override, not a raised
    # default (see the no-override test above).
    resources = build_cks_values(["iris"], manager_memory_limit="2Gi")["kueue"]["controllerManager"]["manager"][
        "resources"
    ]
    assert resources == {"limits": {"memory": "2Gi"}, "requests": {"memory": "2Gi"}}
