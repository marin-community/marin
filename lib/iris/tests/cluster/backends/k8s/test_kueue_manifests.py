# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for the Kueue helm values Iris renders."""

import yaml
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


def test_cks_values_set_client_connection_rate_limit():
    values = build_cks_values(["iris"], client_connection_qps=100, client_connection_burst=200)

    config_yaml = values["kueue"]["managerConfig"]["controllerManagerConfigYaml"]
    config = yaml.safe_load(config_yaml)

    assert config["clientConnection"] == {"qps": 100, "burst": 200}
