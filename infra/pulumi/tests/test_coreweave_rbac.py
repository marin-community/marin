# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iac.coreweave.rbac import GRAFANA_OBSERVER_ROLE, grafana_observer_manifests


def test_grafana_observer_can_read_node_inventory_without_mutating_it():
    role, binding = grafana_observer_manifests(("cwtoken-current", "cwtoken-next"))

    assert role["metadata"]["name"] == GRAFANA_OBSERVER_ROLE
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["nodes"], "verbs": ["get", "list", "watch"]},
        {
            "apiGroups": ["compute.coreweave.com"],
            "resources": ["nodepools"],
            "verbs": ["get", "list", "watch"],
        },
    ]
    assert binding["roleRef"]["name"] == GRAFANA_OBSERVER_ROLE
    assert [subject["name"] for subject in binding["subjects"]] == ["cwtoken-current", "cwtoken-next"]
