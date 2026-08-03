# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Least-privilege RBAC rendered for Grafana's CoreWeave observer token."""

from iac.coreweave.rbac import grafana_finelog_probe_manifests


def test_finelog_probe_rbac_can_only_get_the_named_service_proxy():
    role, binding = grafana_finelog_probe_manifests("iris", ("grafana-reader",), "finelog-cw-a")

    assert role["metadata"]["namespace"] == "iris"
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["services/proxy"],
            "resourceNames": ["http:finelog-cw-a:rpc"],
            "verbs": ["get"],
        }
    ]
    assert binding["metadata"]["namespace"] == "iris"
    assert binding["roleRef"]["kind"] == "Role"
    assert binding["subjects"] == [
        {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "User",
            "name": "grafana-reader",
        }
    ]
