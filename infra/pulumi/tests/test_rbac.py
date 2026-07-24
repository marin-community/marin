# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave RBAC contracts owned by the Pulumi stack."""

from iac.config import load_provisioning
from iac.coreweave.rbac import GRAFANA_OBSERVER_ROLE, grafana_observer_manifests


def test_grafana_observer_can_only_read_nodes():
    role, binding = grafana_observer_manifests("cwtoken-observer")

    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["nodes"],
            "verbs": ["get", "list", "watch"],
        }
    ]
    assert binding["roleRef"]["name"] == GRAFANA_OBSERVER_ROLE
    assert binding["subjects"] == [
        {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "User",
            "name": "cwtoken-observer",
        }
    ]


def test_grafana_clusters_bind_the_same_observer_identity():
    usernames = {
        load_provisioning(cluster).coreweave.grafana_observer_rbac.username
        for cluster in ("cw-us-east-02a", "cw-us-east-08a", "cw-rno2a")
    }

    assert len(usernames) == 1
