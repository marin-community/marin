# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iac.config import load_provisioning
from iac.coreweave.rbac import (
    GRAFANA_OBSERVER_ROLE,
    LOOM_SESSION_ROLE,
    grafana_observer_manifests,
    loom_session_manifests,
)
from iris.cluster.platforms.k8s.rbac_manifests import node_agent_rbac_manifests

COREWEAVE_CLUSTERS = ("cw-us-east-02a", "cw-us-east-08a", "cw-rno2a", "cw-us-west-04a")


def test_node_agent_can_collect_and_publish_health_without_controller_authority():
    account, role, binding = node_agent_rbac_manifests("iris")

    assert account["metadata"] == {"name": "iris-node-agent", "namespace": "iris"}
    assert binding["subjects"] == [{"kind": "ServiceAccount", **account["metadata"]}]
    assert binding["roleRef"] == {
        "kind": "ClusterRole",
        "name": role["metadata"]["name"],
        "apiGroup": "rbac.authorization.k8s.io",
    }
    permissions = {
        (group, resource, verb)
        for rule in role["rules"]
        for group in rule["apiGroups"]
        for resource in rule["resources"]
        for verb in rule["verbs"]
    }
    # This is the complete grant: workload creation/exec, secrets and TokenReview
    # must not become available to the host-network collector.
    assert permissions == {
        ("", "nodes", "get"),
        ("", "nodes", "patch"),
        ("", "pods", "list"),
        ("", "nodes/metrics", "get"),
    }


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


def test_loom_session_can_connect_to_namespaced_pods_without_mutating_objects():
    role, binding = loom_session_manifests("iris", ("cwtoken-current", "cwtoken-next"))

    assert role["metadata"] == {
        "name": LOOM_SESSION_ROLE,
        "namespace": "iris",
        "labels": {
            "app.kubernetes.io/name": "marin-loom",
            "app.kubernetes.io/component": "session-connect",
        },
    }
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["pods/exec", "pods/portforward"],
            "verbs": ["create"],
        }
    ]
    assert binding["metadata"] == {
        "name": LOOM_SESSION_ROLE,
        "namespace": "iris",
        "labels": {
            "app.kubernetes.io/name": "marin-loom",
            "app.kubernetes.io/component": "session-connect",
        },
    }
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": LOOM_SESSION_ROLE,
    }
    assert [subject["name"] for subject in binding["subjects"]] == ["cwtoken-current", "cwtoken-next"]


def test_loom_session_principals_remain_consistent_across_coreweave_clusters():
    usernames = []
    for cluster in COREWEAVE_CLUSTERS:
        coreweave = load_provisioning(cluster).coreweave
        assert coreweave is not None
        assert coreweave.loom_session_rbac is not None
        usernames.append(coreweave.loom_session_rbac.usernames)

    assert len(set(usernames)) == 1
