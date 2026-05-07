# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CloudK8sService helpers and K8sResource enum path construction."""

from __future__ import annotations

import pytest
from iris.cluster.providers.k8s import service as k8s_service
from iris.cluster.providers.k8s.types import K8sResource


class _FakeApiClient:
    def __init__(self) -> None:
        self.request = "normal-request"
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _FakeApiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _FakeCoreV1Api:
    def __init__(self, api_client: _FakeApiClient) -> None:
        self.api_client = api_client

    def connect_get_namespaced_pod_exec(self, **kwargs: object) -> str:
        raise AssertionError("fake stream should own exec method invocation")


def test_exec_stream_does_not_mutate_shared_api_client(monkeypatch: pytest.MonkeyPatch):
    assert k8s_service.kubernetes is not None
    kubernetes = k8s_service.kubernetes

    monkeypatch.setattr(k8s_service, "DynamicClient", lambda api_client: object())
    monkeypatch.setattr(kubernetes.client, "ApiClient", _FakeApiClient)
    monkeypatch.setattr(kubernetes.client, "CoreV1Api", _FakeCoreV1Api)
    monkeypatch.setattr(kubernetes.client, "CustomObjectsApi", lambda api_client: object())
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", lambda: None)

    svc = k8s_service.CloudK8sService(namespace="iris")
    shared_client = svc._api_client
    shared_request = shared_client.request
    stream_client: _FakeApiClient | None = None

    def stream(api_method, **kwargs: object) -> str:
        nonlocal stream_client
        core_v1 = api_method.__self__
        stream_client = core_v1.api_client
        stream_client.request = "websocket-request"
        return "stdout"

    monkeypatch.setattr(kubernetes.stream, "stream", stream)

    result = svc.exec("pod-1", ["true"], container="task", timeout=5)

    assert result.returncode == 0
    assert result.stdout == "stdout"
    assert stream_client is not None
    assert stream_client is not shared_client
    assert shared_client.request == shared_request
    assert stream_client.closed


# Test item_path construction for namespaced resources
@pytest.mark.parametrize(
    "resource,name,namespace,expected",
    [
        (K8sResource.PODS, "mypod", "ns", "/api/v1/namespaces/ns/pods/mypod"),
        (K8sResource.CONFIGMAPS, "cm1", "ns", "/api/v1/namespaces/ns/configmaps/cm1"),
        (K8sResource.SERVICES, "s1", "ns", "/api/v1/namespaces/ns/services/s1"),
        (K8sResource.SECRETS, "sec1", "ns", "/api/v1/namespaces/ns/secrets/sec1"),
        (K8sResource.SERVICE_ACCOUNTS, "sa1", "ns", "/api/v1/namespaces/ns/serviceaccounts/sa1"),
        (K8sResource.DEPLOYMENTS, "d1", "ns", "/apis/apps/v1/namespaces/ns/deployments/d1"),
        (K8sResource.STATEFULSETS, "ss1", "ns", "/apis/apps/v1/namespaces/ns/statefulsets/ss1"),
        (K8sResource.PDBS, "pdb1", "ns", "/apis/policy/v1/namespaces/ns/poddisruptionbudgets/pdb1"),
    ],
)
def test_item_path_namespaced(resource: K8sResource, name: str, namespace: str, expected: str):
    assert resource.item_path(name, namespace) == expected


# Test item_path construction for cluster-scoped resources
@pytest.mark.parametrize(
    "resource,name,expected",
    [
        (K8sResource.NODES, "node1", "/api/v1/nodes/node1"),
        (K8sResource.NAMESPACES, "myns", "/api/v1/namespaces/myns"),
        (K8sResource.CLUSTER_ROLES, "cr1", "/apis/rbac.authorization.k8s.io/v1/clusterroles/cr1"),
        (K8sResource.CLUSTER_ROLE_BINDINGS, "crb1", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/crb1"),
        (K8sResource.NODE_POOLS, "np1", "/apis/compute.coreweave.com/v1alpha1/nodepools/np1"),
    ],
)
def test_item_path_cluster_scoped(resource: K8sResource, name: str, expected: str):
    assert resource.item_path(name) == expected


# Test collection_path for namespaced resources
@pytest.mark.parametrize(
    "resource,namespace,expected",
    [
        (K8sResource.PODS, "ns", "/api/v1/namespaces/ns/pods"),
        (K8sResource.CONFIGMAPS, "ns", "/api/v1/namespaces/ns/configmaps"),
        (K8sResource.DEPLOYMENTS, "ns", "/apis/apps/v1/namespaces/ns/deployments"),
        (K8sResource.PDBS, "ns", "/apis/policy/v1/namespaces/ns/poddisruptionbudgets"),
    ],
)
def test_collection_path_namespaced(resource: K8sResource, namespace: str, expected: str):
    assert resource.collection_path(namespace) == expected


# Test collection_path for cluster-scoped resources
@pytest.mark.parametrize(
    "resource,expected",
    [
        (K8sResource.NODES, "/api/v1/nodes"),
        (K8sResource.NAMESPACES, "/api/v1/namespaces"),
        (K8sResource.CLUSTER_ROLES, "/apis/rbac.authorization.k8s.io/v1/clusterroles"),
        (K8sResource.CLUSTER_ROLE_BINDINGS, "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings"),
        (K8sResource.NODE_POOLS, "/apis/compute.coreweave.com/v1alpha1/nodepools"),
    ],
)
def test_collection_path_cluster_scoped(resource: K8sResource, expected: str):
    assert resource.collection_path() == expected


# Test from_kind mapping
@pytest.mark.parametrize(
    "kind,expected_resource",
    [
        ("Pod", K8sResource.PODS),
        ("ConfigMap", K8sResource.CONFIGMAPS),
        ("Service", K8sResource.SERVICES),
        ("Secret", K8sResource.SECRETS),
        ("ServiceAccount", K8sResource.SERVICE_ACCOUNTS),
        ("Namespace", K8sResource.NAMESPACES),
        ("Node", K8sResource.NODES),
        ("Deployment", K8sResource.DEPLOYMENTS),
        ("StatefulSet", K8sResource.STATEFULSETS),
        ("PodDisruptionBudget", K8sResource.PDBS),
        ("ClusterRole", K8sResource.CLUSTER_ROLES),
        ("ClusterRoleBinding", K8sResource.CLUSTER_ROLE_BINDINGS),
        ("NodePool", K8sResource.NODE_POOLS),
        ("Event", K8sResource.EVENTS),
    ],
)
def test_from_kind_valid(kind: str, expected_resource: K8sResource):
    assert K8sResource.from_kind(kind) == expected_resource


def test_from_kind_invalid():
    with pytest.raises(ValueError, match="Unknown kind: 'Bogus'"):
        K8sResource.from_kind("Bogus")


def test_all_required_kinds_are_enum_members():
    """Every kind that callers pass to apply_json must be in the enum."""
    required_kinds = {
        "Pod",
        "ConfigMap",
        "Service",
        "Secret",
        "ServiceAccount",
        "Namespace",
        "Deployment",
        "PodDisruptionBudget",
        "ClusterRole",
        "ClusterRoleBinding",
        "NodePool",
    }
    enum_kinds = {member.kind for member in K8sResource}
    missing = required_kinds - enum_kinds
    assert not missing, f"Missing kinds in K8sResource: {missing}"


def test_api_base_paths():
    """Test that api_base() returns correct paths for core and custom API groups."""
    assert K8sResource.PODS.api_base() == "/api/v1"
    assert K8sResource.DEPLOYMENTS.api_base() == "/apis/apps/v1"
    assert K8sResource.CLUSTER_ROLES.api_base() == "/apis/rbac.authorization.k8s.io/v1"
    assert K8sResource.NODE_POOLS.api_base() == "/apis/compute.coreweave.com/v1alpha1"
