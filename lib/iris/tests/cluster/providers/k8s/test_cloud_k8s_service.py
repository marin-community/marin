# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CloudK8sService helpers and K8sResource enum path construction."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from iris.cluster.providers.k8s import service as k8s_service
from iris.cluster.providers.k8s.types import K8sResource


class _LogResponse:
    def __init__(self, data: bytes):
        self.data = data
        self.released = False

    def release_conn(self):
        self.released = True


class _CoreV1WithLogResponse:
    def __init__(self, response: _LogResponse):
        self.response = response
        self.kwargs = None

    def read_namespaced_pod_log(self, **kwargs):
        self.kwargs = kwargs
        return self.response


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


def test_bearer_token_alias_is_added_for_incluster_auth():
    if k8s_service.kubernetes is None:
        pytest.skip("kubernetes client is not installed")

    config = k8s_service.kubernetes.client.Configuration()
    config.api_key["authorization"] = "bearer token-1"

    k8s_service._keep_bearer_token_alias_fresh(config)

    assert config.api_key["BearerToken"] == "bearer token-1"
    assert config.get_api_key_with_prefix("BearerToken") == "bearer token-1"


def test_bearer_token_alias_tracks_incluster_token_refresh():
    if k8s_service.kubernetes is None:
        pytest.skip("kubernetes client is not installed")

    config = k8s_service.kubernetes.client.Configuration()
    config.api_key["authorization"] = "bearer token-1"

    def refresh(client_configuration):
        client_configuration.api_key["authorization"] = "bearer token-2"
        client_configuration.refresh_api_key_hook = refresh

    config.refresh_api_key_hook = refresh
    k8s_service._keep_bearer_token_alias_fresh(config)

    assert config.get_api_key_with_prefix("BearerToken") == "bearer token-2"
    assert config.api_key["BearerToken"] == "bearer token-2"
    assert config.refresh_api_key_hook is not refresh


def test_create_api_client_syncs_bearer_token_for_kubeconfig(monkeypatch):
    if k8s_service.kubernetes is None:
        pytest.skip("kubernetes client is not installed")

    config = k8s_service.kubernetes.client.Configuration()
    config.api_key["authorization"] = "bearer token-1"
    api_client = k8s_service.kubernetes.client.ApiClient(config)

    def new_client_from_config(config_file=None):
        assert config_file == "/tmp/kubeconfig"
        return api_client

    monkeypatch.setattr(k8s_service.kubernetes.config, "new_client_from_config", new_client_from_config)
    service = k8s_service.CloudK8sService.__new__(k8s_service.CloudK8sService)
    service.kubeconfig_path = "/tmp/kubeconfig"

    assert service.create_api_client() is api_client
    assert config.api_key["BearerToken"] == "bearer token-1"


def test_logs_decode_raw_response_bytes():
    response = _LogResponse(b"line 1\nline 2\n")
    core_v1 = _CoreV1WithLogResponse(response)
    service = k8s_service.CloudK8sService.__new__(k8s_service.CloudK8sService)
    service.namespace = "test-ns"
    service.timeout = 60.0
    service._core_v1 = core_v1

    assert service.logs("pod-1", container="task", tail=10) == "line 1\nline 2\n"
    assert core_v1.kwargs["_preload_content"] is False
    assert core_v1.kwargs["container"] == "task"
    assert response.released


def test_stream_logs_decodes_raw_response_bytes():
    response = _LogResponse(
        b"2026-05-22T16:45:12.158721540Z I20260522 16:45:12 iris.test.verbose info-marker\n"
        b"2026-05-22T16:45:12.158728857Z W20260522 16:45:12 iris.test.verbose warning-marker\n"
    )
    core_v1 = _CoreV1WithLogResponse(response)
    service = k8s_service.CloudK8sService.__new__(k8s_service.CloudK8sService)
    service.namespace = "test-ns"
    service._core_v1 = core_v1

    result = service.stream_logs("pod-1", container="task", limit_bytes=100_000)

    assert core_v1.kwargs["_preload_content"] is False
    assert core_v1.kwargs["container"] == "task"
    assert response.released
    assert [line.data for line in result.lines] == [
        "I20260522 16:45:12 iris.test.verbose info-marker",
        "W20260522 16:45:12 iris.test.verbose warning-marker",
    ]
    assert result.last_timestamp == datetime(2026, 5, 22, 16, 45, 12, 158728, tzinfo=timezone.utc)
