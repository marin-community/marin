# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CloudK8sService helpers and K8sResource enum path construction."""

from types import SimpleNamespace

import pytest
from iris.cluster.platforms.k8s import service as k8s_service
from iris.cluster.platforms.k8s.service import CloudK8sService
from iris.cluster.platforms.k8s.types import K8sResource


def test_construct_without_kubernetes_client(monkeypatch):
    """Constructing CloudK8sService must not require the kubernetes python client.

    Read-only controller-URL resolution and the kubectl port-forward tunnel run on
    a plain install with only the kubectl binary, so __post_init__ builds just the
    kubectl subprocess prefix and never touches the DynamicClient.
    """
    monkeypatch.setattr(k8s_service, "kubernetes", None)
    monkeypatch.setattr(k8s_service, "DynamicClient", None)

    svc = CloudK8sService(namespace="iris", context="my-ctx")

    assert svc._kubectl_prefix == ["kubectl", "--context", "my-ctx"]


@pytest.mark.parametrize("client_attr", ["_api_client", "_dyn", "_core_v1", "_custom"])
def test_crud_client_requires_kubernetes(monkeypatch, client_attr: str):
    """The kubernetes-client requirement lands at first CRUD use, not construction."""
    monkeypatch.setattr(k8s_service, "kubernetes", None)
    monkeypatch.setattr(k8s_service, "DynamicClient", None)

    svc = CloudK8sService(namespace="iris")

    with pytest.raises(ImportError, match=r"iris\[controller\]"):
        getattr(svc, client_attr)


class _FakeApiServer:
    """Paginating stand-in for one DynamicClient resource handle.

    Serves `pages` of item names, obeying the `continue` query param the way the API
    server does, and records every request it received.
    """

    def __init__(self, pages: list[list[str]]):
        self._pages = pages
        self.requests: list[dict] = []
        self.deletes: list[dict] = []

    def get(self, **kwargs):
        self.requests.append(kwargs)
        index = int(kwargs["_continue"]) if kwargs.get("_continue") else 0
        names = self._pages[index]
        more = index + 1 < len(self._pages)
        return _FakeListResponse(names, str(index + 1) if more else "")

    def delete(self, **kwargs) -> None:
        self.deletes.append(kwargs)


class _FakeListResponse:
    def __init__(self, names: list[str], continue_token: str):
        self.items = [_FakeItem({"metadata": {"name": n}}) for n in names]
        self.metadata = {"continue": continue_token}


class _FakeItem:
    def __init__(self, body: dict):
        self._body = body

    def to_dict(self) -> dict:
        return self._body


def _service_with_api(pages: list[list[str]]) -> tuple[CloudK8sService, _FakeApiServer]:
    """A CloudK8sService whose every resource handle is one fake API server."""
    svc = CloudK8sService(namespace="iris")
    api = _FakeApiServer(pages)
    # Seed the cached_property so no real kubernetes client is built.
    svc.__dict__["_dyn"] = SimpleNamespace(resources=SimpleNamespace(get=lambda **kwargs: api))
    return svc, api


def test_list_json_walks_all_pages():
    """A list must be chunked: an unpaginated response body has no read bound (#7881)."""
    svc, api = _service_with_api([["a", "b"], ["c", "d"], ["e"]])

    names = [pod["metadata"]["name"] for pod in svc.list_json(K8sResource.PODS)]

    assert names == ["a", "b", "c", "d", "e"]
    # Every request carries a page limit: an unpaginated one is the wedge itself.
    assert [req.get("limit") for req in api.requests] == [k8s_service._LIST_PAGE_LIMIT] * 3


def test_iter_json_stops_fetching_when_abandoned():
    """A caller that stops early stops paying: the later pages are never requested."""
    svc, api = _service_with_api([["a", "b"], ["c", "d"], ["e"]])

    first = next(iter(svc.iter_json(K8sResource.PODS)))

    assert first["metadata"]["name"] == "a"
    assert len(api.requests) == 1


def test_delete_by_labels_deletes_each_match_by_name():
    """Deletes go by name, never as a DELETE on the collection URL.

    Kubernetes treats a collection DELETE as the `deletecollection` verb, which the
    controller ClusterRole does not grant, so it would 403 at runtime.
    """
    svc, api = _service_with_api([["a", "b"], ["c"]])

    svc.delete_by_labels(K8sResource.CONFIGMAPS, {"iris.task-hash": "abc"})

    assert [d.get("name") for d in api.deletes] == ["a", "b", "c"]
    assert all(d.get("label_selector") is None for d in api.deletes)


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
        (K8sResource.DAEMONSETS, "ds1", "ns", "/apis/apps/v1/namespaces/ns/daemonsets/ds1"),
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
        (K8sResource.DAEMONSETS, "ns", "/apis/apps/v1/namespaces/ns/daemonsets"),
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
        ("DaemonSet", K8sResource.DAEMONSETS),
        ("StatefulSet", K8sResource.STATEFULSETS),
        ("PodDisruptionBudget", K8sResource.PDBS),
        ("ClusterRole", K8sResource.CLUSTER_ROLES),
        ("ClusterRoleBinding", K8sResource.CLUSTER_ROLE_BINDINGS),
        ("NodePool", K8sResource.NODE_POOLS),
    ],
)
def test_from_kind_valid(kind: str, expected_resource: K8sResource):
    assert K8sResource.from_kind(kind) == expected_resource


def test_from_kind_invalid():
    with pytest.raises(ValueError, match="Unknown kind: 'Bogus'"):
        K8sResource.from_kind("Bogus")


def test_api_base_paths():
    """Test that api_base() returns correct paths for core and custom API groups."""
    assert K8sResource.PODS.api_base() == "/api/v1"
    assert K8sResource.DEPLOYMENTS.api_base() == "/apis/apps/v1"
    assert K8sResource.CLUSTER_ROLES.api_base() == "/apis/rbac.authorization.k8s.io/v1"
    assert K8sResource.NODE_POOLS.api_base() == "/apis/compute.coreweave.com/v1alpha1"
