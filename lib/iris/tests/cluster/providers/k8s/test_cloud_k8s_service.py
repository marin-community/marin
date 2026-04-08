# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CloudK8sService URL construction and resource dispatch.

These test the pure functions that construct API paths from resource types
and manifests. They caught regressions during the kubectl-to-Python-client
migration where missing kind entries caused CI failures.
"""

from __future__ import annotations

import pytest

from iris.cluster.providers.k8s.service import (
    _KIND_TO_PLURAL,
    _RESOURCE_INFO,
    _api_path_for_manifest,
    _api_path_for_resource,
)


# Every manifest kind used in the codebase must resolve.
_ALL_MANIFESTS = [
    ({"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p1"}}, "/api/v1/namespaces/ns/pods/p1"),
    ({"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm1"}}, "/api/v1/namespaces/ns/configmaps/cm1"),
    ({"apiVersion": "v1", "kind": "Service", "metadata": {"name": "s1"}}, "/api/v1/namespaces/ns/services/s1"),
    ({"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "sec1"}}, "/api/v1/namespaces/ns/secrets/sec1"),
    (
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": "sa1"}},
        "/api/v1/namespaces/ns/serviceaccounts/sa1",
    ),
    ({"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "myns"}}, "/api/v1/namespaces/myns"),
    (
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "d1"}},
        "/apis/apps/v1/namespaces/ns/deployments/d1",
    ),
    (
        {"apiVersion": "policy/v1", "kind": "PodDisruptionBudget", "metadata": {"name": "pdb1"}},
        "/apis/policy/v1/namespaces/ns/poddisruptionbudgets/pdb1",
    ),
    (
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRole", "metadata": {"name": "cr1"}},
        "/apis/rbac.authorization.k8s.io/v1/clusterroles/cr1",
    ),
    (
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRoleBinding", "metadata": {"name": "crb1"}},
        "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/crb1",
    ),
    (
        {"apiVersion": "compute.coreweave.com/v1alpha1", "kind": "NodePool", "metadata": {"name": "np1"}},
        "/apis/compute.coreweave.com/v1alpha1/nodepools/np1",
    ),
]


@pytest.mark.parametrize("manifest,expected_path", _ALL_MANIFESTS, ids=[m["kind"] for m, _ in _ALL_MANIFESTS])
def test_api_path_for_manifest(manifest: dict, expected_path: str):
    assert _api_path_for_manifest(manifest, "ns") == expected_path


def test_api_path_for_manifest_unknown_kind():
    with pytest.raises(ValueError, match="Unknown kind"):
        _api_path_for_manifest({"apiVersion": "v1", "kind": "Bogus", "metadata": {"name": "x"}}, "ns")


# Every resource string used by callers must resolve.
_RESOURCE_CALLERS = [
    ("pods", False, "/api/v1/namespaces/ns/pods"),
    ("pod", False, "/api/v1/namespaces/ns/pods"),
    ("configmaps", False, "/api/v1/namespaces/ns/configmaps"),
    ("configmap", False, "/api/v1/namespaces/ns/configmaps"),
    ("poddisruptionbudgets", False, "/apis/policy/v1/namespaces/ns/poddisruptionbudgets"),
    ("pdb", False, "/apis/policy/v1/namespaces/ns/poddisruptionbudgets"),
    ("nodes", True, "/api/v1/nodes"),
    ("deployment", False, "/apis/apps/v1/namespaces/ns/deployments"),
    ("service", False, "/api/v1/namespaces/ns/services"),
    ("secret", False, "/api/v1/namespaces/ns/secrets"),
    ("clusterrole", True, "/apis/rbac.authorization.k8s.io/v1/clusterroles"),
    ("clusterrolebinding", True, "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings"),
    ("nodepool", True, "/apis/compute.coreweave.com/v1alpha1/nodepools"),
    ("nodepools", True, "/apis/compute.coreweave.com/v1alpha1/nodepools"),
    ("events", False, "/api/v1/namespaces/ns/events"),
]


@pytest.mark.parametrize("resource,cluster_scoped,expected", _RESOURCE_CALLERS, ids=[r for r, _, _ in _RESOURCE_CALLERS])
def test_api_path_for_resource(resource: str, cluster_scoped: bool, expected: str):
    assert _api_path_for_resource(resource, "ns", cluster_scoped=cluster_scoped) == expected


def test_api_path_for_resource_with_name():
    path = _api_path_for_resource("pods", "ns", name="mypod")
    assert path == "/api/v1/namespaces/ns/pods/mypod"


def test_api_path_for_resource_unknown():
    with pytest.raises(ValueError, match="Unknown resource type"):
        _api_path_for_resource("bogus", "ns")


def test_kind_to_plural_covers_all_applied_kinds():
    """Every kind that callers pass to apply_json must be in _KIND_TO_PLURAL."""
    # These are all kinds used in controller.py and tasks.py apply_json calls.
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
    missing = required_kinds - set(_KIND_TO_PLURAL.keys())
    assert not missing, f"Missing kinds in _KIND_TO_PLURAL: {missing}"


def test_resource_info_covers_all_caller_strings():
    """Every resource string used by callers must be in _RESOURCE_INFO."""
    required = {
        "pods",
        "pod",
        "configmaps",
        "configmap",
        "services",
        "service",
        "secrets",
        "secret",
        "events",
        "nodes",
        "deployments",
        "deployment",
        "poddisruptionbudgets",
        "pdb",
        "clusterroles",
        "clusterrole",
        "clusterrolebindings",
        "clusterrolebinding",
        "nodepools",
        "nodepool",
    }
    missing = required - set(_RESOURCE_INFO.keys())
    assert not missing, f"Missing resource types in _RESOURCE_INFO: {missing}"
