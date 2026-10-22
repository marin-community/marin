# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Define Kubernetes namespace and RBAC manifests for Iris services."""

from typing import NamedTuple

NODE_AGENT_SERVICE_ACCOUNT = "iris-node-agent"

TASK_SECRET_ROLE_NAME = "iris-task-secret-delivery"


def task_secret_role_manifest(namespace: str) -> dict:
    """Limit submitted credential management to the controller's namespace."""
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {"name": TASK_SECRET_ROLE_NAME, "namespace": namespace},
        "rules": [{"apiGroups": [""], "resources": ["secrets"], "verbs": ["create", "patch", "delete"]}],
    }


def task_secret_binding_manifest(namespace: str, service_account: str) -> dict:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {"name": TASK_SECRET_ROLE_NAME, "namespace": namespace},
        "subjects": [{"kind": "ServiceAccount", "name": service_account, "namespace": namespace}],
        "roleRef": {"kind": "Role", "name": TASK_SECRET_ROLE_NAME, "apiGroup": "rbac.authorization.k8s.io"},
    }


# Permissions the iris-controller ServiceAccount needs at runtime.
CLUSTER_ROLE_RULES = [
    {
        "apiGroups": ["compute.coreweave.com"],
        "resources": ["nodepools"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {
        "apiGroups": [""],
        "resources": ["pods", "pods/exec", "pods/log"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {"apiGroups": [""], "resources": ["nodes"], "verbs": ["get", "list", "watch", "patch"]},
    {"apiGroups": [""], "resources": ["nodes/metrics"], "verbs": ["get"]},
    {
        "apiGroups": [""],
        "resources": ["configmaps"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {
        "apiGroups": ["policy"],
        "resources": ["poddisruptionbudgets"],
        "verbs": ["get", "list", "create", "update", "patch", "delete"],
    },
    {
        # Kueue gang admission: Iris deletes the per-pod-group Workload to release a
        # torn-down gang's reserved quota (Kueue parks it in WaitingForReplacementPods
        # otherwise).
        "apiGroups": ["kueue.x-k8s.io"],
        "resources": ["workloads"],
        "verbs": ["get", "list", "watch", "delete"],
    },
    {
        # Iris creates iris-{system,production,interactive,batch} PriorityClass objects at
        # startup so pods can be stamped without manual setup.
        "apiGroups": ["scheduling.k8s.io"],
        "resources": ["priorityclasses"],
        "verbs": ["get", "create", "update", "patch", "delete"],
    },
]


def cluster_role_name(namespace: str) -> str:
    """Namespace-qualified ClusterRole/ClusterRoleBinding name.

    Qualified so multiple Iris instances on one CKS cluster don't collide on these
    cluster-scoped resources.
    """
    return f"iris-controller-{namespace}"


def namespace_manifest(namespace: str) -> dict:
    """Return the controller's Namespace.

    Declares the default ``kubernetes`` finalizer explicitly: a caller applying this
    under server-side-apply with forced ownership (e.g. Pulumi's ``enable_patch_force``)
    otherwise prunes fields it doesn't declare, silently stripping the finalizer on adopt.
    """
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": namespace},
        "spec": {"finalizers": ["kubernetes"]},
    }


def service_account_manifest(namespace: str, service_account: str) -> dict:
    """Return a namespaced ServiceAccount."""
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": service_account, "namespace": namespace},
    }


def cluster_role_manifest(role_name: str) -> dict:
    """Return the namespace-qualified ClusterRole granting CLUSTER_ROLE_RULES."""
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRole",
        "metadata": {"name": role_name},
        "rules": CLUSTER_ROLE_RULES,
    }


def cluster_role_binding_manifest(role_name: str, namespace: str, service_account: str) -> dict:
    """Return the ClusterRoleBinding tying ``service_account`` to ``role_name``."""
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRoleBinding",
        "metadata": {"name": role_name},
        "subjects": [{"kind": "ServiceAccount", "name": service_account, "namespace": namespace}],
        "roleRef": {"kind": "ClusterRole", "name": role_name, "apiGroup": "rbac.authorization.k8s.io"},
    }


def node_agent_cluster_role_name(namespace: str) -> str:
    return f"{NODE_AGENT_SERVICE_ACCOUNT}-{namespace}"


class NodeAgentRbac(NamedTuple):
    service_account: dict
    cluster_role: dict
    cluster_role_binding: dict


def node_agent_rbac_manifests(namespace: str) -> NodeAgentRbac:
    """Grant Kubernetes node collection and health publication separately."""
    role_name = node_agent_cluster_role_name(namespace)
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRole",
        "metadata": {"name": role_name},
        "rules": [
            # Node identity and the storage-health annotation written by the agent.
            {"apiGroups": [""], "resources": ["nodes"], "verbs": ["get", "patch"]},
            # Task Pods and DCGM exporters live in different namespaces.
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["list"]},
            # The kubelet authorizes /metrics/resource against this subresource.
            {"apiGroups": [""], "resources": ["nodes/metrics"], "verbs": ["get"]},
        ],
    }
    return NodeAgentRbac(
        service_account_manifest(namespace, NODE_AGENT_SERVICE_ACCOUNT),
        role,
        cluster_role_binding_manifest(role_name, namespace, NODE_AGENT_SERVICE_ACCOUNT),
    )
