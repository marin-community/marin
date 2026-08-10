# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Define Kubernetes namespace and RBAC manifests for Iris controllers."""

# Permissions the iris-controller ServiceAccount needs at runtime.
CLUSTER_ROLE_RULES = [
    {
        "apiGroups": ["compute.coreweave.com"],
        "resources": ["nodepools"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {
        # Bound via ClusterRoleBinding, so this grants pod access in ALL namespaces —
        # required for blocker eviction in kubernetes_provider.preempt_namespaces.
        "apiGroups": [""],
        "resources": ["pods", "pods/exec", "pods/log"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {"apiGroups": [""], "resources": ["nodes"], "verbs": ["get", "list", "watch"]},
    {
        "apiGroups": [""],
        "resources": ["configmaps"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {"apiGroups": ["metrics.k8s.io"], "resources": ["pods"], "verbs": ["get", "list"]},
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
        # Iris creates iris-{production,interactive,batch} PriorityClass objects at
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
    """Return the controller's ServiceAccount."""
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
