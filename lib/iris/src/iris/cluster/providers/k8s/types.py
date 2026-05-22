# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for the k8s cluster layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class KubectlError(RuntimeError):
    """Error raised for kubectl command failures."""


class K8sResource(Enum):
    """Kubernetes resource type with API metadata.

    Each member carries the information needed to construct API URL paths:
    (api_group, api_version, is_namespaced, plural, kind).

    Use the enum members instead of freeform strings when calling K8sService
    methods like get_json, list_json, delete, etc.
    """

    # Core v1
    PODS = ("", "v1", True, "pods", "Pod")
    CONFIGMAPS = ("", "v1", True, "configmaps", "ConfigMap")
    SERVICES = ("", "v1", True, "services", "Service")
    SECRETS = ("", "v1", True, "secrets", "Secret")
    EVENTS = ("", "v1", True, "events", "Event")
    NAMESPACES = ("", "v1", False, "namespaces", "Namespace")
    NODES = ("", "v1", False, "nodes", "Node")
    SERVICE_ACCOUNTS = ("", "v1", True, "serviceaccounts", "ServiceAccount")

    # Apps v1
    DEPLOYMENTS = ("apps", "v1", True, "deployments", "Deployment")
    STATEFULSETS = ("apps", "v1", True, "statefulsets", "StatefulSet")

    # Policy v1
    PDBS = ("policy", "v1", True, "poddisruptionbudgets", "PodDisruptionBudget")

    # RBAC v1
    CLUSTER_ROLES = ("rbac.authorization.k8s.io", "v1", False, "clusterroles", "ClusterRole")
    CLUSTER_ROLE_BINDINGS = ("rbac.authorization.k8s.io", "v1", False, "clusterrolebindings", "ClusterRoleBinding")

    # CoreWeave custom resources
    NODE_POOLS = ("compute.coreweave.com", "v1alpha1", False, "nodepools", "NodePool")

    def __init__(self, api_group: str, api_version: str, is_namespaced: bool, plural: str, kind: str) -> None:
        self.api_group = api_group
        self.api_version = api_version
        self.is_namespaced = is_namespaced
        self.plural = plural
        self.kind = kind

    def api_base(self) -> str:
        """URL prefix for this resource type (e.g. '/api/v1' or '/apis/apps/v1')."""
        if self.api_group:
            return f"/apis/{self.api_group}/{self.api_version}"
        return f"/api/{self.api_version}"

    def collection_path(self, namespace: str | None = None) -> str:
        """URL path for listing/creating resources."""
        base = self.api_base()
        if self.is_namespaced and namespace:
            return f"{base}/namespaces/{namespace}/{self.plural}"
        return f"{base}/{self.plural}"

    def item_path(self, name: str, namespace: str | None = None) -> str:
        """URL path for a specific resource by name."""
        return f"{self.collection_path(namespace)}/{name}"

    @classmethod
    def from_kind(cls, kind: str) -> K8sResource:
        """Look up a resource by its manifest kind string."""
        for member in cls:
            if member.kind == kind:
                return member
        raise ValueError(f"Unknown kind: {kind!r}")


@dataclass
class KubectlLogLine:
    """A single parsed log line from kubectl logs --timestamps."""

    timestamp: datetime
    stream: str  # "stdout" or "stderr"
    data: str


@dataclass
class KubectlLogResult:
    """Result of an incremental log fetch."""

    lines: list[KubectlLogLine]
    last_timestamp: datetime | None


@dataclass(frozen=True)
class ExecResult:
    """Domain type replacing subprocess.CompletedProcess in the K8sService protocol."""

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PodResourceUsage:
    """CPU and memory usage for a single pod."""

    cpu_millicores: int
    memory_bytes: int


def parse_k8s_quantity(val: str) -> int:
    """Parse K8s resource quantity strings like '4000m', '16Gi', '8'.

    Handles binary suffixes (Ki, Mi, Gi, Ti), SI suffixes (K, M, G, T),
    millicore 'm' suffix, and plain integers.
    """
    if not val:
        return 0
    binary_suffixes = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50}
    si_suffixes = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}
    for suffix, mult in binary_suffixes.items():
        if val.endswith(suffix):
            return int(float(val[: -len(suffix)]) * mult)
    for suffix, mult in si_suffixes.items():
        if val.endswith(suffix) and not val.endswith("i"):
            return int(float(val[: -len(suffix)]) * mult)
    if val.endswith("m"):
        return int(val[:-1])
    return int(float(val))


def parse_k8s_cpu(value: str) -> int:
    """Parse Kubernetes CPU notation to millicores.

    Examples: '250m' -> 250, '1' -> 1000, '0.5' -> 500, '2500m' -> 2500
    """
    if value.endswith("m"):
        return int(value[:-1])
    return int(float(value) * 1000)
