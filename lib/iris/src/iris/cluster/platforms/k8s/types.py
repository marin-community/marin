# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for the k8s cluster layer."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Default PriorityClass names Iris creates at controller startup and uses when
# the cluster config does not override priority_class_names. Override via
# kubernetes_provider.priority_classes.
#
# iris-system is the control plane band (controller, finelog, Kueue manager). It
# is NOT a user band and is never mapped from a PriorityBand — user jobs cannot
# request it. It sits above every user band so a user pod can never preempt the
# control plane off the shared control node; PreemptLowerPriority lets it evict a
# lower-priority pod to stay scheduled when that node is full.
IRIS_PRIORITY_CLASS_SYSTEM = "iris-system"
IRIS_PRIORITY_CLASS_PRODUCTION = "iris-production"
IRIS_PRIORITY_CLASS_INTERACTIVE = "iris-interactive"
IRIS_PRIORITY_CLASS_BATCH = "iris-batch"

# Canonical (name, value, preemptionPolicy) for every PriorityClass Iris owns.
# Single source of truth: the controller applies all of them at startup and the
# Kueue installer reuses the iris-system entry so the manager is pinned to the
# same class. value/preemptionPolicy are immutable on a PriorityClass, so a
# change here forces a delete+recreate (see ensure_priority_classes).
IRIS_PRIORITY_CLASSES: tuple[tuple[str, int, str], ...] = (
    (IRIS_PRIORITY_CLASS_SYSTEM, 10000, "PreemptLowerPriority"),
    (IRIS_PRIORITY_CLASS_PRODUCTION, 1000, "PreemptLowerPriority"),
    (IRIS_PRIORITY_CLASS_INTERACTIVE, 10, "PreemptLowerPriority"),
    (IRIS_PRIORITY_CLASS_BATCH, 0, "Never"),
)


def build_priority_class_manifest(name: str, value: int, preemption_policy: str) -> dict:
    """Build a cluster-scoped PriorityClass manifest dict."""
    return {
        "apiVersion": "scheduling.k8s.io/v1",
        "kind": "PriorityClass",
        "metadata": {"name": name},
        "value": value,
        "preemptionPolicy": preemption_policy,
        "globalDefault": False,
        "description": f"Iris {name.removeprefix('iris-')} priority band",
    }


def iris_priority_class_manifest(name: str) -> dict:
    """PriorityClass manifest for one Iris-owned band, resolved from IRIS_PRIORITY_CLASSES."""
    for class_name, value, preemption_policy in IRIS_PRIORITY_CLASSES:
        if class_name == name:
            return build_priority_class_manifest(class_name, value, preemption_policy)
    raise ValueError(f"unknown Iris priority class {name!r}")


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
    NAMESPACES = ("", "v1", False, "namespaces", "Namespace")
    NODES = ("", "v1", False, "nodes", "Node")
    SERVICE_ACCOUNTS = ("", "v1", True, "serviceaccounts", "ServiceAccount")
    PERSISTENT_VOLUME_CLAIMS = ("", "v1", True, "persistentvolumeclaims", "PersistentVolumeClaim")

    # Apps v1
    DEPLOYMENTS = ("apps", "v1", True, "deployments", "Deployment")
    STATEFULSETS = ("apps", "v1", True, "statefulsets", "StatefulSet")

    # Networking v1
    INGRESSES = ("networking.k8s.io", "v1", True, "ingresses", "Ingress")
    INGRESS_CLASSES = ("networking.k8s.io", "v1", False, "ingressclasses", "IngressClass")

    # Policy v1
    PDBS = ("policy", "v1", True, "poddisruptionbudgets", "PodDisruptionBudget")

    # RBAC v1
    CLUSTER_ROLES = ("rbac.authorization.k8s.io", "v1", False, "clusterroles", "ClusterRole")
    CLUSTER_ROLE_BINDINGS = ("rbac.authorization.k8s.io", "v1", False, "clusterrolebindings", "ClusterRoleBinding")

    # Scheduling v1
    PRIORITY_CLASSES = ("scheduling.k8s.io", "v1", False, "priorityclasses", "PriorityClass")

    # CoreWeave custom resources
    NODE_POOLS = ("compute.coreweave.com", "v1alpha1", False, "nodepools", "NodePool")

    # Kueue: the per-pod-group Workload (gang admission). Kueue names the
    # Workload after the pod-group-name, so Iris can address it directly to
    # release quota when it tears down a coscheduled gang generation.
    WORKLOADS = ("kueue.x-k8s.io", "v1beta1", True, "workloads", "Workload")
    # Kueue: the namespaced LocalQueue Iris reconciles at controller start to
    # bind its namespace to the admin-provisioned ClusterQueue.
    LOCAL_QUEUES = ("kueue.x-k8s.io", "v1beta1", True, "localqueues", "LocalQueue")

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
    def from_kind(cls, kind: str) -> "K8sResource":
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


def parse_k8s_timestamp(value: str) -> datetime:
    """Parse a Kubernetes RFC3339 timestamp into an aware UTC datetime.

    Accepts the ``Z`` zulu suffix that the Kubernetes API emits (e.g.
    ``2024-01-01T00:00:00Z``); fractional seconds beyond microsecond
    precision are truncated by ``datetime.fromisoformat``. Raises
    ``ValueError`` on malformed input.
    """
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
