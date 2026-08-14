# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure Kueue manifest + helm-value builders and their constants.

One source of truth for the Kueue configuration Iris installs: the install
script (``scripts/install_kueue.py``) and the IaC component both import these
builders so they render byte-identical manifests. Everything here is pure — the
functions return plain dicts and do no I/O.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import yaml

from iris.cluster.platforms.k8s.coreweave_topology import (
    CW_INFINIBAND_TOPOLOGY_LABELS,
    CW_MULTINODE_TOPOLOGY_LABELS,
)
from iris.cluster.platforms.k8s.nodepool_manifests import KUEUE_NODE_LABEL
from iris.cluster.platforms.k8s.types import IRIS_PRIORITY_CLASS_SYSTEM, IRIS_PRIORITY_CLASSES

# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
VARIANT_COREWEAVE = "coreweave"
VARIANT_UPSTREAM = "upstream"

# CoreWeave cks-kueue chart (wraps upstream kueue as a subchart).
CW_REPO_NAME = "coreweave"
CW_REPO_URL = "https://charts.core-services.ingress.coreweave.com"
CW_CHART = f"{CW_REPO_NAME}/cks-kueue"

RELEASE_DEFAULT = "kueue"
OPERATOR_NS = "kueue-system"

# Controller-manager feature gates for the cks-kueue chart. Helm replaces list values
# wholesale, so this enumerates the full set the chart ships with, changing one entry:
# TASBalancedPlacement stays OFF. That Alpha gate's balanced-placement scheduler divides
# the pod-slice count by the number of selected topology domains and panics (integer
# divide by zero) when that count is zero, crashing the controller-manager process — which
# drops the admission-webhook endpoints and fail-closes every pod CREATE in the Iris
# namespace. Iris requests explicit per-rack slice sizes for balanced multi-rack placement
# (podset-slice-size, under TopologyAwareScheduling), so it never relies on this heuristic;
# every other gate stays at the chart default.
CKS_KUEUE_FEATURE_GATES = [
    {"name": "VisibilityOnDemand", "enabled": True},
    {"name": "LendingLimit", "enabled": True},
    {"name": "ObjectRetentionPolicies", "enabled": True},
    {"name": "TopologyAwareScheduling", "enabled": True},
    {"name": "TASBalancedPlacement", "enabled": False},
    {"name": "TASMultiLayerTopology", "enabled": True},
]

# Namespace(s) Iris submits gang pods into (the k8s provider namespace, default
# "iris"). Kueue's admission webhooks are scoped to ONLY these — see
# build_controller_manager_config for why a broad selector is dangerous.
DEFAULT_POD_NAMESPACES = ("iris",)

# Kueue's Kubernetes API client is shared by reconcilers, event recording, and
# leader election. Iris clusters routinely carry enough Pods and Workloads for
# Kueue's upstream 20-QPS/30-burst defaults to delay a lease renewal during a
# restart resync.
DEFAULT_CLIENT_CONNECTION_QPS = 100.0
DEFAULT_CLIENT_CONNECTION_BURST = 200

# Standard k8s per-node label, the finest topology level.
_K8S_HOSTNAME_LABEL = "kubernetes.io/hostname"

# Topology CRs. Iris's preferred "leafgroup" topology rides on
# backend.coreweave.cloud/leafgroup and the required "nvlink.domain" topology on
# ds.coreweave.com/nvlink.domain — both are levels here, so TAS can satisfy the
# podset-topology annotations Iris stamps. Label keys come from coreweave_topology
# so the provider, this script, and the kind smoke share one source.
INFINIBAND_LEVELS = [*CW_INFINIBAND_TOPOLOGY_LABELS, _K8S_HOSTNAME_LABEL]
MULTINODE_NVLINK_IB_LEVELS = [*CW_MULTINODE_TOPOLOGY_LABELS, _K8S_HOSTNAME_LABEL]
INFINIBAND_TOPOLOGY_NAME = "infiniband"
MULTINODE_TOPOLOGY_NAME = "multinode-nvlink-ib"
TOPOLOGIES = {
    INFINIBAND_TOPOLOGY_NAME: INFINIBAND_LEVELS,
    MULTINODE_TOPOLOGY_NAME: MULTINODE_NVLINK_IB_LEVELS,
}

TOPOLOGY_CRD = "topologies.kueue.x-k8s.io"
RESOURCE_FLAVOR_NAME = "cw-tas"
# One TAS ResourceFlavor spans every Iris-managed CoreWeave NodePool. CPU
# NodePools carry synthetic topology levels so Kueue can assign them at the
# hostname level; GPU/RDMA requests exclude those nodes by allocatable capacity.
RESOURCE_FLAVOR_NODE_LABELS = {KUEUE_NODE_LABEL: "true"}

# Resources the ClusterQueue covers when --with-queues is set. A Kueue
# ClusterQueue can only admit a workload if *every* resource the pods request is
# covered here AND has a nominalQuota; an uncovered resource leaves the workload
# stuck at QuotaReserved=False (pods SchedulingGated) forever. Iris IB-GPU pods
# request cpu/memory/nvidia.com/gpu plus ephemeral-storage (from the disk request)
# and rdma/ib (the InfiniBand devices), so all five must be covered.
#
# Iris does NOT use Kueue for capacity *enforcement*: the Iris autoscaler bounds
# capacity via scale-group max_slices, so every resource's nominalQuota is a
# sentinel large enough never to bind — Kueue never rejects on quota, and the real
# capacity authority stays the scheduler/autoscaler.
#
# It DOES use Kueue for preemption (see build_cluster_queue's preemption stanza).
# The pressure signal is Topology-Aware Scheduling, not quota. Every Iris Pod
# uses the same TAS flavor so simulated victim removal can reclaim lower-priority
# CPU reservations before retrying a GPU gang's topology fit. Quota stays
# non-binding so it does not fight the autoscaler.
NON_BINDING_QUOTA = {
    # Use "1G" not "1000000000" because the Kubernetes API server canonicalizes to 1G
    # and always returns that, which causes a perpetual, cosmetic `pulumi preview` diff
    "cpu": "1G",  # cores
    "memory": "1Pi",
    "ephemeral-storage": "1Pi",
    "nvidia.com/gpu": "1G",
    "rdma/ib": "1G",
}
COVERED_RESOURCES = list(NON_BINDING_QUOTA)


# Kueue-only priorities within each Iris band. Ordinary, standalone-accelerator,
# and co-scheduled Workloads occupy consecutive values starting at the native
# band. This keeps the lowest batch Workload priority at 0, above CoreWeave's
# priority -1 node-health-check Pods even though Kueue and Pod priority are
# separate scheduling domains.
class WorkloadPriorityKind(StrEnum):
    CPU = "cpu"
    ACCELERATOR = "accelerator"
    COSCHEDULED = "coscheduled"


@dataclass(frozen=True)
class IrisWorkloadPriorityClass:
    band: str
    kind: WorkloadPriorityKind
    name: str
    value: int


def workload_priority_class_name(band: str, kind: WorkloadPriorityKind) -> str:
    return f"iris-{kind.value}-{band}"


IRIS_WORKLOAD_PRIORITY_CLASSES = tuple(
    IrisWorkloadPriorityClass(
        band=class_name.removeprefix("iris-"),
        kind=kind,
        name=workload_priority_class_name(class_name.removeprefix("iris-"), kind),
        value=value + offset,
    )
    for class_name, value, _ in IRIS_PRIORITY_CLASSES
    if class_name != IRIS_PRIORITY_CLASS_SYSTEM
    for kind, offset in (
        (WorkloadPriorityKind.CPU, 0),
        (WorkloadPriorityKind.ACCELERATOR, 1),
        (WorkloadPriorityKind.COSCHEDULED, 2),
    )
)


# --------------------------------------------------------------------------
# Pure builders (return plain dicts; no I/O).
# --------------------------------------------------------------------------


def build_controller_manager_config(
    pod_namespaces: Sequence[str] = DEFAULT_POD_NAMESPACES,
    *,
    client_connection_qps: float = DEFAULT_CLIENT_CONNECTION_QPS,
    client_connection_burst: int = DEFAULT_CLIENT_CONNECTION_BURST,
) -> dict:
    """Return the kueue ``Configuration`` (controller-manager config) as a dict.

    Serialized to YAML and embedded as the chart's ``controllerManagerConfigYaml``
    value. Enables the "pod" framework (gang admission for plain pods) alongside
    "batch/job" cluster-wide. ``manageJobsWithoutQueueName`` stays false so Kueue
    only *gates* pods carrying ``kueue.x-k8s.io/queue-name`` (the ones Iris stamps).
    internalCertManagement is enabled so Kueue self-signs its webhook certs (no
    cert-manager dependency); the names match both charts' webhook service/secret.

    ``managedJobsNamespaceSelector`` scopes Kueue's *admission webhooks* to only
    ``pod_namespaces`` (the namespace Iris submits into). This is critical and
    separate from ``manageJobsWithoutQueueName``: that flag governs whether Kueue
    *gates* an already-intercepted pod, but the fail-closed webhooks still
    *intercept* every CREATE in every selected namespace. Both charts' webhook
    templates render each webhook's ``namespaceSelector`` from this top-level key
    (NOT from the legacy ``integrations.podOptions.namespaceSelector``, which
    never reaches the webhook objects), falling back to a broad selector that
    excludes only kube-system + the release namespace. On a shared CoreWeave
    cluster that broad default intercepts CNI/system pods (e.g. cilium in
    cw-cilium-system): a freshly delivered node's CNI pod is gated by a webhook
    it can't reach (no network yet) → the pod is rejected → the node never goes
    Ready. Opt-in scoping keeps the webhooks off every namespace but our own.
    """
    config: dict[str, object] = {
        "apiVersion": "config.kueue.x-k8s.io/v1beta1",
        "kind": "Configuration",
        "health": {"healthProbeBindAddress": ":8081"},
        "metrics": {"bindAddress": ":8080"},
        "webhook": {"port": 9443},
        "manageJobsWithoutQueueName": False,
        # Rendered by the charts into every webhook's namespaceSelector; also
        # scopes controller-side management. Must NOT match kube-system or the
        # kueue namespace (kueue config validation rejects it).
        "managedJobsNamespaceSelector": {
            "matchExpressions": [
                {
                    "key": "kubernetes.io/metadata.name",
                    "operator": "In",
                    "values": list(pod_namespaces),
                }
            ]
        },
        "internalCertManagement": {
            "enable": True,
            "webhookServiceName": "kueue-webhook-service",
            "webhookSecretName": "kueue-webhook-server-cert",
        },
        "integrations": {
            "frameworks": ["batch/job", "pod"],
        },
        "clientConnection": {
            "qps": client_connection_qps,
            "burst": client_connection_burst,
        },
    }
    return config


def build_cks_values(
    pod_namespaces: Sequence[str] = DEFAULT_POD_NAMESPACES,
    *,
    manager_memory_limit: str | None = None,
    client_connection_qps: float = DEFAULT_CLIENT_CONNECTION_QPS,
    client_connection_burst: int = DEFAULT_CLIENT_CONNECTION_BURST,
) -> dict:
    """Return the ``cks-kueue`` (CoreWeave) helm values (managerConfig only).

    cks-kueue nests the upstream kueue subchart under ``kueue:``. The chart's
    ``topologies:`` value is deliberately NOT set — it renders Topology CRs at an
    apiVersion the CRD no longer serves (see module docstring); the Topology CRs
    are kubectl-applied after install instead.

    ``controllerManager.featureGates`` is CKS_KUEUE_FEATURE_GATES — the chart's own
    list shape with the crash-prone TASBalancedPlacement gate turned off. The chart
    takes this value as a *list*; overriding it as a map breaks the chart's
    ``kueue.featureGates`` template.

    ``manager_memory_limit``, when set, overrides ``controllerManager.manager.resources``
    (requests == limits for memory). CPU is left out of the override: Helm deep-merges map
    values against the chart's own ``values.yaml`` (unlike lists, which replace wholesale —
    see the featureGates note above), so omitting ``cpu`` here preserves the chart's own CPU
    request/limit instead of duplicating it.
    """
    config_yaml = yaml.safe_dump(
        build_controller_manager_config(
            pod_namespaces,
            client_connection_qps=client_connection_qps,
            client_connection_burst=client_connection_burst,
        ),
        default_flow_style=False,
        sort_keys=False,
    )
    controller_manager: dict = {"featureGates": CKS_KUEUE_FEATURE_GATES}
    if manager_memory_limit is not None:
        controller_manager["manager"] = {
            "resources": {
                "limits": {"memory": manager_memory_limit},
                "requests": {"memory": manager_memory_limit},
            }
        }
    return {
        "kueue": {
            "enableKueueViz": False,
            "controllerManager": controller_manager,
            "managerConfig": {"controllerManagerConfigYaml": config_yaml},
        },
    }


def build_upstream_values(pod_namespaces: Sequence[str] = DEFAULT_POD_NAMESPACES) -> dict:
    """Return the upstream Kueue OCI-chart helm values.

    The upstream chart puts ``managerConfig`` at the top level and takes feature
    gates as a *list* under ``controllerManager.featureGates``. TopologyAwareScheduling
    is NOT on by default upstream, so we enable it here.
    """
    config_yaml = yaml.safe_dump(
        build_controller_manager_config(pod_namespaces), default_flow_style=False, sort_keys=False
    )
    return {
        "enableKueueViz": False,
        "controllerManager": {
            "featureGates": [{"name": "TopologyAwareScheduling", "enabled": True}],
        },
        "managerConfig": {"controllerManagerConfigYaml": config_yaml},
    }


def build_topology_cr(name: str, levels: list[str], api_version: str) -> dict:
    """Return a Topology CR dict at ``api_version`` (the CRD's served version)."""
    return {
        "apiVersion": api_version,
        "kind": "Topology",
        "metadata": {"name": name},
        "spec": {"levels": [{"nodeLabel": label} for label in levels]},
    }


def build_resource_flavor(topology_name: str = INFINIBAND_TOPOLOGY_NAME) -> dict:
    """Return the cluster-scoped ResourceFlavor tied to the named Kueue Topology.

    Defaults to the InfiniBand topology (fabric/superpod/leafgroup). CPU
    NodePools carry a synthetic value for these levels; pass
    ``multinode-nvlink-ib`` to also expose nvlink.domain for GB200 placement.
    """
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "ResourceFlavor",
        "metadata": {"name": RESOURCE_FLAVOR_NAME},
        "spec": {
            # Kueue requires at least one nodeLabel when topologyName is set.
            "nodeLabels": RESOURCE_FLAVOR_NODE_LABELS,
            # Tie the flavor to the Topology so podset-topology annotations resolve.
            "topologyName": topology_name,
        },
    }


def build_workload_priority_class(name: str, value: int) -> dict:
    """Return a Kueue WorkloadPriorityClass for Iris admission ordering."""
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "WorkloadPriorityClass",
        "metadata": {"name": name},
        "value": value,
        "description": "Iris workload admission priority within its user-selected band",
    }


def build_cluster_queue(name: str) -> dict:
    """Return the cluster-scoped, admin-owned ClusterQueue.

    Covers every resource Iris pods request (COVERED_RESOURCES) with a non-binding
    nominalQuota (NON_BINDING_QUOTA) — Kueue does not enforce capacity here (the Iris
    autoscaler does). It DOES enforce priority: ``preemption.withinClusterQueue:
    LowerPriority`` lets a higher-priority pending Workload evict compatible
    lower-priority admitted Workloads when it cannot otherwise be admitted. One
    flavor covers all resources and nodes so CPU reservations on accelerator
    nodes are compatible preemption candidates for GPU gangs.
    """
    flavors = [
        {
            "name": RESOURCE_FLAVOR_NAME,
            "resources": [{"name": r, "nominalQuota": NON_BINDING_QUOTA[r]} for r in COVERED_RESOURCES],
        },
    ]
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "ClusterQueue",
        "metadata": {"name": name},
        "spec": {
            "namespaceSelector": {},
            "preemption": {"withinClusterQueue": "LowerPriority"},
            "resourceGroups": [
                {
                    "coveredResources": COVERED_RESOURCES,
                    "flavors": flavors,
                }
            ],
        },
    }
