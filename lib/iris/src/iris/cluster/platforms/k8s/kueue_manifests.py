# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure Kueue manifest + helm-value builders and their constants.

One source of truth for the Kueue configuration Iris installs: the install
script (``scripts/install_kueue.py``) and the IaC component both import these
builders so they render byte-identical manifests. Everything here is pure — the
functions return plain dicts and do no I/O.
"""

from collections.abc import Sequence

import yaml

from iris.cluster.platforms.k8s.coreweave_topology import (
    CW_FLAVOR_INFINIBAND,
    CW_LABEL_FABRIC,
    CW_LABEL_FLAVOR,
    CW_LABEL_LEAFGROUP,
    CW_LABEL_NVLINK_DOMAIN,
    CW_LABEL_SUPERPOD,
)

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

# Standard k8s per-node label, the finest topology level.
_K8S_HOSTNAME_LABEL = "kubernetes.io/hostname"

# Topology CRs. Iris's preferred "leafgroup" topology rides on
# backend.coreweave.cloud/leafgroup and the required "nvlink.domain" topology on
# ds.coreweave.com/nvlink.domain — both are levels here, so TAS can satisfy the
# podset-topology annotations Iris stamps. Label keys come from coreweave_topology
# so the provider, this script, and the kind smoke share one source.
INFINIBAND_LEVELS = [
    CW_LABEL_FABRIC,
    CW_LABEL_SUPERPOD,
    CW_LABEL_LEAFGROUP,
    _K8S_HOSTNAME_LABEL,
]
MULTINODE_NVLINK_IB_LEVELS = [
    CW_LABEL_FABRIC,
    CW_LABEL_SUPERPOD,
    CW_LABEL_LEAFGROUP,
    CW_LABEL_NVLINK_DOMAIN,
    _K8S_HOSTNAME_LABEL,
]
INFINIBAND_TOPOLOGY_NAME = "infiniband"
MULTINODE_TOPOLOGY_NAME = "multinode-nvlink-ib"
TOPOLOGIES = {
    INFINIBAND_TOPOLOGY_NAME: INFINIBAND_LEVELS,
    MULTINODE_TOPOLOGY_NAME: MULTINODE_NVLINK_IB_LEVELS,
}

TOPOLOGY_CRD = "topologies.kueue.x-k8s.io"
RESOURCE_FLAVOR_NAME = "cw-ib"
# Node selector for the cw-ib ResourceFlavor. Kueue requires a topology-aware
# flavor (spec.topologyName set) to carry at least one nodeLabel; CoreWeave stamps
# backend.coreweave.cloud/flavor=infiniband on every IB-fabric node, which is
# exactly the capacity this flavor represents. On kind the smoke harness stamps
# the same label on its worker nodes.
RESOURCE_FLAVOR_NODE_LABELS = {CW_LABEL_FLAVOR: CW_FLAVOR_INFINIBAND}

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
# The pressure signal is Topology-Aware Scheduling, not quota: when TAS cannot place
# a higher-priority Workload on real nodes, Kueue preempts lower-priority Workloads
# occupying the topology to free room. Quota stays non-binding precisely so this
# stays TAS-driven and does not fight the autoscaler.
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

# cw-cpu shares one resourceGroup with cw-ib but zeroes the accelerator quotas, so a
# GPU pod (requesting nvidia.com/gpu + rdma/ib) can never be admitted under it and
# falls through to cw-ib, while a CPU-only pod matches cw-cpu. Listed first so CPU pods
# pick it before the GPU flavor.
CPU_RESOURCE_FLAVOR_NAME = "cw-cpu"
CPU_FLAVOR_QUOTA = {**NON_BINDING_QUOTA, "nvidia.com/gpu": "0", "rdma/ib": "0"}


# --------------------------------------------------------------------------
# Pure builders (return plain dicts; no I/O).
# --------------------------------------------------------------------------
def build_controller_manager_config(pod_namespaces: Sequence[str] = DEFAULT_POD_NAMESPACES) -> dict:
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
    return {
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
    }


def build_cks_values(
    pod_namespaces: Sequence[str] = DEFAULT_POD_NAMESPACES,
    *,
    manager_memory_limit: str | None = None,
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
        build_controller_manager_config(pod_namespaces), default_flow_style=False, sort_keys=False
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

    Defaults to the InfiniBand topology (fabric/superpod/leafgroup) — the only
    levels real H100 IB nodes carry. Pass ``multinode-nvlink-ib`` to also expose
    the nvlink.domain level (the kind smoke does this to mock a GB200 layout and
    exercise the hard/required nvlink.domain placement).
    """
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "ResourceFlavor",
        "metadata": {"name": RESOURCE_FLAVOR_NAME},
        "spec": {
            # nodeLabels select the nodes this flavor represents (the IB fabric).
            # Required by Kueue whenever topologyName is set.
            "nodeLabels": RESOURCE_FLAVOR_NODE_LABELS,
            # Tie the flavor to the Topology so podset-topology annotations resolve.
            "topologyName": topology_name,
        },
    }


def build_cpu_resource_flavor(node_label: tuple[str, str] | None = None) -> dict:
    """Return the cluster-scoped CPU ResourceFlavor (cw-cpu), no topology.

    ``node_label`` as ``(key, value)`` selects those nodes; omitted (the default)
    leaves the spec empty, so Kueue injects no nodeSelector for admitted CPU pods. No
    ``topologyName``: CPU jobs need no topology-aware placement.
    """
    spec: dict = {}
    if node_label is not None:
        spec["nodeLabels"] = {node_label[0]: node_label[1]}
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "ResourceFlavor",
        "metadata": {"name": CPU_RESOURCE_FLAVOR_NAME},
        "spec": spec,
    }


def build_cluster_queue(name: str) -> dict:
    """Return the cluster-scoped, admin-owned ClusterQueue.

    Covers every resource Iris pods request (COVERED_RESOURCES) with a non-binding
    nominalQuota (NON_BINDING_QUOTA) — Kueue does not enforce capacity here (the Iris
    autoscaler does). It DOES enforce priority: ``preemption.withinClusterQueue:
    LowerPriority`` lets a higher-priority pending Workload evict lower-priority
    admitted ones when it cannot otherwise be admitted — including when TAS cannot
    place it (topology pressure), which is how a higher-priority gang reclaims nodes
    from running batch gangs even though quota never binds.

    Both flavors sit in one resourceGroup, cw-cpu first, so CPU pods match cw-cpu and
    GPU pods fall through to cw-ib.
    """
    flavors = [
        {
            "name": CPU_RESOURCE_FLAVOR_NAME,
            "resources": [{"name": r, "nominalQuota": CPU_FLAVOR_QUOTA[r]} for r in COVERED_RESOURCES],
        },
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
