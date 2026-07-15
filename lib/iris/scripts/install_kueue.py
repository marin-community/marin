#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install + configure Kueue for Iris gang admission, on CoreWeave or upstream.

Iris's K8s direct provider relies on Kueue's *plain-Pod* integration to gang-admit
coscheduled pod groups (all-or-nothing) and on Topology-Aware Scheduling (TAS) to
honor the podset-topology annotations it stamps. Neither is on by default in any
Kueue chart, so this script renders that configuration and installs it.

Two variants share one code path (``--variant``):

  * ``coreweave`` — the CoreWeave ``cks-kueue`` helm chart (wraps upstream kueue).
  * ``upstream`` — the upstream OCI helm chart
    (``oci://registry.k8s.io/kueue/charts/kueue``), used for kind / generic
    clusters. TAS is enabled via ``controllerManager`` feature gates. The smoke
    harness (tests/e2e/gpu_gang_smoke.py) drives this variant on kind.

Neither variant uses cks-kueue's ``topologies:`` values templating: the chart
(1.3.0) renders Topology CRs at ``kueue.x-k8s.io/v1alpha1`` while the CRD it
itself installs serves only v1beta1+, so any helm pass carrying ``topologies``
fails with 'no matches for kind "Topology"'. Instead both variants apply the
Topology CRs with kubectl after install, at the apiVersion the installed CRD
actually serves.

Both variants:
  1. Install the operator into ``kueue-system`` (``helm upgrade --install``).
  2. Enable the plain-Pod integration via the controller-manager ``Configuration``
     (``integrations.frameworks: ["batch/job","pod"]``). ``manageJobsWithoutQueueName``
     stays false, so Kueue only gates pods carrying ``kueue.x-k8s.io/queue-name``
     (the ones Iris stamps). The *admission webhooks* are opt-in scoped via the
     top-level ``managedJobsNamespaceSelector`` (which both charts render into
     every webhook's ``namespaceSelector``) to only ``--pod-namespace`` (default
     ``iris``) — NOT the chart default (every namespace except
     kube-system/kueue-system), which fail-closed-intercepts CNI/system pods on a
     shared cluster and deadlocks node delivery. See build_controller_manager_config.
  3. Create the Topology CRs (infiniband + multinode-nvlink-ib) so TAS can resolve
     the podset-topology annotations (``backend.coreweave.cloud/leafgroup``,
     ``ds.coreweave.com/nvlink.domain``).
  4. (``--with-queues``) Create the cluster-scoped, admin-owned ResourceFlavor
     (``cw-ib``, selecting ``backend.coreweave.cloud/flavor=infiniband`` nodes) +
     ClusterQueue. The ClusterQueue enables priority preemption within the queue
     (``preemption.withinClusterQueue: LowerPriority``): a higher-priority Workload
     evicts lower-priority admitted ones when it cannot otherwise be admitted —
     including when TAS cannot place it on real nodes (topology pressure), which is
     how a higher-priority gang reclaims capacity from running lower-priority gangs.
     Quota stays non-binding, so the pressure signal is TAS, not quota. Because Iris
     now routes *every* pod through Kueue (not just gangs), pass
     ``--cpu-flavor-node-label KEY=VALUE`` to also provision the ``cw-cpu``
     ResourceFlavor so CPU-only pods have a flavor to match. The namespaced
     LocalQueue is NOT created here: Iris reconciles its own (``{label_prefix}-lq``)
     at controller start (``K8sControllerProvider.ensure_kueue_queues``), binding it
     to this ClusterQueue via ``kubernetes_provider.kueue.cluster_queue``.

NB on Kueue version: TAS-aware preemption is version-sensitive. On too-old a Kueue
a ClusterQueue that combines a topology-bound flavor with a ``preemption`` stanza
can be marked Inactive, which breaks all gang admission. Validate on the target
version (kind smoke first) before applying to a shared cluster.

NB on the topology levels / flavor node-labels: to Kueue these are just node-label
*keys* and a node selector — nothing CoreWeave-specific. The ``upstream`` variant
reuses the identical CoreWeave level names and flavor labels; on a synthetic
cluster (kind) the caller must stamp those labels onto the nodes first (the smoke
harness does this), so TAS sees the kind nodes as one IB fabric.

SAFE BY DEFAULT: prints the rendered helm values + the would-be queue manifests,
then stops. Pass ``--apply`` to mutate the cluster. The coreweave variant touches
a SHARED cluster — review the plan before applying.

Requires: helm >= 3.8, kubectl. Point at the cluster with ``--kubeconfig`` and/or
``--context`` (or the usual ``KUBECONFIG`` env var).

Why this exists / what the CoreWeave docs leave out:
  https://docs.coreweave.com/products/cks/clusters/coreweave-charts/kueue
  documents the repo, the install, and ``topologies:`` but NOT how to enable the
  plain-Pod integration Iris relies on. That is the ``integrations.frameworks``
  block this script injects via the chart's ``managerConfig``.
"""

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence

import click
import yaml
from iris.cluster.platforms.k8s.coreweave_topology import (
    CW_FLAVOR_INFINIBAND,
    CW_LABEL_FABRIC,
    CW_LABEL_FLAVOR,
    CW_LABEL_LEAFGROUP,
    CW_LABEL_NVLINK_DOMAIN,
    CW_LABEL_SUPERPOD,
)
from iris.cluster.platforms.k8s.types import IRIS_PRIORITY_CLASS_SYSTEM, iris_priority_class_manifest

# Right after a fresh install Kueue's internal cert manager has not yet populated
# the webhook caBundle, so admission/conversion webhook calls fail transiently
# ("certificate signed by unknown authority"). Retry kubectl reads/writes through
# this window rather than fail the install.
_WEBHOOK_WARMUP_RETRIES = 6
_WEBHOOK_WARMUP_DELAY = 5.0

# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
VARIANT_COREWEAVE = "coreweave"
VARIANT_UPSTREAM = "upstream"

# CoreWeave cks-kueue chart (wraps upstream kueue as a subchart).
CW_REPO_NAME = "coreweave"
CW_REPO_URL = "https://charts.core-services.ingress.coreweave.com"
CW_CHART = f"{CW_REPO_NAME}/cks-kueue"

# Upstream Kueue OCI helm chart (kind / generic clusters).
UPSTREAM_CHART = "oci://registry.k8s.io/kueue/charts/kueue"
UPSTREAM_DEFAULT_VERSION = "0.11.0"

RELEASE_DEFAULT = "kueue"
OPERATOR_NS = "kueue-system"

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
    "cpu": "1000000000",  # cores
    "memory": "1Pi",
    "ephemeral-storage": "1Pi",
    "nvidia.com/gpu": "1000000000",
    "rdma/ib": "1000000000",
}
COVERED_RESOURCES = list(NON_BINDING_QUOTA)

# The CPU ResourceFlavor (cw-cpu) covers the SAME resources so it can share one
# resourceGroup with cw-ib, but pins the accelerator quotas to ZERO: a GPU pod can
# never match cw-cpu and falls through to the IB flavor, while a CPU-only pod (which
# requests no GPU/RDMA) matches cw-cpu and lands on CPU capacity. Listed first in the
# resourceGroup so CPU pods pick it before the GPU flavor.
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


def build_cks_values(pod_namespaces: Sequence[str] = DEFAULT_POD_NAMESPACES) -> dict:
    """Return the ``cks-kueue`` (CoreWeave) helm values (managerConfig only).

    cks-kueue nests the upstream kueue subchart under ``kueue:``. The chart's
    ``topologies:`` value is deliberately NOT set — it renders Topology CRs at an
    apiVersion the CRD no longer serves (see module docstring); the Topology CRs
    are kubectl-applied after install instead.

    NB: the chart already enables ``--feature-gates=TopologyAwareScheduling=true``
    by default (its ``controllerManager.featureGates`` value is a *list*), so we
    deliberately do NOT set ``featureGates`` — overriding it (especially as a map)
    breaks the chart's ``kueue.featureGates`` template.
    """
    config_yaml = yaml.safe_dump(
        build_controller_manager_config(pod_namespaces), default_flow_style=False, sort_keys=False
    )
    return {
        "kueue": {
            "enableKueueViz": False,
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


def build_cpu_resource_flavor(node_label_key: str, node_label_value: str) -> dict:
    """Return the cluster-scoped CPU ResourceFlavor (cw-cpu), no topology.

    Selects CPU nodes via a caller-supplied node label so CPU-only pods routed
    through Kueue land on CPU capacity, never the IB GPU flavor. The correct label
    is cluster-specific (which nodes are CPU nodes), so it is a required input, not
    a default. No ``topologyName``: CPU jobs need no topology-aware placement.
    """
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "ResourceFlavor",
        "metadata": {"name": CPU_RESOURCE_FLAVOR_NAME},
        "spec": {"nodeLabels": {node_label_key: node_label_value}},
    }


def build_cluster_queue(name: str, *, include_cpu_flavor: bool = False) -> dict:
    """Return the cluster-scoped, admin-owned ClusterQueue.

    Covers every resource Iris pods request (COVERED_RESOURCES) with a non-binding
    nominalQuota (NON_BINDING_QUOTA) — Kueue does not enforce capacity here (the Iris
    autoscaler does). It DOES enforce priority: ``preemption.withinClusterQueue:
    LowerPriority`` lets a higher-priority pending Workload evict lower-priority
    admitted ones when it cannot otherwise be admitted — including when TAS cannot
    place it (topology pressure), which is how a higher-priority gang reclaims nodes
    from running batch gangs even though quota never binds.

    With ``include_cpu_flavor`` the resourceGroup carries cw-cpu (first) as well as
    cw-ib, so CPU-only pods routed through Kueue match a CPU flavor instead of the IB
    GPU flavor. Both flavors cover the same resources; cw-cpu pins GPU/RDMA to 0.
    """
    flavors = [
        {
            "name": RESOURCE_FLAVOR_NAME,
            "resources": [{"name": r, "nominalQuota": NON_BINDING_QUOTA[r]} for r in COVERED_RESOURCES],
        }
    ]
    if include_cpu_flavor:
        flavors.insert(
            0,
            {
                "name": CPU_RESOURCE_FLAVOR_NAME,
                "resources": [{"name": r, "nominalQuota": CPU_FLAVOR_QUOTA[r]} for r in COVERED_RESOURCES],
            },
        )
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


def _parse_node_label(spec: str) -> tuple[str, str]:
    """Parse a ``KEY=VALUE`` node-label selector; raise on a malformed value."""
    key, sep, value = spec.partition("=")
    if not sep or not key or not value:
        raise click.BadParameter(f"expected KEY=VALUE, got {spec!r}", param_hint="--cpu-flavor-node-label")
    return key, value


# --------------------------------------------------------------------------
# Thin I/O helpers (subprocess via arg lists — never shell=True).
# --------------------------------------------------------------------------
def helm_flags(kubeconfig: str | None, context: str | None) -> list[str]:
    """Shared flags threaded into every helm invocation (helm spells it --kube-context)."""
    flags: list[str] = []
    if kubeconfig:
        flags += ["--kubeconfig", kubeconfig]
    if context:
        flags += ["--kube-context", context]
    return flags


def kubectl_flags(kubeconfig: str | None, context: str | None) -> list[str]:
    """Shared flags threaded into every kubectl invocation (kubectl spells it --context)."""
    flags: list[str] = []
    if kubeconfig:
        flags += ["--kubeconfig", kubeconfig]
    if context:
        flags += ["--context", context]
    return flags


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command from an arg list, echoing it first."""
    click.secho(f"$ {' '.join(cmd)}", fg="bright_black")
    return subprocess.run(cmd, **kwargs)


def write_values_file(values: dict) -> str:
    """Serialize a values dict to a temp YAML file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="kueue-values.")
    with os.fdopen(fd, "w") as handle:
        yaml.safe_dump(values, handle, default_flow_style=False, sort_keys=False)
    return path


def topology_api_version(kc_flags: list[str]) -> str:
    """Return the served apiVersion (group/version) of the Topology CRD on the cluster.

    TAS's Topology kind has lived at different versions across Kueue releases
    (v1alpha1 in 0.11), so we read the served version off the installed CRD rather
    than hardcode it.
    """
    result = run(
        [
            "kubectl",
            *kc_flags,
            "get",
            "crd",
            TOPOLOGY_CRD,
            "-o",
            "jsonpath={.spec.versions[?(@.served)].name}",
        ],
        capture_output=True,
        text=True,
    )
    version = (result.stdout or "").split()[0] if result.stdout.strip() else ""
    if not version:
        raise RuntimeError(f"could not determine served version of {TOPOLOGY_CRD} (is the Kueue CRD installed?)")
    return f"kueue.x-k8s.io/{version}"


def kubectl_apply_docs(docs: list[dict], kc_flags: list[str]) -> None:
    """Apply a list of manifest dicts via ``kubectl apply -f -``, retrying webhook warmup."""
    manifest = yaml.safe_dump_all(docs, default_flow_style=False, sort_keys=False)
    last = subprocess.CompletedProcess([], 1, "", "")
    for attempt in range(_WEBHOOK_WARMUP_RETRIES + 1):
        last = run(["kubectl", *kc_flags, "apply", "-f", "-"], input=manifest, text=True, capture_output=True)
        if last.returncode == 0:
            click.echo((last.stdout or "").rstrip())
            return
        if attempt < _WEBHOOK_WARMUP_RETRIES:
            click.secho(
                f"   apply failed (attempt {attempt + 1}); retrying in {_WEBHOOK_WARMUP_DELAY:g}s "
                "(Kueue webhook likely still warming up)",
                fg="yellow",
                err=True,
            )
            time.sleep(_WEBHOOK_WARMUP_DELAY)
    click.secho((last.stderr or "").strip(), fg="red", err=True)
    raise RuntimeError("kubectl apply failed after webhook-warmup retries")


def kubectl_get_topologies(kc_flags: list[str]) -> None:
    """Print the Topology CRs, tolerating transient conversion-webhook warmup errors."""
    last = subprocess.CompletedProcess([], 1, "", "")
    for attempt in range(_WEBHOOK_WARMUP_RETRIES + 1):
        last = run(["kubectl", *kc_flags, "get", TOPOLOGY_CRD], capture_output=True, text=True)
        if last.returncode == 0:
            click.echo((last.stdout or "").rstrip())
            return
        if attempt < _WEBHOOK_WARMUP_RETRIES:
            time.sleep(_WEBHOOK_WARMUP_DELAY)
    click.secho(f"warn: could not list Topologies: {(last.stderr or '').strip()}", fg="yellow", err=True)


# --------------------------------------------------------------------------
# Install core (importable; the click command and the smoke harness both call it).
# --------------------------------------------------------------------------
def run_install(
    *,
    variant: str,
    kubeconfig: str | None = None,
    context: str | None = None,
    chart_version: str | None = None,
    release: str = RELEASE_DEFAULT,
    with_queues: bool = False,
    cluster_queue: str = "iris-cq",
    flavor_topology: str = INFINIBAND_TOPOLOGY_NAME,
    cpu_flavor_node_label: tuple[str, str] | None = None,
    pod_namespaces: Sequence[str] = DEFAULT_POD_NAMESPACES,
    apply: bool = False,
) -> None:
    """Install + configure Kueue for the given ``variant`` (coreweave | upstream).

    Idempotent. Prints the plan and returns without mutating the cluster unless
    ``apply`` is set. ``flavor_topology`` selects the Topology the ResourceFlavor
    binds (default InfiniBand; the kind smoke passes multinode-nvlink-ib).
    ``cpu_flavor_node_label`` is a ``(key, value)`` node label selecting CPU nodes;
    when set, ``--with-queues`` also provisions the cw-cpu ResourceFlavor and adds it
    to the ClusterQueue so CPU-only pods routed through Kueue have a flavor to match.
    ``pod_namespaces`` scopes the plain-Pod admission webhook (default: the ``iris``
    namespace) — never widen this to system namespaces on a shared cluster.
    """
    if variant not in (VARIANT_COREWEAVE, VARIANT_UPSTREAM):
        raise ValueError(f"unknown variant {variant!r} (expected {VARIANT_COREWEAVE!r} or {VARIANT_UPSTREAM!r})")

    hflags = helm_flags(kubeconfig, context)
    kflags = kubectl_flags(kubeconfig, context)
    if with_queues:
        queue_docs = [build_resource_flavor(flavor_topology)]
        if cpu_flavor_node_label is not None:
            queue_docs.append(build_cpu_resource_flavor(*cpu_flavor_node_label))
        queue_docs.append(build_cluster_queue(cluster_queue, include_cpu_flavor=cpu_flavor_node_label is not None))
    else:
        queue_docs = []

    if variant == VARIANT_COREWEAVE:
        values = build_cks_values(pod_namespaces)
        chart = CW_CHART
        version = chart_version
    else:
        values = build_upstream_values(pod_namespaces)
        chart = UPSTREAM_CHART
        version = chart_version or UPSTREAM_DEFAULT_VERSION

    version_args = ["--version", version] if version else []

    # Always assemble + print the plan (chart, values, queue manifests). The only
    # branch is the final apply: print and stop unless --apply.
    click.secho(f"==> Variant: {variant} (chart={chart}, version={version or 'latest'})", fg="blue", bold=True)
    click.secho("==> Rendered helm values:", fg="blue", bold=True)
    click.echo(yaml.safe_dump(values, default_flow_style=False, sort_keys=False))
    if with_queues:
        click.secho(f"==> ResourceFlavor + ClusterQueue ({cluster_queue}):", fg="blue", bold=True)
        click.echo(yaml.safe_dump_all(queue_docs, default_flow_style=False, sort_keys=False))

    if not apply:
        click.secho("\nwarn: dry run — nothing applied. Re-run with --apply to install.", fg="yellow", err=True)
        return

    if variant == VARIANT_COREWEAVE:
        # helm repo add/update only touches local helm config (no cluster mutation).
        click.secho(f"==> Adding/updating helm repo {CW_REPO_NAME} ({CW_REPO_URL})", fg="blue", bold=True)
        run(["helm", "repo", "add", CW_REPO_NAME, CW_REPO_URL], check=True, stdout=subprocess.DEVNULL)
        run(["helm", "repo", "update", CW_REPO_NAME], check=True, stdout=subprocess.DEVNULL)

    _apply(values, chart, release, hflags, kflags, version_args)

    if with_queues:
        click.secho(f"==> Applying ResourceFlavor + ClusterQueue ({cluster_queue})", fg="blue", bold=True)
        kubectl_apply_docs(queue_docs, kflags)

    click.secho(
        "==> Done. Point the Iris cluster config at this admin ClusterQueue. Iris creates its own "
        "LocalQueue ({label_prefix}-lq) in its namespace at controller start:",
        fg="green",
        bold=True,
    )
    click.echo("  kubernetes_provider:\n    kueue:\n" f"      cluster_queue: {cluster_queue}")


def _helm_upgrade(chart: str, release: str, values_file: str, hflags: list[str], version_args: list[str]) -> None:
    run(
        [
            "helm",
            "upgrade",
            "--install",
            release,
            chart,
            "--namespace",
            OPERATOR_NS,
            "--create-namespace",
            "--values",
            values_file,
            *version_args,
            *hflags,
        ],
        check=True,
    )


def _pin_manager_priority(kflags: list[str]) -> None:
    """Pin kueue-controller-manager to the iris-system PriorityClass.

    The manager serves Kueue's admission webhook — a hard dependency of every pod
    CREATE in the Iris namespace. The helm charts leave it at priority 0, below
    every Iris user job (iris-interactive=10), so a user pod can legally preempt
    it; when it dies the webhook loses its endpoint and all pod admission fails
    clusterwide until it reschedules. Pinning it to iris-system (10000, above
    iris-production) makes it non-preemptible.

    Applied out of band because neither chart variant exposes a priorityClassName
    value. Helm 3's 3-way merge preserves fields it never set, so this survives
    later `helm upgrade`s; install_kueue also re-applies it on every run.
    """
    click.secho("==> Pinning kueue-controller-manager to the iris-system PriorityClass", fg="blue", bold=True)
    kubectl_apply_docs([iris_priority_class_manifest(IRIS_PRIORITY_CLASS_SYSTEM)], kflags)
    patch = json.dumps({"spec": {"template": {"spec": {"priorityClassName": IRIS_PRIORITY_CLASS_SYSTEM}}}})
    run(
        [
            "kubectl",
            *kflags,
            "-n",
            OPERATOR_NS,
            "patch",
            "deploy/kueue-controller-manager",
            "--type=strategic",
            "-p",
            patch,
        ],
        check=True,
    )


def _wait_controller(kflags: list[str]) -> None:
    click.secho("==> Waiting for the Kueue controller to become available", fg="blue", bold=True)
    run(
        [
            "kubectl",
            *kflags,
            "-n",
            OPERATOR_NS,
            "rollout",
            "status",
            "deploy/kueue-controller-manager",
            "--timeout=180s",
        ],
        check=True,
    )


def _apply(
    values: dict, chart: str, release: str, hflags: list[str], kflags: list[str], version_args: list[str]
) -> None:
    """Install/upgrade Kueue, then ensure the Topology CRs exist.

    One helm pass installs the operator + CRDs for both charts (cks-kueue templates
    its CRDs, the upstream chart ships them in ``crds/`` — either way the CRDs land
    before any Topology CR is needed). The Topology CRs are then kubectl-applied at
    the apiVersion the installed CRD actually serves; see the module docstring for
    why the cks chart cannot template them itself. Idempotent on re-runs.
    """
    click.secho(f"==> Installing/upgrading {chart} as '{release}' in {OPERATOR_NS}", fg="blue", bold=True)
    _helm_upgrade(chart, release, write_values_file(values), hflags, version_args)
    click.secho(f"==> Waiting for {TOPOLOGY_CRD} to be Established", fg="blue", bold=True)
    run(
        ["kubectl", *kflags, "wait", "--for=condition=Established", f"crd/{TOPOLOGY_CRD}", "--timeout=120s"],
        check=True,
    )
    _pin_manager_priority(kflags)
    _wait_controller(kflags)

    api_version = topology_api_version(kflags)
    click.secho(f"==> Applying Topology CRs ({api_version})", fg="blue", bold=True)
    topology_docs = [build_topology_cr(name, levels, api_version) for name, levels in TOPOLOGIES.items()]
    kubectl_apply_docs(topology_docs, kflags)

    click.secho("==> Topologies on the cluster:", fg="blue", bold=True)
    kubectl_get_topologies(kflags)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
@click.command()
@click.option(
    "--variant",
    type=click.Choice([VARIANT_COREWEAVE, VARIANT_UPSTREAM]),
    default=VARIANT_COREWEAVE,
    help="Which chart to install: 'coreweave' (cks-kueue) or 'upstream' (kind/generic).",
)
@click.option("--kubeconfig", default=None, help="kubeconfig to use (else $KUBECONFIG / ~/.kube/config).")
@click.option("--context", default=None, help="kube context to target.")
@click.option("--chart-version", default=None, help="Pin the chart version (upstream default: 0.11.0; cw: latest).")
@click.option("--release", default=RELEASE_DEFAULT, help="helm release name (default: kueue).")
@click.option(
    "--with-queues/--no-with-queues",
    default=False,
    help="Also create the cluster-scoped ResourceFlavor + ClusterQueue.",
)
@click.option("--cluster-queue", default="iris-cq", help="ClusterQueue name for --with-queues (default: iris-cq).")
@click.option(
    "--flavor-topology",
    type=click.Choice([INFINIBAND_TOPOLOGY_NAME, MULTINODE_TOPOLOGY_NAME]),
    default=INFINIBAND_TOPOLOGY_NAME,
    help="Topology the cw-ib ResourceFlavor binds (default: infiniband; multinode-nvlink-ib exposes nvlink.domain).",
)
@click.option(
    "--cpu-flavor-node-label",
    default=None,
    metavar="KEY=VALUE",
    help="Node label selecting CPU nodes. When set, --with-queues also provisions the cw-cpu "
    "ResourceFlavor so CPU-only pods routed through Kueue land on CPU capacity (never the IB GPU flavor).",
)
@click.option(
    "--pod-namespace",
    "pod_namespaces",
    multiple=True,
    default=DEFAULT_POD_NAMESPACES,
    show_default=True,
    help="Namespace(s) the plain-Pod webhook is scoped to (where Iris submits gang pods). Repeatable.",
)
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
def main(
    variant: str,
    kubeconfig: str | None,
    context: str | None,
    chart_version: str | None,
    release: str,
    with_queues: bool,
    cluster_queue: str,
    flavor_topology: str,
    cpu_flavor_node_label: str | None,
    pod_namespaces: tuple[str, ...],
    apply: bool,
) -> None:
    """Install + configure Kueue (coreweave or upstream) for Iris gang admission."""
    cpu_label = _parse_node_label(cpu_flavor_node_label) if cpu_flavor_node_label else None
    run_install(
        variant=variant,
        kubeconfig=kubeconfig,
        context=context,
        chart_version=chart_version,
        release=release,
        with_queues=with_queues,
        cluster_queue=cluster_queue,
        flavor_topology=flavor_topology,
        cpu_flavor_node_label=cpu_label,
        pod_namespaces=pod_namespaces,
        apply=apply,
    )


if __name__ == "__main__":
    main()
