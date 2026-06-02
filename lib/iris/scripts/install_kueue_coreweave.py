#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install + configure Kueue on a CoreWeave CKS cluster for Iris gang admission.

What it does (idempotent; ``helm upgrade --install``):
  1. Adds the CoreWeave helm repo and installs the ``cks-kueue`` chart into the
     ``kueue-system`` namespace.
  2. Creates the CoreWeave Topology CRs (infiniband + multinode-nvlink-ib) so
     topology-aware scheduling can honor the podset-topology annotations Iris
     stamps (``backend.coreweave.cloud/leafgroup``, ``ds.coreweave.com/nvlink.domain``).
  3. Enables the plain-Pod integration (NOT on by default in the chart) so
     Kueue gang-admits Iris's scheduling-gated pod groups. This is a broad,
     whole-cluster install: the pod webhook uses the chart default selector
     (all namespaces except kube-system/kueue-system) and Kueue only gates pods
     carrying the kueue.x-k8s.io/queue-name label (manageJobsWithoutQueueName is
     false), so unlabeled pods anywhere pass through untouched.
  4. (optional, ``--with-queues``) Creates the cluster-scoped, admin-owned
     ResourceFlavor + ClusterQueue (the quota). The namespaced LocalQueue is NOT
     created here: Iris reconciles its own LocalQueue at controller start
     (``K8sControllerProvider.ensure_kueue_queues``), binding it to this
     ClusterQueue via ``kubernetes_provider.kueue.cluster_queue``.

SAFE BY DEFAULT: prints the rendered helm values, a client-side validation, and
the would-be queue manifests, then stops. Pass ``--apply`` to mutate the
cluster. This touches a SHARED CoreWeave cluster — review the plan and the
namespace allowlist before applying.

Requires: helm >= 3.8, kubectl. Point at the cluster with ``--kubeconfig``
and/or ``--context`` (or the usual ``KUBECONFIG`` env var).

Why this exists / what the CoreWeave docs leave out:
  https://docs.coreweave.com/products/cks/clusters/coreweave-charts/kueue
  documents the repo, the install, and ``topologies:`` but NOT how to enable the
  plain-Pod integration that Iris's direct provider relies on. That is the
  ``integrations.frameworks: ["pod"]`` block this script injects via the wrapped
  upstream kueue subchart's ``managerConfig``.
"""

import os
import subprocess
import sys
import tempfile

import click
import yaml

# --------------------------------------------------------------------------
# Defaults / config
# --------------------------------------------------------------------------
REPO_NAME = "coreweave"
REPO_URL = "https://charts.core-services.ingress.coreweave.com"
CHART = f"{REPO_NAME}/cks-kueue"
RELEASE_DEFAULT = "kueue"
OPERATOR_NS = "kueue-system"

# The CoreWeave Topology CRs. Iris's preferred "pool" topology rides on
# backend.coreweave.cloud/leafgroup (NOT ib.coreweave.cloud/leafgroup) and the
# required "nvlink"/"tpu-name" topology on ds.coreweave.com/nvlink.domain — both
# are levels here, so TAS can satisfy the podset-topology annotations.
INFINIBAND_LEVELS = [
    "backend.coreweave.cloud/fabric",
    "backend.coreweave.cloud/superpod",
    "backend.coreweave.cloud/leafgroup",
    "kubernetes.io/hostname",
]
MULTINODE_NVLINK_IB_LEVELS = [
    "backend.coreweave.cloud/fabric",
    "backend.coreweave.cloud/superpod",
    "backend.coreweave.cloud/leafgroup",
    "ds.coreweave.com/nvlink.domain",
    "kubernetes.io/hostname",
]

# The IB Topology CR that the ResourceFlavor ties to.
TOPOLOGY_CRD = "topologies.kueue.x-k8s.io"
RESOURCE_FLAVOR_NAME = "cw-ib"
INFINIBAND_TOPOLOGY_NAME = "infiniband"
# Node selector for the cw-ib ResourceFlavor. Kueue requires a topology-aware
# flavor (spec.topologyName set) to also carry at least one nodeLabel; CoreWeave
# stamps backend.coreweave.cloud/flavor=infiniband on every IB-fabric node, which
# is exactly the capacity this flavor represents.
RESOURCE_FLAVOR_NODE_LABELS = {"backend.coreweave.cloud/flavor": "infiniband"}

# Resources the ClusterQueue covers when --with-queues is set.
COVERED_RESOURCES = ["cpu", "memory", "nvidia.com/gpu"]


# --------------------------------------------------------------------------
# Pure value/manifest builders (return plain dicts; no I/O).
# --------------------------------------------------------------------------
def build_controller_manager_config() -> dict:
    """Return the upstream kueue ``Configuration`` as a nested dict.

    This is serialized to YAML and embedded as the chart's
    ``controllerManagerConfigYaml`` string value. We enable the "pod" framework
    (gang admission for plain pods) alongside "batch/job" cluster-wide — this is
    a broad, whole-cluster Kueue install. ``manageJobsWithoutQueueName`` stays
    false so Kueue only gates pods that carry the ``kueue.x-k8s.io/queue-name``
    label (the ones Iris stamps); every other pod passes through untouched, so
    we do not need a podOptions.namespaceSelector. The pod webhook keeps the
    chart's default namespaceSelector (NotIn kube-system/kueue-system).
    """
    return {
        "apiVersion": "config.kueue.x-k8s.io/v1beta1",
        "kind": "Configuration",
        "health": {"healthProbeBindAddress": ":8081"},
        "metrics": {"bindAddress": ":8080"},
        "webhook": {"port": 9443},
        "manageJobsWithoutQueueName": False,
        "internalCertManagement": {
            "enable": True,
            "webhookServiceName": "kueue-webhook-service",
            "webhookSecretName": "kueue-webhook-server-cert",
        },
        "integrations": {
            "frameworks": ["batch/job", "pod"],
        },
    }


def build_base_values() -> dict:
    """Return the ``kueue:`` helm values block (managerConfig + viz).

    NB: the chart already enables ``--feature-gates=TopologyAwareScheduling=true``
    by default (its ``controllerManager.featureGates`` value is a *list*, not the
    map you might expect), so we deliberately do NOT set ``featureGates`` here.
    Overriding it — especially as a map — breaks the chart's ``kueue.featureGates``
    template (``can't evaluate field name in type interface {}``).
    """
    config_yaml = yaml.safe_dump(
        build_controller_manager_config(),
        default_flow_style=False,
        sort_keys=False,
    )
    return {
        "kueue": {
            "enableKueueViz": False,
            "managerConfig": {"controllerManagerConfigYaml": config_yaml},
        }
    }


def build_topology_values() -> dict:
    """Return the ``topologies:`` helm values block (the two Topology CRs)."""
    return {
        "topologies": [
            {"name": INFINIBAND_TOPOLOGY_NAME, "levels": INFINIBAND_LEVELS},
            {"name": "multinode-nvlink-ib", "levels": MULTINODE_NVLINK_IB_LEVELS},
        ]
    }


def build_resource_flavor() -> dict:
    """Return the cluster-scoped ResourceFlavor tied to the IB Topology."""
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "ResourceFlavor",
        "metadata": {"name": RESOURCE_FLAVOR_NAME},
        "spec": {
            # nodeLabels select the nodes this flavor represents (the IB fabric).
            # Required by Kueue whenever topologyName is set.
            "nodeLabels": RESOURCE_FLAVOR_NODE_LABELS,
            # Tie the flavor to the IB Topology so podset-topology annotations resolve.
            "topologyName": INFINIBAND_TOPOLOGY_NAME,
        },
    }


def build_cluster_queue(name: str, cpu: str, memory: str, gpu: str) -> dict:
    """Return the cluster-scoped, admin-owned ClusterQueue (the quota)."""
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta1",
        "kind": "ClusterQueue",
        "metadata": {"name": name},
        "spec": {
            "namespaceSelector": {},
            "resourceGroups": [
                {
                    "coveredResources": COVERED_RESOURCES,
                    "flavors": [
                        {
                            "name": RESOURCE_FLAVOR_NAME,
                            "resources": [
                                {"name": "cpu", "nominalQuota": cpu},
                                {"name": "memory", "nominalQuota": memory},
                                {"name": "nvidia.com/gpu", "nominalQuota": gpu},
                            ],
                        }
                    ],
                }
            ],
        },
    }


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


def crd_exists(crd: str, kc_flags: list[str]) -> bool:
    """Return True if the named CRD is present on the cluster."""
    result = run(
        ["kubectl", *kc_flags, "get", "crd", crd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
@click.command()
@click.option("--kubeconfig", default=None, help="kubeconfig to use (else $KUBECONFIG / ~/.kube/config).")
@click.option("--context", default=None, help="kube context to target.")
@click.option("--chart-version", default=None, help="Pin the cks-kueue chart version (default: latest).")
@click.option("--release", default=RELEASE_DEFAULT, help="helm release name (default: kueue).")
@click.option(
    "--with-queues/--no-with-queues",
    default=False,
    help="Also create the cluster-scoped ResourceFlavor + ClusterQueue.",
)
@click.option("--cluster-queue", default="iris-cq", help="ClusterQueue name for --with-queues (default: iris-cq).")
@click.option("--cq-cpu", default="384", help="ClusterQueue cpu nominal quota (default: 384).")
@click.option("--cq-memory", default="1536Gi", help="ClusterQueue memory nominal quota (default: 1536Gi).")
@click.option("--cq-gpu", default="24", help="ClusterQueue nvidia.com/gpu quota (default: 24).")
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
def main(
    kubeconfig: str | None,
    context: str | None,
    chart_version: str | None,
    release: str,
    with_queues: bool,
    cluster_queue: str,
    cq_cpu: str,
    cq_memory: str,
    cq_gpu: str,
    apply: bool,
) -> None:
    """Install + configure the CoreWeave cks-kueue chart for Iris gang admission."""
    hflags = helm_flags(kubeconfig, context)
    kflags = kubectl_flags(kubeconfig, context)

    base_values = build_base_values()
    topo_values = build_topology_values()

    click.secho("==> Rendered helm base values (kueue: block):", fg="blue", bold=True)
    click.echo(yaml.safe_dump(base_values, default_flow_style=False, sort_keys=False))
    click.secho("==> Rendered helm topology values (topologies: block):", fg="blue", bold=True)
    click.echo(yaml.safe_dump(topo_values, default_flow_style=False, sort_keys=False))

    queue_docs = (
        [build_resource_flavor(), build_cluster_queue(cluster_queue, cq_cpu, cq_memory, cq_gpu)] if with_queues else []
    )

    # helm repo add/update only touches local helm config (no cluster mutation).
    click.secho(f"==> Adding/updating helm repo {REPO_NAME} ({REPO_URL})", fg="blue", bold=True)
    run(["helm", "repo", "add", REPO_NAME, REPO_URL], check=True, stdout=subprocess.DEVNULL)
    run(["helm", "repo", "update", REPO_NAME], check=True, stdout=subprocess.DEVNULL)

    version_args = ["--version", chart_version] if chart_version else []

    if not apply:
        _dry_run(base_values, topo_values, queue_docs, release, hflags, version_args, with_queues)
        return

    _apply(
        base_values,
        topo_values,
        queue_docs,
        release,
        hflags,
        kflags,
        version_args,
        with_queues,
        cluster_queue,
    )


def _dry_run(
    base_values: dict,
    topo_values: dict,
    queue_docs: list[dict],
    release: str,
    hflags: list[str],
    version_args: list[str],
    with_queues: bool,
) -> None:
    """Client-side validation only — never mutates the cluster.

    NB: a server-side ``helm upgrade --dry-run=server`` would spuriously ERROR on
    a FIRST install of this chart. The chart TEMPLATES its CRDs (they are not in
    the chart's crds/ dir), so helm maps every manifest against the live API
    server's discovery before applying — and the Topology CRD does not yet
    exist, so the Topology CRs fail to map. We therefore validate client-side via
    ``helm template`` (no cluster mutation) instead.
    """
    base_file = write_values_file(base_values)
    topo_file = write_values_file(topo_values)

    click.secho(
        "==> DRY RUN — client-side validating the chart via `helm template` (no changes)",
        fg="blue",
        bold=True,
    )
    result = run(
        [
            "helm",
            "template",
            release,
            CHART,
            "--namespace",
            OPERATOR_NS,
            "--values",
            base_file,
            "--values",
            topo_file,
            *version_args,
            *hflags,
        ],
        stdout=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        click.secho("error: `helm template` failed to render the chart with these values.", fg="red", err=True)
        sys.exit(result.returncode)
    click.secho("    helm template rendered cleanly.", fg="green")

    click.secho(
        "\nwarn: This was a dry run (client-side `helm template` only; a server-side "
        "--dry-run cannot validate a first install of this chart). Re-run with --apply "
        "to install on the cluster.",
        fg="yellow",
        err=True,
    )
    if with_queues:
        click.secho(
            "warn: --with-queues objects (ResourceFlavor/ClusterQueue) are printed below but NOT applied.",
            fg="yellow",
            err=True,
        )
        click.echo(yaml.safe_dump_all(queue_docs, default_flow_style=False, sort_keys=False))


def _apply(
    base_values: dict,
    topo_values: dict,
    queue_docs: list[dict],
    release: str,
    hflags: list[str],
    kflags: list[str],
    version_args: list[str],
    with_queues: bool,
    cluster_queue: str,
) -> None:
    """Two-phase helm install that mutates the cluster."""
    base_file = write_values_file(base_values)
    topo_file = write_values_file(topo_values)

    # --------------------------------------------------------------------
    # Two-phase install. The cks-kueue chart TEMPLATES its CRDs (helm show crds
    # is empty), so helm maps EVERY manifest against the live API server's
    # discovery BEFORE applying anything. On a fresh cluster you therefore cannot
    # create the Topology CRD and a Topology CR in the same `helm install` — it
    # fails with "resource mapping not found for kind Topology ... ensure CRDs
    # are installed first". So: if the Topology CRD is absent, do a BOOTSTRAP
    # pass with base_values only (no topologies => no CRs => CRDs get created
    # cleanly), wait for the CRD to be Established, then do the FULL pass with
    # topologies. We only bootstrap when the CRD is missing so re-runs on an
    # already-installed cluster are a single idempotent pass (avoids briefly
    # removing the Topology CRs).
    # --------------------------------------------------------------------
    if not crd_exists(TOPOLOGY_CRD, kflags):
        click.secho(
            f"==> Topology CRD absent — BOOTSTRAP pass to create CRDs (release '{release}', no topologies)",
            fg="blue",
            bold=True,
        )
        run(
            [
                "helm",
                "upgrade",
                "--install",
                release,
                CHART,
                "--namespace",
                OPERATOR_NS,
                "--create-namespace",
                "--values",
                base_file,
                *version_args,
                *hflags,
            ],
            check=True,
        )
        click.secho(f"==> Waiting for {TOPOLOGY_CRD} to be Established", fg="blue", bold=True)
        run(
            [
                "kubectl",
                *kflags,
                "wait",
                "--for=condition=Established",
                f"crd/{TOPOLOGY_CRD}",
                "--timeout=120s",
            ],
            check=True,
        )

    click.secho(
        f"==> FULL pass: installing/upgrading {CHART} as release '{release}' in {OPERATOR_NS}",
        fg="blue",
        bold=True,
    )
    run(
        [
            "helm",
            "upgrade",
            "--install",
            release,
            CHART,
            "--namespace",
            OPERATOR_NS,
            "--create-namespace",
            "--values",
            base_file,
            "--values",
            topo_file,
            *version_args,
            *hflags,
        ],
        check=True,
    )

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

    click.secho("==> Topologies on the cluster:", fg="blue", bold=True)
    result = run(["kubectl", *kflags, "get", "topologies.kueue.x-k8s.io"])
    if result.returncode != 0:
        click.secho("warn: no Topology CRs found yet (the chart may still be reconciling)", fg="yellow", err=True)

    if with_queues:
        click.secho(f"==> Applying ResourceFlavor + ClusterQueue ({cluster_queue})", fg="blue", bold=True)
        manifest = yaml.safe_dump_all(queue_docs, default_flow_style=False, sort_keys=False)
        run(["kubectl", *kflags, "apply", "-f", "-"], input=manifest, text=True, check=True)

    click.secho(
        "==> Done. Point the Iris cluster config at this admin ClusterQueue. Iris creates its own "
        "LocalQueue ({label_prefix}-lq) in its namespace at controller start:",
        fg="green",
        bold=True,
    )
    click.echo("  kubernetes_provider:\n    kueue:\n" f"      cluster_queue: {cluster_queue}")


if __name__ == "__main__":
    main()
