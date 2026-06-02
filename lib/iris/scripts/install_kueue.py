#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install + configure Kueue for Iris gang admission, on CoreWeave or upstream.

Iris's K8s direct provider relies on Kueue's *plain-Pod* integration to gang-admit
coscheduled pod groups (all-or-nothing) and on Topology-Aware Scheduling (TAS) to
honor the podset-topology annotations it stamps. Neither is on by default in any
Kueue chart, so this script renders that configuration and installs it.

Two variants share one code path (``--variant``):

  * ``coreweave`` — the CoreWeave ``cks-kueue`` helm chart (wraps upstream kueue),
    which TEMPLATES its CRDs and renders Topology CRs from ``topologies:`` values.
    Because the CRDs are templated (not in the chart's ``crds/`` dir), a first
    install needs a two-phase bootstrap (CRDs first, then the Topology CRs).
  * ``upstream`` — the upstream OCI helm chart
    (``oci://registry.k8s.io/kueue/charts/kueue``), used for kind / generic
    clusters. CRDs ship in the chart, TAS is enabled via ``controllerManager``
    feature gates, and the Topology CRs are applied with kubectl after install.
    The smoke harness (scripts/gpu_gang_smoke.py) drives this variant on kind.

Both variants:
  1. Install the operator into ``kueue-system`` (``helm upgrade --install``).
  2. Enable the plain-Pod integration via the controller-manager ``Configuration``
     (``integrations.frameworks: ["batch/job","pod"]``). ``manageJobsWithoutQueueName``
     stays false, so Kueue only gates pods carrying ``kueue.x-k8s.io/queue-name``
     (the ones Iris stamps); every other pod passes through untouched — this is a
     broad, whole-cluster install with the chart-default webhook namespace
     selector (all namespaces except kube-system/kueue-system).
  3. Create the Topology CRs (infiniband + multinode-nvlink-ib) so TAS can resolve
     the podset-topology annotations (``backend.coreweave.cloud/leafgroup``,
     ``ds.coreweave.com/nvlink.domain``).
  4. (``--with-queues``) Create the cluster-scoped, admin-owned ResourceFlavor
     (``cw-ib``, selecting ``backend.coreweave.cloud/flavor=infiniband`` nodes) +
     ClusterQueue (the quota). The namespaced LocalQueue is NOT created here: Iris
     reconciles its own (``{label_prefix}-lq``) at controller start
     (``K8sControllerProvider.ensure_kueue_queues``), binding it to this
     ClusterQueue via ``kubernetes_provider.kueue.cluster_queue``.

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

import os
import subprocess
import sys
import tempfile
import time

import click
import yaml

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

# Topology CRs. Iris's preferred "pool" topology rides on
# backend.coreweave.cloud/leafgroup and the required "nvlink"/"tpu-name" topology
# on ds.coreweave.com/nvlink.domain — both are levels here, so TAS can satisfy the
# podset-topology annotations Iris stamps.
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
RESOURCE_FLAVOR_NODE_LABELS = {"backend.coreweave.cloud/flavor": "infiniband"}

# Resources the ClusterQueue covers when --with-queues is set.
COVERED_RESOURCES = ["cpu", "memory", "nvidia.com/gpu"]


# --------------------------------------------------------------------------
# Pure builders (return plain dicts; no I/O).
# --------------------------------------------------------------------------
def build_controller_manager_config() -> dict:
    """Return the kueue ``Configuration`` (controller-manager config) as a dict.

    Serialized to YAML and embedded as the chart's ``controllerManagerConfigYaml``
    value. Enables the "pod" framework (gang admission for plain pods) alongside
    "batch/job" cluster-wide. ``manageJobsWithoutQueueName`` stays false so Kueue
    only gates pods carrying ``kueue.x-k8s.io/queue-name`` (the ones Iris stamps);
    every other pod passes through, so no podOptions.namespaceSelector is needed.
    internalCertManagement is enabled so Kueue self-signs its webhook certs (no
    cert-manager dependency); the names match both charts' webhook service/secret.
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


def build_cks_values() -> dict:
    """Return the ``cks-kueue`` (CoreWeave) helm values: managerConfig + topologies.

    cks-kueue nests the upstream kueue subchart under ``kueue:`` and renders
    Topology CRs from a top-level ``topologies:`` list.

    NB: the chart already enables ``--feature-gates=TopologyAwareScheduling=true``
    by default (its ``controllerManager.featureGates`` value is a *list*), so we
    deliberately do NOT set ``featureGates`` — overriding it (especially as a map)
    breaks the chart's ``kueue.featureGates`` template.
    """
    config_yaml = yaml.safe_dump(build_controller_manager_config(), default_flow_style=False, sort_keys=False)
    return {
        "kueue": {
            "enableKueueViz": False,
            "managerConfig": {"controllerManagerConfigYaml": config_yaml},
        },
        "topologies": [{"name": name, "levels": levels} for name, levels in TOPOLOGIES.items()],
    }


def build_upstream_values() -> dict:
    """Return the upstream Kueue OCI-chart helm values.

    The upstream chart puts ``managerConfig`` at the top level and takes feature
    gates as a *list* under ``controllerManager.featureGates``. TopologyAwareScheduling
    is NOT on by default upstream, so we enable it here. The chart ships CRDs in
    ``crds/`` (installed by helm before templates), so no bootstrap pass is needed;
    the Topology CRs are applied with kubectl after the operator is up.
    """
    config_yaml = yaml.safe_dump(build_controller_manager_config(), default_flow_style=False, sort_keys=False)
    return {
        "enableKueueViz": False,
        "controllerManager": {
            "featureGates": [{"name": "TopologyAwareScheduling", "enabled": True}],
        },
        "managerConfig": {"controllerManagerConfigYaml": config_yaml},
    }


def build_topology_cr(name: str, levels: list[str], api_version: str) -> dict:
    """Return a Topology CR dict (for the upstream variant's kubectl-applied CRs)."""
    return {
        "apiVersion": api_version,
        "kind": "Topology",
        "metadata": {"name": name},
        "spec": {"levels": [{"nodeLabel": label} for label in levels]},
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
    cq_cpu: str = "384",
    cq_memory: str = "1536Gi",
    cq_gpu: str = "24",
    apply: bool = False,
) -> None:
    """Install + configure Kueue for the given ``variant`` (coreweave | upstream).

    Idempotent. Prints the plan and returns without mutating the cluster unless
    ``apply`` is set.
    """
    if variant not in (VARIANT_COREWEAVE, VARIANT_UPSTREAM):
        raise ValueError(f"unknown variant {variant!r} (expected {VARIANT_COREWEAVE!r} or {VARIANT_UPSTREAM!r})")

    hflags = helm_flags(kubeconfig, context)
    kflags = kubectl_flags(kubeconfig, context)
    queue_docs = (
        [build_resource_flavor(), build_cluster_queue(cluster_queue, cq_cpu, cq_memory, cq_gpu)] if with_queues else []
    )

    if variant == VARIANT_COREWEAVE:
        values = build_cks_values()
        chart = CW_CHART
        version = chart_version
    else:
        values = build_upstream_values()
        chart = UPSTREAM_CHART
        version = chart_version or UPSTREAM_DEFAULT_VERSION

    click.secho(f"==> Variant: {variant} (chart={chart}, version={version or 'latest'})", fg="blue", bold=True)
    click.secho("==> Rendered helm values:", fg="blue", bold=True)
    click.echo(yaml.safe_dump(values, default_flow_style=False, sort_keys=False))

    version_args = ["--version", version] if version else []

    if variant == VARIANT_COREWEAVE:
        # helm repo add/update only touches local helm config (no cluster mutation).
        click.secho(f"==> Adding/updating helm repo {CW_REPO_NAME} ({CW_REPO_URL})", fg="blue", bold=True)
        run(["helm", "repo", "add", CW_REPO_NAME, CW_REPO_URL], check=True, stdout=subprocess.DEVNULL)
        run(["helm", "repo", "update", CW_REPO_NAME], check=True, stdout=subprocess.DEVNULL)

    if not apply:
        _dry_run(values, chart, release, hflags, version_args, queue_docs, with_queues)
        return

    if variant == VARIANT_COREWEAVE:
        _apply_coreweave(values, chart, release, hflags, kflags, version_args)
    else:
        _apply_upstream(values, chart, release, hflags, kflags, version_args)

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


def _dry_run(
    values: dict,
    chart: str,
    release: str,
    hflags: list[str],
    version_args: list[str],
    queue_docs: list[dict],
    with_queues: bool,
) -> None:
    """Client-side validation only — never mutates the cluster.

    NB: a server-side ``helm upgrade --dry-run=server`` would spuriously ERROR on
    a FIRST install of the cks-kueue chart (it templates its CRDs, so helm maps
    every manifest against live discovery before applying, and the Topology CRD
    does not yet exist). We therefore validate client-side via ``helm template``.
    """
    values_file = write_values_file(values)
    click.secho("==> DRY RUN — client-side validating via `helm template` (no changes)", fg="blue", bold=True)
    result = run(
        [
            "helm",
            "template",
            release,
            chart,
            "--namespace",
            OPERATOR_NS,
            "--values",
            values_file,
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
        "\nwarn: This was a dry run (client-side `helm template` only). Re-run with --apply to install.",
        fg="yellow",
        err=True,
    )
    if with_queues:
        click.secho("warn: --with-queues objects printed below but NOT applied.", fg="yellow", err=True)
        click.echo(yaml.safe_dump_all(queue_docs, default_flow_style=False, sort_keys=False))


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


def _apply_coreweave(
    values: dict, chart: str, release: str, hflags: list[str], kflags: list[str], version_args: list[str]
) -> None:
    """Two-phase helm install for the cks-kueue chart (it templates its CRDs).

    On a fresh cluster you cannot create the Topology CRD and a Topology CR in the
    same `helm install` (helm maps every manifest against live discovery first, and
    the CRD does not exist yet). So if the Topology CRD is absent, do a BOOTSTRAP
    pass with NO topologies (CRDs get created cleanly), wait for the CRD to be
    Established, then the FULL pass with topologies. Re-runs on an already-installed
    cluster are a single idempotent pass.
    """
    full_file = write_values_file(values)
    if not crd_exists(TOPOLOGY_CRD, kflags):
        bootstrap = {k: v for k, v in values.items() if k != "topologies"}
        bootstrap_file = write_values_file(bootstrap)
        click.secho(
            f"==> Topology CRD absent — BOOTSTRAP pass to create CRDs (release '{release}', no topologies)",
            fg="blue",
            bold=True,
        )
        _helm_upgrade(chart, release, bootstrap_file, hflags, version_args)
        click.secho(f"==> Waiting for {TOPOLOGY_CRD} to be Established", fg="blue", bold=True)
        run(
            ["kubectl", *kflags, "wait", "--for=condition=Established", f"crd/{TOPOLOGY_CRD}", "--timeout=120s"],
            check=True,
        )

    click.secho(f"==> FULL pass: installing/upgrading {chart} as '{release}' in {OPERATOR_NS}", fg="blue", bold=True)
    _helm_upgrade(chart, release, full_file, hflags, version_args)
    _wait_controller(kflags)

    click.secho("==> Topologies on the cluster:", fg="blue", bold=True)
    kubectl_get_topologies(kflags)


def _apply_upstream(
    values: dict, chart: str, release: str, hflags: list[str], kflags: list[str], version_args: list[str]
) -> None:
    """Single-pass helm install for the upstream OCI chart, then apply Topology CRs.

    The upstream chart ships its CRDs in ``crds/`` (helm installs them before
    templates), so no bootstrap is needed. After the operator is up we apply the
    Topology CRs with kubectl, reading the served apiVersion off the installed CRD.
    """
    values_file = write_values_file(values)
    click.secho(f"==> Installing/upgrading {chart} as '{release}' in {OPERATOR_NS}", fg="blue", bold=True)
    _helm_upgrade(chart, release, values_file, hflags, version_args)
    _wait_controller(kflags)

    api_version = topology_api_version(kflags)
    click.secho(f"==> Applying Topology CRs ({api_version})", fg="blue", bold=True)
    topology_docs = [build_topology_cr(name, levels, api_version) for name, levels in TOPOLOGIES.items()]
    kubectl_apply_docs(topology_docs, kflags)
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
@click.option("--cq-cpu", default="384", help="ClusterQueue cpu nominal quota (default: 384).")
@click.option("--cq-memory", default="1536Gi", help="ClusterQueue memory nominal quota (default: 1536Gi).")
@click.option("--cq-gpu", default="24", help="ClusterQueue nvidia.com/gpu quota (default: 24).")
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
def main(
    variant: str,
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
    """Install + configure Kueue (coreweave or upstream) for Iris gang admission."""
    run_install(
        variant=variant,
        kubeconfig=kubeconfig,
        context=context,
        chart_version=chart_version,
        release=release,
        with_queues=with_queues,
        cluster_queue=cluster_queue,
        cq_cpu=cq_cpu,
        cq_memory=cq_memory,
        cq_gpu=cq_gpu,
        apply=apply,
    )


if __name__ == "__main__":
    main()
