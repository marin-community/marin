#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install the CoreWeave (CKS) cluster-wide prerequisites for a public /proxy ingress.

CKS ships no ingress controller and no TLS issuer, so the controller's ``/proxy``
Ingress (created per-cluster by ``K8sControllerProvider.start_controller``) needs
two cluster-wide components installed first. This script installs them, once, from
the operator's kubeconfig (the controller's own ServiceAccount can't install CRDs):

  * **Traefik** — the ingress controller; CKS gives its LoadBalancer a wildcard
    ``*.coreweave.app`` DNS name.
  * **cert-manager** — issues the TLS cert Traefik terminates.

It also creates **HTTP-01** Let's Encrypt ClusterIssuers. CoreWeave's bundled
issuers use DNS-01 against ``acme.coreweave.com`` and only cover
``*.coreweave.app``; a custom host (e.g. ``iris-cw.oa.dev`` CNAME'd to the
coreweave.app FQDN) needs HTTP-01, which validates through Traefik for any host
that resolves to the LoadBalancer.

Target a cluster with ``--cluster <name>`` (resolved the same way as ``iris
--cluster``): the host, ingress class, and kubeconfig come from that config's
``controller.coreweave`` block, so it matches what the controller will use. For a
cluster with no checked-in config, point at it with ``--context``/``--kubeconfig``
and pass ``--host`` directly. Run it once per CoreWeave cluster (Traefik +
cert-manager are per-cluster).

Without ``--apply`` it prints the helm commands and ClusterIssuer manifests and
stops. With ``--apply`` it installs them, then reads the Traefik LoadBalancer's
allocated ``*.coreweave.app`` FQDN and prints the DNS record to create and the
restart/config step.

``uninstall`` tears the whole stack back down — releases, namespaces, and the
cluster-scoped CRDs/webhooks/RBAC/IngressClass a namespace delete would orphan —
then verifies nothing remains. Both subcommands only print the plan without
``--apply``.

Usage:
    uv run lib/iris/scripts/install_traefik_proxy.py --cluster cw-us-east-02a \\
        install --acme-email you@oa.dev [--apply]
    uv run lib/iris/scripts/install_traefik_proxy.py --cluster cw-us-east-02a uninstall [--apply]
"""

import subprocess
import time
from typing import NamedTuple

import click
import yaml
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.cluster.config import load_config
from rigging.config_discovery import resolve_cluster_config

DEFAULT_INGRESS_CLASS = "traefik"

CW_REPO_NAME = "coreweave"
CW_REPO_URL = "https://charts.core-services.ingress.coreweave.com"
TRAEFIK_CHART = f"{CW_REPO_NAME}/traefik"
CERT_MANAGER_CHART = f"{CW_REPO_NAME}/cert-manager"

LE_ACME = {
    "prod": "https://acme-v02.api.letsencrypt.org/directory",
    "staging": "https://acme-staging-v02.api.letsencrypt.org/directory",
}
# ClusterIssuer names this script creates; use as controller.coreweave.cluster_issuer.
ISSUER_NAMES = {"prod": "letsencrypt-http01-prod", "staging": "letsencrypt-http01-staging"}

CLUSTERISSUER_CRD = "clusterissuers.cert-manager.io"
_CRD_WAIT_SECONDS = 120.0

# API groups whose CRDs the charts register; a namespace delete orphans these.
CRD_GROUPS = ("cert-manager.io", "traefik.io")
# Cluster-scoped kinds the charts create that `helm uninstall` normally removes
# but a namespace-only teardown leaves behind.
SWEEP_KINDS = (
    "validatingwebhookconfiguration",
    "mutatingwebhookconfiguration",
    "clusterrole",
    "clusterrolebinding",
)
INSTANCE_LABEL = "app.kubernetes.io/instance"

# CoreWeave's External Hostname Controller allocates the LB's *.coreweave.app
# FQDN asynchronously and reports it in the Service's ExternalRecords condition.
_FQDN_JSONPATH = '{.status.conditions[?(@.type=="ExternalRecords")].message}'
_FQDN_WAIT_SECONDS = 90.0
_COREWEAVE_APP = ".coreweave.app"


# --------------------------------------------------------------------------
# Thin I/O helpers (subprocess via arg lists — never shell=True).
# --------------------------------------------------------------------------
def helm_flags(kubeconfig: str | None, context: str | None) -> list[str]:
    flags: list[str] = []
    if kubeconfig:
        flags += ["--kubeconfig", kubeconfig]
    if context:
        flags += ["--kube-context", context]
    return flags


def kubectl_flags(kubeconfig: str | None, context: str | None) -> list[str]:
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


def kubectl_apply_docs(docs: list[dict], kflags: list[str]) -> None:
    """Apply a list of manifests via ``kubectl apply -f -`` (stdin)."""
    manifests = yaml.safe_dump_all(docs, default_flow_style=False, sort_keys=False)
    proc = run(["kubectl", *kflags, "apply", "-f", "-"], input=manifests, text=True)
    if proc.returncode != 0:
        raise click.ClickException("kubectl apply of ClusterIssuers failed")


def wait_for_crd(crd: str, kflags: list[str]) -> None:
    """Block until ``crd`` is established (cert-manager registers its CRDs async)."""
    deadline = time.monotonic() + _CRD_WAIT_SECONDS
    while time.monotonic() < deadline:
        result = run(
            ["kubectl", *kflags, "get", "crd", crd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return
        time.sleep(3.0)
    raise click.ClickException(f"CRD {crd} not present after {_CRD_WAIT_SECONDS:.0f}s (is cert-manager installed?)")


def read_traefik_fqdn(release: str, namespace: str, kflags: list[str]) -> str:
    """Poll the Traefik LoadBalancer Service for its allocated ``*.coreweave.app`` FQDN.

    Returns the bare FQDN, or "" if not allocated within the wait window (the
    External Hostname Controller assigns it asynchronously). Polls quietly (no
    per-attempt command echo) since it may take several tries.
    """
    click.secho(f"==> Reading Traefik LoadBalancer FQDN (svc/{release} -n {namespace}) …", fg="blue", bold=True)
    deadline = time.monotonic() + _FQDN_WAIT_SECONDS
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["kubectl", *kflags, "get", "svc", release, "-n", namespace, "-o", f"jsonpath={_FQDN_JSONPATH}"],
            capture_output=True,
            text=True,
        )
        for token in (result.stdout or "").replace(",", " ").split():
            candidate = token.strip(".;\"'")
            if candidate.endswith(_COREWEAVE_APP):
                return candidate
        time.sleep(3.0)
    return ""


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------
def _http01_issuer(env: str, email: str, ingress_class: str) -> dict:
    """A Let's Encrypt HTTP-01 ClusterIssuer validated through ``ingress_class``.

    HTTP-01 (not CoreWeave's bundled DNS-01) so it can issue for a custom host,
    which the coreweave.app DNS-01 webhook cannot. Requires the host to already
    resolve to the Traefik LoadBalancer before issuance.
    """
    return {
        "apiVersion": "cert-manager.io/v1",
        "kind": "ClusterIssuer",
        "metadata": {"name": ISSUER_NAMES[env]},
        "spec": {
            "acme": {
                "server": LE_ACME[env],
                "email": email,
                "privateKeySecretRef": {"name": f"{ISSUER_NAMES[env]}-account-key"},
                "solvers": [{"http01": {"ingress": {"ingressClassName": ingress_class}}}],
            }
        },
    }


# --------------------------------------------------------------------------
# Shared settings, resolved once at the command group
# --------------------------------------------------------------------------
class StackSettings(NamedTuple):
    """Target cluster and chart placement shared by every subcommand."""

    cluster: str
    host: str
    ingress_class: str
    traefik_namespace: str
    traefik_release: str
    cert_manager_namespace: str
    cert_manager_release: str
    kubeconfig: str | None
    context: str | None


class ClusterIngressSettings(NamedTuple):
    """CoreWeave ingress settings derived from a named Iris cluster config."""

    host: str
    ingress_class: str
    kubeconfig: str


def _derive_from_cluster(name: str) -> ClusterIngressSettings:
    """Read a named Iris cluster config and return its CoreWeave ingress settings.

    Resolves ``name`` the same way ``iris --cluster`` does, so the host, ingress
    class, and kubeconfig come from the one config the controller also reads.
    """
    try:
        path = resolve_cluster_config(name, dirs=IRIS_CLUSTER_CONFIG_DIRS)
    except FileNotFoundError as exc:
        raise click.ClickException(f"Unknown cluster {name!r}; run `iris cluster list`.") from exc
    config = load_config(str(path))
    cw = config.controller.coreweave
    if cw is None:
        raise click.ClickException(f"Cluster {name!r} has no controller.coreweave block — not a CoreWeave cluster.")
    platform = config.platform.coreweave
    return ClusterIngressSettings(
        host=cw.public_proxy_host,
        ingress_class=cw.ingress_class,
        kubeconfig=platform.kubeconfig_path if platform else "",
    )


# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
def install(
    settings: StackSettings,
    *,
    acme_email: str | None,
    traefik_version: str | None,
    cert_manager_version: str | None,
    skip_traefik: bool,
    skip_cert_manager: bool,
    skip_issuers: bool,
    apply: bool,
) -> None:
    hflags = helm_flags(settings.kubeconfig, settings.context)
    kflags = kubectl_flags(settings.kubeconfig, settings.context)

    if not skip_issuers and not acme_email:
        raise click.ClickException(
            "--acme-email is required to create the HTTP-01 ClusterIssuers (or pass --skip-issuers)"
        )

    issuer_docs = (
        [_http01_issuer(env, acme_email, settings.ingress_class) for env in ("staging", "prod")]
        if not skip_issuers and acme_email
        else []
    )

    click.secho("==> Plan (CoreWeave /proxy ingress prerequisites):", fg="blue", bold=True)
    if not skip_traefik:
        click.echo(
            f"  helm upgrade --install {settings.traefik_release} {TRAEFIK_CHART} "
            f"-n {settings.traefik_namespace} --create-namespace"
        )
    if not skip_cert_manager:
        click.echo(
            f"  helm upgrade --install {settings.cert_manager_release} {CERT_MANAGER_CHART} "
            f"-n {settings.cert_manager_namespace} --create-namespace"
        )
    if issuer_docs:
        click.secho("==> ClusterIssuers (HTTP-01 via Traefik):", fg="blue", bold=True)
        click.echo(yaml.safe_dump_all(issuer_docs, default_flow_style=False, sort_keys=False))

    if not apply:
        click.secho("\nwarn: dry run — nothing applied. Re-run with --apply to install.", fg="yellow", err=True)
        return

    click.secho(f"==> Adding/updating helm repo {CW_REPO_NAME} ({CW_REPO_URL})", fg="blue", bold=True)
    run(["helm", "repo", "add", CW_REPO_NAME, CW_REPO_URL], check=True, stdout=subprocess.DEVNULL)
    run(["helm", "repo", "update", CW_REPO_NAME], check=True, stdout=subprocess.DEVNULL)

    if not skip_traefik:
        click.secho(
            f"==> Installing Traefik ({TRAEFIK_CHART}) in namespace {settings.traefik_namespace}", fg="blue", bold=True
        )
        _helm_install(TRAEFIK_CHART, settings.traefik_release, settings.traefik_namespace, traefik_version, hflags)

    if not skip_cert_manager:
        click.secho(
            f"==> Installing cert-manager ({CERT_MANAGER_CHART}) in namespace {settings.cert_manager_namespace}",
            fg="blue",
            bold=True,
        )
        _helm_install(
            CERT_MANAGER_CHART,
            settings.cert_manager_release,
            settings.cert_manager_namespace,
            cert_manager_version,
            hflags,
        )

    if issuer_docs:
        click.secho("==> Waiting for the cert-manager ClusterIssuer CRD, then applying issuers", fg="blue", bold=True)
        wait_for_crd(CLUSTERISSUER_CRD, kflags)
        kubectl_apply_docs(issuer_docs, kflags)

    _print_next_steps(settings, skip_traefik=skip_traefik, skip_issuers=skip_issuers, kflags=kflags)


def _helm_install(chart: str, release: str, namespace: str, version: str | None, hflags: list[str]) -> None:
    cmd = ["helm", *hflags, "upgrade", "--install", release, chart, "-n", namespace, "--create-namespace", "--wait"]
    if version:
        cmd += ["--version", version]
    result = run(cmd)
    if result.returncode != 0:
        raise click.ClickException(f"helm install of {chart} failed")


def _print_next_steps(settings: StackSettings, *, skip_traefik: bool, skip_issuers: bool, kflags: list[str]) -> None:
    """Print exactly what the operator must do next — the DNS record and config."""
    host_label = settings.host or "<your-host>.oa.dev"
    click.secho("==> Done. Two steps to finish wiring the public /proxy route:", fg="green", bold=True)

    # 1. The concrete DNS record. Read the real Traefik FQDN so we can print it.
    click.secho("  1) Create this DNS record at your DNS provider (e.g. Namecheap):", fg="green", bold=True)
    fqdn = "" if skip_traefik else read_traefik_fqdn(settings.traefik_release, settings.traefik_namespace, kflags)
    if fqdn:
        click.echo(f"        {host_label}   CNAME   {fqdn}")
    else:
        click.secho("       (Traefik's FQDN isn't allocated yet — read it in a minute, then CNAME to it:)", fg="yellow")
        click.echo(
            f"       kubectl get svc {settings.traefik_release} -n {settings.traefik_namespace} "
            "-o=jsonpath='{.status.conditions[?(@.type==\"ExternalRecords\")].message}'"
        )
        click.echo(f"        {host_label}   CNAME   <that>{_COREWEAVE_APP}")

    # 2. The controller config. With --cluster and a host already set, the block
    # is already in the config — just restart. Otherwise print the block to add.
    if settings.cluster and settings.host:
        click.secho(
            f"  2) {settings.cluster} already sets controller.coreweave (public_proxy_host={settings.host}) — "
            "restart the controller so it creates the /proxy Ingress:",
            fg="green",
            bold=True,
        )
        click.echo(f"       iris --cluster {settings.cluster} cluster restart")
    else:
        click.secho(
            "  2) Set the cluster config's controller.coreweave block, then (re)start the controller:",
            fg="green",
            bold=True,
        )
        click.echo(f"       public_proxy_host: {settings.host or 'your-host.oa.dev'}")
        click.echo(f"       ingress_class: {settings.ingress_class}")
        click.echo("       tls_secret: iris-controller-proxy-tls")
        if not skip_issuers:
            click.echo(
                f"       cluster_issuer: {ISSUER_NAMES['staging']}   # switch to {ISSUER_NAMES['prod']} once verified"
            )
    click.secho(
        "  HTTP-01 issuance needs the CNAME live first (Let's Encrypt fetches http://<host>/.well-known/…); "
        "use the staging issuer first to avoid LE rate limits, then flip to prod.",
        fg="yellow",
    )


# --------------------------------------------------------------------------
# Uninstall
# --------------------------------------------------------------------------
# kubectl's error for a kind the API server doesn't know (e.g. a CRD already
# deleted); from k8s.io/cli-runtime's resource builder: "the server doesn't
# have a resource type %q". Any other nonzero exit (auth, network, …) is a
# real failure and must not read as "nothing left".
_UNKNOWN_KIND_MARKER = "doesn't have a resource type"


def _resource_names(kind: str, kflags: list[str], selector: str | None = None) -> list[str]:
    """List ``kind/name`` strings for ``kind``; [] when the kind itself is unknown (CRD gone)."""
    cmd = ["kubectl", *kflags, "get", kind, "-o", "name"]
    if selector:
        cmd += ["-l", selector]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.split()
    if _UNKNOWN_KIND_MARKER in result.stderr:
        return []
    raise click.ClickException(f"`{' '.join(cmd)}` failed: {result.stderr.strip()}")


def _short_name(qualified: str) -> str:
    """``customresourcedefinition.apiextensions.k8s.io/foo.cert-manager.io`` -> ``foo.cert-manager.io``."""
    return qualified.rsplit("/", 1)[-1]


def _group_crds(kflags: list[str]) -> list[str]:
    """CRDs belonging to the chart-registered API groups (``CRD_GROUPS``)."""
    suffixes = tuple(f".{group}" for group in CRD_GROUPS)
    return [crd for crd in _resource_names("crd", kflags) if _short_name(crd).endswith(suffixes)]


def _release_scoped(kind: str, releases: tuple[str, ...], kflags: list[str]) -> list[str]:
    """Cluster-scoped ``kind`` objects belonging to ``releases``.

    Matches by the ``app.kubernetes.io/instance`` label where the charts set it,
    plus a name-prefix fallback for objects created without instance labels.
    """
    found: set[str] = set()
    all_names = _resource_names(kind, kflags)
    for release in releases:
        found.update(_resource_names(kind, kflags, selector=f"{INSTANCE_LABEL}={release}"))
        found.update(name for name in all_names if _short_name(name).startswith(release))
    return sorted(found)


def _kubectl_delete(names: list[str], kflags: list[str]) -> None:
    run(["kubectl", *kflags, "delete", *names, "--ignore-not-found"], check=True)


def _helm_release_installed(release: str, namespace: str, hflags: list[str]) -> bool:
    result = subprocess.run(
        ["helm", *hflags, "status", release, "-n", namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def uninstall(settings: StackSettings, *, apply: bool) -> None:
    hflags = helm_flags(settings.kubeconfig, settings.context)
    kflags = kubectl_flags(settings.kubeconfig, settings.context)
    ingress_class = settings.ingress_class
    release_pairs = (
        (settings.traefik_release, settings.traefik_namespace),
        (settings.cert_manager_release, settings.cert_manager_namespace),
    )
    releases = (settings.traefik_release, settings.cert_manager_release)
    issuers = [ISSUER_NAMES[env] for env in ("staging", "prod")]
    namespaces = list(dict.fromkeys([settings.traefik_namespace, settings.cert_manager_namespace]))

    click.secho("==> Plan (teardown of the CoreWeave /proxy ingress prerequisites):", fg="blue", bold=True)
    click.echo(f"  kubectl delete clusterissuer {' '.join(issuers)}   # first, while the CRD still exists")
    for release, namespace in release_pairs:
        click.echo(f"  helm uninstall {release} -n {namespace}")
    click.echo(f"  kubectl delete namespace {' '.join(namespaces)}")
    groups = " / ".join(f"*.{group}" for group in CRD_GROUPS)
    click.echo(f"  kubectl delete crd <every {groups} CRD>")
    click.echo(
        f"  kubectl delete <{', '.join(SWEEP_KINDS)} with "
        f"{INSTANCE_LABEL} in ({', '.join(releases)}) or name-prefixed by a release>"
    )
    click.echo(f"  kubectl delete ingressclass {ingress_class}")
    click.echo("  then verify nothing is left")

    if not apply:
        click.secho("\nwarn: dry run — nothing deleted. Re-run with --apply to uninstall.", fg="yellow", err=True)
        return

    # 1. ClusterIssuers first: once the CRD is deleted they become unaddressable.
    click.secho("==> Deleting ClusterIssuers (before the CRD goes)", fg="blue", bold=True)
    if CLUSTERISSUER_CRD in {_short_name(crd) for crd in _resource_names("crd", kflags)}:
        present = {_short_name(name) for name in _resource_names("clusterissuer", kflags)}
        for name in (issuer for issuer in issuers if issuer not in present):
            click.echo(f"  clusterissuer/{name}: already absent")
        found = [f"clusterissuer/{issuer}" for issuer in issuers if issuer in present]
        if found:
            _kubectl_delete(found, kflags)
    else:
        click.echo(f"  CRD {CLUSTERISSUER_CRD} already absent — no ClusterIssuers to delete")

    # 2. helm uninstall both releases (helm removes its own cluster-scoped objects).
    for release, namespace in release_pairs:
        click.secho(f"==> Uninstalling helm release {release} (namespace {namespace})", fg="blue", bold=True)
        if not _helm_release_installed(release, namespace, hflags):
            click.echo(f"  release {release} not installed in {namespace} — already absent")
            continue
        result = run(["helm", *hflags, "uninstall", release, "-n", namespace, "--wait"])
        if result.returncode != 0:
            raise click.ClickException(f"helm uninstall of {release} failed")

    # 3. The namespaces the installs created.
    click.secho("==> Deleting namespaces", fg="blue", bold=True)
    existing = {_short_name(name) for name in _resource_names("namespace", kflags)}
    for namespace in (ns for ns in namespaces if ns not in existing):
        click.echo(f"  namespace/{namespace}: already absent")
    found = [f"namespace/{ns}" for ns in namespaces if ns in existing]
    if found:
        _kubectl_delete(found, kflags)

    # 4. Sweep the cluster-scoped leftovers a namespace delete orphans.
    click.secho("==> Sweeping cluster-scoped leftovers", fg="blue", bold=True)
    crds = _group_crds(kflags)
    if crds:
        _kubectl_delete(crds, kflags)
    else:
        click.echo(f"  CRDs ({groups}): already absent")
    for kind in SWEEP_KINDS:
        names = _release_scoped(kind, releases, kflags)
        if names:
            _kubectl_delete(names, kflags)
        else:
            click.echo(f"  {kind}: already absent")
    if any(_short_name(name) == ingress_class for name in _resource_names("ingressclass", kflags)):
        _kubectl_delete([f"ingressclass/{ingress_class}"], kflags)
    else:
        click.echo(f"  ingressclass/{ingress_class}: already absent")

    # 5. Verify: list anything still present rather than assuming the sweeps worked.
    click.secho("==> Verifying teardown", fg="blue", bold=True)
    leftovers: list[str] = []
    for release, namespace in release_pairs:
        if _helm_release_installed(release, namespace, hflags):
            leftovers.append(f"helm release {release} (namespace {namespace})")
    existing = {_short_name(name) for name in _resource_names("namespace", kflags)}
    leftovers += [f"namespace/{ns}" for ns in namespaces if ns in existing]
    leftovers += _group_crds(kflags)
    for kind in SWEEP_KINDS:
        leftovers += _release_scoped(kind, releases, kflags)
    leftovers += [
        f"ingressclass/{ingress_class}"
        for name in _resource_names("ingressclass", kflags)
        if _short_name(name) == ingress_class
    ]
    if leftovers:
        click.secho("warn: still present after teardown:", fg="yellow", err=True)
        for item in leftovers:
            click.echo(f"  {item}", err=True)
        raise click.ClickException("teardown incomplete — delete the resources above by hand")
    click.secho(
        "==> Teardown complete — no releases, namespaces, or cluster-scoped leftovers remain.", fg="green", bold=True
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
@click.group()
@click.option(
    "--cluster",
    default="",
    help="Iris cluster name; supplies --host, --ingress-class, and --kubeconfig from its controller.coreweave block.",
)
@click.option("--host", default="", help="Public endpoint host (e.g. iris-cw.oa.dev) [default: the cluster's].")
@click.option(
    "--ingress-class",
    default="",
    help=f"IngressClass the HTTP-01 solver routes through [default: the cluster's, else {DEFAULT_INGRESS_CLASS}].",
)
@click.option("--traefik-namespace", default="traefik", show_default=True)
@click.option("--traefik-release", default="traefik", show_default=True, help="helm release name for Traefik.")
@click.option("--cert-manager-namespace", default="cert-manager", show_default=True)
@click.option(
    "--cert-manager-release", default="cert-manager", show_default=True, help="helm release name for cert-manager."
)
@click.option(
    "--kubeconfig", default="", help="kubeconfig to use [default: the cluster's, else $KUBECONFIG / ~/.kube/config]."
)
@click.option("--context", default=None, help="kube context to target.")
@click.pass_context
def main(
    ctx: click.Context,
    cluster: str,
    host: str,
    ingress_class: str,
    traefik_namespace: str,
    traefik_release: str,
    cert_manager_namespace: str,
    cert_manager_release: str,
    kubeconfig: str,
    context: str | None,
) -> None:
    """Traefik + cert-manager + HTTP-01 issuers for the CoreWeave /proxy ingress."""
    derived = _derive_from_cluster(cluster) if cluster else ClusterIngressSettings("", "", "")
    ctx.obj = StackSettings(
        cluster=cluster,
        host=host or derived.host,
        ingress_class=ingress_class or derived.ingress_class or DEFAULT_INGRESS_CLASS,
        traefik_namespace=traefik_namespace,
        traefik_release=traefik_release,
        cert_manager_namespace=cert_manager_namespace,
        cert_manager_release=cert_manager_release,
        kubeconfig=kubeconfig or derived.kubeconfig or None,
        context=context,
    )


@main.command("install")
@click.option(
    "--acme-email",
    default=None,
    help="Email for the Let's Encrypt HTTP-01 ClusterIssuers (required unless --skip-issuers).",
)
@click.option("--traefik-version", default=None, help="Pin the Traefik chart version (default: latest).")
@click.option("--cert-manager-version", default=None, help="Pin the cert-manager chart version (default: latest).")
@click.option("--skip-traefik", is_flag=True, help="Do not install Traefik (already present).")
@click.option("--skip-cert-manager", is_flag=True, help="Do not install cert-manager (already present).")
@click.option("--skip-issuers", is_flag=True, help="Do not create the HTTP-01 ClusterIssuers.")
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
@click.pass_obj
def install_cmd(
    settings: StackSettings,
    acme_email: str | None,
    traefik_version: str | None,
    cert_manager_version: str | None,
    skip_traefik: bool,
    skip_cert_manager: bool,
    skip_issuers: bool,
    apply: bool,
) -> None:
    """Install the cluster-wide ingress prerequisites."""
    install(
        settings,
        acme_email=acme_email,
        traefik_version=traefik_version,
        cert_manager_version=cert_manager_version,
        skip_traefik=skip_traefik,
        skip_cert_manager=skip_cert_manager,
        skip_issuers=skip_issuers,
        apply=apply,
    )


@main.command("uninstall")
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
@click.pass_obj
def uninstall_cmd(settings: StackSettings, apply: bool) -> None:
    """Tear the whole stack back down and verify nothing remains."""
    uninstall(settings, apply=apply)


if __name__ == "__main__":
    main()
