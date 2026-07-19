#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Set up everything a CoreWeave (CKS) controller needs on the network that the controller can't install itself.

CKS ships no ingress controller and no TLS issuer, and the controller's own
ServiceAccount can't install CRDs, so an operator runs this once per cluster from
their kubeconfig. It installs, idempotently, the whole networking stack in one
pass so there is no "ran one script but not the other" gap:

  * **Traefik** — the ingress controller; CKS gives its LoadBalancer a wildcard
    ``*.coreweave.app`` DNS name.
  * **cert-manager** — issues the TLS certs Traefik terminates.
  * **HTTP-01 Let's Encrypt ClusterIssuers** — CoreWeave's bundled issuers use
    DNS-01 against ``acme.coreweave.com`` and only cover ``*.coreweave.app``; a
    custom host (``iris-cw-<cluster>.oa.dev`` CNAME'd to the coreweave.app FQDN)
    needs HTTP-01, which validates through Traefik for any host that resolves to
    the LoadBalancer.
  * **The federation ingress** — an IP-locked route publishing the controller's
    RPC surface to the GCP ``marin`` controller (see below).

Federation ingress — how the GCP ``marin`` controller reaches this one:

    marin federates whole jobs to the CoreWeave controllers by dialing their RPC
    surface directly (the pull model — CoreWeave must be reachable *inbound*).
    CoreWeave has no user surface of its own: end users reach Iris only through
    ``iris.oa.dev`` (IAP), and marin federates outward, so the only external caller
    of a CoreWeave controller is the marin controller. That makes the auth surface
    two factors, both required and neither alone sufficient:

      1. **IP allowlist** — a Traefik ``ipAllowList`` Middleware admits only the
         marin-side controllers' egress IPs (``FEDERATION_ALLOW_SOURCES``). The
         *network* factor.
      2. **The controller's own auth** — the ``aud="federation"`` verifier for the
         handoff RPCs, the general auth chain for the rest. The *identity* factor.

    Because the controller is the identity gate, it MUST be enforcing. A permissive
    (null-auth) controller behind only an IP lock hands anonymous admin over its
    entire control plane to anything arriving from the allowlisted IP — so this
    script warns loudly when its target is still permissive. Both CoreWeave
    controllers enforce via ``trusted_cidrs``.

    A single ``ipAllowList`` catch-all route covers the WHOLE controller host
    (``iris-cw-<cluster>.oa.dev``) — there is no world-open surface. cert-manager
    issues the TLS cert. Install removes the controller's legacy world-open
    ``/proxy`` Ingress (``iris-controller-proxy``) so no path stays public.

Target a cluster with ``--cluster <name>`` (resolved the same way ``iris --cluster``
does): the host, ingress class, namespace, controller Service, and kubeconfig all
come from that config's ``controller.coreweave`` block, so they match what the
controller itself uses.

Without ``--apply`` every subcommand only prints its plan. ``uninstall`` tears the
whole stack back down — the federation route, the helm releases, their namespaces,
and the cluster-scoped CRDs/webhooks/RBAC/IngressClass a namespace delete would
orphan — then verifies nothing remains.

Usage:
    # Dry-run (default): prints the helm commands, manifests, and pre-flight findings.
    uv run lib/iris/scripts/install_cw_network.py --cluster cw-rno2a \\
        install --acme-email you@oa.dev
    # Apply the whole stack, then CNAME the host to the Traefik LoadBalancer FQDN:
    uv run lib/iris/scripts/install_cw_network.py --cluster cw-rno2a \\
        install --acme-email you@oa.dev --apply
    # Re-run to converge just the federation route (skip the already-installed stack):
    uv run lib/iris/scripts/install_cw_network.py --cluster cw-rno2a install \\
        --skip-traefik --skip-cert-manager --skip-issuers \\
        --cluster-issuer letsencrypt-http01-prod --apply
    # Tear the whole stack down and verify nothing remains:
    uv run lib/iris/scripts/install_cw_network.py --cluster cw-rno2a uninstall --apply
"""

import os
import subprocess
import time
from typing import NamedTuple

import click
import yaml
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.cluster.config import load_config
from iris.cluster.platforms.k8s.network_manifests import (
    CERT_MANAGER_CHART,
    CLUSTERISSUER_CRD,
    CONTROLLER_PROXY_INGRESS_NAME,
    CW_REPO_NAME,
    CW_REPO_URL,
    DEFAULT_CLUSTER_ISSUER,
    DEFAULT_TLS_SECRET,
    INGRESS_NAME,
    ISSUER_NAMES,
    MIDDLEWARE_CRD,
    MIDDLEWARE_NAME,
    TRAEFIK_CHART,
    default_federation_host,
    federation_ingress,
    http01_issuer,
    ipallowlist_middleware,
    normalize_source,
)
from rigging.config_discovery import resolve_cluster_config

DEFAULT_INGRESS_CLASS = "traefik"

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

# Egress addresses of the marin-side controllers that federate into a CoreWeave cluster,
# reserved as iris-marin-fed-egress and iris-marin-dev-fed-egress in hai-gcp-models. The
# Middleware's sourceRange is replaced wholesale on every install, never merged, so an
# install that names a subset silently strands the omitted cluster at a 403.
FEDERATION_ALLOW_SOURCES = ("34.27.183.11", "35.254.13.19")
# _auth_mode's value for a permissive controller (no login arm, no trusted_cidrs).
_NULL_AUTH = "null-auth"


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


def kubectl_apply_docs(docs: list[dict], kflags: list[str], what: str) -> None:
    """Apply a list of manifests via ``kubectl apply -f -`` (stdin)."""
    manifests = yaml.safe_dump_all(docs, default_flow_style=False, sort_keys=False)
    proc = run(["kubectl", *kflags, "apply", "-f", "-"], input=manifests, text=True)
    if proc.returncode != 0:
        raise click.ClickException(f"kubectl apply of {what} failed")


def resource_present(kind: str, name: str, kflags: list[str]) -> bool:
    """True if ``kind/name`` exists; queries quietly, emitting no output."""
    result = subprocess.run(
        ["kubectl", *kflags, "get", kind, name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def wait_for_crd(crd: str, kflags: list[str]) -> None:
    """Block until ``crd`` is established (cert-manager registers its CRDs async)."""
    deadline = time.monotonic() + _CRD_WAIT_SECONDS
    while time.monotonic() < deadline:
        if resource_present("crd", crd, kflags):
            return
        time.sleep(3.0)
    raise click.ClickException(f"CRD {crd} not present after {_CRD_WAIT_SECONDS:.0f}s (is cert-manager installed?)")


def read_traefik_fqdn(release: str, namespace: str, kflags: list[str], host: str) -> str:
    """Poll the Traefik LoadBalancer Service for a CNAME target under ``*.coreweave.app``.

    The External Hostname Controller allocates a wildcard record — every label under
    it resolves to the LoadBalancer — so the wildcard is resolved against ``host``'s
    own first label to yield a name a CNAME can actually point at. Returns "" if the
    record is not allocated within the wait window (allocation is asynchronous).
    Polls quietly (no per-attempt command echo) since it may take several tries.
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
                return candidate.replace("*", host.split(".", 1)[0], 1)
        time.sleep(3.0)
    return ""


# --------------------------------------------------------------------------
# Settings, resolved once at the command group
# --------------------------------------------------------------------------
class NetworkSettings(NamedTuple):
    """Target cluster, controller coordinates, and chart placement for the CW network stack."""

    cluster: str
    namespace: str  # controller namespace; the federation Ingress + Middleware live here
    service_name: str  # controller Service the federation route backends onto
    port: int
    ingress_class: str
    auth_mode: str  # "null-auth" (permissive) | "iap" | "cidr"
    kubeconfig: str | None
    context: str | None
    traefik_namespace: str
    traefik_release: str
    cert_manager_namespace: str
    cert_manager_release: str


def _auth_mode(config) -> str:
    """The controller's request-auth mode.

    ``null-auth`` means the RPC surface is PERMISSIVE: no login-provider arm and
    no ``trusted_cidrs``, so every request is admitted as the anonymous admin.
    """
    auth = config.auth
    if auth is None or (auth.provider_kind() is None and not auth.trusted_cidrs):
        return _NULL_AUTH
    return auth.provider_kind() or "cidr"


class _DerivedConfig(NamedTuple):
    """Controller coordinates read from a named Iris cluster config."""

    namespace: str
    service_name: str
    port: int
    ingress_class: str
    auth_mode: str
    kubeconfig: str
    context: str


def _derive_from_cluster(name: str) -> _DerivedConfig:
    """Read a named Iris cluster config and return its CoreWeave controller coordinates.

    Resolves ``name`` the same way ``iris --cluster`` does, so the namespace,
    controller Service, kubeconfig, and kube context match what the controller
    itself uses.
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
    return _DerivedConfig(
        namespace=(platform.namespace if platform else "") or "iris",
        service_name=cw.service_name or "iris-controller-svc",
        port=cw.port or 10000,
        ingress_class=cw.ingress_class or DEFAULT_INGRESS_CLASS,
        auth_mode=_auth_mode(config),
        kubeconfig=(platform.kubeconfig_path if platform else "") or "",
        context=(platform.kube_context if platform else "") or "",
    )


def _normalize_source(value: str) -> str:
    """CLI-friendly wrapper around ``normalize_source``: bad input -> ``click.BadParameter``."""
    try:
        return normalize_source(value)
    except ValueError as exc:
        raise click.BadParameter(f"{value!r} is not a valid IP or CIDR: {exc}", param_hint="--allow-source") from exc


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
def _warn_if_permissive(settings: NetworkSettings) -> None:
    """Warn when the target controller admits unauthenticated callers.

    With the federation route exposing the controller behind only an IP allowlist,
    a permissive controller hands anonymous admin over its whole control plane to
    anything from the allowlisted IP — so the controller must be enforcing before
    that route is a real boundary.
    """
    if settings.auth_mode != _NULL_AUTH:
        return
    click.secho(
        f"\nwarn: {settings.cluster} runs NULL-AUTH (permissive) — its RPC surface admits any caller\n"
        "      as the anonymous admin. Behind only an IP allowlist that means the ENTIRE control\n"
        "      plane is exposed to the allowlisted IP with no identity check. Make the controller\n"
        "      enforcing first (add auth.trusted_cidrs, as cw-us-east-02a has) so an off-cluster\n"
        "      request must present a valid bearer. See docs/coreweave.md.",
        fg="yellow",
        err=True,
    )


# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
def install(
    settings: NetworkSettings,
    *,
    host: str,
    source_ranges: list[str],
    tls_secret: str,
    cluster_issuer: str,
    xff_depth: int,
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
        [http01_issuer(env, acme_email, settings.ingress_class) for env in ("staging", "prod")]
        if not skip_issuers and acme_email
        else []
    )
    middleware = ipallowlist_middleware(namespace=settings.namespace, source_ranges=source_ranges, xff_depth=xff_depth)
    ingress = federation_ingress(
        namespace=settings.namespace,
        service_name=settings.service_name,
        port=settings.port,
        host=host,
        ingress_class=settings.ingress_class,
        tls_secret=tls_secret,
        cluster_issuer=cluster_issuer,
    )

    click.secho(f"==> Plan (CoreWeave network stack for {settings.cluster}):", fg="blue", bold=True)
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
    click.secho("==> Federation ingress (IP-locked; the whole controller host):", fg="blue", bold=True)
    click.echo(f"  host:         {host}  (CNAME -> the Traefik LoadBalancer FQDN)")
    click.echo(f"  allowlist:    {', '.join(source_ranges)}")
    click.echo(f"  backend:      {settings.service_name}:{settings.port} (namespace {settings.namespace})")
    click.echo(f"  tls secret:   {tls_secret}  (cert-manager issues via {cluster_issuer})")
    click.echo(f"  removes:      the controller's world-open {CONTROLLER_PROXY_INGRESS_NAME} Ingress, if present")
    click.echo(yaml.safe_dump_all([middleware, ingress], default_flow_style=False, sort_keys=False))

    _warn_if_permissive(settings)

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
        kubectl_apply_docs(issuer_docs, kflags, "ClusterIssuers")

    click.secho("==> Applying the federation ingress", fg="blue", bold=True)
    if not resource_present("crd", MIDDLEWARE_CRD, kflags):
        raise click.ClickException(
            f"Traefik Middleware CRD {MIDDLEWARE_CRD} not found — the ipAllowList cannot be applied.\n"
            "Install Traefik (drop --skip-traefik) before the federation route."
        )
    kubectl_apply_docs([middleware, ingress], kflags, "the federation ingress")
    _delete_controller_proxy_ingress(settings, kflags)

    _print_next_steps(settings, host=host, cluster_issuer=cluster_issuer)


def _helm_install(chart: str, release: str, namespace: str, version: str | None, hflags: list[str]) -> None:
    cmd = ["helm", *hflags, "upgrade", "--install", release, chart, "-n", namespace, "--create-namespace", "--wait"]
    if version:
        cmd += ["--version", version]
    result = run(cmd)
    if result.returncode != 0:
        raise click.ClickException(f"helm install of {chart} failed")


def _print_next_steps(settings: NetworkSettings, *, host: str, cluster_issuer: str) -> None:
    """Print exactly what the operator must do next — DNS, controller wiring, verification."""
    kflags = kubectl_flags(settings.kubeconfig, settings.context)
    click.secho("==> Done. To finish wiring the CoreWeave network:", fg="green", bold=True)

    # 1. DNS. The whole host resolves to Traefik's LB; the IP-locked ingress serves it.
    click.secho(f"  1) CNAME {host} at the Traefik LoadBalancer FQDN:", fg="green", bold=True)
    fqdn = read_traefik_fqdn(settings.traefik_release, settings.traefik_namespace, kflags, host)
    if fqdn:
        click.echo(f"        {host}   CNAME   {fqdn}")
    else:
        click.secho("       (Traefik's FQDN isn't allocated yet — read it in a minute, then CNAME to it:)", fg="yellow")
        click.echo(
            f"       kubectl get svc {settings.traefik_release} -n {settings.traefik_namespace} "
            f"-o=jsonpath='{_FQDN_JSONPATH}'"
        )
        click.echo(f"       (substitute {host.split('.', 1)[0]} for the '*' label — the record is a wildcard)")
        click.echo(f"        {host}   CNAME   <that>{_COREWEAVE_APP}")

    click.secho("  2) Verify federation reach from the marin controller VM:", fg="green", bold=True)
    click.echo("       - ListBackends WITH the federation JWT from the allowlisted egress IP  -> succeeds")
    click.echo("       - the same call WITHOUT the JWT (controller enforcing)                 -> UNAUTHENTICATED")
    click.echo("       - the same call from a non-allowlisted IP                              -> refused (403)")
    if cluster_issuer and "staging" in cluster_issuer:
        click.secho(
            "  3) Staging issuer in use — once the cert validates, re-run with --skip-traefik "
            "--skip-cert-manager --skip-issuers --cluster-issuer letsencrypt-http01-prod --apply.",
            fg="green",
        )
    if settings.auth_mode == _NULL_AUTH:
        click.secho(
            "  NOTE: this controller is still permissive — add auth.trusted_cidrs and restart so an "
            "off-cluster request must present a bearer; the IP allowlist alone is not an identity gate.",
            fg="yellow",
        )
    if cluster_issuer != DEFAULT_CLUSTER_ISSUER:
        click.secho(
            f"  NOTE: cluster-issuer is {cluster_issuer!r}, not the default {DEFAULT_CLUSTER_ISSUER!r}. If this "
            f"cluster is IaC-managed (has a provisioning.coreweave.ingress block), that annotation is also set "
            f"declaratively via active_cluster_issuer in lib/iris/config/{settings.cluster}.yaml — update it to "
            f"match, or the next `pulumi up` will silently revert this flip.",
            fg="yellow",
        )
    click.echo(f"  (inspect: kubectl get ingress {INGRESS_NAME} -n {settings.namespace} {' '.join(kflags)} -o wide)")


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


def _delete_federation_ingress(settings: NetworkSettings, kflags: list[str]) -> None:
    """Delete the federation Ingress and its ipAllowList Middleware from the controller namespace."""
    run(
        ["kubectl", *kflags, "delete", "ingress", INGRESS_NAME, "-n", settings.namespace, "--ignore-not-found"],
        check=True,
    )
    if resource_present("crd", MIDDLEWARE_CRD, kflags):
        run(
            [
                "kubectl",
                *kflags,
                "delete",
                MIDDLEWARE_CRD,
                MIDDLEWARE_NAME,
                "-n",
                settings.namespace,
                "--ignore-not-found",
            ],
            check=True,
        )


def _delete_controller_proxy_ingress(settings: NetworkSettings, kflags: list[str]) -> None:
    """Remove the controller's legacy world-open /proxy Ingress, if present.

    The IP-locked route supersedes it and covers the whole host; leaving it would
    keep /proxy world-open (Traefik prefers its longer prefix). Harmless where it
    was never created (e.g. cw-rno2a).
    """
    run(
        [
            "kubectl",
            *kflags,
            "delete",
            "ingress",
            CONTROLLER_PROXY_INGRESS_NAME,
            "-n",
            settings.namespace,
            "--ignore-not-found",
        ],
        check=True,
    )


def _scan_leftovers(
    settings: NetworkSettings,
    *,
    release_pairs: tuple[tuple[str, str], ...],
    releases: tuple[str, ...],
    namespaces: list[str],
    ingress_class: str,
    hflags: list[str],
    kflags: list[str],
) -> list[str]:
    """Everything the teardown should have removed that is still present.

    Re-queries the cluster rather than trusting the delete calls returned success,
    so an empty result is real evidence the teardown is complete.
    """
    leftovers: list[str] = []
    fed_ingress = subprocess.run(
        ["kubectl", *kflags, "get", "ingress", INGRESS_NAME, "-n", settings.namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if fed_ingress.returncode == 0:
        leftovers.append(f"ingress/{INGRESS_NAME} (namespace {settings.namespace})")
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
    return leftovers


def uninstall(settings: NetworkSettings, *, apply: bool) -> None:
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

    click.secho("==> Plan (teardown of the CoreWeave network stack):", fg="blue", bold=True)
    click.echo(f"  kubectl delete ingress {INGRESS_NAME} + middleware {MIDDLEWARE_NAME} -n {settings.namespace}")
    click.echo(
        f"  kubectl delete ingress {CONTROLLER_PROXY_INGRESS_NAME} -n {settings.namespace}   # the legacy /proxy route"
    )
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

    # 0. The federation route + the legacy /proxy Ingress (both in the controller
    # namespace, outside the chart teardown).
    click.secho("==> Deleting the federation ingress", fg="blue", bold=True)
    _delete_federation_ingress(settings, kflags)
    _delete_controller_proxy_ingress(settings, kflags)

    # 1. ClusterIssuers next: once the CRD is deleted they become unaddressable.
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

    # 5. Verify: re-scan for anything still present rather than assuming the sweeps worked.
    click.secho("==> Verifying teardown", fg="blue", bold=True)
    leftovers = _scan_leftovers(
        settings,
        release_pairs=release_pairs,
        releases=releases,
        namespaces=namespaces,
        ingress_class=ingress_class,
        hflags=hflags,
        kflags=kflags,
    )
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
    required=True,
    help="Iris cluster name; supplies host, ingress class, namespace, controller Service, and kubeconfig.",
)
@click.option("--kubeconfig", default="", help="kubeconfig to use [default: the cluster's].")
@click.option("--context", default="", help="kube context to target [default: the cluster's kube_context].")
@click.option("--traefik-namespace", default="traefik", show_default=True)
@click.option("--traefik-release", default="traefik", show_default=True, help="helm release name for Traefik.")
@click.option("--cert-manager-namespace", default="cert-manager", show_default=True)
@click.option(
    "--cert-manager-release", default="cert-manager", show_default=True, help="helm release name for cert-manager."
)
@click.pass_context
def main(
    ctx: click.Context,
    cluster: str,
    kubeconfig: str,
    context: str,
    traefik_namespace: str,
    traefik_release: str,
    cert_manager_namespace: str,
    cert_manager_release: str,
) -> None:
    """Traefik + cert-manager + HTTP-01 issuers + the IP-locked federation ingress for a CoreWeave cluster."""
    derived = _derive_from_cluster(cluster)
    ctx.obj = NetworkSettings(
        cluster=cluster,
        namespace=derived.namespace,
        service_name=derived.service_name,
        port=derived.port,
        ingress_class=derived.ingress_class,
        auth_mode=derived.auth_mode,
        # helm and kubectl are handed this path as an argv value, and neither expands
        # a leading '~' the way a shell would; cluster configs write one.
        kubeconfig=os.path.expanduser(kubeconfig or derived.kubeconfig) or None,
        context=context or derived.context or None,
        traefik_namespace=traefik_namespace,
        traefik_release=traefik_release,
        cert_manager_namespace=cert_manager_namespace,
        cert_manager_release=cert_manager_release,
    )


@main.command("install")
@click.option(
    "--acme-email",
    default=None,
    help="Email for the Let's Encrypt HTTP-01 ClusterIssuers (required unless --skip-issuers).",
)
@click.option(
    "--allow-source",
    "allow_sources",
    multiple=True,
    default=FEDERATION_ALLOW_SOURCES,
    show_default=True,
    help="IP or CIDR permitted to reach the federation route (repeatable). Defaults to every "
    "marin-side controller egress; passing this replaces the whole allowlist rather than adding to it.",
)
@click.option("--host", default="", help="Ingress host [default: iris-cw-<cluster>.oa.dev].")
@click.option("--tls-secret", default="", help="TLS secret [default: iris-controller-fed-tls].")
@click.option(
    "--cluster-issuer",
    default="",
    help="cert-manager ClusterIssuer for the cert [default: letsencrypt-http01-staging].",
)
@click.option(
    "--xff-depth",
    default=0,
    show_default=True,
    help="X-Forwarded-For depth for the ipAllowList (0 = match the direct peer; >0 if the LB SNATs).",
)
@click.option("--traefik-version", default=None, help="Pin the Traefik chart version (default: latest).")
@click.option("--cert-manager-version", default=None, help="Pin the cert-manager chart version (default: latest).")
@click.option("--skip-traefik", is_flag=True, help="Do not install Traefik (already present).")
@click.option("--skip-cert-manager", is_flag=True, help="Do not install cert-manager (already present).")
@click.option("--skip-issuers", is_flag=True, help="Do not create the HTTP-01 ClusterIssuers (already present).")
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
@click.pass_obj
def install_cmd(
    settings: NetworkSettings,
    acme_email: str | None,
    allow_sources: tuple[str, ...],
    host: str,
    tls_secret: str,
    cluster_issuer: str,
    xff_depth: int,
    traefik_version: str | None,
    cert_manager_version: str | None,
    skip_traefik: bool,
    skip_cert_manager: bool,
    skip_issuers: bool,
    apply: bool,
) -> None:
    """Install the whole CoreWeave network stack (dry-run without --apply)."""
    source_ranges = [_normalize_source(value) for value in allow_sources]
    install(
        settings,
        host=host or default_federation_host(settings.cluster),
        source_ranges=source_ranges,
        tls_secret=tls_secret or DEFAULT_TLS_SECRET,
        cluster_issuer=cluster_issuer or DEFAULT_CLUSTER_ISSUER,
        xff_depth=xff_depth,
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
def uninstall_cmd(settings: NetworkSettings, apply: bool) -> None:
    """Tear the whole network stack back down and verify nothing remains."""
    uninstall(settings, apply=apply)


if __name__ == "__main__":
    main()
