#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand up the CoreWeave federation ingress: an IP-restricted, JWT-gated RPC route.

The GCP ``marin`` controller federates whole jobs to the CoreWeave controllers by
dialing their RPC surface directly (the pull model — CW must be reachable inbound).
This script publishes ONLY the federation RPC subset on a dedicated host, distinct
from the ``/proxy`` route ``install_traefik_proxy.py`` sets up:

  * A Traefik ``Middleware`` (``ipAllowList``) that admits only the marin
    controller's egress IP — the network gate.
  * An ``Ingress`` that routes just the six federation methods
    (``LaunchJob``, ``TerminateJob``, ``FederationSync``, ``ListBackends``,
    ``ProfileTask``, ``ExecInContainer``) to the controller Service, TLS
    terminated in-cluster by cert-manager. The rest of the RPC surface
    (job submission for arbitrary users, budgets, scheduler state, the
    dashboard) stays ClusterIP-internal — it is never routed in.

Two independent gates, both required (neither alone suffices):
  1. **IP allowlist** (this ingress) — only the marin egress IP reaches the route.
  2. **Federation JWT** (the controller) — the method-scoped ``aud="federation"``
     verifier rejects an unauthenticated request. On a null-auth CoreWeave
     controller (``cw-rno2a`` today) the base RPC chain is PERMISSIVE — it admits
     any caller as the anonymous admin — so the federation verifier must be
     activated in the controller's ``auth`` config (``federation_peers``) or the
     IP allowlist becomes the sole gate. This script warns loudly when the target
     controller is still permissive.

This is intentionally a standalone operator step, NOT part of ``start_controller``
(unlike the ``/proxy`` Ingress): an Ingress + Middleware are independent objects,
so applying them does not restart the controller. That lets the networking land
(rollout P2) before the config + controller restart that turns on enforcement
(rollout P3). See ``.agents/projects/iris_federation/rollout.md`` (WS-5) and
``docs/coreweave.md``.

Prerequisites (install once per cluster, before this script):
  * Traefik + cert-manager + the HTTP-01 ClusterIssuers —
    ``install_traefik_proxy.py``. This script reuses them; it installs nothing.

Usage:
    # Dry-run (default): prints the manifests and the pre-flight findings.
    uv run lib/iris/scripts/install_federation_ingress.py --cluster cw-rno2a \\
        install --allow-source 203.0.113.7
    # Apply for real once DNS for the host CNAMEs to the Traefik LoadBalancer:
    uv run lib/iris/scripts/install_federation_ingress.py --cluster cw-rno2a \\
        install --allow-source 203.0.113.7 --cluster-issuer letsencrypt-http01-prod --apply
    # Tear the federation ingress back down (leaves Traefik/cert-manager alone):
    uv run lib/iris/scripts/install_federation_ingress.py --cluster cw-rno2a uninstall --apply
"""

import ipaddress
import subprocess
from typing import NamedTuple

import click
import yaml
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.cluster.config import load_config
from rigging.config_discovery import resolve_cluster_config

# The federation RPC subset (rollout WS-4.4): the only methods the ingress exposes.
# ConnectRPC maps each to POST /{package}.{Service}/{Method}; the whole service
# shares one flat prefix, so the ingress matches these paths EXACTLY (never the
# bare prefix, which would expose the entire control plane).
_RPC_SERVICE_PATH = "/iris.cluster.ControllerService"
FEDERATION_RPC_METHODS = (
    "LaunchJob",  # federation handoff (a plain LaunchJob is gated by the JWT verifier)
    "TerminateJob",  # routed cancel of a handed-off job
    "FederationSync",  # delta-sync mirror-back pull
    "ListBackends",  # capability heartbeat
    "ProfileTask",  # proxied profiling
    "ExecInContainer",  # proxied exec
)

_INGRESS_NAME = "iris-federation"
_MIDDLEWARE_NAME = "iris-federation-ipallowlist"
_MIDDLEWARE_CRD = "middlewares.traefik.io"

DEFAULT_TLS_SECRET = "iris-federation-tls"
# Staging first to avoid Let's Encrypt rate limits while DNS/allowlist are shaken
# out; flip to letsencrypt-http01-prod once the staging cert validates.
DEFAULT_CLUSTER_ISSUER = "letsencrypt-http01-staging"
DEFAULT_INGRESS_CLASS = "traefik"


# --------------------------------------------------------------------------
# Thin I/O helpers (subprocess via arg lists — never shell=True).
# --------------------------------------------------------------------------
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
        raise click.ClickException("kubectl apply of the federation ingress failed")


def resource_present(kind: str, name: str, kflags: list[str]) -> bool:
    """True if ``kind/name`` exists (quietly; used for pre-flight and teardown)."""
    result = subprocess.run(
        ["kubectl", *kflags, "get", kind, name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


# --------------------------------------------------------------------------
# Cluster settings, resolved from the Iris config
# --------------------------------------------------------------------------
class FederationIngressSettings(NamedTuple):
    """Target cluster and controller coordinates for the federation ingress."""

    cluster: str
    namespace: str
    service_name: str
    port: int
    ingress_class: str
    kubeconfig: str | None
    context: str | None
    auth_mode: str  # "null-auth" (permissive) | "iap" | "cidr"


def _auth_mode(config) -> str:
    """The controller's request-auth mode (mirrors controller/auth.py:436).

    ``null-auth`` means the RPC surface is PERMISSIVE: no login-provider arm and
    no ``trusted_cidrs``, so every request is admitted as the anonymous admin.
    """
    auth = config.auth
    if auth is None or (auth.provider_kind() is None and not auth.trusted_cidrs):
        return "null-auth"
    return auth.provider_kind() or "cidr"


def _derive_from_cluster(name: str, kubeconfig_override: str, context_override: str | None) -> FederationIngressSettings:
    """Read a named Iris cluster config and return its federation-ingress settings.

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
    namespace = (platform.namespace if platform else "") or "iris"
    return FederationIngressSettings(
        cluster=name,
        namespace=namespace,
        service_name=cw.service_name or "iris-controller-svc",
        port=cw.port or 10000,
        ingress_class=cw.ingress_class or DEFAULT_INGRESS_CLASS,
        kubeconfig=kubeconfig_override or (platform.kubeconfig_path if platform else "") or None,
        context=context_override or (platform.kube_context if platform else "") or None,
        auth_mode=_auth_mode(config),
    )


def _default_host(cluster: str) -> str:
    """``cw-rno2a`` -> ``iris-fed-rno2a.oa.dev`` (rollout WS-5.1 hostnames)."""
    short = cluster[len("cw-") :] if cluster.startswith("cw-") else cluster
    return f"iris-fed-{short}.oa.dev"


def _normalize_source(value: str) -> str:
    """Validate one allowlist entry and return it in CIDR form (bare IP -> /32,/128)."""
    try:
        if "/" in value:
            ipaddress.ip_network(value, strict=False)
            return value
        ip = ipaddress.ip_address(value)
    except ValueError as exc:
        raise click.BadParameter(f"{value!r} is not a valid IP or CIDR: {exc}", param_hint="--allow-source") from exc
    return f"{value}/{32 if ip.version == 4 else 128}"


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------
def _build_ipallowlist_middleware(*, namespace: str, source_ranges: list[str], xff_depth: int) -> dict:
    """A Traefik ``ipAllowList`` Middleware admitting only ``source_ranges``.

    By default Traefik matches the client's direct transport peer (``RemoteAddr``).
    If the CoreWeave LoadBalancer SNATs (so Traefik sees the LB, not the real
    client), set ``xff_depth`` to the number of trusted proxy hops and Traefik
    reads the client IP from ``X-Forwarded-For`` instead. Verify which applies by
    testing a refused request from a non-allowlisted host (rollout P2).
    """
    ip_allow_list: dict = {"sourceRange": source_ranges}
    if xff_depth > 0:
        ip_allow_list["ipStrategy"] = {"depth": xff_depth}
    return {
        "apiVersion": "traefik.io/v1alpha1",
        "kind": "Middleware",
        "metadata": {"name": _MIDDLEWARE_NAME, "namespace": namespace},
        "spec": {"ipAllowList": ip_allow_list},
    }


def _build_federation_ingress(
    *,
    namespace: str,
    service_name: str,
    port: int,
    host: str,
    ingress_class: str,
    tls_secret: str,
    cluster_issuer: str,
) -> dict:
    """Ingress routing ONLY the federation RPC methods to the controller Service.

    Each method is an exact-path rule, so the surface is exactly the six
    federation methods and nothing else. The Traefik ``ipAllowList`` Middleware is
    attached via the router-middlewares annotation (``<ns>-<name>@kubernetescrd``);
    cert-manager auto-issues the TLS cert into ``tls_secret`` from ``cluster_issuer``.
    cert-manager's HTTP-01 solver runs on its own unrestricted Ingress, so the
    allowlist does not block ACME validation.
    """
    annotations = {
        "traefik.ingress.kubernetes.io/router.middlewares": f"{namespace}-{_MIDDLEWARE_NAME}@kubernetescrd",
    }
    if cluster_issuer:
        annotations["cert-manager.io/cluster-issuer"] = cluster_issuer
    paths = [
        {
            "path": f"{_RPC_SERVICE_PATH}/{method}",
            "pathType": "Exact",
            "backend": {"service": {"name": service_name, "port": {"number": port}}},
        }
        for method in FEDERATION_RPC_METHODS
    ]
    spec: dict = {
        "ingressClassName": ingress_class,
        "rules": [{"host": host, "http": {"paths": paths}}],
    }
    if tls_secret:
        spec["tls"] = [{"hosts": [host], "secretName": tls_secret}]
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {"name": _INGRESS_NAME, "namespace": namespace, "annotations": annotations},
        "spec": spec,
    }


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
def _warn_if_permissive(settings: FederationIngressSettings) -> None:
    """Warn when the target controller admits unauthenticated federation calls."""
    if settings.auth_mode != "null-auth":
        return
    click.secho(
        f"\nwarn: {settings.cluster} runs NULL-AUTH (permissive) — its RPC surface admits any caller\n"
        "      as the anonymous admin. Until the method-scoped federation verifier is activated\n"
        "      (auth.federation_peers in the controller config + a controller restart, rollout P3),\n"
        "      the IP allowlist below is the ONLY gate on the federation route. Both the IP\n"
        "      allowlist AND the federation JWT are required (rollout WS-5.4); do not treat this\n"
        "      ingress as enforcing until the controller is no longer permissive.",
        fg="yellow",
        err=True,
    )


def _check_prerequisites(settings: FederationIngressSettings, kflags: list[str], *, apply: bool) -> None:
    """Fail fast without the Middleware CRD; warn on a missing IngressClass.

    The Middleware CRD is load-bearing: without it the ``ipAllowList`` cannot be
    applied and the route would have no network gate, so this fails rather than
    ship an ungated ingress. A missing IngressClass only delays serving (the
    Ingress waits for Traefik), so it is a warning.
    """
    if not apply:
        return
    if not resource_present("crd", _MIDDLEWARE_CRD, kflags):
        raise click.ClickException(
            f"Traefik Middleware CRD {_MIDDLEWARE_CRD} not found — the ipAllowList cannot be applied.\n"
            "Install Traefik first: install_traefik_proxy.py --cluster "
            f"{settings.cluster} install --apply"
        )
    if not resource_present("ingressclass", settings.ingress_class, kflags):
        click.secho(
            f"warn: IngressClass {settings.ingress_class!r} not found — the federation Ingress will stay "
            "pending (no external address) until Traefik is installed.",
            fg="yellow",
            err=True,
        )


# --------------------------------------------------------------------------
# Install / uninstall
# --------------------------------------------------------------------------
def install(
    settings: FederationIngressSettings,
    *,
    host: str,
    source_ranges: list[str],
    tls_secret: str,
    cluster_issuer: str,
    xff_depth: int,
    apply: bool,
) -> None:
    kflags = kubectl_flags(settings.kubeconfig, settings.context)
    middleware = _build_ipallowlist_middleware(
        namespace=settings.namespace, source_ranges=source_ranges, xff_depth=xff_depth
    )
    ingress = _build_federation_ingress(
        namespace=settings.namespace,
        service_name=settings.service_name,
        port=settings.port,
        host=host,
        ingress_class=settings.ingress_class,
        tls_secret=tls_secret,
        cluster_issuer=cluster_issuer,
    )

    click.secho(f"==> Plan (federation ingress for {settings.cluster}):", fg="blue", bold=True)
    click.echo(f"  host:         {host}  (CNAME -> the Traefik LoadBalancer FQDN)")
    click.echo(f"  allowlist:    {', '.join(source_ranges)}")
    click.echo(f"  backend:      {settings.service_name}:{settings.port} (namespace {settings.namespace})")
    click.echo(f"  methods:      {', '.join(FEDERATION_RPC_METHODS)}")
    click.echo(f"  tls secret:   {tls_secret}  (issuer {cluster_issuer or '<none — bring your own cert>'})")
    click.secho("==> Manifests:", fg="blue", bold=True)
    click.echo(yaml.safe_dump_all([middleware, ingress], default_flow_style=False, sort_keys=False))

    _warn_if_permissive(settings)

    if not apply:
        click.secho("\nwarn: dry run — nothing applied. Re-run with --apply to install.", fg="yellow", err=True)
        return

    _check_prerequisites(settings, kflags, apply=apply)
    click.secho("==> Applying the federation ingress", fg="blue", bold=True)
    kubectl_apply_docs([middleware, ingress], kflags)
    _print_next_steps(settings, host=host, cluster_issuer=cluster_issuer)


def _print_next_steps(settings: FederationIngressSettings, *, host: str, cluster_issuer: str) -> None:
    kflags = kubectl_flags(settings.kubeconfig, settings.context)
    click.secho("==> Done. To finish wiring the federation route:", fg="green", bold=True)
    click.secho(f"  1) CNAME {host} at the Traefik LoadBalancer FQDN (same target as /proxy):", fg="green", bold=True)
    click.echo(
        "       kubectl get svc traefik -n traefik "
        "-o=jsonpath='{.status.conditions[?(@.type==\"ExternalRecords\")].message}'"
    )
    click.secho("  2) Verify from the marin controller VM (rollout P2):", fg="green", bold=True)
    click.echo("       - ListBackends WITH the federation JWT from the allowlisted egress IP  -> succeeds")
    click.echo("       - the same call WITHOUT the JWT (once the controller is enforcing)      -> UNAUTHENTICATED")
    click.echo("       - the same call from a non-allowlisted IP                               -> refused (403)")
    if cluster_issuer and "staging" in cluster_issuer:
        click.secho(
            "  3) Staging issuer in use — once the cert validates, re-run with "
            "--cluster-issuer letsencrypt-http01-prod --apply.",
            fg="green",
            bold=True,
        )
    if settings.auth_mode == "null-auth":
        click.secho(
            "  NOTE: this controller is still permissive — the IP allowlist is the only gate until "
            "auth.federation_peers is configured and the controller restarted (rollout P3).",
            fg="yellow",
        )
    click.echo(
        f"  (inspect: kubectl get ingress {_INGRESS_NAME} -n {settings.namespace} " f"{' '.join(kflags)} -o wide)"
    )


def uninstall(settings: FederationIngressSettings, *, apply: bool) -> None:
    kflags = kubectl_flags(settings.kubeconfig, settings.context)
    click.secho(f"==> Plan (teardown of the federation ingress for {settings.cluster}):", fg="blue", bold=True)
    click.echo(f"  kubectl delete ingress {_INGRESS_NAME} -n {settings.namespace}")
    click.echo(f"  kubectl delete middleware.traefik.io {_MIDDLEWARE_NAME} -n {settings.namespace}")
    click.echo("  (Traefik, cert-manager, and the /proxy ingress are left untouched.)")

    if not apply:
        click.secho("\nwarn: dry run — nothing deleted. Re-run with --apply to uninstall.", fg="yellow", err=True)
        return

    click.secho("==> Deleting the federation ingress", fg="blue", bold=True)
    run(
        ["kubectl", *kflags, "delete", "ingress", _INGRESS_NAME, "-n", settings.namespace, "--ignore-not-found"],
        check=True,
    )
    run(
        [
            "kubectl",
            *kflags,
            "delete",
            "middleware.traefik.io",
            _MIDDLEWARE_NAME,
            "-n",
            settings.namespace,
            "--ignore-not-found",
        ],
        check=True,
    )
    click.secho("==> Teardown complete — the TLS secret (if any) is left for cert-manager to GC.", fg="green", bold=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
@click.group()
@click.option(
    "--cluster",
    required=True,
    help="Iris cluster name; supplies namespace, controller Service, kubeconfig, and kube context.",
)
@click.option("--kubeconfig", default="", help="kubeconfig to use [default: the cluster's].")
@click.option("--context", default="", help="kube context to target [default: the cluster's kube_context].")
@click.pass_context
def main(ctx: click.Context, cluster: str, kubeconfig: str, context: str) -> None:
    """IP-restricted, JWT-gated federation ingress for a CoreWeave controller."""
    ctx.obj = _derive_from_cluster(cluster, kubeconfig, context or None)


@main.command("install")
@click.option(
    "--allow-source",
    "allow_sources",
    multiple=True,
    required=True,
    help="IP or CIDR permitted to reach the federation route (repeatable). The marin egress IP.",
)
@click.option("--host", default="", help="Ingress host [default: iris-fed-<cluster>.oa.dev].")
@click.option("--tls-secret", default=DEFAULT_TLS_SECRET, show_default=True, help="Secret cert-manager issues into.")
@click.option(
    "--cluster-issuer",
    default=DEFAULT_CLUSTER_ISSUER,
    show_default=True,
    help="cert-manager ClusterIssuer (empty = bring your own cert in --tls-secret).",
)
@click.option(
    "--xff-depth",
    default=0,
    show_default=True,
    help="X-Forwarded-For depth for the ipAllowList (0 = match the direct peer; >0 if the LB SNATs).",
)
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
@click.pass_obj
def install_cmd(
    settings: FederationIngressSettings,
    allow_sources: tuple[str, ...],
    host: str,
    tls_secret: str,
    cluster_issuer: str,
    xff_depth: int,
    apply: bool,
) -> None:
    """Apply the federation Middleware + Ingress (dry-run without --apply)."""
    source_ranges = [_normalize_source(value) for value in allow_sources]
    install(
        settings,
        host=host or _default_host(settings.cluster),
        source_ranges=source_ranges,
        tls_secret=tls_secret,
        cluster_issuer=cluster_issuer,
        xff_depth=xff_depth,
        apply=apply,
    )


@main.command("uninstall")
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
@click.pass_obj
def uninstall_cmd(settings: FederationIngressSettings, apply: bool) -> None:
    """Delete the federation Ingress + Middleware (Traefik/cert-manager untouched)."""
    uninstall(settings, apply=apply)


if __name__ == "__main__":
    main()
