#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a CoreWeave controller's RPC surface to the GCP marin controller, IP-locked.

The GCP ``marin`` controller federates whole jobs to the CoreWeave controllers by
dialing their RPC surface directly (the pull model — CoreWeave must be reachable
*inbound*). CoreWeave has no user surface of its own: end users reach Iris only
through ``iris.oa.dev`` (IAP), and marin federates outward, so the only external
caller of a CoreWeave controller is the marin controller. That makes the auth
surface two factors, both required and neither alone sufficient:

  1. **IP allowlist** (this script) — a Traefik ``ipAllowList`` Middleware admits
     only the marin controller's egress IP. This is the *network* factor.
  2. **The controller's own auth** (identity factor) — the method-scoped
     ``aud="federation"`` verifier gates the handoff RPCs, and the general auth
     chain gates everything else. The ingress does NOT do method policy; the
     controller does.

Because the controller is the identity gate, it MUST be enforcing. A permissive
(null-auth) controller behind only an IP lock hands anonymous admin over its
entire control plane to anything arriving from the allowlisted IP — so this script
refuses to treat such a target as safe and warns loudly. ``cw-us-east-02a`` is
enforcing (``trusted_cidrs``); ``cw-rno2a`` must gain ``trusted_cidrs`` too (see
``docs/coreweave.md``).

The route reuses the controller's ``iris-cw-*.oa.dev`` host rather than a dedicated
name. Where the cluster already publishes ``/proxy`` (``cw-us-east-02a``), that
world-open route keeps its own more-specific Traefik router with no allowlist,
while this Ingress adds an allowlisted catch-all router for the rest of the surface
and reuses the existing TLS cert. Where there is no ``/proxy`` host
(``cw-rno2a``), this Ingress publishes the whole controller host, allowlisted, and
cert-manager issues its cert.

This is intentionally a standalone operator step, NOT part of ``start_controller``:
an Ingress + Middleware are independent objects, so applying them does not restart
the controller. That lets the networking land before the config change and
controller restart that turn on enforcement. See ``docs/coreweave.md``.

Prerequisites (install once per cluster, before this script):
  * Traefik + cert-manager + the HTTP-01 ClusterIssuers —
    ``install_traefik_proxy.py``. This script reuses them; it installs nothing.

Usage:
    # Dry-run (default): prints the manifests and the pre-flight findings.
    uv run lib/iris/scripts/install_federation_ingress.py --cluster cw-rno2a \\
        install --allow-source 203.0.113.7
    # Apply once the host CNAMEs to the Traefik LoadBalancer FQDN:
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

_INGRESS_NAME = "iris-federation"
_MIDDLEWARE_NAME = "iris-federation-ipallowlist"
_MIDDLEWARE_CRD = "middlewares.traefik.io"

# TLS secret for the ingress when the cluster has no existing controller cert to
# reuse (e.g. cw-rno2a, which publishes no /proxy host today).
DEFAULT_TLS_SECRET = "iris-controller-fed-tls"
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
    proxy_host: str  # the cluster's existing /proxy host, if any (host + cert to reuse)
    proxy_tls_secret: str  # the cluster's existing /proxy TLS secret, if any
    kubeconfig: str | None
    context: str | None
    auth_mode: str  # "null-auth" (permissive) | "iap" | "cidr"


def _auth_mode(config) -> str:
    """The controller's request-auth mode.

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
        proxy_host=cw.public_proxy_host,
        proxy_tls_secret=cw.tls_secret,
        kubeconfig=kubeconfig_override or (platform.kubeconfig_path if platform else "") or None,
        context=context_override or (platform.kube_context if platform else "") or None,
        auth_mode=_auth_mode(config),
    )


def _default_host(settings: FederationIngressSettings) -> str:
    """Reuse the cluster's /proxy host, else ``iris-cw-<cluster>.oa.dev``."""
    if settings.proxy_host:
        return settings.proxy_host
    short = settings.cluster[len("cw-") :] if settings.cluster.startswith("cw-") else settings.cluster
    return f"iris-cw-{short}.oa.dev"


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
    testing a refused request from a non-allowlisted host.
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
    """Ingress routing the controller host to the RPC surface, IP-locked by the Middleware.

    A single catch-all path attaches the ``ipAllowList`` Middleware via the
    router-middlewares annotation (``<ns>-<name>@kubernetescrd``). Where the
    cluster also publishes ``/proxy``, that route keeps its own more-specific
    Traefik router (Traefik prefers the longer path match), so ``/proxy`` stays
    world-open while everything else is allowlisted here.

    ``cluster_issuer`` empty means the TLS cert is managed elsewhere (the ``/proxy``
    Ingress) and this route only references ``tls_secret``; set it to have
    cert-manager issue the cert here. cert-manager's HTTP-01 solver runs on its own
    unrestricted Ingress, so the allowlist does not block ACME validation.
    """
    annotations = {
        "traefik.ingress.kubernetes.io/router.middlewares": f"{namespace}-{_MIDDLEWARE_NAME}@kubernetescrd",
    }
    if cluster_issuer:
        annotations["cert-manager.io/cluster-issuer"] = cluster_issuer
    spec: dict = {
        "ingressClassName": ingress_class,
        "rules": [
            {
                "host": host,
                "http": {
                    "paths": [
                        {
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {"service": {"name": service_name, "port": {"number": port}}},
                        }
                    ]
                },
            }
        ],
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
    """Warn when the target controller admits unauthenticated callers.

    With the ingress exposing the controller behind only an IP allowlist, a
    permissive controller hands anonymous admin over its whole control plane to
    anything from the allowlisted IP — so the controller must be enforcing before
    this ingress is a real boundary.
    """
    if settings.auth_mode != "null-auth":
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


def _check_prerequisites(settings: FederationIngressSettings, kflags: list[str]) -> None:
    """Fail fast without the Middleware CRD; warn on a missing IngressClass.

    The Middleware CRD is load-bearing: without it the ``ipAllowList`` cannot be
    applied and the route would have no network gate, so this fails rather than
    ship an ungated ingress. A missing IngressClass only delays serving (the
    Ingress waits for Traefik), so it is a warning.
    """
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
    reusing_cert = bool(cluster_issuer) is False and host == settings.proxy_host
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
    if reusing_cert:
        click.echo(f"  tls secret:   {tls_secret}  (reusing the existing /proxy cert on this host)")
    else:
        click.echo(f"  tls secret:   {tls_secret}  (cert-manager issues via {cluster_issuer})")
    if settings.proxy_host == host:
        click.echo("  note:         /proxy keeps its own world-open router; this allowlists the rest.")
    click.secho("==> Manifests:", fg="blue", bold=True)
    click.echo(yaml.safe_dump_all([middleware, ingress], default_flow_style=False, sort_keys=False))

    _warn_if_permissive(settings)

    if not apply:
        click.secho("\nwarn: dry run — nothing applied. Re-run with --apply to install.", fg="yellow", err=True)
        return

    _check_prerequisites(settings, kflags)
    click.secho("==> Applying the federation ingress", fg="blue", bold=True)
    kubectl_apply_docs([middleware, ingress], kflags)
    _print_next_steps(settings, host=host, cluster_issuer=cluster_issuer)


def _print_next_steps(settings: FederationIngressSettings, *, host: str, cluster_issuer: str) -> None:
    kflags = kubectl_flags(settings.kubeconfig, settings.context)
    click.secho("==> Done. To finish wiring the federation route:", fg="green", bold=True)
    if host != settings.proxy_host:
        click.secho(
            f"  1) CNAME {host} at the Traefik LoadBalancer FQDN (same target as /proxy):", fg="green", bold=True
        )
        click.echo(
            "       kubectl get svc traefik -n traefik "
            "-o=jsonpath='{.status.conditions[?(@.type==\"ExternalRecords\")].message}'"
        )
    else:
        click.secho(
            f"  1) {host} already resolves (the existing /proxy host) — no new DNS needed.", fg="green", bold=True
        )
    click.secho("  2) Verify from the marin controller VM:", fg="green", bold=True)
    click.echo("       - ListBackends WITH the federation JWT from the allowlisted egress IP  -> succeeds")
    click.echo("       - the same call WITHOUT the JWT (controller enforcing)                 -> UNAUTHENTICATED")
    click.echo("       - the same call from a non-allowlisted IP                              -> refused (403)")
    if cluster_issuer and "staging" in cluster_issuer:
        click.secho(
            "  3) Staging issuer in use — once the cert validates, re-run with "
            "--cluster-issuer letsencrypt-http01-prod --apply.",
            fg="green",
            bold=True,
        )
    if settings.auth_mode == "null-auth":
        click.secho(
            "  NOTE: this controller is still permissive — add auth.trusted_cidrs and restart so an "
            "off-cluster request must present a bearer; the IP allowlist alone is not an identity gate.",
            fg="yellow",
        )
    click.echo(f"  (inspect: kubectl get ingress {_INGRESS_NAME} -n {settings.namespace} {' '.join(kflags)} -o wide)")


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
    """IP-restricted ingress publishing a CoreWeave controller's RPC surface for federation."""
    ctx.obj = _derive_from_cluster(cluster, kubeconfig, context or None)


@main.command("install")
@click.option(
    "--allow-source",
    "allow_sources",
    multiple=True,
    required=True,
    help="IP or CIDR permitted to reach the route (repeatable). The marin egress IP.",
)
@click.option(
    "--host", default="", help="Ingress host [default: the cluster's /proxy host, else iris-cw-<cluster>.oa.dev]."
)
@click.option(
    "--tls-secret", default="", help="TLS secret [default: reuse the cluster's /proxy secret, else a new one]."
)
@click.option(
    "--cluster-issuer",
    default="",
    help="cert-manager ClusterIssuer to issue the cert [default: none when reusing an existing cert, "
    "else letsencrypt-http01-staging].",
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
    resolved_host = host or _default_host(settings)
    # Reuse the controller's existing /proxy cert when publishing on that same
    # host; otherwise cert-manager issues a fresh cert for this route.
    reuse_existing_cert = resolved_host == settings.proxy_host and bool(settings.proxy_tls_secret)
    resolved_tls_secret = tls_secret or (settings.proxy_tls_secret if reuse_existing_cert else DEFAULT_TLS_SECRET)
    resolved_issuer = cluster_issuer or ("" if reuse_existing_cert else DEFAULT_CLUSTER_ISSUER)
    install(
        settings,
        host=resolved_host,
        source_ranges=source_ranges,
        tls_secret=resolved_tls_secret,
        cluster_issuer=resolved_issuer,
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
