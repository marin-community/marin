#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install the CoreWeave (CKS) cluster-wide prerequisites for a public /proxy ingress.

CKS ships no ingress controller and no TLS issuer, so the controller's ``/proxy``
Ingress (created per-cluster by ``K8sControllerProvider.start_controller``) needs
two cluster-wide components installed first. This script installs them, once, from
the operator's kubeconfig (the controller's own ServiceAccount can't install CRDs):

  * **Traefik** (``coreweave/traefik``) — the ingress controller. Its chart exposes
    Traefik as a ``Service type=LoadBalancer`` and sets
    ``service.beta.kubernetes.io/external-hostname: '*'``, so CKS allocates it a
    wildcard ``*.<ORG-ID>-<CLUSTER>.coreweave.app`` DNS name. Registers IngressClass
    ``traefik``.
  * **cert-manager** (``coreweave/cert-manager``) — issues the TLS cert Traefik
    terminates with.

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

Usage:
    uv run lib/iris/scripts/install_traefik_proxy.py --cluster cw-us-east-02a \\
        --acme-email you@oa.dev                                              # dry run
    uv run lib/iris/scripts/install_traefik_proxy.py --cluster cw-us-east-02a \\
        --acme-email you@oa.dev --apply
"""

import subprocess
import time

import click
import yaml
from click.core import ParameterSource
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.cluster.config import load_config
from rigging.config_discovery import resolve_cluster_config

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
# Install core (importable; the click command calls it).
# --------------------------------------------------------------------------
def install(
    *,
    acme_email: str | None,
    host: str,
    from_cluster: str = "",
    ingress_class: str,
    traefik_namespace: str,
    traefik_release: str,
    cert_manager_namespace: str,
    cert_manager_release: str,
    traefik_version: str | None,
    cert_manager_version: str | None,
    skip_traefik: bool,
    skip_cert_manager: bool,
    skip_issuers: bool,
    kubeconfig: str | None,
    context: str | None,
    apply: bool,
) -> None:
    hflags = helm_flags(kubeconfig, context)
    kflags = kubectl_flags(kubeconfig, context)

    if not skip_issuers and not acme_email:
        raise click.ClickException(
            "--acme-email is required to create the HTTP-01 ClusterIssuers (or pass --skip-issuers)"
        )

    issuer_docs = (
        [_http01_issuer(env, acme_email, ingress_class) for env in ("staging", "prod")]
        if not skip_issuers and acme_email
        else []
    )

    # Assemble + print the plan; the only branch is the final apply.
    click.secho("==> Plan (CoreWeave /proxy ingress prerequisites):", fg="blue", bold=True)
    if not skip_traefik:
        click.echo(
            f"  helm upgrade --install {traefik_release} {TRAEFIK_CHART} -n {traefik_namespace} --create-namespace"
        )
    if not skip_cert_manager:
        click.echo(
            f"  helm upgrade --install {cert_manager_release} {CERT_MANAGER_CHART} "
            f"-n {cert_manager_namespace} --create-namespace"
        )
    if issuer_docs:
        click.secho("==> ClusterIssuers (HTTP-01 via Traefik):", fg="blue", bold=True)
        click.echo(yaml.safe_dump_all(issuer_docs, default_flow_style=False, sort_keys=False))

    if not apply:
        click.secho("\nwarn: dry run — nothing applied. Re-run with --apply to install.", fg="yellow", err=True)
        return

    # helm repo add/update touches only local helm config (no cluster mutation).
    click.secho(f"==> Adding/updating helm repo {CW_REPO_NAME} ({CW_REPO_URL})", fg="blue", bold=True)
    run(["helm", "repo", "add", CW_REPO_NAME, CW_REPO_URL], check=True, stdout=subprocess.DEVNULL)
    run(["helm", "repo", "update", CW_REPO_NAME], check=True, stdout=subprocess.DEVNULL)

    if not skip_traefik:
        click.secho(f"==> Installing Traefik ({TRAEFIK_CHART}) in namespace {traefik_namespace}", fg="blue", bold=True)
        _helm_install(TRAEFIK_CHART, traefik_release, traefik_namespace, traefik_version, hflags)

    if not skip_cert_manager:
        click.secho(
            f"==> Installing cert-manager ({CERT_MANAGER_CHART}) in namespace {cert_manager_namespace}",
            fg="blue",
            bold=True,
        )
        _helm_install(CERT_MANAGER_CHART, cert_manager_release, cert_manager_namespace, cert_manager_version, hflags)

    if issuer_docs:
        click.secho("==> Waiting for the cert-manager ClusterIssuer CRD, then applying issuers", fg="blue", bold=True)
        wait_for_crd(CLUSTERISSUER_CRD, kflags)
        kubectl_apply_docs(issuer_docs, kflags)

    _print_next_steps(
        host=host,
        from_cluster=from_cluster,
        ingress_class=ingress_class,
        traefik_namespace=traefik_namespace,
        traefik_release=traefik_release,
        skip_traefik=skip_traefik,
        skip_issuers=skip_issuers,
        kflags=kflags,
    )


def _helm_install(chart: str, release: str, namespace: str, version: str | None, hflags: list[str]) -> None:
    cmd = ["helm", *hflags, "upgrade", "--install", release, chart, "-n", namespace, "--create-namespace", "--wait"]
    if version:
        cmd += ["--version", version]
    result = run(cmd)
    if result.returncode != 0:
        raise click.ClickException(f"helm install of {chart} failed")


def _print_next_steps(
    *,
    host: str,
    from_cluster: str,
    ingress_class: str,
    traefik_namespace: str,
    traefik_release: str,
    skip_traefik: bool,
    skip_issuers: bool,
    kflags: list[str],
) -> None:
    """Print exactly what the operator must do next — the DNS record and config."""
    host_label = host or "<your-host>.oa.dev"
    click.secho("==> Done. Two steps to finish wiring the public /proxy route:", fg="green", bold=True)

    # 1. The concrete DNS record. Read the real Traefik FQDN so we can print it.
    click.secho("  1) Create this DNS record at your DNS provider (e.g. Namecheap):", fg="green", bold=True)
    fqdn = "" if skip_traefik else read_traefik_fqdn(traefik_release, traefik_namespace, kflags)
    if fqdn:
        click.echo(f"        {host_label}   CNAME   {fqdn}")
    else:
        click.secho("       (Traefik's FQDN isn't allocated yet — read it in a minute, then CNAME to it:)", fg="yellow")
        click.echo(
            f"       kubectl get svc {traefik_release} -n {traefik_namespace} "
            "-o=jsonpath='{.status.conditions[?(@.type==\"ExternalRecords\")].message}'"
        )
        click.echo(f"        {host_label}   CNAME   <that>{_COREWEAVE_APP}")

    # 2. The controller config. With --cluster and a host already set, the block
    # is already in the config — just restart. Otherwise print the block to add.
    if from_cluster and host:
        click.secho(
            f"  2) {from_cluster} already sets controller.coreweave (public_proxy_host={host}) — "
            "restart the controller so it creates the /proxy Ingress:",
            fg="green",
            bold=True,
        )
        click.echo(f"       iris --cluster {from_cluster} cluster restart")
    else:
        click.secho(
            "  2) Set the cluster config's controller.coreweave block, then (re)start the controller:",
            fg="green",
            bold=True,
        )
        click.echo(f"       public_proxy_host: {host or 'your-host.oa.dev'}")
        click.echo(f"       ingress_class: {ingress_class}")
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


def _derive_from_cluster(name: str) -> dict:
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
    return {
        "host": cw.public_proxy_host,
        "ingress_class": cw.ingress_class,
        "kubeconfig": platform.kubeconfig_path if platform else "",
    }


@click.command()
@click.option(
    "--cluster",
    default="",
    help="Iris cluster name; derives --host, --ingress-class, and --kubeconfig from its controller.coreweave block.",
)
@click.option(
    "--acme-email",
    default=None,
    help="Email for the Let's Encrypt HTTP-01 ClusterIssuers (required unless --skip-issuers).",
)
@click.option(
    "--host", default="", help="Public endpoint host (e.g. iris-cw.oa.dev); used to print the exact DNS + config."
)
@click.option(
    "--ingress-class", default="traefik", show_default=True, help="IngressClass the HTTP-01 solver routes through."
)
@click.option("--traefik-namespace", default="traefik", show_default=True)
@click.option("--traefik-release", default="traefik", show_default=True, help="helm release name for Traefik.")
@click.option("--cert-manager-namespace", default="cert-manager", show_default=True)
@click.option(
    "--cert-manager-release", default="cert-manager", show_default=True, help="helm release name for cert-manager."
)
@click.option("--traefik-version", default=None, help="Pin the Traefik chart version (default: latest).")
@click.option("--cert-manager-version", default=None, help="Pin the cert-manager chart version (default: latest).")
@click.option("--skip-traefik", is_flag=True, help="Do not install Traefik (already present).")
@click.option("--skip-cert-manager", is_flag=True, help="Do not install cert-manager (already present).")
@click.option("--skip-issuers", is_flag=True, help="Do not create the HTTP-01 ClusterIssuers.")
@click.option("--kubeconfig", default=None, help="kubeconfig to use (else $KUBECONFIG / ~/.kube/config).")
@click.option("--context", default=None, help="kube context to target.")
@click.option("--apply/--no-apply", default=False, help="Actually mutate the cluster (default: dry-run only).")
@click.pass_context
def main(
    ctx: click.Context,
    cluster: str,
    acme_email: str | None,
    host: str,
    ingress_class: str,
    traefik_namespace: str,
    traefik_release: str,
    cert_manager_namespace: str,
    cert_manager_release: str,
    traefik_version: str | None,
    cert_manager_version: str | None,
    skip_traefik: bool,
    skip_cert_manager: bool,
    skip_issuers: bool,
    kubeconfig: str | None,
    context: str | None,
    apply: bool,
) -> None:
    """Install Traefik + cert-manager + HTTP-01 issuers for the CoreWeave /proxy ingress."""
    if cluster:
        # Fill host / ingress-class / kubeconfig from the cluster config, but let
        # an explicitly-passed flag win over the derived value.
        derived = _derive_from_cluster(cluster)

        def pick(name: str, cli_value, cfg_value):
            explicit = ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE
            return cli_value if explicit else (cfg_value or cli_value)

        host = pick("host", host, derived["host"])
        ingress_class = pick("ingress_class", ingress_class, derived["ingress_class"])
        kubeconfig = pick("kubeconfig", kubeconfig, derived["kubeconfig"])

    install(
        acme_email=acme_email,
        host=host,
        from_cluster=cluster,
        ingress_class=ingress_class,
        traefik_namespace=traefik_namespace,
        traefik_release=traefik_release,
        cert_manager_namespace=cert_manager_namespace,
        cert_manager_release=cert_manager_release,
        traefik_version=traefik_version,
        cert_manager_version=cert_manager_version,
        skip_traefik=skip_traefik,
        skip_cert_manager=skip_cert_manager,
        skip_issuers=skip_issuers,
        kubeconfig=kubeconfig,
        context=context,
        apply=apply,
    )


if __name__ == "__main__":
    main()
