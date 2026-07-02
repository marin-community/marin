#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install the CoreWeave (CKS) cluster-wide prerequisites for a public /proxy ingress.

CKS ships no ingress controller and no TLS issuer, so the controller's ``/proxy``
Ingress (created per-cluster by ``K8sControllerProvider.start_controller``) needs
two cluster-wide, install-once components in place first:

  * **Traefik** — CoreWeave's blessed ingress controller (``coreweave/traefik``).
    Installed as a ``Service type=LoadBalancer`` that CKS fronts with a stable
    wildcard ``*.<ORG-ID>-<CLUSTER>.coreweave.app`` DNS name (the chart sets the
    ``service.beta.kubernetes.io/external-hostname: '*'`` annotation for you).
    Registers IngressClass ``traefik``.
  * **cert-manager** — issues the TLS cert Traefik terminates with
    (``coreweave/cert-manager``).

CoreWeave's *bundled* Let's Encrypt issuers validate via DNS-01 against
``acme.coreweave.com`` and therefore only cover ``*.coreweave.app``. To serve a
**custom** host (e.g. ``iris-cw.oa.dev`` CNAME'd to the coreweave.app FQDN) this
script instead installs **HTTP-01** ClusterIssuers validated through Traefik,
which work for any host that already resolves to the Traefik LoadBalancer.

This mirrors ``install_kueue.py``: the heavyweight cluster-wide controllers live
here (operator-run, one-time); the per-cluster objects (the ``/proxy`` Ingress)
are reconciled by the Iris controller at start. This is the operator's job, not
the controller Pod's — the controller has no authority to install CRDs.

SAFE BY DEFAULT: prints the plan (helm commands + ClusterIssuer manifests) and
stops. Pass ``--apply`` to mutate the cluster.

After it runs:
  1. Read the Traefik LoadBalancer FQDN:
       kubectl get svc traefik -n traefik \\
         -o=jsonpath='{.status.conditions[?(@.type=="ExternalRecords")].message}'
  2. CNAME your host at your DNS provider (e.g. Namecheap):
       iris-cw.oa.dev  CNAME  <that>.coreweave.app
  3. Point the cluster config's controller.coreweave block at it:
       public_proxy_host: iris-cw.oa.dev
       ingress_class: traefik
       tls_secret: iris-controller-proxy-tls
       cluster_issuer: letsencrypt-http01-prod   # use -staging first to avoid LE rate limits
     start_controller then creates the /proxy Ingress and cert-manager issues the cert.

Usage:
    uv run lib/iris/scripts/install_traefik_proxy.py --acme-email you@oa.dev            # dry run
    uv run lib/iris/scripts/install_traefik_proxy.py --acme-email you@oa.dev --apply
"""

import subprocess
import time

import click
import yaml

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

    _print_next_steps(traefik_namespace, traefik_release, ingress_class, skip_issuers)


def _helm_install(chart: str, release: str, namespace: str, version: str | None, hflags: list[str]) -> None:
    cmd = ["helm", *hflags, "upgrade", "--install", release, chart, "-n", namespace, "--create-namespace", "--wait"]
    if version:
        cmd += ["--version", version]
    result = run(cmd)
    if result.returncode != 0:
        raise click.ClickException(f"helm install of {chart} failed")


def _print_next_steps(traefik_namespace: str, traefik_release: str, ingress_class: str, skip_issuers: bool) -> None:
    issuer = ISSUER_NAMES["prod"]
    click.secho("==> Done. Finish wiring the public /proxy route:", fg="green", bold=True)
    click.echo(
        "  1. Read the Traefik LoadBalancer's stable FQDN:\n"
        f"       kubectl get svc {traefik_release} -n {traefik_namespace} "
        "-o=jsonpath='{.status.conditions[?(@.type==\"ExternalRecords\")].message}'"
    )
    click.echo("  2. CNAME your host to it at your DNS provider (Namecheap): iris-cw.oa.dev CNAME <that>.coreweave.app")
    click.echo(
        "  3. Set the cluster config's controller.coreweave block, then (re)start the controller:\n"
        "       public_proxy_host: iris-cw.oa.dev\n"
        f"       ingress_class: {ingress_class}\n"
        "       tls_secret: iris-controller-proxy-tls\n"
        + ("" if skip_issuers else f"       cluster_issuer: {issuer}   # use letsencrypt-http01-staging first\n")
    )
    click.secho(
        "  HTTP-01 issuance needs the CNAME live first (Let's Encrypt fetches http://<host>/.well-known/...). "
        "Validate with letsencrypt-http01-staging before switching to prod.",
        fg="yellow",
    )


@click.command()
@click.option(
    "--acme-email",
    default=None,
    help="Email for the Let's Encrypt HTTP-01 ClusterIssuers (required unless --skip-issuers).",
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
def main(
    acme_email: str | None,
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
    install(
        acme_email=acme_email,
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
