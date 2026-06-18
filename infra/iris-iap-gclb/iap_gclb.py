#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand up (idempotently) the external HTTPS Load Balancer + IAP stack that
fronts an Iris controller VM.

    client --HTTPS:443--> GCLB --(IAP)--> backend service --HTTP:10000--> controller VM

This supersedes the Cloud Run proxy in ``../iris-iap-proxy/`` (retired as this
rolls out): GCLB talks straight to the controller VM (no extra serverless hop
and no Cloud Run 300s request cap that would truncate long-poll requests).

Every resource is a single ``gcloud`` create guarded by an existence probe, so
the whole rollout — or any single stage — is safe to re-run. ``deploy`` runs all
stages in dependency order; the per-stage subcommands (``oauth``, ``address``,
``cert``, ``firewall``, ``backend``, ``frontend``, ``grant``) expose each one on
its own. ``status`` reports what exists and ``teardown`` deletes the stack.

One stack per cluster: the cluster name is both the resource-name prefix
(``iris-<cluster>-*``) and the GCE network tag / label used to find and firewall
the controller VM (``iris-<cluster>-controller``).

Usage:
    uv run infra/iris-iap-gclb/iap_gclb.py deploy marin \\
        --domain iris-marin.example.com \\
        --support-email you@example.com \\
        --member user:you@example.com
    uv run infra/iris-iap-gclb/iap_gclb.py status marin
    uv run infra/iris-iap-gclb/iap_gclb.py teardown marin
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
from collections.abc import Sequence

import click

logger = logging.getLogger("iap-gclb")

DEFAULT_PROJECT = "hai-gcp-models"
DEFAULT_ZONE = "us-central1-a"
CONTROLLER_PORT = 10000

# Google front-end / health-check / IAP source ranges that legitimately reach
# the controller port; a lower-priority rule denies everything else so nobody
# can bypass IAP by hitting the VM's IP directly.
GOOGLE_LB_RANGES = "130.211.0.0/22,35.191.0.0/16"
IAP_ACCESSOR_ROLE = "roles/iap.httpsResourceAccessor"
OAUTH_CLIENT_DISPLAY = "Iris GCLB IAP"


@dataclasses.dataclass(frozen=True)
class Stack:
    """Names of every GCLB+IAP resource for one cluster, derived from its name."""

    cluster: str
    project: str = DEFAULT_PROJECT
    zone: str = DEFAULT_ZONE
    domain: str | None = None

    @property
    def prefix(self) -> str:
        return f"iris-{self.cluster}"

    @property
    def controller_label(self) -> str:
        """GCE label (for discovery) and network tag (for the firewall)."""
        return f"{self.prefix}-controller"

    @property
    def neg(self) -> str:
        return f"{self.prefix}-neg"

    @property
    def health_check(self) -> str:
        return f"{self.prefix}-hc"

    @property
    def backend(self) -> str:
        return f"{self.prefix}-be"

    @property
    def url_map(self) -> str:
        return f"{self.prefix}-urlmap"

    @property
    def https_proxy(self) -> str:
        return f"{self.prefix}-https-proxy"

    @property
    def forwarding_rule(self) -> str:
        return f"{self.prefix}-fr"

    @property
    def address(self) -> str:
        return f"{self.prefix}-ip"

    @property
    def cert(self) -> str:
        return f"{self.prefix}-cert"

    @property
    def allow_firewall(self) -> str:
        return f"{self.prefix}-allow-lb"

    @property
    def deny_firewall(self) -> str:
        return f"{self.prefix}-deny-public-{CONTROLLER_PORT}"


# --------------------------------------------------------------------------- #
# gcloud helpers
# --------------------------------------------------------------------------- #


def _compute(stack: Stack, *args: str) -> list[str]:
    """Build a ``gcloud compute ... --project=<project>`` command."""
    return ["gcloud", "compute", *args, f"--project={stack.project}"]


def _iap(stack: Stack, *args: str) -> list[str]:
    """Build a ``gcloud iap ... --project=<project>`` command."""
    return ["gcloud", "iap", *args, f"--project={stack.project}"]


def _run(cmd: Sequence[str], *, dry_run: bool = False, check: bool = True, capture: bool = False):
    """Run a gcloud command, tracing it. With *dry_run*, log and skip."""
    logger.info("$ %s", " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(list(cmd), 0, stdout="", stderr="")
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def _exists(describe_cmd: Sequence[str]) -> bool:
    """True if a ``gcloud ... describe`` probe succeeds (resource present)."""
    result = subprocess.run([*describe_cmd, "--format=value(name)"], text=True, capture_output=True)
    return result.returncode == 0


def _ensure(label: str, exists: bool, create_cmd: Sequence[str], *, dry_run: bool) -> bool:
    """Create a resource if it is missing. Returns True if a create ran."""
    if exists:
        logger.info("✓ %s already exists", label)
        return False
    logger.info("→ creating %s", label)
    _run(create_cmd, dry_run=dry_run)
    return True


# --------------------------------------------------------------------------- #
# Stage: IAP OAuth brand + client
# --------------------------------------------------------------------------- #


def _list_brands(stack: Stack) -> list[dict]:
    result = _run(_iap(stack, "oauth-brands", "list", "--format=json"), capture=True)
    return json.loads(result.stdout or "[]")


def _list_oauth_clients(stack: Stack, brand: str) -> list[dict]:
    result = _run(_iap(stack, "oauth-clients", "list", brand, "--format=json"), capture=True)
    return json.loads(result.stdout or "[]")


def ensure_brand(stack: Stack, support_email: str | None, *, dry_run: bool) -> str:
    """Return the project's IAP OAuth brand resource name, creating it if absent.

    A project has at most one brand (the OAuth consent screen), so an existing
    brand is reused. Creation needs a support email owned by the caller.
    """
    brands = [] if dry_run else _list_brands(stack)
    if brands:
        logger.info("✓ IAP brand exists: %s", brands[0]["name"])
        return brands[0]["name"]
    if not support_email:
        raise click.ClickException("no IAP brand exists yet; pass --support-email to create one")
    _run(
        _iap(
            stack,
            "oauth-brands",
            "create",
            "--application_title=Iris Controller",
            f"--support_email={support_email}",
        ),
        dry_run=dry_run,
    )
    if dry_run:
        return "projects/<PROJECT_NUMBER>/brands/<BRAND_ID>"
    return _list_brands(stack)[0]["name"]


def ensure_oauth_client(stack: Stack, brand: str, *, dry_run: bool) -> tuple[str, str]:
    """Return ``(client_id, client_secret)`` for this cluster's IAP OAuth client.

    Reuses an existing client matched by display name (``describe``/``list``
    returns the secret for brand-owned clients), otherwise creates one.
    """
    display = f"{OAUTH_CLIENT_DISPLAY} ({stack.cluster})"
    clients = [] if dry_run else _list_oauth_clients(stack, brand)
    for client in clients:
        if client.get("displayName") == display:
            client_id = client["name"].split("/")[-1]
            logger.info("✓ IAP OAuth client exists: %s", client_id)
            return client_id, client["secret"]
    result = _run(
        _iap(stack, "oauth-clients", "create", brand, f"--display_name={display}", "--format=json"),
        dry_run=dry_run,
        capture=True,
    )
    if dry_run:
        return "<oauth-client-id>", "<oauth-client-secret>"
    created = json.loads(result.stdout)
    return created["name"].split("/")[-1], created["secret"]


def resolve_oauth(
    stack: Stack,
    client_id: str | None,
    client_secret: str | None,
    support_email: str | None,
    *,
    dry_run: bool,
) -> tuple[str, str]:
    """Use explicit credentials if given, otherwise find/create the OAuth client."""
    if client_id and client_secret:
        return client_id, client_secret
    if client_id or client_secret:
        raise click.ClickException("pass both --oauth-client-id and --oauth-client-secret, or neither")
    brand = ensure_brand(stack, support_email, dry_run=dry_run)
    return ensure_oauth_client(stack, brand, dry_run=dry_run)


# --------------------------------------------------------------------------- #
# Stage: static IP, managed cert, firewall
# --------------------------------------------------------------------------- #


def ensure_address(stack: Stack, *, dry_run: bool) -> str:
    """Reserve the global static IP if missing and return its address."""
    describe = _compute(stack, "addresses", "describe", stack.address, "--global")
    _ensure(
        f"static IP {stack.address}",
        _exists(describe),
        _compute(stack, "addresses", "create", stack.address, "--global"),
        dry_run=dry_run,
    )
    if dry_run:
        return "<reserved-ip>"
    result = _run([*describe, "--format=value(address)"], capture=True)
    return result.stdout.strip()


def ensure_cert(stack: Stack, *, dry_run: bool) -> None:
    """Create the Google-managed SSL certificate for the cluster domain.

    The cert stays PROVISIONING until the domain's DNS A record resolves to the
    reserved static IP.
    """
    if not stack.domain:
        raise click.ClickException("--domain is required to create the managed SSL certificate")
    _ensure(
        f"managed SSL cert {stack.cert} ({stack.domain})",
        _exists(_compute(stack, "ssl-certificates", "describe", stack.cert, "--global")),
        _compute(stack, "ssl-certificates", "create", stack.cert, "--global", f"--domains={stack.domain}"),
        dry_run=dry_run,
    )


def ensure_firewall(stack: Stack, *, dry_run: bool) -> None:
    """Allow the controller port only from Google LB ranges; deny all other ingress."""
    _ensure(
        f"firewall allow-LB {stack.allow_firewall}",
        _exists(_compute(stack, "firewall-rules", "describe", stack.allow_firewall)),
        _compute(
            stack,
            "firewall-rules",
            "create",
            stack.allow_firewall,
            "--network=default",
            "--direction=INGRESS",
            "--action=ALLOW",
            f"--rules=tcp:{CONTROLLER_PORT}",
            f"--source-ranges={GOOGLE_LB_RANGES}",
            f"--target-tags={stack.controller_label}",
            "--priority=900",
        ),
        dry_run=dry_run,
    )
    _ensure(
        f"firewall deny-public {stack.deny_firewall}",
        _exists(_compute(stack, "firewall-rules", "describe", stack.deny_firewall)),
        _compute(
            stack,
            "firewall-rules",
            "create",
            stack.deny_firewall,
            "--network=default",
            "--direction=INGRESS",
            "--action=DENY",
            f"--rules=tcp:{CONTROLLER_PORT}",
            "--source-ranges=0.0.0.0/0",
            f"--target-tags={stack.controller_label}",
            "--priority=1000",
        ),
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Stage: backend (NEG -> health check -> backend service -> IAP)
# --------------------------------------------------------------------------- #


def discover_controller_ip(stack: Stack) -> str:
    """Resolve the controller VM's internal IP from its GCE label."""
    result = _run(
        _compute(
            stack,
            "instances",
            "list",
            f"--filter=labels.{stack.controller_label}=true",
            "--format=value(networkInterfaces[0].networkIP)",
        ),
        capture=True,
    )
    ips = (result.stdout or "").split()
    if not ips:
        raise click.ClickException(
            f"no controller VM labelled {stack.controller_label}=true; pass --controller-ip explicitly"
        )
    if len(ips) > 1:
        raise click.ClickException(f"multiple VMs match {stack.controller_label}=true ({ips}); pass --controller-ip")
    return ips[0]


def _neg_has_endpoint(stack: Stack, ip: str) -> bool:
    result = _run(
        _compute(
            stack,
            "network-endpoint-groups",
            "list-network-endpoints",
            stack.neg,
            f"--zone={stack.zone}",
            "--format=value(ipAddress)",
        ),
        capture=True,
    )
    return ip in (result.stdout or "").split()


def _backend_has_neg(stack: Stack) -> bool:
    result = _run(
        _compute(
            stack,
            "backend-services",
            "describe",
            stack.backend,
            "--global",
            "--format=value(backends[].group)",
        ),
        capture=True,
    )
    return stack.neg in (result.stdout or "")


def ensure_backend(
    stack: Stack,
    client_id: str,
    client_secret: str,
    controller_ip: str,
    *,
    dry_run: bool,
) -> None:
    """Build the backend half: zonal NEG -> controller endpoint -> health check
    -> backend service -> NEG attachment -> IAP enablement."""
    _ensure(
        f"zonal NEG {stack.neg}",
        _exists(_compute(stack, "network-endpoint-groups", "describe", stack.neg, f"--zone={stack.zone}")),
        _compute(
            stack,
            "network-endpoint-groups",
            "create",
            stack.neg,
            f"--zone={stack.zone}",
            "--network=default",
            "--subnet=default",
            "--network-endpoint-type=GCE_VM_IP_PORT",
            f"--default-port={CONTROLLER_PORT}",
        ),
        dry_run=dry_run,
    )

    if dry_run or not _neg_has_endpoint(stack, controller_ip):
        logger.info("→ attaching controller endpoint %s:%d to %s", controller_ip, CONTROLLER_PORT, stack.neg)
        _run(
            _compute(
                stack,
                "network-endpoint-groups",
                "update",
                stack.neg,
                f"--zone={stack.zone}",
                f"--add-endpoint=ip={controller_ip},port={CONTROLLER_PORT}",
            ),
            dry_run=dry_run,
        )
    else:
        logger.info("✓ endpoint %s:%d already attached to %s", controller_ip, CONTROLLER_PORT, stack.neg)

    _ensure(
        f"health check {stack.health_check} (HTTP /health :{CONTROLLER_PORT})",
        _exists(_compute(stack, "health-checks", "describe", stack.health_check, "--global")),
        _compute(
            stack,
            "health-checks",
            "create",
            "http",
            stack.health_check,
            "--global",
            f"--port={CONTROLLER_PORT}",
            "--request-path=/health",
            "--check-interval=10s",
            "--timeout=5s",
            "--healthy-threshold=2",
            "--unhealthy-threshold=3",
        ),
        dry_run=dry_run,
    )

    _ensure(
        f"backend service {stack.backend}",
        _exists(_compute(stack, "backend-services", "describe", stack.backend, "--global")),
        _compute(
            stack,
            "backend-services",
            "create",
            stack.backend,
            "--global",
            "--protocol=HTTP",
            "--port-name=http",
            f"--health-checks={stack.health_check}",
            "--timeout=120s",
            "--load-balancing-scheme=EXTERNAL_MANAGED",
        ),
        dry_run=dry_run,
    )

    if dry_run or not _backend_has_neg(stack):
        logger.info("→ adding NEG %s to backend service %s", stack.neg, stack.backend)
        _run(
            _compute(
                stack,
                "backend-services",
                "add-backend",
                stack.backend,
                "--global",
                f"--network-endpoint-group={stack.neg}",
                f"--network-endpoint-group-zone={stack.zone}",
                "--balancing-mode=RATE",
                "--max-rate-per-endpoint=1000",
            ),
            dry_run=dry_run,
        )
    else:
        logger.info("✓ NEG %s already attached to backend %s", stack.neg, stack.backend)

    # Enabling IAP is an update, so it is idempotent; re-run to reconcile the
    # configured OAuth client.
    logger.info("→ enabling IAP on backend service %s", stack.backend)
    _run(
        _compute(
            stack,
            "backend-services",
            "update",
            stack.backend,
            "--global",
            f"--iap=enabled,oauth2-client-id={client_id},oauth2-client-secret={client_secret}",
        ),
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Stage: frontend (URL map -> HTTPS proxy -> forwarding rule)
# --------------------------------------------------------------------------- #


def ensure_frontend(stack: Stack, *, dry_run: bool) -> None:
    """Build the frontend half: URL map -> target HTTPS proxy -> :443 forwarding rule."""
    _ensure(
        f"URL map {stack.url_map}",
        _exists(_compute(stack, "url-maps", "describe", stack.url_map, "--global")),
        _compute(stack, "url-maps", "create", stack.url_map, "--global", f"--default-service={stack.backend}"),
        dry_run=dry_run,
    )
    _ensure(
        f"target HTTPS proxy {stack.https_proxy}",
        _exists(_compute(stack, "target-https-proxies", "describe", stack.https_proxy, "--global")),
        _compute(
            stack,
            "target-https-proxies",
            "create",
            stack.https_proxy,
            "--global",
            f"--url-map={stack.url_map}",
            f"--ssl-certificates={stack.cert}",
        ),
        dry_run=dry_run,
    )
    _ensure(
        f"forwarding rule {stack.forwarding_rule} ({stack.address}:443)",
        _exists(_compute(stack, "forwarding-rules", "describe", stack.forwarding_rule, "--global")),
        _compute(
            stack,
            "forwarding-rules",
            "create",
            stack.forwarding_rule,
            "--global",
            f"--address={stack.address}",
            f"--target-https-proxy={stack.https_proxy}",
            "--ports=443",
            "--load-balancing-scheme=EXTERNAL_MANAGED",
        ),
        dry_run=dry_run,
    )


def grant_access(stack: Stack, member: str, *, dry_run: bool) -> None:
    """Grant *member* IAP access (roles/iap.httpsResourceAccessor) on the backend."""
    logger.info("→ granting %s %s on %s", member, IAP_ACCESSOR_ROLE, stack.backend)
    _run(
        _iap(
            stack,
            "web",
            "add-iam-policy-binding",
            "--resource-type=backend-services",
            f"--service={stack.backend}",
            f"--member={member}",
            f"--role={IAP_ACCESSOR_ROLE}",
        ),
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _common_options(func):
    func = click.argument("cluster")(func)
    func = click.option("--project", default=DEFAULT_PROJECT, show_default=True, help="GCP project id")(func)
    func = click.option("--zone", default=DEFAULT_ZONE, show_default=True, help="Zone of the controller VM / NEG")(func)
    func = click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")(func)
    return func


@click.group(help=__doc__)
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG-level logging")
def cli(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


@cli.command()
@_common_options
@click.option("--domain", required=True, help="Domain whose DNS A record points at the reserved static IP")
@click.option("--controller-ip", help="Controller VM internal IP (default: discover from the GCE label)")
@click.option("--support-email", help="Support email for the IAP brand (only needed to create one)")
@click.option("--oauth-client-id", help="Existing IAP OAuth client id (default: find/create one)")
@click.option("--oauth-client-secret", help="Existing IAP OAuth client secret")
@click.option("--member", help="Principal to grant IAP access, e.g. user:you@example.com")
def deploy(
    cluster: str,
    project: str,
    zone: str,
    dry_run: bool,
    domain: str,
    controller_ip: str | None,
    support_email: str | None,
    oauth_client_id: str | None,
    oauth_client_secret: str | None,
    member: str | None,
) -> None:
    """Run every stage in dependency order to stand up the full stack."""
    stack = Stack(cluster=cluster, project=project, zone=zone, domain=domain)
    controller_ip = controller_ip or discover_controller_ip(stack)
    client_id, client_secret = resolve_oauth(stack, oauth_client_id, oauth_client_secret, support_email, dry_run=dry_run)
    reserved_ip = ensure_address(stack, dry_run=dry_run)
    ensure_cert(stack, dry_run=dry_run)
    ensure_firewall(stack, dry_run=dry_run)
    ensure_backend(stack, client_id, client_secret, controller_ip, dry_run=dry_run)
    ensure_frontend(stack, dry_run=dry_run)
    if member:
        grant_access(stack, member, dry_run=dry_run)

    click.echo()
    click.echo(f"Stack for cluster={cluster} reconciled.")
    click.echo(f"  Reserved IP    : {reserved_ip}")
    click.echo(f"  Domain         : {domain}  (ensure a DNS A record -> {reserved_ip})")
    click.echo(f"  URL            : https://{domain}")
    click.echo(f"  OAuth client   : {client_id}")
    click.echo()
    click.echo(f"The managed cert ({stack.cert}) provisions only after DNS resolves to the")
    click.echo("reserved IP; expect a few minutes to ACTIVE. Grant more users with:")
    click.echo(f"  uv run {sys.argv[0]} grant {cluster} --member user:NAME@example.com")


@cli.command()
@_common_options
@click.option("--support-email", help="Support email for the IAP brand (only needed to create one)")
def oauth(cluster: str, project: str, zone: str, dry_run: bool, support_email: str | None) -> None:
    """Ensure the IAP OAuth brand + client exist; print the client id/secret."""
    stack = Stack(cluster=cluster, project=project, zone=zone)
    brand = ensure_brand(stack, support_email, dry_run=dry_run)
    client_id, client_secret = ensure_oauth_client(stack, brand, dry_run=dry_run)
    click.echo(f"oauth-client-id     : {client_id}")
    click.echo(f"oauth-client-secret : {client_secret}")


@cli.command()
@_common_options
def address(cluster: str, project: str, zone: str, dry_run: bool) -> None:
    """Reserve the global static IP and print it (point the domain's A record here)."""
    stack = Stack(cluster=cluster, project=project, zone=zone)
    reserved_ip = ensure_address(stack, dry_run=dry_run)
    click.echo(f"reserved-ip : {reserved_ip}")


@cli.command()
@_common_options
@click.option("--domain", required=True, help="Domain for the managed SSL certificate")
def cert(cluster: str, project: str, zone: str, dry_run: bool, domain: str) -> None:
    """Create the Google-managed SSL certificate for the domain."""
    ensure_cert(Stack(cluster=cluster, project=project, zone=zone, domain=domain), dry_run=dry_run)


@cli.command()
@_common_options
def firewall(cluster: str, project: str, zone: str, dry_run: bool) -> None:
    """Allow the controller port from Google LB ranges only; deny public ingress."""
    ensure_firewall(Stack(cluster=cluster, project=project, zone=zone), dry_run=dry_run)


@cli.command()
@_common_options
@click.option("--controller-ip", help="Controller VM internal IP (default: discover from the GCE label)")
@click.option("--support-email", help="Support email for the IAP brand (only needed to create one)")
@click.option("--oauth-client-id", help="Existing IAP OAuth client id (default: find/create one)")
@click.option("--oauth-client-secret", help="Existing IAP OAuth client secret")
def backend(
    cluster: str,
    project: str,
    zone: str,
    dry_run: bool,
    controller_ip: str | None,
    support_email: str | None,
    oauth_client_id: str | None,
    oauth_client_secret: str | None,
) -> None:
    """Build the NEG -> health check -> backend service and enable IAP on it."""
    stack = Stack(cluster=cluster, project=project, zone=zone)
    controller_ip = controller_ip or discover_controller_ip(stack)
    client_id, client_secret = resolve_oauth(stack, oauth_client_id, oauth_client_secret, support_email, dry_run=dry_run)
    ensure_backend(stack, client_id, client_secret, controller_ip, dry_run=dry_run)


@cli.command()
@_common_options
def frontend(cluster: str, project: str, zone: str, dry_run: bool) -> None:
    """Build the URL map -> HTTPS proxy -> :443 forwarding rule (needs cert + IP)."""
    ensure_frontend(Stack(cluster=cluster, project=project, zone=zone), dry_run=dry_run)


@cli.command()
@_common_options
@click.option("--member", required=True, help="Principal to grant, e.g. user:you@example.com")
def grant(cluster: str, project: str, zone: str, dry_run: bool, member: str) -> None:
    """Grant a principal IAP access on the backend service."""
    grant_access(Stack(cluster=cluster, project=project, zone=zone), member, dry_run=dry_run)


@cli.command()
@_common_options
def status(cluster: str, project: str, zone: str, dry_run: bool) -> None:
    """Report which resources exist, the reserved IP, and the cert state."""
    stack = Stack(cluster=cluster, project=project, zone=zone)
    checks = [
        ("static IP", _compute(stack, "addresses", "describe", stack.address, "--global")),
        ("managed cert", _compute(stack, "ssl-certificates", "describe", stack.cert, "--global")),
        ("allow-LB firewall", _compute(stack, "firewall-rules", "describe", stack.allow_firewall)),
        ("deny-public firewall", _compute(stack, "firewall-rules", "describe", stack.deny_firewall)),
        ("NEG", _compute(stack, "network-endpoint-groups", "describe", stack.neg, f"--zone={stack.zone}")),
        ("health check", _compute(stack, "health-checks", "describe", stack.health_check, "--global")),
        ("backend service", _compute(stack, "backend-services", "describe", stack.backend, "--global")),
        ("URL map", _compute(stack, "url-maps", "describe", stack.url_map, "--global")),
        ("HTTPS proxy", _compute(stack, "target-https-proxies", "describe", stack.https_proxy, "--global")),
        ("forwarding rule", _compute(stack, "forwarding-rules", "describe", stack.forwarding_rule, "--global")),
    ]
    click.echo(f"Stack status for cluster={cluster} (project={project}):")
    for label, describe in checks:
        click.echo(f"  [{'OK ' if _exists(describe) else 'MISSING'}] {label}")

    addr = _run(
        [*_compute(stack, "addresses", "describe", stack.address, "--global"), "--format=value(address)"],
        check=False,
        capture=True,
    )
    if addr.returncode == 0 and addr.stdout.strip():
        click.echo(f"  reserved IP : {addr.stdout.strip()}")
    cert_state = _run(
        [*_compute(stack, "ssl-certificates", "describe", stack.cert, "--global"), "--format=value(managed.status)"],
        check=False,
        capture=True,
    )
    if cert_state.returncode == 0 and cert_state.stdout.strip():
        click.echo(f"  cert state  : {cert_state.stdout.strip()}")


@cli.command()
@_common_options
@click.option("--release-ip", is_flag=True, help="Also release the reserved static IP (breaks the DNS A record)")
@click.option("--delete-oauth-client", is_flag=True, help="Also delete this cluster's IAP OAuth client")
def teardown(
    cluster: str,
    project: str,
    zone: str,
    dry_run: bool,
    release_ip: bool,
    delete_oauth_client: bool,
) -> None:
    """Delete the LB stack in dependency order. Keeps the static IP + OAuth client
    by default so a redeploy reuses the DNS record and consent screen."""
    stack = Stack(cluster=cluster, project=project, zone=zone)

    def _delete(label: str, cmd: Sequence[str]) -> None:
        logger.info("→ deleting %s", label)
        result = _run([*cmd, "--quiet"], dry_run=dry_run, check=False, capture=True)
        if not dry_run and result.returncode != 0:
            logger.info("  (skip: %s missing or already deleted)", label)

    _delete(stack.forwarding_rule, _compute(stack, "forwarding-rules", "delete", stack.forwarding_rule, "--global"))
    _delete(stack.https_proxy, _compute(stack, "target-https-proxies", "delete", stack.https_proxy, "--global"))
    _delete(stack.url_map, _compute(stack, "url-maps", "delete", stack.url_map, "--global"))
    _delete(stack.backend, _compute(stack, "backend-services", "delete", stack.backend, "--global"))
    _delete(stack.neg, _compute(stack, "network-endpoint-groups", "delete", stack.neg, f"--zone={stack.zone}"))
    _delete(stack.health_check, _compute(stack, "health-checks", "delete", stack.health_check, "--global"))
    _delete(stack.cert, _compute(stack, "ssl-certificates", "delete", stack.cert, "--global"))
    _delete(stack.allow_firewall, _compute(stack, "firewall-rules", "delete", stack.allow_firewall))
    _delete(stack.deny_firewall, _compute(stack, "firewall-rules", "delete", stack.deny_firewall))

    if release_ip:
        _delete(stack.address, _compute(stack, "addresses", "delete", stack.address, "--global"))
    else:
        click.echo(f"Kept static IP {stack.address}; pass --release-ip to release it.")

    if delete_oauth_client and not dry_run:
        brands = _list_brands(stack)
        display = f"{OAUTH_CLIENT_DISPLAY} ({cluster})"
        for client in (brands and _list_oauth_clients(stack, brands[0]["name"])) or []:
            if client.get("displayName") == display:
                _delete(client["name"], _iap(stack, "oauth-clients", "delete", client["name"]))
    elif not delete_oauth_client:
        click.echo("Kept the IAP OAuth client; pass --delete-oauth-client to remove it.")


if __name__ == "__main__":
    cli()
