# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand up (idempotently) the external HTTPS Load Balancer + IAP stack that
fronts an Iris controller VM.

    client --HTTPS:443--> GCLB --(IAP)--> backend service --HTTP:10000--> controller VM

GCLB talks straight to the controller VM (no extra serverless hop and no Cloud Run
300s request cap that would truncate long-poll requests).

OAuth clients are *not* created here — the IAP OAuth Admin API is being turned
down, so the clients are created once in the Cloud Console and handed to this
module as their downloaded JSON secrets files:

* a **Web** OAuth client — IAP's anchor (``oauthSettings.clientId``); also serves
  the browser sign-in page. Needs the redirect URI
  ``https://iap.googleapis.com/v1/oauth/clientIds/<id>:handleRedirect``.
* a **Desktop** OAuth client — what the ``iris`` CLI drives for the browser login
  flow. Its id is added to ``oauthSettings.programmaticClients`` so IAP admits
  the CLI's bearer ID token (whose ``aud`` is the desktop client id).

Every resource is a single ``gcloud`` create guarded by an existence probe, so
the whole rollout — or any single stage — is safe to re-run.

The project, zone, domain, resource prefix, controller port, and discovery label
all come from the cluster's ``provisioning:`` config (:class:`Stack.from_config`),
never hard-coded constants.
"""

import dataclasses
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Sequence

import click

from marin_cluster.config import ClusterConfig

logger = logging.getLogger("iap-gclb")

# Google front-end / health-check / IAP source ranges that legitimately reach
# the controller port; a lower-priority deny rule blocks everything else so
# nobody can bypass IAP by hitting the VM's IP directly.
GOOGLE_LB_RANGES = "130.211.0.0/22,35.191.0.0/16"
IAP_ACCESSOR_ROLE = "roles/iap.httpsResourceAccessor"


@dataclasses.dataclass(frozen=True)
class Stack:
    """Names of every GCLB+IAP resource for one cluster, derived from its prefix."""

    cluster: str
    project: str
    zone: str
    prefix: str
    controller_port: int = 10000
    domain: str | None = None
    discovery_label: str | None = None

    @classmethod
    def from_config(cls, cfg: ClusterConfig) -> "Stack":
        """Build a :class:`Stack` from a cluster's ``provisioning`` config.

        Raises:
            ValueError: if the cluster is not a GCP cluster (no ``provisioning.gcp``).
            click.ClickException: if the cluster has no ``provisioning.iap_gclb``.
        """
        gcp = cfg.gcp
        prov = cfg.provisioning
        iap = prov.iap_gclb if prov is not None else None
        if iap is None:
            raise click.ClickException(f"cluster {cfg.name!r} has no provisioning.iap_gclb — nothing to roll out")
        return cls(
            cluster=cfg.name,
            project=gcp.project,
            zone=gcp.default_zone,
            prefix=iap.resource_prefix,
            controller_port=iap.controller_port,
            domain=iap.domain,
            discovery_label=iap.discovery_label,
        )

    @property
    def controller_label(self) -> str:
        """GCE label (for discovery) and network tag (for the firewall)."""
        return self.discovery_label or f"{self.prefix}-controller"

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
        return f"{self.prefix}-deny-public-{self.controller_port}"


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


def read_oauth_client(path: str) -> tuple[str, str]:
    """Return ``(client_id, client_secret)`` from a downloaded Google OAuth client JSON.

    Accepts both the ``{"web": {...}}`` and ``{"installed": {...}}`` shapes the
    Cloud Console emits.
    """
    with open(path) as fh:
        data = json.load(fh)
    for key in ("web", "installed"):
        block = data.get(key)
        if block:
            return block["client_id"], block["client_secret"]
    raise click.ClickException(f"{path}: not a Google OAuth client secrets JSON (no 'web'/'installed' key)")


# --------------------------------------------------------------------------- #
# Controller discovery
# --------------------------------------------------------------------------- #


def _discover_controller(stack: Stack, field: str) -> str:
    """Return a single field of the controller VM, found by its GCE label."""
    result = _run(
        _compute(
            stack,
            "instances",
            "list",
            f"--filter=labels.{stack.controller_label}=true",
            f"--format=value({field})",
        ),
        capture=True,
    )
    values = (result.stdout or "").split()
    if not values:
        raise click.ClickException(f"no controller VM labelled {stack.controller_label}=true")
    if len(values) > 1:
        raise click.ClickException(f"multiple VMs match {stack.controller_label}=true ({values})")
    return values[0]


def discover_controller_ip(stack: Stack) -> str:
    """Resolve the controller VM's internal IP from its GCE label."""
    return _discover_controller(stack, "networkInterfaces[0].networkIP")


def discover_controller_name(stack: Stack) -> str:
    """Resolve the controller VM's instance name from its GCE label."""
    return _discover_controller(stack, "name")


def discover_signed_header_audience(stack: Stack, *, dry_run: bool = False) -> str | None:
    """Return the IAP signed-header JWT audience for the backend service.

    The controller verifies this audience on IAP's ``X-Goog-IAP-JWT-Assertion``
    header to grant tokenless browsers the read-only dashboard role. Its value is
    ``/projects/<PROJECT_NUMBER>/global/backendServices/<BACKEND_SERVICE_ID>``.
    Returns None if the project number or backend service can't be resolved yet
    (e.g. dry-run, or before the backend stage has created the service).
    """
    if dry_run:
        return None
    project_number = _run(
        ["gcloud", "projects", "describe", stack.project, "--format=value(projectNumber)"],
        capture=True,
        check=False,
    ).stdout.strip()
    backend_id = _run(
        _compute(stack, "backend-services", "describe", stack.backend, "--global", "--format=value(id)"),
        capture=True,
        check=False,
    ).stdout.strip()
    if not project_number or not backend_id:
        return None
    return f"/projects/{project_number}/global/backendServices/{backend_id}"


# --------------------------------------------------------------------------- #
# Stage: static IP, managed cert
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
        raise click.ClickException("provisioning.iap_gclb.domain is required to create the managed SSL certificate")
    _ensure(
        f"managed SSL cert {stack.cert} ({stack.domain})",
        _exists(_compute(stack, "ssl-certificates", "describe", stack.cert, "--global")),
        _compute(stack, "ssl-certificates", "create", stack.cert, "--global", f"--domains={stack.domain}"),
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Stage: firewall (allow LB ranges; optionally deny public)
# --------------------------------------------------------------------------- #


def ensure_controller_tag(stack: Stack, *, dry_run: bool) -> None:
    """Tag the controller VM so the firewall rules apply to it (idempotent)."""
    name = discover_controller_name(stack)
    logger.info("→ ensuring network tag %s on controller VM %s", stack.controller_label, name)
    _run(
        _compute(stack, "instances", "add-tags", name, f"--zone={stack.zone}", f"--tags={stack.controller_label}"),
        dry_run=dry_run,
    )


def ensure_allow_firewall(stack: Stack, *, dry_run: bool) -> None:
    """Allow the controller port from the Google front-end / health-check ranges.

    Additive: without it the LB health check cannot reach the controller, so the
    backend never becomes healthy.
    """
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
            f"--rules=tcp:{stack.controller_port}",
            f"--source-ranges={GOOGLE_LB_RANGES}",
            f"--target-tags={stack.controller_label}",
            "--priority=900",
        ),
        dry_run=dry_run,
    )


def ensure_deny_firewall(stack: Stack, *, dry_run: bool) -> None:
    """Deny all other ingress to the controller port (defence in depth).

    Risky: this overrides ``default-allow-internal`` for the controller port, so
    any in-cluster component that reaches the controller over the network (e.g.
    task blob fetch) is cut. Apply only after confirming nothing internal needs
    direct ``:{port}`` access.
    """
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
            f"--rules=tcp:{stack.controller_port}",
            "--source-ranges=0.0.0.0/0",
            f"--target-tags={stack.controller_label}",
            "--priority=1000",
        ),
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Stage: backend (NEG -> controller endpoint -> health check -> backend service)
# --------------------------------------------------------------------------- #


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


def ensure_backend(stack: Stack, controller_name: str, controller_ip: str, *, dry_run: bool) -> None:
    """Build the backend half: zonal NEG -> controller endpoint -> health check
    -> backend service -> NEG attachment. IAP is enabled separately (``ensure_iap``)."""
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
            f"--default-port={stack.controller_port}",
        ),
        dry_run=dry_run,
    )

    if dry_run or not _neg_has_endpoint(stack, controller_ip):
        logger.info("→ attaching controller endpoint %s:%d to %s", controller_ip, stack.controller_port, stack.neg)
        _run(
            _compute(
                stack,
                "network-endpoint-groups",
                "update",
                stack.neg,
                f"--zone={stack.zone}",
                f"--add-endpoint=instance={controller_name},ip={controller_ip},port={stack.controller_port}",
            ),
            dry_run=dry_run,
        )
    else:
        logger.info("✓ endpoint %s:%d already attached to %s", controller_ip, stack.controller_port, stack.neg)

    _ensure(
        f"health check {stack.health_check} (HTTP /health :{stack.controller_port})",
        _exists(_compute(stack, "health-checks", "describe", stack.health_check, "--global")),
        _compute(
            stack,
            "health-checks",
            "create",
            "http",
            stack.health_check,
            "--global",
            f"--port={stack.controller_port}",
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


# --------------------------------------------------------------------------- #
# Stage: enable + configure IAP on the backend service
# --------------------------------------------------------------------------- #


def ensure_iap(stack: Stack, web_id: str, web_secret: str, desktop_id: str, *, dry_run: bool) -> None:
    """Enable IAP on the backend and bind the OAuth clients.

    The web client anchors IAP (``oauthSettings.clientId``) and renders the
    browser sign-in page; the desktop client is added to
    ``oauthSettings.programmaticClients`` so the CLI's bearer ID token (whose
    ``aud`` is the desktop client id) is admitted. Both ``backend-services
    update --iap=enabled`` and ``iap settings set`` are reconciling updates, so
    re-running is safe.
    """
    logger.info("→ enabling IAP on backend service %s", stack.backend)
    _run(
        _compute(stack, "backend-services", "update", stack.backend, "--global", "--iap=enabled"),
        dry_run=dry_run,
    )

    settings = {
        "access_settings": {
            "oauth_settings": {
                "client_id": web_id,
                "client_secret": web_secret,
                "programmatic_clients": [desktop_id],
            }
        }
    }
    logger.info(
        "→ IAP oauth_settings: clientId=%s programmaticClients=[%s]",
        web_id,
        desktop_id,
    )
    if dry_run:
        return

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(settings, fh)
        settings_path = fh.name
    try:
        _run(
            _iap(
                stack,
                "settings",
                "set",
                settings_path,
                "--resource-type=backend-services",
                f"--service={stack.backend}",
            )
        )
    finally:
        os.unlink(settings_path)


# --------------------------------------------------------------------------- #
# Stage: frontend (URL map -> HTTPS proxy -> forwarding rule) + IAM grant
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


def print_auth_block(
    stack: Stack,
    desktop_id: str,
    desktop_secret: str,
    member: str | None,
    signed_header_audience: str | None,
) -> None:
    """Print the cluster ``auth.iap`` block to paste into the cluster config."""
    admin = member.split(":", 1)[-1] if member else "you@example.com"
    click.echo()
    click.echo("Add this to the cluster config (the desktop client secret is non-confidential,")
    click.echo("RFC 8252 §8.5). optional=true keeps tokenless callers working until you cut over:")
    click.echo()
    click.echo("auth:")
    click.echo("  iap:")
    click.echo(f"    url: https://{stack.domain}")
    click.echo(f"    oauth_client_id: {desktop_id}")
    click.echo(f"    oauth_client_secret: {desktop_secret}")
    click.echo("    audiences:")
    click.echo(f"      - {desktop_id}")
    # The signed-header audience opts an IAP-authenticated browser into the
    # read-only dashboard role without `iris login`.
    if signed_header_audience:
        click.echo(f"    signed_header_audience: {signed_header_audience}")
    else:
        click.echo("    # signed_header_audience: /projects/<NUM>/global/backendServices/<ID>")
        click.echo("    #   (re-run `status` once the backend service exists to print it)")
    click.echo("  admin_users:")
    click.echo(f"    - {admin}")
    click.echo("  optional: true")
