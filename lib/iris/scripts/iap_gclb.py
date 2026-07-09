#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand up (idempotently) the external HTTPS Load Balancer + IAP stack that
fronts the Iris controller VMs.

    client --HTTPS:443--> GCLB --(IAP)--> backend service --HTTP:10000--> controller VM

One **shared frontend** carries every cluster: a single static IP, a URL map
that routes by ``Host`` header to per-cluster backends, an HTTPS proxy holding
each cluster's managed cert, and a ``:443`` forwarding rule. The frontend is
named after the cluster that first stood it up (``SHARED_FRONTEND``); its IP is
the one the cluster domains' DNS A records point at.

Each cluster contributes a **backend**: a zonal NEG to its controller VM, a
health check, an IAP-gated backend service, a managed cert for its domain, and a
host rule in the shared URL map (``iris.oa.dev`` -> marin, ``iris-dev.oa.dev`` ->
marin-dev, …). The frontend-owning cluster is the URL map's default service, so
it needs no explicit host rule.

GCLB talks straight to the controller VM — no serverless hop, and no 300s
request cap that would truncate long-poll requests.

OAuth clients are *not* created here — the IAP OAuth Admin API is being turned
down, so the clients are created once in the Cloud Console and handed to this
script as their downloaded JSON secrets files:

* a **Web** OAuth client — IAP's anchor (``oauthSettings.clientId``); also serves
  the browser sign-in page. Needs the redirect URI
  ``https://iap.googleapis.com/v1/oauth/clientIds/<id>:handleRedirect``.
* a **Desktop** OAuth client — what the ``iris`` CLI drives for the browser login
  flow. Its id is added to ``oauthSettings.programmaticClients`` so IAP admits
  the CLI's bearer ID token (whose ``aud`` is the desktop client id).

The same pair of clients can protect every cluster's backend service.

Every resource is a single ``gcloud`` create guarded by an existence probe, so
the whole rollout — or any single stage — is safe to re-run. ``deploy`` runs the
stages in dependency order; the per-stage subcommands (``address``, ``cert``,
``backend``, ``iap``, ``frontend``, ``route``, ``grant``, ``firewall``,
``token-proxy``) expose each on its own. ``status`` reports what exists and
``teardown`` removes a cluster's backend.

The ``firewall`` stage is kept separate and is *not* run by ``deploy`` unless
``--with-firewall`` is passed: its allow-rule is a prerequisite for the LB health
check, but its deny-public rule can cut internal task->controller traffic, so it
stays an explicit, deliberate step.

The ``token-proxy`` stage opens only the capability-URL path ``/proxy/t/*`` past
IAP (an IAP-free backend on the same controller NEG plus a URL-map path rule) so
off-cluster callers can reach a shared endpoint through a gist-style URL that
carries its scoped token in the path — possession of the URL is the credential.
Everything else under ``/proxy`` stays IAP-gated, so the dashboard's own log
viewer and any PRIVATE endpoint keep the browser's IAP identity. It runs by
default as part of ``deploy`` (idempotent); pass ``--no-token-proxy`` to keep the
controller fully IAP-gated, or run the ``token-proxy`` subcommand standalone.

Usage:
    uv run lib/iris/scripts/iap_gclb.py deploy marin \\
        --domain iris.oa.dev \\
        --web-client-secrets scratch/web.json \\
        --desktop-client-secrets scratch/desktop.json \\
        --member user:you@example.com
    uv run lib/iris/scripts/iap_gclb.py deploy marin-dev \\
        --domain iris-dev.oa.dev \\
        --web-client-secrets scratch/web.json \\
        --desktop-client-secrets scratch/desktop.json
    uv run lib/iris/scripts/iap_gclb.py status marin
    uv run lib/iris/scripts/iap_gclb.py teardown marin-dev
"""

import dataclasses
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import click
import yaml

logger = logging.getLogger("iap-gclb")

DEFAULT_PROJECT = "hai-gcp-models"
DEFAULT_ZONE = "us-central1-a"
CONTROLLER_PORT = 10000

# Path prefix opened past IAP by the token-proxy stage. Only capability URLs
# (``/proxy/t/<token>/<endpoint>/...``, carrying the scoped token in the path)
# route to the IAP-free backend. Everything else under ``/proxy`` — the
# dashboard's own system endpoints (``/proxy/system.log-server/...``), PRIVATE
# endpoints, in-browser access — stays IAP-gated, so the dashboard keeps its IAP
# identity. The controller verifies the path token before forwarding.
TOKEN_PROXY_PATHS = ["/proxy/t", "/proxy/t/*"]

# The cluster that owns the shared LB frontend (static IP, URL map, HTTPS proxy,
# forwarding rule). Its backend service is the URL map's default route.
SHARED_FRONTEND = "marin"

# Google front-end / health-check / IAP source ranges that legitimately reach
# the controller port; a lower-priority deny rule blocks everything else so
# nobody can bypass IAP by hitting the VM's IP directly.
GOOGLE_LB_RANGES = "130.211.0.0/22,35.191.0.0/16"
IAP_ACCESSOR_ROLE = "roles/iap.httpsResourceAccessor"


@dataclasses.dataclass(frozen=True)
class Frontend:
    """Names of the shared LB frontend resources, derived from its owning cluster.

    One per project: a static IP, a URL map that routes by Host to per-cluster
    backends, an HTTPS proxy holding every cluster's cert, and a :443 forwarding
    rule. Cluster domains' DNS A records all point at this frontend's IP.
    """

    name: str = SHARED_FRONTEND
    project: str = DEFAULT_PROJECT

    @property
    def prefix(self) -> str:
        return f"iris-{self.name}"

    @property
    def address(self) -> str:
        return f"{self.prefix}-ip"

    @property
    def url_map(self) -> str:
        return f"{self.prefix}-urlmap"

    @property
    def https_proxy(self) -> str:
        return f"{self.prefix}-https-proxy"

    @property
    def forwarding_rule(self) -> str:
        return f"{self.prefix}-fr"


@dataclasses.dataclass(frozen=True)
class Backend:
    """Names of one cluster's backend resources behind the shared frontend.

    A zonal NEG to the controller VM, a health check, an IAP-gated backend
    service, allow/deny firewall rules, a managed cert for the cluster's domain,
    and the cluster's host route (path matcher) in the shared URL map.
    """

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
    def service(self) -> str:
        """The backend service name."""
        return f"{self.prefix}-be"

    @property
    def proxy_service(self) -> str:
        """IAP-free backend service (token-proxy stage) on the same NEG.

        Fronts only the ``/proxy/t/*`` capability path, routed here by a URL-map
        path rule; the controller verifies the path-carried token via
        ``_authorize_proxy`` before forwarding.
        """
        return f"{self.prefix}-proxy-be"

    @property
    def path_matcher(self) -> str:
        """The shared URL map's path-matcher name for this cluster's host rule."""
        return self.cluster

    @property
    def allow_firewall(self) -> str:
        return f"{self.prefix}-allow-lb"

    @property
    def deny_firewall(self) -> str:
        return f"{self.prefix}-deny-public-{CONTROLLER_PORT}"

    @property
    def cert(self) -> str:
        """Managed-cert name for the cluster's domain (e.g. iris-oa-dev-cert)."""
        if not self.domain:
            raise click.ClickException(f"cluster {self.cluster} has no --domain, cannot name its cert")
        return f"{self.domain.replace('.', '-')}-cert"


# --------------------------------------------------------------------------- #
# gcloud helpers
# --------------------------------------------------------------------------- #


def _compute(project: str, *args: str) -> list[str]:
    """Build a ``gcloud compute ... --project=<project>`` command."""
    return ["gcloud", "compute", *args, f"--project={project}"]


def _iap(project: str, *args: str) -> list[str]:
    """Build a ``gcloud iap ... --project=<project>`` command."""
    return ["gcloud", "iap", *args, f"--project={project}"]


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


def _discover_controller(backend: Backend, field: str) -> str:
    """Return a single field of the controller VM, found by its GCE label."""
    result = _run(
        _compute(
            backend.project,
            "instances",
            "list",
            f"--filter=labels.{backend.controller_label}=true",
            f"--format=value({field})",
        ),
        capture=True,
    )
    values = (result.stdout or "").split()
    if not values:
        raise click.ClickException(f"no controller VM labelled {backend.controller_label}=true")
    if len(values) > 1:
        raise click.ClickException(f"multiple VMs match {backend.controller_label}=true ({values})")
    return values[0]


def discover_controller_ip(backend: Backend) -> str:
    """Resolve the controller VM's internal IP from its GCE label."""
    return _discover_controller(backend, "networkInterfaces[0].networkIP")


def discover_controller_name(backend: Backend) -> str:
    """Resolve the controller VM's instance name from its GCE label."""
    return _discover_controller(backend, "name")


def discover_signed_header_audience(backend: Backend, *, dry_run: bool = False) -> str | None:
    """Return the IAP signed-header JWT audience for the cluster's backend service.

    The controller verifies this audience on IAP's ``X-Goog-IAP-JWT-Assertion``
    header to grant tokenless browsers the read-only dashboard role. Its value is
    ``/projects/<PROJECT_NUMBER>/global/backendServices/<BACKEND_SERVICE_ID>``.
    Returns None if the project number or backend service can't be resolved yet
    (e.g. dry-run, or before the backend stage has created the service).
    """
    if dry_run:
        return None
    project_number = _run(
        ["gcloud", "projects", "describe", backend.project, "--format=value(projectNumber)"],
        capture=True,
        check=False,
    ).stdout.strip()
    backend_id = _run(
        _compute(backend.project, "backend-services", "describe", backend.service, "--global", "--format=value(id)"),
        capture=True,
        check=False,
    ).stdout.strip()
    if not project_number or not backend_id:
        return None
    return f"/projects/{project_number}/global/backendServices/{backend_id}"


# --------------------------------------------------------------------------- #
# Frontend stage: static IP
# --------------------------------------------------------------------------- #


def ensure_address(frontend: Frontend, *, dry_run: bool) -> str:
    """Reserve the shared global static IP if missing and return its address."""
    describe = _compute(frontend.project, "addresses", "describe", frontend.address, "--global")
    _ensure(
        f"static IP {frontend.address}",
        _exists(describe),
        _compute(frontend.project, "addresses", "create", frontend.address, "--global"),
        dry_run=dry_run,
    )
    if dry_run:
        return "<reserved-ip>"
    result = _run([*describe, "--format=value(address)"], capture=True)
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# Backend stage: managed cert for the cluster's domain
# --------------------------------------------------------------------------- #


def ensure_cert(backend: Backend, *, dry_run: bool) -> None:
    """Create the Google-managed SSL certificate for the cluster's domain.

    The cert stays PROVISIONING until the domain's DNS A record resolves to the
    shared frontend's static IP and the cert is served by the HTTPS proxy.
    """
    if not backend.domain:
        raise click.ClickException("--domain is required to create the managed SSL certificate")
    _ensure(
        f"managed SSL cert {backend.cert} ({backend.domain})",
        _exists(_compute(backend.project, "ssl-certificates", "describe", backend.cert, "--global")),
        _compute(backend.project, "ssl-certificates", "create", backend.cert, "--global", f"--domains={backend.domain}"),
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Backend stage: firewall (allow LB ranges; optionally deny public)
# --------------------------------------------------------------------------- #


def ensure_controller_tag(backend: Backend, *, dry_run: bool) -> None:
    """Tag the controller VM so the firewall rules apply to it (idempotent)."""
    name = discover_controller_name(backend)
    logger.info("→ ensuring network tag %s on controller VM %s", backend.controller_label, name)
    _run(
        _compute(
            backend.project,
            "instances",
            "add-tags",
            name,
            f"--zone={backend.zone}",
            f"--tags={backend.controller_label}",
        ),
        dry_run=dry_run,
    )


def ensure_allow_firewall(backend: Backend, *, dry_run: bool) -> None:
    """Allow the controller port from the Google front-end / health-check ranges.

    Additive: without it the LB health check cannot reach the controller, so the
    backend never becomes healthy.
    """
    _ensure(
        f"firewall allow-LB {backend.allow_firewall}",
        _exists(_compute(backend.project, "firewall-rules", "describe", backend.allow_firewall)),
        _compute(
            backend.project,
            "firewall-rules",
            "create",
            backend.allow_firewall,
            "--network=default",
            "--direction=INGRESS",
            "--action=ALLOW",
            f"--rules=tcp:{CONTROLLER_PORT}",
            f"--source-ranges={GOOGLE_LB_RANGES}",
            f"--target-tags={backend.controller_label}",
            "--priority=900",
        ),
        dry_run=dry_run,
    )


def ensure_deny_firewall(backend: Backend, *, dry_run: bool) -> None:
    """Deny all other ingress to the controller port (defence in depth).

    Risky: this overrides ``default-allow-internal`` for the controller port, so
    any in-cluster component that reaches the controller over the network (e.g.
    task blob fetch) is cut. Apply only after confirming nothing internal needs
    direct ``:{port}`` access.
    """
    _ensure(
        f"firewall deny-public {backend.deny_firewall}",
        _exists(_compute(backend.project, "firewall-rules", "describe", backend.deny_firewall)),
        _compute(
            backend.project,
            "firewall-rules",
            "create",
            backend.deny_firewall,
            "--network=default",
            "--direction=INGRESS",
            "--action=DENY",
            f"--rules=tcp:{CONTROLLER_PORT}",
            "--source-ranges=0.0.0.0/0",
            f"--target-tags={backend.controller_label}",
            "--priority=1000",
        ),
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Backend stage: NEG -> controller endpoint -> health check -> backend service
# --------------------------------------------------------------------------- #


def _neg_has_endpoint(backend: Backend, ip: str) -> bool:
    result = _run(
        _compute(
            backend.project,
            "network-endpoint-groups",
            "list-network-endpoints",
            backend.neg,
            f"--zone={backend.zone}",
            "--format=value(ipAddress)",
        ),
        capture=True,
    )
    return ip in (result.stdout or "").split()


def _backend_has_neg(backend: Backend, service: str | None = None) -> bool:
    result = _run(
        _compute(
            backend.project,
            "backend-services",
            "describe",
            service or backend.service,
            "--global",
            "--format=value(backends[].group)",
        ),
        capture=True,
    )
    return backend.neg in (result.stdout or "")


def ensure_backend(backend: Backend, controller_name: str, controller_ip: str, *, dry_run: bool) -> None:
    """Build the backend half: zonal NEG -> controller endpoint -> health check
    -> backend service -> NEG attachment. IAP is enabled separately (``ensure_iap``)."""
    _ensure(
        f"zonal NEG {backend.neg}",
        _exists(_compute(backend.project, "network-endpoint-groups", "describe", backend.neg, f"--zone={backend.zone}")),
        _compute(
            backend.project,
            "network-endpoint-groups",
            "create",
            backend.neg,
            f"--zone={backend.zone}",
            "--network=default",
            "--subnet=default",
            "--network-endpoint-type=GCE_VM_IP_PORT",
            f"--default-port={CONTROLLER_PORT}",
        ),
        dry_run=dry_run,
    )

    if dry_run or not _neg_has_endpoint(backend, controller_ip):
        logger.info("→ attaching controller endpoint %s:%d to %s", controller_ip, CONTROLLER_PORT, backend.neg)
        _run(
            _compute(
                backend.project,
                "network-endpoint-groups",
                "update",
                backend.neg,
                f"--zone={backend.zone}",
                f"--add-endpoint=instance={controller_name},ip={controller_ip},port={CONTROLLER_PORT}",
            ),
            dry_run=dry_run,
        )
    else:
        logger.info("✓ endpoint %s:%d already attached to %s", controller_ip, CONTROLLER_PORT, backend.neg)

    _ensure(
        f"health check {backend.health_check} (HTTP /health :{CONTROLLER_PORT})",
        _exists(_compute(backend.project, "health-checks", "describe", backend.health_check, "--global")),
        _compute(
            backend.project,
            "health-checks",
            "create",
            "http",
            backend.health_check,
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
        f"backend service {backend.service}",
        _exists(_compute(backend.project, "backend-services", "describe", backend.service, "--global")),
        _compute(
            backend.project,
            "backend-services",
            "create",
            backend.service,
            "--global",
            "--protocol=HTTP",
            "--port-name=http",
            f"--health-checks={backend.health_check}",
            "--timeout=120s",
            "--load-balancing-scheme=EXTERNAL_MANAGED",
        ),
        dry_run=dry_run,
    )

    if dry_run or not _backend_has_neg(backend):
        logger.info("→ adding NEG %s to backend service %s", backend.neg, backend.service)
        _run(
            _compute(
                backend.project,
                "backend-services",
                "add-backend",
                backend.service,
                "--global",
                f"--network-endpoint-group={backend.neg}",
                f"--network-endpoint-group-zone={backend.zone}",
                "--balancing-mode=RATE",
                "--max-rate-per-endpoint=1000",
            ),
            dry_run=dry_run,
        )
    else:
        logger.info("✓ NEG %s already attached to backend %s", backend.neg, backend.service)


# --------------------------------------------------------------------------- #
# Backend stage: enable + configure IAP on the backend service
# --------------------------------------------------------------------------- #


def ensure_iap(backend: Backend, web_id: str, web_secret: str, desktop_id: str, *, dry_run: bool) -> None:
    """Enable IAP on the cluster's backend and bind the OAuth clients.

    The web client anchors IAP (``oauthSettings.clientId``) and renders the
    browser sign-in page; the desktop client is added to
    ``oauthSettings.programmaticClients`` so the CLI's bearer ID token (whose
    ``aud`` is the desktop client id) is admitted. The same client pair can
    protect every cluster's backend. Both ``backend-services update
    --iap=enabled`` and ``iap settings set`` are reconciling updates, so
    re-running is safe.
    """
    logger.info("→ enabling IAP on backend service %s", backend.service)
    _run(
        _compute(backend.project, "backend-services", "update", backend.service, "--global", "--iap=enabled"),
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
                backend.project,
                "settings",
                "set",
                settings_path,
                "--resource-type=backend-services",
                f"--service={backend.service}",
            )
        )
    finally:
        os.unlink(settings_path)


# --------------------------------------------------------------------------- #
# Frontend stage: URL map -> host routing, HTTPS proxy + certs, forwarding rule
# --------------------------------------------------------------------------- #


def _proxy_cert_names(frontend: Frontend) -> list[str]:
    """Return the basenames of the certs currently attached to the HTTPS proxy."""
    result = _run(
        _compute(
            frontend.project,
            "target-https-proxies",
            "describe",
            frontend.https_proxy,
            "--global",
            "--format=value(sslCertificates)",
        ),
        capture=True,
        check=False,
    )
    raw = (result.stdout or "").replace(";", " ").split()
    return [ref.rsplit("/", 1)[-1] for ref in raw]


def add_proxy_cert(frontend: Frontend, cert: str, *, dry_run: bool) -> None:
    """Attach *cert* to the shared HTTPS proxy, preserving the others (idempotent)."""
    current = _proxy_cert_names(frontend)
    if cert in current:
        logger.info("✓ cert %s already on proxy %s", cert, frontend.https_proxy)
        return
    desired = [*current, cert]
    logger.info("→ attaching cert %s to proxy %s (certs now: %s)", cert, frontend.https_proxy, ",".join(desired))
    _run(
        _compute(
            frontend.project,
            "target-https-proxies",
            "update",
            frontend.https_proxy,
            "--global",
            f"--ssl-certificates={','.join(desired)}",
        ),
        dry_run=dry_run,
    )


def ensure_url_map(frontend: Frontend, default_service: str, *, dry_run: bool) -> None:
    """Create the shared URL map (default route -> the frontend cluster's backend)."""
    _ensure(
        f"URL map {frontend.url_map} (default -> {default_service})",
        _exists(_compute(frontend.project, "url-maps", "describe", frontend.url_map, "--global")),
        _compute(
            frontend.project, "url-maps", "create", frontend.url_map, "--global", f"--default-service={default_service}"
        ),
        dry_run=dry_run,
    )


def _url_map_has_matcher(frontend: Frontend, name: str) -> bool:
    result = _run(
        _compute(
            frontend.project,
            "url-maps",
            "describe",
            frontend.url_map,
            "--global",
            "--format=value(pathMatchers[].name)",
        ),
        capture=True,
        check=False,
    )
    return name in (result.stdout or "").replace(";", " ").split()


def ensure_route(frontend: Frontend, backend: Backend, *, dry_run: bool) -> None:
    """Route ``backend.domain`` to its backend service in the shared URL map.

    The frontend-owning cluster is already the URL map's default service, so it
    needs no host rule. Other clusters get a path matcher + host rule keyed on
    their domain.
    """
    if backend.cluster == frontend.name:
        logger.info("✓ %s is the frontend default service; no host rule needed", backend.cluster)
        return
    if not backend.domain:
        raise click.ClickException(f"--domain is required to route cluster {backend.cluster}")
    if _url_map_has_matcher(frontend, backend.path_matcher):
        logger.info("✓ host rule %s -> %s already in %s", backend.domain, backend.service, frontend.url_map)
        return
    logger.info("→ routing %s -> %s in %s", backend.domain, backend.service, frontend.url_map)
    _run(
        _compute(
            frontend.project,
            "url-maps",
            "add-path-matcher",
            frontend.url_map,
            "--global",
            f"--path-matcher-name={backend.path_matcher}",
            f"--default-service={backend.service}",
            f"--new-hosts={backend.domain}",
        ),
        dry_run=dry_run,
    )


def ensure_https_proxy(frontend: Frontend, cert: str, *, dry_run: bool) -> None:
    """Create the HTTPS proxy (with *cert*) or attach *cert* to the existing one."""
    if _exists(_compute(frontend.project, "target-https-proxies", "describe", frontend.https_proxy, "--global")):
        add_proxy_cert(frontend, cert, dry_run=dry_run)
        return
    logger.info("→ creating target HTTPS proxy %s", frontend.https_proxy)
    _run(
        _compute(
            frontend.project,
            "target-https-proxies",
            "create",
            frontend.https_proxy,
            "--global",
            f"--url-map={frontend.url_map}",
            f"--ssl-certificates={cert}",
        ),
        dry_run=dry_run,
    )


def ensure_forwarding_rule(frontend: Frontend, *, dry_run: bool) -> None:
    """Bind the shared static IP:443 to the HTTPS proxy."""
    _ensure(
        f"forwarding rule {frontend.forwarding_rule} ({frontend.address}:443)",
        _exists(_compute(frontend.project, "forwarding-rules", "describe", frontend.forwarding_rule, "--global")),
        _compute(
            frontend.project,
            "forwarding-rules",
            "create",
            frontend.forwarding_rule,
            "--global",
            f"--address={frontend.address}",
            f"--target-https-proxy={frontend.https_proxy}",
            "--ports=443",
            "--load-balancing-scheme=EXTERNAL_MANAGED",
        ),
        dry_run=dry_run,
    )


def ensure_frontend(frontend: Frontend, backend: Backend, *, dry_run: bool) -> str:
    """Reconcile the shared frontend and route *backend*'s domain to its service.

    Reserves the shared IP, ensures the URL map (default route -> the frontend
    cluster's backend service), adds this cluster's host rule, attaches its cert
    to the HTTPS proxy, and binds the :443 forwarding rule. Returns the reserved IP.
    """
    primary = Backend(cluster=frontend.name, project=frontend.project, zone=backend.zone)
    reserved_ip = ensure_address(frontend, dry_run=dry_run)
    ensure_url_map(frontend, primary.service, dry_run=dry_run)
    ensure_route(frontend, backend, dry_run=dry_run)
    ensure_https_proxy(frontend, backend.cert, dry_run=dry_run)
    ensure_forwarding_rule(frontend, dry_run=dry_run)
    return reserved_ip


# --------------------------------------------------------------------------- #
# token-proxy stage: open the capability-URL route (/proxy/t/*) to the controller
# without IAP. Callers reach a shared endpoint through a URL that carries its
# scoped token in the path; the Iris controller verifies that token before
# forwarding. Everything else under /proxy stays IAP-gated.
# --------------------------------------------------------------------------- #


def _backend_self_link(backend: Backend, service: str) -> str:
    """Return a backend service's full selfLink (the form a URL map references)."""
    result = _run(
        _compute(backend.project, "backend-services", "describe", service, "--global", "--format=value(selfLink)"),
        capture=True,
    )
    link = (result.stdout or "").strip()
    if not link:
        raise click.ClickException(f"backend service {service} not found; run the backend/token-proxy stage first")
    return link


def ensure_token_proxy_backend(backend: Backend, *, dry_run: bool) -> None:
    """Create the IAP-free backend service on the existing NEG + health check.

    Reuses the NEG and health check the ``backend`` stage created, so this must
    run after ``deploy``/``backend``. IAP is never enabled on it — the controller
    verifies the path token on ``/proxy/t`` requests itself.
    """
    _ensure(
        f"IAP-free backend service {backend.proxy_service}",
        _exists(_compute(backend.project, "backend-services", "describe", backend.proxy_service, "--global")),
        _compute(
            backend.project,
            "backend-services",
            "create",
            backend.proxy_service,
            "--global",
            "--protocol=HTTP",
            "--port-name=http",
            f"--health-checks={backend.health_check}",
            "--timeout=120s",
            "--load-balancing-scheme=EXTERNAL_MANAGED",
        ),
        dry_run=dry_run,
    )

    if dry_run or not _backend_has_neg(backend, backend.proxy_service):
        logger.info("→ adding NEG %s to IAP-free backend %s", backend.neg, backend.proxy_service)
        _run(
            _compute(
                backend.project,
                "backend-services",
                "add-backend",
                backend.proxy_service,
                "--global",
                f"--network-endpoint-group={backend.neg}",
                f"--network-endpoint-group-zone={backend.zone}",
                "--balancing-mode=RATE",
                "--max-rate-per-endpoint=1000",
            ),
            dry_run=dry_run,
        )
    else:
        logger.info("✓ NEG %s already attached to IAP-free backend %s", backend.neg, backend.proxy_service)


def _export_url_map(frontend: Frontend) -> dict:
    """Export the shared URL map as a dict (the import-round-trippable form)."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        path = fh.name
    try:
        _run(_compute(frontend.project, "url-maps", "export", frontend.url_map, "--global", f"--destination={path}"))
        return yaml.safe_load(Path(path).read_text())
    finally:
        os.unlink(path)


def _import_url_map(frontend: Frontend, doc: dict, *, dry_run: bool) -> None:
    """Write *doc* back to the shared URL map (atomic, fingerprint-guarded)."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
        path = fh.name
    try:
        _run(
            _compute(
                frontend.project, "url-maps", "import", frontend.url_map, "--global", f"--source={path}", "--quiet"
            ),
            dry_run=dry_run,
        )
    finally:
        os.unlink(path)


def ensure_token_proxy_route(frontend: Frontend, backend: Backend, *, dry_run: bool) -> None:
    """Route ``<domain>/proxy/t/*`` to the IAP-free backend, leaving the rest IAP-gated.

    Idempotent; touches only this cluster's host rule and its ``/proxy/t/*`` path
    rule. Only capability URLs (``/proxy/t/...``) stay unauthenticated; the
    dashboard keeps its IAP identity on every other ``/proxy`` path.
    """
    if not backend.domain:
        raise click.ClickException(f"--domain is required to open the token proxy route for {backend.cluster}")
    iap_link = _backend_self_link(backend, backend.service)
    proxy_link = _backend_self_link(backend, backend.proxy_service)

    doc = _export_url_map(frontend)
    doc.setdefault("hostRules", [])
    doc.setdefault("pathMatchers", [])

    host_rule = next((h for h in doc["hostRules"] if backend.domain in h.get("hosts", [])), None)
    matcher: dict
    if host_rule is None:
        # Frontend-owning cluster serves off the map default; give it a host
        # rule whose default stays the IAP backend.
        matcher = {"name": backend.path_matcher, "defaultService": iap_link, "pathRules": []}
        doc["hostRules"].append({"hosts": [backend.domain], "pathMatcher": backend.path_matcher})
        doc["pathMatchers"].append(matcher)
    else:
        matcher = next(m for m in doc["pathMatchers"] if m["name"] == host_rule["pathMatcher"])
        matcher.setdefault("pathRules", [])

    rules = matcher["pathRules"]
    existing = next((r for r in rules if set(r.get("paths", [])) == set(TOKEN_PROXY_PATHS)), None)
    if existing is not None and existing.get("service") == proxy_link:
        logger.info("✓ /proxy/t/* already routes to %s for %s", backend.proxy_service, backend.domain)
        return
    if existing is not None:
        existing["service"] = proxy_link
    else:
        rules.append({"paths": list(TOKEN_PROXY_PATHS), "service": proxy_link})
    logger.info(
        "→ routing %s/proxy/t/* -> %s (IAP-free capability URLs); everything else stays IAP-gated",
        backend.domain,
        backend.proxy_service,
    )
    _import_url_map(frontend, doc, dry_run=dry_run)


def remove_token_proxy_route(frontend: Frontend, backend: Backend, *, dry_run: bool) -> None:
    """Remove this cluster's ``/proxy/t/*`` capability path rule from the shared URL map (best-effort)."""
    if not backend.domain:
        return
    doc = _export_url_map(frontend)
    host_rule = next((h for h in doc.get("hostRules", []) if backend.domain in h.get("hosts", [])), None)
    if host_rule is None:
        return
    matcher = next((m for m in doc.get("pathMatchers", []) if m["name"] == host_rule["pathMatcher"]), None)
    if matcher is None:
        return
    before = matcher.get("pathRules", [])
    kept = [r for r in before if set(r.get("paths", [])) != set(TOKEN_PROXY_PATHS)]
    if len(kept) == len(before):
        return
    matcher["pathRules"] = kept
    logger.info("→ removing /proxy route(s) for %s from %s", backend.domain, frontend.url_map)
    _import_url_map(frontend, doc, dry_run=dry_run)


def grant_access(backend: Backend, member: str, *, dry_run: bool) -> None:
    """Grant *member* IAP access (roles/iap.httpsResourceAccessor) on the backend."""
    logger.info("→ granting %s %s on %s", member, IAP_ACCESSOR_ROLE, backend.service)
    _run(
        _iap(
            backend.project,
            "web",
            "add-iam-policy-binding",
            "--resource-type=backend-services",
            f"--service={backend.service}",
            f"--member={member}",
            f"--role={IAP_ACCESSOR_ROLE}",
        ),
        dry_run=dry_run,
    )


def print_auth_block(
    backend: Backend,
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
    click.echo(f"    url: https://{backend.domain}")
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _common_options(func):
    func = click.argument("cluster")(func)
    func = click.option("--project", default=DEFAULT_PROJECT, show_default=True, help="GCP project id")(func)
    func = click.option("--zone", default=DEFAULT_ZONE, show_default=True, help="Zone of the controller VM / NEG")(func)
    func = click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")(func)
    return func


def _frontend_option(func):
    return click.option(
        "--frontend",
        "frontend_name",
        default=SHARED_FRONTEND,
        show_default=True,
        help="Cluster that owns the shared LB frontend (IP / URL map / proxy / forwarding rule)",
    )(func)


def _client_options(func):
    func = click.option(
        "--web-client-secrets",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Downloaded JSON for the Web OAuth client (IAP anchor + browser sign-in)",
    )(func)
    func = click.option(
        "--desktop-client-secrets",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Downloaded JSON for the Desktop OAuth client (the CLI login flow)",
    )(func)
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
@_client_options
@_frontend_option
@click.option("--domain", required=True, help="Domain whose DNS A record points at the shared frontend IP")
@click.option("--controller-ip", help="Controller VM internal IP (default: discover from the GCE label)")
@click.option("--member", help="Principal to grant IAP access, e.g. user:you@example.com")
@click.option("--with-firewall", is_flag=True, help="Also run the allow-LB firewall stage (tag VM + allow rule)")
@click.option(
    "--no-token-proxy",
    is_flag=True,
    help="Skip the token-proxy stage (leave the controller fully IAP-gated; no off-cluster capability URLs)",
)
def deploy(
    cluster: str,
    project: str,
    zone: str,
    dry_run: bool,
    web_client_secrets: str,
    desktop_client_secrets: str,
    frontend_name: str,
    domain: str,
    controller_ip: str | None,
    member: str | None,
    with_firewall: bool,
    no_token_proxy: bool,
) -> None:
    """Stand up a cluster's IAP backend and route it through the shared frontend.

    Runs the stages in dependency order. The shared frontend (IP / URL map /
    proxy / forwarding rule) is created on first use and reused thereafter; the
    deny-public firewall rule is never part of ``deploy`` — add it explicitly
    with ``firewall <cluster> --deny-public`` once internal access is confirmed.
    """
    backend = Backend(cluster=cluster, project=project, zone=zone, domain=domain)
    frontend = Frontend(name=frontend_name, project=project)
    web_id, web_secret = read_oauth_client(web_client_secrets)
    desktop_id, desktop_secret = read_oauth_client(desktop_client_secrets)
    controller_name = discover_controller_name(backend)
    controller_ip = controller_ip or discover_controller_ip(backend)

    # Backend half.
    ensure_cert(backend, dry_run=dry_run)
    ensure_backend(backend, controller_name, controller_ip, dry_run=dry_run)
    ensure_iap(backend, web_id, web_secret, desktop_id, dry_run=dry_run)
    if with_firewall:
        ensure_controller_tag(backend, dry_run=dry_run)
        ensure_allow_firewall(backend, dry_run=dry_run)
    if member:
        grant_access(backend, member, dry_run=dry_run)

    # Shared frontend + this cluster's route.
    reserved_ip = ensure_frontend(frontend, backend, dry_run=dry_run)

    if not no_token_proxy:
        ensure_token_proxy_backend(backend, dry_run=dry_run)
        ensure_token_proxy_route(frontend, backend, dry_run=dry_run)

    click.echo()
    click.echo(f"Backend for cluster={cluster} reconciled behind frontend={frontend_name}.")
    click.echo(f"  Shared IP      : {reserved_ip}")
    click.echo(f"  Domain         : {domain}  (ensure a DNS A record -> {reserved_ip})")
    click.echo(f"  URL            : https://{domain}")
    click.echo(f"  Web client     : {web_id}")
    click.echo(f"  Desktop client : {desktop_id}  (programmatic / CLI)")
    if not with_firewall:
        click.echo()
        click.echo("Firewall NOT applied. The backend health check needs the allow-LB rule:")
        click.echo(f"  uv run {sys.argv[0]} firewall {cluster}")
    click.echo()
    if no_token_proxy:
        click.echo("Token proxy NOT opened (--no-token-proxy). To open capability URLs off-cluster later:")
        click.echo(f"  uv run {sys.argv[0]} token-proxy {cluster} --domain {domain}")
    else:
        click.echo(
            f"Opened https://{domain}/proxy/t/* (IAP-free capability URLs) -> {backend.proxy_service}; "
            "the controller verifies the path token."
        )
    signed_header_audience = discover_signed_header_audience(backend, dry_run=dry_run)
    print_auth_block(backend, desktop_id, desktop_secret, member, signed_header_audience)


@cli.command()
@_common_options
@_frontend_option
def address(cluster: str, project: str, zone: str, dry_run: bool, frontend_name: str) -> None:
    """Reserve the shared static IP and print it (point cluster domains' A records here)."""
    reserved_ip = ensure_address(Frontend(name=frontend_name, project=project), dry_run=dry_run)
    click.echo(f"reserved-ip : {reserved_ip}")


@cli.command()
@_common_options
@click.option("--domain", required=True, help="Domain for the managed SSL certificate")
def cert(cluster: str, project: str, zone: str, dry_run: bool, domain: str) -> None:
    """Create the Google-managed SSL certificate for the cluster's domain."""
    ensure_cert(Backend(cluster=cluster, project=project, zone=zone, domain=domain), dry_run=dry_run)


@cli.command()
@_common_options
@click.option("--deny-public", is_flag=True, help="Also add the deny-public rule (blocks all non-Google ingress)")
def firewall(cluster: str, project: str, zone: str, dry_run: bool, deny_public: bool) -> None:
    """Tag the controller VM and allow the Google LB ranges to reach the controller port.

    The allow rule is additive. Pass --deny-public to *also* block every other
    source — only do that once you've confirmed nothing internal reaches the
    controller port directly.
    """
    backend = Backend(cluster=cluster, project=project, zone=zone)
    ensure_controller_tag(backend, dry_run=dry_run)
    ensure_allow_firewall(backend, dry_run=dry_run)
    if deny_public:
        ensure_deny_firewall(backend, dry_run=dry_run)


@cli.command()
@_common_options
@click.option("--controller-ip", help="Controller VM internal IP (default: discover from the GCE label)")
def backend(cluster: str, project: str, zone: str, dry_run: bool, controller_ip: str | None) -> None:
    """Build the NEG -> health check -> backend service (no IAP; see the iap stage)."""
    backend = Backend(cluster=cluster, project=project, zone=zone)
    controller_name = discover_controller_name(backend)
    controller_ip = controller_ip or discover_controller_ip(backend)
    ensure_backend(backend, controller_name, controller_ip, dry_run=dry_run)


@cli.command()
@_common_options
@_client_options
def iap(
    cluster: str,
    project: str,
    zone: str,
    dry_run: bool,
    web_client_secrets: str,
    desktop_client_secrets: str,
) -> None:
    """Enable IAP on the cluster's backend and bind the web + desktop OAuth clients."""
    backend = Backend(cluster=cluster, project=project, zone=zone)
    web_id, web_secret = read_oauth_client(web_client_secrets)
    desktop_id, _ = read_oauth_client(desktop_client_secrets)
    ensure_iap(backend, web_id, web_secret, desktop_id, dry_run=dry_run)


@cli.command()
@_common_options
@_frontend_option
@click.option("--domain", required=True, help="Domain whose DNS A record points at the shared frontend IP")
def frontend(cluster: str, project: str, zone: str, dry_run: bool, frontend_name: str, domain: str) -> None:
    """Build/extend the shared frontend and route the cluster's domain to its backend.

    Reserves the IP, ensures the URL map (default -> the frontend cluster's
    backend), adds this cluster's host rule, attaches its cert to the HTTPS
    proxy, and binds the :443 forwarding rule.
    """
    fe = Frontend(name=frontend_name, project=project)
    backend = Backend(cluster=cluster, project=project, zone=zone, domain=domain)
    ensure_frontend(fe, backend, dry_run=dry_run)


@cli.command()
@_common_options
@_frontend_option
@click.option("--domain", required=True, help="Domain whose DNS A record points at the shared frontend IP")
def route(cluster: str, project: str, zone: str, dry_run: bool, frontend_name: str, domain: str) -> None:
    """Route a cluster's domain to its backend service in the shared URL map."""
    fe = Frontend(name=frontend_name, project=project)
    ensure_route(fe, Backend(cluster=cluster, project=project, zone=zone, domain=domain), dry_run=dry_run)


@cli.command()
@_common_options
@click.option("--member", required=True, help="Principal to grant, e.g. user:you@example.com")
def grant(cluster: str, project: str, zone: str, dry_run: bool, member: str) -> None:
    """Grant a principal IAP access on the cluster's backend service."""
    grant_access(Backend(cluster=cluster, project=project, zone=zone), member, dry_run=dry_run)


@cli.command("token-proxy")
@_common_options
@_frontend_option
@click.option(
    "--domain", required=True, help="Cluster domain whose /proxy/t/* capability path should be opened past IAP"
)
def token_proxy(cluster: str, project: str, zone: str, dry_run: bool, frontend_name: str, domain: str) -> None:
    """Open ONLY ``<domain>/proxy/t/*`` (capability URLs) off-cluster via an IAP-free backend.

    Adds a second backend service (IAP disabled) on the same controller NEG and a
    URL-map path rule routing ``/proxy/t/*`` to it, leaving the dashboard, RPC
    surface, and all identity-gated ``/proxy`` traffic IAP-gated. A capability URL
    carries its scoped token in the path, so the controller verifies that token
    before forwarding — possession of the URL is the credential. Run after
    ``deploy``/``backend`` (it reuses the NEG + health check).
    """
    fe = Frontend(name=frontend_name, project=project)
    backend = Backend(cluster=cluster, project=project, zone=zone, domain=domain)
    ensure_token_proxy_backend(backend, dry_run=dry_run)
    ensure_token_proxy_route(fe, backend, dry_run=dry_run)
    click.echo()
    click.echo(f"Opened https://{domain}/proxy/t/* (IAP-free capability URLs) -> {backend.proxy_service}.")
    click.echo("  The scoped token rides in the path; the controller verifies it before forwarding.")
    click.echo("  Everything else under /proxy stays IAP-gated (dashboard identity preserved).")


@cli.command()
@_common_options
@_frontend_option
def status(cluster: str, project: str, zone: str, dry_run: bool, frontend_name: str) -> None:
    """Report which resources exist for the shared frontend and the cluster's backend."""
    fe = Frontend(name=frontend_name, project=project)
    backend = Backend(cluster=cluster, project=project, zone=zone)
    frontend_checks = [
        ("static IP", _compute(project, "addresses", "describe", fe.address, "--global")),
        ("URL map", _compute(project, "url-maps", "describe", fe.url_map, "--global")),
        ("HTTPS proxy", _compute(project, "target-https-proxies", "describe", fe.https_proxy, "--global")),
        ("forwarding rule", _compute(project, "forwarding-rules", "describe", fe.forwarding_rule, "--global")),
    ]
    backend_checks = [
        ("allow-LB firewall", _compute(project, "firewall-rules", "describe", backend.allow_firewall)),
        ("deny-public firewall", _compute(project, "firewall-rules", "describe", backend.deny_firewall)),
        ("NEG", _compute(project, "network-endpoint-groups", "describe", backend.neg, f"--zone={zone}")),
        ("health check", _compute(project, "health-checks", "describe", backend.health_check, "--global")),
        ("backend service", _compute(project, "backend-services", "describe", backend.service, "--global")),
    ]
    click.echo(f"Shared frontend={frontend_name} (project={project}):")
    for label, describe in frontend_checks:
        click.echo(f"  [{'OK ' if _exists(describe) else 'MISSING'}] {label}")
    addr = _run(
        [*_compute(project, "addresses", "describe", fe.address, "--global"), "--format=value(address)"],
        check=False,
        capture=True,
    )
    if addr.returncode == 0 and addr.stdout.strip():
        click.echo(f"  reserved IP : {addr.stdout.strip()}")
    certs = _proxy_cert_names(fe)
    if certs:
        click.echo(f"  proxy certs : {', '.join(certs)}")

    click.echo(f"Backend cluster={cluster}:")
    for label, describe in backend_checks:
        click.echo(f"  [{'OK ' if _exists(describe) else 'MISSING'}] {label}")
    has_route = backend.cluster == frontend_name or _url_map_has_matcher(fe, backend.path_matcher)
    click.echo(f"  [{'OK ' if has_route else 'MISSING'}] host route in {fe.url_map}")
    has_proxy = _exists(_compute(project, "backend-services", "describe", backend.proxy_service, "--global"))
    click.echo(f"  [{'OK ' if has_proxy else 'off '}] token-proxy (IAP-free) backend {backend.proxy_service}")
    audience = discover_signed_header_audience(backend)
    if audience:
        click.echo(f"  iap jwt aud : {audience}  (auth.iap.signed_header_audience)")


@cli.command()
@_common_options
@_frontend_option
@click.option("--domain", help="Cluster domain (needed to name the cert to delete)")
@click.option(
    "--include-frontend",
    is_flag=True,
    help="Also delete the SHARED frontend (IP / URL map / proxy / forwarding rule) — affects every cluster",
)
@click.option("--release-ip", is_flag=True, help="With --include-frontend, also release the static IP (breaks DNS)")
def teardown(
    cluster: str,
    project: str,
    zone: str,
    dry_run: bool,
    frontend_name: str,
    domain: str | None,
    include_frontend: bool,
    release_ip: bool,
) -> None:
    """Delete a cluster's backend and its route. Leaves the shared frontend intact
    unless --include-frontend is passed. OAuth clients are Console-managed and untouched."""
    fe = Frontend(name=frontend_name, project=project)
    backend = Backend(cluster=cluster, project=project, zone=zone, domain=domain)

    def _delete(label: str, cmd: Sequence[str]) -> None:
        logger.info("→ deleting %s", label)
        result = _run([*cmd, "--quiet"], dry_run=dry_run, check=False, capture=True)
        if not dry_run and result.returncode != 0:
            logger.info("  (skip: %s missing or already deleted)", label)

    # Drop this cluster's host route first so the URL map stops referencing it.
    if backend.cluster != frontend_name and _url_map_has_matcher(fe, backend.path_matcher):
        if domain:
            _delete(
                f"host rule {domain}",
                _compute(project, "url-maps", "remove-host-rule", fe.url_map, "--global", f"--host={domain}"),
            )
        _delete(
            f"path matcher {backend.path_matcher}",
            _compute(
                project,
                "url-maps",
                "remove-path-matcher",
                fe.url_map,
                "--global",
                f"--path-matcher-name={backend.path_matcher}",
            ),
        )

    # The IAP-free token-proxy backend + its /proxy route, if the stage was run.
    # Its route must be dropped and the service deleted before the shared NEG.
    if _exists(_compute(project, "backend-services", "describe", backend.proxy_service, "--global")):
        remove_token_proxy_route(fe, backend, dry_run=dry_run)
        _delete(
            backend.proxy_service, _compute(project, "backend-services", "delete", backend.proxy_service, "--global")
        )

    _delete(backend.service, _compute(project, "backend-services", "delete", backend.service, "--global"))
    _delete(backend.neg, _compute(project, "network-endpoint-groups", "delete", backend.neg, f"--zone={zone}"))
    _delete(backend.health_check, _compute(project, "health-checks", "delete", backend.health_check, "--global"))
    _delete(backend.allow_firewall, _compute(project, "firewall-rules", "delete", backend.allow_firewall))
    _delete(backend.deny_firewall, _compute(project, "firewall-rules", "delete", backend.deny_firewall))
    # The cert can only be deleted once it is off the proxy; detach by re-setting
    # the proxy's cert list without it (skipped if the proxy is already gone).
    if domain:
        remaining = [c for c in _proxy_cert_names(fe) if c != backend.cert]
        if remaining and backend.cert in _proxy_cert_names(fe):
            _run(
                _compute(
                    project,
                    "target-https-proxies",
                    "update",
                    fe.https_proxy,
                    "--global",
                    f"--ssl-certificates={','.join(remaining)}",
                ),
                dry_run=dry_run,
                check=False,
            )
        _delete(backend.cert, _compute(project, "ssl-certificates", "delete", backend.cert, "--global"))

    if include_frontend:
        _delete(fe.forwarding_rule, _compute(project, "forwarding-rules", "delete", fe.forwarding_rule, "--global"))
        _delete(fe.https_proxy, _compute(project, "target-https-proxies", "delete", fe.https_proxy, "--global"))
        _delete(fe.url_map, _compute(project, "url-maps", "delete", fe.url_map, "--global"))
        if release_ip:
            _delete(fe.address, _compute(project, "addresses", "delete", fe.address, "--global"))
        else:
            click.echo(f"Kept static IP {fe.address}; pass --release-ip to release it.")


if __name__ == "__main__":
    cli()
