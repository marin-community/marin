# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``marin-cluster admin`` — config-driven GCP cluster-admin verbs.

Every verb reads the active cluster from the click context and loads its
``provisioning:`` config; nothing about project/zone/domain/SA identity is a flag.
GCP-only verbs call :pyattr:`ClusterConfig.gcp` early, so a non-GCP cluster
(e.g. CoreWeave) fails with a clear "not a GCP cluster" error before any work.
"""

import logging
import sys

import click

from marin_cluster.admin import iam, iap_gclb
from marin_cluster.admin.iap_gclb import Stack
from marin_cluster.config import ClusterConfig

# Iris scheduler priority bands, duplicated as a constant so `--max-band` validates
# without importing iris at module load (the lazy budget step is the real check).
_PRIORITY_BAND_NAMES = ("production", "interactive", "batch")


def _load(ctx: click.Context) -> ClusterConfig:
    """Load the active cluster's full config from the parent CLI context."""
    return ClusterConfig.load(ctx.obj.get("cluster") if ctx.obj else None)


def _stack(ctx: click.Context) -> Stack:
    """Build the IAP/GCLB :class:`Stack` for the active cluster.

    Surfaces the GCP gate's "not a GCP cluster" error as a clean ClickException.
    """
    try:
        return Stack.from_config(_load(ctx))
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _client_options(func):
    """The two per-invocation OAuth client-secret JSON paths (genuine secrets)."""
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


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG-level logging")
def admin(verbose: bool) -> None:
    """Administer a cluster's GCP provisioning (IAM, IAP/GCLB, users)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


# --------------------------------------------------------------------------- #
# admin iap — the GCLB + IAP ingress stack
# --------------------------------------------------------------------------- #


@admin.group()
def iap() -> None:
    """Stand up / inspect the external HTTPS LB + IAP front door."""


@iap.command()
@_client_options
@click.option("--controller-ip", help="Controller VM internal IP (default: discover from the GCE label)")
@click.option("--member", help="Principal to grant IAP access, e.g. user:you@example.com")
@click.option("--with-firewall", is_flag=True, help="Also run the allow-LB firewall stage (tag VM + allow rule)")
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def deploy(
    ctx: click.Context,
    web_client_secrets: str,
    desktop_client_secrets: str,
    controller_ip: str | None,
    member: str | None,
    with_firewall: bool,
    dry_run: bool,
) -> None:
    """Run the stages in dependency order to stand up the full stack.

    The deny-public firewall rule is never part of ``deploy`` — add it explicitly
    with ``firewall --deny-public`` once internal access is confirmed.
    """
    stack = _stack(ctx)
    web_id, web_secret = iap_gclb.read_oauth_client(web_client_secrets)
    desktop_id, desktop_secret = iap_gclb.read_oauth_client(desktop_client_secrets)
    controller_name = iap_gclb.discover_controller_name(stack)
    controller_ip = controller_ip or iap_gclb.discover_controller_ip(stack)

    reserved_ip = iap_gclb.ensure_address(stack, dry_run=dry_run)
    iap_gclb.ensure_cert(stack, dry_run=dry_run)
    if with_firewall:
        iap_gclb.ensure_controller_tag(stack, dry_run=dry_run)
        iap_gclb.ensure_allow_firewall(stack, dry_run=dry_run)
    iap_gclb.ensure_backend(stack, controller_name, controller_ip, dry_run=dry_run)
    iap_gclb.ensure_iap(stack, web_id, web_secret, desktop_id, dry_run=dry_run)
    iap_gclb.ensure_frontend(stack, dry_run=dry_run)
    if member:
        iap_gclb.grant_access(stack, member, dry_run=dry_run)

    click.echo()
    click.echo(f"Stack for cluster={stack.cluster} reconciled.")
    click.echo(f"  Reserved IP    : {reserved_ip}")
    click.echo(f"  Domain         : {stack.domain}  (ensure a DNS A record -> {reserved_ip})")
    click.echo(f"  URL            : https://{stack.domain}")
    click.echo(f"  Web client     : {web_id}")
    click.echo(f"  Desktop client : {desktop_id}  (programmatic / CLI)")
    if not with_firewall:
        click.echo()
        click.echo("Firewall NOT applied. The backend health check needs the allow-LB rule:")
        click.echo(f"  marin-cluster admin iap firewall  (cluster {stack.cluster})")
    signed_header_audience = iap_gclb.discover_signed_header_audience(stack, dry_run=dry_run)
    iap_gclb.print_auth_block(stack, desktop_id, desktop_secret, member, signed_header_audience)


@iap.command()
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def address(ctx: click.Context, dry_run: bool) -> None:
    """Reserve the global static IP and print it (point the domain's A record here)."""
    stack = _stack(ctx)
    reserved_ip = iap_gclb.ensure_address(stack, dry_run=dry_run)
    click.echo(f"reserved-ip : {reserved_ip}")


@iap.command()
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def cert(ctx: click.Context, dry_run: bool) -> None:
    """Create the Google-managed SSL certificate for the configured domain."""
    iap_gclb.ensure_cert(_stack(ctx), dry_run=dry_run)


@iap.command()
@click.option("--deny-public", is_flag=True, help="Also add the deny-public rule (blocks all non-Google ingress)")
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def firewall(ctx: click.Context, deny_public: bool, dry_run: bool) -> None:
    """Tag the controller VM and allow the Google LB ranges to reach the controller port.

    The allow rule is additive. Pass --deny-public to *also* block every other
    source — only do that once you've confirmed nothing internal reaches the
    controller port directly.
    """
    stack = _stack(ctx)
    iap_gclb.ensure_controller_tag(stack, dry_run=dry_run)
    iap_gclb.ensure_allow_firewall(stack, dry_run=dry_run)
    if deny_public:
        iap_gclb.ensure_deny_firewall(stack, dry_run=dry_run)


@iap.command()
@click.option("--controller-ip", help="Controller VM internal IP (default: discover from the GCE label)")
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def backend(ctx: click.Context, controller_ip: str | None, dry_run: bool) -> None:
    """Build the NEG -> health check -> backend service (no IAP; see the iap stage)."""
    stack = _stack(ctx)
    controller_name = iap_gclb.discover_controller_name(stack)
    controller_ip = controller_ip or iap_gclb.discover_controller_ip(stack)
    iap_gclb.ensure_backend(stack, controller_name, controller_ip, dry_run=dry_run)


@iap.command(name="iap")
@_client_options
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def iap_stage(
    ctx: click.Context,
    web_client_secrets: str,
    desktop_client_secrets: str,
    dry_run: bool,
) -> None:
    """Enable IAP on the backend and bind the web + desktop OAuth clients."""
    stack = _stack(ctx)
    web_id, web_secret = iap_gclb.read_oauth_client(web_client_secrets)
    desktop_id, _ = iap_gclb.read_oauth_client(desktop_client_secrets)
    iap_gclb.ensure_iap(stack, web_id, web_secret, desktop_id, dry_run=dry_run)


@iap.command()
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def frontend(ctx: click.Context, dry_run: bool) -> None:
    """Build the URL map -> HTTPS proxy -> :443 forwarding rule (needs cert + IP)."""
    iap_gclb.ensure_frontend(_stack(ctx), dry_run=dry_run)


@iap.command()
@click.option("--member", required=True, help="Principal to grant, e.g. user:you@example.com")
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def grant(ctx: click.Context, member: str, dry_run: bool) -> None:
    """Grant a principal IAP access on the backend service."""
    iap_gclb.grant_access(_stack(ctx), member, dry_run=dry_run)


@iap.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Report which resources exist, the reserved IP, and the cert state."""
    stack = _stack(ctx)
    checks = [
        ("static IP", iap_gclb._compute(stack, "addresses", "describe", stack.address, "--global")),
        ("managed cert", iap_gclb._compute(stack, "ssl-certificates", "describe", stack.cert, "--global")),
        ("allow-LB firewall", iap_gclb._compute(stack, "firewall-rules", "describe", stack.allow_firewall)),
        ("deny-public firewall", iap_gclb._compute(stack, "firewall-rules", "describe", stack.deny_firewall)),
        ("NEG", iap_gclb._compute(stack, "network-endpoint-groups", "describe", stack.neg, f"--zone={stack.zone}")),
        ("health check", iap_gclb._compute(stack, "health-checks", "describe", stack.health_check, "--global")),
        ("backend service", iap_gclb._compute(stack, "backend-services", "describe", stack.backend, "--global")),
        ("URL map", iap_gclb._compute(stack, "url-maps", "describe", stack.url_map, "--global")),
        ("HTTPS proxy", iap_gclb._compute(stack, "target-https-proxies", "describe", stack.https_proxy, "--global")),
        ("forwarding rule", iap_gclb._compute(stack, "forwarding-rules", "describe", stack.forwarding_rule, "--global")),
    ]
    click.echo(f"Stack status for cluster={stack.cluster} (project={stack.project}):")
    for label, describe in checks:
        click.echo(f"  [{'OK ' if iap_gclb._exists(describe) else 'MISSING'}] {label}")

    addr = iap_gclb._run(
        [*iap_gclb._compute(stack, "addresses", "describe", stack.address, "--global"), "--format=value(address)"],
        check=False,
        capture=True,
    )
    if addr.returncode == 0 and addr.stdout.strip():
        click.echo(f"  reserved IP : {addr.stdout.strip()}")
    cert_state = iap_gclb._run(
        [
            *iap_gclb._compute(stack, "ssl-certificates", "describe", stack.cert, "--global"),
            "--format=value(managed.status)",
        ],
        check=False,
        capture=True,
    )
    if cert_state.returncode == 0 and cert_state.stdout.strip():
        click.echo(f"  cert state  : {cert_state.stdout.strip()}")
    audience = iap_gclb.discover_signed_header_audience(stack)
    if audience:
        click.echo(f"  iap jwt aud : {audience}  (auth.iap.signed_header_audience)")


@iap.command()
@click.option("--release-ip", is_flag=True, help="Also release the reserved static IP (breaks the DNS A record)")
@click.option("--dry-run", is_flag=True, help="Trace gcloud commands without running them")
@click.pass_context
def teardown(ctx: click.Context, release_ip: bool, dry_run: bool) -> None:
    """Delete the LB stack in dependency order. Keeps the static IP by default so a
    redeploy reuses the DNS record. OAuth clients are Console-managed and untouched."""
    stack = _stack(ctx)

    def _delete(label: str, cmd) -> None:
        logging.getLogger("iap-gclb").info("→ deleting %s", label)
        result = iap_gclb._run([*cmd, "--quiet"], dry_run=dry_run, check=False, capture=True)
        if not dry_run and result.returncode != 0:
            logging.getLogger("iap-gclb").info("  (skip: %s missing or already deleted)", label)

    c = iap_gclb._compute
    _delete(stack.forwarding_rule, c(stack, "forwarding-rules", "delete", stack.forwarding_rule, "--global"))
    _delete(stack.https_proxy, c(stack, "target-https-proxies", "delete", stack.https_proxy, "--global"))
    _delete(stack.url_map, c(stack, "url-maps", "delete", stack.url_map, "--global"))
    _delete(stack.backend, c(stack, "backend-services", "delete", stack.backend, "--global"))
    _delete(stack.neg, c(stack, "network-endpoint-groups", "delete", stack.neg, f"--zone={stack.zone}"))
    _delete(stack.health_check, c(stack, "health-checks", "delete", stack.health_check, "--global"))
    _delete(stack.cert, c(stack, "ssl-certificates", "delete", stack.cert, "--global"))
    _delete(stack.allow_firewall, c(stack, "firewall-rules", "delete", stack.allow_firewall))
    _delete(stack.deny_firewall, c(stack, "firewall-rules", "delete", stack.deny_firewall))

    if release_ip:
        _delete(stack.address, c(stack, "addresses", "delete", stack.address, "--global"))
    else:
        click.echo(f"Kept static IP {stack.address}; pass --release-ip to release it.")


# --------------------------------------------------------------------------- #
# admin iam — service accounts + project IAM
# --------------------------------------------------------------------------- #


def _iam_inputs(cfg: ClusterConfig) -> tuple[str, str, str, tuple[str, ...], str | None]:
    """Derive (project, controller_sa_id, worker_sa_id, operators, ci_principal) from config.

    Raises:
        click.ClickException: if the cluster is not a GCP cluster, or has no
            ``provisioning.iam``.
    """
    try:
        project = cfg.gcp.project
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    prov = cfg.provisioning
    iam_cfg = prov.iam if prov is not None else None
    if iam_cfg is None:
        raise click.ClickException(f"cluster {cfg.name!r} has no provisioning.iam — nothing to reconcile")
    return (
        project,
        iam_cfg.controller_service_account,
        iam_cfg.worker_service_account,
        iam_cfg.operators,
        iam_cfg.ci_principal,
    )


@admin.group(name="iam")
def iam_group() -> None:
    """Create service accounts and reconcile project IAM bindings."""


@iam_group.command()
@click.option("--dry-run", is_flag=True, help="Print planned actions without changing IAM")
@click.pass_context
def init(ctx: click.Context, dry_run: bool) -> None:
    """Create service accounts and wire IAM bindings to desired state."""
    project, controller_sa_id, worker_sa_id, operators, ci_principal = _iam_inputs(_load(ctx))
    operator_members = tuple(iam._principal_member(o) for o in operators)
    iam.reconcile_iam(project, controller_sa_id, worker_sa_id, operator_members, ci_principal, dry_run=dry_run)


@iam_group.command()
@click.option("--dry-run", is_flag=True, help="Print planned actions without changing IAM")
@click.pass_context
def reconcile(ctx: click.Context, dry_run: bool) -> None:
    """Re-apply the desired-state IAM bindings (alias of init; both are idempotent)."""
    project, controller_sa_id, worker_sa_id, operators, ci_principal = _iam_inputs(_load(ctx))
    operator_members = tuple(iam._principal_member(o) for o in operators)
    iam.reconcile_iam(project, controller_sa_id, worker_sa_id, operator_members, ci_principal, dry_run=dry_run)


# --------------------------------------------------------------------------- #
# admin user — per-user lifecycle (check + create)
# --------------------------------------------------------------------------- #


@admin.group()
def user() -> None:
    """Per-user access: check existing access or grant a new user."""


@user.command()
@click.option(
    "--probe-gcs-path",
    multiple=True,
    help="Optional gs:// bucket or path to verify worker-SA bucket metadata access, e.g. gs://marin-us-east5",
)
@click.argument("email")
@click.pass_context
def check(ctx: click.Context, probe_gcs_path: tuple[str, ...], email: str) -> None:
    """Check whether EMAIL has the IAM bindings and live credentials to use Iris."""
    project, controller_sa_id, worker_sa_id, _operators, _ci = _iam_inputs(_load(ctx))
    iam.check_user(project, controller_sa_id, worker_sa_id, probe_gcs_path, email)


@user.command()
@click.argument("email")
@click.option("--budget-limit", required=True, type=int, help="Iris budget limit (0 = unlimited)")
@click.option(
    "--max-band",
    required=True,
    type=click.Choice(_PRIORITY_BAND_NAMES),
    help="Highest priority band this user may submit to",
)
@click.option("--dry-run", is_flag=True, help="Print planned actions without changing IAM or Iris policy")
@click.pass_context
def create(ctx: click.Context, email: str, budget_limit: int, max_band: str, dry_run: bool) -> None:
    """Grant EMAIL full access: GCP IAM binding, IAP allowlist, and an Iris budget.

    Idempotent and desired-state: re-running reconciles each layer. The Iris
    policy step needs ``marin-cluster[iris]`` and a logged-in session
    (``marin-cluster login``) so the controller call carries credentials.
    """
    cfg = _load(ctx)
    project, controller_sa_id, worker_sa_id, _operators, _ci = _iam_inputs(cfg)
    member = iam._principal_member(email)

    # 1. GCP IAM: project + service-account bindings for the user.
    iam.bind_user(project, controller_sa_id, worker_sa_id, member, dry_run=dry_run)

    # 2. IAP allowlist: roles/iap.httpsResourceAccessor on the backend.
    iap_gclb.grant_access(Stack.from_config(cfg), member, dry_run=dry_run)

    # 3. Iris policy: the user's budget + max band on the controller.
    if dry_run:
        click.echo(f"[dry-run] would set Iris budget for {email}: limit={budget_limit}, max_band={max_band}")
    else:
        _set_iris_budget(cfg, email, budget_limit, max_band)
        click.echo(f"Iris budget set for {email}: limit={budget_limit}, max_band={max_band}")

    click.echo(f"User {email} provisioned (IAM + IAP + Iris policy).")


def _set_iris_budget(cfg: ClusterConfig, email: str, budget_limit: int, max_band: str) -> None:
    """Set ``email``'s budget on the cluster's Iris controller.

    Resolves the controller endpoint from the cluster's IAP url and the bearer
    material from the unified ``marin-cluster login`` credential store, so the
    call carries both the Iris app JWT and the IAP proxy token.
    """
    try:
        from iris.rpc import controller_pb2  # noqa: PLC0415  (optional iris install)
        from iris.rpc.controller_connect import ControllerServiceClientSync  # noqa: PLC0415
        from iris.rpc.proto_display import priority_band_value  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("setting an Iris budget needs iris; install `marin-cluster[iris]`") from e

    from rigging.credentials import credentials_for  # noqa: PLC0415

    auth = cfg.auth
    if auth.iap is None or not auth.iap.url:
        raise click.ClickException(
            f"cluster {cfg.name!r} has no auth.iap.url; the Iris budget step needs an IAP-fronted controller"
        )
    credentials = credentials_for(cfg.name, auth)
    if credentials.token_provider is None:
        raise click.ClickException(f"no cached credentials for cluster {cfg.name!r}; run `marin-cluster login` first")
    client = ControllerServiceClientSync(auth.iap.url, interceptors=credentials.interceptors())
    with client:
        client.set_user_budget(
            controller_pb2.Controller.SetUserBudgetRequest(
                user_id=email,
                budget_limit=budget_limit,
                max_band=priority_band_value(max_band),
            )
        )


# --------------------------------------------------------------------------- #
# admin tunnel — the SSH/IAP escape hatch (debug only)
# --------------------------------------------------------------------------- #


@admin.command()
@click.pass_context
def tunnel(ctx: click.Context) -> None:
    """Print (do not open) the gcloud IAP-TCP tunnel command for the controller VM.

    The supported path to the controller is the IAP-fronted HTTPS endpoint; this
    raw tunnel is a debug/admin-only escape hatch, so it is printed for a human to
    run deliberately rather than opened implicitly.
    """
    stack = _stack(ctx)
    controller_name = iap_gclb.discover_controller_name(stack)
    click.echo("# debug/admin only — the supported controller path is the IAP HTTPS endpoint")
    click.echo(
        "gcloud compute start-iap-tunnel "
        f"{controller_name} {stack.controller_port} "
        f"--project={stack.project} --zone={stack.zone} "
        f"--local-host-port=localhost:{stack.controller_port}"
    )
