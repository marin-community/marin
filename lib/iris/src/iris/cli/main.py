# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level Iris CLI entry point.

Defines the ``iris`` Click group and registers all subcommands.
"""

import base64
import logging
import sys

import click
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT, IapCredentialsUnavailable, run_iap_desktop_login
from rigging.config_discovery import resolve_cluster_config
from rigging.credential_store import CredentialRecord, save_credentials
from rigging.log_setup import configure_logging

from iris.cli.connect import (
    IRIS_CLUSTER_CONFIG_DIRS,
    client_credentials,
    iap_config,
    require_controller_url,
    resolve_cluster_name,
    rpc_client_for_ctx,
)
from iris.cluster.config import load_config
from iris.cluster.platforms.k8s.controller import configure_client_s3
from iris.cluster.platforms.k8s.service import CloudK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.rpc import controller_pb2
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_name, priority_band_value

logger = logging.getLogger(__name__)


def _configure_client_s3(config) -> None:
    """Configure S3 env vars for fsspec access. Delegates to the canonical implementation."""
    configure_client_s3(config)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--traceback", "show_traceback", is_flag=True, help="Show full stack traces on errors")
@click.option("--controller-url", help="Controller URL (e.g., http://localhost:10000)")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    help="Exact cluster config YAML path; use for custom configs or pinned files",
)
@click.option(
    "--cluster",
    "cluster_name",
    default=None,
    help="Named cluster to resolve from config search paths; preferred for known clusters",
)
@click.pass_context
def iris(
    ctx,
    verbose: bool,
    show_traceback: bool,
    controller_url: str | None,
    config_file: str | None,
    cluster_name: str | None,
):
    """Iris cluster management."""
    ctx.ensure_object(dict)
    ctx.obj["traceback"] = show_traceback
    ctx.obj["verbose"] = verbose
    ctx.obj["cluster_name"] = cluster_name

    if verbose:
        configure_logging(level=logging.DEBUG)
    else:
        configure_logging(level=logging.INFO)

    # Resolve cluster name to config file if no explicit config or URL given
    if cluster_name and not config_file and not controller_url:
        try:
            resolved = resolve_cluster_config(cluster_name, dirs=IRIS_CLUSTER_CONFIG_DIRS)
            logger.info("Resolved cluster %r to config: %s", cluster_name, resolved)
            config_file = str(resolved)
        except FileNotFoundError:
            raise click.UsageError(
                f"Unknown cluster {cluster_name!r}. Run `iris cluster list` to see available clusters."
            ) from None

    # Validate mutually exclusive options
    if controller_url and config_file:
        raise click.UsageError("Cannot specify both --controller-url and --config")

    # Skip expensive operations when showing help or doing shell completion.
    # Only check for help flags before "--" to avoid matching help flags
    # intended for the user's command (e.g., "job run -- python script.py --help").
    argv_before_separator = sys.argv[: sys.argv.index("--")] if "--" in sys.argv else sys.argv
    if ctx.resilient_parsing or "--help" in argv_before_separator or "-h" in argv_before_separator:
        return

    # Load config if provided
    if config_file:
        config = load_config(config_file)
        ctx.obj["config"] = config
        ctx.obj["config_file"] = config_file
        _configure_client_s3(config)

        name = resolve_cluster_name(config, controller_url, cluster_name)
        ctx.obj["cluster_name"] = name
        ctx.obj["credentials"] = client_credentials(config, name)
    else:
        name = resolve_cluster_name(None, controller_url, cluster_name)
        ctx.obj["cluster_name"] = name
        ctx.obj["credentials"] = client_credentials(None, name)

    # Store direct controller URL; tunnel from config is established lazily
    # in require_controller_url() so commands like ``cluster start`` don't block.
    if controller_url:
        ctx.obj["controller_url"] = controller_url


@iris.command()
@click.option(
    "--headless",
    is_flag=True,
    help="Print the authorization URL and read back the pasted code instead of opening a browser. "
    "Use on a remote box whose localhost is unreachable from your browser; auto-detected when no browser is registered.",
)
@click.pass_context
def login(ctx, headless):
    """Authenticate to the cluster's IAP edge via the browser and cache the refresh token.

    Pure-IAP: the controller mints no token. This runs the desktop OAuth browser
    flow and caches the long-lived IAP edge refresh token, from which each later
    RPC silently re-mints the short-lived edge token IAP requires. Non-IAP
    clusters need no login (in-network / loopback trust admits the caller).

    On a box with no local browser but where you can open a URL elsewhere, pass
    ``--headless`` (auto-detected when no browser is registered) to authenticate by
    pasting the returned code. Fully unattended callers (CI / agents, no human at
    all) skip login entirely and instead give the process service-account
    credentials on the cluster's IAP allowlist — e.g. ``gcloud auth
    application-default login --impersonate-service-account=<sa>``; see
    ``lib/iris/docs/iap-gclb.md``.
    """
    controller_url = require_controller_url(ctx)
    config = ctx.obj.get("config")
    cluster_name = ctx.obj.get("cluster_name", "default")

    iap = iap_config(config)
    if iap is None:
        raise click.ClickException(
            "This cluster is not fronted by IAP; no login is needed (in-network / loopback trust applies)."
        )

    # Config may front a cluster-specific desktop client; otherwise the Marin
    # desktop client shipped in rigging drives the flow (same fallback as
    # rigging.credentials._desktop_client).
    client_id = iap.oauth_client_id or MARIN_DESKTOP_OAUTH_CLIENT.client_id
    client_secret = iap.oauth_client_secret or MARIN_DESKTOP_OAUTH_CLIENT.client_secret
    if headless:
        click.echo("Authenticating with Google IAP (headless: open the printed URL, then paste the result)...")
    else:
        click.echo("Opening browser to authenticate with Google IAP...")
    try:
        _id_token, refresh_token = run_iap_desktop_login(client_id, client_secret, headless=headless)
    except Exception as e:
        raise click.ClickException(
            f"IAP authentication failed: {e}\n"
            "If you are fully unattended (CI / agent), skip `iris login` and give the "
            "process service-account credentials on the cluster's IAP allowlist, e.g. "
            "`gcloud auth application-default login --impersonate-service-account=<sa>`. "
            "See lib/iris/docs/iap-gclb.md."
        ) from e

    save_credentials(
        CredentialRecord(
            cluster=cluster_name,
            endpoint=controller_url,
            edge_refresh_token=refresh_token,
        )
    )
    click.echo(f"IAP edge credentials cached for cluster '{cluster_name}'.")
    click.echo("The controller authenticates each request via IAP; no cluster token is minted.")


# ---------------------------------------------------------------------------
# User budget management
# ---------------------------------------------------------------------------


@iris.command("operator-login")
@click.option("--principal", required=True, help="Identity recorded for this Kubernetes operator session")
@click.pass_context
def operator_login(ctx: click.Context, principal: str) -> None:
    """Cache a one-hour admin JWT using existing Kubernetes controller-Pod exec access."""
    config = ctx.obj.get("config")
    if config is None or config.credential_broker is None or config.kubernetes_provider is None:
        raise click.ClickException("operator-login requires a broker-enabled Kubernetes cluster config")
    kp = config.kubernetes_provider
    platform = config.platform.coreweave
    assert platform is not None
    kubernetes = CloudK8sService(
        namespace=kp.namespace or "iris",
        kubeconfig_path=platform.kubeconfig_path or kp.kubeconfig or None,
        context=platform.kube_context or kp.kube_context or None,
    )
    pods = kubernetes.list_json(K8sResource.PODS, labels={"app": "iris-controller"})
    running = [pod for pod in pods if pod.get("status", {}).get("phase") == "Running"]
    if len(running) != 1:
        raise click.ClickException("operator-login requires exactly one running controller Pod")
    result = kubernetes.exec(
        running[0]["metadata"]["name"],
        [
            ".venv/bin/python",
            "-m",
            "iris.cluster.controller.operator_token",
            "--config",
            "/etc/iris/config.json",
            "--principal",
            principal,
        ],
        container="iris-controller",
        timeout=60,
    )
    if result.returncode != 0:
        raise click.ClickException("Controller rejected operator-token creation; inspect controller-Pod exec access")
    token = result.stdout.strip()
    if not token or len(token.split(".")) != 3:
        raise click.ClickException("Controller returned an invalid operator-token response")
    certificate = kubernetes.get_json(K8sResource.SECRETS, config.credential_broker.tls_secret)
    if certificate is None:
        raise click.ClickException("Controller TLS certificate is unavailable")
    save_credentials(
        CredentialRecord(
            cluster=ctx.obj["cluster_name"],
            endpoint=kp.controller_address,
            token=token,
            ca_bundle=base64.b64decode(certificate["data"]["ca.crt"]).decode(),
        )
    )
    click.echo("Operator credential cached for one hour. Kubernetes Pod-exec authority granted this session.")


@iris.command("task-sessions")
@click.option("--attempt", "attempt_uid", default="", help="Exact task attempt UID")
@click.option("--job", "job_id", default="", help="Job name, including its descendants")
@click.option("--principal", default="", help="Verified submitting identity")
@click.option("--action", type=click.Choice(["rollover", "disable", "enable"]), required=True)
@click.pass_context
def task_sessions(ctx: click.Context, attempt_uid: str, job_id: str, principal: str, action: str) -> None:
    """Roll or disable task credentials (admin only). Select exactly one scope."""
    request_type = controller_pb2.Controller.ManageTaskSessionsRequest
    with rpc_client_for_ctx(ctx) as client:
        response = client.manage_task_sessions(
            request_type(
                attempt_uid=attempt_uid,
                job_id=job_id,
                principal=principal,
                action=request_type.Action.Value(action.upper()),
            )
        )
    click.echo(f"Updated {response.affected} task sessions.")


@iris.group()
@click.pass_context
def user(ctx):
    """User management commands."""
    pass


@user.group()
@click.pass_context
def budget(ctx):
    """Manage user budgets."""
    pass


@budget.command("set")
@click.argument("user_id")
@click.option("--limit", "budget_limit", required=True, type=int, help="Budget limit (0 = unlimited)")
@click.option(
    "--max-band",
    required=True,
    type=click.Choice(PRIORITY_BAND_NAMES),
    help="Highest priority band this user can submit to",
)
@click.pass_context
def budget_set(ctx, user_id: str, budget_limit: int, max_band: str):
    """Set budget limit and max band for a user."""
    with rpc_client_for_ctx(ctx) as client:
        client.set_user_budget(
            controller_pb2.Controller.SetUserBudgetRequest(
                user_id=user_id,
                budget_limit=budget_limit,
                max_band=priority_band_value(max_band),
            )
        )

    click.echo(f"Budget set for {user_id}: limit={budget_limit}, max_band={max_band}")


@budget.command("get")
@click.argument("user_id")
@click.pass_context
def budget_get(ctx, user_id: str):
    """Get budget config and current spend for a user."""
    with rpc_client_for_ctx(ctx) as client:
        resp = client.get_user_budget(controller_pb2.Controller.GetUserBudgetRequest(user_id=user_id))

    click.echo(f"User:      {resp.user_id}")
    click.echo(f"Limit:     {resp.budget_limit}")
    click.echo(f"Spent:     {resp.budget_spent}")
    click.echo(f"Max band:  {priority_band_name(resp.max_band)}")


@budget.command("list")
@click.pass_context
def budget_list(ctx):
    """List all user budgets with current spend."""
    with rpc_client_for_ctx(ctx) as client:
        resp = client.list_user_budgets(controller_pb2.Controller.ListUserBudgetsRequest())

    if not resp.users:
        click.echo("No user budgets found.")
        return

    click.echo(f"{'USER':<30s} {'LIMIT':>10s} {'SPENT':>10s} {'MAX BAND':<15s}")
    for u in resp.users:
        click.echo(f"{u.user_id:<30s} {u.budget_limit:>10d} {u.budget_spent:>10d} {priority_band_name(u.max_band):<15s}")


# Register subcommand groups — imported at module level to ensure they are
# always available when the ``iris`` group is used.
from iris.cli.actor import actor as actor_cmd  # noqa: E402
from iris.cli.attempt import attempt as attempt_cmd  # noqa: E402
from iris.cli.build import build  # noqa: E402
from iris.cli.cluster import cluster  # noqa: E402
from iris.cli.endpoints import endpoints  # noqa: E402
from iris.cli.job import job  # noqa: E402
from iris.cli.process_status import register_process_status_commands  # noqa: E402
from iris.cli.query import query_cmd  # noqa: E402
from iris.cli.rpc import register_rpc_commands  # noqa: E402
from iris.cli.task import task  # noqa: E402

iris.add_command(actor_cmd)
iris.add_command(attempt_cmd)
iris.add_command(cluster)
iris.add_command(build)
iris.add_command(endpoints)
iris.add_command(job)
iris.add_command(task)
iris.add_command(query_cmd)
register_rpc_commands(iris)
register_process_status_commands(iris)


def main() -> None:
    """Console-script entry point: run the ``iris`` group with clean auth errors.

    IAP credential failures surface lazily from deep in an RPC call as a bare
    ``IapCredentialsUnavailable``. Catch it here to print an actionable hint
    instead of a stack trace; every other exception keeps its normal traceback.
    """
    try:
        iris()
    except IapCredentialsUnavailable as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        click.echo(
            "Run `iris login` for the interactive path, or for an unattended caller "
            "`gcloud auth application-default login --impersonate-service-account=<sa>`. "
            "See lib/iris/docs/iap-gclb.md.",
            err=True,
        )
        sys.exit(1)
