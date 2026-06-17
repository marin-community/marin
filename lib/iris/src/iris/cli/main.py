# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level Iris CLI entry point.

Defines the ``iris`` Click group and registers all subcommands.
"""

import logging
import sys

import click
from rigging.config_discovery import resolve_cluster_config
from rigging.log_setup import configure_logging

from iris.cli.connect import (
    IRIS_CLUSTER_CONFIG_DIRS,
    build_iap_provider,
    iap_config,
    require_controller_url,
    rpc_client,
    rpc_client_for_ctx,
)
from iris.cluster.backends.k8s.controller import configure_client_s3
from iris.cluster.config import IrisConfig
from iris.cluster.token_store import cluster_name_from_url, load_any_token, load_token, store_token
from iris.rpc import config_pb2, controller_pb2, job_pb2
from iris.rpc.auth import GcpAccessTokenProvider, StaticTokenProvider, TokenProvider, run_iap_desktop_login
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_name, priority_band_value

logger = logging.getLogger(__name__)


def resolve_cluster_name(
    config: config_pb2.IrisClusterConfig | None,
    controller_url: str | None,
    cli_cluster_name: str | None,
) -> str:
    if cli_cluster_name:
        return cli_cluster_name
    if config and config.name:
        return config.name
    if config and config.controller.WhichOneof("controller") == "local":
        return "local"
    if controller_url:
        return cluster_name_from_url(controller_url)
    return "default"


def create_client_token_provider(
    auth_config: config_pb2.AuthConfig, cluster_name: str = "default"
) -> TokenProvider | None:
    """Create a TokenProvider from an AuthConfig proto for CLI usage.

    Checks the named-cluster token store first (from ``iris login``),
    then falls back to config-based token providers.
    """
    credential = load_token(cluster_name)
    if credential is None:
        credential = load_any_token()
    if credential is not None:
        return StaticTokenProvider(credential.token)

    provider = auth_config.WhichOneof("provider")
    if provider is None:
        return None
    if provider == "gcp":
        return GcpAccessTokenProvider()
    elif provider == "static":
        tokens = dict(auth_config.static.tokens)
        if not tokens:
            raise ValueError("Static auth config requires at least one token")
        first_token = next(iter(tokens))
        return StaticTokenProvider(first_token)
    elif provider == "iap":
        # The Iris JWT for an IAP cluster comes only from the token store after
        # `iris login` (handled above). With none cached yet there is no
        # config-derived fallback — return None so commands fail clearly until
        # the user logs in.
        return None
    raise ValueError(f"Unknown auth provider: {provider}")


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
        iris_config = IrisConfig.load(config_file)
        ctx.obj["config"] = iris_config.proto
        ctx.obj["config_file"] = config_file
        _configure_client_s3(iris_config.proto)

        name = resolve_cluster_name(iris_config.proto, controller_url, cluster_name)
        ctx.obj["cluster_name"] = name

        if iris_config.proto.HasField("auth"):
            ctx.obj["token_provider"] = create_client_token_provider(iris_config.proto.auth, cluster_name=name)

        # For an IAP-fronted cluster, also attach the IAP ID-token provider so
        # every RPC carries the Proxy-Authorization header IAP requires.
        iap = iap_config(iris_config.proto)
        if iap is not None:
            ctx.obj["iap_provider"] = build_iap_provider(iap, name)
    else:
        name = resolve_cluster_name(None, controller_url, cluster_name)
        ctx.obj["cluster_name"] = name

        # Load stored token from `iris login` when no config is available
        credential = load_token(name)
        if credential is None:
            credential = load_any_token()
        if credential is not None:
            ctx.obj["token_provider"] = StaticTokenProvider(credential.token)

    # Store direct controller URL; tunnel from config is established lazily
    # in require_controller_url() so commands like ``cluster start`` don't block.
    if controller_url:
        ctx.obj["controller_url"] = controller_url


@iris.command()
@click.pass_context
def login(ctx):
    """Authenticate with the cluster and store a JWT locally."""
    controller_url = require_controller_url(ctx)
    config = ctx.obj.get("config")
    cluster_name = ctx.obj.get("cluster_name", "default")

    # IAP-fronted cluster: two-step login. First authenticate to IAP via the
    # desktop OAuth flow (browser), then exchange the resulting OIDC ID token for
    # an Iris JWT over the IAP transport. The ID token rides in Proxy-Authorization
    # so IAP admits the exchange request to the controller.
    iap = iap_config(config)
    if iap is not None:
        if not iap.oauth_client_id or not iap.oauth_client_secret:
            raise click.ClickException("IAP auth config is missing oauth_client_id/oauth_client_secret")
        click.echo("Opening browser to authenticate with Google IAP...")
        try:
            id_token, refresh_token = run_iap_desktop_login(iap.oauth_client_id, iap.oauth_client_secret)
        except Exception as e:
            raise click.ClickException(f"IAP authentication failed: {e}") from e

        iap_provider = StaticTokenProvider(id_token)
        with rpc_client(controller_url, iap_provider=iap_provider) as client:
            try:
                response = client.login(job_pb2.LoginRequest(identity_token=id_token))
            except Exception as e:
                raise click.ClickException(f"Login failed: {e}") from e

        store_token(cluster_name, controller_url, response.token, iap_refresh_token=refresh_token)
        click.echo(f"Authenticated as {response.user_id}")
        click.echo(f"Token stored for cluster '{cluster_name}' (IAP + Iris credentials cached)")
        return

    if config and config.HasField("auth"):
        provider = config.auth.WhichOneof("provider")
    else:
        with rpc_client(controller_url) as client:
            try:
                auth_info = client.get_auth_info(job_pb2.GetAuthInfoRequest())
            except Exception as e:
                raise click.ClickException(f"Failed to discover auth method: {e}") from e
        provider = auth_info.provider or None
        if not provider:
            raise click.ClickException("Controller has no authentication configured")

    if provider == "gcp":
        gcp_provider = GcpAccessTokenProvider()
        try:
            identity_token = gcp_provider.get_token()
        except Exception as e:
            raise click.ClickException(f"Failed to get GCP access token: {e}") from e
    elif provider == "static":
        if not config:
            raise click.ClickException("Static auth requires --config (tokens are in the config file)")
        tokens = dict(config.auth.static.tokens)
        if not tokens:
            raise click.ClickException("No static tokens configured")
        identity_token = next(iter(tokens))
    else:
        raise click.ClickException(f"Unsupported auth provider: {provider}")

    # All providers converge: exchange identity_token for JWT via Login RPC
    with rpc_client(controller_url) as client:
        try:
            response = client.login(job_pb2.LoginRequest(identity_token=identity_token))
        except Exception as e:
            raise click.ClickException(f"Login failed: {e}") from e

    store_token(cluster_name, controller_url, response.token)

    click.echo(f"Authenticated as {response.user_id}")
    # Token in URL is visible in browser history/logs — acceptable for internal clusters
    click.echo(f"Dashboard: {controller_url}/auth/session_bootstrap?token={response.token}")
    click.echo(f"Token stored for cluster '{cluster_name}'")


@iris.group()
@click.pass_context
def key(ctx):
    """Manage API keys."""
    pass


@key.command("create")
@click.option("--name", required=True, help="Human-readable key name")
@click.option("--user", "user_id", default="", help="Target user (admin only for other users)")
@click.option("--ttl", "ttl_ms", default=0, type=int, help="Time-to-live in milliseconds (0 = no expiry)")
@click.pass_context
def key_create(ctx, name: str, user_id: str, ttl_ms: int):
    """Create a new API key."""
    with rpc_client_for_ctx(ctx) as client:
        response = client.create_api_key(job_pb2.CreateApiKeyRequest(user_id=user_id, name=name, ttl_ms=ttl_ms))

    click.echo(f"Key ID:  {response.key_id}")
    click.echo(f"Token:   {response.token}")
    click.echo(f"Prefix:  {response.key_prefix}")
    click.echo("Store this token securely — it cannot be retrieved again.")


@key.command("list")
@click.option("--user", "user_id", default="", help="Filter by user (admin only for other users)")
@click.pass_context
def key_list(ctx, user_id: str):
    """List API keys."""
    with rpc_client_for_ctx(ctx) as client:
        response = client.list_api_keys(job_pb2.ListApiKeysRequest(user_id=user_id))

    if not response.keys:
        click.echo("No API keys found.")
        return

    for k in response.keys:
        status = "REVOKED" if k.revoked else "active"
        click.echo(f"  {k.key_id}  {k.key_prefix}...  {k.name:<20s}  {k.user_id:<20s}  {status}")


@key.command("revoke")
@click.argument("key_id")
@click.pass_context
def key_revoke(ctx, key_id: str):
    """Revoke an API key."""
    with rpc_client_for_ctx(ctx) as client:
        client.revoke_api_key(job_pb2.RevokeApiKeyRequest(key_id=key_id))

    click.echo(f"Revoked key: {key_id}")


# ---------------------------------------------------------------------------
# User budget management
# ---------------------------------------------------------------------------


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
from iris.cli.build import build  # noqa: E402
from iris.cli.cluster import cluster  # noqa: E402
from iris.cli.job import job  # noqa: E402
from iris.cli.process_status import register_process_status_commands  # noqa: E402
from iris.cli.query import query_cmd  # noqa: E402
from iris.cli.rpc import register_rpc_commands  # noqa: E402
from iris.cli.task import task  # noqa: E402

iris.add_command(actor_cmd)
iris.add_command(cluster)
iris.add_command(build)
iris.add_command(job)
iris.add_command(task)
iris.add_command(query_cmd)
register_rpc_commands(iris)
register_process_status_commands(iris)
