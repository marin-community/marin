# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level Iris CLI entry point.

Defines the ``iris`` Click group and registers all subcommands.
"""

import logging as _logging_module
import sys
from pathlib import Path

import click
from rigging.config_discovery import resolve_cluster_config
from rigging.log_setup import configure_logging

from iris.cli.token_store import cluster_name_from_url, load_any_token, load_token, store_token
from iris.rpc import config_pb2, job_pb2
from iris.rpc import controller_pb2 as _controller_pb2
from iris.rpc.auth import AuthTokenInjector, GcpAccessTokenProvider, StaticTokenProvider, TokenProvider
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.proto_utils import PRIORITY_BAND_NAMES, priority_band_name, priority_band_value

logger = _logging_module.getLogger(__name__)


def _bundled_iris_examples_dir() -> str | None:
    """Return the iris package's bundled examples/ dir when it ships on disk.

    Probes two layouts because the examples directory can physically live in
    two places depending on how iris was installed:

    1. Wheel installs (site-packages): hatchling force-include places the
       yamls at ``iris/examples/`` inside the package. Resolve that via
       ``Path(__file__).parent.parent / "examples"``.
    2. Editable workspace installs: the yamls stay at their source location
       ``lib/iris/examples/`` — reachable via ``parents[3] / "examples"`` from
       ``lib/iris/src/iris/cli/main.py``.

    Returns the first directory that exists, or ``None`` for wheel installs
    that don't ship examples at all.
    """
    here = Path(__file__).resolve()
    # Wheel install: examples is a sibling of cli/ inside the iris package.
    wheel_path = here.parent.parent / "examples"
    if wheel_path.is_dir():
        return str(wheel_path)
    # Editable install: examples lives at lib/iris/examples/ (parents[3]).
    editable_path = here.parents[3] / "examples"
    if editable_path.is_dir():
        return str(editable_path)
    return None


# Directories searched (in priority order) to resolve ``--cluster=<name>`` to
# a YAML config file. Relative paths are resolved against the marin project
# root by ``rigging.config_discovery``; absolute paths are used as-is.
IRIS_CLUSTER_CONFIG_DIRS: tuple[str, ...] = tuple(
    p
    for p in (
        "~/.config/marin/clusters",  # user override — checked first
        "lib/iris/examples",  # in-tree marin checkout
        _bundled_iris_examples_dir(),  # editable install from sibling workspace
    )
    if p is not None
)


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
    raise ValueError(f"Unknown auth provider: {provider}")


def _configure_client_s3(config) -> None:
    """Configure S3 env vars for fsspec access. Delegates to the canonical implementation."""
    from iris.cluster.providers.k8s.controller import configure_client_s3

    configure_client_s3(config)


def rpc_client(
    address: str,
    token_provider: TokenProvider | None = None,
    timeout_ms: int = 30_000,
) -> ControllerServiceClientSync:
    """Create an RPC client with optional auth. Use as a context manager: ``with rpc_client(url) as c:``."""
    interceptors = [AuthTokenInjector(token_provider)] if token_provider else []
    return ControllerServiceClientSync(address, timeout_ms=timeout_ms, interceptors=interceptors)


def require_controller_url(ctx: click.Context) -> str:
    """Get controller_url from context, establishing a tunnel lazily if needed.

    On first call with a --config, this establishes the tunnel to the controller
    and caches the result. Subsequent calls return the cached URL.
    Commands that don't call this (e.g. ``cluster start``) never pay tunnel cost.
    """
    controller_url = ctx.obj.get("controller_url") if ctx.obj else None
    if controller_url:
        return controller_url

    # Lazy tunnel establishment from config
    config = ctx.obj.get("config") if ctx.obj else None
    if config:
        from iris.cluster.config import IrisConfig

        iris_config = IrisConfig(config)
        bundle = iris_config.provider_bundle()
        ctx.obj["provider_bundle"] = bundle

        if iris_config.proto.controller.WhichOneof("controller") == "local":
            from iris.cluster.providers.local.cluster import LocalCluster

            cluster = LocalCluster(iris_config.proto)
            controller_address = cluster.start()
            ctx.call_on_close(cluster.close)
        else:
            controller_address = iris_config.controller_address()
            if not controller_address:
                controller_address = bundle.controller.discover_controller(iris_config.proto.controller)

        # Establish tunnel and keep it alive for command duration
        try:
            logger.info("Establishing tunnel to controller...")
            tunnel_cm = bundle.controller.tunnel(address=controller_address)
            tunnel_url = tunnel_cm.__enter__()
            ctx.obj["controller_url"] = tunnel_url
            # Clean up tunnel when context closes
            ctx.call_on_close(lambda: tunnel_cm.__exit__(None, None, None))
            return tunnel_url
        except Exception as e:
            raise click.ClickException(f"Could not connect to controller: {e}") from e

    config_file = ctx.obj.get("config_file") if ctx.obj else None
    if config_file:
        raise click.ClickException(
            f"Could not connect to controller (config: {config_file}). "
            "Check that the controller is running and reachable."
        )
    raise click.ClickException(
        "No controller specified. Pass --cluster=<name> (see `iris cluster list`), --controller-url, or --config."
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--traceback", "show_traceback", is_flag=True, help="Show full stack traces on errors")
@click.option("--controller-url", help="Controller URL (e.g., http://localhost:10000)")
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config file")
@click.option(
    "--cluster",
    "cluster_name",
    default=None,
    help="Cluster name (resolves config automatically) or used for token lookup",
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
        configure_logging(level=_logging_module.DEBUG)
    else:
        configure_logging(level=_logging_module.INFO)

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
        from iris.cluster.config import IrisConfig

        iris_config = IrisConfig.load(config_file)
        ctx.obj["config"] = iris_config.proto
        ctx.obj["config_file"] = config_file
        _configure_client_s3(iris_config.proto)

        name = resolve_cluster_name(iris_config.proto, controller_url, cluster_name)
        ctx.obj["cluster_name"] = name

        if iris_config.proto.HasField("auth"):
            ctx.obj["token_provider"] = create_client_token_provider(iris_config.proto.auth, cluster_name=name)
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

    cluster_name = ctx.obj.get("cluster_name", "default")
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
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider")

    with rpc_client(controller_url, token_provider) as client:
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
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider")

    with rpc_client(controller_url, token_provider) as client:
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
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider")

    with rpc_client(controller_url, token_provider) as client:
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
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider")

    with rpc_client(controller_url, token_provider) as client:
        client.set_user_budget(
            _controller_pb2.Controller.SetUserBudgetRequest(
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
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider")

    with rpc_client(controller_url, token_provider) as client:
        resp = client.get_user_budget(_controller_pb2.Controller.GetUserBudgetRequest(user_id=user_id))

    click.echo(f"User:      {resp.user_id}")
    click.echo(f"Limit:     {resp.budget_limit}")
    click.echo(f"Spent:     {resp.budget_spent}")
    click.echo(f"Max band:  {priority_band_name(resp.max_band)}")


@budget.command("list")
@click.pass_context
def budget_list(ctx):
    """List all user budgets with current spend."""
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider")

    with rpc_client(controller_url, token_provider) as client:
        resp = client.list_user_budgets(_controller_pb2.Controller.ListUserBudgetsRequest())

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
