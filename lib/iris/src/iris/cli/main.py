# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level Iris CLI entry point.

Defines the ``iris`` Click group and registers all subcommands.
"""

import json
import logging as _logging_module
import os
import sys

import click

from iris.cli.token_store import cluster_name_from_url, load_any_token, load_token, store_token
from iris.logging import configure_logging
from iris.rpc import config_pb2
from iris.rpc.auth import GcpAccessTokenProvider, StaticTokenProvider, TokenProvider

logger = _logging_module.getLogger(__name__)


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
    """Configure S3 env vars for fsspec access (e.g. bundle downloads).

    fsspec needs AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (mapped from
    R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY), AWS_ENDPOINT_URL, and FSSPEC_S3
    with the correct endpoint.
    """
    from iris.cluster.platform.coreweave import _needs_virtual_host_addressing

    endpoint = config.platform.coreweave.object_storage_endpoint
    if not endpoint:
        return

    r2_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if r2_key and r2_secret:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", r2_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", r2_secret)

    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)

    if "FSSPEC_S3" not in os.environ:
        fsspec_conf: dict = {"endpoint_url": endpoint}
        if _needs_virtual_host_addressing(endpoint):
            fsspec_conf["config_kwargs"] = {"s3": {"addressing_style": "virtual"}}
        os.environ["FSSPEC_S3"] = json.dumps(fsspec_conf)


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
        platform = iris_config.platform()
        ctx.obj["platform"] = platform

        if iris_config.proto.controller.WhichOneof("controller") == "local":
            from iris.cluster.local_cluster import LocalCluster

            cluster = LocalCluster(iris_config.proto)
            controller_address = cluster.start()
            ctx.call_on_close(cluster.close)
        else:
            controller_address = iris_config.controller_address()
            if not controller_address:
                controller_address = platform.discover_controller(iris_config.proto.controller)

        # Establish tunnel and keep it alive for command duration
        try:
            logger.info("Establishing tunnel to controller...")
            tunnel_cm = platform.tunnel(address=controller_address)
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
    raise click.ClickException("Either --controller-url or --config is required")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--traceback", "show_traceback", is_flag=True, help="Show full stack traces on errors")
@click.option("--controller-url", help="Controller URL (e.g., http://localhost:10000)")
@click.option("--config", "config_file", type=click.Path(exists=True), help="Cluster config file")
@click.option("--cluster", "cluster_name", default=None, help="Cluster name for token lookup")
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

    from iris.rpc import cluster_pb2
    from iris.rpc.cluster_connect import ControllerServiceClientSync

    if config and config.HasField("auth"):
        provider = config.auth.WhichOneof("provider")
    else:
        client = ControllerServiceClientSync(address=controller_url, timeout_ms=30000)
        try:
            auth_info = client.get_auth_info(cluster_pb2.GetAuthInfoRequest())
        except Exception as e:
            raise click.ClickException(f"Failed to discover auth method: {e}") from e
        finally:
            client.close()
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
    client = ControllerServiceClientSync(address=controller_url, timeout_ms=30000)
    try:
        response = client.login(cluster_pb2.LoginRequest(identity_token=identity_token))
    except Exception as e:
        raise click.ClickException(f"Login failed: {e}") from e
    finally:
        client.close()

    cluster_name = ctx.obj.get("cluster_name", "default")
    store_token(cluster_name, controller_url, response.token)

    click.echo(f"Authenticated as {response.user_id}")
    # Token in URL is visible in browser history/logs — acceptable for internal clusters
    click.echo(f"Dashboard: {controller_url}?session_token={response.token}")
    click.echo(f"Token stored for cluster '{cluster_name}'")


def _make_authenticated_client(controller_url: str, token_provider: TokenProvider | None):
    """Create a ControllerServiceClientSync with auth interceptor if available."""
    from iris.rpc.auth import AuthTokenInjector
    from iris.rpc.cluster_connect import ControllerServiceClientSync

    interceptors = [AuthTokenInjector(token_provider)] if token_provider else []
    return ControllerServiceClientSync(address=controller_url, timeout_ms=30000, interceptors=interceptors)


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

    from iris.rpc import cluster_pb2

    client = _make_authenticated_client(controller_url, token_provider)
    try:
        response = client.create_api_key(cluster_pb2.CreateApiKeyRequest(user_id=user_id, name=name, ttl_ms=ttl_ms))
    finally:
        client.close()

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

    from iris.rpc import cluster_pb2

    client = _make_authenticated_client(controller_url, token_provider)
    try:
        response = client.list_api_keys(cluster_pb2.ListApiKeysRequest(user_id=user_id))
    finally:
        client.close()

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

    from iris.rpc import cluster_pb2

    client = _make_authenticated_client(controller_url, token_provider)
    try:
        client.revoke_api_key(cluster_pb2.RevokeApiKeyRequest(key_id=key_id))
    finally:
        client.close()

    click.echo(f"Revoked key: {key_id}")


# Register subcommand groups — imported at module level to ensure they are
# always available when the ``iris`` group is used.
from iris.cli.build import build  # noqa: E402
from iris.cli.cluster import cluster  # noqa: E402
from iris.cli.job import job  # noqa: E402
from iris.cli.process_status import register_process_status_commands  # noqa: E402
from iris.cli.query import query_cmd  # noqa: E402
from iris.cli.rpc import register_rpc_commands  # noqa: E402

iris.add_command(cluster)
iris.add_command(build)
iris.add_command(job)
iris.add_command(query_cmd)
register_rpc_commands(iris)
register_process_status_commands(iris)
