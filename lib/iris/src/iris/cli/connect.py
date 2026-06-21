# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI helpers used by every ``iris`` subcommand module.

This module is a leaf in the ``iris.cli`` dependency graph: it imports only
infrastructure (``iris.rpc``, ``iris.cluster``) and never imports from sibling
spoke modules. Spokes import ``require_controller_url`` / ``rpc_client`` from
here instead of from ``iris.cli.main``, which is what lets ``main`` aggregate
spokes without forming an import cycle.
"""

import logging
from pathlib import Path

import click
from rigging.auth import IapRefreshTokenProvider

from iris.cluster.backends.local.cluster import LocalCluster
from iris.cluster.config import IrisConfig
from iris.cluster.token_store import load_token
from iris.rpc import config_pb2
from iris.rpc.auth import ClientCredentials
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync

logger = logging.getLogger(__name__)


def _bundled_iris_config_dir() -> str | None:
    """Return the iris package's bundled config/ dir when it ships on disk.

    Probes two layouts because the config directory can physically live in
    two places depending on how iris was installed:

    1. Wheel installs (site-packages): hatchling force-include places the
       yamls at ``iris/config/`` inside the package. Resolve that via
       ``Path(__file__).parent.parent / "config"``.
    2. Editable workspace installs: the yamls stay at their source location
       ``lib/iris/config/`` — reachable via ``parents[3] / "config"`` from
       ``lib/iris/src/iris/cli/connect.py``.

    Returns the first directory that exists, or ``None`` for wheel installs
    that don't ship configs at all.
    """
    here = Path(__file__).resolve()
    wheel_path = here.parent.parent / "config"
    if wheel_path.is_dir():
        return str(wheel_path)
    editable_path = here.parents[3] / "config"
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
        "lib/iris/config",  # in-tree marin checkout
        _bundled_iris_config_dir(),  # editable install from sibling workspace
    )
    if p is not None
)


def rpc_client(
    address: str,
    credentials: ClientCredentials | None = None,
    timeout_ms: int = 30_000,
) -> ControllerServiceClientSync:
    """Create an RPC client with optional auth. Use as a context manager: ``with rpc_client(url) as c:``.

    ``credentials`` carries the Iris JWT (``Authorization``) and, for an
    IAP-fronted cluster, the IAP OIDC ID token (``Proxy-Authorization``).
    """
    interceptors = credentials.interceptors() if credentials is not None else []
    return ControllerServiceClientSync(
        address,
        timeout_ms=timeout_ms,
        interceptors=interceptors,
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
    )


def rpc_client_for_ctx(
    ctx: click.Context,
    *,
    url: str | None = None,
    timeout_ms: int = 30_000,
) -> ControllerServiceClientSync:
    """Build an RPC client from the CLI context, threading both auth tokens.

    Resolves the controller URL (establishing a tunnel if needed, unless ``url``
    is given) and attaches the ``ClientCredentials`` stashed on the context by the
    ``iris`` group. Prefer this over ``rpc_client`` in subcommands so IAP-fronted
    clusters work uniformly.
    """
    controller_url = url or require_controller_url(ctx)
    obj = ctx.obj or {}
    return rpc_client(controller_url, obj.get("credentials"), timeout_ms=timeout_ms)


def iap_config(config: config_pb2.IrisClusterConfig | None) -> config_pb2.IapAuthConfig | None:
    """Return the IAP auth config if this cluster is IAP-fronted, else None."""
    if config is None or not config.HasField("auth"):
        return None
    if config.auth.WhichOneof("provider") != "iap":
        return None
    return config.auth.iap


def build_iap_provider(
    iap: config_pb2.IapAuthConfig,
    cluster_name: str,
) -> IapRefreshTokenProvider | None:
    """Build an IAP ID-token provider from the cached refresh token, or None.

    Returns None when no refresh token is cached yet (i.e. before ``iris login``),
    so pre-login commands degrade to a clear UNAUTHENTICATED error rather than
    crashing on a missing credential.
    """
    credential = load_token(cluster_name)
    if credential is None or not credential.iap_refresh_token:
        return None
    return IapRefreshTokenProvider(
        client_id=iap.oauth_client_id,
        client_secret=iap.oauth_client_secret,
        refresh_token=credential.iap_refresh_token,
    )


def require_controller_url(ctx: click.Context) -> str:
    """Get controller_url from context, establishing a tunnel lazily if needed.

    On first call with a loaded config, this establishes the tunnel to the controller
    and caches the result. Subsequent calls return the cached URL.
    Commands that don't call this (e.g. ``cluster start``) never pay tunnel cost.
    """
    controller_url = ctx.obj.get("controller_url") if ctx.obj else None
    if controller_url:
        return controller_url

    config = ctx.obj.get("config") if ctx.obj else None

    # IAP-fronted clusters are reachable directly over HTTPS (gated by IAP at the
    # ingress) — no SSH tunnel. The public URL comes from the auth config.
    iap = iap_config(config)
    if iap is not None:
        if not iap.url:
            raise click.ClickException("IAP auth config is missing the ingress 'url'")
        ctx.obj["controller_url"] = iap.url
        return iap.url

    # Lazy tunnel establishment from config
    if config:
        iris_config = IrisConfig(config)
        bundle = iris_config.provider_bundle()
        ctx.obj["provider_bundle"] = bundle

        if iris_config.proto.controller.WhichOneof("controller") == "local":
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
