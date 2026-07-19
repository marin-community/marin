# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI helpers used by every ``iris`` subcommand module.

This module is a leaf in the ``iris.cli`` dependency graph: it never imports
from sibling spoke modules. Spokes import controller/client helpers from here
instead of from ``iris.cli.main``, which lets ``main`` aggregate spokes without
forming an import cycle.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import click
from rigging.cluster_manifest import AuthProvider, ClusterAuth, IapAuth
from rigging.config_discovery import resolve_cluster_config
from rigging.credential_store import cluster_name_from_url
from rigging.credentials import ClientCredentials, credentials_for

from iris.client import IrisClient
from iris.cluster.composer import provider_bundle
from iris.cluster.config import AuthConfig, IapAuthConfig, IrisClusterConfig, load_config
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.platforms.factory import ProviderBundle
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
DEFAULT_CONTROLLER_TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class ControllerEndpoint:
    """Resolved controller URL plus auth/config context for client construction."""

    url: str
    credentials: ClientCredentials
    config: IrisClusterConfig | None


def resolve_cluster_name(
    config: IrisClusterConfig | None,
    controller_url: str | None,
    cli_cluster_name: str | None,
) -> str:
    """Pick the credential-store cluster name for a CLI invocation."""
    if cli_cluster_name:
        return cli_cluster_name
    if config and config.name:
        return config.name
    if config and config.controller.controller_kind() == "local":
        return "local"
    if controller_url:
        return cluster_name_from_url(controller_url)
    return "default"


def _cluster_auth_from_config(auth: AuthConfig) -> ClusterAuth:
    """Adapt iris's ``AuthConfig`` to rigging's shared credential vocabulary."""
    provider = auth.provider_kind()
    if provider == "iap":
        # `programmatic_audiences` are the service-account edge audiences the
        # client mints against. Empty is fine: rigging's edge resolver falls back
        # to the desktop client id, which IAP registers as a programmatic client.
        return ClusterAuth(
            AuthProvider.IAP,
            iap=IapAuth(
                url=auth.iap.url,
                desktop_oauth_client_id=auth.iap.oauth_client_id or None,
                desktop_oauth_client_secret=auth.iap.oauth_client_secret or None,
                programmatic_audiences=tuple(auth.iap.programmatic_audiences),
                signed_header_audience=auth.iap.signed_header_audience or None,
            ),
        )
    return ClusterAuth(AuthProvider.NONE)


def client_credentials(
    config: IrisClusterConfig | None,
    cluster_name: str,
) -> ClientCredentials:
    """Resolve the cluster's client credentials via the shared rigging resolver."""
    auth = (
        _cluster_auth_from_config(config.auth)
        if config is not None and config.auth is not None
        else ClusterAuth(AuthProvider.NONE)
    )
    return credentials_for(cluster_name, auth)


@contextmanager
def open_controller_endpoint(
    *,
    config_file: Path | None = None,
    controller_url: str | None = None,
    cluster_name: str | None = None,
) -> Iterator[ControllerEndpoint]:
    """Resolve a reachable controller URL and keep its local resources alive.

    Click-free, so programmatic callers (harbor, deploy tooling) can use it from
    any thread: raises ValueError on bad arguments and ConnectionError when the
    controller is unreachable. Local resources (an SSH tunnel, a local cluster)
    live until the context exits.
    """
    if controller_url is not None and config_file is not None:
        raise ValueError("Cannot specify both controller_url and config_file")

    if cluster_name and config_file is None and controller_url is None:
        try:
            config_file = resolve_cluster_config(cluster_name, dirs=IRIS_CLUSTER_CONFIG_DIRS)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Unknown cluster {cluster_name!r}. Run `iris cluster list` to see available clusters."
            ) from exc

    config = load_config(str(config_file)) if config_file is not None else None
    resolved_cluster_name = resolve_cluster_name(config, controller_url, cluster_name)
    credentials = client_credentials(config, resolved_cluster_name)

    with ExitStack() as stack:
        if controller_url is not None:
            url = controller_url
        else:
            url, _ = _connect_controller(
                stack, config=config, config_file=str(config_file) if config_file is not None else None
            )
        yield ControllerEndpoint(url=url, credentials=credentials, config=config)


def _connect_controller(
    stack: ExitStack,
    *,
    config: IrisClusterConfig | None,
    config_file: str | None,
) -> tuple[str, ProviderBundle | None]:
    """Resolve the controller URL from a loaded config, registering cleanup on ``stack``.

    IAP-fronted clusters return their public ingress URL directly. Otherwise a
    local cluster is started and/or an SSH tunnel established, with teardown
    pushed onto ``stack``. Returns the URL plus the provider bundle when one was
    built (the CLI caches it on the click context for later subcommand use).
    """
    iap = iap_config(config)
    if iap is not None:
        if not iap.url:
            raise ValueError("IAP auth config is missing the ingress 'url'")
        return iap.url, None

    if config:
        bundle = provider_bundle(config)
        if config.controller.controller_kind() == "local":
            cluster = LocalCluster(config)
            controller_address = cluster.start()
            stack.callback(cluster.close)
        else:
            controller_address = config.controller_address()
            if not controller_address:
                controller_address = bundle.controller.discover_controller(config.controller)

        try:
            logger.info("Establishing tunnel to controller...")
            url = stack.enter_context(bundle.controller.tunnel(address=controller_address))
        except Exception as e:
            raise ConnectionError(f"Could not connect to controller: {e}") from e
        return url, bundle

    if config_file:
        raise ConnectionError(
            f"Could not connect to controller (config: {config_file}). "
            "Check that the controller is running and reachable."
        )
    raise ValueError(
        "No controller specified. Pass --cluster=<name> (see `iris cluster list`), --controller-url, or --config."
    )


def iris_client_for_ctx(
    ctx: click.Context,
    *,
    workspace: Path | None,
    timeout_ms: int = DEFAULT_CONTROLLER_TIMEOUT_MS,
    extra_bundle_includes: Sequence[str] = (),
) -> IrisClient:
    """Build an IrisClient from an active Iris CLI context."""
    obj = ctx.obj or {}
    return IrisClient.remote(
        require_controller_url(ctx),
        workspace=workspace,
        credentials=obj.get("credentials"),
        timeout_ms=timeout_ms,
        extra_bundle_includes=extra_bundle_includes,
    )


@contextmanager
def open_iris_client(
    *,
    config_file: Path | None = None,
    controller_url: str | None = None,
    cluster_name: str | None = None,
    workspace: Path | None,
    timeout_ms: int = DEFAULT_CONTROLLER_TIMEOUT_MS,
    extra_bundle_includes: Sequence[str] = (),
) -> Iterator[IrisClient]:
    """Open an IrisClient from a config file, cluster name, or direct controller URL."""
    with open_controller_endpoint(
        config_file=config_file,
        controller_url=controller_url,
        cluster_name=cluster_name,
    ) as endpoint:
        with IrisClient.remote(
            endpoint.url,
            workspace=workspace,
            credentials=endpoint.credentials,
            timeout_ms=timeout_ms,
            extra_bundle_includes=extra_bundle_includes,
        ) as client:
            yield client


def rpc_client(
    address: str,
    credentials: ClientCredentials | None = None,
    timeout_ms: int = DEFAULT_CONTROLLER_TIMEOUT_MS,
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
    timeout_ms: int = DEFAULT_CONTROLLER_TIMEOUT_MS,
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


def iap_config(config: IrisClusterConfig | None) -> IapAuthConfig | None:
    """Return the IAP auth config if this cluster is IAP-fronted, else None."""
    if config is None or config.auth is None:
        return None
    if config.auth.provider_kind() != "iap":
        return None
    return config.auth.iap


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
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    stack = ExitStack()
    try:
        url, bundle = _connect_controller(stack, config=config, config_file=config_file)
    except (ValueError, ConnectionError) as e:
        stack.close()
        raise click.ClickException(str(e)) from e
    except BaseException:
        stack.close()
        raise
    ctx.call_on_close(stack.close)
    ctx.obj["controller_url"] = url
    if bundle is not None:
        ctx.obj["provider_bundle"] = bundle
    return url
