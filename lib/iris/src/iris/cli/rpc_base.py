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

from iris.rpc.auth import AuthTokenInjector, TokenProvider
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
       ``lib/iris/src/iris/cli/rpc_base.py``.

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
    token_provider: TokenProvider | None = None,
    timeout_ms: int = 30_000,
) -> ControllerServiceClientSync:
    """Create an RPC client with optional auth. Use as a context manager: ``with rpc_client(url) as c:``."""
    interceptors = [AuthTokenInjector(token_provider)] if token_provider else []
    return ControllerServiceClientSync(
        address,
        timeout_ms=timeout_ms,
        interceptors=interceptors,
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
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
