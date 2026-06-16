# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-free cluster connection helpers shared by the CLI and the SDK.

The ``iris`` CLI resolves a named cluster, opens a tunnel to its controller,
and builds an authenticated :class:`~iris.client.client.IrisClient` — but that
logic was historically spread across the click command tree (``iris.cli.main``
group + ``iris.cli.connect.require_controller_url`` + per-command client
construction), reachable only from inside a ``click.Context``.

This module hoists the click-free primitives — cluster-config search dirs,
cluster-name resolution, token-provider creation — into one place that both the
CLI (``iris.cli.*`` import down from here) and non-CLI callers can use, and adds
:func:`connect_to_cluster`: a context manager that performs the full
resolve → tunnel → authenticate → connect sequence and yields a live client.
This is what lets experiment scripts hoist their own Iris client instead of
being launched via ``uv run iris ... job run -- python -m ...``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

from rigging.config_discovery import resolve_cluster_config

from iris.client.client import IrisClient
from iris.cluster.backends.k8s.controller import configure_client_s3
from iris.cluster.backends.local.cluster import LocalCluster
from iris.cluster.config import IrisConfig
from iris.cluster.token_store import cluster_name_from_url, load_any_token, load_token
from iris.rpc import config_pb2
from iris.rpc.auth import GcpAccessTokenProvider, StaticTokenProvider, TokenProvider

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
       ``lib/iris/src/iris/client/connect.py``.

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


def resolve_cluster_name(
    config: config_pb2.IrisClusterConfig | None,
    controller_url: str | None,
    cli_cluster_name: str | None,
) -> str:
    """Resolve a cluster name, preferring the CLI name, then the config name,
    then ``local`` for a local controller, then a name derived from the URL,
    falling back to ``default``."""
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
    """Create a TokenProvider from an AuthConfig proto for client usage.

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


def _resolve_token_provider(proto: config_pb2.IrisClusterConfig, cluster_name: str) -> TokenProvider | None:
    """Return a provider from the config's auth block, else the stored ``iris login`` token, else None."""
    if proto.HasField("auth"):
        return create_client_token_provider(proto.auth, cluster_name=cluster_name)
    credential = load_token(cluster_name) or load_any_token()
    return StaticTokenProvider(credential.token) if credential is not None else None


@contextlib.contextmanager
def connect_to_cluster(
    cluster: str,
    *,
    workspace: Path | None = None,
    timeout_ms: int = 30_000,
) -> Iterator[IrisClient]:
    """Resolve a named cluster, open a tunnel, and yield a connected IrisClient.

    The non-click twin of the ``iris`` CLI connection path. Resolves ``cluster``
    against the same config search dirs (:data:`IRIS_CLUSTER_CONFIG_DIRS`),
    loads the cluster config, configures storage credentials, builds the
    provider bundle, discovers/tunnels to the controller, loads the stored
    ``iris login`` token, and yields ``IrisClient.remote(...)``. The tunnel,
    any local controller, and the client are all torn down on exit.

    Args:
        cluster: Named cluster to resolve (e.g. ``"marin"``). Must match a config
            YAML in the search dirs; raises ``FileNotFoundError`` otherwise.
        workspace: Directory bundled and shipped to workers (the git repo root,
            containing ``pyproject.toml``). Required for external job submission;
            without it workers have no code to run.
        timeout_ms: RPC timeout for the controller client.

    Yields:
        A connected :class:`~iris.client.client.IrisClient`.
    """
    config_path = resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)
    logger.info("Resolved cluster %r to config: %s", cluster, config_path)
    iris_config = IrisConfig.load(str(config_path))
    proto = iris_config.proto

    configure_client_s3(proto)
    name = resolve_cluster_name(proto, None, cluster)
    token_provider = _resolve_token_provider(proto, name)

    bundle = iris_config.provider_bundle()

    with contextlib.ExitStack() as stack:
        if proto.controller.WhichOneof("controller") == "local":
            local_cluster = LocalCluster(proto)
            controller_address = local_cluster.start()
            stack.callback(local_cluster.close)
        else:
            controller_address = iris_config.controller_address()
            if not controller_address:
                controller_address = bundle.controller.discover_controller(proto.controller)

        logger.info("Establishing tunnel to controller at %s ...", controller_address)
        tunnel_url = stack.enter_context(bundle.controller.tunnel(address=controller_address))
        logger.info("Connected to cluster %r via %s", name, tunnel_url)

        client = stack.enter_context(
            IrisClient.remote(
                tunnel_url,
                workspace=workspace,
                timeout_ms=timeout_ms,
                token_provider=token_provider,
            )
        )
        yield client
