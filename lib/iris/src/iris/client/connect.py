# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-free cluster connection helpers shared by the CLI and the SDK.

Connecting to a cluster means resolving a named cluster to its config, opening a
tunnel to its controller, and building an authenticated
:class:`~iris.client.client.IrisClient`. The primitives for that — cluster-config
search dirs, cluster-name resolution, token-provider creation — live here so both
the CLI (``iris.cli.*`` imports down from here) and non-CLI callers reach them
without a ``click.Context``.

:func:`connect_to_cluster` runs the full resolve → tunnel → authenticate →
connect sequence and yields a live client; :func:`stream_until_complete` waits on
a submitted job, streaming its logs and surviving a dropped connection. Together
they let an experiment script connect and submit on its own (see
``experiments/launch.py``) instead of being wrapped in ``uv run iris ... job run``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

from rigging.config_discovery import resolve_cluster_config

from iris.client.client import IrisClient, Job, JobFailedError
from iris.cluster.backends.k8s.controller import configure_client_s3
from iris.cluster.backends.local.cluster import LocalCluster
from iris.cluster.config import IrisConfig
from iris.cluster.token_store import cluster_name_from_url, load_any_token, load_token
from iris.rpc import config_pb2, job_pb2
from iris.rpc.auth import ClientCredentials, GcpAccessTokenProvider, StaticTokenProvider, TokenProvider

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
    elif provider == "iap":
        # The Iris JWT for an IAP cluster comes only from the token store after
        # `iris login` (handled above). With none cached yet there is no
        # config-derived fallback — return None so commands fail clearly until
        # the user logs in.
        return None
    raise ValueError(f"Unknown auth provider: {provider}")


@contextlib.contextmanager
def connect_to_cluster(
    cluster: str,
    *,
    workspace: Path | None = None,
    timeout_ms: int = 30_000,
) -> Iterator[IrisClient]:
    """Resolve a named cluster, tunnel to its controller, and yield a connected client.

    A context manager: the tunnel, any local controller it starts, and the
    client are all torn down on exit. Requests authenticate with the token
    cached by ``iris login`` for this cluster, falling back to the cluster
    config's auth provider.

    Args:
        cluster: Named cluster to resolve (e.g. ``"marin"``) against
            :data:`IRIS_CLUSTER_CONFIG_DIRS`. Raises ``FileNotFoundError`` if no
            matching config YAML exists.
        workspace: Repo root (containing ``pyproject.toml``) bundled and shipped
            to workers. Required for external job submission; without it workers
            have no code to run.
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
    token_provider = create_client_token_provider(proto.auth, cluster_name=name)

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
                credentials=ClientCredentials(token_provider=token_provider),
            )
        )
        yield client


def stream_until_complete(client: IrisClient, job: Job, *, terminate_on_exit: bool = True) -> int:
    """Stream a job's logs until it finishes; return a shell-style exit code.

    Disconnect-safe, matching ``iris job run``: a dropped connection leaves the
    job — and anything it spawned — running on the cluster (reconnect with
    ``iris job logs -f <id>``). Ctrl-C terminates the job and its children when
    ``terminate_on_exit`` is set.

    Returns 0 on success, 1 on job failure, 130 on Ctrl-C.
    """
    logger.info("Streaming logs (Ctrl+C to stop). Reconnect with: iris job logs -f %s", job.job_id)
    try:
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            logger.info("Job %s finished: %s", job.job_id, job_pb2.JobState.Name(status.state))
            return 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except JobFailedError as e:
            logger.error("Job failed: %s", e)
            return 1
    except KeyboardInterrupt:
        if terminate_on_exit:
            logger.info("Terminating job %s and its children ...", job.job_id)
            client.terminate_prefix(job.job_id, exclude_finished=True)
        return 130
    except Exception:
        logger.warning(
            "Connection lost; job %s is still running. Reconnect with: iris job logs -f %s",
            job.job_id,
            job.job_id,
        )
        raise
