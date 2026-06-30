# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-free cluster connection helpers shared by the CLI and the SDK.

Connecting to a cluster means resolving a named cluster to its config, reaching
its controller (a tunnel, or a direct HTTPS endpoint for an IAP-fronted
cluster), resolving its credentials, and building an authenticated
:class:`~iris.client.client.IrisClient`. The primitives for that — the
cluster-config search dirs, cluster-name resolution, and credential assembly —
live here so both the CLI (``iris.cli.*`` imports down from here) and non-CLI
callers reach them without a ``click.Context``.

:func:`connect_to_cluster` runs the full resolve → reach → authenticate →
connect sequence and yields a live client; :func:`stream_until_complete` waits on
a submitted job, streaming its logs and surviving a dropped connection. Together
they let an experiment script connect to a cluster and submit jobs on its own.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

from rigging.cluster_manifest import AuthProvider, ClusterAuth, IapAuth
from rigging.config_discovery import resolve_cluster_config
from rigging.credential_store import cluster_name_from_url
from rigging.credentials import ClientCredentials, credentials_for

from iris.client.client import IrisClient, Job, JobFailedError
from iris.cluster.composer import provider_bundle
from iris.cluster.config import AuthConfig, IrisClusterConfig, load_config
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.platforms.k8s.controller import configure_client_s3
from iris.rpc import job_pb2

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
    config: IrisClusterConfig | None,
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
    if config and config.controller.controller_kind() == "local":
        return "local"
    if controller_url:
        return cluster_name_from_url(controller_url)
    return "default"


def _cluster_auth_from_config(auth: AuthConfig) -> ClusterAuth:
    """Adapt iris's ``AuthConfig`` to rigging's ``ClusterAuth``.

    The single boundary where iris's wire config meets the shared credential
    vocabulary; everything downstream resolves through ``rigging.credentials``.
    """
    provider = auth.provider_kind()
    if provider == "iap":
        # ``audiences`` are the login audiences the controller accepts; they
        # include the desktop client id (interactive flow). A service-account
        # edge token must carry an IAP-secured audience — never the desktop one,
        # which IAP rejects — so the desktop client id is dropped here and only
        # genuine programmatic audiences reach the service-account token path.
        desktop_oauth_client_id = auth.iap.oauth_client_id or None
        programmatic_audiences = tuple(a for a in auth.iap.audiences if a != desktop_oauth_client_id)
        return ClusterAuth(
            AuthProvider.IAP,
            iap=IapAuth(
                url=auth.iap.url,
                desktop_oauth_client_id=desktop_oauth_client_id,
                desktop_oauth_client_secret=auth.iap.oauth_client_secret or None,
                programmatic_audiences=programmatic_audiences,
                signed_header_audience=auth.iap.signed_header_audience or None,
            ),
        )
    if provider == "gcp":
        return ClusterAuth(AuthProvider.GCP)
    if provider == "static":
        return ClusterAuth(AuthProvider.STATIC)
    return ClusterAuth(AuthProvider.NONE)


def client_credentials(config: IrisClusterConfig | None, cluster_name: str) -> ClientCredentials:
    """Resolve the cluster's client credentials via the shared rigging resolver."""
    if config is None or config.auth is None:
        return credentials_for(cluster_name, ClusterAuth(AuthProvider.NONE))
    auth = config.auth
    static_token = next(iter(auth.static.tokens), None) if auth.provider_kind() == "static" else None
    return credentials_for(cluster_name, _cluster_auth_from_config(auth), static_token=static_token)


def _iap_url(config: IrisClusterConfig) -> str | None:
    """The public HTTPS controller URL for an IAP-fronted cluster, else None.

    IAP clusters are reachable directly (gated by IAP at the ingress), so they
    skip the SSH tunnel; the URL comes from the auth config.
    """
    if config.auth is None or config.auth.provider_kind() != "iap":
        return None
    iap = config.auth.iap
    if iap is None or not iap.url:
        raise ValueError("IAP auth config is missing the ingress 'url'")
    return iap.url


@contextlib.contextmanager
def connect_to_cluster(
    cluster: str,
    *,
    workspace: Path | None = None,
    timeout_ms: int = 30_000,
) -> Iterator[IrisClient]:
    """Resolve a named cluster, reach its controller, and yield a connected client.

    A context manager: the tunnel (or local controller) and the client are torn
    down on exit. Requests authenticate with the cluster's resolved credentials
    (the token cached by ``iris login``, or the config's auth provider).

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
    config = load_config(str(config_path))

    configure_client_s3(config)
    name = resolve_cluster_name(config, None, cluster)
    credentials = client_credentials(config, name)

    with contextlib.ExitStack() as stack:
        controller_url = _iap_url(config)
        if controller_url is None and config.controller.controller_kind() == "local":
            local_cluster = LocalCluster(config)
            controller_url = local_cluster.start()
            stack.callback(local_cluster.close)
        elif controller_url is None:
            bundle = provider_bundle(config)
            controller_address = config.controller_address() or bundle.controller.discover_controller(config.controller)
            logger.info("Establishing tunnel to controller at %s ...", controller_address)
            controller_url = stack.enter_context(bundle.controller.tunnel(address=controller_address))

        logger.info("Connected to cluster %r via %s", name, controller_url)
        client = stack.enter_context(
            IrisClient.remote(controller_url, workspace=workspace, timeout_ms=timeout_ms, credentials=credentials)
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
