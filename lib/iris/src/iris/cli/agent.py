# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``iris agent`` CLI: run a remote-backend agent that dials home to the root.

The agent owns a real in-cluster :class:`TaskBackend` (built from the local
cluster config) and reconciles it with a root controller over the
RemoteAgentService poll RPC. The root stays authoritative; this process is the
recoverable cache that converges the local backend to the root's desired set.
"""

import logging

import click
from rigging.log_setup import configure_logging

from iris.cluster.agent.loop import AgentLoop, ConnectPollTransport
from iris.cluster.composer import make_task_backend
from iris.cluster.config import load_config

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def agent(ctx):
    """Remote-backend agent commands."""
    parent_obj = ctx.obj or {}
    ctx.ensure_object(dict)
    ctx.obj.update(parent_obj)


@agent.command("serve")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    required=True,
    help="Local cluster config describing the in-cluster backend the agent drives",
)
@click.option("--root-address", required=True, help="Root controller URL the agent dials home to")
@click.option("--backend-id", required=True, help="Root-side backend id this agent serves")
@click.option(
    "--token",
    envvar="IRIS_AGENT_TOKEN",
    default="",
    help="Bearer token presented to the root (or set IRIS_AGENT_TOKEN)",
)
@click.option("--poll-interval", default=2.0, type=float, show_default=True, help="Seconds between poll rounds")
def agent_serve(config_file: str, root_address: str, backend_id: str, token: str, poll_interval: float):
    """Run the remote-backend agent loop until interrupted.

    Loads the local cluster config, builds its task backend, and reconciles it
    with the root over the poll RPC, presenting ``--token`` as a Bearer token.
    """
    configure_logging(level=logging.INFO)
    config = load_config(config_file)
    local_backend = make_task_backend(config)
    transport = ConnectPollTransport(root_address, token)
    loop = AgentLoop(backend_id=backend_id, local_backend=local_backend, transport=transport)

    logger.info(
        "Iris remote agent starting: backend_id=%s root=%s poll_interval=%.1fs",
        backend_id,
        root_address,
        poll_interval,
    )
    click.echo(f"Remote agent serving backend {backend_id!r} -> {root_address}")
    loop.run(poll_interval_seconds=poll_interval)
