# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI for identity tokens.

``auth mint-scoped`` mints a job-prefix-scoped bearer token: an agent holding
it acts as the calling user but only on jobs under the prefix, so several
agents sharing one Iris account cannot reach each other's jobs.
"""

import time

import click
from rigging.timing import Duration

from iris.cli.connect import rpc_client_for_ctx
from iris.rpc import controller_pb2
from iris.time_proto import duration_to_proto


@click.group()
def auth():
    """Identity tokens for headless callers."""
    pass


@auth.command("mint-scoped")
@click.option(
    "--prefix",
    required=True,
    help="Job-name prefix the token may act on, e.g. /benfeuer/exp2- (must include a job component).",
)
@click.option(
    "--ttl-hours",
    type=float,
    default=None,
    help="Token lifetime; clamped server-side to the controller's maximum (30 days).",
)
@click.pass_context
def mint_scoped(ctx, prefix: str, ttl_hours: float | None):
    """Mint a job-prefix-scoped token for your user.

    The token authenticates as you on every RPC, but owner-gated actions —
    submit, cancel, exec, kick, endpoint mint — only reach jobs whose name
    starts with --prefix. Give each agent its own prefix so one agent cannot
    cancel another's jobs. The token is not revocable. The controller sets
    the default lifetime when --ttl-hours is omitted.

    Pass it to the agent as MARIN_CLUSTER_TOKEN.
    """
    request = controller_pb2.Controller.MintScopedTokenRequest(prefix=prefix)
    if ttl_hours is not None:
        request.ttl.CopyFrom(duration_to_proto(Duration.from_hours(ttl_hours)))

    with rpc_client_for_ctx(ctx) as client:
        resp = client.mint_scoped_token(request)

    hours_left = max(0.0, (resp.expires_at.epoch_ms - int(time.time() * 1000)) / 3_600_000)
    click.echo(f"Scoped token for jobs under {resp.prefix} (expires in {hours_left:.1f}h):")
    click.echo(f"  {resp.token}")
    click.echo("  export MARIN_CLUSTER_TOKEN='<token>'   # in the agent's environment")
    click.echo("  Store it like a password: it is not revocable before it expires.")
