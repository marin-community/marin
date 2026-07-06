# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI for registered controller endpoints.

``mint`` issues a capability URL for an endpoint that is already registered, so a
caller can be handed off-cluster access without relaunching the job that serves
it.
"""

import time

import click
from rigging.connect import capability_path
from rigging.timing import Duration

from iris.cli.connect import rpc_client_for_ctx
from iris.rpc import controller_pb2
from iris.time_proto import duration_to_proto


@click.group()
def endpoints():
    """Registered controller endpoints."""


@endpoints.command("mint")
@click.argument("name")
@click.option(
    "--ttl-hours",
    type=float,
    default=24.0,
    show_default=True,
    help="Token lifetime; clamped server-side to the controller's maximum.",
)
@click.pass_context
def mint(ctx, name: str, ttl_hours: float):
    """Mint a capability URL for an already-registered endpoint NAME (e.g. /serve/foo).

    The scoped token rides in the URL path (/proxy/t/<token>/<name>/...), so
    possession of the URL is the credential — no auth header. It authorizes only
    this endpoint and expires after --ttl-hours. The mint runs under your
    identity, so the controller's owner check passes.
    """
    with rpc_client_for_ctx(ctx) as client:
        resp = client.mint_endpoint_token(
            controller_pb2.Controller.MintEndpointTokenRequest(
                endpoint_name=name,
                ttl=duration_to_proto(Duration.from_hours(ttl_hours)),
            )
        )

    config = (ctx.obj or {}).get("config")
    dashboard_url = (config.dashboard_url if config else "").rstrip("/")
    hours_left = max(0.0, (resp.expires_at.epoch_ms - int(time.time() * 1000)) / 3_600_000)
    path = capability_path(name, resp.token)

    click.echo(f"Capability URL for {name} (token in the path — anyone with the URL can call it):")
    if dashboard_url:
        click.echo(f"  url        {dashboard_url}{path}/")
        click.echo(f"  expires    in {hours_left:.1f}h")
        click.echo("  note       append the app path (e.g. /v1 for an OpenAI server); no auth header needed")
    else:
        # No public origin on this config — front the controller's /proxy/t route.
        click.echo(f"  path       {path}/  (front the controller's /proxy/t route to reach it off-cluster)")
        click.echo(f"  expires    in {hours_left:.1f}h")
