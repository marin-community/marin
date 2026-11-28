# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command line reader and producer interface for ops.events."""

import json
from pathlib import Path

import click
import yaml

from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client
from finelog.ops_feed import DEFAULT_STATE, Severity, Subscription, ack, emit, event_dict, poll


@click.group()
@click.option("--deployment", default="marin", show_default=True)
@click.option("--state", type=click.Path(path_type=Path), default=DEFAULT_STATE, show_default=True)
@click.pass_context
def cli(ctx: click.Context, deployment: str, state: Path) -> None:
    """Emit, poll, or acknowledge typed fleet events."""
    ctx.obj = (deployment, state)


@cli.command("emit")
@click.option("--source", required=True)
@click.option("--kind", required=True)
@click.option("--severity", type=click.Choice([level.value for level in Severity]), required=True)
@click.option("--subject", required=True)
@click.option("--payload", default="{}", help="JSON object without personal data or secrets.")
@click.option("--dedup-key")
@click.pass_obj
def emit_cmd(settings, source, kind, severity, subject, payload, dedup_key) -> None:
    deployment, _ = settings
    with open_client(load_finelog_config(deployment), deployment) as client:
        click.echo(
            emit(source, kind, Severity(severity), subject, json.loads(payload), dedup_key=dedup_key, client=client)
        )


@cli.command("poll")
@click.option("--subscriptions", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--reader", required=True)
@click.pass_obj
def poll_cmd(settings, subscriptions: Path, reader: str) -> None:
    """Print one page as JSONL and advance this reader's saved position."""
    deployment, state = settings
    raw = yaml.safe_load(subscriptions.read_text())
    entries = [Subscription(s["source"], Severity(s["min_severity"])) for s in raw["subscriptions"]]
    with open_client(load_finelog_config(deployment), deployment) as client:
        for event in poll(client, entries, deployment + ":" + reader, state=state):
            click.echo(json.dumps(event_dict(event)))


@cli.command("ack")
@click.argument("event_id")
@click.pass_obj
def ack_cmd(settings, event_id: str) -> None:
    deployment, _ = settings
    with open_client(load_finelog_config(deployment), deployment) as client:
        click.echo(ack(client, event_id))


if __name__ == "__main__":
    cli()
