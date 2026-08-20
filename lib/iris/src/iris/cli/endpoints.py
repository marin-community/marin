# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint inventory and capability tokens."""

import click
from rigging.connect import capability_path
from rigging.timing import Duration

from iris.cli.connect import resource_client_for_ctx
from iris.cli.resource_commands import (
    DEFAULT_RESOURCE_LIST_LIMIT,
    MAX_RESOURCE_LIST_LIMIT,
    echo_source_warnings,
    resource_key,
)
from iris.resources.endpoint import EndpointQuery
from iris.resources.identity import ResourceKey, ResourceKind
from iris.rpc.resource_client import ResourceRpcClient


@click.group("endpoint")
def endpoint() -> None:
    """Inspect Endpoints and mint capability links."""


def _resolve_endpoint_reference(
    client: ResourceRpcClient,
    key: ResourceKey,
    reference: str,
) -> ResourceKey:
    if not reference.startswith("/"):
        return key
    matches = client.resolve_endpoints(reference)
    if not matches:
        raise click.ClickException(f"Endpoint {reference!r} was not found")
    if len(matches) > 1:
        ids = ", ".join(sorted(detail.summary.endpoint_id for detail in matches))
        raise click.ClickException(f"Endpoint name {reference!r} is ambiguous; use one of these IDs: {ids}")
    return matches[0].summary.key


@endpoint.command("list")
@click.option("--prefix", help="Only Endpoint names starting with this prefix.")
@click.option("--task", "task_id", help="Only Endpoints registered by this Task ID.")
@click.option(
    "--limit",
    type=click.IntRange(1, MAX_RESOURCE_LIST_LIMIT),
    default=DEFAULT_RESOURCE_LIST_LIMIT,
    show_default=True,
)
@click.pass_context
def endpoint_list(ctx: click.Context, prefix: str | None, task_id: str | None, limit: int) -> None:
    """List registered Endpoints."""
    task = resource_key(ctx, ResourceKind.TASK, task_id) if task_id is not None else None
    with resource_client_for_ctx(ctx) as client:
        page = client.list_endpoints(EndpointQuery(name_prefix=prefix, task=task, page_size=limit))
    if not page.items:
        click.echo("No endpoints found.")
    else:
        click.echo(f"{'ENDPOINT ID':<36} {'NAME':<40} {'ACCESS':<9} TASK")
        for item in page.items:
            click.echo(
                f"{item.endpoint_id:<36} {item.name:<40} {item.access.value:<9} "
                f"{item.task.resource_id if item.task is not None else '-'}"
            )
    echo_source_warnings(page.source_statuses)


@endpoint.command("describe")
@click.argument("endpoint")
@click.pass_context
def endpoint_describe(ctx: click.Context, endpoint: str) -> None:
    """Describe one Endpoint by name or stable registration ID."""
    with resource_client_for_ctx(ctx) as client:
        key = _resolve_endpoint_reference(
            client,
            resource_key(ctx, ResourceKind.ENDPOINT, endpoint),
            endpoint,
        )
        detail = client.describe_endpoint(key)
    summary = detail.summary
    click.echo(f"Endpoint: {summary.endpoint_id}")
    click.echo(f"Name: {summary.name}")
    click.echo(f"Access: {summary.access.value}")
    click.echo(f"Address: {detail.address}")
    click.echo(f"Execution cluster: {summary.execution_cluster_id or '-'}")
    click.echo(f"Task: {summary.task.resource_id if summary.task is not None else '-'}")
    if detail.metadata:
        click.echo("Metadata:")
        for key, value in sorted(detail.metadata.items()):
            click.echo(f"  {key}: {value}")


@endpoint.command("mint")
@click.argument("endpoint")
@click.option("--ttl-hours", type=click.FloatRange(min=0.01), default=24.0, show_default=True)
@click.pass_context
def endpoint_mint(ctx: click.Context, endpoint: str, ttl_hours: float) -> None:
    """Mint a time-limited capability link for one Endpoint name or ID."""
    with resource_client_for_ctx(ctx) as client:
        key = _resolve_endpoint_reference(
            client,
            resource_key(ctx, ResourceKind.ENDPOINT, endpoint),
            endpoint,
        )
        detail = client.describe_endpoint(key)
        token = client.mint_endpoint_token(key, ttl=Duration.from_hours(ttl_hours))
    if token.capability_url:
        click.echo(f"{token.capability_url}/")
    else:
        click.echo(f"{capability_path(detail.summary.name, token.token)}/")
    click.echo(f"Expires: {token.expires_at.epoch_ms()}")
