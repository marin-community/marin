# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global Node inventory commands."""

import click

from iris.cli.connect import resource_client_for_ctx
from iris.cli.resource_commands import (
    DEFAULT_RESOURCE_LIST_LIMIT,
    MAX_RESOURCE_LIST_LIMIT,
    echo_source_warnings,
    resource_key,
)
from iris.cluster.resources.identity import NodeLocator, ResourceKind
from iris.cluster.resources.node import NodeHealth, NodeQuery


@click.group()
def node() -> None:
    """Inspect execution Nodes across all backends."""


@node.command("list")
@click.option("--backend", "backend_id", help="Only Nodes from this backend.")
@click.option("--contains", help="Only Nodes whose provider identifier contains this text.")
@click.option("--health", type=click.Choice([item.value for item in NodeHealth]), multiple=True)
@click.option(
    "--limit",
    type=click.IntRange(1, MAX_RESOURCE_LIST_LIMIT),
    default=DEFAULT_RESOURCE_LIST_LIMIT,
    show_default=True,
)
@click.pass_context
def node_list(
    ctx: click.Context,
    backend_id: str | None,
    contains: str | None,
    health: tuple[str, ...],
    limit: int,
) -> None:
    """List Nodes, retaining partial results when one backend is unavailable."""
    with resource_client_for_ctx(ctx) as client:
        page = client.list_nodes(
            NodeQuery(
                backend_id=backend_id,
                contains=contains,
                health=frozenset(NodeHealth(item) for item in health),
                page_size=limit,
            )
        )
    if not page.items:
        click.echo("No nodes found.")
    else:
        click.echo(f"{'NODE':<40} {'BACKEND':<16} {'HEALTH':<12} {'TASKS':>5} ACCELERATOR")
        for item in page.items:
            accelerator = item.capacity.accelerator_variant or item.capacity.accelerator_kind or "-"
            if item.capacity.accelerator_count:
                accelerator = f"{accelerator}x{item.capacity.accelerator_count}"
            click.echo(
                f"{item.identity.key.resource_id:<40} {item.identity.backend_id:<16} "
                f"{item.health.value:<12} {item.running_task_count:>5} {accelerator}"
            )
    echo_source_warnings(page.source_statuses)


@node.command("describe")
@click.argument("node_id")
@click.option("--backend", "backend_id", required=True, help="Backend that owns the Node.")
@click.option("--uid", "node_uid", help="Require this exact Node incarnation.")
@click.pass_context
def node_describe(ctx: click.Context, node_id: str, backend_id: str, node_uid: str | None) -> None:
    """Describe one logical or exact Node."""
    locator = NodeLocator(resource_key(ctx, ResourceKind.NODE, node_id), backend_id, node_uid)
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_node(locator)
    summary = detail.summary
    click.echo(f"Node: {summary.identity.key.resource_id}")
    click.echo(f"UID: {summary.identity.node_uid}")
    click.echo(f"Backend: {summary.identity.backend_id}")
    click.echo(f"Health: {summary.health.value}")
    click.echo(f"Schedulable: {'yes' if summary.schedulable else 'no'}")
    click.echo(f"Running tasks: {summary.running_task_count}")
    if detail.address:
        click.echo(f"Address: {detail.address}")
    if detail.attributes:
        click.echo("Attributes:")
        for attribute in detail.attributes:
            value = attribute.string_value
            if value is None:
                value = attribute.integer_value
            if value is None:
                value = attribute.float_value
            click.echo(f"  {attribute.key}: {value}")
    echo_source_warnings(detail.source_statuses)
