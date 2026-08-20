# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-qualified Slice inventory commands."""

import click

from iris.cli.connect import resource_client_for_ctx
from iris.cli.resource_commands import (
    DEFAULT_RESOURCE_LIST_LIMIT,
    MAX_RESOURCE_LIST_LIMIT,
    echo_source_warnings,
    resource_key,
)
from iris.resources.identity import ResourceKind, SliceLocator
from iris.resources.slice import SliceQuery


@click.group("slice")
def slice_group() -> None:
    """Inspect capacity Slices across all backends."""


@slice_group.command("list")
@click.option("--backend", "backend_id", help="Only Slices from this backend.")
@click.option("--scaling-group", "scaling_group_id")
@click.option(
    "--limit",
    type=click.IntRange(1, MAX_RESOURCE_LIST_LIMIT),
    default=DEFAULT_RESOURCE_LIST_LIMIT,
    show_default=True,
)
@click.pass_context
def slice_list(
    ctx: click.Context,
    backend_id: str | None,
    scaling_group_id: str | None,
    limit: int,
) -> None:
    """List backend-qualified capacity Slices."""
    with resource_client_for_ctx(ctx) as client:
        page = client.list_slices(SliceQuery(backend_id=backend_id, scaling_group_id=scaling_group_id, page_size=limit))
    if not page.items:
        click.echo("No slices found.")
    else:
        click.echo(f"{'SLICE':<40} {'BACKEND':<16} {'STATE':<12} {'MEMBERSHIP':<12} MEMBERS")
        for item in page.items:
            click.echo(
                f"{item.identity.key.resource_id:<40} {item.identity.backend_id:<16} "
                f"{item.lifecycle.value:<12} {item.membership_state.value:<12} {item.observed_member_count}"
            )
    echo_source_warnings(page.source_statuses)


@slice_group.command("describe")
@click.argument("slice_id")
@click.option("--backend", "backend_id", required=True, help="Backend that owns the Slice.")
@click.option("--uid", "slice_uid", help="Require this exact Slice incarnation.")
@click.pass_context
def slice_describe(ctx: click.Context, slice_id: str, backend_id: str, slice_uid: str | None) -> None:
    """Describe one logical or exact Slice and its observed membership."""
    locator = SliceLocator(resource_key(ctx, ResourceKind.SLICE, slice_id), backend_id, slice_uid)
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_slice(locator)
    summary = detail.summary
    click.echo(f"Slice: {summary.identity.key.resource_id}")
    click.echo(f"UID: {summary.identity.slice_uid}")
    click.echo(f"Backend: {summary.identity.backend_id}")
    click.echo(f"State: {summary.lifecycle.value}")
    click.echo(f"Membership: {summary.membership_state.value}")
    if detail.members:
        click.echo("Members:")
        for member in detail.members:
            node = member.node.key.resource_id if member.node is not None else "unresolved"
            click.echo(f"  {member.provider_node_id}: {node}")
    echo_source_warnings(detail.source_statuses)
