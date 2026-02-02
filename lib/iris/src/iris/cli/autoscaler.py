# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autoscaler status subcommands."""

import click

from iris.cluster.vm.vm_platform import compute_slice_state_counts
from iris.rpc.proto_utils import format_accelerator_display

from iris.cli._helpers import format_timestamp, get_autoscaler_status


@click.group()
def autoscaler():
    """Autoscaler commands."""


@autoscaler.command("status")
@click.option("--controller-url", required=True, help="Controller URL (e.g., http://localhost:10000)")
@click.pass_context
def autoscaler_status_cmd(ctx, controller_url: str):
    """Show autoscaler status and scaling decisions."""
    try:
        status = get_autoscaler_status(controller_url)
    except Exception as e:
        click.echo(f"Error connecting to controller: {e}", err=True)
        raise SystemExit(1) from None

    click.echo("Autoscaler Status")
    click.echo("=================")
    click.echo(f"Last evaluation: {format_timestamp(status.last_evaluation_ms)}")

    if status.current_demand:
        click.echo("\nCurrent Demand:")
        for group_name, demand in status.current_demand.items():
            click.echo(f"  {group_name}: {demand} pending tasks")

    if not status.groups:
        click.echo("\nNo scale groups configured")
        return

    click.echo("\nScale Groups:")
    for group in status.groups:
        counts = compute_slice_state_counts(group.slices)
        total = sum(counts.values())
        click.echo(f"\n  {group.name}:")
        click.echo(
            f"    Type: {format_accelerator_display(group.config.accelerator_type, group.config.accelerator_variant)}"
        )
        click.echo(f"    Min/Max slices: {group.config.min_slices}/{group.config.max_slices}")
        click.echo(
            f"    Current slices: {total} "
            f"(booting: {counts.get('booting', 0)}, initializing: {counts.get('initializing', 0)}, "
            f"ready: {counts.get('ready', 0)}, failed: {counts.get('failed', 0)})"
        )
        click.echo(f"    Demand: current={group.current_demand}, peak={group.peak_demand}")

        if group.last_scale_up_ms > 0:
            click.echo(f"    Last scale up: {format_timestamp(group.last_scale_up_ms)}")
        if group.last_scale_down_ms > 0:
            click.echo(f"    Last scale down: {format_timestamp(group.last_scale_down_ms)}")

        if group.consecutive_failures > 0:
            click.echo(f"    Consecutive failures: {group.consecutive_failures}")
            if group.backoff_until_ms > 0:
                click.echo(f"    Backoff until: {format_timestamp(group.backoff_until_ms)}")
