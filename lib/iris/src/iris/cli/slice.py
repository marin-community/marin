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

"""Slice management subcommands."""

import json

import click

from iris.rpc import vm_pb2
from iris.rpc.proto_utils import vm_state_name

from iris.cli._helpers import (
    format_timestamp,
    load_autoscaler,
    require_config,
    terminate_slices_parallel,
    wait_for_slice_obj,
)


@click.group("slice")
def slice_group():
    """Slice management commands."""
    pass


@slice_group.command("create")
@click.option("--scale-group", required=True, help="Scale group name")
@click.option("--tag", "tags", multiple=True, help="Tags in KEY=VALUE format")
@click.option("--dry-run", is_flag=True, help="Log commands without executing them")
@click.option("--wait/--no-wait", default=True, help="Wait for slice to reach READY (default: wait)")
@click.pass_context
def slice_create(ctx, scale_group: str, tags: tuple[str, ...], dry_run: bool, wait: bool):
    """Create a new slice in the specified scale group."""
    config_file = require_config(ctx)
    autoscaler_obj, _config = load_autoscaler(config_file, dry_run=dry_run)

    group = autoscaler_obj.get_group(scale_group)
    if not group:
        available = ", ".join(autoscaler_obj.groups.keys()) or "(none)"
        click.echo(f"Error: Unknown scale group '{scale_group}'. Available: {available}", err=True)
        raise SystemExit(1)

    tag_dict: dict[str, str] = {}
    for tag in tags:
        if "=" not in tag:
            click.echo(f"Error: Invalid tag format '{tag}', expected KEY=VALUE", err=True)
            raise SystemExit(1)
        k, v = tag.split("=", 1)
        tag_dict[k] = v

    try:
        slice_obj = group.scale_up(tags=tag_dict or None)
        click.echo(f"Created slice: {slice_obj.slice_id}")

        if wait:
            success = wait_for_slice_obj(slice_obj)
            if not success:
                raise SystemExit(1)
        else:
            click.echo("VMs are now booting. Use 'iris cluster slice list --config ...' to monitor progress.")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None


@slice_group.command("terminate")
@click.argument("slice_ids", nargs=-1, required=False)
@click.option("--all", "terminate_all", is_flag=True, help="Terminate all slices")
@click.option("--scale-group", default=None, help="Only terminate slices in this scale group (with --all)")
@click.pass_context
def slice_terminate(ctx, slice_ids: tuple[str, ...], terminate_all: bool, scale_group: str | None):
    """Terminate one or more slices and all their VMs.

    Examples:
        iris cluster --config=... slice terminate iris-tpu_v5e_16-123456
        iris cluster --config=... slice terminate --all
    """
    config_file = require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file)

    if terminate_all:
        slice_ids = tuple(
            vm_group.slice_id
            for name, group in autoscaler_obj.groups.items()
            if not scale_group or name == scale_group
            for vm_group in group.vm_groups()
        )
        if not slice_ids:
            click.echo("No slices found to terminate")
            return
    elif not slice_ids:
        click.echo("Error: Either provide SLICE_IDS or use --all", err=True)
        raise SystemExit(1)

    terminate_slices_parallel(autoscaler_obj, slice_ids)


@slice_group.command("list")
@click.option("--scale-group", default=None, help="Filter by scale group name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def slice_list(ctx, scale_group: str | None, as_json: bool):
    """List all slices."""
    config_file = require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file)

    all_slices = []
    for name, group in autoscaler_obj.groups.items():
        if scale_group and name != scale_group:
            continue
        for vm_group in group.vm_groups():
            status = vm_group.status()
            all_slices.append((name, vm_group, status))

    if as_json:
        output = []
        for group_name, vm_group, status in all_slices:
            output.append(
                {
                    "slice_id": vm_group.slice_id,
                    "scale_group": group_name,
                    "all_ready": status.all_ready,
                    "any_failed": status.any_failed,
                    "vm_count": len(list(vm_group.vms())),
                    "created_at_ms": vm_group.created_at_ms,
                }
            )
        click.echo(json.dumps(output, indent=2))
    else:
        if not all_slices:
            click.echo("No slices found")
            return

        click.echo(f"{'SLICE ID':<40} {'SCALE GROUP':<20} {'STATUS':<10} {'VMs'}")
        click.echo("-" * 80)
        for group_name, vm_group, status in all_slices:
            status_str = "READY" if status.all_ready else ("FAILED" if status.any_failed else "PENDING")
            vm_count = len(list(vm_group.vms()))
            click.echo(f"{vm_group.slice_id:<40} {group_name:<20} {status_str:<10} {vm_count}")


@slice_group.command("get")
@click.argument("slice_id")
@click.pass_context
def slice_get(ctx, slice_id: str):
    """Get detailed info for a specific slice."""
    config_file = require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file)

    for group_name, group in autoscaler_obj.groups.items():
        for vm_group in group.vm_groups():
            if vm_group.slice_id == slice_id:
                status = vm_group.status()
                status_str = "READY" if status.all_ready else ("FAILED" if status.any_failed else "PENDING")
                vms = list(vm_group.vms())

                click.echo(f"Slice: {vm_group.slice_id}")
                click.echo(f"Scale Group: {group_name}")
                click.echo(f"Status: {status_str}")
                click.echo(f"Created: {format_timestamp(vm_group.created_at_ms)}")
                click.echo(f"VMs: {len(vms)}")
                click.echo()

                for vm_item in vms:
                    state = vm_state_name(vm_item.info.state)
                    click.echo(f"  {vm_item.info.vm_id}:")
                    click.echo(f"    State: {state}")
                    click.echo(f"    Address: {vm_item.info.address or '(none)'}")
                    if vm_item.info.zone:
                        click.echo(f"    Zone: {vm_item.info.zone}")
                    if vm_item.info.worker_id:
                        click.echo(f"    Worker ID: {vm_item.info.worker_id}")
                    if vm_item.info.init_phase:
                        click.echo(f"    Init Phase: {vm_item.info.init_phase}")
                    if vm_item.info.init_error:
                        click.echo(f"    Error: {vm_item.info.init_error}")
                return

    click.echo(f"Error: Slice {slice_id} not found", err=True)
    raise SystemExit(1)
