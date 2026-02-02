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

"""VM management subcommands."""

import click
from connectrpc.errors import ConnectError

from iris.cluster.vm.vm_platform import compute_slice_state_counts, slice_all_ready, slice_any_failed
from iris.rpc import vm_pb2
from iris.rpc.proto_utils import format_accelerator_display, vm_state_name

from iris.cli._helpers import (
    format_timestamp,
    get_autoscaler_status,
    get_vm_logs,
    load_autoscaler,
    require_config,
)


@click.group()
@click.option("--controller-url", help="Controller URL for RPC mode (e.g., http://localhost:10000)")
@click.pass_context
def vm(ctx, controller_url: str | None):
    """VM management commands.

    Uses --config from parent cluster group, or --controller-url for direct RPC access.
    """
    ctx.obj["controller_url"] = controller_url


@vm.command("status")
@click.option("--scale-group", default=None, help="Filter by scale group name")
@click.pass_context
def vm_status(ctx, scale_group: str | None):
    """Show VM and slice status."""
    controller_url = ctx.obj.get("controller_url")
    config_file = ctx.obj.get("config_file")

    if controller_url:
        _status_via_controller(controller_url, scale_group)
    elif config_file:
        _status_via_autoscaler(config_file, scale_group)
    else:
        click.echo("Error: Either --controller-url or cluster --config is required", err=True)
        raise SystemExit(1)


def _status_via_controller(controller_url: str, scale_group: str | None):
    """Show status via controller RPC."""
    try:
        autoscaler_status = get_autoscaler_status(controller_url)
    except Exception as e:
        click.echo(f"Error connecting to controller: {e}", err=True)
        raise SystemExit(1) from None

    if not autoscaler_status.groups:
        click.echo("No scale groups configured")
        return

    for group in autoscaler_status.groups:
        if scale_group and group.name != scale_group:
            continue

        counts = compute_slice_state_counts(group.slices)
        total = sum(counts.values())
        click.echo(f"\nScale Group: {group.name}")
        accel_display = format_accelerator_display(group.config.accelerator_type, group.config.accelerator_variant)
        click.echo(f"  Accelerator: {accel_display}")
        click.echo(f"  Slices: {counts.get('ready', 0)}/{total} ready")
        click.echo(f"    Booting: {counts.get('booting', 0)}")
        click.echo(f"    Initializing: {counts.get('initializing', 0)}")
        click.echo(f"    Failed: {counts.get('failed', 0)}")
        click.echo(f"  Demand: {group.current_demand} (peak: {group.peak_demand})")

        if group.backoff_until_ms > 0:
            click.echo(f"  Backoff until: {format_timestamp(group.backoff_until_ms)}")
            click.echo(f"  Consecutive failures: {group.consecutive_failures}")

        if group.slices:
            click.echo("  Slices:")
            for slice_info in group.slices:
                status_str = (
                    "READY" if slice_all_ready(slice_info) else ("FAILED" if slice_any_failed(slice_info) else "PENDING")
                )
                click.echo(f"    {slice_info.slice_id}: {status_str}")
                for vm_info in slice_info.vms:
                    state = vm_state_name(vm_info.state)
                    click.echo(f"      {vm_info.vm_id}: {state} ({vm_info.address})")
                    if vm_info.init_error:
                        click.echo(f"        Error: {vm_info.init_error}")

    click.echo(f"\nLast evaluation: {format_timestamp(autoscaler_status.last_evaluation_ms)}")


def _status_via_autoscaler(config_file: str, scale_group: str | None):
    """Show status via direct Autoscaler."""
    autoscaler_obj, _ = load_autoscaler(config_file)

    groups = autoscaler_obj.groups
    if scale_group:
        groups = {k: v for k, v in groups.items() if k == scale_group}

    if not groups:
        click.echo("No scale groups found")
        return

    for name, group in groups.items():
        vm_groups = group.vm_groups()
        click.echo(f"\nScale Group: {name}")
        accel_display = format_accelerator_display(group.config.accelerator_type, group.config.accelerator_variant)
        click.echo(f"  Accelerator: {accel_display}")
        click.echo(f"  Slices: {len(vm_groups)}")

        for vm_group in vm_groups:
            status = vm_group.status()
            status_str = "READY" if status.all_ready else ("FAILED" if status.any_failed else "PENDING")
            click.echo(f"\n    Slice: {vm_group.slice_id} [{status_str}]")
            click.echo(f"      Created: {format_timestamp(vm_group.created_at_ms)}")

            for vm_item in vm_group.vms():
                state = vm_state_name(vm_item.info.state)
                addr = vm_item.info.address or "no address"
                click.echo(f"        {vm_item.info.vm_id}: {state} ({addr})")
                if vm_item.info.init_error:
                    click.echo(f"          Error: {vm_item.info.init_error}")


@vm.command("logs")
@click.argument("vm_id")
@click.option("--tail", type=int, default=0, help="Show last N lines (0 = all)")
@click.pass_context
def vm_logs(ctx, vm_id: str, tail: int):
    """Show VM initialization logs."""
    controller_url = ctx.obj.get("controller_url")
    config_file = ctx.obj.get("config_file")

    if controller_url:
        _logs_via_controller(controller_url, vm_id, tail)
    elif config_file:
        _logs_via_autoscaler(config_file, vm_id, tail)
    else:
        click.echo("Error: Either --controller-url or cluster --config is required", err=True)
        raise SystemExit(1)


def _logs_via_controller(controller_url: str, vm_id: str, tail: int):
    """Show logs via controller RPC."""
    try:
        log_content, returned_vm_id, state = get_vm_logs(controller_url, vm_id, tail)
    except ConnectError as e:
        from connectrpc.code import Code

        if e.code == Code.NOT_FOUND:
            click.echo(f"VM not found: {vm_id}", err=True)
        else:
            click.echo(f"Error fetching logs: {e}", err=True)
        raise SystemExit(1) from None
    except Exception as e:
        click.echo(f"Error connecting to controller: {e}", err=True)
        raise SystemExit(1) from None

    click.echo(f"VM: {returned_vm_id}")
    click.echo(f"State: {vm_state_name(state)}")
    click.echo("---")
    if log_content:
        click.echo(log_content)
    else:
        click.echo("(no logs available)")


def _logs_via_autoscaler(config_file: str, vm_id: str, tail: int):
    """Show logs via direct Autoscaler."""
    autoscaler_obj, _ = load_autoscaler(config_file)

    for group in autoscaler_obj.groups.values():
        for vm_group in group.vm_groups():
            for vm_item in vm_group.vms():
                if vm_item.info.vm_id == vm_id:
                    log_content = vm_item.init_log(tail if tail > 0 else None)
                    click.echo(f"VM: {vm_id}")
                    click.echo(f"State: {vm_state_name(vm_item.info.state)}")
                    click.echo("---")
                    if log_content:
                        click.echo(log_content)
                    else:
                        click.echo("(no logs available)")
                    return

    click.echo(f"Error: VM {vm_id} not found", err=True)
    raise SystemExit(1)


@vm.command("get")
@click.argument("vm_id")
@click.pass_context
def vm_get(ctx, vm_id: str):
    """Get detailed info for a specific VM."""
    config_file = require_config(ctx)
    autoscaler_obj, _ = load_autoscaler(config_file)

    for group_name, group in autoscaler_obj.groups.items():
        for vm_group in group.vm_groups():
            for vm_item in vm_group.vms():
                if vm_item.info.vm_id == vm_id:
                    state = vm_state_name(vm_item.info.state)
                    click.echo(f"VM: {vm_item.info.vm_id}")
                    click.echo(f"Slice: {vm_item.info.slice_id}")
                    click.echo(f"Scale Group: {group_name}")
                    click.echo(f"State: {state}")
                    click.echo(f"Address: {vm_item.info.address or '(none)'}")
                    if vm_item.info.zone:
                        click.echo(f"Zone: {vm_item.info.zone}")
                    click.echo(f"Created: {format_timestamp(vm_item.info.created_at_ms)}")
                    click.echo(f"State Changed: {format_timestamp(vm_item.info.state_changed_at_ms)}")

                    if vm_item.info.worker_id:
                        click.echo(f"Worker ID: {vm_item.info.worker_id}")
                        click.echo(f"Worker Healthy: {vm_item.info.worker_healthy}")

                    if vm_item.info.init_phase:
                        click.echo(f"Init Phase: {vm_item.info.init_phase}")
                    if vm_item.info.init_error:
                        click.echo(f"Init Error: {vm_item.info.init_error}")
                    return

    click.echo(f"Error: VM {vm_id} not found", err=True)
    raise SystemExit(1)
