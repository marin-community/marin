# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for process status, logs, and profiling.

Provides ``iris process <status|logs|profile>`` with ``--target`` to address
a specific process by its RPC path (e.g. ``/system/worker/<id>`` for a worker,
``/alice/job/0`` for a task container). Omitting ``--target`` defaults to the
controller itself.
"""

import click
import humanfriendly
from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync

from iris.cli.main import require_controller_url, rpc_client
from iris.cluster.runtime.profile import SYSTEM_PROCESS_TARGET
from iris.rpc import job_pb2

_CONTROLLER_LOG_TARGET = "/system/controller"


def _format_cpu_millicores(millicores: int) -> str:
    """Format CPU usage in cores with the raw millicore value."""
    return f"{millicores / 1000:g} cores ({millicores}m)"


def _print_status(resp: job_pb2.GetProcessStatusResponse, label: str) -> None:
    """Print process status to stdout in human-readable form."""
    info = resp.process_info
    click.echo(f"=== {label} Process Status ===")
    click.echo(f"Hostname:        {info.hostname}")
    click.echo(f"PID:             {info.pid}")
    click.echo(f"Python:          {info.python_version}")
    click.echo(f"Uptime:          {humanfriendly.format_timespan(info.uptime_ms / 1000)}")
    click.echo(f"CPU:             {_format_cpu_millicores(info.cpu_millicores)}")
    click.echo(f"Memory RSS:      {humanfriendly.format_size(info.memory_rss_bytes, binary=True)}")
    click.echo(f"Memory VMS:      {humanfriendly.format_size(info.memory_vms_bytes, binary=True)}")
    click.echo(f"Memory Total:    {humanfriendly.format_size(info.memory_total_bytes, binary=True)}")
    click.echo(f"Threads:         {info.thread_count}")
    click.echo(f"Open FDs:        {info.open_fd_count}")


@click.group(name="process")
def process_group():
    """Process status, logs, and profiling for controller or workers."""


@process_group.command()
@click.option(
    "--target",
    "-t",
    default=None,
    help="RPC target path, e.g. /system/worker/<id> or /alice/job/0 (default: controller)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx, target: str | None, as_json: bool):
    """Show process status (host info, resource usage)."""
    from google.protobuf import json_format

    url = require_controller_url(ctx)
    label = target or "Controller"
    with rpc_client(url) as client:
        # GetProcessStatus uses empty string for controller
        resp = client.get_process_status(job_pb2.GetProcessStatusRequest(max_log_lines=0, target=target or ""))
    if as_json:
        click.echo(json_format.MessageToJson(resp.process_info, preserving_proto_field_name=True, indent=2))
    else:
        _print_status(resp, label)


@process_group.command()
@click.option(
    "--target",
    "-t",
    default=None,
    help="RPC target path, e.g. /system/worker/<id> (default: controller)",
)
@click.option("--level", default="", help="Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)")
@click.option("--follow", "-f", is_flag=True, help="Stream logs continuously")
@click.option("--max-lines", default=200, help="Max lines to show")
@click.option("--substring", default="", help="Substring filter")
@click.pass_context
def logs(ctx, target: str | None, level: str, follow: bool, max_lines: int, substring: str):
    """Show process logs."""
    import time
    from datetime import datetime, timezone

    url = require_controller_url(ctx)
    log_client = LogServiceClientSync(url)
    source = target or _CONTROLLER_LOG_TARGET

    cursor = 0
    first = True
    while True:
        req = logging_pb2.FetchLogsRequest(
            source=source,
            max_lines=max_lines if first else 100,
            tail=first,
            min_level=level,
            cursor=cursor if not first else 0,
        )
        if substring:
            req.substring = substring

        resp = log_client.fetch_logs(req)
        for entry in resp.entries:
            ts = ""
            if entry.timestamp and entry.timestamp.epoch_ms:
                dt = datetime.fromtimestamp(entry.timestamp.epoch_ms / 1000, tz=timezone.utc)
                ts = dt.strftime("%H:%M:%S")
            click.echo(f"[{ts}] {entry.data}")

        cursor = resp.cursor
        first = False

        if not follow:
            break
        time.sleep(2)


@process_group.command()
@click.option(
    "--target",
    "-t",
    default=None,
    help="RPC target path, e.g. /system/worker/<id> or /alice/job/0 (default: controller)",
)
@click.argument("profiler", type=click.Choice(["threads", "cpu", "mem"]))
@click.option("--duration", "-d", default=10, help="Profiling duration in seconds")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--locals", "include_locals", is_flag=True, help="Include local variables in thread dump")
@click.pass_context
def profile(
    ctx,
    target: str | None,
    profiler: str,
    duration: int,
    output: str | None,
    include_locals: bool,
):
    """Profile the process (threads, cpu, or mem).

    By default profiles the controller. Use --target with the full RPC path:
    /system/worker/<id> for a worker, /alice/job/0 for a task container.
    """
    url = require_controller_url(ctx)
    rpc_target = target or SYSTEM_PROCESS_TARGET
    label = target or "Controller"

    if profiler == "threads":
        profile_type = job_pb2.ProfileType(threads=job_pb2.ThreadsProfile(locals=include_locals))
    elif profiler == "cpu":
        profile_type = job_pb2.ProfileType(cpu=job_pb2.CpuProfile(format=job_pb2.CpuProfile.SPEEDSCOPE))
    elif profiler == "mem":
        profile_type = job_pb2.ProfileType(memory=job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH))
    else:
        raise click.ClickException(f"Unknown profiler type: {profiler}")

    click.echo(f"Profiling {label} ({profiler}, {duration}s)...")
    with rpc_client(url) as client:
        resp = client.profile_task(
            job_pb2.ProfileTaskRequest(
                target=rpc_target,
                duration_seconds=duration,
                profile_type=profile_type,
            )
        )

    if resp.error:
        raise click.ClickException(f"Profiling failed: {resp.error}")

    if output:
        with open(output, "wb") as f:
            f.write(resp.profile_data)
        click.echo(f"Profile written to {output}")
    elif profiler == "threads":
        click.echo(resp.profile_data.decode("utf-8"))
    else:
        ext = {"cpu": ".speedscope.json", "mem": ".html"}[profiler]
        safe_label = label.lower().replace(" ", "-").replace("/", "-").strip("-")
        default_name = f"profile-{profiler}-{safe_label}{ext}"
        with open(default_name, "wb") as f:
            f.write(resp.profile_data)
        click.echo(f"Profile written to {default_name}")


def register_process_status_commands(iris_group: click.Group) -> None:
    """Register the ``iris process`` CLI group."""
    iris_group.add_command(process_group)
