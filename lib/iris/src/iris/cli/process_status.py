# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for process status, logs, and profiling.

Provides ``iris process <status|logs|profile>`` with ``--worker=<id>`` to
target a specific worker (proxied through the controller) or the controller
itself (default).
"""


import click
import humanfriendly

from iris.cli.main import require_controller_url
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync


def _print_status(resp: cluster_pb2.GetProcessStatusResponse, label: str) -> None:
    """Print process status to stdout in human-readable form."""
    info = resp.process_info
    click.echo(f"=== {label} Process Status ===")
    click.echo(f"Hostname:        {info.hostname}")
    click.echo(f"PID:             {info.pid}")
    click.echo(f"Python:          {info.python_version}")
    click.echo(f"Uptime:          {humanfriendly.format_timespan(info.uptime_ms / 1000)}")
    click.echo(f"CPU:             {info.cpu_percent:.1f}% ({info.cpu_count} cores)")
    click.echo(f"Memory RSS:      {humanfriendly.format_size(info.memory_rss_bytes, binary=True)}")
    click.echo(f"Memory VMS:      {humanfriendly.format_size(info.memory_vms_bytes, binary=True)}")
    click.echo(f"Memory Total:    {humanfriendly.format_size(info.memory_total_bytes, binary=True)}")
    click.echo(f"Threads:         {info.thread_count}")
    click.echo(f"Open FDs:        {info.open_fd_count}")


@click.group(name="process")
def process_group():
    """Process status, logs, and profiling for controller or workers."""


@process_group.command()
@click.option("--worker", "-w", default=None, help="Worker ID to target (default: controller)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx, worker: str | None, as_json: bool):
    """Show process status (host info, resource usage)."""
    from google.protobuf import json_format

    url = require_controller_url(ctx)
    client = ControllerServiceClientSync(url)
    target = f"/system/worker/{worker}" if worker else ""
    resp = client.get_process_status(cluster_pb2.GetProcessStatusRequest(max_log_lines=0, target=target))
    label = f"Worker {worker}" if worker else "Controller"
    if as_json:
        click.echo(json_format.MessageToJson(resp.process_info, preserving_proto_field_name=True, indent=2))
    else:
        _print_status(resp, label)


@process_group.command()
@click.option("--worker", "-w", default=None, help="Worker ID to target (default: controller)")
@click.option("--level", default="", help="Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)")
@click.option("--follow", "-f", is_flag=True, help="Stream logs continuously")
@click.option("--max-lines", default=200, help="Max lines to show")
@click.option("--substring", default="", help="Substring filter")
@click.pass_context
def logs(ctx, worker: str | None, level: str, follow: bool, max_lines: int, substring: str):
    """Show process logs."""
    import time
    from datetime import datetime, timezone

    url = require_controller_url(ctx)
    client = ControllerServiceClientSync(url)

    # Use /system/worker/<id> as the log source so the controller proxies to the worker
    source = f"/system/worker/{worker}" if worker else "/system/process"

    cursor = 0
    first = True
    while True:
        req = cluster_pb2.FetchLogsRequest(
            source=source,
            max_lines=max_lines if first else 100,
            tail=first,
            min_level=level,
            cursor=cursor if not first else 0,
        )
        if substring:
            req.substring = substring

        resp = client.fetch_logs(req)
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
@click.option("--worker", "-w", default=None, help="Worker ID to target (default: controller)")
@click.argument("profiler", type=click.Choice(["threads", "cpu", "mem"]))
@click.option("--duration", "-d", default=10, help="Profiling duration in seconds")
@click.option("--output", "-o", default=None, help="Output file path")
@click.pass_context
def profile(ctx, worker: str | None, profiler: str, duration: int, output: str | None):
    """Profile the process (threads, cpu, or mem)."""
    url = require_controller_url(ctx)
    client = ControllerServiceClientSync(url)

    # Build profile type
    if profiler == "threads":
        profile_type = cluster_pb2.ProfileType(threads=cluster_pb2.ThreadsProfile())
    elif profiler == "cpu":
        profile_type = cluster_pb2.ProfileType(cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.SPEEDSCOPE))
    elif profiler == "mem":
        profile_type = cluster_pb2.ProfileType(
            memory=cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH)
        )
    else:
        raise click.ClickException(f"Unknown profiler type: {profiler}")

    # Target: /system/worker/<id> for workers, /system/process for controller
    target = f"/system/worker/{worker}" if worker else "/system/process"
    label = f"Worker {worker}" if worker else "Controller"

    click.echo(f"Profiling {label} ({profiler}, {duration}s)...")
    resp = client.profile_task(
        cluster_pb2.ProfileTaskRequest(
            target=target,
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
        default_name = f"profile-{profiler}-{label.lower().replace(' ', '-')}{ext}"
        with open(default_name, "wb") as f:
            f.write(resp.profile_data)
        click.echo(f"Profile written to {default_name}")


def register_process_status_commands(iris_group: click.Group) -> None:
    """Register the ``iris process`` CLI group."""
    iris_group.add_command(process_group)
