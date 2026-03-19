# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for process status, logs, and profiling.

Provides ``iris process <status|logs|profile>`` with ``--target`` to address
a specific worker (plain worker ID like ``abc123`` maps to
``/system/worker/abc123``) or a task container (full path like
``/alice/job/0``). Omitting ``--target`` defaults to the controller itself.
"""


import click
import humanfriendly

from iris.cli.main import require_controller_url
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync


def _resolve_profile_target(target: str | None) -> tuple[str, str]:
    """Resolve --target to (rpc_target, label).

    - ``None`` → controller (``/system/process``)
    - starts with ``/`` → raw path passed directly (task or system path)
    - plain string → worker ID → ``/system/worker/<id>``

    Returns:
        (target, label) where target is passed to ProfileTask and
        label is used for display and default output filename.
    """
    if target is None:
        return "/system/process", "Controller"
    if target.startswith("/"):
        return target, target
    return f"/system/worker/{target}", f"Worker {target}"


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
@click.option("--target", "-t", default=None, help="Worker ID or task path (default: controller)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx, target: str | None, as_json: bool):
    """Show process status (host info, resource usage)."""
    from google.protobuf import json_format

    url = require_controller_url(ctx)
    client = ControllerServiceClientSync(url)
    rpc_target, label = _resolve_profile_target(target)
    # GetProcessStatus uses empty string for controller (not /system/process)
    get_target = "" if target is None else rpc_target
    resp = client.get_process_status(cluster_pb2.GetProcessStatusRequest(max_log_lines=0, target=get_target))
    if as_json:
        click.echo(json_format.MessageToJson(resp.process_info, preserving_proto_field_name=True, indent=2))
    else:
        _print_status(resp, label)


@process_group.command()
@click.option("--target", "-t", default=None, help="Worker ID or task path (default: controller)")
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
    client = ControllerServiceClientSync(url)

    source, _ = _resolve_profile_target(target)

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
@click.option(
    "--target",
    "-t",
    default=None,
    help="Worker ID, task path (e.g. /alice/job/0), or omit for controller",
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

    By default profiles the controller process. Use --target with a worker ID
    to profile a worker, or a task path (e.g. /alice/job/0) for a task container.
    """
    url = require_controller_url(ctx)
    client = ControllerServiceClientSync(url)

    # Build profile type
    if profiler == "threads":
        profile_type = cluster_pb2.ProfileType(threads=cluster_pb2.ThreadsProfile(locals=include_locals))
    elif profiler == "cpu":
        profile_type = cluster_pb2.ProfileType(cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.SPEEDSCOPE))
    elif profiler == "mem":
        profile_type = cluster_pb2.ProfileType(
            memory=cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH)
        )
    else:
        raise click.ClickException(f"Unknown profiler type: {profiler}")

    rpc_target, label = _resolve_profile_target(target)

    click.echo(f"Profiling {label} ({profiler}, {duration}s)...")
    resp = client.profile_task(
        cluster_pb2.ProfileTaskRequest(
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
