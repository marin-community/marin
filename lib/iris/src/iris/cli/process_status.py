# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for process status, logs, and profiling.

Provides ``iris controller <status|logs|profile>`` and
``iris worker <status|logs|profile>`` commands. Both share the same
implementation, differing only in which RPC service they target.
"""

import sys

import click
from google.protobuf import json_format

from iris.cli.main import require_controller_url
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync, WorkerServiceClientSync


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


def _format_uptime(ms: int) -> str:
    """Format milliseconds as human-readable uptime."""
    seconds = ms // 1000
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m {seconds % 60}s"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h {minutes % 60}m"
    days = hours // 24
    return f"{days}d {hours % 24}h {minutes % 60}m"


def _get_status(url: str, service: str) -> cluster_pb2.GetProcessStatusResponse:
    """Call GetProcessStatus on the specified service."""
    if service == "controller":
        client = ControllerServiceClientSync(url)
    else:
        client = WorkerServiceClientSync(url)
    return client.get_process_status(cluster_pb2.GetProcessStatusRequest(max_log_lines=0))


def _print_status(resp: cluster_pb2.GetProcessStatusResponse, service_name: str) -> None:
    """Print process status to stdout."""
    info = resp.process_info
    click.echo(f"=== {service_name} Process Status ===")
    click.echo(f"Hostname:        {info.hostname}")
    click.echo(f"PID:             {info.pid}")
    click.echo(f"Python:          {info.python_version}")
    click.echo(f"Uptime:          {_format_uptime(info.uptime_ms)}")
    click.echo(f"CPU:             {info.cpu_percent:.1f}% ({info.cpu_count} cores)")
    click.echo(f"Memory RSS:      {_format_bytes(info.memory_rss_bytes)}")
    click.echo(f"Memory VMS:      {_format_bytes(info.memory_vms_bytes)}")
    click.echo(f"Memory Total:    {_format_bytes(info.memory_total_bytes)}")
    click.echo(f"Threads:         {info.thread_count}")
    click.echo(f"Open FDs:        {info.open_fd_count}")


def _fetch_and_print_logs(
    url: str,
    service: str,
    level: str,
    follow: bool,
    max_lines: int,
    substring: str,
) -> None:
    """Fetch and print process logs."""
    if service == "controller":
        client = ControllerServiceClientSync(url)
    else:
        client = WorkerServiceClientSync(url)

    import time

    cursor = 0
    first = True
    while True:
        req = cluster_pb2.FetchLogsRequest(
            source="/system/process",
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
                from datetime import datetime, timezone

                dt = datetime.fromtimestamp(entry.timestamp.epoch_ms / 1000, tz=timezone.utc)
                ts = dt.strftime("%H:%M:%S")
            click.echo(f"[{ts}] {entry.data}")

        cursor = resp.cursor
        first = False

        if not follow:
            break
        time.sleep(2)


def _run_profile(
    url: str,
    service: str,
    profiler_type: str,
    duration: int,
    output: str | None,
) -> None:
    """Run a profile and write output to file or stdout."""
    if service == "controller":
        client = ControllerServiceClientSync(url)
    else:
        client = WorkerServiceClientSync(url)

    # Build profile type
    if profiler_type == "threads":
        profile_type = cluster_pb2.ProfileType(threads=cluster_pb2.ThreadsProfile())
    elif profiler_type == "cpu":
        profile_type = cluster_pb2.ProfileType(
            cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.SPEEDSCOPE)
        )
    elif profiler_type == "mem":
        profile_type = cluster_pb2.ProfileType(
            memory=cluster_pb2.MemoryProfile(format=cluster_pb2.MemoryProfile.FLAMEGRAPH)
        )
    else:
        raise click.ClickException(f"Unknown profiler type: {profiler_type}")

    click.echo(f"Profiling {service} process ({profiler_type}, {duration}s)...")
    resp = client.profile_task(
        cluster_pb2.ProfileTaskRequest(
            target="/system/process",
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
    else:
        # Write to stdout for text formats, or suggest --output for binary
        if profiler_type == "threads":
            click.echo(resp.profile_data.decode("utf-8"))
        else:
            ext = {"cpu": ".speedscope.json", "mem": ".html"}[profiler_type]
            default_name = f"profile-{profiler_type}-{service}{ext}"
            with open(default_name, "wb") as f:
                f.write(resp.profile_data)
            click.echo(f"Profile written to {default_name}")


def _build_service_group(service_name: str) -> click.Group:
    """Build a CLI group for a service (controller or worker) with status/logs/profile commands."""

    @click.group(name=service_name)
    def group():
        pass

    group.help = f"{service_name.capitalize()} process status, logs, and profiling."

    @group.command()
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    @click.pass_context
    def status(ctx, as_json: bool):
        """Show process status (host info, resource usage)."""
        url = require_controller_url(ctx)
        resp = _get_status(url, service_name)
        if as_json:
            click.echo(json_format.MessageToJson(resp.process_info, preserving_proto_field_name=True, indent=2))
        else:
            _print_status(resp, service_name.capitalize())

    @group.command()
    @click.option("--level", default="", help="Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)")
    @click.option("--follow", "-f", is_flag=True, help="Stream logs continuously")
    @click.option("--max-lines", default=200, help="Max lines to show")
    @click.option("--substring", default="", help="Substring filter")
    @click.pass_context
    def logs(ctx, level: str, follow: bool, max_lines: int, substring: str):
        """Show process logs."""
        url = require_controller_url(ctx)
        _fetch_and_print_logs(url, service_name, level, follow, max_lines, substring)

    @group.command()
    @click.argument("profiler", type=click.Choice(["threads", "cpu", "mem"]))
    @click.option("--duration", "-d", default=10, help="Profiling duration in seconds")
    @click.option("--output", "-o", default=None, help="Output file path")
    @click.pass_context
    def profile(ctx, profiler: str, duration: int, output: str | None):
        """Profile the process (threads, cpu, or mem)."""
        url = require_controller_url(ctx)
        _run_profile(url, service_name, profiler, duration, output)

    return group


controller_group = _build_service_group("controller")
worker_group = _build_service_group("worker")


def register_process_status_commands(iris_group: click.Group) -> None:
    """Register controller and worker process status CLI groups."""
    iris_group.add_command(controller_group)
    iris_group.add_command(worker_group)
