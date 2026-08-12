# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for process status, logs, and profiling.

Provides ``iris process <status|logs|profile>`` with ``--target`` to address
a specific process by its RPC path (e.g. ``/system/worker/<id>`` for a worker,
``/alice/job/0`` for a task container). Omitting ``--target`` defaults to the
controller itself.
"""

import contextlib
import time
from datetime import UTC, datetime

import click
import humanfriendly
from finelog.client import LogClient
from finelog.rpc import logging_pb2
from rigging.connect import proxy_path
from rigging.timing import Duration

from iris.cli.connect import require_controller_url, resource_client_for_ctx, rpc_client_for_ctx
from iris.cli.resource_commands import attempt_locator
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.runtime.profile import SYSTEM_PROCESS_TARGET
from iris.resources.endpoint import (
    CpuProfileConfiguration,
    CpuProfileFormat,
    MemoryProfileConfiguration,
    MemoryProfileFormat,
    ThreadsProfileConfiguration,
)
from iris.resources.system import ProcessInfo
from iris.rpc import job_pb2
from iris.rpc.profile_codec import profile_configuration_to_proto
from iris.rpc.worker_codec import process_info_from_proto

_CONTROLLER_LOG_TARGET = "/system/controller"


def _format_cpu_millicores(millicores: int) -> str:
    """Format CPU usage in cores with the raw millicore value."""
    return f"{millicores / 1000:g} cores ({millicores}m)"


def _print_status(info: ProcessInfo, label: str) -> None:
    """Print process status to stdout in human-readable form."""
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
@click.pass_context
def status(ctx, target: str | None):
    """Show process status (host info, resource usage)."""
    url = require_controller_url(ctx)
    label = target or "Controller"
    with rpc_client_for_ctx(ctx, url=url) as client:
        # GetProcessStatus uses empty string for controller
        resp = client.get_process_status(job_pb2.GetProcessStatusRequest(max_log_lines=0, target=target or ""))
    _print_status(process_info_from_proto(resp.process_info), label)


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
    url = require_controller_url(ctx)
    source = target or _CONTROLLER_LOG_TARGET
    credentials = ctx.obj.get("credentials") if ctx.obj else None
    interceptors = credentials.interceptors() if credentials is not None else ()

    log_server_url = f"{url.rstrip('/')}{proxy_path(LOG_SERVER_ENDPOINT_NAME)}"
    with contextlib.closing(LogClient.connect(log_server_url, interceptors=interceptors)) as log_client:
        cursor = 0
        first = True
        while True:
            req = logging_pb2.FetchLogsRequest(
                source=source,
                match_scope=logging_pb2.MATCH_SCOPE_EXACT,
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
                    dt = datetime.fromtimestamp(entry.timestamp.epoch_ms / 1000, tz=UTC)
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
@click.option("--native", "include_native", is_flag=True, help="Include native frames in thread dump")
@click.pass_context
def profile(
    ctx,
    target: str | None,
    profiler: str,
    duration: int,
    output: str | None,
    include_locals: bool,
    include_native: bool,
):
    """Profile the process (threads, cpu, or mem).

    By default profiles the controller. Use --target with the full RPC path:
    /system/worker/<id> for a worker, /alice/job/0 for a task container.
    """
    url = require_controller_url(ctx)
    rpc_target = target or SYSTEM_PROCESS_TARGET
    label = target or "Controller"

    if profiler == "threads":
        profile_configuration = ThreadsProfileConfiguration(
            include_locals=include_locals,
            include_native=include_native,
        )
    elif profiler == "cpu":
        profile_configuration = CpuProfileConfiguration(format=CpuProfileFormat.SPEEDSCOPE, rate_hz=0, native=None)
    elif profiler == "mem":
        profile_configuration = MemoryProfileConfiguration(format=MemoryProfileFormat.FLAMEGRAPH, leaks=False)
    else:
        raise click.ClickException(f"Unknown profiler type: {profiler}")

    click.echo(f"Profiling {label} ({profiler}, {duration}s)...")
    if rpc_target.startswith("/system/"):
        with rpc_client_for_ctx(ctx, url=url) as client:
            resp = client.profile_task(
                job_pb2.ProfileTaskRequest(
                    target=rpc_target,
                    duration_seconds=duration,
                    profile_type=profile_configuration_to_proto(profile_configuration),
                )
            )
        profile_data = resp.profile_data
        error = resp.error
    else:
        with resource_client_for_ctx(ctx) as client:
            attempt = client.describe_attempt(attempt_locator(ctx, rpc_target)).summary.identity
            result = client.profile_attempt(
                attempt,
                profile=profile_configuration,
                duration=Duration.from_seconds(duration),
            )
        profile_data = result.profile_data
        error = result.error_message

    if error:
        raise click.ClickException(f"Profiling failed: {error}")

    if output:
        with open(output, "wb") as f:
            f.write(profile_data)
        click.echo(f"Profile written to {output}")
    elif profiler == "threads":
        click.echo(profile_data.decode("utf-8"))
    else:
        ext = {"cpu": ".speedscope.json", "mem": ".html"}[profiler]
        safe_label = label.lower().replace(" ", "-").replace("/", "-").strip("-")
        default_name = f"profile-{profiler}-{safe_label}{ext}"
        with open(default_name, "wb") as f:
            f.write(profile_data)
        click.echo(f"Profile written to {default_name}")


def register_process_status_commands(iris_group: click.Group) -> None:
    """Register the ``iris process`` CLI group."""
    iris_group.add_command(process_group)
