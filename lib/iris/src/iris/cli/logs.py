# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared workload-log input, streaming, and output handling."""

from collections.abc import Iterable

import click
from rigging.timing import Timestamp

from iris.client.client import Attempt, Job, Task, TaskLogEntry


def workload_log_options(command):
    """Apply the log filters shared by Job, Task, and Attempt commands."""
    options = [
        click.option("--follow", "-f", is_flag=True, help="Continue reading until the selected resource finishes."),
        click.option(
            "--since-ms",
            type=int,
            default=None,
            help="Only show logs after this epoch millisecond timestamp.",
        ),
        click.option("--since-seconds", type=int, default=None, help="Only show logs from the last N seconds."),
        click.option("--max-lines", type=int, default=0, help="Maximum lines per fetch; zero uses the server default."),
        click.option("--tail/--no-tail", default=True, help="Begin with the most recent lines instead of the earliest."),
        click.option(
            "--level",
            type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
            default=None,
            help="Minimum log level to return.",
        ),
        click.option("--substring", default="", help="Only return lines containing this text."),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def log_start(since_ms: int | None, since_seconds: int | None) -> Timestamp | None:
    if since_ms is not None and since_seconds is not None:
        raise click.UsageError("Specify only one of --since-ms or --since-seconds.")
    if since_seconds is not None:
        return Timestamp.from_ms(Timestamp.now().epoch_ms() - since_seconds * 1_000)
    return Timestamp.from_ms(since_ms) if since_ms is not None else None


def echo_log_entries(entries: Iterable[TaskLogEntry]) -> None:
    for entry in entries:
        click.echo(f"[{entry.timestamp.as_short_time()}] task={entry.task_id} attempt={entry.attempt_id} | {entry.data}")


def echo_workload_logs(
    handle: Job | Task | Attempt,
    *,
    since_ms: int | None,
    since_seconds: int | None,
    follow: bool,
    max_lines: int,
    tail: bool,
    level: str | None,
    substring: str,
) -> None:
    """Read or follow logs with the common workload command semantics."""
    start = log_start(since_ms, since_seconds)
    min_level = level.upper() if level else ""
    if follow:
        entries = handle.follow_logs(
            start=start,
            max_lines=max_lines,
            tail=tail,
            min_level=min_level,
            substring=substring,
        )
    else:
        entries = handle.logs(
            start=start,
            max_lines=max_lines,
            tail=tail,
            min_level=min_level,
            substring=substring,
        )
    echo_log_entries(entries)
