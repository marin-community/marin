# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared workload-log input and output handling."""

import click
from rigging.timing import Timestamp

from iris.client.client import TaskLogEntry


def log_start(since_ms: int | None, since_seconds: int | None) -> Timestamp | None:
    if since_ms is not None and since_seconds is not None:
        raise click.UsageError("Specify only one of --since-ms or --since-seconds.")
    if since_seconds is not None:
        return Timestamp.from_ms(Timestamp.now().epoch_ms() - since_seconds * 1_000)
    return Timestamp.from_ms(since_ms) if since_ms is not None else None


def echo_log_entries(entries: list[TaskLogEntry]) -> None:
    for entry in entries:
        click.echo(f"[{entry.timestamp.as_short_time()}] task={entry.task_id} attempt={entry.attempt_id} | {entry.data}")
