# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared presentation and target parsing for resource commands."""

import uuid

import click

from iris.resources.action import ActionReceipt
from iris.resources.activity import ActivityEntry
from iris.resources.identity import AttemptLocator, ResourceKey, ResourceKind
from iris.resources.log import LogEntry, LogLevel
from iris.resources.source import ResourceSourceStatus, SourceState
from iris.resources.state import JobState, TaskState
from iris.rpc.proto_display import job_state_friendly, task_state_friendly

DEFAULT_RESOURCE_LIST_LIMIT = 100
MAX_RESOURCE_LIST_LIMIT = 10_000
DEFAULT_ACTIVITY_LIMIT = 200
MAX_ACTIVITY_LIMIT = 10_000
DEFAULT_LOG_LINES = 1_000
MAX_LOG_LINES = 100_000
LOG_LEVEL_NAMES = tuple(level.name.lower() for level in LogLevel)


def cluster_id(ctx: click.Context) -> str:
    obj = ctx.obj or {}
    config = obj.get("config")
    if config is not None and config.name:
        return config.name
    return obj.get("cluster_name") or "default"


def resource_key(ctx: click.Context, kind: ResourceKind, resource_id: str) -> ResourceKey:
    return ResourceKey(cluster_id(ctx), kind, resource_id)


def attempt_locator(ctx: click.Context, value: str) -> AttemptLocator:
    task_id, separator, raw_number = value.rpartition(":")
    if not separator:
        return AttemptLocator(resource_key(ctx, ResourceKind.TASK, value), None)
    if not task_id or not raw_number.isdecimal():
        raise click.BadParameter("expected TASK_ID or TASK_ID:ATTEMPT_NUMBER")
    return AttemptLocator(resource_key(ctx, ResourceKind.TASK, task_id), int(raw_number))


def action_key(value: str | None) -> str:
    return value or uuid.uuid4().hex


def log_level(value: str) -> LogLevel:
    return LogLevel[value.upper()]


def echo_action(receipt: ActionReceipt) -> None:
    click.echo(f"Action: {receipt.action_id}")
    click.echo(f"State: {receipt.state.value}")
    click.echo(f"Result: {receipt.result_code.value}")
    if receipt.result_message:
        click.echo(f"Message: {receipt.result_message}")


def echo_source_warnings(statuses: tuple[ResourceSourceStatus, ...]) -> None:
    for status in statuses:
        if status.state is SourceState.AVAILABLE:
            continue
        source = status.backend_id or status.source_id
        message = status.error_message or status.error_code or status.state.value
        click.echo(f"Warning: {source}: {message}", err=True)


def echo_activity(entries: tuple[ActivityEntry, ...]) -> None:
    if not entries:
        click.echo("No activity found.")
        return
    for entry in entries:
        click.echo(f"{entry.occurred_at.epoch_ms()}  {entry.severity:<8} {entry.kind:<20} {entry.message}")


def echo_logs(entries: tuple[LogEntry, ...]) -> None:
    for entry in entries:
        prefix = f"{entry.timestamp} " if entry.timestamp is not None else ""
        source = f"{entry.source}: " if entry.source else ""
        click.echo(f"{prefix}{source}{entry.data}")


def job_state_name(state: JobState) -> str:
    return job_state_friendly(state)


def task_state_name(state: TaskState) -> str:
    return task_state_friendly(state)
