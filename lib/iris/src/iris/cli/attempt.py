# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact Attempt inspection and actions."""

from enum import StrEnum

import click
from rigging.timing import Duration

from iris.cli.connect import resource_client_for_ctx
from iris.cli.resource_commands import (
    DEFAULT_ACTIVITY_LIMIT,
    DEFAULT_LOG_LINES,
    MAX_ACTIVITY_LIMIT,
    MAX_LOG_LINES,
    action_key,
    attempt_locator,
    echo_action,
    echo_activity,
    echo_logs,
    task_state_name,
)
from iris.cluster.resources.activity import ActivityQuery
from iris.cluster.resources.log import LogQuery
from iris.rpc import job_pb2


class ProfileName(StrEnum):
    CPU = "cpu"
    MEMORY = "memory"
    THREAD = "thread"


@click.group()
def attempt() -> None:
    """Inspect and act on exact Task Attempts."""


@attempt.command("describe")
@click.argument("attempt_ref")
@click.pass_context
def attempt_describe(ctx: click.Context, attempt_ref: str) -> None:
    """Describe ATTEMPT_REF, written as TASK_ID or TASK_ID:ATTEMPT_NUMBER."""
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_attempt(attempt_locator(ctx, attempt_ref))
    summary = detail.summary
    click.echo(f"Attempt: {summary.identity.task.resource_id}:{summary.identity.attempt_number}")
    click.echo(f"UID: {summary.identity.attempt_uid}")
    click.echo(f"State: {task_state_name(summary.state)}")
    click.echo(f"Backend: {summary.backend_id or '-'}")
    if summary.node is not None:
        click.echo(f"Node: {summary.node.key.resource_id}")
    if detail.runtime is not None:
        click.echo(f"Runtime: {detail.runtime.provider_kind}:{detail.runtime.namespace}/{detail.runtime.name}")
    if summary.exit_code is not None:
        click.echo(f"Exit code: {summary.exit_code}")
    if summary.terminal_reason:
        click.echo(f"Reason: {summary.terminal_reason}")
    if summary.error_message:
        click.echo(f"Error: {summary.error_message}")


@attempt.command("logs")
@click.argument("attempt_ref")
@click.option("--tail", is_flag=True, help="Continue reading new log entries.")
@click.option("--max-lines", type=click.IntRange(1, MAX_LOG_LINES), default=DEFAULT_LOG_LINES, show_default=True)
@click.pass_context
def attempt_logs(ctx: click.Context, attempt_ref: str, tail: bool, max_lines: int) -> None:
    """Read logs for one exact Attempt."""
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_attempt(attempt_locator(ctx, attempt_ref))
        query = LogQuery(max_lines=max_lines, tail=tail)
        if tail:
            for entry in client.stream_attempt_logs(detail.summary.identity, query):
                echo_logs((entry,))
            return
        echo_logs(client.fetch_attempt_logs(detail.summary.identity, query).entries)


@attempt.command("activity")
@click.argument("attempt_ref")
@click.option(
    "--limit",
    type=click.IntRange(1, MAX_ACTIVITY_LIMIT),
    default=DEFAULT_ACTIVITY_LIMIT,
    show_default=True,
)
@click.pass_context
def attempt_activity(ctx: click.Context, attempt_ref: str, limit: int) -> None:
    """Show authorized activity for one exact Attempt."""
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_attempt(attempt_locator(ctx, attempt_ref))
        page = client.list_activity(
            ActivityQuery(
                target=detail.summary.identity.task,
                attempt_uid=detail.summary.identity.attempt_uid,
                page_size=limit,
            )
        )
    echo_activity(page.items)


@attempt.command("exec", context_settings={"ignore_unknown_options": True})
@click.argument("attempt_ref")
@click.argument("command", nargs=-1, required=True)
@click.option("--timeout", type=click.IntRange(1), default=60, show_default=True)
@click.pass_context
def attempt_exec(ctx: click.Context, attempt_ref: str, command: tuple[str, ...], timeout: int) -> None:
    """Execute COMMAND in one exact active Attempt."""
    with resource_client_for_ctx(ctx) as client:
        result = client.exec_attempt(
            attempt_locator(ctx, attempt_ref),
            command=command,
            timeout=Duration.from_seconds(timeout),
        )
    if result.stdout:
        click.echo(result.stdout, nl=False)
    if result.stderr:
        click.echo(result.stderr, nl=False, err=True)
    if result.error_message:
        raise click.ClickException(result.error_message)
    if result.exit_code:
        raise SystemExit(result.exit_code)


@attempt.command("profile")
@click.argument("attempt_ref")
@click.option("--type", "profile_name", type=click.Choice(ProfileName), default=ProfileName.CPU)
@click.option("--duration", type=click.IntRange(1), default=30, show_default=True)
@click.pass_context
def attempt_profile(ctx: click.Context, attempt_ref: str, profile_name: ProfileName, duration: int) -> None:
    """Capture and write a diagnostic profile for one exact Attempt."""
    profile_by_name = {
        ProfileName.CPU: job_pb2.ProfileType(cpu=job_pb2.CpuProfile(format=job_pb2.CpuProfile.SPEEDSCOPE)),
        ProfileName.MEMORY: job_pb2.ProfileType(memory=job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH)),
        ProfileName.THREAD: job_pb2.ProfileType(threads=job_pb2.ThreadsProfile()),
    }
    with resource_client_for_ctx(ctx) as client:
        result = client.profile_attempt(
            attempt_locator(ctx, attempt_ref),
            profile=profile_by_name[profile_name],
            duration=Duration.from_seconds(duration),
        )
    if result.error_message:
        raise click.ClickException(result.error_message)
    click.echo(result.profile_data, nl=False)


@attempt.command("terminate")
@click.argument("attempt_ref")
@click.option("--idempotency-key")
@click.pass_context
def attempt_terminate(ctx: click.Context, attempt_ref: str, idempotency_key: str | None) -> None:
    """Terminate one exact Attempt and disable automatic retry."""
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_attempt(attempt_locator(ctx, attempt_ref))
        receipt = client.terminate_attempt(
            detail.summary.identity,
            idempotency_key=action_key(idempotency_key),
        )
    echo_action(receipt)
