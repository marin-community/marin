# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global Task inventory, diagnostics, logs, and retry."""

import click

from iris.cli.connect import resource_client_for_ctx
from iris.cli.resource_commands import (
    DEFAULT_ACTIVITY_LIMIT,
    DEFAULT_LOG_LINES,
    DEFAULT_RESOURCE_LIST_LIMIT,
    MAX_ACTIVITY_LIMIT,
    MAX_LOG_LINES,
    MAX_RESOURCE_LIST_LIMIT,
    action_key,
    echo_action,
    echo_activity,
    echo_logs,
    echo_source_warnings,
    resource_key,
    task_state_name,
)
from iris.resources.activity import ActivityQuery
from iris.resources.identity import ResourceKind
from iris.resources.log import LogQuery
from iris.resources.task import TaskQuery


@click.group()
def task() -> None:
    """Inspect and act on global Tasks."""


@task.command("list")
@click.option("--job", "job_id", help="Only Tasks belonging to this Job ID.")
@click.option("--prefix", help="Only Tasks whose Job ID starts with this prefix.")
@click.option("--backend", "backend_id", help="Only Tasks placed on this backend.")
@click.option(
    "--limit",
    type=click.IntRange(1, MAX_RESOURCE_LIST_LIMIT),
    default=DEFAULT_RESOURCE_LIST_LIMIT,
    show_default=True,
)
@click.pass_context
def task_list(
    ctx: click.Context,
    job_id: str | None,
    prefix: str | None,
    backend_id: str | None,
    limit: int,
) -> None:
    """List Tasks globally or narrow them by Job and backend."""
    job_key = resource_key(ctx, ResourceKind.JOB, job_id) if job_id is not None else None
    with resource_client_for_ctx(ctx) as client:
        page = client.list_tasks(TaskQuery(job=job_key, job_id_prefix=prefix, backend_id=backend_id, page_size=limit))
    if not page.items:
        click.echo("No tasks found.")
    else:
        click.echo(f"{'TASK':<48} {'STATE':<15} {'BACKEND':<16} ATTEMPT")
        for item in page.items:
            attempt_number = str(item.current_attempt.attempt_number) if item.current_attempt is not None else "-"
            click.echo(
                f"{item.identity.key.resource_id:<48} {task_state_name(item.state):<15} "
                f"{item.backend_id or '-':<16} {attempt_number}"
            )
    echo_source_warnings(page.source_statuses)


@task.command("describe")
@click.argument("task_id")
@click.pass_context
def task_describe(ctx: click.Context, task_id: str) -> None:
    """Describe one Task and its complete Attempt chain."""
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_task(resource_key(ctx, ResourceKind.TASK, task_id))
    summary = detail.summary
    click.echo(f"Task: {summary.identity.key.resource_id}")
    click.echo(f"UID: {summary.identity.task_uid}")
    click.echo(f"State: {task_state_name(summary.state)}")
    click.echo(f"Backend: {summary.backend_id or '-'}")
    if summary.current_attempt is not None:
        click.echo(f"Current attempt: {summary.current_attempt.attempt_number} ({summary.current_attempt.attempt_uid})")
    if summary.current_node is not None:
        click.echo(f"Current node: {summary.current_node.key.resource_id}")
    if summary.status_message:
        click.echo(f"Status: {summary.status_message}")
    if summary.error_message:
        click.echo(f"Error: {summary.error_message}")
    click.echo("Attempts:")
    for attempt_summary in detail.attempts:
        click.echo(
            f"  {attempt_summary.identity.attempt_number:<4} "
            f"{task_state_name(attempt_summary.state):<15} {attempt_summary.identity.attempt_uid}"
        )
    if detail.root_cause_highlights:
        click.echo("Root cause:")
        for highlight in detail.root_cause_highlights:
            click.echo(f"  {highlight}")
    echo_source_warnings(detail.source_statuses)


@task.command("activity")
@click.argument("task_id")
@click.option(
    "--limit",
    type=click.IntRange(min=1, max=MAX_ACTIVITY_LIMIT),
    default=DEFAULT_ACTIVITY_LIMIT,
    show_default=True,
)
@click.pass_context
def task_activity(ctx: click.Context, task_id: str, limit: int) -> None:
    """Show authorized activity for one Task."""
    key = resource_key(ctx, ResourceKind.TASK, task_id)
    with resource_client_for_ctx(ctx) as client:
        page = client.list_activity(ActivityQuery(target=key, page_size=limit))
    echo_activity(page.items)
    echo_source_warnings(page.source_statuses)


@task.command("logs")
@click.argument("task_id")
@click.option("--tail", is_flag=True, help="Continue reading new log entries.")
@click.option("--max-lines", type=click.IntRange(1, MAX_LOG_LINES), default=DEFAULT_LOG_LINES, show_default=True)
@click.pass_context
def task_logs(ctx: click.Context, task_id: str, tail: bool, max_lines: int) -> None:
    """Read logs for one exact Task incarnation."""
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_task(resource_key(ctx, ResourceKind.TASK, task_id))
        query = LogQuery(max_lines=max_lines, tail=tail)
        if tail:
            for entry in client.stream_task_logs(detail.summary.identity, query):
                echo_logs((entry,))
            return
        page = client.fetch_task_logs(detail.summary.identity, query)
    echo_logs(page.entries)
    echo_source_warnings(page.source_statuses)


@task.command("retry")
@click.argument("task_id")
@click.option("--idempotency-key")
@click.pass_context
def task_retry(ctx: click.Context, task_id: str, idempotency_key: str | None) -> None:
    """Preempt the exact current Attempt and make its Task retry-eligible."""
    with resource_client_for_ctx(ctx) as client:
        detail = client.describe_task(resource_key(ctx, ResourceKind.TASK, task_id))
        current = detail.summary.current_attempt
        if current is None:
            raise click.ClickException(f"Task {task_id} has no current Attempt")
        receipt = client.retry_task(
            detail.summary.identity,
            expected_attempt_uid=current.attempt_uid,
            idempotency_key=action_key(idempotency_key),
        )
    echo_action(receipt)
