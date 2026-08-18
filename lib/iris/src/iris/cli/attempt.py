# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attempt inspection, logs, and lifecycle actions."""

import click

from iris.cli.connect import iris_client_for_ctx
from iris.cli.logs import echo_log_entries, log_start
from iris.cli.process_status import run_profile
from iris.cli.targets import collect_resource_ids
from iris.cli.task import (
    attempt_status,
    echo_task_action_results,
    fetch_task_description,
    fetch_task_events,
    render_attempt_detail_text,
    render_task_events_text,
    task_event_attempt_uids,
)
from iris.cluster.types import TaskAttempt
from iris.resources.state import TaskState


def _attempt_ref(raw: str) -> TaskAttempt:
    ref = TaskAttempt.from_wire(raw)
    ref.task_id.require_task()
    ref.require_attempt()
    return ref


@click.group()
def attempt() -> None:
    """Inspect and act on numbered Task Attempts."""


@attempt.command("describe")
@click.argument("attempt_ref")
@click.pass_context
def attempt_describe(ctx: click.Context, attempt_ref: str) -> None:
    """Describe one Attempt addressed as TASK_ID:ATTEMPT_NUMBER."""
    ref = _attempt_ref(attempt_ref)
    description = fetch_task_description(ctx, attempt_ref)
    click.echo(render_attempt_detail_text(description, attempt_status(description, ref.require_attempt())))


@attempt.command("events")
@click.argument("attempt_ref")
@click.option("--limit", type=click.IntRange(min=1, max=10_000), default=200, show_default=True)
@click.pass_context
def attempt_events(ctx: click.Context, attempt_ref: str, limit: int) -> None:
    """Show retained backend and controller events for one Attempt."""
    ref = _attempt_ref(attempt_ref)
    description = fetch_task_description(ctx, attempt_ref)
    attempt_uids = task_event_attempt_uids(description.status, ref)
    events = fetch_task_events(ctx, ref, attempt_uids, limit)
    click.echo(render_task_events_text(ref.to_wire(), events))


@attempt.command("logs")
@click.argument("attempt_ref")
@click.option("--since-ms", type=int, default=None, help="Only show logs after this epoch millisecond timestamp.")
@click.option("--since-seconds", type=int, default=None, help="Only show logs from the last N seconds.")
@click.option("--max-lines", type=int, default=0, help="Maximum lines to return; zero uses the server default.")
@click.option("--tail/--no-tail", default=True, help="Return the most recent lines instead of the earliest.")
@click.option(
    "--level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
    default=None,
)
@click.option("--substring", default="", help="Only return lines containing this text.")
@click.pass_context
def attempt_logs(
    ctx: click.Context,
    attempt_ref: str,
    since_ms: int | None,
    since_seconds: int | None,
    max_lines: int,
    tail: bool,
    level: str | None,
    substring: str,
) -> None:
    """Read logs for one numbered Attempt."""
    ref = _attempt_ref(attempt_ref)
    with iris_client_for_ctx(ctx, workspace=None) as client:
        entries = client.attempt(ref).logs(
            start=log_start(since_ms, since_seconds),
            max_lines=max_lines,
            tail=tail,
            min_level=level.upper() if level else "",
            substring=substring,
        )
    echo_log_entries(entries)


@attempt.command("wait")
@click.argument("attempt_ref")
@click.pass_context
def attempt_wait(ctx: click.Context, attempt_ref: str) -> None:
    """Wait for one numbered Attempt to reach a terminal state."""
    ref = _attempt_ref(attempt_ref)
    with iris_client_for_ctx(ctx, workspace=None) as client:
        status = client.attempt(ref).wait(timeout=float("inf"))
    click.echo(status.state.value)
    if status.state is not TaskState.SUCCEEDED:
        raise SystemExit(1)


def _attempt_targets(raw_targets: tuple[str, ...], read_stdin: bool) -> tuple[TaskAttempt, ...]:
    targets = collect_resource_ids(raw_targets, read_stdin)
    if not targets:
        raise click.UsageError("No Attempts given. Pass IDs or use --stdin.")
    return tuple(_attempt_ref(target) for target in targets)


@attempt.command("preempt")
@click.argument("attempt_refs", nargs=-1)
@click.option("--stdin", is_flag=True, help="Read additional Attempt IDs from stdin CSV rows.")
@click.option("--reason", default="", help="Record an operator reason on the Attempts.")
@click.option("--dry-run", is_flag=True, help="Print the Attempts that would be preempted.")
@click.pass_context
def attempt_preempt(
    ctx: click.Context,
    attempt_refs: tuple[str, ...],
    stdin: bool,
    reason: str,
    dry_run: bool,
) -> None:
    """Preempt Attempts that are still current under their Task retry policies."""
    targets = _attempt_targets(attempt_refs, stdin)
    if dry_run:
        for target in targets:
            click.echo(target.to_wire())
        return
    with iris_client_for_ctx(ctx, workspace=None) as client:
        echo_task_action_results(client.preempt_tasks(targets, reason=reason), "preempted")


@attempt.command("fail")
@click.argument("attempt_refs", nargs=-1)
@click.option("--stdin", is_flag=True, help="Read additional Attempt IDs from stdin CSV rows.")
@click.option("--reason", default="", help="Record an operator reason on the Attempts.")
@click.option("--dry-run", is_flag=True, help="Print the Attempts that would be failed.")
@click.pass_context
def attempt_fail(
    ctx: click.Context,
    attempt_refs: tuple[str, ...],
    stdin: bool,
    reason: str,
    dry_run: bool,
) -> None:
    """Fail Attempts without retry if they are still current."""
    targets = _attempt_targets(attempt_refs, stdin)
    if dry_run:
        for target in targets:
            click.echo(target.to_wire())
        return
    with iris_client_for_ctx(ctx, workspace=None) as client:
        echo_task_action_results(client.fail_tasks(targets, reason=reason), "failed")


@attempt.command("profile")
@click.argument("attempt_ref")
@click.argument("profiler", type=click.Choice(["threads", "cpu", "mem"]))
@click.option("--duration", "-d", default=10, help="Profiling duration in seconds.")
@click.option("--output", "-o", default=None, help="Output file path.")
@click.option("--locals", "include_locals", is_flag=True, help="Include local variables in a thread dump.")
@click.option("--native", "include_native", is_flag=True, help="Include native frames in a thread dump.")
@click.pass_context
def attempt_profile(
    ctx: click.Context,
    attempt_ref: str,
    profiler: str,
    duration: int,
    output: str | None,
    include_locals: bool,
    include_native: bool,
) -> None:
    """Capture a profile from one active Attempt."""
    target = _attempt_ref(attempt_ref)
    run_profile(ctx, target.to_wire(), profiler, duration, output, include_locals, include_native)
