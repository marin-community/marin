# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-attempt operations CLI.

Usage:
    iris --config cluster.yaml attempt describe /user/job/0:3
"""

import click

from iris.cli.task import build_attempt_detail, fetch_task_status, render_attempt_detail_text
from iris.cluster.types import TaskAttempt


@click.group()
def attempt():
    """Task-attempt operations."""
    pass


@attempt.command("describe")
@click.argument("attempt_ref")
@click.pass_context
def attempt_describe(ctx, attempt_ref: str) -> None:
    """Describe one task attempt addressed as /user/job/0:3.

    Shows the attempt's terminal state, exit code, worker, error, and — for the
    current attempt — its backing pod name and root cause.

    Examples:

      iris attempt describe /user/job/0:3
    """
    target = TaskAttempt.from_wire(attempt_ref)
    if target.attempt_id is None:
        raise click.ClickException(
            f"attempt describe needs an attempt id: {attempt_ref!r} has no ':<attempt>' suffix "
            "(e.g. /user/job/0:3). Use `task describe` for the whole task."
        )
    response = fetch_task_status(ctx, attempt_ref)
    try:
        detail = build_attempt_detail(response, target.attempt_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_attempt_detail_text(detail))
