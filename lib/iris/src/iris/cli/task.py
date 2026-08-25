# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task operations CLI.

Usage:
    iris --config cluster.yaml task describe /user/job/0
    iris --config cluster.yaml task exec /user/job/0 -- bash -c "ls /app"
"""

import contextlib
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import click
import humanfriendly
from finelog.client import LogClient
from rigging.connect import proxy_path
from rigging.timing import Timestamp
from tabulate import tabulate

from iris.cli.connect import iris_client_for_ctx, require_controller_url, rpc_client_for_ctx
from iris.cli.logs import echo_workload_logs, workload_log_options
from iris.cli.process_status import run_profile, workload_profile_options
from iris.cli.targets import collect_resource_ids, workload_action_options
from iris.client.workload import (
    AttemptStatus,
    DeviceKind,
    ResourceRequest,
    TaskActionResult,
    TaskDescription,
    TaskStatus,
)
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.stats.tables import TASK_EVENT_NAMESPACE
from iris.cluster.types import JobName, TaskAttempt
from iris.resources.state import TERMINAL_TASK_STATES, TaskState
from iris.rpc import controller_pb2
from iris.rpc.proto_display import signal_name

logger = logging.getLogger(__name__)


def _format_exit(exit_code: int) -> str:
    """Render an exit code, naming the signal for the shell's 128+signal convention."""
    if exit_code > 128:
        return f"{exit_code} ({signal_name(exit_code - 128)})"
    return str(exit_code)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_resources(resources: ResourceRequest) -> str:
    parts = [
        f"cpu={resources.cpu_millicores / 1000:g}",
        f"memory={humanfriendly.format_size(resources.memory_bytes)}",
        f"disk={humanfriendly.format_size(resources.disk_bytes)}",
    ]
    if resources.device is not None:
        device = resources.device
        if device.kind is DeviceKind.TPU:
            parts.append(f"tpu={device.variant or 'any'}")
        elif device.kind is DeviceKind.GPU:
            parts.append(f"gpu={device.count}x{device.variant or 'any'}")
    return ", ".join(parts)


def attempt_status(description: TaskDescription, attempt_number: int) -> AttemptStatus:
    match = next(
        (attempt for attempt in description.status.attempts if attempt.attempt_number == attempt_number),
        None,
    )
    if match is not None:
        return match
    available = sorted(attempt.attempt_number for attempt in description.status.attempts)
    raise ValueError(
        f"task {description.status.task_id} has no attempt {attempt_number}; attempts: {available or '(none)'}"
    )


def render_task_description_text(description: TaskDescription) -> str:
    status = description.status
    state_line = f"State: {status.state.value}"
    if status.state in TERMINAL_TASK_STATES:
        state_line += f"  exit={_format_exit(status.exit_code)}"
    if status.backend_id:
        state_line += f"  backend={status.backend_id}"
    if status.execution_cluster_id and status.execution_cluster_id != "local":
        state_line += f"  cluster={status.execution_cluster_id}"
    lines = [f"Task: {status.task_id}", state_line]
    if status.worker_id:
        worker = status.worker_id
        if status.worker_address:
            worker += f" ({status.worker_address})"
        lines.append(f"Worker: {worker}")
    if status.container_id:
        lines.append(f"Backend object (current attempt {status.current_attempt_number}): {status.container_id}")
    lines.append(f"Resources: {_format_resources(description.resources)}")
    if status.pending_reason:
        lines.append(f"Pending: {status.pending_reason}")
    if status.status_message:
        lines.append(f"Backend status: {status.status_message}")
    if status.error_message:
        lines.append(f"Error: {status.error_message}")

    lines.extend(["", "Attempts:"])
    rows = [
        [
            attempt.attempt_number,
            attempt.attempt_uid[:12],
            attempt.state.value + (" (worker)" if attempt.is_worker_failure else ""),
            "-" if attempt.state not in TERMINAL_TASK_STATES else _format_exit(attempt.exit_code),
            attempt.worker_id or "-",
            _truncate(attempt.terminal_reason or attempt.error_message, 60),
        ]
        for attempt in sorted(status.attempts, key=lambda item: item.attempt_number)
    ]
    lines.append(tabulate(rows, headers=["ATTEMPT", "UID", "STATE", "EXIT", "WORKER", "REASON"], tablefmt="plain"))

    if description.root_cause_highlights:
        lines.extend(["", "Root cause:"])
        lines.extend(f"  {line}" for line in description.root_cause_highlights)
    return "\n".join(lines)


def render_attempt_detail_text(description: TaskDescription, attempt: AttemptStatus) -> str:
    is_current = description.status.current_attempt_number == attempt.attempt_number
    header = f"Attempt: {description.status.task_id}:{attempt.attempt_number}"
    if is_current:
        header += "  (current)"
    state_line = f"State: {attempt.state.value}"
    if attempt.is_worker_failure:
        state_line += "  (worker failure)"
    if attempt.state in TERMINAL_TASK_STATES:
        state_line += f"  exit={_format_exit(attempt.exit_code)}"
    lines = [header, f"UID: {attempt.attempt_uid}", state_line]
    if attempt.worker_id:
        lines.append(f"Worker: {attempt.worker_id}")
    if attempt.pod_name:
        backend = attempt.pod_name
        if attempt.node_name:
            backend += f" on {attempt.node_name}"
        lines.append(f"Backend object: {backend}")
    if attempt.terminal_reason:
        lines.append(f"Terminal reason: {attempt.terminal_reason}")
    if attempt.error_message and attempt.error_message != attempt.terminal_reason:
        lines.append(f"Error: {attempt.error_message}")
    if is_current and description.root_cause_highlights:
        lines.extend(["", "Root cause:"])
        lines.extend(f"  {line}" for line in description.root_cause_highlights)
    return "\n".join(lines)


def fetch_task_description(ctx: click.Context, task_id: str) -> TaskDescription:
    """Resolve a Task or Attempt ID and fetch its public description."""
    target = TaskAttempt.from_wire(task_id)
    with iris_client_for_ctx(ctx, workspace=None) as client:
        return client.task(target.task_id).describe()


def build_task_events_sql(target: TaskAttempt, attempt_uids: list[str], limit: int) -> str:
    """Build a query for the newest retained events in one task incarnation."""
    task_id = target.task_id.to_wire().replace("'", "''")
    escaped_uids = [uid.replace("'", "''") for uid in attempt_uids]
    uid_literals = ", ".join("'" + uid + "'" for uid in escaped_uids)
    predicates = [
        f"task_id = '{task_id}'",
        f"attempt_uid IN ({uid_literals})",
    ]
    where = " AND ".join(predicates)
    return (
        "SELECT attempt_id, ts, type, reason, message, source, count FROM ("
        f'SELECT attempt_id, ts, type, reason, message, source, count FROM "{TASK_EVENT_NAMESPACE}" '
        f"WHERE {where} ORDER BY ts DESC LIMIT {limit}"
        ") AS recent ORDER BY ts ASC"
    )


@dataclass(frozen=True, slots=True)
class TaskEventView:
    """Typed task-event row returned by finelog."""

    attempt_id: int
    ts: Timestamp
    event_type: str
    reason: str
    message: str
    source: str
    count: int

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "TaskEventView":
        attempt_id = row["attempt_id"]
        ts = row["ts"]
        event_type = row["type"]
        reason = row["reason"]
        message = row["message"]
        source = row["source"]
        count = row["count"]
        if not isinstance(attempt_id, int) or not isinstance(count, int):
            raise ValueError("finelog task event has a non-integer attempt_id or count")
        if not isinstance(ts, datetime):
            raise ValueError("finelog task event has a non-datetime timestamp")
        strings = (event_type, reason, message, source)
        if not all(isinstance(value, str) for value in strings):
            raise ValueError("finelog task event has a non-string type, reason, message, or source")
        normalized_ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
        return cls(
            attempt_id=attempt_id,
            ts=Timestamp.from_seconds(normalized_ts.timestamp()),
            event_type=event_type,
            reason=reason,
            message=message,
            source=source,
            count=count,
        )


def build_task_event_display_rows(events: list[TaskEventView]) -> list[list[object]]:
    return [
        [
            event.ts.as_formatted_date(),
            event.attempt_id,
            event.event_type,
            event.source,
            event.reason,
            event.count,
            _truncate(event.message, 100),
        ]
        for event in events
    ]


def render_task_events_text(task_id: str, events: list[TaskEventView]) -> str:
    """Render finelog task-event rows as a chronological operator timeline."""
    lines = [f"Task: {task_id}", ""]
    if not events:
        lines.append("No task events found.")
        return "\n".join(lines)
    lines.append(
        tabulate(
            build_task_event_display_rows(events),
            headers=["TIME (UTC)", "ATTEMPT", "TYPE", "SOURCE", "ACTION", "COUNT", "MESSAGE"],
            tablefmt="plain",
        )
    )
    return "\n".join(lines)


def task_event_attempt_uids(
    status: TaskStatus,
    target: TaskAttempt,
) -> list[str]:
    """Return the current job incarnation's attempt UIDs selected by ``target``."""
    attempts = status.attempts
    if target.attempt_id is None:
        return [attempt.attempt_uid for attempt in attempts if attempt.attempt_uid]
    uid = next(
        (attempt.attempt_uid for attempt in attempts if attempt.attempt_number == target.attempt_id),
        "",
    )
    if not uid:
        raise click.ClickException(f"Attempt {target.attempt_id} not found for task {target.task_id}")
    return [uid]


def fetch_task_events(
    ctx: click.Context,
    target: TaskAttempt,
    attempt_uids: list[str],
    limit: int,
) -> list[TaskEventView]:
    """Fetch retained events for the selected Task incarnation."""
    if not attempt_uids:
        return []
    url = require_controller_url(ctx)
    credentials = ctx.obj.get("credentials") if ctx.obj else None
    interceptors = credentials.interceptors() if credentials is not None else ()
    log_server_url = f"{url.rstrip('/')}{proxy_path(LOG_SERVER_ENDPOINT_NAME)}"
    with contextlib.closing(LogClient.connect(log_server_url, interceptors=interceptors)) as log_client:
        return [
            TaskEventView.from_row(row)
            for row in log_client.query(build_task_events_sql(target, attempt_uids, limit)).to_pylist()
        ]


@click.group()
def task():
    """Task operations."""
    pass


@task.command("describe")
@click.argument("task_id")
@click.pass_context
def task_describe(ctx, task_id: str) -> None:
    """Describe one task: state, backend object, attempt chain, and root cause.

    The attempt chain shows each attempt's terminal state, exit code, worker,
    and error. The current attempt's backing pod name is printed under
    "Backend object". Use `attempt describe` for one numbered Attempt.

    Examples:

      iris task describe /user/job/0
    """
    target = TaskAttempt.from_wire(task_id)
    if target.attempt_id is not None:
        raise click.UsageError("task describe accepts a Task ID; use `iris attempt describe` for one Attempt")
    target.task_id.require_task()
    click.echo(render_task_description_text(fetch_task_description(ctx, task_id)))


@task.command("events")
@click.argument("task_id")
@click.option("--limit", type=click.IntRange(min=1, max=10_000), default=200, show_default=True)
@click.pass_context
def task_events(ctx, task_id: str, limit: int) -> None:
    """Show backend events and controller actions for a task.

    Returns all retained Attempts in chronological order. Use ``attempt events``
    to select one Attempt.

    Examples:

      iris task events /user/job/0

      iris attempt events /user/job/0:2
    """
    target = TaskAttempt.from_wire(task_id)
    if target.attempt_id is not None:
        raise click.UsageError("task events accepts a Task ID; use `iris attempt events` for one Attempt")
    description = fetch_task_description(ctx, task_id)
    attempt_uids = task_event_attempt_uids(description.status, target)
    events = fetch_task_events(ctx, target, attempt_uids, limit)
    click.echo(render_task_events_text(target.task_id.to_wire(), events))


@task.command("list")
@click.argument("job_id")
@click.pass_context
def task_list(ctx: click.Context, job_id: str) -> None:
    """List the Tasks belonging to a Job."""
    with iris_client_for_ctx(ctx, workspace=None) as client:
        statuses = client.list_tasks(JobName.from_wire(job_id))
    rows = [
        [
            status.task_id.require_task()[1],
            status.state.value,
            status.backend_id or "-",
            status.current_attempt_number if status.attempts else "-",
            status.status_message or status.error_message,
        ]
        for status in statuses
    ]
    click.echo(tabulate(rows, headers=["TASK", "STATE", "BACKEND", "ATTEMPT", "STATUS"], tablefmt="plain"))


@task.command("logs")
@click.argument("task_id")
@workload_log_options
@click.pass_context
def task_logs(
    ctx: click.Context,
    task_id: str,
    since_ms: int | None,
    since_seconds: int | None,
    follow: bool,
    max_lines: int,
    tail: bool,
    level: str | None,
    substring: str,
) -> None:
    """Read logs across all Attempts of a Task."""
    target = TaskAttempt.from_wire(task_id)
    if target.attempt_id is not None:
        raise click.UsageError("task logs accepts a Task ID; use `iris attempt logs` for one Attempt")
    with iris_client_for_ctx(ctx, workspace=None) as client:
        echo_workload_logs(
            client.task(target.task_id),
            since_ms=since_ms,
            since_seconds=since_seconds,
            follow=follow,
            max_lines=max_lines,
            tail=tail,
            level=level,
            substring=substring,
        )


@task.command("wait")
@click.argument("task_id")
@click.pass_context
def task_wait(ctx: click.Context, task_id: str) -> None:
    """Wait for a Task to reach a terminal state."""
    target = TaskAttempt.from_wire(task_id)
    if target.attempt_id is not None:
        raise click.UsageError("task wait accepts a Task ID; use `iris attempt wait` for one Attempt")
    with iris_client_for_ctx(ctx, workspace=None) as client:
        status = client.task(target.task_id).wait(timeout=float("inf"))
    click.echo(status.state.value)
    if status.state is not TaskState.SUCCEEDED:
        raise SystemExit(1)


def finish_task_actions(results: tuple[TaskActionResult, ...], verb: str) -> None:
    """Print action results and exit nonzero when any target was rejected."""
    for result in results:
        label = result.task_id or result.target
        if result.accepted:
            click.echo(f"{verb}: {label}")
        else:
            click.echo(f"rejected: {label}: {result.message}", err=True)
    if any(not result.accepted for result in results):
        raise SystemExit(1)


def _task_action_targets(raw_targets: tuple[str, ...], read_stdin: bool) -> tuple[TaskAttempt, ...]:
    targets = collect_resource_ids(raw_targets, read_stdin)
    if not targets:
        raise click.UsageError("No Tasks given. Pass IDs or use --stdin.")
    refs: list[TaskAttempt] = []
    for raw in targets:
        ref = TaskAttempt.from_wire(raw)
        ref.task_id.require_task()
        if ref.attempt_id is not None:
            raise click.UsageError(f"{raw} is an Attempt; use `iris attempt preempt` or `iris attempt fail`")
        refs.append(ref)
    return tuple(refs)


@task.command("preempt")
@click.argument("task_ids", nargs=-1)
@workload_action_options
@click.pass_context
def task_preempt(
    ctx: click.Context,
    task_ids: tuple[str, ...],
    stdin: bool,
    reason: str,
    dry_run: bool,
) -> None:
    """Preempt current Attempts under their Task retry policies."""
    targets = _task_action_targets(task_ids, stdin)
    if dry_run:
        for target in targets:
            click.echo(target.to_wire())
        return
    with iris_client_for_ctx(ctx, workspace=None) as client:
        finish_task_actions(client.preempt_tasks(targets, reason=reason), "preempted")


@task.command("fail")
@click.argument("task_ids", nargs=-1)
@workload_action_options
@click.pass_context
def task_fail(
    ctx: click.Context,
    task_ids: tuple[str, ...],
    stdin: bool,
    reason: str,
    dry_run: bool,
) -> None:
    """Fail current Attempts without retry."""
    targets = _task_action_targets(task_ids, stdin)
    if dry_run:
        for target in targets:
            click.echo(target.to_wire())
        return
    with iris_client_for_ctx(ctx, workspace=None) as client:
        finish_task_actions(client.fail_tasks(targets, reason=reason), "failed")


@task.command("profile")
@click.argument("task_id")
@workload_profile_options
@click.pass_context
def task_profile(
    ctx: click.Context,
    task_id: str,
    profiler: str,
    duration: int,
    output: str | None,
    include_locals: bool,
    include_native: bool,
) -> None:
    """Capture a profile from the current Attempt of a Task."""
    target = TaskAttempt.from_wire(task_id)
    if target.attempt_id is not None:
        raise click.UsageError("task profile accepts a Task ID; use `iris attempt profile` for one Attempt")
    target.task_id.require_task()
    run_profile(ctx, target.to_wire(), profiler, duration, output, include_locals, include_native)


@task.command("exec")
@click.argument("task_id")
@click.argument("command", nargs=-1, required=True)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=int,
    default=60,
    help="Command timeout in seconds (default: 60, -1 for no timeout)",
)
@click.pass_context
def task_exec(ctx, task_id: str, command: tuple[str, ...], timeout_seconds: int):
    """Execute a command in a running task's container.

    Works across platforms: docker exec on Docker, kubectl exec on K8s.

    Examples:

      iris task exec /user/job/0 -- bash -c "ls /app"

      iris task exec /user/job/0 --timeout 300 -- cat /proc/1/status
    """
    with rpc_client_for_ctx(ctx) as client:
        request = controller_pb2.Controller.ExecInContainerRequest(
            task_id=task_id,
            command=list(command),
            timeout_seconds=timeout_seconds,
        )
        response = client.exec_in_container(request)

    if response.error:
        click.echo(f"Error: {response.error}", err=True)
        sys.exit(1)

    if response.stdout:
        click.echo(response.stdout, nl=False)
    if response.stderr:
        click.echo(response.stderr, nl=False, err=True)

    sys.exit(response.exit_code)
