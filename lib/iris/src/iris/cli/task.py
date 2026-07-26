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
from finelog.client import LogClient
from rigging.connect import proxy_path
from rigging.timing import Timestamp
from tabulate import tabulate

from iris.cli.connect import require_controller_url, rpc_client_for_ctx
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.stats.tables import TASK_EVENT_NAMESPACE
from iris.cluster.types import TERMINAL_TASK_STATES, TaskAttempt
from iris.rpc import controller_pb2
from iris.rpc.proto_display import format_resources, signal_name, task_state_friendly

logger = logging.getLogger(__name__)


def _format_exit(exit_code: int) -> str:
    """Render an exit code, naming the signal for the shell's 128+signal convention."""
    if exit_code > 128:
        return f"{exit_code} ({signal_name(exit_code - 128)})"
    return str(exit_code)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_task_description(response: controller_pb2.Controller.GetTaskStatusResponse) -> dict:
    """Build a structured single-task description from a GetTaskStatus response.

    Pure over the proto (no RPC), so it is unit-testable without a cluster. Surfaces
    the attempt chain and the current attempt's backend object — the pod name that
    ``job summary`` never printed — alongside the distilled failure root cause.
    """
    task_status = response.task
    attempts = [
        {
            "attempt_id": int(a.attempt_id),
            "attempt_uid": a.attempt_uid,
            "state": task_state_friendly(a.state),
            "exit_code": int(a.exit_code) if a.state in TERMINAL_TASK_STATES else None,
            "worker_id": a.worker_id,
            "is_worker_failure": bool(a.is_worker_failure),
            "error": a.error,
            "pod_name": a.pod_name,
            "node_name": a.node_name,
            "terminal_reason": a.terminal_reason,
        }
        for a in task_status.attempts
    ]
    attempts.sort(key=lambda a: a["attempt_id"])
    return {
        "task_id": task_status.task_id,
        "state": task_state_friendly(task_status.state),
        "exit_code": int(task_status.exit_code) if task_status.state in TERMINAL_TASK_STATES else None,
        "error": task_status.error,
        "backend_id": task_status.backend_id,
        "cluster": task_status.cluster,
        "worker_id": task_status.worker_id,
        "worker_address": task_status.worker_address,
        "container_id": task_status.container_id,
        "current_attempt_id": int(task_status.current_attempt_id),
        "pending_reason": task_status.pending_reason,
        "status_message": task_status.status_message,
        "resources": format_resources(response.job_resources),
        "root_cause_highlights": list(response.root_cause_highlights),
        "attempts": attempts,
    }


def build_attempt_detail(response: controller_pb2.Controller.GetTaskStatusResponse, attempt_id: int) -> dict:
    """Build a single-attempt detail from a GetTaskStatus response.

    Pure over the proto. Raises ``ValueError`` if the task has no such attempt. The
    backend object (pod name, node) and terminal reason are persisted per attempt,
    so a past failed attempt is described as fully as the current one. The
    distilled ``root_cause_highlights`` are task-scoped and shown only for the
    current attempt.
    """
    task_status = response.task
    match = next((a for a in task_status.attempts if int(a.attempt_id) == attempt_id), None)
    if match is None:
        available = sorted(int(a.attempt_id) for a in task_status.attempts)
        raise ValueError(f"task {task_status.task_id} has no attempt {attempt_id}; attempts: {available or '(none)'}")
    is_current = int(task_status.current_attempt_id) == attempt_id
    return {
        "task_id": task_status.task_id,
        "attempt_id": attempt_id,
        "attempt_uid": match.attempt_uid,
        "state": task_state_friendly(match.state),
        "exit_code": int(match.exit_code) if match.state in TERMINAL_TASK_STATES else None,
        "worker_id": match.worker_id,
        "is_worker_failure": bool(match.is_worker_failure),
        "error": match.error,
        "is_current": is_current,
        "pod_name": match.pod_name,
        "node_name": match.node_name,
        "terminal_reason": match.terminal_reason,
        "root_cause_highlights": list(response.root_cause_highlights) if is_current else [],
    }


def render_task_description_text(desc: dict) -> str:
    state_line = f"State: {desc['state']}"
    if desc["exit_code"] is not None:
        state_line += f"  exit={_format_exit(desc['exit_code'])}"
    if desc["backend_id"]:
        state_line += f"  backend={desc['backend_id']}"
    if desc["cluster"] and desc["cluster"] != "local":
        state_line += f"  cluster={desc['cluster']}"
    lines = [f"Task: {desc['task_id']}", state_line]
    if desc["worker_id"]:
        worker = desc["worker_id"]
        if desc["worker_address"]:
            worker += f" ({desc['worker_address']})"
        lines.append(f"Worker: {worker}")
    if desc["container_id"]:
        lines.append(f"Backend object (current attempt {desc['current_attempt_id']}): {desc['container_id']}")
    if desc["resources"] and desc["resources"] != "-":
        lines.append(f"Resources: {desc['resources']}")
    if desc["pending_reason"]:
        lines.append(f"Pending: {desc['pending_reason']}")
    if desc["status_message"]:
        lines.append(f"Backend status: {desc['status_message']}")
    if desc["error"]:
        lines.append(f"Error: {desc['error']}")

    lines.extend(["", "Attempts:"])
    rows = [
        [
            a["attempt_id"],
            (a["attempt_uid"] or "")[:12],
            a["state"] + (" (worker)" if a["is_worker_failure"] else ""),
            "-" if a["exit_code"] is None else _format_exit(a["exit_code"]),
            a["worker_id"] or "-",
            # Prefer the terminal reason (carries the init-container failure) over
            # the task container's own error.
            _truncate(a["terminal_reason"] or a["error"], 60),
        ]
        for a in desc["attempts"]
    ]
    lines.append(tabulate(rows, headers=["ATTEMPT", "UID", "STATE", "EXIT", "WORKER", "REASON"], tablefmt="plain"))

    if desc["root_cause_highlights"]:
        lines.extend(["", "Root cause:"])
        lines.extend(f"  {line}" for line in desc["root_cause_highlights"])
    return "\n".join(lines)


def render_attempt_detail_text(detail: dict) -> str:
    header = f"Attempt: {detail['task_id']}:{detail['attempt_id']}"
    if detail["is_current"]:
        header += "  (current)"
    state_line = f"State: {detail['state']}"
    if detail["is_worker_failure"]:
        state_line += "  (worker failure)"
    if detail["exit_code"] is not None:
        state_line += f"  exit={_format_exit(detail['exit_code'])}"
    lines = [header, f"UID: {detail['attempt_uid']}", state_line]
    if detail["worker_id"]:
        lines.append(f"Worker: {detail['worker_id']}")
    if detail["pod_name"]:
        backend = detail["pod_name"]
        if detail["node_name"]:
            backend += f" on {detail['node_name']}"
        lines.append(f"Backend object: {backend}")
    if detail["terminal_reason"]:
        lines.append(f"Terminal reason: {detail['terminal_reason']}")
    if detail["error"] and detail["error"] != detail["terminal_reason"]:
        lines.append(f"Error: {detail['error']}")
    if detail["root_cause_highlights"]:
        lines.extend(["", "Root cause:"])
        lines.extend(f"  {line}" for line in detail["root_cause_highlights"])
    return "\n".join(lines)


def fetch_task_status(ctx, task_id: str) -> controller_pb2.Controller.GetTaskStatusResponse:
    """Resolve a task/attempt id to its task and fetch its GetTaskStatus response."""
    target = TaskAttempt.from_wire(task_id)
    with rpc_client_for_ctx(ctx) as client:
        return client.get_task_status(controller_pb2.Controller.GetTaskStatusRequest(task_id=target.task_id.to_wire()))


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
    """Build table rows from typed task events."""
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
    response: controller_pb2.Controller.GetTaskStatusResponse,
    target: TaskAttempt,
) -> list[str]:
    """Return the current job incarnation's attempt UIDs selected by ``target``."""
    attempts = response.task.attempts
    if target.attempt_id is None:
        return [attempt.attempt_uid for attempt in attempts if attempt.attempt_uid]
    uid = next(
        (attempt.attempt_uid for attempt in attempts if attempt.attempt_id == target.attempt_id),
        "",
    )
    if not uid:
        raise click.ClickException(f"Attempt {target.attempt_id} not found for task {target.task_id}")
    return [uid]


@click.group()
def task():
    """Task operations."""
    pass


@task.command("describe")
@click.argument("task_id")
@click.pass_context
def task_describe(ctx, task_id: str) -> None:
    """Describe one task: state, backend object, attempt chain, and root cause.

    Reads the controller's GetTaskStatus RPC. The attempt chain shows each attempt's
    terminal state, exit code, worker, and error; the current attempt's backing pod
    name is printed under "Backend object". Accepts a task id (/user/job/0); an
    attempt suffix is ignored here — use `attempt describe` for a single attempt.

    Examples:

      iris task describe /user/job/0
    """
    response = fetch_task_status(ctx, task_id)
    click.echo(render_task_description_text(build_task_description(response)))


@task.command("events")
@click.argument("task_id")
@click.option("--limit", type=click.IntRange(min=1, max=10_000), default=200, show_default=True)
@click.pass_context
def task_events(ctx, task_id: str, limit: int) -> None:
    """Show backend events and controller actions for a task.

    Without an attempt suffix, returns all retained attempts in chronological
    order. Add ``:ATTEMPT`` to select one attempt.

    Examples:

      iris task events /user/job/0

      iris task events /user/job/0:2
    """
    target = TaskAttempt.from_wire(task_id)
    response = fetch_task_status(ctx, task_id)
    attempt_uids = task_event_attempt_uids(response, target)
    if not attempt_uids:
        click.echo(render_task_events_text(target.task_id.to_wire(), []))
        return
    url = require_controller_url(ctx)
    credentials = ctx.obj.get("credentials") if ctx.obj else None
    interceptors = credentials.interceptors() if credentials is not None else ()
    log_server_url = f"{url.rstrip('/')}{proxy_path(LOG_SERVER_ENDPOINT_NAME)}"
    with contextlib.closing(LogClient.connect(log_server_url, interceptors=interceptors)) as log_client:
        events = [
            TaskEventView.from_row(row)
            for row in log_client.query(build_task_events_sql(target, attempt_uids, limit)).to_pylist()
        ]
    click.echo(render_task_events_text(target.task_id.to_wire(), events))


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
