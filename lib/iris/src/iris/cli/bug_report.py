# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic bug report generation for failed Iris jobs.

Gathers job status, task details, worker health, and recent logs into a
structured Markdown report suitable for GitHub issues or agent consumption.
"""

import logging
import subprocess
from dataclasses import dataclass, field

from iris.cluster.types import JobName
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AttemptReport:
    attempt_id: int
    worker_id: str
    state: str
    exit_code: int
    error: str
    is_worker_failure: bool
    started_at: str
    finished_at: str


@dataclass
class TaskReport:
    task_id: str
    state: str
    worker_id: str
    worker_address: str
    exit_code: int
    error: str
    started_at: str
    finished_at: str
    duration: str
    pending_reason: str
    log_directory: str
    attempts: list[AttemptReport]
    recent_logs: list[str]


@dataclass
class WorkerReport:
    worker_id: str
    address: str
    healthy: bool
    status_message: str
    hostname: str
    gpu_info: str
    tpu_info: str
    memory: str
    zone: str
    process_log_path: str


@dataclass
class BugReport:
    job_id: str
    job_name: str
    state_name: str
    error_summary: str
    error: str
    submitted_at: str
    started_at: str
    finished_at: str
    duration: str
    resources: str
    task_count: int
    completed_count: int
    failure_count: int
    preemption_count: int
    task_state_counts: dict[str, int]
    pending_reason: str
    tasks: list[TaskReport]
    workers: dict[str, WorkerReport] = field(default_factory=dict)
    entrypoint: str = ""
    log_prefix: str | None = None


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def gather_bug_report(
    controller_url: str,
    job_id: JobName,
    *,
    tail: int = 50,
) -> BugReport:
    """Gather all diagnostic data for a job into a BugReport."""
    client = ControllerServiceClientSync(controller_url, timeout_ms=30000)
    try:
        return _gather(client, job_id, tail=tail)
    finally:
        client.close()


def _gather(
    client: ControllerServiceClientSync,
    job_id: JobName,
    *,
    tail: int,
) -> BugReport:
    # 1. Job status + original request
    resp = client.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire()))
    job = resp.job
    request = resp.request

    # 2. List tasks
    tasks_resp = client.list_tasks(cluster_pb2.Controller.ListTasksRequest(job_id=job_id.to_wire()))

    # 3. List workers, filter to those involved in this job
    workers_resp = client.list_workers(cluster_pb2.Controller.ListWorkersRequest())
    involved_worker_ids: set[str] = set()
    for t in tasks_resp.tasks:
        if t.worker_id:
            involved_worker_ids.add(t.worker_id)
        for a in t.attempts:
            if a.worker_id:
                involved_worker_ids.add(a.worker_id)

    # 4. Fetch recent logs per task
    task_logs: dict[str, list[str]] = {}
    for task in tasks_resp.tasks:
        try:
            log_resp = client.get_task_logs(
                cluster_pb2.Controller.GetTaskLogsRequest(
                    id=task.task_id,
                    max_total_lines=tail,
                )
            )
            lines: list[str] = []
            for batch in log_resp.task_logs:
                for entry in batch.logs:
                    lines.append(entry.data)
            task_logs[task.task_id] = lines[-tail:]
        except Exception:
            logger.debug("Failed to fetch logs for task %s", task.task_id, exc_info=True)
            task_logs[task.task_id] = ["(failed to fetch logs)"]

    # 5. Derive log prefix
    log_prefix = _derive_log_prefix(tasks_resp.tasks)

    # 6. Build entrypoint string
    entrypoint_str = ""
    if request and request.entrypoint and request.entrypoint.run_command:
        entrypoint_str = " ".join(request.entrypoint.run_command.argv)

    # 7. Build task reports
    task_reports = [_build_task_report(t, task_logs.get(t.task_id, [])) for t in tasks_resp.tasks]

    # 8. Build worker reports
    worker_reports: dict[str, WorkerReport] = {}
    for w in workers_resp.workers:
        if w.worker_id in involved_worker_ids:
            worker_reports[w.worker_id] = _build_worker_report(w, log_prefix)

    # 9. Assemble — prefer ListTasks count over the JobStatus convenience field
    state_name = _job_state_name(job.state)
    error = job.error or ""
    error_summary = error[:100] if error else state_name
    task_count = job.task_count or len(task_reports)

    return BugReport(
        job_id=job.job_id,
        job_name=job.name,
        state_name=state_name,
        error_summary=error_summary,
        error=error,
        submitted_at=_format_timestamp(job.submitted_at),
        started_at=_format_timestamp(job.started_at),
        finished_at=_format_timestamp(job.finished_at),
        duration=_compute_duration(job.started_at, job.finished_at),
        resources=_format_resources(job.resources if job.HasField("resources") else None),
        task_count=task_count,
        completed_count=job.completed_count,
        failure_count=job.failure_count,
        preemption_count=job.preemption_count,
        task_state_counts=dict(job.task_state_counts),
        pending_reason=job.pending_reason,
        tasks=task_reports,
        workers=worker_reports,
        entrypoint=entrypoint_str,
        log_prefix=log_prefix,
    )


# ---------------------------------------------------------------------------
# Report building helpers
# ---------------------------------------------------------------------------


def _build_task_report(task: cluster_pb2.TaskStatus, logs: list[str]) -> TaskReport:
    attempts = [
        AttemptReport(
            attempt_id=a.attempt_id,
            worker_id=a.worker_id,
            state=_task_state_name(a.state),
            exit_code=a.exit_code,
            error=a.error,
            is_worker_failure=a.is_worker_failure,
            started_at=_format_timestamp(a.started_at),
            finished_at=_format_timestamp(a.finished_at),
        )
        for a in task.attempts
    ]
    return TaskReport(
        task_id=task.task_id,
        state=_task_state_name(task.state),
        worker_id=task.worker_id,
        worker_address=task.worker_address,
        exit_code=task.exit_code,
        error=task.error,
        started_at=_format_timestamp(task.started_at),
        finished_at=_format_timestamp(task.finished_at),
        duration=_compute_duration(task.started_at, task.finished_at),
        pending_reason=task.pending_reason,
        log_directory=task.log_directory,
        attempts=attempts,
        recent_logs=logs,
    )


def _build_worker_report(
    w: cluster_pb2.Controller.WorkerHealthStatus,
    log_prefix: str | None,
) -> WorkerReport:
    meta = w.metadata
    gpu_info = ""
    if meta.gpu_count > 0:
        gpu_info = f"{meta.gpu_count}x {meta.gpu_name}"
        if meta.gpu_memory_mb > 0:
            gpu_info += f" ({meta.gpu_memory_mb // 1024}GB)"
    tpu_info = ""
    if meta.device and meta.device.HasField("tpu"):
        tpu_info = meta.device.tpu.variant
    memory = ""
    if meta.memory_bytes > 0:
        memory = f"{meta.memory_bytes // (1024**3)} GiB"

    process_log_path = ""
    if log_prefix:
        process_log_path = _worker_process_log_path(log_prefix, w.worker_id)

    return WorkerReport(
        worker_id=w.worker_id,
        address=w.address,
        healthy=w.healthy,
        status_message=w.status_message,
        hostname=meta.hostname,
        gpu_info=gpu_info,
        tpu_info=tpu_info,
        memory=memory,
        zone=meta.gce_zone,
        process_log_path=process_log_path,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _job_state_name(state: cluster_pb2.JobState) -> str:
    return cluster_pb2.JobState.Name(state).replace("JOB_STATE_", "").lower()


def _task_state_name(state: cluster_pb2.TaskState) -> str:
    return cluster_pb2.TaskState.Name(state).replace("TASK_STATE_", "").lower()


def _format_timestamp(ts) -> str:
    if not ts or not ts.epoch_ms:
        return "-"
    return Timestamp.from_proto(ts).as_formatted_date()


def _compute_duration(start, end) -> str:
    if not start or not start.epoch_ms or not end or not end.epoch_ms:
        return "-"
    delta_s = (end.epoch_ms - start.epoch_ms) / 1000.0
    if delta_s < 0:
        return "-"
    if delta_s < 60:
        return f"{delta_s:.0f}s"
    minutes = int(delta_s // 60)
    seconds = int(delta_s % 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {seconds}s"


def _format_exit_code(code: int) -> str:
    if code == 0:
        return "0 (success)"
    if code > 128:
        signal_num = code - 128
        signals = {9: "SIGKILL (likely OOM)", 15: "SIGTERM", 6: "SIGABRT"}
        sig_name = signals.get(signal_num, f"signal {signal_num}")
        return f"{code} ({sig_name})"
    return str(code)


def _format_resources(resources: cluster_pb2.ResourceSpecProto | None) -> str:
    if not resources:
        return "-"
    parts: list[str] = []
    if resources.cpu:
        parts.append(f"{resources.cpu} cpu")
    if resources.memory_bytes:
        gib = resources.memory_bytes / (1024**3)
        parts.append(f"{gib:.0f} GiB")
    if resources.HasField("device"):
        device = resources.device
        if device.HasField("tpu"):
            parts.append(device.tpu.variant)
        elif device.HasField("gpu"):
            gpu = device.gpu
            gpu_str = f"{gpu.count}x {gpu.variant}" if gpu.variant else f"{gpu.count} gpu"
            parts.append(gpu_str)
    return ", ".join(parts) if parts else "-"


def _derive_log_prefix(tasks: list[cluster_pb2.TaskStatus]) -> str | None:
    """Extract the log storage prefix from task log directories.

    Given log_directory = gs://bucket/ttl=30d/iris-logs/worker-abc/job/name/task/0/0
    and worker_id = worker-abc:
    Returns gs://bucket/ttl=30d/iris-logs
    """
    for task in tasks:
        if not task.log_directory or not task.worker_id:
            continue
        marker = f"/{task.worker_id}/"
        idx = task.log_directory.find(marker)
        if idx >= 0:
            return task.log_directory[:idx]
    return None


def _worker_process_log_path(log_prefix: str, worker_id: str) -> str:
    return f"{log_prefix}/process/worker/{worker_id}/logs.jsonl"


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def format_bug_report(report: BugReport) -> str:
    """Format a BugReport as Markdown suitable for GitHub issues."""
    lines: list[str] = []

    # Job Summary
    lines.append("# Iris Job Bug Report\n")
    lines.append("## Job Summary\n")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Job ID | `{report.job_id}` |")
    if report.job_name:
        lines.append(f"| Name | `{report.job_name}` |")
    lines.append(f"| State | `{report.state_name}` |")
    if report.error:
        lines.append(f"| Error | {_escape_md(report.error)} |")
    if report.pending_reason:
        lines.append(f"| Pending Reason | {_escape_md(report.pending_reason)} |")
    lines.append(f"| Submitted | {report.submitted_at} |")
    lines.append(f"| Started | {report.started_at} |")
    lines.append(f"| Finished | {report.finished_at} |")
    lines.append(f"| Duration | {report.duration} |")

    task_summary = f"{report.task_count} total"
    if report.completed_count:
        task_summary += f" ({report.completed_count} completed"
        if report.failure_count:
            task_summary += f", {report.failure_count} failed"
        task_summary += ")"
    lines.append(f"| Tasks | {task_summary} |")

    lines.append(f"| Failures | {report.failure_count} |")
    lines.append(f"| Preemptions | {report.preemption_count} |")
    lines.append(f"| Resources | {report.resources} |")
    if report.entrypoint:
        lines.append(f"| Entrypoint | `{report.entrypoint}` |")
    lines.append("")

    # Task Details
    lines.append("## Task Details\n")
    for task in report.tasks:
        exit_info = f" (exit {_format_exit_code(task.exit_code)})" if task.exit_code != 0 else ""
        lines.append(f"### `{task.task_id}` — {task.state}{exit_info}\n")

        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        if task.worker_id:
            addr = f" (`{task.worker_address}`)" if task.worker_address else ""
            lines.append(f"| Worker | `{task.worker_id}`{addr} |")
        lines.append(f"| Started | {task.started_at} |")
        lines.append(f"| Finished | {task.finished_at} |")
        lines.append(f"| Duration | {task.duration} |")
        if task.error:
            lines.append(f"| Error | {_escape_md(task.error)} |")
        if task.pending_reason:
            lines.append(f"| Pending Reason | {_escape_md(task.pending_reason)} |")
        if task.log_directory:
            lines.append(f"| Log directory | `{task.log_directory}` |")
        lines.append("")

        if task.attempts:
            lines.append("**Attempts:**\n")
            lines.append("| # | Worker | State | Exit | Error | Worker Failure |")
            lines.append("|---|--------|-------|------|-------|---------------|")
            for a in task.attempts:
                error_col = _escape_md(a.error) if a.error else "-"
                wf = "yes" if a.is_worker_failure else "no"
                lines.append(
                    f"| {a.attempt_id} | {a.worker_id} | {a.state} | "
                    f"{_format_exit_code(a.exit_code)} | {error_col} | {wf} |"
                )
            lines.append("")

    # Log Paths
    lines.append("## Log Paths\n")
    task_log_rows = [t for t in report.tasks if t.log_directory]
    if task_log_rows:
        lines.append("### Task Logs\n")
        lines.append("| Task | Log Directory |")
        lines.append("|------|--------------|")
        for t in task_log_rows:
            lines.append(f"| `{t.task_id}` | `{t.log_directory}` |")
        lines.append("")

    worker_log_rows = [(wid, w) for wid, w in report.workers.items() if w.process_log_path]
    if worker_log_rows:
        lines.append("### Worker Process Logs\n")
        lines.append("| Worker | Path |")
        lines.append("|--------|------|")
        for wid, w in worker_log_rows:
            lines.append(f"| {wid} | `{w.process_log_path}` |")
        lines.append("")

    lines.append("### Controller Logs\n")
    lines.append("Controller process logs are available via:\n")
    lines.append("```")
    lines.append("iris --config cluster.yaml rpc controller GetProcessLogs")
    lines.append("```\n")

    # Recent Logs (for non-succeeded tasks)
    failed_tasks = [t for t in report.tasks if t.state not in ("succeeded", "pending") and t.recent_logs]
    if failed_tasks:
        lines.append("## Recent Logs\n")
        for t in failed_tasks:
            lines.append(f"### `{t.task_id}` (last {len(t.recent_logs)} lines)\n")
            lines.append("```")
            for log_line in t.recent_logs:
                lines.append(log_line)
            lines.append("```\n")

    # Involved Workers
    if report.workers:
        lines.append("## Involved Workers\n")
        lines.append("| Worker | Address | Healthy | Device | Memory | Zone |")
        lines.append("|--------|---------|---------|--------|--------|------|")
        for wid, w in report.workers.items():
            healthy = "yes" if w.healthy else "no"
            device = w.gpu_info or w.tpu_info or "cpu"
            lines.append(f"| {wid} | {w.address} | {healthy} | {device} | {w.memory} | {w.zone} |")
        lines.append("")

    # Useful Commands
    lines.append("## Useful Commands\n")
    lines.append("```bash")
    lines.append("# Stream full task logs")
    lines.append(f"iris --config cluster.yaml job logs {report.job_id}\n")
    if report.log_prefix:
        for t in task_log_rows:
            lines.append("# Read task log file directly")
            lines.append(f"gsutil cat {t.log_directory}/logs.jsonl | tail -100\n")
            break
        for _wid, w in report.workers.items():
            if w.process_log_path:
                lines.append("# Read worker process logs")
                lines.append(f"gsutil cat {w.process_log_path} | tail -100\n")
                break
    lines.append("# Check autoscaler status")
    lines.append("iris --config cluster.yaml rpc controller GetAutoscalerStatus")
    lines.append("```\n")

    return "\n".join(lines)


def _escape_md(text: str) -> str:
    """Escape pipe characters for Markdown tables and collapse to single line."""
    return text.replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------------
# GitHub issue filing
# ---------------------------------------------------------------------------


def file_github_issue(
    title: str,
    body: str,
    repo: str | None,
    labels: list[str],
) -> str | None:
    """File a GitHub issue using the gh CLI. Returns the issue URL or None."""
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    if repo:
        cmd.extend(["--repo", repo])
    for label in labels:
        label = label.strip()
        if label:
            cmd.extend(["--label", label])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        logger.warning("gh issue create failed: %s", result.stderr)
        return None
    return result.stdout.strip()
