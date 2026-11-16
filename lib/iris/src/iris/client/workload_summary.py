# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JSON-ready workload summaries for diagnostics and performance reports."""

from typing import Any

from rigging.timing import Timestamp

from iris.client.workload import JobStatus, TaskStatus
from iris.resources.state import TERMINAL_TASK_STATES


def _task_duration_ms(task: TaskStatus) -> int | None:
    if task.started_at is None:
        return None
    end_ms = task.finished_at.epoch_ms() if task.finished_at is not None else Timestamp.now().epoch_ms()
    return max(0, end_ms - task.started_at.epoch_ms())


def job_summary_data(job_status: JobStatus, tasks: list[TaskStatus]) -> dict[str, Any]:
    """Return the stable JSON shape consumed by diagnostics and perf reports."""
    task_summaries = []
    for task in sorted(tasks, key=lambda item: item.task_id.require_task()[1]):
        usage = task.resource_usage
        task_summaries.append(
            {
                "task_id": str(task.task_id),
                "index": str(task.task_id.require_task()[1]),
                "state": task.state.value,
                "exit_code": int(task.exit_code) if task.state in TERMINAL_TASK_STATES else None,
                "duration_ms": _task_duration_ms(task),
                "memory_mb": usage.memory_mb if usage is not None else 0,
                "memory_peak_mb": usage.memory_peak_mb if usage is not None else 0,
                "cpu_millicores": usage.cpu_millicores if usage is not None else 0,
                "disk_mb": usage.disk_mb if usage is not None else 0,
                "worker_id": task.worker_id,
                "status_message": task.status_message,
                "error": task.error_message,
            }
        )

    return {
        "job_id": str(job_status.job_id),
        "name": job_status.name,
        "state": job_status.state.value,
        "exit_code": int(job_status.exit_code),
        "error": job_status.error_message,
        "failure_count": int(job_status.failure_count),
        "preemption_count": int(job_status.preemption_count),
        "task_count": int(job_status.task_count),
        "completed_count": int(job_status.completed_count),
        "task_state_counts": {state.name.lower(): count for state, count in job_status.task_state_counts.items()},
        "tasks": task_summaries,
    }
