# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound a long-loop child stage's allocated H100 time and watchdog retries."""

import json
import math
import time
import uuid
from dataclasses import dataclass, replace

from fray.current_client import current_client
from fray.types import GpuConfig, JobRequest
from iris.client.client import iris_ctx
from iris.client.workload import JobStatus, TaskStatus
from iris.cluster.types import JobName
from iris.resources.state import JobState, is_job_finished
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Timestamp


@dataclass(frozen=True)
class StageGuard:
    h100_hour_limit: float
    poll_interval: float
    stall_retries: int

    def __post_init__(self):
        if (
            not math.isfinite(self.h100_hour_limit)
            or self.h100_hour_limit <= 0
            or not math.isfinite(self.poll_interval)
            or self.poll_interval <= 0
            or not 0 <= self.stall_retries <= 2
        ):
            raise ValueError("Stage guards require a positive budget/poll interval and at most two stall retries")


def allocated_seconds(task: TaskStatus, now: Timestamp) -> float:
    """Charge build/worker allocation from timestamps, excluding queue time."""
    starts = [task.started_at, *(attempt.started_at for attempt in task.attempts)]
    if task.build_metrics is not None:
        starts.append(task.build_metrics.started_at)
    present = [stamp.epoch_seconds() for stamp in starts if stamp is not None]
    if not present:
        return 0.0
    return max(0.0, (task.finished_at or now).epoch_seconds() - min(present))


def watchdog_stall(status: JobStatus) -> bool:
    """Return whether the stage reports a progress-watchdog timeout."""
    signals = [status.error_message, status.status_message]
    for task in status.tasks:
        signals.extend((task.error_message, task.status_message))
        for attempt in task.attempts:
            signals.append(attempt.error_message)
            if attempt.exit_code == 124:
                return True
    return any("No progress after process_started" in text or "exit code 124" in text for text in signals)


def run_guarded_stage(request: JobRequest, policy: StageGuard, output: str) -> dict:
    """Retry only watchdog stalls under fresh child identities; cancel only this stage on budget exhaustion."""
    context = iris_ctx()
    if context.client is None or context.job_id is None:
        raise RuntimeError("Guarded stages require an Iris coordinator")
    device = request.resources.device
    if not isinstance(device, GpuConfig) or device.variant.upper() != "H100":
        raise ValueError("The H100-hour guard requires explicit H100 resources")
    root = StoragePath(output)
    root.mkdirs(exist_ok=True)
    (root / "policy.json").write_text(json.dumps(vars(policy), sort_keys=True, indent=2))
    completed_hours = 0.0
    attempts = []
    for retry in range(policy.stall_retries + 1):
        name = request.name if retry == 0 else f"{request.name}-stall-retry-{retry}-{uuid.uuid4().hex[:8]}"
        child = current_client().submit(
            replace(request, name=name, max_retries_failure=0, max_retries_preemption=0), adopt_existing=False
        )
        identity = JobName.from_wire(child.job_id)
        if not identity.to_wire().startswith(context.job_id.to_wire() + "/"):
            raise ValueError("Submitted stage is outside this coordinator tree")
        while True:
            status = context.client.job(identity).status()
            tasks = context.client.list_tasks(identity)
            hours = device.count * sum(allocated_seconds(task, Timestamp.now()) for task in tasks) / 3600
            finished = is_job_finished(status.state)
            receipt = {
                "job_id": child.job_id,
                "attempt": retry,
                "state": status.state.value,
                "h100_hours": completed_hours + hours,
                "h100_hour_limit": policy.h100_hour_limit,
                "terminal_error": status.error_message,
                "terminal_status": status.status_message,
            }
            (root / "allocation-ledger.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
            active = sum(
                task.finished_at is None
                and (
                    task.started_at is not None
                    or (task.build_metrics is not None and task.build_metrics.started_at is not None)
                )
                for task in tasks
            )
            reserve = 0 if finished else device.count * active * policy.poll_interval / 3600
            if completed_hours + hours + reserve >= policy.h100_hour_limit:
                if not finished:
                    context.client.job(identity).cancel()
                receipt["result"] = "budget_exhausted"
                (root / f"attempt-{retry}.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
                raise RuntimeError(
                    f"Stage H100-hour ceiling reached: observed={completed_hours + hours:.6f}, "
                    f"ceiling={policy.h100_hour_limit:.6f}; see {root}"
                )
            if finished:
                receipt["result"] = "succeeded" if status.state == JobState.SUCCEEDED else "failed"
                attempts.append(receipt)
                (root / f"attempt-{retry}.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
                if status.state == JobState.SUCCEEDED:
                    return {"attempts": attempts, "h100_hours": completed_hours + hours}
                completed_hours += hours
                if not watchdog_stall(status) or retry == policy.stall_retries:
                    raise RuntimeError(f"Stage failed: {child.job_id}: {status.error_message}")
                break
            time.sleep(policy.poll_interval)
    raise AssertionError("Stage retry loop exited without a terminal result")
