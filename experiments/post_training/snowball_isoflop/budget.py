# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for allocated GPU time within one approved experiment job tree."""

from iris.client.client import IrisClient
from iris.client.workload import DeviceKind, TaskStatus
from iris.cluster.types import JobName
from rigging.timing import Timestamp


def allocated_seconds(task: TaskStatus, now: Timestamp) -> float:
    """Include setup and all attempts, conservatively charging any retry gaps."""
    starts = [task.started_at]
    if task.build_metrics is not None:
        starts.append(task.build_metrics.started_at)
    starts.extend(attempt.started_at for attempt in task.attempts)
    present = [value.epoch_seconds() for value in starts if value is not None]
    if not present:
        return 0.0
    end = task.finished_at or now
    return max(0.0, end.epoch_seconds() - min(present))


def allocation_ledger(client: IrisClient, root: JobName) -> dict:
    """Read only descendants of the current Snowball experiment coordinator."""
    root_name = root.to_wire()
    if root.parent is not None or not root_name.startswith("/benfeuer/snowball-isoflop-"):
        raise ValueError(f"Refusing to monitor a job outside the approved experiment: {root_name}")
    now = Timestamp.now()
    rows = []
    for job in client.list_jobs(prefix=f"{root_name}/"):
        if not job.job_id.to_wire().startswith(f"{root_name}/"):
            raise ValueError("Controller returned a job outside this coordinator tree")
        # ListJobs omits resources; GetJobStatus supplies the submitted request.
        job = client.job(job.job_id).status()
        device = job.resources.device
        if device is None or device.kind != DeviceKind.GPU:
            continue
        if device.variant.upper() != "H100" or device.count != 8:
            raise ValueError(f"Unexpected GPU allocation in {job.job_id}")
        for task in client.list_tasks(job.job_id):
            rows.append(
                {
                    "task": task.task_id.to_wire(),
                    "state": task.state.value,
                    "h100_hours": device.count * allocated_seconds(task, now) / 3600,
                }
            )
    return {
        "root": root_name,
        "timestamp": now.as_formatted_date(),
        "h100_hours": sum(row["h100_hours"] for row in rows),
        "tasks": rows,
    }
