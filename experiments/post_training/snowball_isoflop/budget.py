# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for allocated GPU time within one approved experiment job tree."""

import contextvars
import json
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from iris.client.client import IrisClient, iris_ctx
from iris.client.workload import DeviceKind, TaskStatus
from iris.cluster.types import JobName
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Timestamp

logger = logging.getLogger(__name__)


def allocated_seconds(task: TaskStatus, now: Timestamp) -> float:
    """Charge timestamped worker/build time; job submission and queue waits are excluded."""
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


def budgeted_map[T, R](
    tasks: Mapping[str, T],
    operation: Callable[[T], R],
    manager_root: StoragePath,
    h100_hour_limit: float,
    max_concurrent: int,
) -> dict[str, R]:
    """Run named tasks under one live cap, stopping this tree on any failure."""
    context = iris_ctx()
    if context.client is None or context.job_id is None:
        raise RuntimeError("A budgeted phase must run inside an Iris coordinator")
    allocation_ledger(context.client, context.job_id)
    concurrency = min(max_concurrent, len(tasks))
    results = {}
    control = manager_root / "concurrency.json"
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending = iter(tasks.items())
        futures = {}
        exhausted = False
        try:
            while futures or not exhausted:
                desired = json.loads(control.read_text())["max_concurrent"] if control.exists() else concurrency
                if not isinstance(desired, int) or not 1 <= desired <= concurrency:
                    raise ValueError("Concurrency control is outside this coordinator's approved bound")
                while len(futures) < desired and not exhausted:
                    next_task = next(pending, None)
                    if next_task is None:
                        exhausted = True
                        break
                    name, task = next_task
                    futures[executor.submit(contextvars.copy_context().run, operation, task)] = name
                if not futures:
                    break
                completed, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                ledger = allocation_ledger(context.client, context.job_id)
                (manager_root / "allocation-ledger.json").write_text(json.dumps(ledger, indent=2))
                if ledger["h100_hours"] + concurrency * 8 / 60 >= h100_hour_limit:
                    raise RuntimeError("Approved experiment H100-hour limit reached")
                for future in completed:
                    name = futures.pop(future)
                    results[name] = future.result()
                    (manager_root / f"completed-{name}.json").write_text(json.dumps({"result": results[name]}))
        except Exception:
            logger.exception("Stopping this experiment tree after a task or budget-monitor failure")
            context.client.job(context.job_id).cancel()
            raise
    return results
