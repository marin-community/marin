# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve safe Iris timing/resource evidence for matched runtime comparisons."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from iris.cli.connect import open_iris_client
from rigging.filesystem.buckets import filesystem_for
from rigging.timing import Timestamp


def milliseconds(value):
    return value.epoch_ms() if value is not None else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    observed = Timestamp.now().epoch_ms()
    groups = []
    with open_iris_client(cluster_name="marin", workspace=None) as client:
        for prefix in args.job_root:
            listed = client.list_jobs(prefix=prefix, limit=1000)
            if not listed:
                raise ValueError(f"No jobs found for {prefix}")
            jobs = []
            gpu_ms = 0
            core_ms = 0
            memory_byte_ms = 0
            for item in listed:
                job = client.job_status(item.job_id)
                tasks = []
                device = job.resources.device
                listed_tasks = client.list_tasks(job.job_id)
                if len(listed_tasks) != job.task_count:
                    raise ValueError(f"Incomplete task inventory for {job.job_id}")
                for listed_task in listed_tasks:
                    task = client.task_status(listed_task.task_id)
                    if task.started_at is not None and not task.attempts:
                        raise ValueError(f"Missing attempt timing evidence for {task.task_id}")
                    attempts = []
                    for attempt in task.attempts:
                        start = milliseconds(attempt.started_at)
                        finish = milliseconds(attempt.finished_at)
                        elapsed = max(0, (finish or observed) - start) if start is not None else 0
                        if device is not None and device.kind == "gpu":
                            gpu_ms += elapsed * device.count
                        core_ms += elapsed * job.resources.cpu_millicores / 1000
                        memory_byte_ms += elapsed * job.resources.memory_bytes
                        attempts.append(
                            {
                                "attempt": attempt.attempt_number,
                                "state": str(attempt.state),
                                "started_ms": start,
                                "finished_ms": finish,
                                "elapsed_ms": elapsed,
                                "node": attempt.node_name,
                                "terminal_reason": attempt.terminal_reason,
                            }
                        )
                    tasks.append({"task": str(task.task_id), "state": str(task.state), "attempts": attempts})
                jobs.append(
                    {
                        "job": str(job.job_id),
                        "state": str(job.state),
                        "resources": asdict(job.resources),
                        "submitted_ms": milliseconds(job.submitted_at),
                        "started_ms": milliseconds(job.started_at),
                        "finished_ms": milliseconds(job.finished_at),
                        "failure_count": job.failure_count,
                        "preemption_count": job.preemption_count,
                        "tasks": tasks,
                        "cluster": job.execution_cluster_id,
                    }
                )
            root = next(job for job in jobs if job["job"] == prefix)
            groups.append(
                {
                    "root": prefix,
                    "jobs": jobs,
                    "gpu_task_execution_hours": gpu_ms / 3_600_000,
                    "cpu_core_task_execution_hours": core_ms / 3_600_000,
                    "memory_gib_task_execution_hours": memory_byte_ms / (3_600_000 * 2**30),
                    "root_wall_seconds": (
                        ((root["finished_ms"] or observed) - root["started_ms"]) / 1000 if root["started_ms"] else None
                    ),
                    "complete": (
                        root["finished_ms"] is not None
                        and all(
                            a["finished_ms"] is not None
                            for j in jobs
                            for t in j["tasks"]
                            for a in t["attempts"]
                            if a["started_ms"] is not None
                        )
                    ),
                }
            )
    receipt = {
        "observed_ms": observed,
        "scope": (
            "Iris task attempt execution, including setup/retries; scheduler queue and "
            "pre-container GPU reservation are not billed-cost measurements."
        ),
        "groups": groups,
    }
    payload = json.dumps(receipt, indent=2, default=str).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    fs, path = filesystem_for(args.destination)
    fs.pipe_file(path, payload)
    print(
        json.dumps(
            [
                {
                    "root": group["root"],
                    "complete": group["complete"],
                    "gpu_task_execution_hours": group["gpu_task_execution_hours"],
                }
                for group in groups
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
