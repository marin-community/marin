#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report Iris jobs that stay active past an age threshold.

The ``old-iris-jobs`` Loom watch (``infra/loom/watches/old-iris-jobs.md``) runs
this script, posts the message it writes to Slack, and keeps the state it writes
for the next run. The unit is the root job: running children add their active
tasks to their root, because the owner cancels the root.

A job carries the name of the cluster that runs it. The federation hub mirrors
the jobs it hands off, children included, so scanning the hub covers them; a job
submitted directly to a peer appears only when that peer is scanned. A job found
on both the hub and its peer is reported once.

A job is reported once while it stays old and active. The state holds the keys of
the reported jobs that still are; a job that stops being active leaves the state
and is reported again if it becomes old and active later.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
from iris.cli.connect import open_iris_client
from iris.client.workload import DeviceKind, JobStatus, ResourceRequest
from iris.cluster.types import LOCAL_CLUSTER, JobName
from iris.resources.state import JobState, TaskState
from rigging.timing import Timestamp

ACTIVE_TASK_STATES = (TaskState.ASSIGNED, TaskState.BUILDING, TaskState.RUNNING)
MS_PER_DAY = 86_400_000
# Loom's Slack post accepts at most 4,000 bytes; jobs that do not fit go in the next report.
MAX_MESSAGE_BYTES = 3_500
# Room kept for the closing line that counts the deferred jobs.
DEFERRED_LINE_BYTES = 60


def job_key(cluster: str, root: JobName) -> str:
    return f"{cluster}:{root}"


@dataclass(frozen=True)
class ActiveTree:
    """A root job's running jobs on the cluster that runs them, with the earliest start time."""

    cluster: str
    root: JobName
    started_at_ms: int
    jobs: tuple[JobStatus, ...]


@dataclass(frozen=True)
class OldJob:
    """A root job active past the threshold, with the resources its tree holds."""

    cluster: str
    root: JobName
    started_at_ms: int
    # Active task count per per-task resource description.
    occupied: Mapping[str, int]

    @property
    def key(self) -> str:
        return job_key(self.cluster, self.root)


def active_task_count(job: JobStatus) -> int:
    return sum(job.task_state_counts.get(state, 0) for state in ACTIVE_TASK_STATES)


def old_trees(scanned_cluster: str, running_jobs: Iterable[JobStatus], cutoff_ms: int) -> list[ActiveTree]:
    """Group running jobs by running cluster and root; keep trees started before ``cutoff_ms`` with an active task.

    ``scanned_cluster`` names the jobs the scanned controller runs itself; a
    federated job keeps the name of the peer that runs it.
    """
    # Offset paging over a changing job set can return a job twice.
    unique = {job.job_id: job for job in running_jobs}
    by_root: dict[tuple[str, JobName], list[tuple[int, JobStatus]]] = defaultdict(list)
    for job in unique.values():
        if job.started_at is None:
            continue
        cluster = scanned_cluster if job.execution_cluster_id == LOCAL_CLUSTER else job.execution_cluster_id
        by_root[(cluster, job.job_id.root_job)].append((job.started_at.epoch_ms(), job))
    trees = []
    for (cluster, root), members in by_root.items():
        started_at_ms = min(started for started, _ in members)
        jobs = tuple(job for _, job in members)
        if started_at_ms < cutoff_ms and any(active_task_count(job) for job in jobs):
            trees.append(ActiveTree(cluster, root, started_at_ms, jobs))
    return trees


def format_resources(resources: ResourceRequest) -> str:
    """Describe one task's resources by its accelerator, or by CPU and memory without one."""
    device = resources.device
    if device is not None and device.kind is DeviceKind.TPU:
        return f"tpu={device.variant}"
    if device is not None and device.kind is DeviceKind.GPU:
        return f"gpu={device.count}x{device.variant}"
    return f"cpu={resources.cpu_millicores / 1000:g} mem={resources.memory_bytes / 2**30:.3g}GiB"


def occupied_resources(
    jobs: Iterable[JobStatus], task_resources: Callable[[JobName], ResourceRequest]
) -> dict[str, int]:
    """Count active tasks per per-task resource description; ``task_resources`` gives a job's per-task request."""
    occupied: Counter[str] = Counter()
    for job in jobs:
        count = active_task_count(job)
        if count:
            occupied[format_resources(task_resources(job.job_id))] += count
    return dict(occupied)


def _job_line(job: OldJob, now_ms: int) -> str:
    age_days = (now_ms - job.started_at_ms) / MS_PER_DAY
    resources = ", ".join(f"{count}x {spec}" for spec, count in sorted(job.occupied.items()))
    return f"• {job.cluster} `{job.root}`: owner {job.root.user}, {age_days:.1f} days, {resources}"


def _compose_message(new_jobs: Sequence[OldJob], now_ms: int, min_age_days: float) -> tuple[str, list[OldJob]]:
    """Return the Slack message for the oldest new jobs that fit, and those jobs."""
    if not new_jobs:
        return "", []
    header = [
        f"*Iris jobs active for more than {min_age_days:g} days*",
        "Cancel a job you no longer need with `iris --cluster=<cluster> job cancel <job>`. "
        "Each job is listed once while it stays active.",
        "",
    ]
    size = sum(len(line.encode()) + 1 for line in header) + DEFERRED_LINE_BYTES
    posted: list[OldJob] = []
    for job in sorted(new_jobs, key=lambda job: job.started_at_ms):
        size += len(_job_line(job, now_ms).encode()) + 1
        if size > MAX_MESSAGE_BYTES:
            break
        posted.append(job)
    lines = header + [_job_line(job, now_ms) for job in sorted(posted, key=lambda job: (job.cluster, job.started_at_ms))]
    deferred = len(new_jobs) - len(posted)
    if deferred:
        lines += ["", f"_{deferred} more jobs will be listed in the next report._"]
    return "\n".join(lines), posted


def report(old_jobs: Sequence[OldJob], reported: set[str], now_ms: int, min_age_days: float) -> tuple[str, set[str]]:
    """Return the Slack message for the jobs not yet reported, and the reported keys for the next run.

    The message is empty when no job is new. The next keys drop jobs that are no
    longer old and active, and omit new jobs the message had no room for.
    """
    new_jobs = [job for job in old_jobs if job.key not in reported]
    message, posted = _compose_message(new_jobs, now_ms, min_age_days)
    current = {job.key for job in old_jobs}
    return message, (reported & current) | {job.key for job in posted}


def read_reported(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    return set(json.loads(state_path.read_text())["reported"])


@click.command()
@click.option(
    "--cluster",
    "clusters",
    multiple=True,
    required=True,
    help="Iris cluster to scan; repeat for each cluster. The hub covers the jobs it federates.",
)
@click.option(
    "--min-age-days",
    type=click.FloatRange(min=0, min_open=True),
    required=True,
    help="Report root jobs whose running tree started more than this many days ago.",
)
@click.option(
    "--state",
    "state_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help='Reported-job state, {"reported": [...]}; read when it exists, then replaced.',
)
@click.option(
    "--message",
    "message_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Slack message output; empty when no job is new.",
)
def main(clusters: tuple[str, ...], min_age_days: float, state_path: Path, message_path: Path) -> None:
    now_ms = Timestamp.now().epoch_ms()
    cutoff_ms = now_ms - int(min_age_days * MS_PER_DAY)
    old_jobs: dict[str, OldJob] = {}
    for scanned_cluster in clusters:
        with open_iris_client(cluster_name=scanned_cluster, workspace=None) as client:
            for tree in old_trees(scanned_cluster, client.list_jobs(state=JobState.RUNNING), cutoff_ms):
                key = job_key(tree.cluster, tree.root)
                if key in old_jobs:
                    continue
                # ListJobs omits resources; the per-job status carries them.
                occupied = occupied_resources(tree.jobs, lambda job_id: client.job_status(job_id).resources)
                old_jobs[key] = OldJob(tree.cluster, tree.root, tree.started_at_ms, occupied)

    reported = read_reported(state_path)
    message, next_reported = report(list(old_jobs.values()), reported, now_ms, min_age_days)
    message_path.write_text(message)
    state_path.write_text(json.dumps({"reported": sorted(next_reported)}, indent=2) + "\n")
    click.echo(f"{len(old_jobs)} old active jobs; {len(next_reported - reported)} in the message")


if __name__ == "__main__":
    main()
