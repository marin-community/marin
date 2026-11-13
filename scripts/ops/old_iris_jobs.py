#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report Iris root jobs that keep running past an age threshold.

The ``old-iris-jobs`` Loom watch (``infra/loom/watches/old-iris-jobs.md``) runs this
script, posts the message it writes to Slack, and keeps its state for the next run.
Children count toward their root, because the owner cancels the root. A job is
posted once while it keeps running, including while a preemption leaves every task
pending; a job that stops running leaves the state.
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


@dataclass(frozen=True)
class OldJob:
    """A root job running past the threshold, with the resources its active tasks hold."""

    # The scanned cluster, or the peer that runs a federated job.
    cluster: str
    root: JobName
    started_at_ms: int
    # Active task count per resource description; empty while every task waits.
    occupied: Mapping[str, int]

    @property
    def key(self) -> str:
        return f"{self.cluster}:{self.root}"


def active_task_count(job: JobStatus) -> int:
    return sum(job.task_state_counts.get(state, 0) for state in ACTIVE_TASK_STATES)


def format_resources(resources: ResourceRequest) -> str:
    """Describe one task's resources by its accelerator, or by CPU and memory without one."""
    device = resources.device
    if device is not None and device.kind is DeviceKind.TPU:
        return f"tpu={device.variant}"
    if device is not None and device.kind is DeviceKind.GPU:
        return f"gpu={device.count}x{device.variant}"
    return f"cpu={resources.cpu_millicores / 1000:g} mem={resources.memory_bytes / 2**30:.3g}GiB"


def find_old_jobs(
    scanned_cluster: str,
    running_jobs: Iterable[JobStatus],
    cutoff_ms: int,
    task_resources: Callable[[JobName], ResourceRequest],
) -> list[OldJob]:
    """Group running jobs by the cluster that runs them and their root; keep the trees started before ``cutoff_ms``.

    ``task_resources`` gives a job's per-task request; only jobs with an active task need it.
    """
    # Offset paging over a changing job set can return a job twice.
    unique = {job.job_id: job for job in running_jobs}
    trees: dict[tuple[str, JobName], list[tuple[int, JobStatus]]] = defaultdict(list)
    for job in unique.values():
        if job.started_at is None:
            continue
        cluster = scanned_cluster if job.execution_cluster_id == LOCAL_CLUSTER else job.execution_cluster_id
        trees[(cluster, job.job_id.root_job)].append((job.started_at.epoch_ms(), job))
    old_jobs = []
    for (cluster, root), members in trees.items():
        started_at_ms = min(started for started, _ in members)
        if started_at_ms >= cutoff_ms:
            continue
        occupied: Counter[str] = Counter()
        for _, job in members:
            if count := active_task_count(job):
                occupied[format_resources(task_resources(job.job_id))] += count
        old_jobs.append(OldJob(cluster, root, started_at_ms, dict(occupied)))
    return old_jobs


def _job_line(job: OldJob, now_ms: int) -> str:
    age_days = (now_ms - job.started_at_ms) / MS_PER_DAY
    resources = ", ".join(f"{count}x {spec}" for spec, count in sorted(job.occupied.items()))
    return f"• {job.cluster} `{job.root}`: owner {job.root.user}, {age_days:.1f} days, {resources}"


def _compose_message(
    new_jobs: Sequence[OldJob], scanned_cluster: str, now_ms: int, min_age_days: float
) -> tuple[str, list[OldJob]]:
    """Return the Slack message for the oldest new jobs that fit, and those jobs.

    The cancel command names the scanned cluster: the hub routes a cancel to the
    peer that runs a federated job, and an owner may lack CoreWeave port-forward access.
    """
    if not new_jobs:
        return "", []
    lines = [
        f"*Iris jobs active for more than {min_age_days:g} days*",
        "Each job is listed once while it keeps running. Cancel a job you no longer need with "
        f"`iris --cluster={scanned_cluster} job cancel <job>`.",
        "",
    ]
    size = sum(len(line.encode()) + 1 for line in lines) + DEFERRED_LINE_BYTES
    posted: list[OldJob] = []
    for job in sorted(new_jobs, key=lambda job: job.started_at_ms):
        size += len(_job_line(job, now_ms).encode()) + 1
        if size > MAX_MESSAGE_BYTES:
            break
        posted.append(job)
    lines += [_job_line(job, now_ms) for job in sorted(posted, key=lambda job: (job.cluster, job.started_at_ms))]
    deferred = len(new_jobs) - len(posted)
    if deferred:
        lines += ["", f"_{deferred} more jobs will be listed in the next report._"]
    return "\n".join(lines), posted


def report(
    old_jobs: Sequence[OldJob], reported: set[str], scanned_cluster: str, now_ms: int, min_age_days: float
) -> tuple[str, set[str]]:
    """Return the Slack message for the jobs not yet reported, and the reported keys for the next run.

    A job with no active task is not posted but keeps its key. The message is
    empty when no job is new. The next keys drop jobs that no longer run, and omit
    new jobs the message had no room for.
    """
    new_jobs = [job for job in old_jobs if job.occupied and job.key not in reported]
    message, posted = _compose_message(new_jobs, scanned_cluster, now_ms, min_age_days)
    current = {job.key for job in old_jobs}
    return message, (reported & current) | {job.key for job in posted}


def read_reported(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    return set(json.loads(state_path.read_text())["reported"])


@click.command()
@click.option(
    "--cluster",
    required=True,
    help="Iris cluster to scan. The federation hub also covers the jobs it runs on peers.",
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
def main(cluster: str, min_age_days: float, state_path: Path, message_path: Path) -> None:
    now_ms = Timestamp.now().epoch_ms()
    cutoff_ms = now_ms - int(min_age_days * MS_PER_DAY)
    with open_iris_client(cluster_name=cluster, workspace=None) as client:
        # ListJobs omits resources; the per-job status carries them.
        old_jobs = find_old_jobs(
            cluster,
            client.list_jobs(state=JobState.RUNNING),
            cutoff_ms,
            lambda job_id: client.job_status(job_id).resources,
        )

    reported = read_reported(state_path)
    message, next_reported = report(old_jobs, reported, cluster, now_ms, min_age_days)
    message_path.write_text(message)
    state_path.write_text(json.dumps({"reported": sorted(next_reported)}, indent=2) + "\n")
    click.echo(f"{len(old_jobs)} old running jobs; {len(next_reported - reported)} in the message")


if __name__ == "__main__":
    main()
