#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report Iris root jobs that keep running past an age threshold.

The ``old-iris-jobs`` Loom watch (``infra/loom/watches/old-iris-jobs.md``) runs this
script, posts the messages it writes to Slack, and keeps its state for the next run.
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
# Loom's Slack post accepts at most 4,000 bytes; a longer report continues in the next message.
MAX_MESSAGE_BYTES = 3_500
REPORTED_KEY = "reported"


@dataclass(frozen=True)
class OldJob:
    """A root job running past the threshold, with the resources its active tasks hold."""

    # The scanned cluster, or the peer that runs a federated job.
    cluster: str
    root: JobName
    started_at_ms: int
    # Active task count per task request; empty while every task waits.
    occupied: Mapping[ResourceRequest, int]

    @property
    def key(self) -> str:
        return f"{self.cluster}:{self.root}"

    @property
    def holds_accelerator(self) -> bool:
        return any(resources.device is not None for resources in self.occupied)


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
        occupied: Counter[ResourceRequest] = Counter()
        for _, job in members:
            if count := active_task_count(job):
                occupied[task_resources(job.job_id)] += count
        old_jobs.append(OldJob(cluster, root, started_at_ms, dict(occupied)))
    return old_jobs


def _job_line(job: OldJob, now_ms: int) -> str:
    age_days = (now_ms - job.started_at_ms) / MS_PER_DAY
    # Requests that differ only in fields the description omits share one entry.
    described: Counter[str] = Counter()
    for resources, count in job.occupied.items():
        described[format_resources(resources)] += count
    resources = ", ".join(f"{count}x {spec}" for spec, count in sorted(described.items()))
    return f"• {job.cluster} `{job.root}`: owner {job.root.user}, {age_days:.1f} days, {resources}"


def _compose_messages(new_jobs: Sequence[OldJob], scanned_cluster: str, now_ms: int, min_age_days: float) -> list[str]:
    """Return the Slack messages that list every new job: accelerator holders first, then oldest first."""
    if not new_jobs:
        return []
    # The cancel command names the scanned cluster: the hub routes a cancel to the peer
    # that runs a federated job, and an owner may lack CoreWeave port-forward access.
    header = [
        f"*Iris jobs active for more than {min_age_days:g} days*",
        "Each job is listed once while it keeps running. Cancel a job you no longer need with "
        f"`iris --cluster={scanned_cluster} job cancel <job>`.",
        "",
    ]
    messages = ["\n".join(header)]
    for job in sorted(new_jobs, key=lambda job: (not job.holds_accelerator, job.started_at_ms)):
        line = _job_line(job, now_ms)
        if len(messages[-1].encode()) + len(line.encode()) + 1 > MAX_MESSAGE_BYTES:
            messages.append(line)
        else:
            messages[-1] += "\n" + line
    return messages


def report(
    old_jobs: Sequence[OldJob], reported: set[str], scanned_cluster: str, now_ms: int, min_age_days: float
) -> tuple[list[str], set[str]]:
    """Return the Slack messages for the jobs not yet reported, and the reported keys for the next run.

    A job with no active task is not posted but keeps its key. There are no
    messages when no job is new. The next keys drop jobs that no longer run.
    """
    new_jobs = [job for job in old_jobs if job.occupied and job.key not in reported]
    messages = _compose_messages(new_jobs, scanned_cluster, now_ms, min_age_days)
    current = {job.key for job in old_jobs}
    return messages, (reported & current) | {job.key for job in new_jobs}


def read_reported(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    return set(json.loads(state_path.read_text())[REPORTED_KEY])


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
    "--messages-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Slack messages output, one file per message, to post in name order; "
    "must not exist yet. Stays empty when no job is new.",
)
def main(cluster: str, min_age_days: float, state_path: Path, messages_dir: Path) -> None:
    # A message left by an earlier run must not be posted again.
    messages_dir.mkdir(parents=True)
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
    messages, next_reported = report(old_jobs, reported, cluster, now_ms, min_age_days)
    for index, message in enumerate(messages, start=1):
        (messages_dir / f"{index:02d}.md").write_text(message)
    state_path.write_text(json.dumps({REPORTED_KEY: sorted(next_reported)}, indent=2) + "\n")
    click.echo(f"{len(old_jobs)} old running jobs; {len(next_reported - reported)} new in {len(messages)} messages")


if __name__ == "__main__":
    main()
