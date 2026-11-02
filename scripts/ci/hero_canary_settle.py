# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Settle the previous hero GB200 canary before the daily workflow submits the next one.

The canary is submitted and left in the Iris queue until a rack is free. The next
day's workflow run settles it:

- still waiting for a rack: cancel it, so that day is skipped;
- training: leave it running and skip today's submission;
- finished: report its outcome so the workflow can score or alert on it.

Writes ``outcome``, ``job_id``, ``run_id`` and ``submit`` to ``$GITHUB_OUTPUT``.
"""

import dataclasses
import os
from enum import StrEnum
from pathlib import Path

import click
from iris.cli.connect import open_iris_client
from iris.client.workload import JobStatus
from iris.cluster.types import JobName
from iris.resources.state import JobState, is_job_finished

COORDINATOR_SUFFIX = "-coord"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Outcome(StrEnum):
    NONE = "none"  # no earlier canary
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"  # cancelled earlier, e.g. a skipped day
    RUNNING = "running"  # training now; today's run is skipped
    SKIPPED = "skipped"  # never got a rack; cancelled now


@dataclasses.dataclass(frozen=True)
class Settlement:
    outcome: Outcome
    job_id: str
    run_id: str
    # Coordinators still waiting for a rack, to cancel before submitting.
    to_cancel: tuple[str, ...]

    @property
    def submit(self) -> bool:
        return self.outcome is not Outcome.RUNNING


def _is_coordinator(job: JobStatus, prefix: str) -> bool:
    job_id = str(job.job_id)
    return job_id.startswith(prefix) and job_id.endswith(COORDINATOR_SUFFIX) and job_id.count("/") == 2


def _training(coordinator: JobStatus, jobs: list[JobStatus]) -> bool:
    children = [job for job in jobs if str(job.job_id).startswith(str(coordinator.job_id) + "/")]
    return any(child.state is JobState.RUNNING for child in children)


def _finished_outcome(state: JobState) -> Outcome:
    if state is JobState.SUCCEEDED:
        return Outcome.SUCCEEDED
    if state is JobState.KILLED:
        return Outcome.KILLED
    return Outcome.FAILED


def settle(jobs: list[JobStatus], prefix: str) -> Settlement:
    """Decide what to do with earlier canaries, given every job under ``prefix``."""
    coordinators = sorted(
        (job for job in jobs if _is_coordinator(job, prefix)),
        key=lambda job: job.submitted_at.epoch_ms() if job.submitted_at else 0,
        reverse=True,
    )
    if not coordinators:
        return Settlement(Outcome.NONE, job_id="", run_id="", to_cancel=())

    live = [job for job in coordinators if not is_job_finished(job.state)]
    to_cancel = tuple(str(job.job_id) for job in live if not _training(job, jobs))
    latest = coordinators[0]
    job_id = str(latest.job_id)
    run_id = job_id.rsplit("/", 1)[1].removesuffix(COORDINATOR_SUFFIX)
    if is_job_finished(latest.state):
        outcome = _finished_outcome(latest.state)
    elif job_id in to_cancel:
        outcome = Outcome.SKIPPED
    else:
        outcome = Outcome.RUNNING
    return Settlement(outcome, job_id=job_id, run_id=run_id, to_cancel=to_cancel)


@click.command()
@click.option("--iris-config", required=True, type=click.Path(path_type=Path))
@click.option("--job-prefix", required=True, help="Coordinator job-ID prefix, e.g. /runner/hero-canary-.")
@click.option("--github-output", is_flag=True, help="Write the settlement to $GITHUB_OUTPUT.")
def main(iris_config: Path, job_prefix: str, github_output: bool) -> None:
    with open_iris_client(config_file=iris_config, workspace=_REPO_ROOT) as client:
        settlement = settle(client.list_jobs(prefix=job_prefix), job_prefix)
        for job_id in settlement.to_cancel:
            click.echo(f"Cancelling {job_id}: it is still waiting for a rack.")
            client.cancel_job(JobName.from_wire(job_id))

    click.echo(f"Previous canary {settlement.job_id or '(none)'}: {settlement.outcome}")
    if settlement.outcome is Outcome.SKIPPED:
        click.echo(f"::notice title=Hero canary skipped::{settlement.run_id} got no rack before the next run.")
    if settlement.outcome is Outcome.RUNNING:
        click.echo(f"::notice title=Hero canary still training::{settlement.run_id}; today's run is skipped.")
    if github_output and (path := os.environ.get("GITHUB_OUTPUT")):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                f"outcome={settlement.outcome}\n"
                f"job_id={settlement.job_id}\n"
                f"run_id={settlement.run_id}\n"
                f"submit={'true' if settlement.submit else 'false'}\n"
            )


if __name__ == "__main__":
    main()
