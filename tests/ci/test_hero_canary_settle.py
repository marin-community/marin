# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from pathlib import Path

import pytest
from click.testing import CliRunner
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from iris.rpc import job_pb2, time_pb2

from scripts.ci import hero_canary_settle

PREFIX = "/runner/hero-canary-"


class FakeClusterClient:
    def __init__(self, jobs: list[job_pb2.JobStatus]) -> None:
        self.jobs = jobs
        self.terminated: list[str] = []

    def list_jobs(self, *, query, **_kwargs):
        return [job for job in self.jobs if job.job_id.startswith(query.job_id_prefix)]

    def terminate_job(self, job_id: JobName) -> None:
        self.terminated.append(job_id.to_wire())


def _canary(day: int, state: int, child_state: int | None = None) -> list[job_pb2.JobStatus]:
    coordinator = f"{PREFIX}{day}-1-coord"
    jobs = [
        job_pb2.JobStatus(
            job_id=coordinator,
            state=state,
            submitted_at=time_pb2.Timestamp(epoch_ms=day * 86_400_000),
            has_children=child_state is not None,
        )
    ]
    if child_state is not None:
        jobs.append(job_pb2.JobStatus(job_id=f"{coordinator}/grug-train-hero-canary-{day}-1", state=child_state))
    return jobs


def _settle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, jobs: list[job_pb2.JobStatus]):
    cluster = FakeClusterClient(jobs)
    monkeypatch.setattr(hero_canary_settle, "open_iris_client", lambda **_kwargs: nullcontext(IrisClient(cluster)))
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    result = CliRunner().invoke(
        hero_canary_settle.main,
        ["--iris-config", "unused.yaml", "--job-prefix", PREFIX, "--github-output"],
    )
    assert result.exit_code == 0, result.output
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    return cluster.terminated, values


def test_canary_still_waiting_for_a_rack_is_cancelled_and_today_submits(monkeypatch, tmp_path):
    jobs = _canary(1, job_pb2.JOB_STATE_SUCCEEDED, job_pb2.JOB_STATE_SUCCEEDED) + _canary(
        2, job_pb2.JOB_STATE_RUNNING, job_pb2.JOB_STATE_PENDING
    )

    terminated, values = _settle(monkeypatch, tmp_path, jobs)

    assert terminated == [f"{PREFIX}2-1-coord"]
    assert values == {
        "outcome": "skipped",
        "job_id": f"{PREFIX}2-1-coord",
        "run_id": "hero-canary-2-1",
        "submit": "true",
    }


def test_training_canary_is_left_running_and_today_is_skipped(monkeypatch, tmp_path):
    terminated, values = _settle(monkeypatch, tmp_path, _canary(2, job_pb2.JOB_STATE_RUNNING, job_pb2.JOB_STATE_RUNNING))

    assert terminated == []
    assert values["outcome"] == "running"
    assert values["submit"] == "false"


@pytest.mark.parametrize(
    ("state", "outcome"),
    [(job_pb2.JOB_STATE_SUCCEEDED, "succeeded"), (job_pb2.JOB_STATE_FAILED, "failed")],
)
def test_finished_canary_reports_its_outcome(monkeypatch, tmp_path, state, outcome):
    terminated, values = _settle(monkeypatch, tmp_path, _canary(2, state, state))

    assert terminated == []
    assert values["outcome"] == outcome
    assert values["run_id"] == "hero-canary-2-1"
    assert values["submit"] == "true"
