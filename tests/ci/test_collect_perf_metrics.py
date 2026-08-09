# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from iris.client import IrisClient
from iris.resources.attempt import AttemptSummary
from iris.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.resources.job import JobSummary
from iris.resources.source import Page
from iris.resources.state import JobState, TaskState
from iris.resources.task import TaskDetail, TaskSummary
from rigging.timing import Timestamp

from scripts.ci import collect_perf_metrics


def _task(job: JobSummary, index: int) -> tuple[TaskSummary, TaskDetail]:
    key = ResourceKey("test", ResourceKind.TASK, f"/owner/perf/{index}")
    identity = TaskIdentity(key, f"task-{index}")
    attempt = AttemptSummary(
        identity=AttemptIdentity(key, 0, f"attempt-{index}"),
        state=TaskState.SUCCEEDED,
        execution_cluster_id="test",
        backend_id="default",
        node=None,
        created_at=Timestamp.from_ms(10),
        started_at=Timestamp.from_ms(20),
        finished_at=Timestamp.from_ms(30),
        exit_code=index,
        error_message="",
        terminal_reason="",
    )
    summary = TaskSummary(
        identity=identity,
        job=job.identity,
        task_index=index,
        state=TaskState.SUCCEEDED,
        execution_cluster_id="test",
        backend_id="default",
        current_attempt=attempt.identity,
        current_node=None,
        failure_count=0,
        preemption_count=0,
        submitted_at=Timestamp.from_ms(10),
        started_at=Timestamp.from_ms(20),
        finished_at=Timestamp.from_ms(30),
        status_message="",
        error_message="",
    )
    return summary, TaskDetail(summary, (attempt,), (), ())


def test_fetch_job_summary_batches_task_details_once_per_page() -> None:
    job_key = ResourceKey("test", ResourceKind.JOB, "/owner/perf")
    job = JobSummary(
        identity=JobIdentity(job_key, "job-uid"),
        owner_id="owner",
        parent=None,
        state=JobState.SUCCEEDED,
        execution_cluster_id="test",
        backend_id="default",
        num_tasks=3,
        submitted_at=Timestamp.from_ms(1),
        started_at=Timestamp.from_ms(2),
        finished_at=Timestamp.from_ms(40),
        error_message="",
        pending_reason="",
    )
    rows = [_task(job, index) for index in range(3)]

    class CurrentJob:
        def status(self):
            return job

    class FakeClient:
        def __init__(self) -> None:
            self.detail_batches = []

        def current_job(self, _job_id):
            return CurrentJob()

        def list_tasks(self, query):
            if query.page_token is None:
                return Page(tuple(summary for summary, _ in rows[:2]), "next", ())
            return Page((rows[2][0],), None, ())

        def describe_tasks(self, keys):
            self.detail_batches.append(keys)
            details = {summary.identity.key: detail for summary, detail in rows}
            return tuple(details[key] for key in keys)

    client = FakeClient()

    result = collect_perf_metrics.fetch_job_summary(cast(IrisClient, client), job_key.resource_id)

    assert result is not None
    assert [(task["task_id"], task["exit_code"]) for task in result["tasks"]] == [
        ("/owner/perf/0", 0),
        ("/owner/perf/1", 1),
        ("/owner/perf/2", 2),
    ]
    assert [[key.resource_id for key in batch] for batch in client.detail_batches] == [
        ["/owner/perf/0", "/owner/perf/1"],
        ["/owner/perf/2"],
    ]
