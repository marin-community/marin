# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import replace
from types import ModuleType

from iris.resources.attempt import AttemptSummary
from iris.resources.identity import (
    JobIdentity,
    NodeIdentity,
    ResourceKey,
    ResourceKind,
)
from iris.resources.job import JobDetail
from iris.resources.source import Page
from iris.resources.state import JobState, TaskState
from iris.resources.task import TaskDetail
from iris.testing.resources import (
    make_attempt_summary,
    make_job_detail,
    make_job_summary,
    make_task_detail,
    make_task_summary,
)
from rigging.timing import Timestamp

discovery = ModuleType("infra.evaldash.src.discovery")
discovery.resolve_internal_ip = lambda *_args, **_kwargs: "127.0.0.1"
sys.modules[discovery.__name__] = discovery

from infra.evaldash.src import cluster  # noqa: E402


def _task(job: JobIdentity, index: int, *, node: NodeIdentity | None, failed_history: bool) -> TaskDetail:
    summary = make_task_summary(
        job,
        index,
        state=TaskState.RUNNING,
        current_node=node,
        started_at=Timestamp.from_ms(70),
    )
    attempts: list[AttemptSummary] = []
    if failed_history:
        attempts.append(
            make_attempt_summary(
                summary.identity,
                0,
                attempt_uid="attempt-0",
                state=TaskState.WORKER_FAILED,
                node=node,
                started_at=Timestamp.from_ms(40),
                finished_at=Timestamp.from_ms(50),
                exit_code=1,
                error_message="worker lost",
            )
        )
    current = make_attempt_summary(
        summary.identity,
        len(attempts),
        attempt_uid="attempt-1" if failed_history else "attempt-second",
        node=node,
        started_at=Timestamp.from_ms(70),
    )
    attempts.append(current)
    return make_task_detail(replace(summary, current_attempt=current.identity), tuple(attempts))


def test_job_status_reads_all_tasks_through_resource_api(monkeypatch) -> None:
    job_key = ResourceKey("iris", ResourceKind.JOB, "/owner/eval")
    job_identity = JobIdentity(job_key, "job-uid")
    job = make_job_detail(
        make_job_summary(
            job_key.resource_id,
            cluster_id=job_key.cluster_id,
            job_uid=job_identity.job_uid,
            owner_id="owner",
            state=JobState.RUNNING,
            num_tasks=2,
            started_at=Timestamp.from_ms(20),
            pending_reason="warming workers",
            exit_code=17,
        ),
        name="evaluation",
    )
    node = NodeIdentity(ResourceKey("iris", ResourceKind.NODE, "worker-a"), "gpu", "worker-uid")
    first_task = _task(job_identity, 0, node=node, failed_history=True)
    second_task = _task(job_identity, 1, node=None, failed_history=False)

    class FakeResourceClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def list_jobs(self, _query):
            return Page((job.summary,), None, ())

        def describe_job(self, _key) -> JobDetail:
            return job

        def list_tasks(self, query):
            if query.page_token is None:
                return Page((first_task.summary,), "next", ())
            return Page((second_task.summary,), None, ())

        def describe_tasks(self, keys):
            details = {
                first_task.summary.identity.key: first_task,
                second_task.summary.identity.key: second_task,
            }
            return tuple(details[key] for key in keys)

        def close(self) -> None:
            pass

    monkeypatch.setattr(cluster, "ResourceRpcClient", FakeResourceClient)
    gateway = cluster.ClusterGateway()
    monkeypatch.setattr(gateway, "_resolve", lambda *_args: "http://controller")

    result = gateway.job_status("/owner/eval")

    assert (result["reachable"], result["error"]) == (True, None)
    assert result["job"] == {
        "state": "JOB_STATE_RUNNING",
        "error": "",
        "exit_code": 17,
        "started_at": {"epoch_ms": 20},
        "name": "evaluation",
        "status_message": "warming workers",
    }
    assert [task["task_id"] for task in result["tasks"]] == ["/owner/eval/0", "/owner/eval/1"]
    first, second = result["tasks"]
    assert (first["worker_id"], first["current_attempt_id"]) == ("worker-a", 1)
    assert first["exit_code"] == 0
    assert [
        (attempt["attempt_uid"], attempt["state"], attempt["exit_code"], attempt["error"])
        for attempt in first["attempts"]
    ] == [
        ("attempt-0", "TASK_STATE_WORKER_FAILED", 1, "worker lost"),
        ("attempt-1", "TASK_STATE_RUNNING", 0, ""),
    ]
    assert first["attempts"][0]["started_at"] == {"epoch_ms": 40}
    assert first["attempts"][0]["finished_at"] == {"epoch_ms": 50}
    assert first["attempts"][1]["started_at"] == {"epoch_ms": 70}
    assert "finished_at" not in first["attempts"][1]
    assert (second["worker_id"], second["current_attempt_id"]) == ("", 0)
