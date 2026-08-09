# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

_EVALDASH_SRC = Path(__file__).resolve().parents[2] / "infra" / "evaldash" / "src"
if str(_EVALDASH_SRC) not in sys.path:
    sys.path.insert(0, str(_EVALDASH_SRC))

discovery = ModuleType("discovery")
discovery.resolve_internal_ip = lambda *_args, **_kwargs: "127.0.0.1"
sys.modules["discovery"] = discovery

from iris.cluster.resources.attempt import AttemptSummary  # noqa: E402
from iris.cluster.resources.identity import (  # noqa: E402
    AttemptIdentity,
    JobIdentity,
    NodeIdentity,
    ResourceKey,
    ResourceKind,
    TaskIdentity,
)
from iris.cluster.resources.job import JobDetail, JobSpec, JobSummary  # noqa: E402
from iris.cluster.resources.source import Page  # noqa: E402
from iris.cluster.resources.task import TaskDetail, TaskSummary  # noqa: E402
from iris.rpc import job_pb2  # noqa: E402
from rigging.timing import Timestamp  # noqa: E402

_CLUSTER_SPEC = importlib.util.spec_from_file_location("evaldash_cluster", _EVALDASH_SRC / "cluster.py")
assert _CLUSTER_SPEC is not None and _CLUSTER_SPEC.loader is not None
cluster = importlib.util.module_from_spec(_CLUSTER_SPEC)
sys.modules[_CLUSTER_SPEC.name] = cluster
_CLUSTER_SPEC.loader.exec_module(cluster)


def test_job_status_reads_all_tasks_through_resource_api(monkeypatch) -> None:
    job_key = ResourceKey("iris", ResourceKind.JOB, "/owner/eval")
    job_identity = JobIdentity(job_key, "job-uid")
    job = JobDetail(
        summary=JobSummary(
            identity=job_identity,
            owner_id="owner",
            parent=None,
            state=job_pb2.JOB_STATE_RUNNING,
            execution_cluster_id="iris",
            backend_id="gpu",
            num_tasks=2,
            submitted_at=Timestamp.from_ms(10),
            started_at=Timestamp.from_ms(20),
            finished_at=None,
            error_message="",
            pending_reason="warming workers",
        ),
        spec=cast(JobSpec, SimpleNamespace(name="evaluation")),
    )
    task_key = ResourceKey("iris", ResourceKind.TASK, "/owner/eval/0")
    task_identity = TaskIdentity(task_key, "task-uid")
    node = NodeIdentity(ResourceKey("iris", ResourceKind.NODE, "worker-a"), "gpu", "worker-uid")
    first_attempt = AttemptSummary(
        identity=AttemptIdentity(task_key, 0, "attempt-0"),
        state=job_pb2.TASK_STATE_WORKER_FAILED,
        execution_cluster_id="iris",
        backend_id="gpu",
        node=node,
        created_at=Timestamp.from_ms(30),
        started_at=Timestamp.from_ms(40),
        finished_at=Timestamp.from_ms(50),
        exit_code=1,
        error_message="worker lost",
        terminal_reason="",
    )
    current_attempt = AttemptSummary(
        identity=AttemptIdentity(task_key, 1, "attempt-1"),
        state=job_pb2.TASK_STATE_RUNNING,
        execution_cluster_id="iris",
        backend_id="gpu",
        node=node,
        created_at=Timestamp.from_ms(60),
        started_at=Timestamp.from_ms(70),
        finished_at=None,
        exit_code=None,
        error_message="",
        terminal_reason="",
    )
    task = TaskDetail(
        summary=TaskSummary(
            identity=task_identity,
            job=job_identity,
            task_index=0,
            state=job_pb2.TASK_STATE_RUNNING,
            execution_cluster_id="iris",
            backend_id="gpu",
            current_attempt=current_attempt.identity,
            current_node=node,
            failure_count=0,
            preemption_count=1,
            submitted_at=Timestamp.from_ms(30),
            started_at=Timestamp.from_ms(70),
            finished_at=None,
            status_message="running",
            error_message="",
        ),
        attempts=(first_attempt, current_attempt),
        source_statuses=(),
        root_cause_highlights=(),
    )
    second_key = ResourceKey("iris", ResourceKind.TASK, "/owner/eval/1")
    second_identity = TaskIdentity(second_key, "task-uid-1")
    second_attempt = replace(
        current_attempt,
        identity=AttemptIdentity(second_key, 0, "attempt-second"),
        node=None,
    )
    second_task = replace(
        task,
        summary=replace(
            task.summary,
            identity=second_identity,
            task_index=1,
            current_attempt=second_attempt.identity,
            current_node=None,
        ),
        attempts=(second_attempt,),
    )
    batch_requests = []

    class FakeResourceClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def list_jobs(self, _query):
            return Page((job.summary,), None, ())

        def describe_job(self, _key):
            return job

        def list_tasks(self, query):
            if query.page_token is None:
                return Page((task.summary, second_task.summary), "next", ())
            return Page((), None, ())

        def describe_tasks(self, keys):
            batch_requests.append(keys)
            details = {task.summary.identity.key: task, second_task.summary.identity.key: second_task}
            return tuple(details[key] for key in keys)

        def close(self) -> None:
            pass

    monkeypatch.setattr(cluster, "ResourceClient", FakeResourceClient)
    gateway = cluster.ClusterGateway()
    monkeypatch.setattr(gateway, "_resolve", lambda *_args: "http://controller")

    assert gateway.job_status("/owner/eval") == {
        "reachable": True,
        "error": None,
        "job": {
            "state": "JOB_STATE_RUNNING",
            "error": "",
            "exit_code": 0,
            "started_at": {"epoch_ms": 20},
            "name": "evaluation",
            "status_message": "warming workers",
        },
        "tasks": [
            {
                "task_id": "/owner/eval/0",
                "state": "TASK_STATE_RUNNING",
                "worker_id": "worker-a",
                "exit_code": 0,
                "error": "",
                "started_at": {"epoch_ms": 70},
                "current_attempt_id": 1,
                "attempts": [
                    {
                        "attempt_id": 0,
                        "state": "TASK_STATE_WORKER_FAILED",
                        "worker_id": "worker-a",
                        "exit_code": 1,
                        "error": "worker lost",
                        "started_at": {"epoch_ms": 40},
                        "finished_at": {"epoch_ms": 50},
                        "is_worker_failure": True,
                        "attempt_uid": "attempt-0",
                    },
                    {
                        "attempt_id": 1,
                        "state": "TASK_STATE_RUNNING",
                        "worker_id": "worker-a",
                        "exit_code": 0,
                        "error": "",
                        "started_at": {"epoch_ms": 70},
                        "is_worker_failure": False,
                        "attempt_uid": "attempt-1",
                    },
                ],
            },
            {
                "task_id": "/owner/eval/1",
                "state": "TASK_STATE_RUNNING",
                "worker_id": "",
                "exit_code": 0,
                "error": "",
                "started_at": {"epoch_ms": 70},
                "current_attempt_id": 0,
                "attempts": [
                    {
                        "attempt_id": 0,
                        "state": "TASK_STATE_RUNNING",
                        "worker_id": "",
                        "exit_code": 0,
                        "error": "",
                        "started_at": {"epoch_ms": 70},
                        "is_worker_failure": False,
                        "attempt_uid": "attempt-second",
                    }
                ],
            },
        ],
    }
    assert batch_requests == [(task_key, second_key)]
