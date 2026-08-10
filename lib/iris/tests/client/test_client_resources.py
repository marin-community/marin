# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import MagicMock

from iris.client import IrisClient
from iris.resources.execution import Entrypoint, EnvironmentSpec, GpuDevice, ResourceSpec
from iris.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.resources.job import JobQuery, JobSummary
from iris.resources.names import JobName
from iris.resources.source import Page
from iris.resources.state import JobState, TaskState
from iris.resources.task import TaskSummary
from rigging.timing import Timestamp


def _job(job_id: str, uid: str) -> JobSummary:
    return JobSummary(
        identity=JobIdentity(ResourceKey("test", ResourceKind.JOB, job_id), uid),
        owner_id="alice",
        parent=None,
        state=JobState.RUNNING,
        execution_cluster_id="test",
        backend_id="default",
        num_tasks=1,
        submitted_at=Timestamp.from_ms(1),
        started_at=None,
        finished_at=None,
        error_message="",
        pending_reason="",
    )


def test_current_job_uses_an_exact_resource_query() -> None:
    exact = _job("/alice/train", "exact-uid")
    cluster = MagicMock()
    cluster.list_jobs.return_value = Page((exact,), None, ())
    client = IrisClient(cluster)

    job = client.current_job(JobName.from_wire("/alice/train"))

    assert job.identity == exact.identity
    assert cluster.list_jobs.call_args.args == (JobQuery(resource_id="/alice/train", page_size=1),)


def test_job_state_and_wait_use_state_only_reads_until_terminal(monkeypatch) -> None:
    running = _job("/alice/train", "exact-uid")
    succeeded = replace(running, state=JobState.SUCCEEDED)
    cluster = MagicMock()
    cluster.list_jobs.return_value = Page((running,), None, ())
    cluster.job_state.side_effect = (JobState.RUNNING, JobState.SUCCEEDED)
    cluster.describe_job.return_value = MagicMock(summary=succeeded)
    monkeypatch.setattr("iris.client.client.time.sleep", lambda _seconds: None)
    job = IrisClient(cluster).current_job(JobName.from_wire("/alice/train"))

    status = job.wait(timeout=1, poll_interval=0)

    assert status == succeeded
    assert cluster.job_state.call_args_list[0].args == (running.identity,)
    assert cluster.describe_job.call_args_list[0].args == (running.identity.key,)


def test_current_task_resolves_a_task_handle_from_its_wire_id() -> None:
    job = _job("/alice/train", "job-uid")
    task_id = JobName.from_wire("/alice/train/7")
    task_identity = TaskIdentity(ResourceKey("test", ResourceKind.TASK, task_id.to_wire()), "task-uid")
    summary = TaskSummary(
        identity=task_identity,
        job=job.identity,
        task_index=7,
        state=TaskState.RUNNING,
        execution_cluster_id="test",
        backend_id="default",
        current_attempt=AttemptIdentity(task_identity.key, 0, "attempt-uid"),
        current_node=None,
        failure_count=0,
        preemption_count=0,
        submitted_at=Timestamp.from_ms(1),
        started_at=Timestamp.from_ms(2),
        finished_at=None,
        status_message="",
        error_message="",
    )
    cluster = MagicMock()
    cluster.list_jobs.return_value = Page((job,), None, ())
    cluster.describe_job.return_value = MagicMock(summary=job)
    cluster.list_tasks.return_value = Page((summary,), None, ())
    client = IrisClient(cluster)

    task = client.current_task(task_id)

    assert task.identity == task_identity


def test_high_level_submit_applies_accelerator_cpu_floor_without_changing_direct_specs() -> None:
    cluster = MagicMock()
    cluster.submit_job.return_value = JobIdentity(
        ResourceKey("test", ResourceKind.JOB, "/alice/train"),
        "job-uid",
    )
    client = IrisClient(cluster)
    requested = ResourceSpec(cpu=0.5, device=GpuDevice("H100"))

    client.submit(
        Entrypoint.from_command("python", "train.py"),
        "train",
        requested,
        environment=EnvironmentSpec(setup_scripts=()),
        user="alice",
    )

    submitted = cluster.submit_job.call_args.args[0]
    assert submitted.resources.cpu == 4
    assert requested.cpu == 0.5

    client.submit(
        Entrypoint.from_command("python", "train.py"),
        "large-train",
        ResourceSpec(cpu=6, device=GpuDevice("H100")),
        environment=EnvironmentSpec(setup_scripts=()),
        user="alice",
    )
    assert cluster.submit_job.call_args.args[0].resources.cpu == 6

    direct = replace(submitted, resources=requested)
    client.submit_job(direct)
    assert cluster.submit_job.call_args.args[0].resources.cpu == 0.5
