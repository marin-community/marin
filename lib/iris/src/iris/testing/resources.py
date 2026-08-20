# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small valid resource records for adapter tests."""

from rigging.timing import Timestamp

from iris.resources.attempt import AttemptSummary
from iris.resources.execution import CommandEntrypoint, Environment, ResourceSpec, RuntimeEntrypoint
from iris.resources.identity import AttemptIdentity, JobIdentity, NodeIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.resources.job import (
    ContainerProfile,
    ExistingJobPolicy,
    JobDetail,
    JobPreemptionPolicy,
    JobSpec,
    JobSummary,
    PriorityBand,
)
from iris.resources.source import ResourceSourceStatus
from iris.resources.state import JobState, TaskState
from iris.resources.task import TaskDetail, TaskSummary

TEST_TIMESTAMP = Timestamp.from_ms(1)


def make_job_summary(
    job_id: str,
    *,
    cluster_id: str = "test",
    job_uid: str = "job-uid",
    owner_id: str = "alice",
    state: JobState = JobState.RUNNING,
    num_tasks: int = 1,
    submitted_at: Timestamp = TEST_TIMESTAMP,
    started_at: Timestamp | None = None,
    finished_at: Timestamp | None = None,
    error_message: str = "",
    pending_reason: str = "",
    exit_code: int | None = None,
) -> JobSummary:
    identity = JobIdentity(ResourceKey(cluster_id, ResourceKind.JOB, job_id), job_uid)
    return JobSummary(
        identity=identity,
        owner_id=owner_id,
        parent=None,
        state=state,
        execution_cluster_id=cluster_id,
        backend_id="default",
        num_tasks=num_tasks,
        submitted_at=submitted_at,
        started_at=started_at,
        finished_at=finished_at,
        error_message=error_message,
        pending_reason=pending_reason,
        exit_code=exit_code,
    )


def make_job_detail(
    summary: JobSummary,
    *,
    name: str,
    resources: ResourceSpec | None = None,
    ports: tuple[str, ...] = (),
) -> JobDetail:
    return JobDetail(
        summary=summary,
        spec=JobSpec(
            version=1,
            name=name,
            entrypoint=RuntimeEntrypoint((), CommandEntrypoint(()), {}, {}),
            resources=resources if resources is not None else ResourceSpec(),
            environment=Environment({}, ()),
            bundle_id="",
            scheduling_timeout=None,
            ports=ports,
            max_task_failures=0,
            max_retries_failure=0,
            max_retries_preemption=0,
            constraints=(),
            coscheduling=None,
            replicas=summary.num_tasks,
            timeout=None,
            fail_if_exists=False,
            preemption_policy=JobPreemptionPolicy.UNSPECIFIED,
            existing_job_policy=ExistingJobPolicy.UNSPECIFIED,
            priority_band=PriorityBand.INHERIT,
            task_image="",
            submit_argv=(),
            client_revision_date="",
            container_profile=ContainerProfile.UNSPECIFIED,
        ),
    )


def make_attempt_summary(
    task: TaskIdentity,
    attempt_number: int,
    *,
    attempt_uid: str | None = None,
    state: TaskState = TaskState.RUNNING,
    node: NodeIdentity | None = None,
    created_at: Timestamp = TEST_TIMESTAMP,
    started_at: Timestamp | None = None,
    finished_at: Timestamp | None = None,
    exit_code: int | None = None,
    error_message: str = "",
    terminal_reason: str = "",
) -> AttemptSummary:
    return AttemptSummary(
        identity=AttemptIdentity(task.key, attempt_number, attempt_uid or f"attempt-uid-{attempt_number}"),
        state=state,
        execution_cluster_id=task.key.cluster_id,
        backend_id="default",
        node=node,
        created_at=created_at,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        error_message=error_message,
        terminal_reason=terminal_reason,
    )


def make_task_summary(
    job: JobIdentity,
    task_index: int,
    *,
    task_uid: str | None = None,
    state: TaskState = TaskState.RUNNING,
    current_attempt: AttemptIdentity | None = None,
    current_node: NodeIdentity | None = None,
    failure_count: int = 0,
    preemption_count: int = 0,
    submitted_at: Timestamp = TEST_TIMESTAMP,
    started_at: Timestamp | None = None,
    finished_at: Timestamp | None = None,
    status_message: str = "",
    error_message: str = "",
) -> TaskSummary:
    key = ResourceKey(job.key.cluster_id, ResourceKind.TASK, f"{job.key.resource_id}/{task_index}")
    return TaskSummary(
        identity=TaskIdentity(key, task_uid or f"task-uid-{task_index}"),
        job=job,
        task_index=task_index,
        state=state,
        execution_cluster_id=job.key.cluster_id,
        backend_id="default",
        current_attempt=current_attempt,
        current_node=current_node,
        failure_count=failure_count,
        preemption_count=preemption_count,
        submitted_at=submitted_at,
        started_at=started_at,
        finished_at=finished_at,
        status_message=status_message,
        error_message=error_message,
    )


def make_task_detail(
    summary: TaskSummary,
    attempts: tuple[AttemptSummary, ...],
    *,
    source_statuses: tuple[ResourceSourceStatus, ...] = (),
    root_cause_highlights: tuple[str, ...] = (),
) -> TaskDetail:
    return TaskDetail(summary, attempts, source_statuses, root_cause_highlights)
