# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable resource-lifecycle smoke for controller rollouts."""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from rigging.timing import Duration, ExponentialBackoff

from iris.client import IrisClient, Job, JobStatus
from iris.resources.action import ActionState
from iris.resources.execution import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.resources.identity import JobIdentity
from iris.resources.job import JobQuery, PriorityBand
from iris.resources.log import LogPage
from iris.resources.state import JobState
from iris.resources.task import TaskQuery
from iris.rpc.resource_client import ResourceRpcClient

logger = logging.getLogger(__name__)

SMOKE_CPU = 0.1
SMOKE_MEMORY = "1GB"
SMOKE_DISK = "5GB"
SMOKE_SUCCESS_MARKER = "iris-rollout-smoke-success"
SMOKE_CANCEL_MARKER = "iris-rollout-smoke-cancel-ready"
SMOKE_FOLLOWUP_MARKER = "iris-rollout-smoke-followup"
SMOKE_SUCCESS_COMMAND = ("bash", "-c", f"echo {SMOKE_SUCCESS_MARKER}")
SMOKE_CANCEL_COMMAND = ("bash", "-c", f"echo {SMOKE_CANCEL_MARKER}; while true; do sleep 5; done")
SMOKE_FOLLOWUP_COMMAND = ("bash", "-c", f"echo {SMOKE_FOLLOWUP_MARKER}")

SMOKE_LOG_TIMEOUT = 60
SMOKE_ACTION_TIMEOUT = 300
SMOKE_POLL_INTERVAL = 5


def _log_progress(message: str) -> None:
    logger.info("%s", message)


@dataclass(frozen=True)
class SmokeSuiteResult:
    """Resource identifiers produced by one successful smoke suite."""

    completed_job: str
    cancelled_job: str
    followup_job: str
    cancel_action_id: str

    def summary(self) -> str:
        return (
            f"completed={self.completed_job}, cancelled={self.cancelled_job}, "
            f"followup={self.followup_job}, cancel_action={self.cancel_action_id}"
        )


def _submit_smoke_job(
    client: IrisClient,
    *,
    suite_id: str,
    role: str,
    command: tuple[str, ...],
    progress: Callable[[str], None],
) -> Job:
    job = client.submit(
        entrypoint=Entrypoint.from_command(*command),
        name=f"deploy-smoke-{role}-{suite_id}",
        resources=ResourceSpec(cpu=SMOKE_CPU, memory=SMOKE_MEMORY, disk=SMOKE_DISK),
        environment=EnvironmentSpec(setup_scripts=[]),
        priority_band=PriorityBand.INTERACTIVE,
    )
    progress(f"Submitted {role} job {job.job_id}")
    return job


def _wait_for_log_marker(
    read_logs: Callable[[], LogPage],
    *,
    marker: str,
    subject: str,
    timeout: float,
    progress: Callable[[str], None],
) -> None:
    def marker_is_visible() -> bool:
        return any(marker in entry.data for entry in read_logs().entries)

    ExponentialBackoff(initial=0.25, maximum=2).wait_until_or_raise(
        marker_is_visible,
        timeout=Duration.from_seconds(timeout),
        error_message=f"{subject} logs did not contain {marker!r} within {timeout:.0f}s",
    )
    progress(f"Read {marker!r} from {subject} logs")


def _job_identity(resources: ResourceRpcClient, job: Job) -> JobIdentity:
    matches = resources.list_jobs(JobQuery(resource_id=job.job_id.to_wire(), page_size=2)).items
    if len(matches) != 1:
        raise RuntimeError(f"Smoke Job {job.job_id} resolved to {len(matches)} resources, expected 1")
    return matches[0].identity


def _verify_job_and_task_logs(
    resources: ResourceRpcClient,
    identity: JobIdentity,
    marker: str,
    *,
    timeout: float,
    progress: Callable[[str], None],
) -> None:
    log_timeout = min(timeout, SMOKE_LOG_TIMEOUT)
    _wait_for_log_marker(
        lambda: resources.fetch_job_logs(identity),
        marker=marker,
        subject=f"Job {identity.key.resource_id}",
        timeout=log_timeout,
        progress=progress,
    )
    tasks = resources.list_tasks(TaskQuery(job=identity.key, page_size=2)).items
    if len(tasks) != 1:
        raise RuntimeError(f"Smoke Job {identity.key.resource_id} has {len(tasks)} Tasks, expected 1")
    task = tasks[0]
    _wait_for_log_marker(
        lambda: resources.fetch_task_logs(task.identity),
        marker=marker,
        subject=f"Task {task.identity.key.resource_id}",
        timeout=log_timeout,
        progress=progress,
    )


def _require_job_state(status: JobStatus, expected: JobState, *, subject: str) -> None:
    if status.state is expected:
        return
    raise RuntimeError(f"{subject} ended {status.state.name}, expected {expected.name}: {status.error_message}")


def run_smoke_suite(
    client: IrisClient,
    resources: ResourceRpcClient,
    *,
    timeout: float,
    progress: Callable[[str], None] = _log_progress,
) -> SmokeSuiteResult:
    """Exercise submit, wait, logs, cancellation, action polling, and reuse."""
    suite_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    progress(f"Waiting up to {timeout:.0f}s per Job; an empty cluster may need to scale up")

    completed = _submit_smoke_job(
        client,
        suite_id=suite_id,
        role="complete",
        command=SMOKE_SUCCESS_COMMAND,
        progress=progress,
    )
    completed_status = completed.wait(
        timeout=timeout,
        poll_interval=SMOKE_POLL_INTERVAL,
        raise_on_failure=False,
    )
    _require_job_state(completed_status, JobState.SUCCEEDED, subject=f"Job {completed.job_id}")
    completed_identity = _job_identity(resources, completed)
    _verify_job_and_task_logs(
        resources,
        completed_identity,
        SMOKE_SUCCESS_MARKER,
        timeout=timeout,
        progress=progress,
    )

    cancellable = _submit_smoke_job(
        client,
        suite_id=suite_id,
        role="cancel",
        command=SMOKE_CANCEL_COMMAND,
        progress=progress,
    )
    cancel_key = f"deploy-smoke-cancel-{suite_id}"
    cancellation_finished = False
    try:
        cancellable_identity = _job_identity(resources, cancellable)
        _verify_job_and_task_logs(
            resources,
            cancellable_identity,
            SMOKE_CANCEL_MARKER,
            timeout=timeout,
            progress=progress,
        )
        receipt = resources.cancel_job(cancellable_identity, idempotency_key=cancel_key)
        completed_receipt = resources.wait_for_action(
            receipt.action_id,
            timeout=Duration.from_seconds(min(timeout, SMOKE_ACTION_TIMEOUT)),
        )
        if completed_receipt.state is not ActionState.SUCCEEDED:
            raise RuntimeError(
                f"Cancellation {completed_receipt.action_id} ended {completed_receipt.state.value}: "
                f"{completed_receipt.result_message}"
            )
        cancelled_status = cancellable.wait(
            timeout=timeout,
            poll_interval=SMOKE_POLL_INTERVAL,
            raise_on_failure=False,
        )
        _require_job_state(cancelled_status, JobState.KILLED, subject=f"Job {cancellable.job_id}")
        cancellation_finished = True
    finally:
        if not cancellation_finished:
            cancellable_identity = _job_identity(resources, cancellable)
            cleanup = resources.cancel_job(cancellable_identity, idempotency_key=cancel_key)
            resources.wait_for_action(
                cleanup.action_id,
                timeout=Duration.from_seconds(min(timeout, SMOKE_ACTION_TIMEOUT)),
            )

    followup = _submit_smoke_job(
        client,
        suite_id=suite_id,
        role="followup",
        command=SMOKE_FOLLOWUP_COMMAND,
        progress=progress,
    )
    followup_status = followup.wait(
        timeout=timeout,
        poll_interval=SMOKE_POLL_INTERVAL,
        raise_on_failure=False,
    )
    _require_job_state(followup_status, JobState.SUCCEEDED, subject=f"Job {followup.job_id}")
    followup_identity = _job_identity(resources, followup)
    _verify_job_and_task_logs(
        resources,
        followup_identity,
        SMOKE_FOLLOWUP_MARKER,
        timeout=timeout,
        progress=progress,
    )

    return SmokeSuiteResult(
        completed_job=completed.job_id.to_wire(),
        cancelled_job=cancellable.job_id.to_wire(),
        followup_job=followup.job_id.to_wire(),
        cancel_action_id=completed_receipt.action_id,
    )
