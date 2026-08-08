# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import replace

import pytest
from click.testing import CliRunner
from iris.cli.attempt import attempt
from iris.cli.job import job
from iris.cli.task import task
from iris.cluster.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.cluster.resources.attempt import AttemptDetail, AttemptRuntimeObject, AttemptSummary
from iris.cluster.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.cluster.resources.job import JobDetail, JobSpec, JobSummary
from iris.cluster.resources.source import Freshness, Page, ResourceSourceStatus, SourceState
from iris.cluster.resources.task import TaskSummary
from iris.cluster.types import ResourceSpec
from iris.rpc import job_pb2
from rigging.timing import Timestamp

_NOW = Timestamp(1_000)


def _job_identity() -> JobIdentity:
    return JobIdentity(ResourceKey("prod", ResourceKind.JOB, "/alice/train"), "job-uid")


def _task_identity(index: int) -> TaskIdentity:
    return TaskIdentity(ResourceKey("prod", ResourceKind.TASK, f"/alice/train/{index}"), f"task-uid-{index}")


def _job_detail() -> JobDetail:
    return JobDetail(
        summary=JobSummary(
            identity=_job_identity(),
            owner_id="alice",
            parent=None,
            state=job_pb2.JOB_STATE_RUNNING,
            execution_cluster_id="prod",
            backend_id="east",
            num_tasks=2,
            submitted_at=_NOW,
            started_at=_NOW,
            finished_at=None,
            error_message="",
        ),
        spec=JobSpec(
            version=1,
            name="train",
            entrypoint=job_pb2.RuntimeEntrypoint(),
            resources=ResourceSpec(cpu=1, memory=1024, disk=2048),
            environment=job_pb2.EnvironmentConfig(),
            bundle_id="bundle",
            scheduling_timeout=None,
            ports=(),
            max_task_failures=0,
            max_retries_failure=0,
            max_retries_preemption=1,
            constraints=(),
            coscheduling=None,
            replicas=2,
            timeout=None,
            fail_if_exists=False,
            preemption_policy=job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED,
            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_UNSPECIFIED,
            priority_band=job_pb2.PRIORITY_BAND_INHERIT,
            task_image="",
            submit_argv=(),
            client_revision_date="",
            container_profile=job_pb2.CONTAINER_PROFILE_UNSPECIFIED,
        ),
    )


def _receipt() -> ActionReceipt:
    return ActionReceipt(
        action_id="action-7",
        kind=ActionKind.CANCEL_JOB,
        target=_job_identity().key,
        expected_target_uid="job-uid",
        expected_attempt_uid=None,
        state=ActionState.ACCEPTED,
        result_code=ActionResult.NONE,
        result_message="",
        created_at=_NOW,
        updated_at=_NOW,
        completed_at=None,
    )


def test_task_list_keeps_rows_and_reports_partial_backend_outage(monkeypatch) -> None:
    current = AttemptIdentity(_task_identity(7).key, 2, "attempt-uid")
    task_summary = TaskSummary(
        identity=_task_identity(7),
        job=_job_identity(),
        task_index=7,
        state=job_pb2.TASK_STATE_RUNNING,
        execution_cluster_id="prod",
        backend_id="east",
        current_attempt=current,
        current_node=None,
        failure_count=0,
        preemption_count=0,
        submitted_at=_NOW,
        started_at=_NOW,
        finished_at=None,
        status_message="",
        error_message="",
    )
    outage = ResourceSourceStatus(
        source_id="backend:west",
        backend_id="west",
        state=SourceState.UNAVAILABLE,
        freshness=Freshness.STALE,
        observed_at=None,
        error_code="backend_unavailable",
        error_message="west did not answer",
    )

    class Client:
        def list_tasks(self, _query):
            return Page((task_summary,), None, (outage,))

    monkeypatch.setattr("iris.cli.task.resource_client_for_ctx", lambda _ctx: nullcontext(Client()))

    result = CliRunner().invoke(task, ["list"], obj={"cluster_name": "prod", "controller_url": "unused"})

    assert result.exit_code == 0, result.output
    assert "/alice/train/7" in result.output
    assert "east" in result.output
    assert "Warning: west: west did not answer" in result.output


def test_job_cancel_uses_described_exact_identity_and_prints_durable_receipt(monkeypatch) -> None:
    accepted_identity: list[JobIdentity] = []

    class Client:
        def describe_job(self, _key):
            return _job_detail()

        def cancel_job(self, identity, *, idempotency_key):
            accepted_identity.append(identity)
            assert idempotency_key == "request-7"
            return _receipt()

    monkeypatch.setattr("iris.cli.job.resource_client_for_ctx", lambda _ctx: nullcontext(Client()))

    result = CliRunner().invoke(
        job,
        ["cancel", "/alice/train", "--idempotency-key", "request-7"],
        obj={"cluster_name": "prod", "controller_url": "unused"},
    )

    assert result.exit_code == 0, result.output
    assert accepted_identity == [_job_identity()]
    assert "Action: action-7" in result.output
    assert "State: accepted" in result.output


def test_attempt_describe_surfaces_exact_runtime_and_terminal_reason(monkeypatch) -> None:
    identity = AttemptIdentity(_task_identity(7).key, 2, "attempt-uid-2")
    detail = AttemptDetail(
        summary=AttemptSummary(
            identity=identity,
            state=job_pb2.TASK_STATE_FAILED,
            execution_cluster_id="prod",
            backend_id="east",
            node=None,
            created_at=_NOW,
            started_at=_NOW,
            finished_at=_NOW,
            exit_code=1,
            error_message="bundle unavailable",
            terminal_reason="init container could not fetch bundle",
        ),
        runtime=AttemptRuntimeObject(
            provider_kind="kubernetes",
            namespace="iris",
            name="iris-train-7-2",
            provider_uid="pod-uid",
            provider_node_id="node-a",
            provider_node_uid="node-uid",
            container_id="container-1",
            observed_at=_NOW,
        ),
        source_statuses=(),
    )

    class Client:
        def describe_attempt(self, _locator):
            return detail

    monkeypatch.setattr("iris.cli.attempt.resource_client_for_ctx", lambda _ctx: nullcontext(Client()))

    result = CliRunner().invoke(
        attempt,
        ["describe", "/alice/train/7:2"],
        obj={"cluster_name": "prod", "controller_url": "unused"},
    )

    assert result.exit_code == 0, result.output
    assert "UID: attempt-uid-2" in result.output
    assert "Runtime: kubernetes:iris/iris-train-7-2" in result.output
    assert "Reason: init container could not fetch bundle" in result.output


@pytest.mark.parametrize(
    ("state", "expected_name", "expected_exit_code"),
    [
        (job_pb2.JOB_STATE_SUCCEEDED, "succeeded", 0),
        (job_pb2.JOB_STATE_FAILED, "failed", 1),
    ],
)
def test_job_wait_reports_terminal_state_and_exit_status(
    monkeypatch,
    state: job_pb2.JobState,
    expected_name: str,
    expected_exit_code: int,
) -> None:
    detail = _job_detail()
    terminal = replace(detail, summary=replace(detail.summary, state=state, finished_at=_NOW))

    class Client:
        def describe_job(self, _key):
            return terminal

    monkeypatch.setattr("iris.cli.job.resource_client_for_ctx", lambda _ctx: nullcontext(Client()))

    result = CliRunner().invoke(
        job,
        ["wait", "/alice/train"],
        obj={"cluster_name": "prod", "controller_url": "unused"},
    )

    assert result.exit_code == expected_exit_code
    assert result.output == f"{expected_name}\n"
