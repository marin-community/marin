# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller.resources.rpc import ResourceServiceImpl
from iris.rpc import job_pb2, resource_pb2, worker_pb2


def _key(key) -> resource_pb2.ResourceKey:
    kinds = {
        "job": resource_pb2.RESOURCE_KIND_JOB,
        "task": resource_pb2.RESOURCE_KIND_TASK,
        "attempt": resource_pb2.RESOURCE_KIND_ATTEMPT,
    }
    return resource_pb2.ResourceKey(
        cluster_id=key.cluster_id,
        kind=kinds[key.kind.value],
        resource_id=key.resource_id,
    )


def _job_target(identity, *, uid: str | None = None) -> resource_pb2.LogTarget:
    return resource_pb2.LogTarget(
        job=resource_pb2.JobIdentity(
            key=_key(identity.key),
            job_uid=identity.job_uid if uid is None else uid,
        )
    )


def _task_target(identity, *, uid: str | None = None) -> resource_pb2.LogTarget:
    return resource_pb2.LogTarget(
        task=resource_pb2.TaskIdentity(
            key=_key(identity.key),
            task_uid=identity.task_uid if uid is None else uid,
        )
    )


def _attempt_target(identity, *, uid: str | None = None) -> resource_pb2.LogTarget:
    return resource_pb2.LogTarget(
        attempt=resource_pb2.AttemptIdentity(
            task=_key(identity.task),
            attempt_number=identity.attempt_number,
            attempt_uid=identity.attempt_uid if uid is None else uid,
        )
    )


def _log_lines(service: ResourceServiceImpl, target: resource_pb2.LogTarget) -> set[str]:
    response = service.fetch_logs(resource_pb2.FetchLogsRequest(target=target), None)
    return {entry.data for entry in response.entries}


def test_logs_are_scoped_to_exact_job_task_and_attempt_identities(journey) -> None:
    selected = journey.submit("logs-selected", tasks=2, preemption_retries=1)
    other = journey.submit("logs-other")
    journey.settle()
    first_attempt = journey.resource_attempt(selected[0]).summary.identity
    journey.preempt(selected[0])
    journey.settle()
    current_attempt = journey.resource_attempt(selected[0]).summary.identity

    journey.push_task_logs(selected[0], ["selected-first-attempt"], attempt_id=0)
    journey.push_task_logs(selected[0], ["selected-current-attempt"], attempt_id=1)
    journey.push_task_logs(selected[1], ["selected-sibling"], attempt_id=0)
    journey.push_task_logs(other[0], ["other-job"], attempt_id=0)

    service = ResourceServiceImpl(journey.controller.resources)
    job = journey.resource_job(selected).summary.identity
    task = journey.resource_task(selected[0]).summary.identity

    assert _log_lines(service, _job_target(job)) == {
        "selected-first-attempt",
        "selected-current-attempt",
        "selected-sibling",
    }
    assert _log_lines(service, _task_target(task)) == {
        "selected-first-attempt",
        "selected-current-attempt",
    }
    assert _log_lines(service, _attempt_target(first_attempt)) == {"selected-first-attempt"}
    assert _log_lines(service, _attempt_target(current_attempt)) == {"selected-current-attempt"}

    stale_targets: tuple[Callable[[], resource_pb2.LogTarget], ...] = (
        lambda: _job_target(job, uid="replacement-job"),
        lambda: _task_target(task, uid="replacement-task"),
        lambda: _attempt_target(current_attempt, uid="replacement-attempt"),
    )
    for stale_target in stale_targets:
        with pytest.raises(ConnectError) as exc_info:
            service.fetch_logs(resource_pb2.FetchLogsRequest(target=stale_target()), None)
        assert exc_info.value.code is Code.FAILED_PRECONDITION


def test_activity_keeps_durable_actions_when_task_events_are_unavailable(journey, monkeypatch) -> None:
    job = journey.submit("activity-partial", preemption_retries=1)
    journey.settle()
    task = journey.resource_task(job[0]).summary
    current = task.current_attempt
    assert current is not None
    receipt = journey.retry_resource_task(
        task.identity,
        expected_attempt_uid=current.attempt_uid,
        idempotency_key="activity-partial",
    )

    def unavailable_query(*_args, **_kwargs):
        raise ConnectionError("finelog task events unavailable")

    monkeypatch.setattr(journey.log_stack.client, "query", unavailable_query)
    service = ResourceServiceImpl(journey.controller.resources)
    response = service.list_activity(
        resource_pb2.ListActivityRequest(
            query=resource_pb2.ActivityQuery(
                target=_key(task.identity.key),
                attempt_uid=current.attempt_uid,
            )
        ),
        None,
    )

    assert [(entry.entry_id, entry.correlation_id) for entry in response.entries] == [
        (f"action:{receipt.action_id}", receipt.action_id)
    ]
    assert {(status.source_id, status.state) for status in response.page.source_statuses} == {
        ("controller:journey", resource_pb2.SOURCE_STATE_AVAILABLE),
        ("finelog:journey", resource_pb2.SOURCE_STATE_UNAVAILABLE),
    }

    with pytest.raises(ConnectError) as exc_info:
        service.list_activity(
            resource_pb2.ListActivityRequest(
                query=resource_pb2.ActivityQuery(
                    target=_key(task.identity.key),
                    attempt_uid="replacement-attempt",
                )
            ),
            None,
        )
    assert exc_info.value.code is Code.FAILED_PRECONDITION


def test_exec_and_profile_refuse_a_superseded_attempt_before_the_runtime_boundary(journey, monkeypatch) -> None:
    job = journey.submit("debug-current", preemption_retries=1)
    journey.settle()
    journey.preempt(job[0])
    journey.settle()
    task_key = journey.resource_task(job[0]).summary.identity.key
    current = journey.resource_attempt(job[0]).summary.identity
    assert current.attempt_number == 1

    def exec_in_container(*_args, **_kwargs) -> worker_pb2.Worker.ExecInContainerResponse:
        return worker_pb2.Worker.ExecInContainerResponse(exit_code=0, stdout="current exec")

    def profile_task(*_args, **_kwargs) -> job_pb2.ProfileTaskResponse:
        return job_pb2.ProfileTaskResponse(profile_data=b"current profile")

    monkeypatch.setattr(journey.backend, "exec_in_container", exec_in_container)
    monkeypatch.setattr(journey.backend, "profile_task", profile_task)
    service = ResourceServiceImpl(journey.controller.resources)

    stale = resource_pb2.AttemptLocator(task=_key(task_key), attempt_number=0)
    with pytest.raises(ConnectError) as exec_error:
        service.exec_attempt(
            resource_pb2.ExecAttemptRequest(attempt=stale, command=["echo", "hello"]),
            None,
        )
    assert exec_error.value.code is Code.FAILED_PRECONDITION

    with pytest.raises(ConnectError) as profile_error:
        service.profile_attempt(
            resource_pb2.ProfileAttemptRequest(
                attempt=stale,
                profile=job_pb2.ProfileType(cpu=job_pb2.CpuProfile()),
            ),
            None,
        )
    assert profile_error.value.code is Code.FAILED_PRECONDITION

    current_locator = resource_pb2.AttemptLocator(task=_key(task_key), attempt_number=current.attempt_number)
    exec_response = service.exec_attempt(
        resource_pb2.ExecAttemptRequest(attempt=current_locator, command=["echo", "hello"]),
        None,
    )
    profile_response = service.profile_attempt(
        resource_pb2.ProfileAttemptRequest(
            attempt=current_locator,
            profile=job_pb2.ProfileType(cpu=job_pb2.CpuProfile()),
        ),
        None,
    )

    assert (exec_response.exit_code, exec_response.stdout) == (0, "current exec")
    assert profile_response.profile_data == b"current profile"
