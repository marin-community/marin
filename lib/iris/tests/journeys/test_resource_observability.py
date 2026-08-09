# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller.persistence.schema import task_attempts_table, tasks_table
from iris.cluster.types import JobName
from iris.resources.activity import ActivityQuery
from iris.resources.endpoint import ExecResult, ProfileResult
from iris.resources.identity import ResourceKey, ResourceKind
from iris.rpc import resource_pb2
from iris.rpc.resource_service import ResourceServiceImpl
from sqlalchemy import update


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
    return resource_pb2.LogTarget(attempt=_attempt_identity(identity, uid=uid))


def _attempt_identity(identity, *, uid: str | None = None) -> resource_pb2.AttemptIdentity:
    return resource_pb2.AttemptIdentity(
        task=_key(identity.task),
        attempt_number=identity.attempt_number,
        attempt_uid=identity.attempt_uid if uid is None else uid,
    )


def _log_lines(service: ResourceServiceImpl, target: resource_pb2.LogTarget) -> set[str]:
    response = service.fetch_logs(resource_pb2.FetchLogsRequest(target=target), None)
    return {entry.data for entry in response.entries}


def test_logs_are_scoped_to_exact_job_task_and_attempt_identities(journey) -> None:
    selected = journey.submit("logs-selected", tasks=2, preemption_retries=1)
    other = journey.submit("logs-other")
    journey.settle()
    first_attempt = journey.attempt(selected[0]).summary.identity
    journey.preempt(selected[0])
    journey.settle()
    current_attempt = journey.attempt(selected[0]).summary.identity

    journey.push_task_logs(selected[0], ["selected-first-attempt"], attempt_id=0)
    journey.push_task_logs(selected[0], ["selected-current-attempt"], attempt_id=1)
    journey.push_task_logs(selected[1], ["selected-sibling"], attempt_id=0)
    journey.push_task_logs(other[0], ["other-job"], attempt_id=0)

    service = ResourceServiceImpl(journey.controller.controller)
    job = journey.job(selected).summary.identity
    task = journey.task(selected[0]).summary.identity

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


def test_activity_after_restart_keeps_durable_actions_when_task_events_are_unavailable(journey, monkeypatch) -> None:
    job = journey.submit("activity-partial", preemption_retries=1)
    journey.settle()
    task = journey.task(job[0]).summary
    current = task.current_attempt
    assert current is not None
    receipt = journey.retry_task(
        task.identity,
        expected_attempt_uid=current.attempt_uid,
        idempotency_key="activity-partial",
    )
    journey.restart()

    def unavailable_query(*_args, **_kwargs):
        raise ConnectionError("finelog task events unavailable")

    monkeypatch.setattr(journey.log_stack.client, "query", unavailable_query)
    service = ResourceServiceImpl(journey.controller.controller)
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


def test_batch_describe_tasks_returns_ordered_details_with_attempts(journey) -> None:
    job = journey.submit("batch-details", tasks=2)
    journey.settle()
    service = ResourceServiceImpl(journey.controller.controller)

    response = service.batch_describe_tasks(
        resource_pb2.BatchDescribeTasksRequest(
            tasks=[_key(journey.task(job[index]).summary.identity.key) for index in (1, 0)]
        ),
        None,
    )

    assert [detail.summary.identity.key.resource_id for detail in response.tasks] == [job[1].wire_id, job[0].wire_id]
    assert [detail.attempts[0].identity.attempt_uid for detail in response.tasks] == [
        journey.attempt(job[1]).summary.identity.attempt_uid,
        journey.attempt(job[0]).summary.identity.attempt_uid,
    ]


def test_batch_describe_many_failed_tasks_does_not_fan_out_to_finelog(journey, monkeypatch) -> None:
    job = journey.submit("batch-failed", tasks=20)
    journey.settle()
    for index in range(job.tasks):
        journey.fail(job[index])
    journey.settle()

    def unexpected_finelog_call(*_args, **_kwargs):
        raise AssertionError("batch Task details must not issue per-Task finelog calls")

    monkeypatch.setattr(journey.log_stack.client, "fetch_logs", unexpected_finelog_call)
    response = ResourceServiceImpl(journey.controller.controller).batch_describe_tasks(
        resource_pb2.BatchDescribeTasksRequest(
            tasks=[
                _key(ResourceKey(job[index].authority_cluster_id, ResourceKind.TASK, job[index].wire_id))
                for index in range(job.tasks)
            ]
        ),
        None,
    )

    assert len(response.tasks) == job.tasks
    assert all(not detail.root_cause_highlights for detail in response.tasks)


def test_activity_page_token_continues_past_each_sources_first_batch(journey) -> None:
    job = journey.submit("activity-pages", preemption_retries=3)
    journey.settle()
    task_identity = journey.task(job[0]).summary.identity
    for index in range(3):
        current = journey.task(job[0]).summary.current_attempt
        assert current is not None
        journey.retry_task(
            task_identity,
            expected_attempt_uid=current.attempt_uid,
            idempotency_key=f"activity-page-{index}",
        )
        journey.settle()

    journey.log_stack.task_event_table.flush()

    all_entries = journey.controller.controller.list_activity(
        ActivityQuery(target=task_identity.key, page_size=100)
    ).items
    paged_entries = []
    page_token = None
    while True:
        page = journey.controller.controller.list_activity(
            ActivityQuery(target=task_identity.key, page_size=1, page_token=page_token)
        )
        paged_entries.extend(page.items)
        if page.next_page_token is None:
            break
        page_token = page.next_page_token

    assert [entry.entry_id for entry in paged_entries] == [entry.entry_id for entry in all_entries]


def test_attempt_runtime_and_history_use_the_attempts_retained_coordinates(journey) -> None:
    job = journey.submit("attempt-runtime", preemption_retries=1)
    journey.settle()
    task_id = JobName.from_wire(job[0].wire_id)
    backend_id = journey.backend.backend_id
    with journey.database.transaction() as tx:
        tx.execute(
            update(task_attempts_table)
            .where(task_attempts_table.c.task_id == task_id, task_attempts_table.c.attempt_id == 0)
            .values(backend_id=backend_id, pod_name="pod-a", pod_uid="pod-uid-a", node_name="node-a")
        )
        tx.execute(update(tasks_table).where(tasks_table.c.task_id == task_id).values(container_id="container-a"))

    current = journey.attempt(job[0], 0)
    assert current.summary.backend_id == backend_id
    assert current.summary.execution_cluster_id == "journey"
    assert current.runtime is not None
    assert (
        current.runtime.provider_kind,
        current.runtime.name,
        current.runtime.provider_uid,
        current.runtime.provider_node_id,
        current.runtime.container_id,
    ) == ("kubernetes", "pod-a", "pod-uid-a", "node-a", "container-a")

    journey.preempt(job[0])
    journey.settle()
    replacement = journey.attempt(job[0])
    assert replacement.summary.identity.attempt_number == 1
    assert replacement.runtime is None
    with journey.database.transaction() as tx:
        tx.execute(
            update(task_attempts_table)
            .where(task_attempts_table.c.task_id == task_id, task_attempts_table.c.attempt_id == 0)
            .values(backend_id="historical-backend")
        )

    historical = journey.attempt(job[0], 0)
    assert (historical.summary.backend_id, historical.summary.execution_cluster_id, historical.summary.node) == (
        "historical-backend",
        "journey",
        None,
    )
    assert historical.runtime is not None
    assert historical.runtime.container_id == ""


def test_exec_and_profile_after_restart_refuse_a_superseded_attempt_before_runtime(journey, monkeypatch) -> None:
    job = journey.submit("debug-current", preemption_retries=1)
    journey.settle()
    journey.preempt(job[0])
    journey.settle()
    detail = journey.task(job[0])
    stale_identity = detail.attempts[0].identity
    current = detail.summary.current_attempt
    assert current is not None
    assert current.attempt_number == 1
    journey.restart()

    def exec_in_container(*_args, **_kwargs) -> ExecResult:
        return ExecResult(0, "current exec", "", "")

    def profile_task(*_args, **_kwargs) -> ProfileResult:
        return ProfileResult(b"current profile", "")

    monkeypatch.setattr(journey.backend, "exec_in_container", exec_in_container)
    monkeypatch.setattr(journey.backend, "profile_task", profile_task)
    service = ResourceServiceImpl(journey.controller.controller)

    stale = _attempt_identity(stale_identity)
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
                profile=resource_pb2.ProfileType(cpu=resource_pb2.CpuProfile()),
            ),
            None,
        )
    assert profile_error.value.code is Code.FAILED_PRECONDITION

    current_identity = _attempt_identity(current)
    exec_response = service.exec_attempt(
        resource_pb2.ExecAttemptRequest(attempt=current_identity, command=["echo", "hello"]),
        None,
    )
    profile_response = service.profile_attempt(
        resource_pb2.ProfileAttemptRequest(
            attempt=current_identity,
            profile=resource_pb2.ProfileType(cpu=resource_pb2.CpuProfile()),
        ),
        None,
    )

    assert (exec_response.exit_code, exec_response.stdout) == (0, "current exec")
    assert profile_response.profile_data == b"current profile"


@pytest.mark.parametrize("operation", ["exec", "profile"])
def test_debug_action_rechecks_attempt_after_target_resolution(journey, monkeypatch, operation: str) -> None:
    job = journey.submit(f"debug-race-{operation}", preemption_retries=1)
    journey.settle()
    stale = journey.attempt(job[0]).summary.identity
    original_target = journey.controller.controller._task_target

    # Stage a competing controller write in the exact window under test; the
    # oracle is the public error plus zero provider calls, not helper dispatch.
    def replace_after_resolution(identity):
        target = original_target(identity)
        task = journey.task(job[0]).summary
        assert task.current_attempt is not None
        journey.retry_task(
            task.identity,
            expected_attempt_uid=task.current_attempt.attempt_uid,
            idempotency_key=f"debug-race-{operation}",
        )
        journey.settle()
        return target

    runtime_calls: list[str] = []
    monkeypatch.setattr(journey.controller.controller, "_task_target", replace_after_resolution)
    monkeypatch.setattr(
        journey.backend,
        "exec_in_container",
        lambda *_args, **_kwargs: runtime_calls.append("exec") or ExecResult(0, "", "", ""),
    )
    monkeypatch.setattr(
        journey.backend,
        "profile_task",
        lambda *_args, **_kwargs: runtime_calls.append("profile") or ProfileResult(b"", ""),
    )
    service = ResourceServiceImpl(journey.controller.controller)

    with pytest.raises(ConnectError) as error:
        if operation == "exec":
            service.exec_attempt(
                resource_pb2.ExecAttemptRequest(attempt=_attempt_identity(stale), command=["true"]),
                None,
            )
        else:
            service.profile_attempt(
                resource_pb2.ProfileAttemptRequest(
                    attempt=_attempt_identity(stale),
                    profile=resource_pb2.ProfileType(cpu=resource_pb2.CpuProfile()),
                ),
                None,
            )

    assert error.value.code is Code.FAILED_PRECONDITION
    assert runtime_calls == []
    assert journey.attempt(job[0]).summary.identity != stale


def test_exec_and_profile_refuse_an_attempt_from_a_replaced_task_incarnation(journey, monkeypatch) -> None:
    original = journey.submit("debug-replaced")
    journey.settle()
    stale = journey.attempt(original[0]).summary.identity
    journey.succeed(original[0])
    journey.settle()
    journey.restart()

    replacement = journey.submit("debug-replaced")
    journey.settle()
    current = journey.attempt(replacement[0]).summary.identity
    assert current.attempt_number == stale.attempt_number == 0
    assert current.attempt_uid != stale.attempt_uid

    runtime_calls: list[str] = []

    def exec_in_container(*_args, **_kwargs) -> ExecResult:
        runtime_calls.append("exec")
        return ExecResult(0, "", "", "")

    def profile_task(*_args, **_kwargs) -> ProfileResult:
        runtime_calls.append("profile")
        return ProfileResult(b"", "")

    monkeypatch.setattr(journey.backend, "exec_in_container", exec_in_container)
    monkeypatch.setattr(journey.backend, "profile_task", profile_task)
    service = ResourceServiceImpl(journey.controller.controller)
    stale_proto = _attempt_identity(stale)

    with pytest.raises(ConnectError) as exec_error:
        service.exec_attempt(resource_pb2.ExecAttemptRequest(attempt=stale_proto, command=["true"]), None)
    with pytest.raises(ConnectError) as profile_error:
        service.profile_attempt(
            resource_pb2.ProfileAttemptRequest(
                attempt=stale_proto,
                profile=resource_pb2.ProfileType(cpu=resource_pb2.CpuProfile()),
            ),
            None,
        )

    assert exec_error.value.code is Code.FAILED_PRECONDITION
    assert profile_error.value.code is Code.FAILED_PRECONDITION
    assert runtime_calls == []
