# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.rpc import resource_pb2
from iris.rpc.auth import DASHBOARD_ROLE
from iris.rpc.resource_service import ResourceServiceImpl
from rigging.server_auth import VerifiedIdentity, identity_scope


def _key(kind: int, resource_id: str) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(cluster_id="journey", kind=kind, resource_id=resource_id)


def test_resource_reads_scope_users_before_rows_cross_the_rpc_boundary(journey) -> None:
    alice = journey.submit("alice-job", user="alice")
    bob = journey.submit("bob-job", user="bob")
    journey.settle()
    service = ResourceServiceImpl(journey.controller.controller)

    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        jobs = service.list_jobs(resource_pb2.ListJobsRequest(), None)
        tasks = service.list_tasks(resource_pb2.ListTasksRequest(), None)
        assert [job.identity.key.resource_id for job in jobs.jobs] == [alice.wire_id]
        assert [task.identity.key.resource_id for task in tasks.tasks] == [f"{alice.wire_id}/0"]

        with pytest.raises(ConnectError) as denied:
            service.describe_job(
                resource_pb2.DescribeJobRequest(job=_key(resource_pb2.RESOURCE_KIND_JOB, bob.wire_id)),
                None,
            )
        assert denied.value.code is Code.PERMISSION_DENIED

        with pytest.raises(ConnectError) as denied:
            service.describe_task(
                resource_pb2.DescribeTaskRequest(task=_key(resource_pb2.RESOURCE_KIND_TASK, f"{bob.wire_id}/0")),
                None,
            )
        assert denied.value.code is Code.PERMISSION_DENIED


def test_dashboard_identity_can_read_logs_activity_and_action_receipts(journey) -> None:
    job = journey.submit("dashboard-diagnostics", user="alice", preemption_retries=1)
    journey.settle()
    task = journey.task(job[0]).summary
    current = task.current_attempt
    assert current is not None
    journey.push_task_logs(job[0], ["worker became ready"])
    receipt = journey.retry_task(
        task.identity,
        expected_attempt_uid=current.attempt_uid,
        idempotency_key="dashboard-diagnostics",
    )
    service = ResourceServiceImpl(journey.controller.controller)

    with identity_scope(VerifiedIdentity(user_id="viewer@example.com", role=DASHBOARD_ROLE)):
        logs = service.fetch_logs(
            resource_pb2.FetchLogsRequest(
                target=resource_pb2.LogTarget(
                    task=resource_pb2.TaskIdentity(
                        key=_key(resource_pb2.RESOURCE_KIND_TASK, task.identity.key.resource_id),
                        task_uid=task.identity.task_uid,
                    )
                )
            ),
            None,
        )
        activity = service.list_activity(
            resource_pb2.ListActivityRequest(
                query=resource_pb2.ActivityQuery(
                    target=_key(resource_pb2.RESOURCE_KIND_TASK, task.identity.key.resource_id)
                )
            ),
            None,
        )
        durable = service.get_action_receipt(
            resource_pb2.GetActionReceiptRequest(action_id=receipt.action_id),
            None,
        )

    assert [entry.data for entry in logs.entries] == ["worker became ready"]
    assert any(entry.correlation_id == receipt.action_id for entry in activity.entries)
    assert durable.receipt.action_id == receipt.action_id
