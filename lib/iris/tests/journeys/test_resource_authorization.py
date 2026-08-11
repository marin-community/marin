# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf import any_pb2
from iris.rpc import iris_logging_pb2, resource_pb2
from iris.rpc.auth import DASHBOARD_ROLE
from iris.rpc.resource_registrations import resource_catalog
from iris.rpc.resource_service import ResourceServiceImpl
from iris.rpc.resource_types import ACTIVITY_ENTRY, ENDPOINT, JOB, LOG_ENTRY, OPERATION, TASK
from rigging.server_auth import VerifiedIdentity, identity_scope


def _key(kind: int, resource_id: str) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(cluster_id="journey", kind=kind, resource_id=resource_id)


def _pack(value) -> any_pb2.Any:
    result = any_pb2.Any()
    result.Pack(value)
    return result


def _service(journey) -> ResourceServiceImpl:
    return ResourceServiceImpl(resource_catalog(journey.controller.controller))


def _ref(resource_type: str, resource_id: str) -> resource_pb2.ResourceRef:
    return resource_pb2.ResourceRef(authority_cluster_id="journey", type=resource_type, id=resource_id)


def test_resource_reads_scope_users_before_rows_cross_the_rpc_boundary(journey) -> None:
    alice = journey.submit("alice-job", user="alice")
    bob = journey.submit("bob-job", user="bob")
    journey.settle()
    service = _service(journey)

    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        jobs = service.list_resources(
            resource_pb2.ListResourcesRequest(
                type=JOB,
                query=_pack(resource_pb2.JobQuery()),
                view=resource_pb2.RESOURCE_VIEW_BASIC,
            ),
            None,
        )
        tasks = service.list_resources(
            resource_pb2.ListResourcesRequest(
                type=TASK,
                query=_pack(resource_pb2.TaskQuery()),
                view=resource_pb2.RESOURCE_VIEW_BASIC,
            ),
            None,
        )
        job_bodies = [resource_pb2.JobSummary.FromString(resource.body.value) for resource in jobs.resources]
        task_bodies = [resource_pb2.TaskSummary.FromString(resource.body.value) for resource in tasks.resources]
        assert [job.identity.key.resource_id for job in job_bodies] == [alice.wire_id]
        assert [task.identity.key.resource_id for task in task_bodies] == [f"{alice.wire_id}/0"]

        with pytest.raises(ConnectError) as denied:
            service.get_resource(
                resource_pb2.GetResourceRequest(
                    ref=_ref(JOB, bob.wire_id),
                    view=resource_pb2.RESOURCE_VIEW_FULL,
                ),
                None,
            )
        assert denied.value.code is Code.PERMISSION_DENIED

        with pytest.raises(ConnectError) as denied:
            service.get_resource(
                resource_pb2.GetResourceRequest(
                    ref=_ref(TASK, f"{bob.wire_id}/0"),
                    view=resource_pb2.RESOURCE_VIEW_FULL,
                ),
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
    service = _service(journey)

    with identity_scope(VerifiedIdentity(user_id="viewer@example.com", role=DASHBOARD_ROLE)):
        logs = service.list_resources(
            resource_pb2.ListResourcesRequest(
                type=LOG_ENTRY,
                query=_pack(
                    resource_pb2.FetchLogsRequest(
                        target=resource_pb2.LogTarget(
                            task=resource_pb2.TaskIdentity(
                                key=_key(resource_pb2.RESOURCE_KIND_TASK, task.identity.key.resource_id),
                                task_uid=task.identity.task_uid,
                            )
                        )
                    )
                ),
                view=resource_pb2.RESOURCE_VIEW_FULL,
            ),
            None,
        )
        activity = service.list_resources(
            resource_pb2.ListResourcesRequest(
                type=ACTIVITY_ENTRY,
                query=_pack(
                    resource_pb2.ActivityQuery(
                        target=_key(resource_pb2.RESOURCE_KIND_TASK, task.identity.key.resource_id)
                    )
                ),
                view=resource_pb2.RESOURCE_VIEW_FULL,
            ),
            None,
        )
        durable = service.get_resource(
            resource_pb2.GetResourceRequest(
                ref=_ref(OPERATION, receipt.action_id),
                view=resource_pb2.RESOURCE_VIEW_FULL,
            ),
            None,
        )

    log_bodies = [iris_logging_pb2.LogEntry.FromString(resource.body.value) for resource in logs.resources]
    activity_bodies = [resource_pb2.ActivityEntry.FromString(resource.body.value) for resource in activity.resources]
    durable_operation = resource_pb2.Operation.FromString(durable.resource.body.value)
    durable_body = resource_pb2.ActionReceipt.FromString(durable_operation.result.value)
    assert [entry.data for entry in log_bodies] == ["worker became ready"]
    assert any(entry.correlation_id == receipt.action_id for entry in activity_bodies)
    assert durable_body.action_id == receipt.action_id


def test_endpoint_reads_are_scoped_to_the_resource_owner(journey) -> None:
    alice = journey.submit("alice-endpoint", user="alice")
    bob = journey.submit("bob-endpoint", user="bob")
    journey.settle()
    journey.register_endpoint(alice[0], "alice-service", "alice:1", endpoint_id="alice-endpoint")
    journey.register_endpoint(bob[0], "bob-service", "bob:1", endpoint_id="bob-endpoint")
    service = _service(journey)

    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        listed = service.list_resources(
            resource_pb2.ListResourcesRequest(
                type=ENDPOINT,
                query=_pack(resource_pb2.EndpointQuery()),
                view=resource_pb2.RESOURCE_VIEW_BASIC,
            ),
            None,
        )
        with pytest.raises(ConnectError) as denied:
            service.get_resource(
                resource_pb2.GetResourceRequest(
                    ref=_ref(ENDPOINT, "bob-endpoint"),
                    view=resource_pb2.RESOURCE_VIEW_FULL,
                ),
                None,
            )

    endpoint_bodies = [resource_pb2.EndpointSummary.FromString(resource.body.value) for resource in listed.resources]
    assert [endpoint.endpoint_id for endpoint in endpoint_bodies] == ["alice-endpoint"]
    assert denied.value.code is Code.PERMISSION_DENIED


def test_worker_endpoint_reads_expose_only_system_discovery(journey) -> None:
    job = journey.submit("worker-endpoint-scope", user="alice")
    journey.settle()
    journey.register_endpoint(job[0], "private-service", "private:1", endpoint_id="private-endpoint")
    journey.controller.endpoint_service.registry.register_system_endpoint(
        "/system/log-server", journey.log_stack.address
    )
    service = _service(journey)

    with identity_scope(VerifiedIdentity(user_id="worker", role="worker")):
        response = service.list_resources(
            resource_pb2.ListResourcesRequest(
                type=ENDPOINT,
                query=_pack(resource_pb2.EndpointQuery()),
                view=resource_pb2.RESOURCE_VIEW_BASIC,
            ),
            None,
        )
        system = service.batch_get_resources(
            resource_pb2.BatchGetResourcesRequest(
                type=ENDPOINT,
                refs=[_ref(ENDPOINT, "/system/log-server")],
                view=resource_pb2.RESOURCE_VIEW_FULL,
            ),
            None,
        )
        with pytest.raises(ConnectError) as denied:
            service.batch_get_resources(
                resource_pb2.BatchGetResourcesRequest(
                    type=ENDPOINT,
                    refs=[_ref(ENDPOINT, "private-endpoint")],
                    view=resource_pb2.RESOURCE_VIEW_FULL,
                ),
                None,
            )

    endpoints = [resource_pb2.EndpointSummary.FromString(resource.body.value) for resource in response.resources]
    assert [endpoint.name for endpoint in endpoints] == ["/system/log-server"]
    system_detail = resource_pb2.EndpointDetail.FromString(system.results[0].resource.body.value)
    assert system_detail.summary.name == "/system/log-server"
    assert denied.value.code is Code.PERMISSION_DENIED
