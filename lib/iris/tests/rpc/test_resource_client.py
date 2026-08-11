# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf import any_pb2
from google.protobuf.message import Message
from iris.client.resolver import ClusterResolver
from iris.resources.action import ActionResult, ActionState
from iris.resources.endpoint import ThreadsProfileConfiguration
from iris.resources.execution import CommandEntrypoint, Environment, ResourceSpec, RuntimeEntrypoint
from iris.resources.identity import AttemptIdentity, AttemptLocator, JobIdentity, ResourceKey, ResourceKind
from iris.resources.job import ContainerProfile, ExistingJobPolicy, JobPreemptionPolicy, JobSpec, PriorityBand
from iris.resources.names import Namespace
from iris.resources.node import NodeQuery
from iris.resources.task import TaskQuery
from iris.rpc import job_pb2, resource_pb2, time_pb2
from iris.rpc.resource_client import ResourceRpcClient
from iris.rpc.resource_types import ATTEMPT, ENDPOINT, EXEC_SESSION, JOB, NODE, PROFILE_CAPTURE, TASK
from rigging.timing import Duration


def _key(kind: int, resource_id: str) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(cluster_id="prod", kind=kind, resource_id=resource_id)


def _pack(value: Message) -> any_pb2.Any:
    result = any_pb2.Any()
    result.Pack(value)
    return result


def _unpack(value: any_pb2.Any, message_type):
    result = message_type()
    assert value.Unpack(result)
    return result


def _resource(ref: resource_pb2.ResourceRef, body: Message) -> resource_pb2.Resource:
    return resource_pb2.Resource(ref=ref, body=_pack(body))


def _operation(result: Message) -> resource_pb2.Operation:
    return resource_pb2.Operation(
        ref=resource_pb2.ResourceRef(authority_cluster_id="prod", type="iris/operation", id="op-1"),
        phase=resource_pb2.OPERATION_PHASE_VERIFIED,
        result=_pack(result),
    )


class _ResourceRpc:
    def __init__(self, *_args, **_kwargs) -> None:
        self.requests = []
        self.rpc_timeouts: list[tuple[str, int | None]] = []

    def close(self) -> None:
        pass

    def create_resource(
        self,
        request: resource_pb2.CreateResourceRequest,
        *,
        timeout_ms: int | None = None,
    ) -> resource_pb2.Operation:
        self.requests.append(request)
        self.rpc_timeouts.append((request.type, timeout_ms))
        if request.type == JOB:
            return _operation(
                resource_pb2.SubmitJobResponse(
                    job=resource_pb2.JobIdentity(
                        key=_key(resource_pb2.RESOURCE_KIND_JOB, "/alice/train"),
                        job_uid="job-uid",
                    )
                )
            )
        if request.type == EXEC_SESSION:
            return _operation(resource_pb2.ExecAttemptResponse(exit_code=0, stdout="done"))
        if request.type == PROFILE_CAPTURE:
            return _operation(resource_pb2.ProfileAttemptResponse(profile_data=b"profile"))
        raise AssertionError(request.type)

    def list_resources(self, request: resource_pb2.ListResourcesRequest) -> resource_pb2.ListResourcesResponse:
        self.requests.append(request)
        if request.type == TASK:
            item = resource_pb2.TaskSummary(
                identity=resource_pb2.TaskIdentity(
                    key=_key(resource_pb2.RESOURCE_KIND_TASK, "/alice/train/7"),
                    task_uid="task-uid-7",
                ),
                job=resource_pb2.JobIdentity(
                    key=_key(resource_pb2.RESOURCE_KIND_JOB, "/alice/train"),
                    job_uid="job-uid",
                ),
                task_index=7,
                state=job_pb2.TASK_STATE_RUNNING,
                execution_cluster_id="prod",
                backend_id="east",
                current_attempt=resource_pb2.AttemptIdentity(
                    task=_key(resource_pb2.RESOURCE_KIND_TASK, "/alice/train/7"),
                    attempt_number=2,
                    attempt_uid="attempt-uid-2",
                ),
                submitted_at=time_pb2.Timestamp(epoch_ms=1_000),
            )
            return resource_pb2.ListResourcesResponse(
                resources=[
                    _resource(
                        resource_pb2.ResourceRef(
                            authority_cluster_id="prod",
                            type=TASK,
                            id="/alice/train/7",
                            uid="task-uid-7",
                        ),
                        item,
                    )
                ],
                source_statuses=[
                    resource_pb2.ResourceSourceStatus(
                        source_id="backend:west",
                        backend_id="west",
                        state=resource_pb2.SOURCE_STATE_UNAVAILABLE,
                        freshness=resource_pb2.FRESHNESS_STALE,
                        error_code="backend_unavailable",
                        error_message="west did not answer",
                    )
                ],
            )
        if request.type == NODE:
            item = resource_pb2.NodeSummary(
                identity=resource_pb2.NodeIdentity(
                    key=_key(resource_pb2.RESOURCE_KIND_NODE, "worker-1"),
                    backend_id="east",
                    node_uid="node-uid-1",
                ),
                health=resource_pb2.NODE_HEALTH_READY,
                schedulable=True,
                capacity=resource_pb2.NodeCapacity(cpu_millicores=2_000, memory_bytes=4_096),
                observed_at=time_pb2.Timestamp(epoch_ms=1_000),
                region="us-east5",
            )
            return resource_pb2.ListResourcesResponse(
                resources=[
                    _resource(
                        resource_pb2.ResourceRef(
                            authority_cluster_id="prod",
                            type=NODE,
                            id="east:worker-1",
                            uid="node-uid-1",
                        ),
                        item,
                    )
                ]
            )
        if request.type == ENDPOINT:
            query = _unpack(request.query, resource_pb2.EndpointQuery)
            if not query.page.page_token:
                endpoints = (
                    ("endpoint-1", "/actors/coordinator"),
                    ("endpoint-noise", "/actors/coordinator-metrics"),
                )
                next_page_token = "next"
            else:
                endpoints = (("endpoint-2", "/actors/coordinator"),)
                next_page_token = ""
            return resource_pb2.ListResourcesResponse(
                resources=[
                    _resource(
                        resource_pb2.ResourceRef(authority_cluster_id="prod", type=ENDPOINT, id=endpoint_id),
                        resource_pb2.EndpointSummary(
                            key=_key(resource_pb2.RESOURCE_KIND_ENDPOINT, endpoint_id),
                            endpoint_id=endpoint_id,
                            name=name,
                            execution_cluster_id="prod",
                        ),
                    )
                    for endpoint_id, name in endpoints
                ],
                next_page_token=next_page_token,
            )
        raise AssertionError(request.type)

    def get_resource(self, request: resource_pb2.GetResourceRequest) -> resource_pb2.GetResourceResponse:
        self.requests.append(request)
        if request.ref.type == JOB:
            summary = resource_pb2.JobSummary(
                identity=resource_pb2.JobIdentity(
                    key=_key(resource_pb2.RESOURCE_KIND_JOB, request.ref.id),
                    job_uid="job-uid",
                ),
                state=resource_pb2.JOB_STATE_RUNNING,
            )
            return resource_pb2.GetResourceResponse(resource=_resource(request.ref, summary))
        if request.ref.type == ATTEMPT:
            _, attempt = request.ref.id.rsplit(":", 1)
            number = 4 if attempt == "current" else int(attempt)
            detail = resource_pb2.AttemptDetail(
                summary=resource_pb2.AttemptSummary(
                    identity=resource_pb2.AttemptIdentity(
                        task=_key(resource_pb2.RESOURCE_KIND_TASK, "/alice/train/7"),
                        attempt_number=number,
                        attempt_uid=f"attempt-{number}",
                    ),
                    state=job_pb2.TASK_STATE_RUNNING,
                    execution_cluster_id="prod",
                    backend_id="east",
                    created_at=time_pb2.Timestamp(epoch_ms=1_000),
                )
            )
            return resource_pb2.GetResourceResponse(resource=_resource(request.ref, detail))
        raise AssertionError(request.ref.type)

    def batch_get_resources(
        self, request: resource_pb2.BatchGetResourcesRequest
    ) -> resource_pb2.BatchGetResourcesResponse:
        self.requests.append(request)
        if request.type == TASK:
            results = []
            for ref in request.refs:
                detail = resource_pb2.TaskDetail(
                    summary=resource_pb2.TaskSummary(
                        identity=resource_pb2.TaskIdentity(
                            key=_key(resource_pb2.RESOURCE_KIND_TASK, ref.id),
                            task_uid=f"uid-{ref.id}",
                        ),
                        job=resource_pb2.JobIdentity(
                            key=_key(resource_pb2.RESOURCE_KIND_JOB, "/alice/train"),
                            job_uid="job-uid",
                        ),
                        state=job_pb2.TASK_STATE_RUNNING,
                        execution_cluster_id="prod",
                        backend_id="east",
                        submitted_at=time_pb2.Timestamp(epoch_ms=1_000),
                    )
                )
                results.append(resource_pb2.BatchGetResourceResult(resource=_resource(ref, detail)))
            return resource_pb2.BatchGetResourcesResponse(results=results)
        if request.type == ENDPOINT:
            return resource_pb2.BatchGetResourcesResponse(
                results=[
                    resource_pb2.BatchGetResourceResult(
                        resource=_resource(
                            ref,
                            resource_pb2.EndpointDetail(
                                summary=resource_pb2.EndpointSummary(
                                    key=_key(resource_pb2.RESOURCE_KIND_ENDPOINT, ref.id),
                                    endpoint_id=ref.id,
                                    name="/actors/coordinator",
                                    execution_cluster_id="prod",
                                ),
                                address=f"{ref.id}:8080",
                                metadata={"replica": ref.id[-1]},
                            ),
                        )
                    )
                    for ref in request.refs
                ]
            )
        raise AssertionError(request.type)

    def update_resource(self, request: resource_pb2.UpdateResourceRequest) -> resource_pb2.Operation:
        self.requests.append(request)
        assert request.ref.type == JOB
        receipt = resource_pb2.ActionReceipt(
            action_id="action-1",
            kind=resource_pb2.ACTION_KIND_CANCEL_JOB,
            target=_key(resource_pb2.RESOURCE_KIND_JOB, request.ref.id),
            expected_target_uid=request.ref.uid,
            state=resource_pb2.ACTION_STATE_SUCCEEDED,
            result_code=resource_pb2.ACTION_RESULT_SATISFIED,
            created_at=time_pb2.Timestamp(epoch_ms=1_000),
            updated_at=time_pb2.Timestamp(epoch_ms=2_000),
            completed_at=time_pb2.Timestamp(epoch_ms=2_000),
        )
        return _operation(receipt)


def _client(monkeypatch) -> tuple[ResourceRpcClient, _ResourceRpc]:
    rpc = _ResourceRpc()
    monkeypatch.setattr("iris.rpc.resource_client.ResourceServiceClientSync", lambda *_a, **_kw: rpc)
    return ResourceRpcClient("http://controller.test"), rpc


@pytest.mark.parametrize("operation", ["submit", "exec", "profile"])
def test_non_idempotent_calls_are_not_automatically_replayed(monkeypatch, operation: str) -> None:
    client, rpc = _client(monkeypatch)
    attempts = 0

    def unavailable(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise ConnectError(Code.UNAVAILABLE, "response lost")

    monkeypatch.setattr(rpc, "create_resource", unavailable)
    with pytest.raises(ConnectError, match="response lost"):
        if operation == "submit":
            client.submit_job(_job_spec())
        elif operation == "exec":
            client.exec_attempt(
                AttemptIdentity(ResourceKey("prod", ResourceKind.TASK, "/alice/train/0"), 0, "attempt-uid"),
                command=("true",),
                timeout=Duration.from_seconds(1),
            )
        else:
            client.profile_attempt(
                AttemptIdentity(ResourceKey("prod", ResourceKind.TASK, "/alice/train/0"), 0, "attempt-uid"),
                profile=ThreadsProfileConfiguration(include_locals=False),
                duration=Duration.from_seconds(1),
            )
    assert attempts == 1


def _job_spec() -> JobSpec:
    return JobSpec(
        version=1,
        name="/alice/train",
        entrypoint=RuntimeEntrypoint((), CommandEntrypoint(()), {}, {}),
        resources=ResourceSpec(cpu=1),
        environment=Environment({}, ()),
        bundle_id="",
        scheduling_timeout=None,
        ports=(),
        max_task_failures=0,
        max_retries_failure=0,
        max_retries_preemption=0,
        constraints=(),
        coscheduling=None,
        replicas=1,
        timeout=None,
        fail_if_exists=False,
        preemption_policy=JobPreemptionPolicy.UNSPECIFIED,
        existing_job_policy=ExistingJobPolicy.RECREATE,
        priority_band=PriorityBand.INHERIT,
        task_image="",
        submit_argv=(),
        client_revision_date="",
        container_profile=ContainerProfile.UNSPECIFIED,
    )


def test_submit_job_uses_generic_create_with_replacement_deadline(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    identity = client.submit_job(_job_spec(), bundle=b"bundle")
    assert identity.job_uid == "job-uid"
    request = rpc.requests[-1]
    body = _unpack(request.body, resource_pb2.SubmitJobRequest)
    assert request.type == JOB
    assert body.bundle_blob == b"bundle"
    assert rpc.rpc_timeouts[-1] == (JOB, 180_000)


def test_list_tasks_returns_rows_alongside_unavailable_source(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    page = client.list_tasks(TaskQuery(backend_id="east", page_size=17))
    assert page.items[0].current_attempt is not None
    assert page.items[0].current_attempt.attempt_uid == "attempt-uid-2"
    assert page.source_statuses[0].error_code == "backend_unavailable"
    request = rpc.requests[0]
    query = _unpack(request.query, resource_pb2.TaskQuery)
    assert request.type == TASK
    assert query.backend_id == "east"
    assert query.page.page_size == 17


def test_job_state_uses_exact_generic_ref(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    identity = JobIdentity(ResourceKey("prod", ResourceKind.JOB, "/alice/train"), "job-uid")
    assert client.job_state(identity).name == "RUNNING"
    request = rpc.requests[0]
    assert request.ref.type == JOB
    assert request.ref.id == "/alice/train"
    assert request.ref.uid == "job-uid"


def test_batch_describe_tasks_preserves_requested_order(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    keys = (
        ResourceKey("prod", ResourceKind.TASK, "/alice/train/2"),
        ResourceKey("prod", ResourceKind.TASK, "/alice/train/7"),
    )
    details = client.describe_tasks(keys)
    assert [detail.summary.identity.key for detail in details] == list(keys)
    assert [ref.id for ref in rpc.requests[0].refs] == ["/alice/train/2", "/alice/train/7"]


def test_describe_attempt_distinguishes_current_and_numbered_refs(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    task = ResourceKey("prod", ResourceKind.TASK, "/alice/train/7")
    current = client.describe_attempt(AttemptLocator(task, None))
    exact = client.describe_attempt(AttemptLocator(task, 2))
    assert current.summary.identity.attempt_number == 4
    assert exact.summary.identity.attempt_uid == "attempt-2"
    assert rpc.requests[0].ref.id.endswith(":current")
    assert rpc.requests[1].ref.id.endswith(":2")


def test_list_nodes_preserves_region_and_typed_query(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    page = client.list_nodes(NodeQuery(page_size=500))
    assert page.items[0].region == "us-east5"
    query = _unpack(rpc.requests[0].query, resource_pb2.NodeQuery)
    assert query.page.page_size == 500


def test_resolve_endpoints_pages_and_batches_exact_resource_details(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    endpoints = client.resolve_endpoints("/actors/coordinator")
    assert [endpoint.summary.endpoint_id for endpoint in endpoints] == ["endpoint-1", "endpoint-2"]
    assert [endpoint.address for endpoint in endpoints] == ["endpoint-1:8080", "endpoint-2:8080"]
    assert [endpoint.metadata for endpoint in endpoints] == [{"replica": "1"}, {"replica": "2"}]


def test_cluster_resolver_uses_exact_resource_matches_across_pages(monkeypatch) -> None:
    rpc = _ResourceRpc()
    monkeypatch.setattr("iris.rpc.resource_client.ResourceServiceClientSync", lambda *_a, **_kw: rpc)
    resolver = ClusterResolver("http://controller.test", namespace=Namespace("/actors"))
    result = resolver.resolve("coordinator")
    assert [(endpoint.actor_id, endpoint.url) for endpoint in result.endpoints] == [
        ("endpoint-1", "http://endpoint-1:8080"),
        ("endpoint-2", "http://endpoint-2:8080"),
    ]
    assert [endpoint.metadata for endpoint in result.endpoints] == [{"replica": "1"}, {"replica": "2"}]


def test_cancel_job_uses_generic_update_with_exact_ref(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    identity = JobIdentity(ResourceKey("prod", ResourceKind.JOB, "/alice/train"), "job-uid")
    receipt = client.cancel_job(identity, idempotency_key="operator-request-9")
    assert receipt.state is ActionState.SUCCEEDED
    assert receipt.result_code is ActionResult.SATISFIED
    request = rpc.requests[0]
    assert request.ref.uid == "job-uid"
    assert request.update.new_state == resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED
    assert request.mutation.request_id == "operator-request-9"


def test_exec_attempt_generic_create_deadline_outlasts_command_timeout(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    task = ResourceKey("prod", ResourceKind.TASK, "/alice/train/7")
    requested_timeout = Duration.from_minutes(20)
    result = client.exec_attempt(
        AttemptIdentity(task, 2, "attempt-uid"),
        command=("sleep", "600"),
        timeout=requested_timeout,
    )
    assert result.stdout == "done"
    request = rpc.requests[-1]
    body = _unpack(request.body, resource_pb2.ExecAttemptRequest)
    assert body.timeout.milliseconds == requested_timeout.to_ms()
    assert request.parent.uid == "attempt-uid"
    operation, rpc_timeout_ms = rpc.rpc_timeouts[-1]
    assert operation == EXEC_SESSION
    assert rpc_timeout_ms is not None and rpc_timeout_ms > requested_timeout.to_ms()


def test_profile_attempt_generic_create_deadline_outlasts_capture_duration(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    task = ResourceKey("prod", ResourceKind.TASK, "/alice/train/7")
    requested_duration = Duration.from_minutes(20)
    result = client.profile_attempt(
        AttemptIdentity(task, 2, "attempt-uid"),
        profile=ThreadsProfileConfiguration(include_locals=False),
        duration=requested_duration,
    )
    assert result.profile_data == b"profile"
    request = rpc.requests[-1]
    body = _unpack(request.body, resource_pb2.ProfileAttemptRequest)
    assert body.duration.milliseconds == requested_duration.to_ms()
    operation, rpc_timeout_ms = rpc.rpc_timeouts[-1]
    assert operation == PROFILE_CAPTURE
    assert rpc_timeout_ms is not None and rpc_timeout_ms > requested_duration.to_ms()
