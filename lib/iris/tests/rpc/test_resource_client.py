# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.client.resolver import ClusterResolver
from iris.cluster.types import Namespace
from iris.resources.action import ActionResult, ActionState
from iris.resources.endpoint import ThreadsProfileConfiguration
from iris.resources.execution import CommandEntrypoint, Environment, ResourceSpec, RuntimeEntrypoint
from iris.resources.identity import AttemptIdentity, AttemptLocator, JobIdentity, ResourceKey, ResourceKind
from iris.resources.job import ContainerProfile, ExistingJobPolicy, JobPreemptionPolicy, JobSpec, PriorityBand
from iris.resources.node import NodeQuery
from iris.resources.task import TaskQuery
from iris.rpc import job_pb2, resource_pb2, time_pb2
from iris.rpc.resource_client import ResourceRpcClient
from rigging.timing import Duration


def _key(kind: int, resource_id: str) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(cluster_id="prod", kind=kind, resource_id=resource_id)


class _ResourceRpc:
    def __init__(self, *_args, **_kwargs) -> None:
        self.requests = []
        self.rpc_timeouts: list[tuple[str, int | None]] = []

    def close(self) -> None:
        pass

    def submit_job(
        self,
        request: resource_pb2.SubmitJobRequest,
        *,
        timeout_ms: int | None = None,
    ) -> resource_pb2.SubmitJobResponse:
        self.requests.append(request)
        self.rpc_timeouts.append(("submit", timeout_ms))
        return resource_pb2.SubmitJobResponse(
            job=resource_pb2.JobIdentity(
                key=_key(resource_pb2.RESOURCE_KIND_JOB, "/alice/train"),
                job_uid="job-uid",
            )
        )

    def list_tasks(self, request: resource_pb2.ListTasksRequest) -> resource_pb2.ListTasksResponse:
        self.requests.append(request)
        return resource_pb2.ListTasksResponse(
            tasks=[
                resource_pb2.TaskSummary(
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
            ],
            page=resource_pb2.PageInfo(
                source_statuses=[
                    resource_pb2.ResourceSourceStatus(
                        source_id="backend:west",
                        backend_id="west",
                        state=resource_pb2.SOURCE_STATE_UNAVAILABLE,
                        freshness=resource_pb2.FRESHNESS_STALE,
                        error_code="backend_unavailable",
                        error_message="west did not answer",
                    )
                ]
            ),
        )

    def batch_describe_tasks(
        self, request: resource_pb2.BatchDescribeTasksRequest
    ) -> resource_pb2.BatchDescribeTasksResponse:
        self.requests.append(request)
        return resource_pb2.BatchDescribeTasksResponse(
            tasks=[
                resource_pb2.TaskDetail(
                    summary=resource_pb2.TaskSummary(
                        identity=resource_pb2.TaskIdentity(key=key, task_uid=f"uid-{key.resource_id}"),
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
                for key in request.tasks
            ]
        )

    def describe_attempt(self, request: resource_pb2.DescribeAttemptRequest) -> resource_pb2.DescribeAttemptResponse:
        self.requests.append(request)
        number = request.attempt.attempt_number if request.attempt.HasField("attempt_number") else 4
        return resource_pb2.DescribeAttemptResponse(
            attempt=resource_pb2.AttemptDetail(
                summary=resource_pb2.AttemptSummary(
                    identity=resource_pb2.AttemptIdentity(
                        task=request.attempt.task,
                        attempt_number=number,
                        attempt_uid=f"attempt-{number}",
                    ),
                    state=job_pb2.TASK_STATE_RUNNING,
                    execution_cluster_id="prod",
                    backend_id="east",
                    created_at=time_pb2.Timestamp(epoch_ms=1_000),
                )
            )
        )

    def list_nodes(self, request: resource_pb2.ListNodesRequest) -> resource_pb2.ListNodesResponse:
        self.requests.append(request)
        return resource_pb2.ListNodesResponse(
            nodes=[
                resource_pb2.NodeSummary(
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
            ]
        )

    def list_endpoints(self, request: resource_pb2.ListEndpointsRequest) -> resource_pb2.ListEndpointsResponse:
        self.requests.append(request)
        if not request.query.page.page_token:
            endpoints = [
                resource_pb2.EndpointSummary(
                    key=_key(resource_pb2.RESOURCE_KIND_ENDPOINT, "endpoint-1"),
                    endpoint_id="endpoint-1",
                    name="/actors/coordinator",
                    execution_cluster_id="prod",
                ),
                resource_pb2.EndpointSummary(
                    key=_key(resource_pb2.RESOURCE_KIND_ENDPOINT, "endpoint-noise"),
                    endpoint_id="endpoint-noise",
                    name="/actors/coordinator-metrics",
                    execution_cluster_id="prod",
                ),
            ]
            next_page_token = "next"
        else:
            endpoints = [
                resource_pb2.EndpointSummary(
                    key=_key(resource_pb2.RESOURCE_KIND_ENDPOINT, "endpoint-2"),
                    endpoint_id="endpoint-2",
                    name="/actors/coordinator",
                    execution_cluster_id="prod",
                )
            ]
            next_page_token = ""
        return resource_pb2.ListEndpointsResponse(
            endpoints=endpoints,
            page=resource_pb2.PageInfo(next_page_token=next_page_token),
        )

    def batch_describe_endpoints(
        self,
        request: resource_pb2.BatchDescribeEndpointsRequest,
    ) -> resource_pb2.BatchDescribeEndpointsResponse:
        self.requests.append(request)
        return resource_pb2.BatchDescribeEndpointsResponse(
            endpoints=[
                resource_pb2.EndpointDetail(
                    summary=resource_pb2.EndpointSummary(
                        key=key,
                        endpoint_id=key.resource_id,
                        name="/actors/coordinator",
                        execution_cluster_id="prod",
                    ),
                    address=f"{key.resource_id}:8080",
                    metadata={"replica": key.resource_id[-1]},
                )
                for key in request.endpoints
            ]
        )

    def cancel_job(self, request: resource_pb2.CancelJobRequest) -> resource_pb2.ActionResponse:
        self.requests.append(request)
        return resource_pb2.ActionResponse(
            receipt=resource_pb2.ActionReceipt(
                action_id="action-1",
                kind=resource_pb2.ACTION_KIND_CANCEL_JOB,
                target=request.job.key,
                expected_target_uid=request.job.job_uid,
                state=resource_pb2.ACTION_STATE_SUCCEEDED,
                result_code=resource_pb2.ACTION_RESULT_SATISFIED,
                created_at=time_pb2.Timestamp(epoch_ms=1_000),
                updated_at=time_pb2.Timestamp(epoch_ms=2_000),
                completed_at=time_pb2.Timestamp(epoch_ms=2_000),
            )
        )

    def exec_attempt(
        self,
        request: resource_pb2.ExecAttemptRequest,
        *,
        timeout_ms: int | None = None,
    ) -> resource_pb2.ExecAttemptResponse:
        self.requests.append(request)
        self.rpc_timeouts.append(("exec", timeout_ms))
        return resource_pb2.ExecAttemptResponse(exit_code=0, stdout="done")

    def profile_attempt(
        self,
        request: resource_pb2.ProfileAttemptRequest,
        *,
        timeout_ms: int | None = None,
    ) -> resource_pb2.ProfileAttemptResponse:
        self.requests.append(request)
        self.rpc_timeouts.append(("profile", timeout_ms))
        return resource_pb2.ProfileAttemptResponse(profile_data=b"profile")


def _client(monkeypatch) -> tuple[ResourceRpcClient, _ResourceRpc]:
    rpc = _ResourceRpc()
    monkeypatch.setattr("iris.rpc.resource_client.ResourceServiceClientSync", lambda *_a, **_kw: rpc)
    return ResourceRpcClient("http://controller.test"), rpc


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


def test_submit_job_rpc_deadline_covers_replacement_drain(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)

    identity = client.submit_job(_job_spec(), bundle=b"bundle")

    assert identity.job_uid == "job-uid"
    assert rpc.requests[-1].bundle_blob == b"bundle"
    operation, rpc_timeout_ms = rpc.rpc_timeouts[-1]
    assert operation == "submit"
    assert rpc_timeout_ms == 180_000


def test_list_tasks_returns_rows_alongside_unavailable_source(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)

    page = client.list_tasks(TaskQuery(backend_id="east", page_size=17))

    task = page.items[0]
    assert task.identity.key.resource_id == "/alice/train/7"
    assert task.current_attempt is not None
    assert task.current_attempt.attempt_uid == "attempt-uid-2"
    assert page.source_statuses[0].error_code == "backend_unavailable"
    request = rpc.requests[0]
    assert request.query.backend_id == "east"
    assert request.query.page.page_size == 17


def test_batch_describe_tasks_preserves_requested_order(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    keys = (
        ResourceKey("prod", ResourceKind.TASK, "/alice/train/2"),
        ResourceKey("prod", ResourceKind.TASK, "/alice/train/7"),
    )

    details = client.describe_tasks(keys)

    assert [detail.summary.identity.key for detail in details] == list(keys)
    assert [key.resource_id for key in rpc.requests[0].tasks] == ["/alice/train/2", "/alice/train/7"]


def test_describe_attempt_distinguishes_current_and_numbered_targets(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    task = ResourceKey("prod", ResourceKind.TASK, "/alice/train/7")

    current = client.describe_attempt(AttemptLocator(task, None))
    exact = client.describe_attempt(AttemptLocator(task, 2))

    assert current.summary.identity.attempt_number == 4
    assert exact.summary.identity.attempt_uid == "attempt-2"
    assert not rpc.requests[0].attempt.HasField("attempt_number")
    assert rpc.requests[1].attempt.attempt_number == 2


def test_list_nodes_preserves_region_in_bounded_summary(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)

    page = client.list_nodes(NodeQuery(page_size=500))

    assert page.items[0].region == "us-east5"
    assert rpc.requests[0].query.page.page_size == 500


def test_resolve_endpoints_pages_and_batches_exact_resource_details(monkeypatch) -> None:
    client, _ = _client(monkeypatch)

    endpoints = client.resolve_endpoints("/actors/coordinator")

    assert [endpoint.summary.endpoint_id for endpoint in endpoints] == ["endpoint-1", "endpoint-2"]
    assert [endpoint.address for endpoint in endpoints] == [
        "endpoint-1:8080",
        "endpoint-2:8080",
    ]
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


def test_cancel_job_preserves_exact_identity_and_decodes_terminal_receipt(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    identity = JobIdentity(ResourceKey("prod", ResourceKind.JOB, "/alice/train"), "job-uid")

    receipt = client.cancel_job(identity, idempotency_key="operator-request-9")

    assert receipt.state is ActionState.SUCCEEDED
    assert receipt.result_code is ActionResult.SATISFIED
    assert receipt.expected_target_uid == "job-uid"
    request = rpc.requests[0]
    assert request.job.job_uid == "job-uid"
    assert request.idempotency_key == "operator-request-9"


def test_exec_attempt_rpc_deadline_outlasts_requested_command_timeout(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    task = ResourceKey("prod", ResourceKind.TASK, "/alice/train/7")
    requested_timeout = Duration.from_minutes(20)

    result = client.exec_attempt(
        AttemptIdentity(task, 2, "attempt-uid"),
        command=("sleep", "600"),
        timeout=requested_timeout,
    )

    assert result.stdout == "done"
    assert rpc.requests[-1].timeout.milliseconds == requested_timeout.to_ms()
    operation, rpc_timeout_ms = rpc.rpc_timeouts[-1]
    assert operation == "exec"
    assert rpc_timeout_ms is not None and rpc_timeout_ms > requested_timeout.to_ms()


def test_profile_attempt_rpc_deadline_outlasts_requested_capture_duration(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    task = ResourceKey("prod", ResourceKind.TASK, "/alice/train/7")
    requested_duration = Duration.from_minutes(20)

    result = client.profile_attempt(
        AttemptIdentity(task, 2, "attempt-uid"),
        profile=ThreadsProfileConfiguration(include_locals=False),
        duration=requested_duration,
    )

    assert result.profile_data == b"profile"
    assert rpc.requests[-1].duration.milliseconds == requested_duration.to_ms()
    operation, rpc_timeout_ms = rpc.rpc_timeouts[-1]
    assert operation == "profile"
    assert rpc_timeout_ms is not None and rpc_timeout_ms > requested_duration.to_ms()
