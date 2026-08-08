# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.client.resource_client import ResourceClient
from iris.cluster.resources.action import ActionResult, ActionState
from iris.cluster.resources.identity import AttemptLocator, JobIdentity, ResourceKey, ResourceKind
from iris.cluster.resources.task import TaskQuery
from iris.rpc import job_pb2, resource_pb2, time_pb2


def _key(kind: int, resource_id: str) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(cluster_id="prod", kind=kind, resource_id=resource_id)


class _ResourceRpc:
    def __init__(self, *_args, **_kwargs) -> None:
        self.requests = []

    def close(self) -> None:
        pass

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


def _client(monkeypatch) -> tuple[ResourceClient, _ResourceRpc]:
    rpc = _ResourceRpc()
    monkeypatch.setattr("iris.cluster.client.resource_client.ResourceServiceClientSync", lambda *_a, **_kw: rpc)
    return ResourceClient("http://controller.test"), rpc


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


def test_describe_attempt_distinguishes_current_and_numbered_targets(monkeypatch) -> None:
    client, rpc = _client(monkeypatch)
    task = ResourceKey("prod", ResourceKind.TASK, "/alice/train/7")

    current = client.describe_attempt(AttemptLocator(task, None))
    exact = client.describe_attempt(AttemptLocator(task, 2))

    assert current.summary.identity.attempt_number == 4
    assert exact.summary.identity.attempt_uid == "attempt-2"
    assert not rpc.requests[0].attempt.HasField("attempt_number")
    assert rpc.requests[1].attempt.attempt_number == 2


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
