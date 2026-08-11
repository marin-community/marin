# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""First-party typed client for the Iris resource service."""

import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from typing import TypeVar

from connectrpc.interceptor import InterceptorSync
from google.protobuf import any_pb2
from google.protobuf.message import Message
from rigging.timing import Deadline, Duration, ExponentialBackoff

from iris.resources.action import ActionReceipt, ActionState
from iris.resources.activity import ActivityEntry, ActivityQuery
from iris.resources.attempt import AttemptDetail
from iris.resources.endpoint import (
    EndpointDetail,
    EndpointQuery,
    EndpointSummary,
    EndpointToken,
    ExecResult,
    ProfileConfiguration,
    ProfileResult,
)
from iris.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    NodeLocator,
    ResourceKey,
    SliceLocator,
    TaskIdentity,
)
from iris.resources.job import JobDetail, JobQuery, JobSpec, JobSummary
from iris.resources.log import LogEntry, LogPage, LogQuery
from iris.resources.node import NodeDetail, NodeQuery, NodeSummary
from iris.resources.slice import SliceDetail, SliceQuery, SliceSummary
from iris.resources.source import Page
from iris.resources.state import JobState
from iris.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.rpc import iris_logging_pb2, resource_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.errors import call_with_retry
from iris.rpc.resource_client_codec import (
    activity_page_from_proto,
    activity_query_to_proto,
    attempt_detail_from_proto,
    endpoint_detail_from_proto,
    endpoint_details_from_proto,
    endpoint_page_from_proto,
    endpoint_query_to_proto,
    endpoint_token_from_proto,
    exec_result_from_proto,
    job_detail_from_proto,
    job_page_from_proto,
    job_query_to_proto,
    log_page_from_proto,
    log_query_to_proto,
    node_detail_from_proto,
    node_page_from_proto,
    node_query_to_proto,
    profile_result_from_proto,
    slice_detail_from_proto,
    slice_page_from_proto,
    slice_query_to_proto,
    task_detail_from_proto,
    task_details_from_proto,
    task_page_from_proto,
    task_query_to_proto,
)
from iris.rpc.resource_codec import (
    action_receipt_from_proto,
    attempt_identity_to_proto,
    job_identity_from_proto,
    job_identity_to_proto,
    job_spec_to_proto,
    profile_configuration_to_proto,
    resource_key_to_proto,
    task_identity_to_proto,
)
from iris.rpc.resource_connect import ResourceServiceClientSync
from iris.rpc.resource_types import (
    ACTIVITY_ENTRY,
    ATTEMPT,
    ENDPOINT,
    ENDPOINT_CAPABILITY,
    EXEC_SESSION,
    JOB,
    LOG_ENTRY,
    NODE,
    OPERATION,
    PROFILE_CAPTURE,
    SLICE,
    TASK,
)
from iris.time_proto import duration_to_proto

_ACTION_POLL_INITIAL = 0.1
_ACTION_POLL_MAXIMUM = 2.0
_LONG_RUNNING_RPC_MARGIN_MS = 60_000
_SUBMIT_JOB_TIMEOUT_MS = 180_000
_ENDPOINT_PAGE_SIZE = 500

_MessageT = TypeVar("_MessageT", bound=Message)


def _pack(value: Message) -> any_pb2.Any:
    result = any_pb2.Any()
    result.Pack(value)
    return result


def _unpack(value: any_pb2.Any, message_type: type[_MessageT]) -> _MessageT:
    if not value.Is(message_type.DESCRIPTOR):
        raise ValueError(f"expected type.googleapis.com/{message_type.DESCRIPTOR.full_name}, got {value.type_url!r}")
    result = message_type()
    if not value.Unpack(result):
        raise ValueError(f"invalid {message_type.DESCRIPTOR.full_name} body")
    return result


def _ref(
    cluster_id: str,
    resource_type: str,
    resource_id: str,
    uid: str | None = None,
) -> resource_pb2.ResourceRef:
    result = resource_pb2.ResourceRef(
        authority_cluster_id=cluster_id,
        type=resource_type,
        id=resource_id,
    )
    if uid is not None:
        result.uid = uid
    return result


def _backend_ref_id(backend_id: str, resource_id: str) -> str:
    return f"{backend_id}:{resource_id}"


def _mutation(reason: str = "") -> resource_pb2.MutationMetadata:
    return resource_pb2.MutationMetadata(request_id=uuid.uuid4().hex, reason=reason)


class ResourceRpcClient:
    """Typed synchronous client for public Iris resources."""

    def __init__(
        self,
        controller_address: str,
        *,
        timeout_ms: int = 30_000,
        interceptors: Iterable[InterceptorSync] = (),
    ) -> None:
        self._client = ResourceServiceClientSync(
            address=controller_address,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=None,
        )

    def __enter__(self) -> "ResourceRpcClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def create_resource(
        self,
        request: resource_pb2.CreateResourceRequest,
        *,
        timeout_ms: int | None = None,
    ) -> resource_pb2.Operation:
        """Create a registered resource through the generic transport."""
        return self._client.create_resource(request, timeout_ms=timeout_ms)

    def get_resource(self, request: resource_pb2.GetResourceRequest) -> resource_pb2.GetResourceResponse:
        """Read one current or exact registered resource."""
        return call_with_retry("get_resource", lambda: self._client.get_resource(request))

    def batch_get_resources(
        self,
        request: resource_pb2.BatchGetResourcesRequest,
    ) -> resource_pb2.BatchGetResourcesResponse:
        """Read an ordered homogeneous batch from one registration."""
        return call_with_retry("batch_get_resources", lambda: self._client.batch_get_resources(request))

    def list_resources(self, request: resource_pb2.ListResourcesRequest) -> resource_pb2.ListResourcesResponse:
        """List one registered resource type with its registered query body."""
        return call_with_retry("list_resources", lambda: self._client.list_resources(request))

    def update_resource(self, request: resource_pb2.UpdateResourceRequest) -> resource_pb2.Operation:
        """Apply an idempotent state update to a current or exact resource."""
        return call_with_retry("update_resource", lambda: self._client.update_resource(request))

    def delete_resource(self, request: resource_pb2.DeleteResourceRequest) -> resource_pb2.Operation:
        """Delete a resource type whose registration supports deletion."""
        return call_with_retry("delete_resource", lambda: self._client.delete_resource(request))

    def get_service_info(self) -> resource_pb2.GetServiceInfoResponse:
        """Return the resource and backend capabilities installed at this endpoint."""
        return call_with_retry(
            "get_service_info",
            lambda: self._client.get_service_info(resource_pb2.GetServiceInfoRequest()),
        )

    def submit_job(self, spec: JobSpec, *, bundle: bytes | None = None) -> JobIdentity:
        body = resource_pb2.SubmitJobRequest(spec=job_spec_to_proto(spec), bundle_blob=bundle or b"")
        operation = self._client.create_resource(
            resource_pb2.CreateResourceRequest(
                mutation=_mutation(),
                type=JOB,
                id=spec.name,
                body=_pack(body),
            ),
            timeout_ms=_SUBMIT_JOB_TIMEOUT_MS,
        )
        return job_identity_from_proto(_unpack(operation.result, resource_pb2.SubmitJobResponse).job)

    def list_jobs(self, query: JobQuery = JobQuery()) -> Page[JobSummary]:
        request = resource_pb2.ListResourcesRequest(
            type=JOB,
            query=_pack(job_query_to_proto(query)),
            view=resource_pb2.RESOURCE_VIEW_BASIC,
        )
        response = call_with_retry("list_resources", lambda: self._client.list_resources(request))
        return job_page_from_proto(
            resource_pb2.ListJobsResponse(
                jobs=[_unpack(item.body, resource_pb2.JobSummary) for item in response.resources],
                page=resource_pb2.PageInfo(
                    next_page_token=response.next_page_token,
                    source_statuses=response.source_statuses,
                ),
            )
        )

    def describe_job(self, key: ResourceKey) -> JobDetail:
        request = resource_pb2.GetResourceRequest(
            ref=_ref(key.cluster_id, JOB, key.resource_id),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("get_resource", lambda: self._client.get_resource(request))
        return job_detail_from_proto(_unpack(response.resource.body, resource_pb2.JobDetail))

    def job_state(self, identity: JobIdentity) -> JobState:
        request = resource_pb2.GetResourceRequest(
            ref=_ref(identity.key.cluster_id, JOB, identity.key.resource_id, identity.job_uid),
            view=resource_pb2.RESOURCE_VIEW_BASIC,
        )
        response = call_with_retry("get_resource", lambda: self._client.get_resource(request))
        return JobState(_unpack(response.resource.body, resource_pb2.JobSummary).state)

    def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]:
        request = resource_pb2.ListResourcesRequest(
            type=TASK,
            query=_pack(task_query_to_proto(query)),
            view=resource_pb2.RESOURCE_VIEW_BASIC,
        )
        response = call_with_retry("list_resources", lambda: self._client.list_resources(request))
        return task_page_from_proto(
            resource_pb2.ListTasksResponse(
                tasks=[_unpack(item.body, resource_pb2.TaskSummary) for item in response.resources],
                page=resource_pb2.PageInfo(
                    next_page_token=response.next_page_token,
                    source_statuses=response.source_statuses,
                ),
            )
        )

    def describe_task(self, key: ResourceKey) -> TaskDetail:
        request = resource_pb2.GetResourceRequest(
            ref=_ref(key.cluster_id, TASK, key.resource_id),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("get_resource", lambda: self._client.get_resource(request))
        return task_detail_from_proto(_unpack(response.resource.body, resource_pb2.TaskDetail))

    def describe_tasks(self, keys: Sequence[ResourceKey]) -> tuple[TaskDetail, ...]:
        request = resource_pb2.BatchGetResourcesRequest(
            type=TASK,
            refs=[_ref(key.cluster_id, TASK, key.resource_id) for key in keys],
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("batch_get_resources", lambda: self._client.batch_get_resources(request))
        details = []
        for result in response.results:
            if result.WhichOneof("result") == "error":
                raise RuntimeError(result.error.message)
            details.append(_unpack(result.resource.body, resource_pb2.TaskDetail))
        return task_details_from_proto(resource_pb2.BatchDescribeTasksResponse(tasks=details))

    def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail:
        attempt = "current" if locator.attempt_number is None else str(locator.attempt_number)
        request = resource_pb2.GetResourceRequest(
            ref=_ref(locator.task.cluster_id, ATTEMPT, f"{locator.task.resource_id}:{attempt}"),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("get_resource", lambda: self._client.get_resource(request))
        return attempt_detail_from_proto(_unpack(response.resource.body, resource_pb2.AttemptDetail))

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]:
        request = resource_pb2.ListResourcesRequest(
            type=NODE,
            query=_pack(node_query_to_proto(query)),
            view=resource_pb2.RESOURCE_VIEW_BASIC,
        )
        response = call_with_retry("list_resources", lambda: self._client.list_resources(request))
        return node_page_from_proto(
            resource_pb2.ListNodesResponse(
                nodes=[_unpack(item.body, resource_pb2.NodeSummary) for item in response.resources],
                page=resource_pb2.PageInfo(
                    next_page_token=response.next_page_token,
                    source_statuses=response.source_statuses,
                ),
            )
        )

    def describe_node(self, locator: NodeLocator) -> NodeDetail:
        request = resource_pb2.GetResourceRequest(
            ref=_ref(
                locator.key.cluster_id,
                NODE,
                _backend_ref_id(locator.backend_id, locator.key.resource_id),
                locator.node_uid,
            ),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("get_resource", lambda: self._client.get_resource(request))
        return node_detail_from_proto(_unpack(response.resource.body, resource_pb2.NodeDetail))

    def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]:
        request = resource_pb2.ListResourcesRequest(
            type=SLICE,
            query=_pack(slice_query_to_proto(query)),
            view=resource_pb2.RESOURCE_VIEW_BASIC,
        )
        response = call_with_retry("list_resources", lambda: self._client.list_resources(request))
        return slice_page_from_proto(
            resource_pb2.ListSlicesResponse(
                slices=[_unpack(item.body, resource_pb2.SliceSummary) for item in response.resources],
                page=resource_pb2.PageInfo(
                    next_page_token=response.next_page_token,
                    source_statuses=response.source_statuses,
                ),
            )
        )

    def describe_slice(self, locator: SliceLocator) -> SliceDetail:
        request = resource_pb2.GetResourceRequest(
            ref=_ref(
                locator.key.cluster_id,
                SLICE,
                _backend_ref_id(locator.backend_id, locator.key.resource_id),
                locator.slice_uid,
            ),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("get_resource", lambda: self._client.get_resource(request))
        return slice_detail_from_proto(_unpack(response.resource.body, resource_pb2.SliceDetail))

    def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]:
        request = resource_pb2.ListResourcesRequest(
            type=ENDPOINT,
            query=_pack(endpoint_query_to_proto(query)),
            view=resource_pb2.RESOURCE_VIEW_BASIC,
        )
        response = call_with_retry("list_resources", lambda: self._client.list_resources(request))
        return endpoint_page_from_proto(
            resource_pb2.ListEndpointsResponse(
                endpoints=[_unpack(item.body, resource_pb2.EndpointSummary) for item in response.resources],
                page=resource_pb2.PageInfo(
                    next_page_token=response.next_page_token,
                    source_statuses=response.source_statuses,
                ),
            )
        )

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail:
        request = resource_pb2.GetResourceRequest(
            ref=_ref(key.cluster_id, ENDPOINT, key.resource_id),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("get_resource", lambda: self._client.get_resource(request))
        return endpoint_detail_from_proto(_unpack(response.resource.body, resource_pb2.EndpointDetail))

    def describe_endpoints(self, keys: Sequence[ResourceKey]) -> tuple[EndpointDetail, ...]:
        request = resource_pb2.BatchGetResourcesRequest(
            type=ENDPOINT,
            refs=[_ref(key.cluster_id, ENDPOINT, key.resource_id) for key in keys],
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("batch_get_resources", lambda: self._client.batch_get_resources(request))
        details = []
        for result in response.results:
            if result.WhichOneof("result") == "error":
                raise RuntimeError(result.error.message)
            details.append(_unpack(result.resource.body, resource_pb2.EndpointDetail))
        return endpoint_details_from_proto(resource_pb2.BatchDescribeEndpointsResponse(endpoints=details))

    def resolve_endpoints(self, name: str) -> tuple[EndpointDetail, ...]:
        """Return every endpoint with the exact resource name."""
        details: list[EndpointDetail] = []
        page_token: str | None = None
        while True:
            page = self.list_endpoints(
                EndpointQuery(
                    name_prefix=name,
                    page_size=_ENDPOINT_PAGE_SIZE,
                    page_token=page_token,
                )
            )
            keys = tuple(endpoint.key for endpoint in page.items if endpoint.name == name)
            if keys:
                details.extend(self.describe_endpoints(keys))
            page_token = page.next_page_token
            if page_token is None:
                return tuple(details)

    def mint_endpoint_token(self, key: ResourceKey, *, ttl: Duration) -> EndpointToken:
        body = resource_pb2.MintEndpointTokenRequest(
            endpoint=resource_key_to_proto(key),
            ttl=duration_to_proto(ttl),
        )
        operation = call_with_retry(
            "create_resource",
            lambda: self._client.create_resource(
                resource_pb2.CreateResourceRequest(
                    mutation=_mutation(),
                    type=ENDPOINT_CAPABILITY,
                    parent=_ref(key.cluster_id, ENDPOINT, key.resource_id),
                    body=_pack(body),
                )
            ),
        )
        return endpoint_token_from_proto(_unpack(operation.result, resource_pb2.MintEndpointTokenResponse))

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]:
        request = resource_pb2.ListResourcesRequest(
            type=ACTIVITY_ENTRY,
            query=_pack(activity_query_to_proto(query)),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("list_resources", lambda: self._client.list_resources(request))
        return activity_page_from_proto(
            resource_pb2.ListActivityResponse(
                entries=[_unpack(item.body, resource_pb2.ActivityEntry) for item in response.resources],
                page=resource_pb2.PageInfo(
                    next_page_token=response.next_page_token,
                    source_statuses=response.source_statuses,
                ),
            )
        )

    def fetch_job_logs(self, identity: JobIdentity, query: LogQuery = LogQuery()) -> LogPage:
        target = resource_pb2.LogTarget(job=job_identity_to_proto(identity))
        return self._fetch_logs(target, query)

    def fetch_task_logs(self, identity: TaskIdentity, query: LogQuery = LogQuery()) -> LogPage:
        target = resource_pb2.LogTarget(task=task_identity_to_proto(identity))
        return self._fetch_logs(target, query)

    def fetch_attempt_logs(self, identity: AttemptIdentity, query: LogQuery = LogQuery()) -> LogPage:
        target = resource_pb2.LogTarget(attempt=attempt_identity_to_proto(identity))
        return self._fetch_logs(target, query)

    def _fetch_logs(self, target: resource_pb2.LogTarget, query: LogQuery) -> LogPage:
        body = resource_pb2.FetchLogsRequest(target=target, query=log_query_to_proto(query))
        request = resource_pb2.ListResourcesRequest(
            type=LOG_ENTRY,
            query=_pack(body),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("list_resources", lambda: self._client.list_resources(request))
        return log_page_from_proto(
            resource_pb2.FetchLogsResponse(
                entries=[_unpack(item.body, iris_logging_pb2.LogEntry) for item in response.resources],
                next_cursor=int(response.next_page_token or 0),
                source_statuses=response.source_statuses,
            )
        )

    def stream_job_logs(
        self,
        identity: JobIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[LogEntry]:
        return self._stream_logs(lambda current: self.fetch_job_logs(identity, current), query)

    def stream_task_logs(
        self,
        identity: TaskIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[LogEntry]:
        return self._stream_logs(lambda current: self.fetch_task_logs(identity, current), query)

    def stream_attempt_logs(
        self,
        identity: AttemptIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[LogEntry]:
        return self._stream_logs(lambda current: self.fetch_attempt_logs(identity, current), query)

    def _stream_logs(self, fetch, query: LogQuery) -> Iterator[LogEntry]:
        current = query
        while True:
            page = fetch(current)
            yield from page.entries
            if not query.tail:
                return
            if page.next_cursor > current.cursor:
                current = replace(current, cursor=page.next_cursor)
                continue
            time.sleep(_ACTION_POLL_INITIAL)

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt:
        request = resource_pb2.UpdateResourceRequest(
            mutation=resource_pb2.MutationMetadata(request_id=idempotency_key),
            ref=_ref(identity.key.cluster_id, JOB, identity.key.resource_id, identity.job_uid),
            update=resource_pb2.ResourceUpdate(new_state=resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED),
        )
        operation = call_with_retry("update_resource", lambda: self._client.update_resource(request))
        return action_receipt_from_proto(_unpack(operation.result, resource_pb2.ActionReceipt))

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
    ) -> ActionReceipt:
        condition = resource_pb2.RetryTaskRequest(
            task=task_identity_to_proto(identity), expected_attempt_uid=expected_attempt_uid
        )
        request = resource_pb2.UpdateResourceRequest(
            mutation=resource_pb2.MutationMetadata(request_id=idempotency_key),
            ref=_ref(identity.key.cluster_id, TASK, identity.key.resource_id, identity.task_uid),
            update=resource_pb2.ResourceUpdate(
                new_state=resource_pb2.REQUESTED_RESOURCE_STATE_PENDING,
                patch=_pack(condition),
            ),
        )
        operation = call_with_retry("update_resource", lambda: self._client.update_resource(request))
        return action_receipt_from_proto(_unpack(operation.result, resource_pb2.ActionReceipt))

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt:
        request = resource_pb2.UpdateResourceRequest(
            mutation=resource_pb2.MutationMetadata(request_id=idempotency_key),
            ref=_ref(
                identity.task.cluster_id,
                ATTEMPT,
                f"{identity.task.resource_id}:{identity.attempt_number}",
                identity.attempt_uid,
            ),
            update=resource_pb2.ResourceUpdate(new_state=resource_pb2.REQUESTED_RESOURCE_STATE_CANCELLED),
        )
        operation = call_with_retry("update_resource", lambda: self._client.update_resource(request))
        return action_receipt_from_proto(_unpack(operation.result, resource_pb2.ActionReceipt))

    def get_action_receipt(self, action_id: str) -> ActionReceipt:
        request = resource_pb2.GetResourceRequest(
            ref=_ref("system", OPERATION, action_id),
            view=resource_pb2.RESOURCE_VIEW_FULL,
        )
        response = call_with_retry("get_resource", lambda: self._client.get_resource(request))
        operation = _unpack(response.resource.body, resource_pb2.Operation)
        return action_receipt_from_proto(_unpack(operation.result, resource_pb2.ActionReceipt))

    def wait_for_action(self, action_id: str, *, timeout: Duration) -> ActionReceipt:
        deadline = Deadline.from_seconds(timeout.to_seconds())
        backoff = ExponentialBackoff(initial=_ACTION_POLL_INITIAL, maximum=_ACTION_POLL_MAXIMUM)
        while True:
            receipt = self.get_action_receipt(action_id)
            if receipt.state in {ActionState.SUCCEEDED, ActionState.FAILED}:
                return receipt
            deadline.raise_if_expired(f"Action {action_id} did not complete")
            time.sleep(min(backoff.next_interval(), deadline.remaining_seconds()))

    def exec_attempt(
        self,
        identity: AttemptIdentity,
        *,
        command: Sequence[str],
        timeout: Duration,
    ) -> ExecResult:
        body = resource_pb2.ExecAttemptRequest(
            attempt=attempt_identity_to_proto(identity),
            command=command,
            timeout=duration_to_proto(timeout),
        )
        rpc_timeout_ms = timeout.to_ms() + _LONG_RUNNING_RPC_MARGIN_MS
        operation = self._client.create_resource(
            resource_pb2.CreateResourceRequest(
                mutation=_mutation(),
                type=EXEC_SESSION,
                parent=_ref(
                    identity.task.cluster_id,
                    ATTEMPT,
                    f"{identity.task.resource_id}:{identity.attempt_number}",
                    identity.attempt_uid,
                ),
                body=_pack(body),
            ),
            timeout_ms=rpc_timeout_ms,
        )
        return exec_result_from_proto(_unpack(operation.result, resource_pb2.ExecAttemptResponse))

    def profile_attempt(
        self,
        identity: AttemptIdentity,
        *,
        profile: ProfileConfiguration,
        duration: Duration,
    ) -> ProfileResult:
        body = resource_pb2.ProfileAttemptRequest(
            attempt=attempt_identity_to_proto(identity),
            profile=profile_configuration_to_proto(profile),
            duration=duration_to_proto(duration),
        )
        rpc_timeout_ms = duration.to_ms() + _LONG_RUNNING_RPC_MARGIN_MS
        operation = self._client.create_resource(
            resource_pb2.CreateResourceRequest(
                mutation=_mutation(),
                type=PROFILE_CAPTURE,
                parent=_ref(
                    identity.task.cluster_id,
                    ATTEMPT,
                    f"{identity.task.resource_id}:{identity.attempt_number}",
                    identity.attempt_uid,
                ),
                body=_pack(body),
            ),
            timeout_ms=rpc_timeout_ms,
        )
        return profile_result_from_proto(_unpack(operation.result, resource_pb2.ProfileAttemptResponse))
