# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""First-party typed client for the Iris resource service."""

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace

from connectrpc.interceptor import InterceptorSync
from rigging.timing import Deadline, Duration, ExponentialBackoff

from iris.cluster.client.resource_conversion import (
    action_receipt_from_proto,
    activity_page_from_proto,
    activity_query_to_proto,
    attempt_detail_from_proto,
    attempt_identity_to_proto,
    attempt_locator_to_proto,
    endpoint_detail_from_proto,
    endpoint_details_from_proto,
    endpoint_page_from_proto,
    endpoint_query_to_proto,
    endpoint_token_from_proto,
    exec_result_from_proto,
    job_detail_from_proto,
    job_identity_from_proto,
    job_identity_to_proto,
    job_page_from_proto,
    job_query_to_proto,
    job_spec_to_proto,
    log_page_from_proto,
    log_query_to_proto,
    node_detail_from_proto,
    node_locator_to_proto,
    node_page_from_proto,
    node_query_to_proto,
    profile_result_from_proto,
    resource_key_to_proto,
    slice_detail_from_proto,
    slice_locator_to_proto,
    slice_page_from_proto,
    slice_query_to_proto,
    task_detail_from_proto,
    task_details_from_proto,
    task_identity_to_proto,
    task_page_from_proto,
    task_query_to_proto,
)
from iris.cluster.resources.action import ActionReceipt, ActionState
from iris.cluster.resources.activity import ActivityEntry, ActivityQuery
from iris.cluster.resources.attempt import AttemptDetail
from iris.cluster.resources.endpoint import (
    EndpointDetail,
    EndpointQuery,
    EndpointSummary,
    EndpointToken,
    ExecResult,
    ProfileResult,
)
from iris.cluster.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    NodeLocator,
    ResourceKey,
    SliceLocator,
    TaskIdentity,
)
from iris.cluster.resources.job import JobDetail, JobQuery, JobSpec, JobSummary
from iris.cluster.resources.log import LogPage, LogQuery
from iris.cluster.resources.node import NodeDetail, NodeQuery, NodeSummary
from iris.cluster.resources.slice import SliceDetail, SliceQuery, SliceSummary
from iris.cluster.resources.source import Page
from iris.cluster.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.rpc import iris_logging_pb2, job_pb2, resource_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.errors import call_with_retry
from iris.rpc.resource_connect import ResourceServiceClientSync
from iris.time_proto import duration_to_proto

_ACTION_POLL_INITIAL = 0.1
_ACTION_POLL_MAXIMUM = 2.0
_LONG_RUNNING_RPC_MARGIN_MS = 60_000
_SUBMIT_JOB_TIMEOUT_MS = 180_000
_ENDPOINT_PAGE_SIZE = 500


class ResourceClient:
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

    def __enter__(self) -> "ResourceClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def submit_job(self, spec: JobSpec, *, bundle: bytes | None = None) -> JobIdentity:
        request = resource_pb2.SubmitJobRequest(spec=job_spec_to_proto(spec), bundle_blob=bundle or b"")
        response = call_with_retry(
            "submit_job",
            lambda: self._client.submit_job(request, timeout_ms=_SUBMIT_JOB_TIMEOUT_MS),
        )
        return job_identity_from_proto(response.job)

    def list_jobs(self, query: JobQuery = JobQuery()) -> Page[JobSummary]:
        request = resource_pb2.ListJobsRequest(query=job_query_to_proto(query))
        response = call_with_retry("list_jobs", lambda: self._client.list_jobs(request))
        return job_page_from_proto(response)

    def describe_job(self, key: ResourceKey) -> JobDetail:
        request = resource_pb2.DescribeJobRequest(job=resource_key_to_proto(key))
        response = call_with_retry("describe_job", lambda: self._client.describe_job(request))
        return job_detail_from_proto(response.job)

    def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]:
        request = resource_pb2.ListTasksRequest(query=task_query_to_proto(query))
        response = call_with_retry("list_tasks", lambda: self._client.list_tasks(request))
        return task_page_from_proto(response)

    def describe_task(self, key: ResourceKey) -> TaskDetail:
        request = resource_pb2.DescribeTaskRequest(task=resource_key_to_proto(key))
        response = call_with_retry("describe_task", lambda: self._client.describe_task(request))
        return task_detail_from_proto(response.task)

    def describe_tasks(self, keys: Sequence[ResourceKey]) -> tuple[TaskDetail, ...]:
        request = resource_pb2.BatchDescribeTasksRequest(tasks=[resource_key_to_proto(key) for key in keys])
        response = call_with_retry("batch_describe_tasks", lambda: self._client.batch_describe_tasks(request))
        return task_details_from_proto(response)

    def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail:
        request = resource_pb2.DescribeAttemptRequest(attempt=attempt_locator_to_proto(locator))
        response = call_with_retry("describe_attempt", lambda: self._client.describe_attempt(request))
        return attempt_detail_from_proto(response.attempt)

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]:
        request = resource_pb2.ListNodesRequest(query=node_query_to_proto(query))
        response = call_with_retry("list_nodes", lambda: self._client.list_nodes(request))
        return node_page_from_proto(response)

    def describe_node(self, locator: NodeLocator) -> NodeDetail:
        request = resource_pb2.DescribeNodeRequest(node=node_locator_to_proto(locator))
        response = call_with_retry("describe_node", lambda: self._client.describe_node(request))
        return node_detail_from_proto(response.node)

    def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]:
        request = resource_pb2.ListSlicesRequest(query=slice_query_to_proto(query))
        response = call_with_retry("list_slices", lambda: self._client.list_slices(request))
        return slice_page_from_proto(response)

    def describe_slice(self, locator: SliceLocator) -> SliceDetail:
        request = resource_pb2.DescribeSliceRequest(slice=slice_locator_to_proto(locator))
        response = call_with_retry("describe_slice", lambda: self._client.describe_slice(request))
        return slice_detail_from_proto(response.slice)

    def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]:
        request = resource_pb2.ListEndpointsRequest(query=endpoint_query_to_proto(query))
        response = call_with_retry("list_endpoints", lambda: self._client.list_endpoints(request))
        return endpoint_page_from_proto(response)

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail:
        request = resource_pb2.DescribeEndpointRequest(endpoint=resource_key_to_proto(key))
        response = call_with_retry("describe_endpoint", lambda: self._client.describe_endpoint(request))
        return endpoint_detail_from_proto(response.endpoint)

    def describe_endpoints(self, keys: Sequence[ResourceKey]) -> tuple[EndpointDetail, ...]:
        request = resource_pb2.BatchDescribeEndpointsRequest(endpoints=[resource_key_to_proto(key) for key in keys])
        response = call_with_retry(
            "batch_describe_endpoints",
            lambda: self._client.batch_describe_endpoints(request),
        )
        return endpoint_details_from_proto(response)

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
        request = resource_pb2.MintEndpointTokenRequest(
            endpoint=resource_key_to_proto(key),
            ttl=duration_to_proto(ttl),
        )
        response = call_with_retry("mint_endpoint_token", lambda: self._client.mint_endpoint_token(request))
        return endpoint_token_from_proto(response)

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]:
        request = resource_pb2.ListActivityRequest(query=activity_query_to_proto(query))
        response = call_with_retry("list_activity", lambda: self._client.list_activity(request))
        return activity_page_from_proto(response)

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
        request = resource_pb2.FetchLogsRequest(target=target, query=log_query_to_proto(query))
        response = call_with_retry("fetch_logs", lambda: self._client.fetch_logs(request))
        return log_page_from_proto(response)

    def stream_job_logs(
        self,
        identity: JobIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]:
        return self._stream_logs(lambda current: self.fetch_job_logs(identity, current), query)

    def stream_task_logs(
        self,
        identity: TaskIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]:
        return self._stream_logs(lambda current: self.fetch_task_logs(identity, current), query)

    def stream_attempt_logs(
        self,
        identity: AttemptIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]:
        return self._stream_logs(lambda current: self.fetch_attempt_logs(identity, current), query)

    def _stream_logs(self, fetch, query: LogQuery) -> Iterator[iris_logging_pb2.LogEntry]:
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
        request = resource_pb2.CancelJobRequest(
            job=job_identity_to_proto(identity),
            idempotency_key=idempotency_key,
        )
        response = call_with_retry("cancel_job", lambda: self._client.cancel_job(request))
        return action_receipt_from_proto(response.receipt)

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
    ) -> ActionReceipt:
        request = resource_pb2.RetryTaskRequest(
            task=task_identity_to_proto(identity),
            expected_attempt_uid=expected_attempt_uid,
            idempotency_key=idempotency_key,
        )
        response = call_with_retry("retry_task", lambda: self._client.retry_task(request))
        return action_receipt_from_proto(response.receipt)

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt:
        request = resource_pb2.TerminateAttemptRequest(
            attempt=attempt_identity_to_proto(identity),
            idempotency_key=idempotency_key,
        )
        response = call_with_retry("terminate_attempt", lambda: self._client.terminate_attempt(request))
        return action_receipt_from_proto(response.receipt)

    def get_action_receipt(self, action_id: str) -> ActionReceipt:
        request = resource_pb2.GetActionReceiptRequest(action_id=action_id)
        response = call_with_retry("get_action_receipt", lambda: self._client.get_action_receipt(request))
        return action_receipt_from_proto(response.receipt)

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
        request = resource_pb2.ExecAttemptRequest(
            attempt=attempt_identity_to_proto(identity),
            command=command,
            timeout=duration_to_proto(timeout),
        )
        rpc_timeout_ms = timeout.to_ms() + _LONG_RUNNING_RPC_MARGIN_MS
        response = call_with_retry(
            "exec_attempt",
            lambda: self._client.exec_attempt(request, timeout_ms=rpc_timeout_ms),
        )
        return exec_result_from_proto(response)

    def profile_attempt(
        self,
        identity: AttemptIdentity,
        *,
        profile: job_pb2.ProfileType,
        duration: Duration,
    ) -> ProfileResult:
        request = resource_pb2.ProfileAttemptRequest(
            attempt=attempt_identity_to_proto(identity),
            profile=profile,
            duration=duration_to_proto(duration),
        )
        rpc_timeout_ms = duration.to_ms() + _LONG_RUNNING_RPC_MARGIN_MS
        response = call_with_retry(
            "profile_attempt",
            lambda: self._client.profile_attempt(request, timeout_ms=rpc_timeout_ms),
        )
        return profile_result_from_proto(response)
