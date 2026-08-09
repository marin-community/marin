# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed protocols for first-party Iris clients."""

from collections.abc import Iterator, Sequence
from typing import Protocol

from rigging.timing import Duration

from iris.cluster.resources.action import ActionReceipt
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
from iris.cluster.types import EndpointAccess, JobName, TaskAttempt
from iris.rpc import iris_logging_pb2, job_pb2


class ResourceClientProtocol(Protocol):
    """The public resource API shared by local and remote clients."""

    def submit_job(self, spec: JobSpec, *, bundle: bytes | None = None) -> JobIdentity: ...

    def list_jobs(self, query: JobQuery = JobQuery()) -> Page[JobSummary]: ...

    def describe_job(self, key: ResourceKey) -> JobDetail: ...

    def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]: ...

    def describe_task(self, key: ResourceKey) -> TaskDetail: ...

    def describe_tasks(self, keys: Sequence[ResourceKey]) -> tuple[TaskDetail, ...]: ...

    def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail: ...

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]: ...

    def describe_node(self, locator: NodeLocator) -> NodeDetail: ...

    def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]: ...

    def describe_slice(self, locator: SliceLocator) -> SliceDetail: ...

    def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]: ...

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail: ...

    def describe_endpoints(self, keys: Sequence[ResourceKey]) -> tuple[EndpointDetail, ...]: ...

    def resolve_endpoints(self, name: str) -> tuple[EndpointDetail, ...]: ...

    def mint_endpoint_token(self, key: ResourceKey, *, ttl: Duration) -> EndpointToken: ...

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]: ...

    def fetch_job_logs(self, identity: JobIdentity, query: LogQuery = LogQuery()) -> LogPage: ...

    def fetch_task_logs(self, identity: TaskIdentity, query: LogQuery = LogQuery()) -> LogPage: ...

    def fetch_attempt_logs(self, identity: AttemptIdentity, query: LogQuery = LogQuery()) -> LogPage: ...

    def stream_job_logs(
        self,
        identity: JobIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]: ...

    def stream_task_logs(
        self,
        identity: TaskIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]: ...

    def stream_attempt_logs(
        self,
        identity: AttemptIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]: ...

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt: ...

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
    ) -> ActionReceipt: ...

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt: ...

    def get_action_receipt(self, action_id: str) -> ActionReceipt: ...

    def wait_for_action(self, action_id: str, *, timeout: Duration) -> ActionReceipt: ...

    def exec_attempt(
        self,
        identity: AttemptIdentity,
        *,
        command: Sequence[str],
        timeout: Duration,
    ) -> ExecResult: ...

    def profile_attempt(
        self,
        identity: AttemptIdentity,
        *,
        profile: job_pb2.ProfileType,
        duration: Duration,
    ) -> ProfileResult: ...


class ClusterClient(ResourceClientProtocol, Protocol):
    """Resource client plus task-runtime transports used by ``IrisClient``."""

    def register_endpoint(
        self,
        name: str,
        address: str,
        task_attempt: TaskAttempt,
        metadata: dict[str, str] | None = None,
        access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    ) -> str: ...

    def unregister_endpoint(self, endpoint_id: str) -> None: ...

    def resolve_endpoint(self, endpoint_name: str) -> str: ...

    def report_task_status_text(
        self,
        task_id: JobName,
        attempt_id: int,
        detail_md: str,
        summary_md: str,
    ) -> None: ...

    def shutdown(self, wait: bool = True) -> None: ...
