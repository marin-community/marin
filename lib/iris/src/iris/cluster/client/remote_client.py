# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote resource client plus task-runtime transports."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from connectrpc.interceptor import InterceptorSync
from finelog.client import LogClient
from rigging.connect import proxy_path
from rigging.timing import Duration

from iris.cluster.client.bundle import create_workspace_zip
from iris.cluster.client.endpoint_client import EndpointClient, EndpointInstance
from iris.cluster.client.resource_client import ResourceClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
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
from iris.cluster.stats.tables import TASK_STATUS_NAMESPACE, TASK_STATUS_STORAGE_POLICY, TaskStatusRow
from iris.cluster.types import EndpointAccess, JobName, TaskAttempt
from iris.rpc import iris_logging_pb2, job_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import EndpointServiceClientSync


class RemoteClusterClient:
    """Connect the typed resource API and private runtime transports."""

    def __init__(
        self,
        controller_address: str,
        bundle_id: str | None = None,
        workspace: Path | None = None,
        timeout_ms: int = 30_000,
        interceptors: Iterable[InterceptorSync] = (),
        use_controller_proxy: bool = True,
        extra_bundle_includes: Sequence[str] = (),
    ) -> None:
        self._address = controller_address
        self._bundle_id = bundle_id
        self._workspace = workspace.resolve() if workspace is not None else None
        self._extra_bundle_includes = extra_bundle_includes
        self._bundle_blob: bytes | None = None
        self._use_controller_proxy = use_controller_proxy
        self._resources = ResourceClient(
            controller_address,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
        )
        self._endpoint_client = EndpointClient(
            EndpointServiceClientSync(
                address=controller_address,
                timeout_ms=timeout_ms,
                interceptors=interceptors,
                accept_compression=IRIS_RPC_COMPRESSIONS,
                send_compression=None,
            )
        )
        self._log_client = LogClient.connect(
            LOG_SERVER_ENDPOINT_NAME,
            resolver=self.resolve_endpoint,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
        )
        self._task_status_table = None

    def submit_job(self, spec: JobSpec, *, bundle: bytes | None = None) -> JobIdentity:
        if not spec.bundle_id and self._bundle_id:
            spec = replace(spec, bundle_id=self._bundle_id)
        if bundle is None and not spec.bundle_id and self._workspace is not None:
            if self._bundle_blob is None:
                self._bundle_blob = create_workspace_zip(
                    self._workspace,
                    extra_includes=self._extra_bundle_includes,
                )
            bundle = self._bundle_blob
        return self._resources.submit_job(spec, bundle=bundle)

    def list_jobs(self, query: JobQuery = JobQuery()) -> Page[JobSummary]:
        return self._resources.list_jobs(query)

    def describe_job(self, key: ResourceKey) -> JobDetail:
        return self._resources.describe_job(key)

    def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]:
        return self._resources.list_tasks(query)

    def describe_task(self, key: ResourceKey) -> TaskDetail:
        return self._resources.describe_task(key)

    def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail:
        return self._resources.describe_attempt(locator)

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]:
        return self._resources.list_nodes(query)

    def describe_node(self, locator: NodeLocator) -> NodeDetail:
        return self._resources.describe_node(locator)

    def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]:
        return self._resources.list_slices(query)

    def describe_slice(self, locator: SliceLocator) -> SliceDetail:
        return self._resources.describe_slice(locator)

    def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]:
        return self._resources.list_endpoints(query)

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail:
        return self._resources.describe_endpoint(key)

    def mint_endpoint_token(self, key: ResourceKey, *, ttl: Duration) -> EndpointToken:
        return self._resources.mint_endpoint_token(key, ttl=ttl)

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]:
        return self._resources.list_activity(query)

    def fetch_job_logs(self, identity: JobIdentity, query: LogQuery = LogQuery()) -> LogPage:
        return self._resources.fetch_job_logs(identity, query)

    def fetch_task_logs(self, identity: TaskIdentity, query: LogQuery = LogQuery()) -> LogPage:
        return self._resources.fetch_task_logs(identity, query)

    def fetch_attempt_logs(self, identity: AttemptIdentity, query: LogQuery = LogQuery()) -> LogPage:
        return self._resources.fetch_attempt_logs(identity, query)

    def stream_job_logs(
        self,
        identity: JobIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]:
        return self._resources.stream_job_logs(identity, query)

    def stream_task_logs(
        self,
        identity: TaskIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]:
        return self._resources.stream_task_logs(identity, query)

    def stream_attempt_logs(
        self,
        identity: AttemptIdentity,
        query: LogQuery = LogQuery(),
    ) -> Iterator[iris_logging_pb2.LogEntry]:
        return self._resources.stream_attempt_logs(identity, query)

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt:
        return self._resources.cancel_job(identity, idempotency_key=idempotency_key)

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
    ) -> ActionReceipt:
        return self._resources.retry_task(
            identity,
            expected_attempt_uid=expected_attempt_uid,
            idempotency_key=idempotency_key,
        )

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt:
        return self._resources.terminate_attempt(identity, idempotency_key=idempotency_key)

    def get_action_receipt(self, action_id: str) -> ActionReceipt:
        return self._resources.get_action_receipt(action_id)

    def wait_for_action(self, action_id: str, *, timeout: Duration) -> ActionReceipt:
        return self._resources.wait_for_action(action_id, timeout=timeout)

    def exec_attempt(
        self,
        locator: AttemptLocator,
        *,
        command: Sequence[str],
        timeout: Duration,
    ) -> ExecResult:
        return self._resources.exec_attempt(locator, command=command, timeout=timeout)

    def profile_attempt(
        self,
        locator: AttemptLocator,
        *,
        profile: job_pb2.ProfileType,
        duration: Duration,
    ) -> ProfileResult:
        return self._resources.profile_attempt(locator, profile=profile, duration=duration)

    def register_endpoint(
        self,
        name: str,
        address: str,
        task_attempt: TaskAttempt,
        metadata: dict[str, str] | None = None,
        access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
    ) -> str:
        return self._endpoint_client.register(name, address, task_attempt, metadata, access)

    def unregister_endpoint(self, endpoint_id: str) -> None:
        self._endpoint_client.unregister(endpoint_id)

    def list_endpoint_instances(self, name: str) -> list[EndpointInstance]:
        """Return leased endpoint instances for in-cluster actor resolution."""
        return self._endpoint_client.list_endpoint_instances(name)

    def resolve_endpoint(self, endpoint_name: str) -> str:
        if self._use_controller_proxy:
            return f"{self._address.rstrip('/')}{proxy_path(endpoint_name)}"
        endpoints = self._endpoint_client.list_endpoint_instances(endpoint_name)
        if not endpoints:
            raise ConnectionError(f"No {endpoint_name!r} endpoint registered on controller")
        return endpoints[0].address

    def report_task_status_text(
        self,
        task_id: JobName,
        attempt_id: int,
        detail_md: str,
        summary_md: str,
    ) -> None:
        """Write task-owned status text to finelog."""
        if self._task_status_table is None:
            self._task_status_table = self._log_client.get_table(
                TASK_STATUS_NAMESPACE,
                TaskStatusRow,
                storage_policy=TASK_STATUS_STORAGE_POLICY,
            )
        self._task_status_table.write(
            [
                TaskStatusRow(
                    ts=datetime.now(UTC).replace(tzinfo=None),
                    task_id=task_id.to_wire(),
                    attempt_id=attempt_id,
                    status_text_detail_md=detail_md,
                    status_text_summary_md=summary_md,
                )
            ]
        )

    def shutdown(self, wait: bool = True) -> None:
        del wait
        self._endpoint_client.close()
        self._log_client.close()
        self._resources.close()
