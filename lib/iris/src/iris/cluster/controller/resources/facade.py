# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed resource facade shared by RPC, clients, and deterministic journeys."""

import base64
import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol

from connectrpc.errors import ConnectError
from finelog.client import LogClient
from finelog.rpc import logging_pb2
from rigging.timing import Duration, Timestamp
from sqlalchemy import and_, or_, select

from iris.cluster.constraints import Constraint
from iris.cluster.controller import reads
from iris.cluster.controller.attempt_counts import AttemptCounts
from iris.cluster.controller.backend import BackendCapability, ProviderError, TaskBackend
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.ops import job as job_ops
from iris.cluster.controller.ops.task import finalize
from iris.cluster.controller.persistence import action as action_persistence
from iris.cluster.controller.projections.endpoints import (
    EndpointQuery as ProjectionEndpointQuery,
)
from iris.cluster.controller.projections.endpoints import (
    EndpointRow,
    EndpointsProjection,
)
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.schema import (
    federated_jobs_table,
    jobs_table,
    task_attempts_table,
    tasks_table,
)
from iris.cluster.federation.store import FederationDirection
from iris.cluster.log_keys import build_log_source
from iris.cluster.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.cluster.resources.activity import ActivityEntry, ActivityQuery
from iris.cluster.resources.attempt import AttemptDetail, AttemptSummary
from iris.cluster.resources.endpoint import (
    EndpointAccess,
    EndpointDetail,
    EndpointQuery,
    EndpointSummary,
    EndpointToken,
)
from iris.cluster.resources.errors import (
    ActionIdempotencyConflict,
    ActionPolicyRejected,
    BackendIdentityUnknown,
    InvalidPageToken,
    InvalidResourceKey,
    ResourceNotFound,
    ResourceReplaced,
)
from iris.cluster.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    NodeIdentity,
    NodeLocator,
    ResourceKey,
    ResourceKind,
    SliceIdentity,
    SliceLocator,
    TaskIdentity,
)
from iris.cluster.resources.job import JobDetail, JobQuery, JobSpec, JobSummary
from iris.cluster.resources.log import LogPage, LogQuery
from iris.cluster.resources.node import (
    NodeAttribute,
    NodeAttributeKind,
    NodeCapacity,
    NodeDetail,
    NodeHealth,
    NodeQuery,
    NodeSummary,
)
from iris.cluster.resources.slice import (
    MembershipState,
    SliceDetail,
    SliceLifecycle,
    SliceMember,
    SliceQuery,
    SliceSummary,
)
from iris.cluster.resources.source import (
    MAX_PROVIDER_SNAPSHOT_ITEMS,
    MAX_SOURCE_ERROR_MESSAGE,
    Freshness,
    Page,
    ResourceSourceStatus,
    SourceState,
)
from iris.cluster.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.cluster.stats.tables import TASK_EVENT_NAMESPACE
from iris.cluster.types import LOCAL_CLUSTER, CoschedulingConfig, JobName, ResourceSpec
from iris.rpc import controller_pb2, iris_logging_pb2, job_pb2
from iris.time_proto import duration_from_proto, duration_to_proto, timestamp_from_proto

_RESOURCE_UID_NAMESPACE = uuid.UUID("2c72b7f4-a156-5d27-8b58-7de28d5ec4cc")
_RESOURCE_UID_PREFIX = "iris-resource-v2"
_MAX_JOB_PAGE = 500
_MAX_TASK_PAGE = 500


class LegacyResourceReader(Protocol):
    """Existing controller reads used while their callers move to typed records."""

    def list_jobs(
        self,
        request: controller_pb2.Controller.ListJobsRequest,
        ctx: object | None,
    ) -> controller_pb2.Controller.ListJobsResponse: ...

    def get_job_status(
        self,
        request: controller_pb2.Controller.GetJobStatusRequest,
        ctx: object | None,
    ) -> controller_pb2.Controller.GetJobStatusResponse: ...

    def launch_job(
        self,
        request: controller_pb2.Controller.LaunchJobRequest,
        ctx: object | None,
    ) -> controller_pb2.Controller.LaunchJobResponse: ...

    def mint_endpoint_token(
        self,
        request: controller_pb2.Controller.MintEndpointTokenRequest,
        ctx: object | None,
    ) -> controller_pb2.Controller.MintEndpointTokenResponse: ...

    def list_workers(
        self,
        request: controller_pb2.Controller.ListWorkersRequest,
        ctx: object | None,
    ) -> controller_pb2.Controller.ListWorkersResponse: ...

    def exec_in_container(
        self,
        request: controller_pb2.Controller.ExecInContainerRequest,
        ctx: object | None,
    ) -> controller_pb2.Controller.ExecInContainerResponse: ...

    def profile_task(
        self,
        request: job_pb2.ProfileTaskRequest,
        ctx: object | None,
    ) -> job_pb2.ProfileTaskResponse: ...


def _uid(kind: ResourceKind, *parts: object) -> str:
    name = "\0".join((_RESOURCE_UID_PREFIX, kind.value, *(str(part) for part in parts)))
    return str(uuid.uuid5(_RESOURCE_UID_NAMESPACE, name))


def _resource_uid(kind: ResourceKind, cluster_id: str, resource_id: str, _submitted_at: Timestamp) -> str:
    if kind is ResourceKind.TASK:
        task_id = JobName.from_wire(resource_id)
        job_id, task_index = task_id.require_task()
        return _uid(kind, _uid(ResourceKind.JOB, cluster_id, job_id.to_wire()), task_index)
    if kind is ResourceKind.NODE:
        backend_id, _, node_id = resource_id.partition(":")
        return _uid(kind, cluster_id, backend_id, node_id)
    return _uid(kind, cluster_id, resource_id)


def _execution_cluster(cluster_id: str, stored: str) -> str:
    return cluster_id if stored == LOCAL_CLUSTER else stored


def _query_fingerprint(kind: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps({"kind": kind, **payload}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode_page_token(fingerprint: str, position: Mapping[str, object]) -> str:
    payload = json.dumps(
        {"query": fingerprint, "position": position},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_page_token(token: str | None, fingerprint: str) -> dict[str, object] | None:
    if token is None:
        return None
    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        if payload["query"] != fingerprint or not isinstance(payload["position"], dict):
            raise InvalidPageToken("page token does not match the query")
        return payload["position"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, InvalidPageToken):
            raise
        raise InvalidPageToken("malformed page token") from exc


def _available_source(source_id: str, *, backend_id: str = "") -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=source_id,
        backend_id=backend_id,
        state=SourceState.AVAILABLE,
        freshness=Freshness.CURRENT,
        observed_at=Timestamp.now(),
        error_code="",
        error_message="",
    )


def _unavailable_backend_source(backend_id: str, error: Exception) -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=f"backend:{backend_id}",
        backend_id=backend_id,
        state=SourceState.UNAVAILABLE,
        freshness=Freshness.UNKNOWN,
        observed_at=None,
        error_code="backend_unavailable",
        error_message=str(error)[:MAX_SOURCE_ERROR_MESSAGE],
    )


def _unavailable_finelog_source(cluster_id: str, error: Exception) -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=f"finelog:{cluster_id}",
        backend_id="",
        state=SourceState.UNAVAILABLE,
        freshness=Freshness.UNKNOWN,
        observed_at=None,
        error_code="finelog_unavailable",
        error_message=str(error)[:MAX_SOURCE_ERROR_MESSAGE],
    )


def _unsupported_source(source_id: str) -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=source_id,
        backend_id="",
        state=SourceState.UNSUPPORTED,
        freshness=Freshness.UNKNOWN,
        observed_at=None,
        error_code="unsupported",
        error_message="",
    )


def _opaque_uid(value: str) -> str:
    return uuid.uuid5(_RESOURCE_UID_NAMESPACE, value).hex


def _string_node_attribute(key: str, value: str) -> NodeAttribute | None:
    if not value:
        return None
    return NodeAttribute(key=key, kind=NodeAttributeKind.STRING, string_value=value)


def _worker_attributes(metadata: job_pb2.WorkerMetadata) -> tuple[NodeAttribute, ...]:
    attributes: list[NodeAttribute] = []
    for key, value in sorted(metadata.attributes.items()):
        selected = value.WhichOneof("value")
        if selected == "string_value":
            attributes.append(NodeAttribute(key, NodeAttributeKind.STRING, string_value=value.string_value))
        elif selected == "int_value":
            attributes.append(NodeAttribute(key, NodeAttributeKind.INTEGER, integer_value=value.int_value))
        elif selected == "float_value":
            attributes.append(NodeAttribute(key, NodeAttributeKind.FLOAT, float_value=value.float_value))
    return tuple(attributes)


def _worker_accelerator(metadata: job_pb2.WorkerMetadata) -> tuple[str, str, int]:
    kind = metadata.device.WhichOneof("device")
    if kind == "gpu":
        return "gpu", metadata.device.gpu.variant, metadata.device.gpu.count
    if kind == "tpu":
        return "tpu", metadata.device.tpu.variant, metadata.device.tpu.count
    return "", "", 0


def _slice_lifecycle(value: str) -> SliceLifecycle:
    if value == "ready":
        return SliceLifecycle.READY
    if value == "failed":
        return SliceLifecycle.FAILED
    if value in {"deleting", "stopping", "terminated"}:
        return SliceLifecycle.DELETING
    return SliceLifecycle.CREATING


class ResourceController:
    """Present typed resources over one controller snapshot and backend set."""

    def __init__(
        self,
        *,
        cluster_id: str,
        db: ControllerDB,
        legacy: LegacyResourceReader,
        backends: Mapping[str, TaskBackend],
        log_client: LogClient | None = None,
    ) -> None:
        if not cluster_id.strip():
            raise ValueError("cluster_id is required for resource identities")
        self._cluster_id = cluster_id
        self._db = db
        self._legacy = legacy
        self._backends = dict(backends)
        self._log_client = log_client

    def submit_job(self, spec: JobSpec, bundle_blob: bytes = b"") -> JobIdentity:
        response = self._legacy.launch_job(_launch_request(spec, bundle_blob), None)
        key = ResourceKey(self._cluster_id, ResourceKind.JOB, response.job_id)
        return self.describe_job(key).summary.identity

    def list_jobs(self, query: JobQuery = JobQuery()) -> Page[JobSummary]:
        page_size = _page_size(query.page_size, _MAX_JOB_PAGE)
        fingerprint = _query_fingerprint(
            "jobs",
            {
                "owner_id": query.owner_id,
                "parent": query.parent.resource_id if query.parent else None,
                "job_id_prefix": query.job_id_prefix,
                "states": sorted(int(state) for state in query.states),
                "backend_id": query.backend_id,
                "execution_cluster_id": query.execution_cluster_id,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        offset = int(position["offset"]) if position is not None else 0
        old_query = controller_pb2.Controller.JobQuery(
            job_id_prefix=query.job_id_prefix or (f"/{query.owner_id}/" if query.owner_id else ""),
            backend_id=query.backend_id or "",
            cluster=_stored_cluster(self._cluster_id, query.execution_cluster_id),
            offset=offset,
            limit=page_size,
        )
        if query.parent is not None:
            old_query.scope = controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN
            old_query.parent_job_id = query.parent.resource_id
        if len(query.states) == 1:
            old_query.state_filter = job_pb2.JobState.Name(next(iter(query.states))).removeprefix("JOB_STATE_").lower()
        response = self._legacy.list_jobs(controller_pb2.Controller.ListJobsRequest(query=old_query), None)
        statuses = [status for status in response.jobs if not query.states or status.state in query.states]
        authority = self._job_authorities(JobName.from_wire(status.job_id) for status in statuses)
        submitted = {JobName.from_wire(status.job_id): timestamp_from_proto(status.submitted_at) for status in statuses}
        parent_ids = [JobName.from_wire(status.parent_job_id) for status in statuses if status.parent_job_id]
        parent_submitted = self._job_submission_times(parent_ids)
        items = tuple(self._job_summary(status, authority, submitted, parent_submitted) for status in statuses)
        next_token = None
        if response.has_more:
            next_token = _encode_page_token(fingerprint, {"offset": offset + len(response.jobs)})
        return Page(
            items=items,
            next_page_token=next_token,
            source_statuses=(_available_source(f"controller:{self._cluster_id}"),),
        )

    def describe_job(self, key: ResourceKey) -> JobDetail:
        _require_key(key, ResourceKind.JOB, self._cluster_id)
        response = self._legacy.get_job_status(
            controller_pb2.Controller.GetJobStatusRequest(job_id=key.resource_id),
            None,
        )
        status = response.job
        if not status.job_id:
            raise ResourceNotFound(key.resource_id)
        job_id = JobName.from_wire(status.job_id)
        authority = self._job_authorities([job_id])
        submitted = {job_id: timestamp_from_proto(status.submitted_at)}
        parent_ids = [JobName.from_wire(status.parent_job_id)] if status.parent_job_id else []
        summary = self._job_summary(status, authority, submitted, self._job_submission_times(parent_ids))
        return JobDetail(summary=summary, spec=_job_spec(response.request))

    def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]:
        page_size = _page_size(query.page_size, _MAX_TASK_PAGE)
        fingerprint = _query_fingerprint(
            "tasks",
            {
                "job": query.job.resource_id if query.job else None,
                "job_id_prefix": query.job_id_prefix,
                "states": sorted(int(state) for state in query.states),
                "backend_id": query.backend_id,
                "authority_cluster_id": query.authority_cluster_id,
                "execution_cluster_id": query.execution_cluster_id,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        with self._db.read_snapshot() as tx:
            stmt = reads.task_detail_query().add_columns(tasks_table.c.task_index)
            if query.job is not None:
                _require_key(query.job, ResourceKind.JOB, query.job.cluster_id)
                stmt = stmt.where(tasks_table.c.job_id == JobName.from_wire(query.job.resource_id))
            if query.job_id_prefix:
                stmt = stmt.where(tasks_table.c.job_id.like(_escaped_prefix(query.job_id_prefix), escape="\\"))
            if query.states:
                stmt = stmt.where(tasks_table.c.state.in_(query.states))
            if query.backend_id is not None:
                stmt = stmt.where(tasks_table.c.backend_id == query.backend_id)
            if query.execution_cluster_id is not None:
                stmt = stmt.where(tasks_table.c.cluster == _stored_cluster(self._cluster_id, query.execution_cluster_id))
            if position is not None:
                last_ms = int(position["submitted_at_ms"])
                last_id = JobName.from_wire(str(position["task_id"]))
                stmt = stmt.where(
                    or_(
                        tasks_table.c.submitted_at_ms < Timestamp.from_ms(last_ms),
                        and_(
                            tasks_table.c.submitted_at_ms == Timestamp.from_ms(last_ms),
                            tasks_table.c.task_id > last_id,
                        ),
                    )
                )
            rows = tx.execute(
                stmt.order_by(tasks_table.c.submitted_at_ms.desc(), tasks_table.c.task_id.asc()).limit(page_size + 1)
            ).all()
            page_rows = rows[:page_size]
            task_ids = [row.task_id for row in page_rows]
            attempt_keys = [
                (row.task_id, int(row.current_attempt_id)) for row in page_rows if int(row.current_attempt_id) >= 0
            ]
            current_attempts = reads.bulk_get_attempts(tx, attempt_keys)
            counts = reads.attempt_counts_for_tasks(tx, task_ids)
            jobs = self._job_rows(tx, {row.job_id for row in page_rows})
        items = tuple(
            self._task_summary(
                row,
                current_attempts.get((row.task_id, int(row.current_attempt_id))),
                counts.get(row.task_id, AttemptCounts()),
                jobs[row.job_id],
            )
            for row in page_rows
            if query.authority_cluster_id is None
            or self._authority_cluster(jobs[row.job_id]) == query.authority_cluster_id
        )
        next_token = None
        if len(rows) > page_size and page_rows:
            last = page_rows[-1]
            next_token = _encode_page_token(
                fingerprint,
                {"submitted_at_ms": last.submitted_at_ms.epoch_ms(), "task_id": last.task_id.to_wire()},
            )
        return Page(items=items, next_page_token=next_token, source_statuses=self._source_statuses())

    def describe_task(self, key: ResourceKey) -> TaskDetail:
        _require_key(key, ResourceKind.TASK, self._cluster_id)
        task_id = JobName.from_wire(key.resource_id)
        with self._db.read_snapshot() as tx:
            row = tx.execute(
                reads.task_detail_query().add_columns(tasks_table.c.task_index).where(tasks_table.c.task_id == task_id)
            ).first()
            if row is None:
                raise ResourceNotFound(key.resource_id)
            attempt_rows = tx.execute(
                select(*reads.ATTEMPT_COLS)
                .where(task_attempts_table.c.task_id == task_id)
                .order_by(task_attempts_table.c.attempt_id.asc())
            ).all()
            counts = reads.attempt_counts_for_tasks(tx, [task_id]).get(task_id, AttemptCounts())
            job = self._job_rows(tx, {row.job_id})[row.job_id]
        current = next((candidate for candidate in attempt_rows if candidate.attempt_id == row.current_attempt_id), None)
        summary = self._task_summary(row, current, counts, job)
        return TaskDetail(
            summary=summary,
            attempts=tuple(self._attempt_summary(row, candidate, job) for candidate in attempt_rows),
            source_statuses=self._source_statuses(),
            root_cause_highlights=(),
        )

    def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail:
        detail = self.describe_task(locator.task)
        if locator.attempt_number is None:
            identity = detail.summary.current_attempt
            if identity is None:
                raise ResourceNotFound(f"{locator.task.resource_id}:current")
            number = identity.attempt_number
        else:
            number = locator.attempt_number
        attempt = next((candidate for candidate in detail.attempts if candidate.identity.attempt_number == number), None)
        if attempt is None:
            raise ResourceNotFound(f"{locator.task.resource_id}:{number}")
        return AttemptDetail(summary=attempt, runtime=None, source_statuses=detail.source_statuses)

    def require_task(self, identity: TaskIdentity) -> TaskDetail:
        detail = self.describe_task(identity.key)
        if detail.summary.identity.task_uid != identity.task_uid:
            raise ResourceReplaced(identity.key.resource_id)
        return detail

    def require_attempt(self, identity: AttemptIdentity) -> AttemptDetail:
        detail = self.describe_attempt(AttemptLocator(identity.task, identity.attempt_number))
        if detail.summary.identity.attempt_uid != identity.attempt_uid:
            raise ResourceReplaced(f"{identity.task.resource_id}:{identity.attempt_number}")
        return detail

    def cancel_job(
        self,
        identity: JobIdentity,
        *,
        idempotency_key: str,
        principal_id: str = "local_admin",
    ) -> ActionReceipt:
        payload_hash = _action_payload_hash(ActionKind.CANCEL_JOB, identity.job_uid, None)
        with self._db.transaction() as tx:
            duplicate = _duplicate_action(
                tx,
                principal_id=principal_id,
                kind=ActionKind.CANCEL_JOB,
                idempotency_key=idempotency_key,
                payload_hash=payload_hash,
            )
            if duplicate is not None:
                return duplicate
            row = tx.execute(
                select(jobs_table).where(jobs_table.c.job_id == JobName.from_wire(identity.key.resource_id))
            ).first()
            if row is None:
                raise ResourceNotFound(identity.key.resource_id)
            authority = self._authority_cluster(self._job_rows(tx, {row.job_id})[row.job_id])
            expected = _resource_uid(ResourceKind.JOB, authority, row.job_id.to_wire(), row.submitted_at_ms)
            if identity.key.cluster_id != authority or identity.job_uid != expected:
                raise ResourceReplaced(identity.key.resource_id)
            job_ops.cancel(tx, job_id=row.job_id, reason="Cancelled by resource action")
            receipt = _completed_action(
                kind=ActionKind.CANCEL_JOB,
                target=identity.key,
                expected_target_uid=identity.job_uid,
                expected_attempt_uid=None,
                result=ActionResult.SATISFIED,
            )
            action_persistence.insert_action(
                tx,
                receipt,
                authority_cluster_id=authority,
                authority_action_id=receipt.action_id,
                backend_id="",
                execution_cluster_id=_execution_cluster(self._cluster_id, row.cluster),
                principal_id=principal_id,
                idempotency_key=_require_idempotency_key(idempotency_key),
                payload_hash=payload_hash,
            )
            return receipt

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
        principal_id: str = "local_admin",
    ) -> ActionReceipt:
        return self._terminal_action(
            identity,
            expected_attempt_uid=expected_attempt_uid,
            idempotency_key=idempotency_key,
            principal_id=principal_id,
            kind=ActionKind.RETRY_TASK,
            terminal_kind=TerminalKind.PREEMPT,
            result=ActionResult.TARGET_ABSENT,
        )

    def terminate_attempt(
        self,
        identity: AttemptIdentity,
        *,
        idempotency_key: str,
        principal_id: str = "local_admin",
    ) -> ActionReceipt:
        task = self.describe_task(identity.task).summary.identity
        if identity.attempt_number < 0:
            raise ActionPolicyRejected("attempt_number must be non-negative")
        return self._terminal_action(
            task,
            expected_attempt_uid=identity.attempt_uid,
            expected_attempt_number=identity.attempt_number,
            idempotency_key=idempotency_key,
            principal_id=principal_id,
            kind=ActionKind.TERMINATE_ATTEMPT,
            terminal_kind=TerminalKind.TERMINATE,
            result=ActionResult.SATISFIED,
        )

    def get_action_receipt(self, action_id: str) -> ActionReceipt:
        with self._db.read_snapshot() as tx:
            receipt = action_persistence.action_by_id(tx, action_id)
        if receipt is None:
            raise ResourceNotFound(action_id)
        return receipt

    def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]:
        page_size = _page_size(query.page_size, _MAX_TASK_PAGE)
        fingerprint = _query_fingerprint(
            "endpoints",
            {
                "name_prefix": query.name_prefix,
                "task": query.task.resource_id if query.task is not None else None,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        task_ids = (JobName.from_wire(query.task.resource_id),) if query.task is not None else ()
        rows = sorted(
            self._db.caches[EndpointsProjection].query(
                ProjectionEndpointQuery(name_prefix=query.name_prefix, task_ids=task_ids)
            ),
            key=lambda row: (row.name, row.endpoint_id),
        )
        if position is not None:
            last_key = (str(position["name"]), str(position["endpoint_id"]))
            rows = [row for row in rows if (row.name, row.endpoint_id) > last_key]
        page_rows = rows[:page_size]
        coordinates = self._endpoint_coordinates(page_rows)
        next_token = None
        if len(rows) > page_size:
            last = page_rows[-1]
            next_token = _encode_page_token(fingerprint, {"name": last.name, "endpoint_id": last.endpoint_id})
        return Page(
            items=tuple(self._endpoint_summary(row, coordinates[row.endpoint_id]) for row in page_rows),
            next_page_token=next_token,
            source_statuses=(_available_source(f"controller:{self._cluster_id}"),),
        )

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail:
        _require_key(key, ResourceKind.ENDPOINT, key.cluster_id)
        row = self._db.caches[EndpointsProjection].get(key.resource_id)
        if row is None:
            raise ResourceNotFound(key.resource_id)
        coordinates = self._endpoint_coordinates([row])[row.endpoint_id]
        if coordinates[0] != key.cluster_id:
            raise ResourceNotFound(key.resource_id)
        return EndpointDetail(
            summary=self._endpoint_summary(row, coordinates),
            address=row.address,
            metadata=dict(row.metadata),
        )

    def mint_endpoint_token(self, key: ResourceKey, ttl: Duration | None) -> EndpointToken:
        detail = self.describe_endpoint(key)
        request = controller_pb2.Controller.MintEndpointTokenRequest(endpoint_name=detail.summary.name)
        if ttl is not None:
            request.ttl.CopyFrom(duration_to_proto(ttl))
        response = self._legacy.mint_endpoint_token(request, None)
        return EndpointToken(
            token=response.token,
            expires_at=timestamp_from_proto(response.expires_at),
            capability_url=response.capability_url,
        )

    def fetch_logs(
        self,
        target: JobIdentity | TaskIdentity | AttemptIdentity,
        query: LogQuery = LogQuery(),
    ) -> LogPage:
        job_name, attempt_number = self._validated_log_target(target)
        if self._log_client is None:
            return LogPage(
                entries=(),
                next_cursor=query.cursor,
                source_statuses=(
                    _unavailable_finelog_source(self._cluster_id, RuntimeError("finelog is not configured")),
                ),
            )
        source, match_scope = build_log_source(job_name, attempt_number)
        minimum_level = ""
        if query.minimum_level != logging_pb2.LOG_LEVEL_UNKNOWN:
            minimum_level = logging_pb2.LogLevel.Name(query.minimum_level).removeprefix("LOG_LEVEL_")
        request = logging_pb2.FetchLogsRequest(
            source=source,
            match_scope=match_scope,
            cursor=query.cursor,
            max_lines=query.max_lines,
            substring=query.substring,
            min_level=minimum_level,
            tail=query.tail,
        )
        if query.after is not None:
            request.since_ms = query.after.epoch_ms()
        try:
            response = self._log_client.fetch_logs(request)
        except (ConnectError, ConnectionError, OSError, RuntimeError) as exc:
            return LogPage(
                entries=(),
                next_cursor=query.cursor,
                source_statuses=(_unavailable_finelog_source(self._cluster_id, exc),),
            )
        return LogPage(
            entries=tuple(iris_logging_pb2.LogEntry.FromString(entry.SerializeToString()) for entry in response.entries),
            next_cursor=response.cursor,
            source_statuses=(_available_source(f"finelog:{self._cluster_id}"),),
        )

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]:
        page_size = _page_size(query.page_size, 500)
        fingerprint = _query_fingerprint(
            "activity",
            {
                "cluster_id": query.target.cluster_id,
                "kind": query.target.kind.value,
                "resource_id": query.target.resource_id,
                "attempt_uid": query.attempt_uid,
                "after_ms": query.after.epoch_ms() if query.after is not None else None,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        attempt_uids = self._activity_attempt_uids(query)
        with self._db.read_snapshot() as tx:
            receipts = action_persistence.actions_for_target(
                tx,
                query.target,
                after=query.after,
                limit=page_size + 1,
            )
        entries = [self._action_activity(receipt) for receipt in receipts]
        source_statuses = [_available_source(f"controller:{self._cluster_id}")]
        if attempt_uids:
            event_entries, event_status = self._task_event_activity(
                query.target, attempt_uids, query.after, page_size + 1
            )
            entries.extend(event_entries)
            source_statuses.append(event_status)
        else:
            source_statuses.append(_unsupported_source(f"finelog:{self._cluster_id}"))
        entries.sort(key=lambda entry: (entry.occurred_at.epoch_ms(), entry.entry_id), reverse=True)
        if position is not None:
            last_key = (int(position["occurred_at_ms"]), str(position["entry_id"]))
            entries = [entry for entry in entries if (entry.occurred_at.epoch_ms(), entry.entry_id) < last_key]
        items = tuple(entries[:page_size])
        next_token = None
        if len(entries) > page_size:
            last = items[-1]
            next_token = _encode_page_token(
                fingerprint,
                {"occurred_at_ms": last.occurred_at.epoch_ms(), "entry_id": last.entry_id},
            )
        return Page(items=items, next_page_token=next_token, source_statuses=tuple(source_statuses))

    def _activity_attempt_uids(self, query: ActivityQuery) -> tuple[str, ...]:
        target = query.target
        if target.kind is ResourceKind.JOB:
            self.describe_job(target)
            return ()
        if target.kind is ResourceKind.TASK:
            detail = self.describe_task(target)
            available = tuple(item.identity.attempt_uid for item in detail.attempts)
        elif target.kind is ResourceKind.ATTEMPT:
            task_id, _, number_text = target.resource_id.rpartition(":")
            task_key = ResourceKey(target.cluster_id, ResourceKind.TASK, task_id)
            detail = self.describe_attempt(AttemptLocator(task_key, int(number_text)))
            available = (detail.summary.identity.attempt_uid,)
        else:
            raise InvalidResourceKey(f"activity is unsupported for {target.kind.value}")
        if query.attempt_uid is None:
            return available
        if query.attempt_uid not in available:
            raise ResourceReplaced(target.resource_id)
        return (query.attempt_uid,)

    def _task_event_activity(
        self,
        target: ResourceKey,
        attempt_uids: tuple[str, ...],
        after: Timestamp | None,
        limit: int,
    ) -> tuple[tuple[ActivityEntry, ...], ResourceSourceStatus]:
        if self._log_client is None:
            error = RuntimeError("finelog is not configured")
            return (), _unavailable_finelog_source(self._cluster_id, error)
        task_id = target.resource_id.rpartition(":")[0] if target.kind is ResourceKind.ATTEMPT else target.resource_id
        task_literal = task_id.replace("'", "''")
        uid_literals = ", ".join(f"'{uid.replace("'", "''")}'" for uid in attempt_uids)
        after_predicate = f" AND ts > to_timestamp({after.epoch_ms()} / 1000.0)" if after is not None else ""
        sql = (
            "SELECT attempt_id, attempt_uid, ts, type, reason, message, source, count "
            f"FROM \"{TASK_EVENT_NAMESPACE}\" WHERE task_id = '{task_literal}' "
            f"AND attempt_uid IN ({uid_literals}){after_predicate} "
            f"ORDER BY ts DESC LIMIT {limit}"
        )
        try:
            rows = self._log_client.query(sql, max_rows=limit).to_pylist()
        except (ConnectError, ConnectionError, OSError, RuntimeError) as exc:
            return (), _unavailable_finelog_source(self._cluster_id, exc)
        entries = tuple(self._task_event_entry(task_id, row) for row in rows)
        return entries, _available_source(f"finelog:{self._cluster_id}")

    def _task_event_entry(self, task_id: str, row: Mapping[str, object]) -> ActivityEntry:
        attempt_number = row["attempt_id"]
        attempt_uid = row["attempt_uid"]
        occurred_at = row["ts"]
        values = (row["type"], row["reason"], row["message"], row["source"])
        if not isinstance(attempt_number, int) or not isinstance(attempt_uid, str):
            raise ValueError("finelog task event has invalid Attempt identity")
        if not isinstance(occurred_at, datetime) or not all(isinstance(value, str) for value in values):
            raise ValueError("finelog task event has invalid typed fields")
        normalized = occurred_at.replace(tzinfo=UTC) if occurred_at.tzinfo is None else occurred_at.astimezone(UTC)
        severity, kind, message, source = values
        sequence_source = json.dumps(
            [task_id, attempt_uid, normalized.isoformat(), kind, message, source],
            separators=(",", ":"),
        ).encode()
        sequence = int.from_bytes(hashlib.sha256(sequence_source).digest()[:8], "big")
        return ActivityEntry(
            entry_id=f"finelog:task-events:{sequence}",
            occurred_at=Timestamp.from_seconds(normalized.timestamp()),
            source=source,
            severity=severity,
            kind=kind,
            message=message,
            target=ResourceKey(self._cluster_id, ResourceKind.ATTEMPT, f"{task_id}:{attempt_number}"),
            attempt_uid=attempt_uid,
            correlation_id=None,
            attributes={"count": str(row["count"])},
        )

    @staticmethod
    def _action_activity(receipt: ActionReceipt) -> ActivityEntry:
        return ActivityEntry(
            entry_id=f"action:{receipt.action_id}",
            occurred_at=receipt.updated_at,
            source="controller",
            severity="error" if receipt.state is ActionState.FAILED else "info",
            kind=receipt.kind.value,
            message=receipt.result_message or receipt.result_code.value,
            target=receipt.target,
            attempt_uid=receipt.expected_attempt_uid,
            correlation_id=receipt.action_id,
            attributes={"state": receipt.state.value, "result": receipt.result_code.value},
        )

    def exec_attempt(
        self,
        locator: AttemptLocator,
        command: tuple[str, ...],
        timeout: Duration | None,
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        attempt = self.describe_attempt(locator)
        self._require_current_attempt(attempt.summary.identity)
        return self._legacy.exec_in_container(
            controller_pb2.Controller.ExecInContainerRequest(
                task_id=locator.task.resource_id,
                command=command,
                timeout_seconds=timeout.to_seconds() if timeout is not None else 0,
            ),
            None,
        )

    def profile_attempt(
        self,
        locator: AttemptLocator,
        profile: job_pb2.ProfileType,
        duration: Duration | None,
    ) -> job_pb2.ProfileTaskResponse:
        attempt = self.describe_attempt(locator)
        self._require_current_attempt(attempt.summary.identity)
        return self._legacy.profile_task(
            job_pb2.ProfileTaskRequest(
                target=locator.task.resource_id,
                duration_seconds=duration.to_seconds() if duration is not None else 0,
                profile_type=profile,
            ),
            None,
        )

    def _validated_log_target(
        self,
        target: JobIdentity | TaskIdentity | AttemptIdentity,
    ) -> tuple[JobName, int]:
        if isinstance(target, JobIdentity):
            detail = self.describe_job(target.key)
            if detail.summary.identity.job_uid != target.job_uid:
                raise ResourceReplaced(target.key.resource_id)
            return JobName.from_wire(target.key.resource_id), -1
        if isinstance(target, TaskIdentity):
            self.require_task(target)
            return JobName.from_wire(target.key.resource_id), -1
        self.require_attempt(target)
        return JobName.from_wire(target.task.resource_id), target.attempt_number

    def _require_current_attempt(self, identity: AttemptIdentity) -> None:
        task = self.describe_task(identity.task)
        if task.summary.current_attempt != identity:
            raise ResourceReplaced(f"{identity.task.resource_id}:{identity.attempt_number}")

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]:
        page_size = _page_size(query.page_size, _MAX_TASK_PAGE)
        fingerprint = _query_fingerprint(
            "nodes",
            {
                "backend_id": query.backend_id,
                "contains": query.contains,
                "health": sorted(value.value for value in query.health),
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        nodes, _details, statuses = self._node_snapshot()
        filtered = [
            node
            for node in nodes
            if (query.backend_id is None or node.identity.backend_id == query.backend_id)
            and (query.contains is None or query.contains.casefold() in node.identity.key.resource_id.casefold())
            and (not query.health or node.health in query.health)
        ]
        filtered.sort(
            key=lambda node: (
                node.identity.backend_id,
                node.identity.key.resource_id,
                node.identity.node_uid,
            )
        )
        if position is not None:
            last_key = (
                str(position["backend_id"]),
                str(position["node_id"]),
                str(position["node_uid"]),
            )
            filtered = [
                node
                for node in filtered
                if (node.identity.backend_id, node.identity.key.resource_id, node.identity.node_uid) > last_key
            ]
        items = tuple(filtered[:page_size])
        next_token = None
        if len(filtered) > page_size:
            last = items[-1]
            next_token = _encode_page_token(
                fingerprint,
                {
                    "backend_id": last.identity.backend_id,
                    "node_id": last.identity.key.resource_id,
                    "node_uid": last.identity.node_uid,
                },
            )
        return Page(items=items, next_page_token=next_token, source_statuses=statuses)

    def describe_node(self, locator: NodeLocator) -> NodeDetail:
        nodes, details, statuses = self._node_snapshot()
        matches = [
            node
            for node in nodes
            if node.identity.key == locator.key
            and node.identity.backend_id == locator.backend_id
            and (locator.node_uid is None or node.identity.node_uid == locator.node_uid)
        ]
        if not matches:
            raise ResourceNotFound(locator.key.resource_id)
        if len(matches) != 1:
            raise ActionPolicyRejected(f"Node locator {locator.key.resource_id!r} is ambiguous")
        node = matches[0]
        address, attributes = details[(node.identity.backend_id, node.identity.node_uid)]
        return NodeDetail(
            summary=node,
            address=address,
            attributes=attributes,
            recent_attempts=(),
            bootstrap_log_key=None,
            source_statuses=statuses,
        )

    def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]:
        page_size = _page_size(query.page_size, _MAX_TASK_PAGE)
        fingerprint = _query_fingerprint(
            "slices",
            {
                "backend_id": query.backend_id,
                "scaling_group_id": query.scaling_group_id,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        slices, _members, statuses = self._slice_snapshot()
        filtered = [
            item
            for item in slices
            if (query.backend_id is None or item.identity.backend_id == query.backend_id)
            and (query.scaling_group_id is None or item.scaling_group_id == query.scaling_group_id)
        ]
        filtered.sort(
            key=lambda item: (
                item.identity.backend_id,
                item.identity.key.resource_id,
                item.identity.slice_uid,
            )
        )
        if position is not None:
            last_key = (
                str(position["backend_id"]),
                str(position["slice_id"]),
                str(position["slice_uid"]),
            )
            filtered = [
                item
                for item in filtered
                if (item.identity.backend_id, item.identity.key.resource_id, item.identity.slice_uid) > last_key
            ]
        items = tuple(filtered[:page_size])
        next_token = None
        if len(filtered) > page_size:
            last = items[-1]
            next_token = _encode_page_token(
                fingerprint,
                {
                    "backend_id": last.identity.backend_id,
                    "slice_id": last.identity.key.resource_id,
                    "slice_uid": last.identity.slice_uid,
                },
            )
        return Page(items=items, next_page_token=next_token, source_statuses=statuses)

    def describe_slice(self, locator: SliceLocator) -> SliceDetail:
        slices, members, statuses = self._slice_snapshot()
        matches = [
            item
            for item in slices
            if item.identity.key == locator.key
            and item.identity.backend_id == locator.backend_id
            and (locator.slice_uid is None or item.identity.slice_uid == locator.slice_uid)
        ]
        if not matches:
            raise ResourceNotFound(locator.key.resource_id)
        if len(matches) != 1:
            raise ActionPolicyRejected(f"Slice locator {locator.key.resource_id!r} is ambiguous")
        item = matches[0]
        return SliceDetail(
            summary=item,
            members=members.get((item.identity.backend_id, item.identity.slice_uid), ()),
            source_statuses=statuses,
        )

    def _node_snapshot(
        self,
    ) -> tuple[
        tuple[NodeSummary, ...],
        dict[tuple[str, str], tuple[str | None, tuple[NodeAttribute, ...]]],
        tuple[ResourceSourceStatus, ...],
    ]:
        worker_response = self._legacy.list_workers(
            controller_pb2.Controller.ListWorkersRequest(query=controller_pb2.Controller.WorkerQuery()),
            None,
        )
        workers_by_backend: dict[str, list] = {}
        for worker in worker_response.workers:
            workers_by_backend.setdefault(worker.backend_id, []).append(worker)
        nodes: list[NodeSummary] = []
        details: dict[tuple[str, str], tuple[str | None, tuple[NodeAttribute, ...]]] = {}
        statuses: list[ResourceSourceStatus] = []
        for backend_id, backend in sorted(self._backends.items()):
            try:
                status = backend.status()
            except (ConnectionError, ProviderError) as exc:
                statuses.append(_unavailable_backend_source(backend_id, exc))
            else:
                if status.HasField("kubernetes"):
                    if len(status.kubernetes.nodes) > MAX_PROVIDER_SNAPSHOT_ITEMS:
                        statuses.append(
                            _unavailable_backend_source(
                                backend_id,
                                ValueError(f"provider returned more than {MAX_PROVIDER_SNAPSHOT_ITEMS} nodes"),
                            )
                        )
                        continue
                    observed_at = Timestamp.now()
                    for node in status.kubernetes.nodes:
                        identity = NodeIdentity(
                            ResourceKey(self._cluster_id, ResourceKind.NODE, node.name),
                            backend_id,
                            _opaque_uid(f"kubernetes:{backend_id}:{node.name}:{node.created}"),
                        )
                        summary = NodeSummary(
                            identity=identity,
                            health=NodeHealth.READY if node.ready else NodeHealth.UNAVAILABLE,
                            schedulable=node.schedulable,
                            capacity=NodeCapacity(
                                cpu_millicores=node.cpu_millicores,
                                memory_bytes=node.memory_bytes,
                                disk_bytes=node.disk_bytes,
                                accelerator_kind="gpu" if node.gpu_count else "",
                                accelerator_variant=node.gpu_model,
                                accelerator_count=node.gpu_count,
                            ),
                            scaling_group_id=None,
                            slice=None,
                            running_task_count=node.running_pods,
                            observed_at=observed_at,
                        )
                        nodes.append(summary)
                        attributes = tuple(
                            attribute
                            for attribute in (
                                _string_node_attribute("instance_type", node.instance_type),
                                _string_node_attribute("region", node.region),
                            )
                            if attribute is not None
                        )
                        details[(backend_id, identity.node_uid)] = (None, attributes)
                statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
            if BackendCapability.WORKER_DAEMON not in backend.capabilities:
                continue
            for worker in workers_by_backend.get(backend_id, []):
                identity = NodeIdentity(
                    ResourceKey(self._cluster_id, ResourceKind.NODE, worker.worker_id),
                    backend_id,
                    worker.worker_id,
                )
                accelerator_kind, accelerator_variant, accelerator_count = _worker_accelerator(worker.metadata)
                slice_identity = None
                slice_id = str(worker.metadata.tpu_name or "")
                if slice_id:
                    slice_identity = SliceIdentity(
                        ResourceKey(self._cluster_id, ResourceKind.SLICE, slice_id),
                        backend_id,
                        _opaque_uid(f"rpc:{backend_id}:{slice_id}"),
                    )
                nodes.append(
                    NodeSummary(
                        identity=identity,
                        health=NodeHealth.READY if worker.healthy else NodeHealth.DEGRADED,
                        schedulable=worker.healthy,
                        capacity=NodeCapacity(
                            cpu_millicores=worker.metadata.cpu_count * 1_000,
                            memory_bytes=worker.metadata.memory_bytes,
                            disk_bytes=worker.metadata.disk_bytes,
                            accelerator_kind=accelerator_kind,
                            accelerator_variant=accelerator_variant,
                            accelerator_count=accelerator_count,
                        ),
                        scaling_group_id=worker.scale_group or None,
                        slice=slice_identity,
                        running_task_count=len(worker.running_job_ids),
                        observed_at=timestamp_from_proto(worker.last_heartbeat),
                    )
                )
                details[(backend_id, identity.node_uid)] = (
                    worker.address or None,
                    _worker_attributes(worker.metadata),
                )
        return tuple(nodes), details, tuple(statuses)

    def _slice_snapshot(
        self,
    ) -> tuple[
        tuple[SliceSummary, ...],
        dict[tuple[str, str], tuple[SliceMember, ...]],
        tuple[ResourceSourceStatus, ...],
    ]:
        slices: list[SliceSummary] = []
        members: dict[tuple[str, str], tuple[SliceMember, ...]] = {}
        statuses: list[ResourceSourceStatus] = []
        for backend_id, backend in sorted(self._backends.items()):
            if BackendCapability.IRIS_AUTOSCALER not in backend.capabilities:
                statuses.append(
                    ResourceSourceStatus(
                        source_id=f"backend:{backend_id}",
                        backend_id=backend_id,
                        state=SourceState.UNSUPPORTED,
                        freshness=Freshness.UNKNOWN,
                        observed_at=None,
                        error_code="unsupported",
                        error_message="",
                    )
                )
                continue
            try:
                status = backend.autoscaler_status()
            except (ConnectionError, ProviderError) as exc:
                statuses.append(_unavailable_backend_source(backend_id, exc))
                continue
            slice_count = sum(len(group.slices) for group in status.groups)
            if slice_count > MAX_PROVIDER_SNAPSHOT_ITEMS:
                statuses.append(
                    _unavailable_backend_source(
                        backend_id,
                        ValueError(f"provider returned more than {MAX_PROVIDER_SNAPSHOT_ITEMS} slices"),
                    )
                )
                continue
            statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
            observed_at = (
                timestamp_from_proto(status.last_evaluation) if status.HasField("last_evaluation") else Timestamp.now()
            )
            for group in status.groups:
                for item in group.slices:
                    slice_uid = _opaque_uid(
                        f"rpc:{backend_id}:{item.slice_id}:"
                        f"{item.created_at.epoch_ms if item.HasField('created_at') else 0}"
                    )
                    identity = SliceIdentity(
                        ResourceKey(self._cluster_id, ResourceKind.SLICE, item.slice_id),
                        backend_id,
                        slice_uid,
                    )
                    lifecycle = _slice_lifecycle(item.state)
                    membership_state = (
                        MembershipState.OBSERVED if lifecycle is SliceLifecycle.READY else MembershipState.UNKNOWN
                    )
                    slices.append(
                        SliceSummary(
                            identity=identity,
                            scaling_group_id=group.name,
                            lifecycle=lifecycle,
                            membership_state=membership_state,
                            observed_member_count=len(item.vms),
                            observed_at=observed_at,
                            error_message=item.error_message,
                        )
                    )
                    members[(backend_id, slice_uid)] = tuple(
                        SliceMember(
                            provider_node_id=vm.vm_id,
                            node=(
                                NodeIdentity(
                                    ResourceKey(
                                        self._cluster_id,
                                        ResourceKind.NODE,
                                        vm.worker_id or vm.vm_id,
                                    ),
                                    backend_id,
                                    vm.worker_id or vm.vm_id,
                                )
                                if vm.worker_id or vm.vm_id
                                else None
                            ),
                            observed_at=observed_at,
                        )
                        for vm in item.vms
                    )
        return tuple(slices), members, tuple(statuses)

    def _terminal_action(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
        principal_id: str,
        kind: ActionKind,
        terminal_kind: TerminalKind,
        result: ActionResult,
        expected_attempt_number: int | None = None,
    ) -> ActionReceipt:
        payload_hash = _action_payload_hash(kind, identity.task_uid, expected_attempt_uid)
        with self._db.transaction() as tx:
            duplicate = _duplicate_action(
                tx,
                principal_id=principal_id,
                kind=kind,
                idempotency_key=idempotency_key,
                payload_hash=payload_hash,
            )
            if duplicate is not None:
                return duplicate
            task_id = JobName.from_wire(identity.key.resource_id)
            row = tx.execute(
                reads.task_detail_query().add_columns(tasks_table.c.task_index).where(tasks_table.c.task_id == task_id)
            ).first()
            if row is None:
                raise ResourceNotFound(identity.key.resource_id)
            job = self._job_rows(tx, {row.job_id})[row.job_id]
            authority = self._authority_cluster(job)
            current_identity = _resource_uid(ResourceKind.TASK, authority, row.task_id.to_wire(), row.submitted_at_ms)
            if identity.key.cluster_id != authority or identity.task_uid != current_identity:
                raise ResourceReplaced(identity.key.resource_id)
            attempt = reads.bulk_get_attempts(tx, [(row.task_id, int(row.current_attempt_id))]).get(
                (row.task_id, int(row.current_attempt_id))
            )
            if attempt is None:
                raise ActionPolicyRejected("Task has no current Attempt")
            if expected_attempt_number is not None and int(attempt.attempt_id) != expected_attempt_number:
                raise ResourceReplaced(f"{identity.key.resource_id}:{expected_attempt_number}")
            if str(attempt.attempt_uid) != expected_attempt_uid:
                raise ResourceReplaced(f"{identity.key.resource_id}:{attempt.attempt_id}")
            finalize(
                tx,
                [
                    TerminalDecision(
                        kind=terminal_kind,
                        task_id=row.task_id,
                        reason="Requested through the resource action API",
                    )
                ],
                now=Timestamp.now(),
            )
            target = identity.key
            if kind is ActionKind.TERMINATE_ATTEMPT:
                target = ResourceKey(
                    authority,
                    ResourceKind.ATTEMPT,
                    f"{row.task_id.to_wire()}:{attempt.attempt_id}",
                )
            receipt = _completed_action(
                kind=kind,
                target=target,
                expected_target_uid=identity.task_uid,
                expected_attempt_uid=expected_attempt_uid,
                result=result,
            )
            action_persistence.insert_action(
                tx,
                receipt,
                authority_cluster_id=authority,
                authority_action_id=receipt.action_id,
                backend_id=self._backend_id(str(row.backend_id)),
                execution_cluster_id=_execution_cluster(self._cluster_id, str(row.cluster)),
                principal_id=principal_id,
                idempotency_key=_require_idempotency_key(idempotency_key),
                payload_hash=payload_hash,
            )
            return receipt

    def _job_summary(
        self,
        status: job_pb2.JobStatus,
        authorities: Mapping[JobName, str],
        submitted: Mapping[JobName, Timestamp],
        parent_submitted: Mapping[JobName, Timestamp],
    ) -> JobSummary:
        job_id = JobName.from_wire(status.job_id)
        authority = authorities.get(job_id, self._cluster_id)
        submitted_at = submitted[job_id]
        parent = None
        if status.parent_job_id:
            parent_id = JobName.from_wire(status.parent_job_id)
            parent_time = parent_submitted.get(parent_id)
            if parent_time is not None:
                parent = JobIdentity(
                    ResourceKey(authority, ResourceKind.JOB, parent_id.to_wire()),
                    _resource_uid(ResourceKind.JOB, authority, parent_id.to_wire(), parent_time),
                )
        return JobSummary(
            identity=JobIdentity(
                ResourceKey(authority, ResourceKind.JOB, job_id.to_wire()),
                _resource_uid(ResourceKind.JOB, authority, job_id.to_wire(), submitted_at),
            ),
            owner_id=job_id.user,
            parent=parent,
            state=status.state,
            execution_cluster_id=_execution_cluster(self._cluster_id, status.cluster),
            backend_id=status.backend_id,
            num_tasks=status.task_count,
            submitted_at=submitted_at,
            started_at=timestamp_from_proto(status.started_at) if status.HasField("started_at") else None,
            finished_at=timestamp_from_proto(status.finished_at) if status.HasField("finished_at") else None,
            error_message=status.error,
        )

    def _task_summary(self, row, current_attempt, counts: AttemptCounts, job) -> TaskSummary:
        authority = self._authority_cluster(job)
        execution = _execution_cluster(self._cluster_id, str(row.cluster))
        task_key = ResourceKey(authority, ResourceKind.TASK, row.task_id.to_wire())
        task_identity = TaskIdentity(
            task_key,
            _resource_uid(ResourceKind.TASK, authority, row.task_id.to_wire(), row.submitted_at_ms),
        )
        attempt_identity = None
        node_identity = None
        backend_id = self._backend_id(str(row.backend_id))
        if current_attempt is not None:
            attempt_identity = AttemptIdentity(
                task_key, int(current_attempt.attempt_id), str(current_attempt.attempt_uid)
            )
            node_id = str(current_attempt.node_name or current_attempt.worker_id or "")
            if node_id:
                node_identity = NodeIdentity(
                    ResourceKey(execution, ResourceKind.NODE, node_id),
                    backend_id,
                    _resource_uid(ResourceKind.NODE, execution, f"{backend_id}:{node_id}", row.submitted_at_ms),
                )
        return TaskSummary(
            identity=task_identity,
            job=self._job_identity(job),
            task_index=int(row.task_index),
            state=row.state,
            execution_cluster_id=execution,
            backend_id=backend_id,
            current_attempt=attempt_identity,
            current_node=node_identity,
            failure_count=counts.failure_count,
            preemption_count=counts.preemption_count,
            submitted_at=row.submitted_at_ms,
            started_at=current_attempt.started_at_ms if current_attempt is not None else None,
            finished_at=current_attempt.finished_at_ms if current_attempt is not None else None,
            status_message=str(row.status_message or ""),
            error_message=str(row.error or ""),
        )

    def _attempt_summary(self, task, attempt, job) -> AttemptSummary:
        task_summary = self._task_summary(
            task, attempt if attempt.attempt_id == task.current_attempt_id else None, AttemptCounts(), job
        )
        execution = _execution_cluster(self._cluster_id, str(task.cluster))
        node_id = str(attempt.node_name or attempt.worker_id or "")
        node = None
        backend_id = self._backend_id(str(task.backend_id))
        if node_id:
            node = NodeIdentity(
                ResourceKey(execution, ResourceKind.NODE, node_id),
                backend_id,
                _resource_uid(ResourceKind.NODE, execution, f"{backend_id}:{node_id}", task.submitted_at_ms),
            )
        return AttemptSummary(
            identity=AttemptIdentity(task_summary.identity.key, int(attempt.attempt_id), str(attempt.attempt_uid)),
            state=attempt.state,
            execution_cluster_id=execution,
            backend_id=backend_id,
            node=node,
            created_at=attempt.created_at_ms,
            started_at=attempt.started_at_ms,
            finished_at=attempt.finished_at_ms,
            exit_code=attempt.exit_code,
            error_message=str(attempt.error or ""),
            terminal_reason=str(attempt.terminal_reason or ""),
        )

    def _job_identity(self, row) -> JobIdentity:
        authority = self._authority_cluster(row)
        return JobIdentity(
            ResourceKey(authority, ResourceKind.JOB, row.job_id.to_wire()),
            _resource_uid(ResourceKind.JOB, authority, row.job_id.to_wire(), row.submitted_at_ms),
        )

    def _authority_cluster(self, row) -> str:
        direction = getattr(row, "direction", None)
        if direction == int(FederationDirection.RECEIVED):
            return str(row.peer_id)
        return self._cluster_id

    def _backend_id(self, stored: str) -> str:
        if stored:
            if stored not in self._backends:
                raise BackendIdentityUnknown(stored)
            return stored
        if len(self._backends) == 1:
            return next(iter(self._backends))
        raise BackendIdentityUnknown("Task has no retained backend coordinate")

    def _endpoint_summary(self, row: EndpointRow, coordinates: tuple[str, str]) -> EndpointSummary:
        authority, execution = coordinates
        task = ResourceKey(authority, ResourceKind.TASK, row.task_id.to_wire())
        return EndpointSummary(
            key=ResourceKey(authority, ResourceKind.ENDPOINT, row.endpoint_id),
            endpoint_id=row.endpoint_id,
            name=row.name,
            task=task,
            execution_cluster_id=execution,
            access=(
                EndpointAccess.LINK
                if row.access == controller_pb2.Controller.ENDPOINT_ACCESS_LINK
                else EndpointAccess.PRIVATE
            ),
            lease_deadline=row.lease_deadline,
        )

    def _endpoint_coordinates(self, rows: list[EndpointRow]) -> dict[str, tuple[str, str]]:
        if not rows:
            return {}
        roots = {row.task_id.root_job for row in rows}
        with self._db.read_snapshot() as tx:
            jobs = self._job_rows(tx, roots)
        coordinates: dict[str, tuple[str, str]] = {}
        for row in rows:
            job = jobs.get(row.task_id.root_job)
            if job is None:
                raise ResourceNotFound(row.task_id.root_job.to_wire())
            coordinates[row.endpoint_id] = (
                self._authority_cluster(job),
                row.peer_id or self._cluster_id,
            )
        return coordinates

    def _job_rows(self, tx, job_ids: set[JobName]) -> dict[JobName, object]:
        if not job_ids:
            return {}
        rows = tx.execute(
            select(
                jobs_table.c.job_id,
                jobs_table.c.submitted_at_ms,
                federated_jobs_table.c.direction,
                federated_jobs_table.c.peer_id,
            )
            .select_from(
                jobs_table.outerjoin(
                    federated_jobs_table,
                    federated_jobs_table.c.job_id == jobs_table.c.job_id,
                )
            )
            .where(jobs_table.c.job_id.in_(job_ids))
        ).all()
        return {row.job_id: row for row in rows}

    def _job_authorities(self, job_ids) -> dict[JobName, str]:
        ids = set(job_ids)
        with self._db.read_snapshot() as tx:
            return {job_id: self._authority_cluster(row) for job_id, row in self._job_rows(tx, ids).items()}

    def _job_submission_times(self, job_ids) -> dict[JobName, Timestamp]:
        ids = set(job_ids)
        if not ids:
            return {}
        with self._db.read_snapshot() as tx:
            rows = tx.execute(
                select(jobs_table.c.job_id, jobs_table.c.submitted_at_ms).where(jobs_table.c.job_id.in_(ids))
            ).all()
        return {row.job_id: row.submitted_at_ms for row in rows}

    def _source_statuses(self) -> tuple[ResourceSourceStatus, ...]:
        statuses = [_available_source(f"controller:{self._cluster_id}")]
        for backend_id, backend in sorted(self._backends.items()):
            try:
                backend.status()
            except (ConnectionError, ProviderError) as exc:
                statuses.append(
                    ResourceSourceStatus(
                        source_id=f"backend:{backend_id}",
                        backend_id=backend_id,
                        state=SourceState.UNAVAILABLE,
                        freshness=Freshness.UNKNOWN,
                        observed_at=None,
                        error_code="backend_unavailable",
                        error_message=str(exc),
                    )
                )
            else:
                statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
        return tuple(statuses)


def _page_size(value: int, maximum: int) -> int:
    if value <= 0 or value > maximum:
        raise ValueError(f"page_size must be between 1 and {maximum}")
    return value


def _require_key(key: ResourceKey, kind: ResourceKind, cluster_id: str) -> None:
    if key.kind is not kind:
        raise ValueError(f"expected {kind.value}, got {key.kind.value}")
    if key.cluster_id != cluster_id:
        raise ResourceNotFound(key.resource_id)


def _stored_cluster(local_cluster_id: str, execution_cluster_id: str | None) -> str:
    if execution_cluster_id is None:
        return ""
    return LOCAL_CLUSTER if execution_cluster_id == local_cluster_id else execution_cluster_id


def _escaped_prefix(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def _require_idempotency_key(value: str) -> str:
    if not value.strip():
        raise ValueError("idempotency_key is required")
    return value


def _action_payload_hash(kind: ActionKind, target_uid: str, attempt_uid: str | None) -> str:
    encoded = json.dumps(
        {"kind": kind.value, "target_uid": target_uid, "attempt_uid": attempt_uid},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _duplicate_action(
    tx: Tx,
    *,
    principal_id: str,
    kind: ActionKind,
    idempotency_key: str,
    payload_hash: str,
) -> ActionReceipt | None:
    existing = action_persistence.action_by_idempotency_key(
        tx,
        principal_id=principal_id,
        kind=kind,
        idempotency_key=_require_idempotency_key(idempotency_key),
    )
    if existing is None:
        return None
    receipt, stored_hash = existing
    if stored_hash != payload_hash:
        raise ActionIdempotencyConflict("idempotency key was already used for a different action")
    return receipt


def _completed_action(
    *,
    kind: ActionKind,
    target: ResourceKey,
    expected_target_uid: str,
    expected_attempt_uid: str | None,
    result: ActionResult,
) -> ActionReceipt:
    now = Timestamp.now()
    return ActionReceipt(
        action_id=uuid.uuid4().hex,
        kind=kind,
        target=target,
        expected_target_uid=expected_target_uid,
        expected_attempt_uid=expected_attempt_uid,
        state=ActionState.SUCCEEDED,
        result_code=result,
        result_message="",
        created_at=now,
        updated_at=now,
        completed_at=now,
    )


def _job_spec(request: controller_pb2.Controller.LaunchJobRequest) -> JobSpec:
    resources = ResourceSpec(
        cpu=request.resources.cpu_millicores / 1_000,
        memory=request.resources.memory_bytes,
        disk=request.resources.disk_bytes,
        device=request.resources.device if request.resources.HasField("device") else None,
    )
    coscheduling = (
        CoschedulingConfig(group_by=request.coscheduling.group_by) if request.HasField("coscheduling") else None
    )
    return JobSpec(
        version=1,
        name=request.name,
        entrypoint=request.entrypoint,
        resources=resources,
        environment=request.environment,
        bundle_id=request.bundle_id,
        scheduling_timeout=(
            duration_from_proto(request.scheduling_timeout) if request.HasField("scheduling_timeout") else None
        ),
        ports=tuple(request.ports),
        max_task_failures=request.max_task_failures,
        max_retries_failure=request.max_retries_failure,
        max_retries_preemption=request.max_retries_preemption,
        constraints=tuple(Constraint.from_proto(candidate) for candidate in request.constraints),
        coscheduling=coscheduling,
        replicas=request.replicas,
        timeout=duration_from_proto(request.timeout) if request.HasField("timeout") else None,
        fail_if_exists=request.fail_if_exists,
        preemption_policy=request.preemption_policy,
        existing_job_policy=request.existing_job_policy,
        priority_band=request.priority_band,
        task_image=request.task_image,
        submit_argv=tuple(request.submit_argv),
        client_revision_date=request.client_revision_date,
        container_profile=request.container_profile,
    )


def _launch_request(spec: JobSpec, bundle_blob: bytes) -> controller_pb2.Controller.LaunchJobRequest:
    request = controller_pb2.Controller.LaunchJobRequest(
        name=spec.name,
        entrypoint=spec.entrypoint,
        resources=spec.resources.to_proto(),
        environment=spec.environment,
        bundle_id=spec.bundle_id,
        bundle_blob=bundle_blob,
        ports=spec.ports,
        max_task_failures=spec.max_task_failures,
        max_retries_failure=spec.max_retries_failure,
        max_retries_preemption=spec.max_retries_preemption,
        constraints=[constraint.to_proto() for constraint in spec.constraints],
        replicas=spec.replicas,
        fail_if_exists=spec.fail_if_exists,
        preemption_policy=spec.preemption_policy,
        existing_job_policy=spec.existing_job_policy,
        priority_band=spec.priority_band,
        task_image=spec.task_image,
        submit_argv=spec.submit_argv,
        client_revision_date=spec.client_revision_date,
        container_profile=spec.container_profile,
    )
    if spec.scheduling_timeout is not None:
        request.scheduling_timeout.CopyFrom(duration_to_proto(spec.scheduling_timeout))
    if spec.coscheduling is not None:
        request.coscheduling.CopyFrom(spec.coscheduling.to_proto())
    if spec.timeout is not None:
        request.timeout.CopyFrom(duration_to_proto(spec.timeout))
    return request
