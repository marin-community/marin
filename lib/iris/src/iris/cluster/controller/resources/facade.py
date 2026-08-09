# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

import base64
import hashlib
import json
import logging
import secrets
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from connectrpc.errors import ConnectError
from finelog.client import LogClient
from finelog.errors import StatsError
from rigging.connect import capability_path, federated_capability_path
from rigging.timing import Duration, Timestamp
from sqlalchemy import Row, and_, bindparam, func, or_, select

from iris.cluster.authorization import authorize_resource_owner
from iris.cluster.bundle import BundleStore
from iris.cluster.config import BackendConfig
from iris.cluster.controller import reads, writes
from iris.cluster.controller.attempt_counts import AttemptCounts
from iris.cluster.controller.auth import (
    DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS,
    MAX_ENDPOINT_TOKEN_TTL_SECONDS,
    ControllerAuth,
)
from iris.cluster.controller.backend import BackendCapability, ProviderError, TaskBackend, TaskTarget
from iris.cluster.controller.codec import (
    decode_attribute_value,
    reconstruct_job_spec,
)
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
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
from iris.cluster.controller.resources.jobs import FederationSubmission, JobResources
from iris.cluster.controller.resources.logs import fetch_log_entries
from iris.cluster.controller.resources.observations import (
    observe_autoscaler_resources,
    observe_backend_resources,
    worker_node_metadata,
)
from iris.cluster.controller.schema import (
    federated_jobs_table,
    job_config_table,
    jobs_table,
    task_attempts_table,
    tasks_table,
)
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.federation.manager import FederationManager, FederationPeerObservation
from iris.cluster.federation.store import CancelTarget, FederationDirection, HandoffState
from iris.cluster.log_highlights import extract_failure_highlights
from iris.cluster.log_keys import build_log_source
from iris.cluster.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.cluster.resources.activity import ActivityEntry, ActivityQuery
from iris.cluster.resources.attempt import AttemptDetail, AttemptRuntimeObject, AttemptSummary
from iris.cluster.resources.endpoint import (
    EndpointAccess,
    EndpointDetail,
    EndpointQuery,
    EndpointSummary,
    EndpointToken,
    ExecRequest,
    ExecResult,
    ProfileConfiguration,
    ProfileRequest,
    ProfileResult,
)
from iris.cluster.resources.errors import (
    ActionIdempotencyConflict,
    ActionPolicyRejected,
    BackendIdentityUnknown,
    InvalidPageToken,
    InvalidResourceKey,
    ResourceNotFound,
    ResourceReplaced,
    ResourceSourceUnavailable,
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
from iris.cluster.resources.job import (
    FederationPosture,
    JobDetail,
    JobObservation,
    JobQuery,
    JobSpec,
    JobSummary,
    JobTaskAggregate,
    TaskStateCount,
)
from iris.cluster.resources.log import LogPage, LogQuery
from iris.cluster.resources.node import (
    NodeAttribute,
    NodeAttributeKind,
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
from iris.cluster.resources.state import JobState, TaskState
from iris.cluster.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.cluster.stats.tables import TASK_EVENT_NAMESPACE
from iris.cluster.types import (
    DEFAULT_BACKEND_ID,
    LOCAL_ADMIN_SUBMITTER,
    LOCAL_CLUSTER,
    JobName,
    UserBudgetDefaults,
    WorkerId,
)

_RESOURCE_UID_NAMESPACE = uuid.UUID("2c72b7f4-a156-5d27-8b58-7de28d5ec4cc")
_RESOURCE_UID_PREFIX = "iris-resource-v2"
_MAX_JOB_PAGE = 500
_MAX_JOB_STATE_BATCH = 32_767
_MAX_TASK_PAGE = 500
_MAX_ENDPOINT_PAGE = 500
_MAX_ACTIVITY_PAGE = 500
_MAX_NODE_RECENT_ATTEMPTS = 50
_NODE_WORKER_SCAN_BATCH = _MAX_TASK_PAGE + 1
_BACKEND_UNAVAILABLE = "backend_unavailable"
_PEER_UNAVAILABLE = "peer_unavailable"
_FINELOG_UNAVAILABLE = "finelog_unavailable"
_SOURCE_UNSUPPORTED = "unsupported"
_FINELOG_NOT_CONFIGURED = "finelog is not configured"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _NodeDetails:
    address: str | None
    attributes: tuple[NodeAttribute, ...]


@dataclass(frozen=True, slots=True)
class _ProviderNodeCandidate:
    summary: NodeSummary
    details: _NodeDetails


@dataclass(frozen=True, slots=True)
class _WorkerNodeCandidate:
    backend_id: str
    worker: Row
    liveness: WorkerLiveness


_NodeCandidate = _ProviderNodeCandidate | _WorkerNodeCandidate


@dataclass(frozen=True, slots=True)
class _ProviderNodeSnapshot:
    candidates: tuple[_ProviderNodeCandidate, ...]
    source_statuses: tuple[ResourceSourceStatus, ...]


@dataclass(frozen=True, slots=True)
class _SliceSnapshot:
    slices: tuple[SliceSummary, ...]
    members: Mapping[tuple[str, str], tuple[SliceMember, ...]]
    source_statuses: tuple[ResourceSourceStatus, ...]


@dataclass(frozen=True, slots=True)
class _FailureHighlights:
    entries: tuple[str, ...]
    source_status: ResourceSourceStatus | None


@dataclass(frozen=True, slots=True)
class _ActivityItem:
    entry: ActivityEntry
    source_rank: int
    source_key: tuple[int | str, ...]

    @property
    def order_key(self) -> tuple[int, int, tuple[int | str, ...]]:
        return (self.entry.occurred_at.epoch_ms(), self.source_rank, self.source_key)


class ResourceRuntime(Protocol):
    """Controller capabilities needed by resource operations."""

    @property
    def backends(self) -> dict[str, TaskBackend]: ...

    @property
    def federation(self) -> FederationManager: ...

    @property
    def capabilities(self) -> frozenset[BackendCapability]: ...

    def wake(self) -> None: ...

    def liveness_for_worker(self, worker_id: WorkerId) -> WorkerLiveness: ...

    def backend_id_for_scale_group(self, scale_group: str) -> str: ...

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class CapabilityUrlConfig:
    """Origins used to construct endpoint capability URLs."""

    cluster_name: str = ""
    local_origin: str = ""
    parent_origin: str = ""

    def build(self, name: str, token: str) -> str:
        if self.parent_origin and self.cluster_name:
            return f"{self.parent_origin.rstrip('/')}{federated_capability_path(self.cluster_name, name, token)}"
        if self.local_origin:
            return f"{self.local_origin.rstrip('/')}{capability_path(name, token)}"
        return ""


@dataclass(frozen=True, slots=True)
class _JobCoordinates:
    job_id: JobName
    submitted_at_ms: Timestamp
    direction: int | None
    peer_id: str | None
    handoff_state: int | None
    handoff_nonce: str | None


def _uid(kind: ResourceKind, *parts: object) -> str:
    name = "\0".join((_RESOURCE_UID_PREFIX, kind.value, *(str(part) for part in parts)))
    return str(uuid.uuid5(_RESOURCE_UID_NAMESPACE, name))


def _job_uid(
    cluster_id: str,
    job_id: JobName,
    submitted_at: Timestamp,
    *,
    handoff_nonce: str = "",
) -> str:
    incarnation = handoff_nonce if job_id.is_root and handoff_nonce else submitted_at.epoch_ms()
    return _uid(ResourceKind.JOB, cluster_id, job_id.to_wire(), incarnation)


def _task_uid(job_uid: str, task_id: JobName) -> str:
    _, task_index = task_id.require_task()
    return _uid(ResourceKind.TASK, job_uid, task_index)


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


def _task_event_source_key(row: Mapping[str, object]) -> tuple[int, str, str, str, str, str, int]:
    return _validated_task_event_source_key(
        (
            row["attempt_id"],
            row["attempt_uid"],
            row["type"],
            row["reason"],
            row["message"],
            row["source"],
            row["count"],
        )
    )


def _validated_task_event_source_key(
    values: tuple[int | str, ...],
) -> tuple[int, str, str, str, str, str, int]:
    if (
        len(values) != 7
        or type(values[0]) is not int
        or not all(isinstance(value, str) for value in values[1:6])
        or type(values[6]) is not int
    ):
        raise InvalidPageToken("malformed task-event activity position")
    return (
        cast(int, values[0]),
        cast(str, values[1]),
        cast(str, values[2]),
        cast(str, values[3]),
        cast(str, values[4]),
        cast(str, values[5]),
        cast(int, values[6]),
    )


def _sql_key_before(columns: tuple[str, ...], values: tuple[int | str, ...]) -> str:
    terms = []
    for index, (column, value) in enumerate(zip(columns, values, strict=True)):
        equal_prefix = " AND ".join(
            f"{prefix_column} = {_sql_literal(prefix_value)}"
            for prefix_column, prefix_value in zip(columns[:index], values[:index], strict=True)
        )
        comparison = f"{column} < {_sql_literal(value)}"
        terms.append(f"({equal_prefix} AND {comparison})" if equal_prefix else comparison)
    return " OR ".join(terms)


def _sql_literal(value: int | str) -> str:
    if type(value) is int:
        return str(value)
    return f"'{value.replace("'", "''")}'"


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
        error_code=_BACKEND_UNAVAILABLE,
        error_message=str(error)[:MAX_SOURCE_ERROR_MESSAGE],
    )


def _unavailable_finelog_source(cluster_id: str, error: Exception) -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=f"finelog:{cluster_id}",
        backend_id="",
        state=SourceState.UNAVAILABLE,
        freshness=Freshness.UNKNOWN,
        observed_at=None,
        error_code=_FINELOG_UNAVAILABLE,
        error_message=str(error)[:MAX_SOURCE_ERROR_MESSAGE],
    )


def _unsupported_source(source_id: str, *, backend_id: str = "") -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=source_id,
        backend_id=backend_id,
        state=SourceState.UNSUPPORTED,
        freshness=Freshness.UNKNOWN,
        observed_at=None,
        error_code=_SOURCE_UNSUPPORTED,
        error_message="",
    )


def _opaque_uid(value: str) -> str:
    return uuid.uuid5(_RESOURCE_UID_NAMESPACE, value).hex


def _string_node_attribute(key: str, value: str) -> NodeAttribute | None:
    if not value:
        return None
    return NodeAttribute(key=key, kind=NodeAttributeKind.STRING, string_value=value)


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
        runtime: ResourceRuntime,
        bundle_store: BundleStore,
        endpoint_service: EndpointServiceImpl,
        auth: ControllerAuth,
        user_budget_defaults: UserBudgetDefaults,
        capability_url_config: CapabilityUrlConfig,
        backends: Mapping[str, TaskBackend],
        backend_configs: Mapping[str, BackendConfig],
        log_client: LogClient | None = None,
    ) -> None:
        if not cluster_id.strip():
            raise ValueError("cluster_id is required for resource identities")
        self._cluster_id = cluster_id
        self._db = db
        self._runtime = runtime
        self._jobs = JobResources(
            db=db,
            runtime=runtime,
            bundle_store=bundle_store,
            auth=auth,
            user_budget_defaults=user_budget_defaults,
        )
        self._endpoint_service = endpoint_service
        self._auth = auth
        self._capability_url_config = capability_url_config
        self._backends = dict(backends)
        if backend_configs.keys() != self._backends.keys():
            raise ValueError("backend_configs keys must exactly match live backend keys")
        self._backend_configs = dict(backend_configs)
        self._log_client = log_client

    @property
    def cluster_id(self) -> str:
        return self._cluster_id

    def received_job_from_peer(self, root_job: JobName, peer_id: str) -> bool:
        """Whether ``root_job`` is a handoff received from ``peer_id``."""
        with self._db.read_snapshot() as tx:
            return reads.has_received_job_from_peer(tx, peer_id, root_job)

    def submit_job(
        self,
        spec: JobSpec,
        bundle_blob: bytes = b"",
        *,
        enforce_client_freshness: bool = True,
    ) -> JobIdentity:
        job_id = self._jobs.submit(
            spec,
            bundle_blob,
            enforce_client_freshness=enforce_client_freshness,
        )
        authority = self._job_authorities({job_id})[job_id]
        key = ResourceKey(authority, ResourceKind.JOB, job_id.to_wire())
        return self.describe_job(key).summary.identity

    def submit_federated_job(
        self,
        spec: JobSpec,
        bundle_blob: bytes,
        federation: FederationSubmission,
    ) -> JobIdentity:
        job_id = self._jobs.submit(
            spec,
            bundle_blob,
            federation=federation,
            enforce_client_freshness=False,
        )
        key = ResourceKey(federation.requester_id, ResourceKind.JOB, job_id.to_wire())
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
        stmt = (
            select(
                *jobs_table.c,
                *job_config_table.c,
                federated_jobs_table.c.direction,
                federated_jobs_table.c.peer_id,
                federated_jobs_table.c.handoff_state,
                federated_jobs_table.c.handoff_nonce,
            )
            .select_from(
                jobs_table.join(job_config_table, job_config_table.c.job_id == jobs_table.c.job_id).outerjoin(
                    federated_jobs_table,
                    federated_jobs_table.c.job_id == jobs_table.c.root_job_id,
                )
            )
            .order_by(jobs_table.c.submitted_at_ms.desc(), jobs_table.c.job_id.asc())
            .offset(offset)
            .limit(page_size + 1)
        )
        if query.owner_id is not None:
            stmt = stmt.where(jobs_table.c.user_id == query.owner_id)
        if query.job_id_prefix is not None:
            stmt = stmt.where(jobs_table.c.job_id.like(_escaped_prefix(query.job_id_prefix), escape="\\"))
        if query.parent is not None:
            _require_kind(query.parent, ResourceKind.JOB)
            stmt = stmt.where(jobs_table.c.parent_job_id == JobName.from_wire(query.parent.resource_id))
        if query.states:
            stmt = stmt.where(jobs_table.c.state.in_(query.states))
        if query.backend_id is not None:
            stmt = stmt.where(jobs_table.c.backend_id == query.backend_id)
        if query.execution_cluster_id is not None:
            stmt = stmt.where(jobs_table.c.cluster == _stored_cluster(self._cluster_id, query.execution_cluster_id))
        with self._db.read_snapshot() as tx:
            rows = tx.execute(stmt).all()
            parent_coordinates = self._job_coordinates_in_snapshot(
                tx,
                {row.parent_job_id for row in rows[:page_size] if row.parent_job_id is not None},
            )
        page_rows = rows[:page_size]
        items = tuple(self._job_summary_from_row(row, parent_coordinates=parent_coordinates) for row in page_rows)
        if query.parent is not None:
            items = tuple(item for item in items if item.parent is not None and item.parent.key == query.parent)
        next_token = None
        if len(rows) > page_size:
            next_token = _encode_page_token(fingerprint, {"offset": offset + len(page_rows)})
        return Page(
            items=items,
            next_page_token=next_token,
            source_statuses=self._source_statuses(),
        )

    def describe_job(self, key: ResourceKey) -> JobDetail:
        _require_kind(key, ResourceKind.JOB)
        job_id = JobName.from_wire(key.resource_id)
        with self._db.read_snapshot() as tx:
            row = reads.get_job_detail(tx, job_id)
            coordinates = self._job_rows(tx, {job_id}).get(job_id)
            workdir_files = reads.get_workdir_files(tx, job_id) if row is not None else {}
            parent_coordinates = self._job_coordinates_in_snapshot(
                tx,
                {row.parent_job_id} if row is not None and row.parent_job_id is not None else set(),
            )
        if row is None or coordinates is None:
            raise ResourceNotFound(key.resource_id)
        summary = self._job_summary_from_row(
            row,
            coordinates=coordinates,
            parent_coordinates=parent_coordinates,
        )
        if summary.identity.key.cluster_id != key.cluster_id:
            raise ResourceNotFound(key.resource_id)
        return JobDetail(summary=summary, spec=reconstruct_job_spec(row, workdir_files=workdir_files))

    def job_states(self, resource_ids: Sequence[str]) -> dict[str, JobState]:
        """Return exact current states for a bounded set of Job IDs."""
        if len(resource_ids) > _MAX_JOB_STATE_BATCH:
            raise ValueError(f"Job state batch cannot exceed {_MAX_JOB_STATE_BATCH} items")
        if not resource_ids:
            return {}
        normalized = [JobName.from_wire(resource_id).to_wire() for resource_id in resource_ids]
        requested = func.json_each(bindparam("job_ids_json")).table_valued("value").alias("requested_job_ids")
        with self._db.read_snapshot() as tx:
            rows = tx.execute(
                select(requested.c.value.label("resource_id"), jobs_table.c.state).select_from(
                    requested.join(jobs_table, jobs_table.c.job_id == requested.c.value)
                ),
                {"job_ids_json": json.dumps(normalized)},
            ).all()
        return {str(row.resource_id): JobState(int(row.state)) for row in rows}

    def observe_jobs(self, summaries: Sequence[JobSummary]) -> tuple[JobObservation, ...]:
        """Read bounded Task, child, and federation aggregates for Jobs in one snapshot."""
        if len(summaries) > _MAX_JOB_PAGE:
            raise ValueError(f"Job observation batch cannot exceed {_MAX_JOB_PAGE} items")
        if not summaries:
            return ()
        job_ids = [JobName.from_wire(summary.identity.key.resource_id) for summary in summaries]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("Job observation keys must be unique")
        with self._db.read_snapshot() as tx:
            attempt_counts = reads.attempt_counts_for_jobs(tx, job_ids)
            task_aggregates = reads.task_summaries_for_jobs(tx, job_ids, attempt_counts=attempt_counts)
            parents = reads.parent_ids_with_children(tx, job_ids)
            handoff_states = reads.handoff_states(tx, job_ids)
        observations = []
        for summary, job_id in zip(summaries, job_ids, strict=True):
            aggregate = task_aggregates.get(job_id, reads.TaskJobSummary(job_id=job_id))
            observations.append(
                JobObservation(
                    summary=summary,
                    tasks=JobTaskAggregate(
                        task_count=aggregate.task_count,
                        completed_count=aggregate.completed_count,
                        failure_count=aggregate.failure_count,
                        preemption_count=aggregate.preemption_count,
                        state_counts=tuple(
                            TaskStateCount(state=TaskState(state), count=count)
                            for state, count in sorted(aggregate.task_state_counts.items())
                        ),
                    ),
                    has_children=job_id in parents,
                    federation_posture=self._federation_posture(summary, handoff_states.get(job_id)),
                )
            )
        return tuple(observations)

    def _federation_posture(self, summary: JobSummary, handoff_state: int | None) -> FederationPosture:
        if handoff_state == int(HandoffState.QUEUED_HANDOFF):
            return FederationPosture.QUEUED
        if handoff_state == int(HandoffState.PENDING_HANDOFF):
            return FederationPosture.PENDING_ACCEPTANCE
        if handoff_state == int(HandoffState.HANDOFF_REJECTED):
            return FederationPosture.REJECTED
        if summary.execution_cluster_id == self._cluster_id:
            return FederationPosture.LOCAL
        return FederationPosture.ACCEPTED

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
                _require_kind(query.job, ResourceKind.JOB)
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
        if query.job is not None:
            items = tuple(item for item in items if item.job.key == query.job)
        next_token = None
        if len(rows) > page_size and page_rows:
            last = page_rows[-1]
            next_token = _encode_page_token(
                fingerprint,
                {"submitted_at_ms": last.submitted_at_ms.epoch_ms(), "task_id": last.task_id.to_wire()},
            )
        return Page(items=items, next_page_token=next_token, source_statuses=self._source_statuses())

    def describe_task(self, key: ResourceKey) -> TaskDetail:
        return self._describe_tasks((key,), include_failure_highlights=True)[0]

    def describe_tasks(self, keys: Sequence[ResourceKey]) -> tuple[TaskDetail, ...]:
        """Describe a bounded Task collection without per-Task finelog enrichment."""
        return self._describe_tasks(keys, include_failure_highlights=False)

    def _describe_tasks(
        self,
        keys: Sequence[ResourceKey],
        *,
        include_failure_highlights: bool,
    ) -> tuple[TaskDetail, ...]:
        if not keys:
            return ()
        if len(keys) > _MAX_TASK_PAGE:
            raise ValueError(f"Task detail batch cannot exceed {_MAX_TASK_PAGE} items")
        for key in keys:
            _require_kind(key, ResourceKind.TASK)
        task_ids = [JobName.from_wire(key.resource_id) for key in keys]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Task detail keys must be unique")
        with self._db.read_snapshot() as tx:
            rows = tx.execute(
                reads.task_detail_query()
                .add_columns(tasks_table.c.task_index)
                .where(tasks_table.c.task_id.in_(bindparam("task_ids", expanding=True))),
                {"task_ids": task_ids},
            ).all()
            rows_by_id = {row.task_id: row for row in rows}
            attempts_by_task = reads.all_attempts_for_tasks(tx, task_ids)
            counts_by_task = reads.attempt_counts_for_tasks(tx, task_ids)
            jobs = self._job_rows(tx, {row.job_id for row in rows})
        source_statuses = self._source_statuses()
        details = []
        for key, task_id in zip(keys, task_ids, strict=True):
            row = rows_by_id.get(task_id)
            if row is None:
                raise ResourceNotFound(key.resource_id)
            attempts = attempts_by_task.get(task_id, ())
            current = next((candidate for candidate in attempts if candidate.attempt_id == row.current_attempt_id), None)
            job = jobs[row.job_id]
            summary = self._task_summary(row, current, counts_by_task.get(task_id, AttemptCounts()), job)
            if summary.identity.key.cluster_id != key.cluster_id:
                raise ResourceNotFound(key.resource_id)
            failure_highlights = (
                self._failure_highlights(task_id, summary.state)
                if include_failure_highlights
                else _FailureHighlights((), None)
            )
            detail_sources = source_statuses
            if failure_highlights.source_status is not None:
                detail_sources += (failure_highlights.source_status,)
            details.append(
                TaskDetail(
                    summary=summary,
                    attempts=tuple(self._attempt_summary(row, candidate, job) for candidate in attempts),
                    source_statuses=detail_sources,
                    root_cause_highlights=failure_highlights.entries,
                )
            )
        return tuple(details)

    def _failure_highlights(self, task_id: JobName, state: TaskState) -> _FailureHighlights:
        if state not in (TaskState.FAILED, TaskState.WORKER_FAILED):
            return _FailureHighlights((), None)
        if self._log_client is None:
            status = _unavailable_finelog_source(self._cluster_id, RuntimeError(_FINELOG_NOT_CONFIGURED))
            return _FailureHighlights((), status)
        source, match_scope = build_log_source(task_id)
        try:
            entries, _ = fetch_log_entries(
                self._log_client,
                source=source,
                match_scope=match_scope,
                query=LogQuery(max_lines=200, tail=True),
            )
        except (ConnectError, ConnectionError, OSError, RuntimeError) as exc:
            logger.warning("Finelog unavailable while reading failure highlights for %s: %s", task_id, exc)
            return _FailureHighlights((), _unavailable_finelog_source(self._cluster_id, exc))
        return _FailureHighlights(
            tuple(extract_failure_highlights([entry.data for entry in entries])),
            _available_source(f"finelog:{self._cluster_id}"),
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
        task_id = JobName.from_wire(locator.task.resource_id)
        with self._db.read_snapshot() as tx:
            attempt_row = reads.bulk_get_attempts(tx, [(task_id, number)]).get((task_id, number))
            task_row = tx.execute(
                select(tasks_table.c.current_attempt_id, tasks_table.c.container_id).where(
                    tasks_table.c.task_id == task_id
                )
            ).first()
        if attempt_row is None or task_row is None:
            raise ResourceNotFound(f"{locator.task.resource_id}:{number}")
        container_id = str(task_row.container_id or "") if number == task_row.current_attempt_id else ""
        return AttemptDetail(
            summary=attempt,
            runtime=self._attempt_runtime(
                attempt_row,
                attempt.execution_cluster_id,
                attempt.backend_id,
                container_id,
            ),
            source_statuses=detail.source_statuses,
        )

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
        principal_id: str = LOCAL_ADMIN_SUBMITTER,
    ) -> ActionReceipt:
        payload_hash = _action_payload_hash(ActionKind.CANCEL_JOB, identity.job_uid, None)
        cancel_target: CancelTarget | None = None
        peer_id: str | None = None
        execution_cluster_id = ""
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
            coordinates = self._job_rows(tx, {row.job_id})[row.job_id]
            authority = self._authority_cluster(coordinates)
            expected = self._job_identity(coordinates).job_uid
            if identity.key.cluster_id != authority or identity.job_uid != expected:
                raise ResourceReplaced(identity.key.resource_id)
            execution_cluster_id = _execution_cluster(self._cluster_id, row.cluster)
            handle = reads.federated_handle(tx, row.job_id.root_job)
            if handle is None:
                job_ops.cancel(tx, job_id=row.job_id, reason="Cancelled by resource action")
                writes.record_federation_change(tx, row.job_id)
            elif row.job_id != row.job_id.root_job:
                peer_id = handle.peer_id
            else:
                writes.bump_cancel_intent(tx, row.job_id)
                if handle.handoff_state in {
                    int(HandoffState.QUEUED_HANDOFF),
                    int(HandoffState.PENDING_HANDOFF),
                }:
                    writes.mark_federated_job_killed(
                        tx,
                        row.job_id,
                        now_ms=Timestamp.now().epoch_ms(),
                        error="Cancelled before handoff",
                    )
                if handle.handoff_state != int(HandoffState.QUEUED_HANDOFF):
                    cancel_target = CancelTarget(row.job_id, handle.peer_id)
            if peer_id is None:
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
                    execution_cluster_id=execution_cluster_id,
                    principal_id=principal_id,
                    idempotency_key=_require_idempotency_key(idempotency_key),
                    payload_hash=payload_hash,
                )
        if peer_id is not None:
            receipt = self._runtime.federation.proxy_to_peer(
                peer_id,
                lambda peer: peer.cancel_job(identity, idempotency_key=idempotency_key),
            )
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
                action_persistence.insert_action(
                    tx,
                    receipt,
                    authority_cluster_id=authority,
                    authority_action_id=receipt.action_id,
                    backend_id="",
                    execution_cluster_id=execution_cluster_id,
                    principal_id=principal_id,
                    idempotency_key=_require_idempotency_key(idempotency_key),
                    payload_hash=payload_hash,
                )
            return receipt
        if cancel_target is not None:
            self._runtime.federation.deliver_cancel(cancel_target)
        else:
            self._runtime.wake()
        return receipt

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
        principal_id: str = LOCAL_ADMIN_SUBMITTER,
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
        principal_id: str = LOCAL_ADMIN_SUBMITTER,
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
        page_size = _page_size(query.page_size, _MAX_ENDPOINT_PAGE)
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
        system_endpoints = ()
        if query.task is None:
            system_endpoints = tuple(
                (name, address)
                for name, address in self._endpoint_service.system_endpoints()
                if query.name_prefix is None or name.startswith(query.name_prefix)
            )
        entries: list[tuple[str, str, EndpointRow | None, str]] = [(row.name, row.endpoint_id, row, "") for row in rows]
        entries.extend((name, name, None, address) for name, address in system_endpoints)
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        if position is not None:
            last_key = (str(position["name"]), str(position["endpoint_id"]))
            entries = [entry for entry in entries if (entry[0], entry[1]) > last_key]
        page_entries = entries[:page_size]
        page_rows = [row for _, _, row, _ in page_entries if row is not None]
        coordinates = self._endpoint_coordinates(page_rows)
        peer_ids = {execution for authority, execution in coordinates.values() if execution != self._cluster_id}
        next_token = None
        if len(entries) > page_size:
            last_name, last_endpoint_id, _, _ = page_entries[-1]
            next_token = _encode_page_token(
                fingerprint,
                {"name": last_name, "endpoint_id": last_endpoint_id},
            )
        return Page(
            items=tuple(
                (
                    self._endpoint_summary(row, coordinates[row.endpoint_id])
                    if row is not None
                    else self._system_endpoint_summary(name)
                )
                for name, _, row, _ in page_entries
            ),
            next_page_token=next_token,
            source_statuses=(
                _available_source(f"controller:{self._cluster_id}"),
                *self._peer_source_statuses(peer_ids),
            ),
        )

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail:
        return self.describe_endpoints((key,))[0]

    def describe_endpoints(self, keys: Sequence[ResourceKey]) -> tuple[EndpointDetail, ...]:
        """Return details for a bounded sequence of Endpoint keys."""
        if len(keys) > _MAX_ENDPOINT_PAGE:
            raise ValueError(f"Endpoint detail batch cannot exceed {_MAX_ENDPOINT_PAGE} items")
        for key in keys:
            _require_kind(key, ResourceKind.ENDPOINT)

        system_endpoints = dict(self._endpoint_service.system_endpoints())
        endpoint_ids = tuple(key.resource_id for key in keys if key.resource_id not in system_endpoints)
        rows = self._db.caches[EndpointsProjection].query(ProjectionEndpointQuery(endpoint_ids=endpoint_ids))
        rows_by_id = {row.endpoint_id: row for row in rows}
        coordinates = self._endpoint_coordinates(rows)
        details: list[EndpointDetail] = []
        for key in keys:
            system_address = system_endpoints.get(key.resource_id)
            if system_address is not None and key.cluster_id == self._cluster_id:
                details.append(
                    EndpointDetail(
                        summary=self._system_endpoint_summary(key.resource_id),
                        address=system_address,
                        metadata={},
                    )
                )
                continue
            row = rows_by_id.get(key.resource_id)
            if row is None:
                raise ResourceNotFound(key.resource_id)
            row_coordinates = coordinates[row.endpoint_id]
            if row_coordinates[0] != key.cluster_id:
                raise ResourceNotFound(key.resource_id)
            details.append(
                EndpointDetail(
                    summary=self._endpoint_summary(row, row_coordinates),
                    address=row.address,
                    metadata=dict(row.metadata),
                )
            )
        return tuple(details)

    def mint_endpoint_token(self, key: ResourceKey, ttl: Duration | None) -> EndpointToken:
        detail = self.describe_endpoint(key)
        if self._auth.jwt_manager is None:
            raise RuntimeError("JWT manager not configured")
        row = self._endpoint_service.resolve_task_endpoint(detail.summary.name)
        if row is None:
            raise ResourceNotFound(detail.summary.name)
        if self._auth.provider:
            authorize_resource_owner(row.task_id.user)
        ttl_seconds = DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS
        if ttl is not None:
            ttl_seconds = max(1, min(int(ttl.to_seconds()), MAX_ENDPOINT_TOKEN_TTL_SECONDS))
        now = Timestamp.now()
        expires_at = Timestamp.from_ms(now.epoch_ms() + ttl_seconds * 1_000)
        token = self._auth.jwt_manager.create_endpoint_token(
            row.name,
            f"iris_ket_{secrets.token_urlsafe(8)}",
            ttl_seconds=ttl_seconds,
        )
        return EndpointToken(
            token=token,
            expires_at=expires_at,
            capability_url=self._capability_url_config.build(row.name, token),
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
                source_statuses=(_unavailable_finelog_source(self._cluster_id, RuntimeError(_FINELOG_NOT_CONFIGURED)),),
            )
        source, match_scope = build_log_source(job_name, attempt_number)
        try:
            entries, next_cursor = fetch_log_entries(
                self._log_client,
                source=source,
                match_scope=match_scope,
                query=query,
            )
        except (ConnectError, ConnectionError, OSError, RuntimeError) as exc:
            return LogPage(
                entries=(),
                next_cursor=query.cursor,
                source_statuses=(_unavailable_finelog_source(self._cluster_id, exc),),
            )
        return LogPage(
            entries=entries,
            next_cursor=next_cursor,
            source_statuses=(_available_source(f"finelog:{self._cluster_id}"),),
        )

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]:
        page_size = _page_size(query.page_size, _MAX_ACTIVITY_PAGE)
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
        before_time = None
        before_source_rank = None
        before_source_key: tuple[int | str, ...] = ()
        if position is not None:
            try:
                before_time = Timestamp.from_ms(int(position["occurred_at_ms"]))
                before_source_rank = int(position["source_rank"])
                source_key = position["source_key"]
                if before_source_rank not in (0, 1) or not isinstance(source_key, list):
                    raise ValueError
                before_source_key = tuple(source_key)
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidPageToken("malformed activity page position") from exc
        attempt_uids = self._activity_attempt_uids(query)
        action_before = None
        if before_time is not None:
            if before_source_rank == 0:
                if len(before_source_key) != 1 or not isinstance(before_source_key[0], str):
                    raise InvalidPageToken("malformed action activity position")
                action_before = (before_time, f"action:{before_source_key[0]}")
            else:
                action_before = (before_time, "\U0010ffff")
        with self._db.read_snapshot() as tx:
            receipts = action_persistence.actions_for_target(
                tx,
                query.target,
                after=query.after,
                before=action_before,
                limit=page_size + 1,
            )
        entries = [_ActivityItem(self._action_activity(receipt), 0, (receipt.action_id,)) for receipt in receipts]
        source_statuses = [_available_source(f"controller:{self._cluster_id}")]
        if attempt_uids:
            event_entries, event_status = self._task_event_activity(
                query.target,
                attempt_uids,
                query.after,
                before_time,
                before_source_rank,
                before_source_key,
                page_size + 1,
            )
            entries.extend(event_entries)
            source_statuses.append(event_status)
        else:
            source_statuses.append(_unsupported_source(f"finelog:{self._cluster_id}"))
        entries.sort(key=lambda item: item.order_key, reverse=True)
        items = tuple(item.entry for item in entries[:page_size])
        next_token = None
        if len(entries) > page_size:
            last = entries[page_size - 1]
            next_token = _encode_page_token(
                fingerprint,
                {
                    "occurred_at_ms": last.entry.occurred_at.epoch_ms(),
                    "source_rank": last.source_rank,
                    "source_key": last.source_key,
                },
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
        before_time: Timestamp | None,
        before_source_rank: int | None,
        before_source_key: tuple[int | str, ...],
        limit: int,
    ) -> tuple[tuple[_ActivityItem, ...], ResourceSourceStatus]:
        if self._log_client is None:
            error = RuntimeError(_FINELOG_NOT_CONFIGURED)
            return (), _unavailable_finelog_source(self._cluster_id, error)
        task_id = target.resource_id.rpartition(":")[0] if target.kind is ResourceKind.ATTEMPT else target.resource_id
        task_literal = task_id.replace("'", "''")
        uid_literals = ", ".join(f"'{uid.replace("'", "''")}'" for uid in attempt_uids)
        after_predicate = f" AND ts > to_timestamp({after.epoch_ms()} / 1000.0)" if after is not None else ""
        before_predicate = ""
        if before_time is not None:
            before_ms = before_time.epoch_ms()
            before_timestamp = f"to_timestamp({before_ms} / 1000.0)"
            before_predicate = f" AND ts < {before_timestamp}"
            if before_source_rank == 1:
                event_key = _validated_task_event_source_key(before_source_key)
                equal_time_predicate = _sql_key_before(
                    ("attempt_id", "attempt_uid", "type", "reason", "message", "source", "count"),
                    event_key,
                )
                before_predicate = (
                    f" AND (ts < {before_timestamp} OR (ts = {before_timestamp} AND ({equal_time_predicate})))"
                )
        sql = (
            "SELECT attempt_id, attempt_uid, ts, type, reason, message, source, count "
            f"FROM \"{TASK_EVENT_NAMESPACE}\" WHERE task_id = '{task_literal}' "
            f"AND attempt_uid IN ({uid_literals}){after_predicate}{before_predicate} "
            "ORDER BY ts DESC, attempt_id DESC, attempt_uid DESC, type DESC, reason DESC, "
            f"message DESC, source DESC, count DESC LIMIT {limit}"
        )
        try:
            rows = self._log_client.query(sql, max_rows=limit).to_pylist()
        except (ConnectError, ConnectionError, OSError, RuntimeError, StatsError) as exc:
            return (), _unavailable_finelog_source(self._cluster_id, exc)
        entries = tuple(
            _ActivityItem(self._task_event_entry(task_id, row), 1, _task_event_source_key(row)) for row in rows
        )
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
            [task_id, attempt_uid, normalized.isoformat(), severity, kind, message, source, row["count"]],
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
        identity: AttemptIdentity,
        command: tuple[str, ...],
        timeout: Duration | None,
    ) -> ExecResult:
        attempt = self.describe_attempt(AttemptLocator(identity.task, identity.attempt_number))
        if attempt.summary.identity != identity:
            raise ResourceReplaced(identity.task.resource_id)
        self._require_current_attempt(identity)
        task_id = JobName.from_wire(identity.task.resource_id)
        request = ExecRequest(identity, command, timeout)
        handle = self._federated_handle(task_id)
        if handle is not None:
            return self._runtime.federation.proxy_to_peer(
                handle.peer_id,
                lambda peer: peer.exec_in_container(request),
            )
        target, backend = self._task_target(identity)
        self._require_current_attempt(identity)
        return backend.exec_in_container(target, request)

    def profile_attempt(
        self,
        identity: AttemptIdentity,
        profile: ProfileConfiguration | None,
        duration: Duration | None,
    ) -> ProfileResult:
        attempt = self.describe_attempt(AttemptLocator(identity.task, identity.attempt_number))
        if attempt.summary.identity != identity:
            raise ResourceReplaced(identity.task.resource_id)
        self._require_current_attempt(identity)
        task_id = JobName.from_wire(identity.task.resource_id)
        request = ProfileRequest(identity, profile, duration)
        handle = self._federated_handle(task_id)
        if handle is not None:
            return self._runtime.federation.proxy_to_peer(
                handle.peer_id,
                lambda peer: peer.profile_task(request),
            )
        target, backend = self._task_target(identity)
        self._require_current_attempt(identity)
        return backend.profile_task(target, request)

    def _federated_handle(self, task_id: JobName):
        with self._db.read_snapshot() as tx:
            return reads.federated_handle(tx, task_id.root_job)

    def _task_target(self, identity: AttemptIdentity) -> tuple[TaskTarget, TaskBackend]:
        task_id = JobName.from_wire(identity.task.resource_id)
        with self._db.read_snapshot() as tx:
            task = reads.get_task_detail(tx, task_id)
            attempt = tx.execute(
                select(*reads.ATTEMPT_COLS).where(
                    task_attempts_table.c.task_id == task_id,
                    task_attempts_table.c.attempt_id == identity.attempt_number,
                )
            ).first()
        if task is None or attempt is None:
            raise ResourceNotFound(identity.task.resource_id)
        if task.current_attempt_id != identity.attempt_number or str(attempt.attempt_uid) != identity.attempt_uid:
            raise ResourceReplaced(f"{identity.task.resource_id}:{identity.attempt_number}")
        backend = self._backends[self._backend_id(str(task.backend_id))]
        if BackendCapability.CLUSTER_VIEW in backend.capabilities:
            return (
                TaskTarget(
                    task_id=task_id.to_wire(),
                    attempt_id=identity.attempt_number,
                    worker_id=None,
                    address=None,
                    attempt_uid=identity.attempt_uid,
                ),
                backend,
            )
        worker_id = attempt.worker_id or task.current_worker_id
        if worker_id is None:
            raise ActionPolicyRejected(f"Task {task_id} is not assigned to a node")
        if not self._runtime.liveness_for_worker(worker_id).healthy:
            raise ResourceSourceUnavailable(f"Node {worker_id} is unavailable")
        return (
            TaskTarget(
                task_id=task_id.to_wire(),
                attempt_id=identity.attempt_number,
                worker_id=worker_id,
                address=task.current_worker_address,
                attempt_uid=identity.attempt_uid,
            ),
            backend,
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
        task_id = JobName.from_wire(identity.task.resource_id)
        with self._db.read_snapshot() as tx:
            row = tx.execute(
                select(tasks_table.c.current_attempt_id, task_attempts_table.c.attempt_uid)
                .select_from(
                    tasks_table.outerjoin(
                        task_attempts_table,
                        (task_attempts_table.c.task_id == tasks_table.c.task_id)
                        & (task_attempts_table.c.attempt_id == tasks_table.c.current_attempt_id),
                    )
                )
                .where(tasks_table.c.task_id == task_id)
            ).first()
        if row is None:
            raise ResourceNotFound(identity.task.resource_id)
        if row.current_attempt_id != identity.attempt_number or str(row.attempt_uid or "") != identity.attempt_uid:
            raise ResourceReplaced(f"{identity.task.resource_id}:{identity.attempt_number}")

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]:
        """List one bounded page of canonical Nodes."""
        page, _details = self.list_nodes_with_details(query)
        return page

    def list_nodes_with_details(
        self,
        query: NodeQuery = NodeQuery(),
    ) -> tuple[Page[NodeSummary], Mapping[tuple[str, str], _NodeDetails]]:
        """List Nodes and their row-local details without loading attempt history."""
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
        last_key = (
            (
                str(position["backend_id"]),
                str(position["node_id"]),
                str(position["node_uid"]),
            )
            if position is not None
            else None
        )
        provider_snapshot = self._provider_node_snapshot()
        candidates: list[_NodeCandidate] = [
            candidate
            for candidate in provider_snapshot.candidates
            if (query.backend_id is None or candidate.summary.identity.backend_id == query.backend_id)
            and (
                query.contains is None
                or query.contains.casefold() in candidate.summary.identity.key.resource_id.casefold()
            )
            and (not query.health or candidate.summary.health in query.health)
            and (last_key is None or _node_candidate_key(candidate) > last_key)
        ]
        with self._db.read_snapshot() as tx:
            candidates.extend(self._worker_node_candidates(tx, query, last_key, page_size + 1))
            candidates.sort(key=_node_candidate_key)
            selected = candidates[:page_size]
            worker_nodes, worker_details = self._materialize_worker_nodes(
                tx,
                [candidate for candidate in selected if isinstance(candidate, _WorkerNodeCandidate)],
            )

        nodes_by_key = {_node_summary_key(node): node for node in worker_nodes}
        details: dict[tuple[str, str], _NodeDetails] = dict(worker_details)
        items: list[NodeSummary] = []
        for candidate in selected:
            if isinstance(candidate, _ProviderNodeCandidate):
                node = candidate.summary
                details[(node.identity.backend_id, node.identity.node_uid)] = candidate.details
            else:
                node = nodes_by_key[_node_candidate_key(candidate)]
            items.append(node)

        next_token = None
        if len(candidates) > page_size:
            last = items[-1]
            next_token = _encode_page_token(
                fingerprint,
                {
                    "backend_id": last.identity.backend_id,
                    "node_id": last.identity.key.resource_id,
                    "node_uid": last.identity.node_uid,
                },
            )
        return (
            Page(
                items=tuple(items),
                next_page_token=next_token,
                source_statuses=provider_snapshot.source_statuses,
            ),
            details,
        )

    def describe_node(self, locator: NodeLocator) -> NodeDetail:
        provider_snapshot = self._provider_node_snapshot()
        matches: list[tuple[NodeSummary, _NodeDetails]] = [
            (candidate.summary, candidate.details)
            for candidate in provider_snapshot.candidates
            if candidate.summary.identity.key == locator.key
            and candidate.summary.identity.backend_id == locator.backend_id
            and (locator.node_uid is None or candidate.summary.identity.node_uid == locator.node_uid)
        ]
        backend = self._backends.get(locator.backend_id)
        if backend is not None and BackendCapability.WORKER_DAEMON in backend.capabilities:
            worker_id = WorkerId(locator.key.resource_id)
            with self._db.read_snapshot() as tx:
                worker = reads.get_worker_detail(tx, worker_id)
                if (
                    worker is not None
                    and self._runtime.backend_id_for_scale_group(str(worker.scale_group or "")) == locator.backend_id
                    and (locator.node_uid is None or locator.node_uid == worker_id)
                ):
                    nodes, details = self._materialize_worker_nodes(
                        tx,
                        [_WorkerNodeCandidate(locator.backend_id, worker, self._runtime.liveness_for_worker(worker_id))],
                    )
                    node = nodes[0]
                    matches.append((node, details[(locator.backend_id, node.identity.node_uid)]))
        if not matches:
            raise ResourceNotFound(locator.key.resource_id)
        if len(matches) != 1:
            raise ActionPolicyRejected(f"Node locator {locator.key.resource_id!r} is ambiguous")
        node, details = matches[0]
        return NodeDetail(
            summary=node,
            address=details.address,
            attributes=details.attributes,
            recent_attempts=self._recent_attempts_for_node(node),
            bootstrap_log_key=None,
            source_statuses=provider_snapshot.source_statuses,
        )

    def _recent_attempts_for_node(self, node: NodeSummary) -> tuple[AttemptSummary, ...]:
        backend = self._backends[node.identity.backend_id]
        with self._db.read_snapshot() as tx:
            if BackendCapability.WORKER_DAEMON in backend.capabilities:
                attempts = reads.recent_attempts_for_worker(
                    tx,
                    WorkerId(node.identity.key.resource_id),
                    limit=_MAX_NODE_RECENT_ATTEMPTS,
                )
            elif BackendCapability.CLUSTER_VIEW in backend.capabilities:
                attempts = reads.recent_attempts_for_provider_node(
                    tx,
                    node.identity.backend_id,
                    node.identity.key.resource_id,
                    limit=_MAX_NODE_RECENT_ATTEMPTS,
                )
            else:
                return ()
            if not attempts:
                return ()
            task_ids = {attempt.task_id for attempt in attempts}
            task_rows = tx.execute(
                reads.task_detail_query().where(tasks_table.c.task_id.in_(bindparam("task_ids", expanding=True))),
                {"task_ids": list(task_ids)},
            ).all()
            tasks = {task.task_id: task for task in task_rows}
            jobs = self._job_rows(tx, {task.job_id for task in task_rows})
        return tuple(
            self._attempt_summary(tasks[attempt.task_id], attempt, jobs[tasks[attempt.task_id].job_id])
            for attempt in attempts
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
        snapshot = self._slice_snapshot()
        filtered = [
            item
            for item in snapshot.slices
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
        return Page(items=items, next_page_token=next_token, source_statuses=snapshot.source_statuses)

    def describe_slice(self, locator: SliceLocator) -> SliceDetail:
        snapshot = self._slice_snapshot()
        matches = [
            item
            for item in snapshot.slices
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
            members=snapshot.members.get((item.identity.backend_id, item.identity.slice_uid), ()),
            source_statuses=snapshot.source_statuses,
        )

    def _provider_node_snapshot(self) -> _ProviderNodeSnapshot:
        candidates: list[_ProviderNodeCandidate] = []
        statuses: list[ResourceSourceStatus] = []
        for backend_id, backend in sorted(self._backends.items()):
            try:
                observation = observe_backend_resources(backend)
            except (ConnectionError, ProviderError) as exc:
                statuses.append(_unavailable_backend_source(backend_id, exc))
            else:
                if len(observation.nodes) > MAX_PROVIDER_SNAPSHOT_ITEMS:
                    statuses.append(
                        _unavailable_backend_source(
                            backend_id,
                            ValueError(f"provider returned more than {MAX_PROVIDER_SNAPSHOT_ITEMS} nodes"),
                        )
                    )
                    continue
                observed_at = Timestamp.now()
                for node in observation.nodes:
                    identity = NodeIdentity(
                        ResourceKey(self._cluster_id, ResourceKind.NODE, node.provider_node_id),
                        backend_id,
                        _opaque_uid(f"kubernetes:{backend_id}:{node.provider_node_id}:{node.incarnation}"),
                    )
                    summary = NodeSummary(
                        identity=identity,
                        health=NodeHealth.READY if node.ready else NodeHealth.UNAVAILABLE,
                        schedulable=node.schedulable,
                        capacity=node.capacity,
                        scaling_group_id=None,
                        slice=None,
                        running_task_count=node.running_task_count,
                        observed_at=observed_at,
                        region=node.region,
                    )
                    attributes = tuple(
                        attribute
                        for attribute in (
                            _string_node_attribute("instance_type", node.instance_type or ""),
                            _string_node_attribute("region", node.region or ""),
                        )
                        if attribute is not None
                    )
                    candidates.append(_ProviderNodeCandidate(summary, _NodeDetails(None, attributes)))
                statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
        return _ProviderNodeSnapshot(tuple(candidates), tuple(statuses))

    def _worker_node_candidates(
        self,
        tx: Tx,
        query: NodeQuery,
        last_key: tuple[str, str, str] | None,
        limit: int,
    ) -> list[_WorkerNodeCandidate]:
        configured_scale_groups = {
            scale_group: backend_id
            for backend_id, config in self._backend_configs.items()
            for scale_group in config.scale_groups
        }
        candidates: list[_WorkerNodeCandidate] = []
        for backend_id, backend in sorted(self._backends.items()):
            if BackendCapability.WORKER_DAEMON not in backend.capabilities:
                continue
            if query.backend_id is not None and query.backend_id != backend_id:
                continue
            if last_key is not None and backend_id < last_key[0]:
                continue
            after_worker_id = WorkerId(last_key[1]) if last_key is not None and backend_id == last_key[0] else None
            include_after = after_worker_id is not None
            while len(candidates) < limit:
                if backend_id == DEFAULT_BACKEND_ID:
                    rows = reads.worker_detail_page_outside_scale_groups(
                        tx,
                        [
                            scale_group
                            for scale_group, owner in configured_scale_groups.items()
                            if owner != DEFAULT_BACKEND_ID
                        ],
                        after_worker_id=after_worker_id,
                        include_after=include_after,
                        limit=_NODE_WORKER_SCAN_BATCH,
                    )
                else:
                    rows = reads.worker_detail_page_in_scale_groups(
                        tx,
                        [scale_group for scale_group, owner in configured_scale_groups.items() if owner == backend_id],
                        after_worker_id=after_worker_id,
                        include_after=include_after,
                        limit=_NODE_WORKER_SCAN_BATCH,
                    )
                if not rows:
                    break
                for worker in rows:
                    worker_id = WorkerId(worker.worker_id)
                    candidate = _WorkerNodeCandidate(
                        backend_id,
                        worker,
                        self._runtime.liveness_for_worker(worker_id),
                    )
                    if last_key is not None and _node_candidate_key(candidate) <= last_key:
                        continue
                    if query.contains is not None and query.contains.casefold() not in worker_id.casefold():
                        continue
                    health = NodeHealth.READY if candidate.liveness.healthy else NodeHealth.DEGRADED
                    if query.health and health not in query.health:
                        continue
                    candidates.append(candidate)
                    if len(candidates) == limit:
                        break
                if len(candidates) == limit or len(rows) < _NODE_WORKER_SCAN_BATCH:
                    break
                after_worker_id = WorkerId(rows[-1].worker_id)
                include_after = False
        return candidates

    def _materialize_worker_nodes(
        self,
        tx: Tx,
        candidates: Sequence[_WorkerNodeCandidate],
    ) -> tuple[tuple[NodeSummary, ...], Mapping[tuple[str, str], _NodeDetails]]:
        if not candidates:
            return (), {}
        worker_ids = [WorkerId(candidate.worker.worker_id) for candidate in candidates]
        attributes_by_worker: dict[WorkerId, dict[str, str | int | float]] = {}
        for row in reads.worker_attribute_rows(tx, worker_ids):
            key, value = decode_attribute_value(row)
            attributes_by_worker.setdefault(WorkerId(row.worker_id), {})[key] = value
        running = reads.running_tasks_by_worker(tx, set(worker_ids))
        nodes: list[NodeSummary] = []
        details: dict[tuple[str, str], _NodeDetails] = {}
        for candidate in candidates:
            worker = candidate.worker
            worker_id = WorkerId(worker.worker_id)
            stored_attributes = attributes_by_worker.get(worker_id, {})
            metadata = worker_node_metadata(worker, stored_attributes)
            identity = NodeIdentity(
                ResourceKey(self._cluster_id, ResourceKind.NODE, worker_id),
                candidate.backend_id,
                worker_id,
            )
            slice_identity = None
            if metadata.slice_id:
                slice_identity = SliceIdentity(
                    ResourceKey(self._cluster_id, ResourceKind.SLICE, metadata.slice_id),
                    candidate.backend_id,
                    _opaque_uid(f"rpc:{candidate.backend_id}:{metadata.slice_id}"),
                )
            nodes.append(
                NodeSummary(
                    identity=identity,
                    health=NodeHealth.READY if candidate.liveness.healthy else NodeHealth.DEGRADED,
                    schedulable=candidate.liveness.healthy,
                    capacity=metadata.capacity,
                    scaling_group_id=str(worker.scale_group or "") or None,
                    slice=slice_identity,
                    running_task_count=len(running.get(worker_id, set())),
                    observed_at=Timestamp.from_ms(candidate.liveness.last_heartbeat_ms),
                    region=metadata.region,
                )
            )
            details[(candidate.backend_id, identity.node_uid)] = _NodeDetails(
                str(worker.address or "") or None,
                metadata.attributes,
            )
        return tuple(nodes), details

    def _slice_snapshot(self) -> _SliceSnapshot:
        slices: list[SliceSummary] = []
        members: dict[tuple[str, str], tuple[SliceMember, ...]] = {}
        statuses: list[ResourceSourceStatus] = []
        for backend_id, backend in sorted(self._backends.items()):
            if BackendCapability.IRIS_AUTOSCALER not in backend.capabilities:
                statuses.append(_unsupported_source(f"backend:{backend_id}", backend_id=backend_id))
                continue
            try:
                observation = observe_autoscaler_resources(backend)
            except (ConnectionError, ProviderError) as exc:
                statuses.append(_unavailable_backend_source(backend_id, exc))
                continue
            if len(observation.slices) > MAX_PROVIDER_SNAPSHOT_ITEMS:
                statuses.append(
                    _unavailable_backend_source(
                        backend_id,
                        ValueError(f"provider returned more than {MAX_PROVIDER_SNAPSHOT_ITEMS} slices"),
                    )
                )
                continue
            statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
            for item in observation.slices:
                slice_uid = _opaque_uid(
                    f"rpc:{backend_id}:{item.slice_id}:{item.created_at.epoch_ms() if item.created_at else 0}"
                )
                identity = SliceIdentity(
                    ResourceKey(self._cluster_id, ResourceKind.SLICE, item.slice_id),
                    backend_id,
                    slice_uid,
                )
                lifecycle = _slice_lifecycle(item.lifecycle_state)
                membership_state = (
                    MembershipState.OBSERVED if lifecycle is SliceLifecycle.READY else MembershipState.UNKNOWN
                )
                slices.append(
                    SliceSummary(
                        identity=identity,
                        scaling_group_id=item.scaling_group_id,
                        lifecycle=lifecycle,
                        membership_state=membership_state,
                        observed_member_count=len(item.provider_node_ids),
                        observed_at=observation.observed_at,
                        error_message=item.error_message,
                    )
                )
                members[(backend_id, slice_uid)] = tuple(
                    SliceMember(
                        provider_node_id=provider_node_id,
                        node=None,
                        observed_at=observation.observed_at,
                    )
                    for provider_node_id in item.provider_node_ids
                )
        return _SliceSnapshot(tuple(slices), members, tuple(statuses))

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
        peer_id: str | None = None
        backend_id = ""
        execution_cluster_id = ""
        remote_attempt: AttemptIdentity | None = None
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
            current_identity = _task_uid(self._job_identity(job).job_uid, row.task_id)
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
            handle = reads.federated_handle(tx, row.task_id.root_job)
            if handle is not None:
                peer_id = handle.peer_id
                backend_id = str(row.backend_id or "")
                execution_cluster_id = _execution_cluster(self._cluster_id, str(row.cluster))
                remote_attempt = AttemptIdentity(identity.key, int(attempt.attempt_id), str(attempt.attempt_uid))
            else:
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
            if peer_id is None:
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

        assert peer_id is not None and remote_attempt is not None
        if kind is ActionKind.RETRY_TASK:
            receipt = self._runtime.federation.proxy_to_peer(
                peer_id,
                lambda peer: peer.retry_task(
                    identity,
                    expected_attempt_uid=expected_attempt_uid,
                    idempotency_key=idempotency_key,
                ),
            )
        else:
            receipt = self._runtime.federation.proxy_to_peer(
                peer_id,
                lambda peer: peer.terminate_attempt(remote_attempt, idempotency_key=idempotency_key),
            )
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
            action_persistence.insert_action(
                tx,
                receipt,
                authority_cluster_id=authority,
                authority_action_id=receipt.action_id,
                backend_id=backend_id,
                execution_cluster_id=execution_cluster_id,
                principal_id=principal_id,
                idempotency_key=_require_idempotency_key(idempotency_key),
                payload_hash=payload_hash,
            )
        return receipt

    def _job_summary_from_row(
        self,
        row,
        *,
        coordinates=None,
        parent_coordinates: Mapping[JobName, _JobCoordinates] | None = None,
    ) -> JobSummary:
        job_id = row.job_id
        authority = self._authority_cluster(coordinates or row)
        execution = _execution_cluster(self._cluster_id, str(row.cluster))
        submitted_at = row.submitted_at_ms
        parent = None
        if row.parent_job_id is not None:
            parent_id = row.parent_job_id
            parent_row = (parent_coordinates or {}).get(parent_id)
            if parent_row is not None:
                parent = JobIdentity(
                    ResourceKey(authority, ResourceKind.JOB, parent_id.to_wire()),
                    _job_uid(
                        authority,
                        parent_id,
                        parent_row.submitted_at_ms,
                        handoff_nonce=str(parent_row.handoff_nonce or ""),
                    ),
                )
        return JobSummary(
            identity=JobIdentity(
                ResourceKey(authority, ResourceKind.JOB, job_id.to_wire()),
                _job_uid(
                    authority,
                    job_id,
                    submitted_at,
                    handoff_nonce=str(getattr(coordinates or row, "handoff_nonce", "") or ""),
                ),
            ),
            owner_id=job_id.user,
            parent=parent,
            state=JobState(row.state),
            execution_cluster_id=execution,
            backend_id=self._known_backend_id(str(row.backend_id or ""), execution),
            num_tasks=int(row.num_tasks),
            submitted_at=submitted_at,
            started_at=row.started_at_ms,
            finished_at=row.finished_at_ms,
            error_message=str(row.error or ""),
            pending_reason=self._job_pending_reason(row, coordinates or row),
        )

    def _job_pending_reason(self, row, coordinates) -> str:
        if JobState(row.state) is not JobState.PENDING:
            return ""
        if getattr(coordinates, "direction", None) == int(FederationDirection.SENT):
            peer_id = str(getattr(coordinates, "peer_id", "") or "")
            handoff_state = getattr(coordinates, "handoff_state", None)
            if handoff_state == int(HandoffState.QUEUED_HANDOFF):
                if peer_id:
                    return f"Queued for peer {peer_id} to report free capacity"
                return "Queued for a federation peer to report free capacity"
            if handoff_state == int(HandoffState.PENDING_HANDOFF):
                return f"Awaiting acceptance by peer {peer_id}"
            return f"Pending on peer {peer_id}"

        scheduler_reason = self._runtime.get_job_scheduling_diagnostics(row.job_id.to_wire())
        pending_reason = scheduler_reason or "Pending scheduler feedback"
        hint = None
        for backend in self._backends.values():
            if backend.autoscaler is not None:
                hint = backend.autoscaler.get_pending_hints().get(row.job_id.to_wire())
                if hint is not None:
                    break
        if hint is None:
            return pending_reason
        scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
        return f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

    def _task_summary(self, row, current_attempt, counts: AttemptCounts, job) -> TaskSummary:
        authority = self._authority_cluster(job)
        execution = _execution_cluster(self._cluster_id, str(row.cluster))
        task_key = ResourceKey(authority, ResourceKind.TASK, row.task_id.to_wire())
        job_identity = self._job_identity(job)
        task_identity = TaskIdentity(task_key, _task_uid(job_identity.job_uid, row.task_id))
        attempt_identity = None
        node_identity = None
        stored_backend_id = str(row.backend_id)
        backend_id = (
            self._execution_backend_id(stored_backend_id, execution)
            if stored_backend_id or current_attempt is not None
            else ""
        )
        if current_attempt is not None:
            attempt_identity = AttemptIdentity(
                task_key, int(current_attempt.attempt_id), str(current_attempt.attempt_uid)
            )
            node_id = str(
                current_attempt.node_name or current_attempt.worker_id or getattr(row, "peer_worker_label", "") or ""
            )
            if node_id and backend_id:
                node_identity = self._current_node_identity(execution, backend_id, node_id)
        return TaskSummary(
            identity=task_identity,
            job=job_identity,
            task_index=int(row.task_index),
            state=TaskState(row.state),
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
        authority = self._authority_cluster(job)
        task_key = ResourceKey(authority, ResourceKind.TASK, task.task_id.to_wire())
        backend_id = str(attempt.backend_id or task.backend_id or "")
        execution = _execution_cluster(self._cluster_id, str(task.cluster)) if backend_id else ""
        node_id = str(attempt.node_name or attempt.worker_id or "")
        node = None
        if node_id and backend_id:
            node = self._current_node_identity(execution, backend_id, node_id)
        return AttemptSummary(
            identity=AttemptIdentity(task_key, int(attempt.attempt_id), str(attempt.attempt_uid)),
            state=TaskState(attempt.state),
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

    def _attempt_runtime(
        self,
        attempt,
        execution_cluster_id: str,
        backend_id: str,
        container_id: str,
    ) -> AttemptRuntimeObject | None:
        config = self._backend_configs.get(backend_id) if execution_cluster_id == self._cluster_id else None
        provider_kind = ""
        namespace = ""
        provider_node_id = ""
        if config is not None and config.kind == "k8s":
            provider_kind = "kubernetes"
            if config.kubernetes_provider is not None:
                namespace = config.kubernetes_provider.namespace
            provider_node_id = str(attempt.node_name or "")
        elif config is not None and config.kind == "worker_daemon":
            provider_kind = "rpc"
            provider_node_id = str(attempt.worker_id or "")
        name = str(attempt.pod_name or "")
        provider_uid = str(attempt.pod_uid or "")
        if not any((namespace, name, provider_uid, provider_node_id, container_id)):
            return None
        observed_at = attempt.finished_at_ms or attempt.started_at_ms or attempt.created_at_ms
        return AttemptRuntimeObject(
            provider_kind=provider_kind,
            namespace=namespace,
            name=name,
            provider_uid=provider_uid,
            provider_node_id=provider_node_id,
            provider_node_uid="",
            container_id=container_id,
            observed_at=observed_at,
        )

    def _job_identity(self, row) -> JobIdentity:
        authority = self._authority_cluster(row)
        return JobIdentity(
            ResourceKey(authority, ResourceKind.JOB, row.job_id.to_wire()),
            _job_uid(
                authority,
                row.job_id,
                row.submitted_at_ms,
                handoff_nonce=str(getattr(row, "handoff_nonce", "") or ""),
            ),
        )

    def _current_node_identity(self, execution: str, backend_id: str, node_id: str) -> NodeIdentity | None:
        if execution != self._cluster_id or not backend_id or not node_id:
            return None
        backend = self._backends.get(backend_id)
        if backend is None or BackendCapability.WORKER_DAEMON not in backend.capabilities:
            return None
        return NodeIdentity(ResourceKey(execution, ResourceKind.NODE, node_id), backend_id, node_id)

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

    def _execution_backend_id(self, stored: str, execution_cluster_id: str) -> str:
        if execution_cluster_id != self._cluster_id:
            return stored
        return self._backend_id(stored)

    def _known_backend_id(self, stored: str, execution_cluster_id: str) -> str:
        if stored or execution_cluster_id != self._cluster_id:
            return stored
        if len(self._backends) == 1:
            return next(iter(self._backends))
        return ""

    def _endpoint_summary(self, row: EndpointRow, coordinates: tuple[str, str]) -> EndpointSummary:
        authority, execution = coordinates
        task = ResourceKey(authority, ResourceKind.TASK, row.task_id.to_wire())
        return EndpointSummary(
            key=ResourceKey(authority, ResourceKind.ENDPOINT, row.endpoint_id),
            endpoint_id=row.endpoint_id,
            name=row.name,
            task=task,
            execution_cluster_id=execution,
            access=EndpointAccess.from_storage(row.access),
            lease_deadline=row.lease_deadline,
        )

    def _system_endpoint_summary(self, name: str) -> EndpointSummary:
        return EndpointSummary(
            key=ResourceKey(self._cluster_id, ResourceKind.ENDPOINT, name),
            endpoint_id=name,
            name=name,
            task=None,
            execution_cluster_id=self._cluster_id,
            access=EndpointAccess.PRIVATE,
            lease_deadline=None,
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

    def _job_rows(self, tx, job_ids: set[JobName]) -> dict[JobName, _JobCoordinates]:
        if not job_ids:
            return {}
        rows = tx.execute(
            select(
                jobs_table.c.job_id,
                jobs_table.c.submitted_at_ms,
                federated_jobs_table.c.direction,
                federated_jobs_table.c.peer_id,
                federated_jobs_table.c.handoff_state,
                federated_jobs_table.c.handoff_nonce,
            )
            .select_from(
                jobs_table.outerjoin(federated_jobs_table, federated_jobs_table.c.job_id == jobs_table.c.root_job_id)
            )
            .where(jobs_table.c.job_id.in_(job_ids))
        ).all()
        return {
            row.job_id: _JobCoordinates(
                job_id=row.job_id,
                submitted_at_ms=row.submitted_at_ms,
                direction=row.direction,
                peer_id=row.peer_id,
                handoff_state=row.handoff_state,
                handoff_nonce=row.handoff_nonce,
            )
            for row in rows
        }

    def _job_authorities(self, job_ids) -> dict[JobName, str]:
        ids = set(job_ids)
        with self._db.read_snapshot() as tx:
            return {job_id: self._authority_cluster(row) for job_id, row in self._job_rows(tx, ids).items()}

    @staticmethod
    def _job_coordinates_in_snapshot(tx: Tx, job_ids: set[JobName]) -> dict[JobName, _JobCoordinates]:
        if not job_ids:
            return {}
        rows = tx.execute(
            select(
                jobs_table.c.job_id,
                jobs_table.c.submitted_at_ms,
                federated_jobs_table.c.direction,
                federated_jobs_table.c.peer_id,
                federated_jobs_table.c.handoff_state,
                federated_jobs_table.c.handoff_nonce,
            )
            .select_from(
                jobs_table.outerjoin(federated_jobs_table, federated_jobs_table.c.job_id == jobs_table.c.root_job_id)
            )
            .where(jobs_table.c.job_id.in_(job_ids))
        ).all()
        return {
            row.job_id: _JobCoordinates(
                job_id=row.job_id,
                submitted_at_ms=row.submitted_at_ms,
                direction=row.direction,
                peer_id=row.peer_id,
                handoff_state=row.handoff_state,
                handoff_nonce=row.handoff_nonce,
            )
            for row in rows
        }

    def _source_statuses(self) -> tuple[ResourceSourceStatus, ...]:
        statuses = [_available_source(f"controller:{self._cluster_id}")]
        for backend_id, backend in sorted(self._backends.items()):
            try:
                observe_backend_resources(backend)
            except (ConnectionError, ProviderError) as exc:
                statuses.append(_unavailable_backend_source(backend_id, exc))
            else:
                statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
        peer_observations = {peer.peer_id: peer for peer in self._runtime.federation.peer_observations()}
        statuses.extend(self._peer_source_statuses(set(peer_observations), observations=peer_observations))
        return tuple(statuses)

    def _peer_source_statuses(
        self,
        peer_ids: set[str],
        *,
        observations: Mapping[str, FederationPeerObservation] | None = None,
    ) -> tuple[ResourceSourceStatus, ...]:
        if not peer_ids:
            return ()
        if observations is None:
            observations = {peer.peer_id: peer for peer in self._runtime.federation.peer_observations()}
        statuses = []
        for peer_id in sorted(peer_ids):
            peer = observations.get(peer_id)
            if peer is None:
                statuses.append(
                    ResourceSourceStatus(
                        source_id=f"federation:{peer_id}",
                        backend_id="",
                        state=SourceState.UNAVAILABLE,
                        freshness=Freshness.UNKNOWN,
                        observed_at=None,
                        error_code=_PEER_UNAVAILABLE,
                        error_message=f"Federation peer {peer_id} is not configured",
                    )
                )
                continue
            observed_at = Timestamp.from_ms(peer.last_contact_ms) if peer.last_contact_ms else None
            if peer.reachable:
                statuses.append(
                    ResourceSourceStatus(
                        source_id=f"federation:{peer.peer_id}",
                        backend_id="",
                        state=SourceState.AVAILABLE,
                        freshness=Freshness.CURRENT,
                        observed_at=observed_at,
                        error_code="",
                        error_message="",
                    )
                )
            else:
                statuses.append(
                    ResourceSourceStatus(
                        source_id=f"federation:{peer.peer_id}",
                        backend_id="",
                        state=SourceState.UNAVAILABLE,
                        freshness=Freshness.STALE if observed_at is not None else Freshness.UNKNOWN,
                        observed_at=observed_at,
                        error_code=_PEER_UNAVAILABLE,
                        error_message=f"Federation peer {peer.peer_id} is unreachable",
                    )
                )
        return tuple(statuses)


def _node_summary_key(node: NodeSummary) -> tuple[str, str, str]:
    return (
        node.identity.backend_id,
        node.identity.key.resource_id,
        node.identity.node_uid,
    )


def _node_candidate_key(candidate: _NodeCandidate) -> tuple[str, str, str]:
    if isinstance(candidate, _ProviderNodeCandidate):
        return _node_summary_key(candidate.summary)
    worker_id = str(candidate.worker.worker_id)
    return (candidate.backend_id, worker_id, worker_id)


def _page_size(value: int, maximum: int) -> int:
    if value <= 0 or value > maximum:
        raise ValueError(f"page_size must be between 1 and {maximum}")
    return value


def _require_kind(key: ResourceKey, kind: ResourceKind) -> None:
    if key.kind is not kind:
        raise ValueError(f"expected {kind.value}, got {key.kind.value}")


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
