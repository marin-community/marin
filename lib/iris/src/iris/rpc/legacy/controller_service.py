# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy ControllerService wire adapter and operational transport RPCs.

Job, Task, Attempt, Node, and Endpoint behavior lives in ``Controller``.
This service translates the retired Job/Task protobuf API and hosts worker and
federation transport methods that do not have resource equivalents.
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import batched
from typing import Any, Protocol, TypeVar

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from finelog.client import LogClient
from rigging.server_auth import get_verified_identity, require_identity
from rigging.timing import Duration, Timer, Timestamp

from iris.backends.protocol import BackendCapability, ProviderError, TaskBackend, TaskTarget
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.auth import (
    ControllerAuth,
)
from iris.cluster.controller.budget import (
    compute_effective_band,
)
from iris.cluster.controller.controller import Controller
from iris.cluster.controller.job import FederationSubmission
from iris.cluster.controller.operations import (
    FederatedRoute,
    OperationalServices,
    TaskOperations,
    UserOperations,
    WorkerOperations,
)
from iris.cluster.controller.scheduling.scheduler import SchedulingContext
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.federation.availability import AVAILABILITY_METRIC_VERSION
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.process_status import get_process_status
from iris.cluster.runtime.profile import (
    build_profile_row,
    profile_local_process,
)
from iris.cluster.stats.tables import (
    PROFILE_NAMESPACE,
    TASK_EVENT_NAMESPACE,
    TASK_EVENT_STORAGE_POLICY,
    IrisProfile,
    TaskEventRow,
)
from iris.cluster.types import (
    LOCAL_CLUSTER,
    TERMINAL_JOB_STATES,
    UserBudgetDefaults,
)
from iris.resources.endpoint import EndpointQuery, ProfileRequest
from iris.resources.errors import ResourceNotFound
from iris.resources.identity import AttemptLocator, ResourceKey
from iris.resources.job import (
    FederationPosture,
    JobInventoryQuery,
    JobListScope,
    JobObservation,
    JobQuery,
    JobSortField,
    JobSummary,
    SortDirection,
)
from iris.resources.names import (
    JobName,
    TaskAttempt,
    WorkerId,
)
from iris.resources.node import NodeAttributeKind, NodeHealth, NodeQuery
from iris.resources.state import JobState
from iris.resources.task import TaskQuery, TaskSummary
from iris.rpc import controller_pb2, job_pb2, query_pb2, vm_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE, AuthzAction, authorize, authorize_resource_owner
from iris.rpc.backend_status_codec import autoscaler_status_to_proto, backend_status_to_proto, kubernetes_status_to_proto
from iris.rpc.endpoint_service import EndpointServiceImpl
from iris.rpc.federation_client import federation_batch_to_legacy, peer_observation_to_legacy, peer_rpc_call
from iris.rpc.legacy.job_codec import resource_spec_to_proto
from iris.rpc.legacy.job_service_codec import (
    job_spec_from_legacy_request,
    job_spec_to_legacy_request,
    job_status_to_legacy,
    redact_request_env_vars,
    task_detail_to_legacy,
)
from iris.rpc.profile_codec import profile_configuration_from_proto
from iris.rpc.proto_display import (
    job_state_friendly,
    task_state_friendly,
)
from iris.rpc.resource_errors import resource_call
from iris.rpc.worker_codec import process_info_to_proto, worker_metadata_from_proto, worker_metadata_to_proto
from iris.time_proto import duration_from_proto, timestamp_to_proto

logger = logging.getLogger(__name__)

_LEGACY_RESOURCE_PAGE_SIZE = 500

_JOB_SCOPE_FROM_LEGACY = {
    controller_pb2.Controller.JOB_QUERY_SCOPE_ALL: JobListScope.ALL,
    controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS: JobListScope.ROOTS,
    controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN: JobListScope.CHILDREN,
}
_JOB_SORT_FROM_LEGACY = {
    controller_pb2.Controller.JOB_SORT_FIELD_DATE: JobSortField.DATE,
    controller_pb2.Controller.JOB_SORT_FIELD_NAME: JobSortField.NAME,
    controller_pb2.Controller.JOB_SORT_FIELD_STATE: JobSortField.STATE,
    controller_pb2.Controller.JOB_SORT_FIELD_FAILURES: JobSortField.FAILURES,
    controller_pb2.Controller.JOB_SORT_FIELD_PREEMPTIONS: JobSortField.PREEMPTIONS,
}

# Return type of a proxied on-demand RPC (a unary controller response).
_T = TypeVar("_T")


def attempt_is_worker_failure(state: int) -> bool:
    """Whether a terminal state (worker-failed or preempted) is a worker-side failure, not an application failure."""
    return state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED)


@dataclass(frozen=True)
class UserStats:
    user: str
    task_state_counts: dict[int, int] = field(default_factory=dict)
    job_state_counts: dict[int, int] = field(default_factory=dict)


# Cap on the merged autoscaler action log returned by GetAutoscalerStatus; matches
# the per-autoscaler action_log deque cap so a single-backend view is unchanged.
_MERGED_AUTOSCALER_ACTIONS = 100

# Max unroutable job sample entries returned by ListBackends.
_UNROUTABLE_SAMPLE_SIZE = 10


def _accumulate_routing_decision(merged: vm_pb2.RoutingDecision, sub: vm_pb2.RoutingDecision) -> None:
    """Fold one backend's routing decision into the merged decision.

    Scale groups partition disjointly across backends (the single
    scale-group->backend key space), so the group-keyed maps never collide and the
    per-group lists concatenate. With a single backend this reproduces that
    backend's decision exactly.
    """
    for group, launch in sub.group_to_launch.items():
        merged.group_to_launch[group] = launch
    for group, reason in sub.group_reasons.items():
        merged.group_reasons[group] = reason
    for group, entries in sub.routed_entries.items():
        merged.routed_entries[group].CopyFrom(entries)
    merged.unmet_entries.extend(sub.unmet_entries)
    merged.group_statuses.extend(sub.group_statuses)


def _encode_query_cell(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<blob:{len(value)} bytes>"
    return value


USER_TASK_STATES = (
    job_pb2.TASK_STATE_PENDING,
    job_pb2.TASK_STATE_ASSIGNED,
    job_pb2.TASK_STATE_BUILDING,
    job_pb2.TASK_STATE_RUNNING,
    job_pb2.TASK_STATE_SUCCEEDED,
    job_pb2.TASK_STATE_FAILED,
    job_pb2.TASK_STATE_KILLED,
    job_pb2.TASK_STATE_UNSCHEDULABLE,
    job_pb2.TASK_STATE_WORKER_FAILED,
    job_pb2.TASK_STATE_PREEMPTED,
    job_pb2.TASK_STATE_COSCHED_FAILED,
)
USER_JOB_STATES = (
    job_pb2.JOB_STATE_PENDING,
    job_pb2.JOB_STATE_BUILDING,
    job_pb2.JOB_STATE_RUNNING,
    job_pb2.JOB_STATE_SUCCEEDED,
    job_pb2.JOB_STATE_FAILED,
    job_pb2.JOB_STATE_KILLED,
    job_pb2.JOB_STATE_WORKER_FAILED,
    job_pb2.JOB_STATE_UNSCHEDULABLE,
)


@dataclass(frozen=True, slots=True)
class TaskWithAttempts:
    """Task detail columns with attempt rows attached."""

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at_ms: Timestamp
    priority_band: int
    error: str | None
    exit_code: int | None
    started_at_ms: Timestamp | None
    finished_at_ms: Timestamp | None
    current_worker_id: WorkerId | None
    current_worker_address: str | None
    container_id: str | None
    # Backend status one-liner for a waiting/building task (why it is not running
    # yet); None/"" when running or quiet. See tasks.status_message.
    status_message: str | None
    backend_id: str
    cluster: str
    # Display worker identity for a federated task (the peer-side worker name from
    # the federated_tasks sidecar); "" for a local task, which uses its worker FK.
    peer_worker_label: str
    attempts: tuple[Any, ...]

    @classmethod
    def from_row(cls, row, attempts: tuple[Any, ...]) -> "TaskWithAttempts":
        """Build from an SA Row (matching TASK_DETAIL_COLS + peer_worker_label) plus attempt rows.

        Per-task failure/preemption counts are not carried: clients derive them
        from ``attempts``.
        """
        return cls(
            task_id=row.task_id,
            job_id=row.job_id,
            state=row.state,
            current_attempt_id=row.current_attempt_id,
            max_retries_failure=row.max_retries_failure,
            max_retries_preemption=row.max_retries_preemption,
            submitted_at_ms=row.submitted_at_ms,
            priority_band=row.priority_band,
            error=row.error,
            exit_code=row.exit_code,
            started_at_ms=row.started_at_ms,
            finished_at_ms=row.finished_at_ms,
            current_worker_id=row.current_worker_id,
            current_worker_address=row.current_worker_address,
            container_id=row.container_id,
            status_message=row.status_message,
            backend_id=str(row.backend_id or ""),
            cluster=str(row.cluster),
            peer_worker_label=str(row.peer_worker_label or ""),
            attempts=attempts,
        )


def _current_attempt(task: TaskWithAttempts):
    """Get the latest attempt for a task, or None."""
    if not task.attempts:
        return None
    return task.attempts[-1]


def _task_worker_id(task: TaskWithAttempts) -> WorkerId | None:
    """Get the effective worker_id for a task."""
    current = _current_attempt(task)
    if current is None:
        return task.current_worker_id
    return current.worker_id


def worker_status_message(liveness: WorkerLiveness) -> str:
    """Build a human-readable status message for unhealthy workers."""
    if liveness.healthy:
        return ""
    age_ms = max(0, Timestamp.now().epoch_ms() - liveness.last_heartbeat_ms)
    return f"Unhealthy (last seen {age_ms // 1000}s ago)"


_WORKER_TARGET_PREFIX = "/system/worker/"


def _parse_worker_target(target: str) -> str | None:
    """Extract worker_id from a /system/worker/<worker_id> target.

    Returns the worker_id string, or None if the target does not match.
    """
    if target.startswith(_WORKER_TARGET_PREFIX):
        worker_id = target[len(_WORKER_TARGET_PREFIX) :]
        if worker_id:
            return worker_id
    return None


def _active_job_count(job_state_counts: dict[int, int]) -> int:
    """Return the count of non-terminal jobs in a user aggregate."""
    return sum(count for state, count in job_state_counts.items() if state not in TERMINAL_JOB_STATES)


def _task_state_counts_for_summary(task_state_counts: dict[int, int]) -> dict[str, int]:
    """Convert enum-keyed task counts to the string-keyed RPC shape."""
    counts = {task_state_friendly(state): 0 for state in USER_TASK_STATES}
    for state, count in task_state_counts.items():
        counts[task_state_friendly(state)] = count
    return counts


def _job_state_counts_for_summary(job_state_counts: dict[int, int]) -> dict[str, int]:
    """Convert enum-keyed job counts to the string-keyed RPC shape."""
    counts = {job_state_friendly(state): 0 for state in USER_JOB_STATES}
    for state, count in job_state_counts.items():
        counts[job_state_friendly(state)] = count
    return counts


# =============================================================================
# DB query helpers — thin wrappers over snapshot() for common read patterns
# =============================================================================


def _read_task_with_attempts(tasks: TaskOperations, task_id: JobName) -> TaskWithAttempts | None:
    """Return a TaskWithAttempts for ``task_id``, or None if absent."""
    result = tasks.task_detail_with_attempts(task_id)
    if result is None:
        return None
    task_row, attempt_rows = result
    return TaskWithAttempts.from_row(task_row, attempt_rows)


def _read_worker(workers: WorkerOperations, worker_id: WorkerId):
    """Return a slim (worker_id, address, scale_group) row for ``worker_id``, or None."""
    return workers.worker(worker_id)


def _read_worker_detail(workers: WorkerOperations, worker_id: WorkerId):
    return workers.worker_detail(worker_id)


MAX_LIST_JOBS_LIMIT = 500
# Hard cap on how deep ListJobs callers may page. A correctly-filtered query
# should narrow the result set; anything reaching offsets this deep is a sign
# of a caller scanning the entire jobs table page-by-page, which is what the
# snapshot is supposed to prevent. Force callers to filter instead.
MAX_LIST_JOBS_OFFSET = 5000
MAX_LIST_WORKERS_LIMIT = 1000


def _peer_status(posture: FederationPosture, has_reported_tasks: bool) -> job_pb2.PeerStatus:
    """Translate a resource federation posture to the retired wire enum."""
    if posture is FederationPosture.LOCAL:
        return job_pb2.PEER_STATUS_NONE
    if posture is FederationPosture.REJECTED:
        return job_pb2.PEER_STATUS_REJECTED
    if has_reported_tasks:
        return job_pb2.PEER_STATUS_SYNCED
    if posture in (FederationPosture.QUEUED, FederationPosture.PENDING_ACCEPTANCE):
        return job_pb2.PEER_STATUS_PENDING_SCHEDULING
    return job_pb2.PEER_STATUS_ASSIGNED


def _federated_pending_reason(cluster: str, posture: FederationPosture, peer_status: int) -> str:
    """Pending reason for a federated job, which the local scheduler never sees.

    A handed-off job's tasks live on the peer and are excluded from the local
    fold, so the local scheduling diagnostic is meaningless for it. Derive the
    message from the handoff posture (single source of truth): waiting in the
    federation queue for a peer with free capacity, awaiting the peer's acceptance,
    awaiting its first status report, or pending on the peer once it has reported
    tasks. A queued job names a peer only when it is pinned to one.
    """
    if posture is FederationPosture.QUEUED:
        if not cluster:
            return "Queued for a federation peer to report free capacity"
        return f"Queued for peer {cluster} to report free capacity"
    if peer_status == job_pb2.PEER_STATUS_PENDING_SCHEDULING:
        return f"Awaiting acceptance by peer {cluster}"
    if peer_status == job_pb2.PEER_STATUS_ASSIGNED:
        return f"Handed off to peer {cluster}; awaiting first status report"
    return f"Pending on peer {cluster}"


def _filter_and_sort_workers(
    workers: list[tuple[Any, dict]],
    liveness_by_id: dict[WorkerId, WorkerLiveness],
    query: controller_pb2.Controller.WorkerQuery,
) -> list[tuple[Any, dict]]:
    """Apply the ``WorkerQuery`` contains filter and sort the cached roster.

    Filtering and sorting happen in Python against the cached worker roster
    rather than in SQL: the roster is bounded by cluster size (low thousands)
    and already cached on the controller, so the marginal cost of a re-scan
    per request is much smaller than reissuing the SELECT + worker_attributes
    fan-out.
    """
    needle = query.contains.lower() if query.contains else ""
    if needle:
        workers = [
            (w, attrs)
            for w, attrs in workers
            if needle in str(w.worker_id).lower() or (w.address and needle in w.address.lower())
        ]

    sort_field = query.sort_field or controller_pb2.Controller.WORKER_SORT_FIELD_WORKER_ID
    descending = query.sort_direction == controller_pb2.Controller.SORT_DIRECTION_DESC
    if sort_field == controller_pb2.Controller.WORKER_SORT_FIELD_LAST_HEARTBEAT:
        workers = sorted(workers, key=lambda wa: liveness_by_id[wa[0].worker_id].last_heartbeat_ms, reverse=descending)
    elif sort_field == controller_pb2.Controller.WORKER_SORT_FIELD_DEVICE_TYPE:
        # CPU workers persist with ``device_type == ""``; under ascending sort
        # they group first (treating CPU as the no-accelerator baseline).
        workers = sorted(workers, key=lambda wa: (wa[0].device_type, str(wa[0].worker_id)), reverse=descending)
    else:
        workers = sorted(workers, key=lambda wa: str(wa[0].worker_id), reverse=descending)
    return workers


def _resolve_state_filter(state_filter: str) -> tuple[int, ...] | None:
    """Resolve a ``JobQuery.state_filter`` string into concrete state ids.

    Returns ``USER_JOB_STATES`` when no filter is set, a single-element tuple
    when it matches a known user-visible state, or ``None`` when the filter
    does not match any known state (caller should return an empty page).
    """
    if not state_filter:
        return USER_JOB_STATES
    normalized = state_filter.lower()
    for st in USER_JOB_STATES:
        if job_state_friendly(st) == normalized:
            return (st,)
    return None


def _query_from_list_jobs_request(
    request: controller_pb2.Controller.ListJobsRequest,
) -> controller_pb2.Controller.JobQuery:
    """Return the request's ``JobQuery`` with paging clamped to safe bounds."""
    query = controller_pb2.Controller.JobQuery()
    if request.HasField("query"):
        query.CopyFrom(request.query)

    # Clamp paging: 0 (unset) or out-of-range values default to MAX. Unbounded
    # listing is not supported because downstream per-page work
    # (task_summaries_for_jobs, parent_ids_with_children) grows an IN-clause
    # with one placeholder per returned row.
    if query.limit <= 0 or query.limit > MAX_LIST_JOBS_LIMIT:
        query.limit = MAX_LIST_JOBS_LIMIT
    if query.offset < 0:
        query.offset = 0
    if query.offset > MAX_LIST_JOBS_OFFSET:
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            f"query.offset={query.offset} exceeds MAX_LIST_JOBS_OFFSET={MAX_LIST_JOBS_OFFSET}; "
            "narrow the result set with state_filter/name_filter/parent_job_id instead of paging deeper.",
        )
    return query


def _live_user_stats(users: UserOperations) -> list[UserStats]:
    """Aggregate job/task counts per user.

    The user set is every observed owner — everyone who has ever submitted a job
    (any state) plus anyone with a budget row — derived directly from those tables,
    never a ``users`` table (there is none). So the landing page lists people even
    when none of their jobs are currently active. The per-state counts only cover
    active (non-terminal) jobs/tasks, so the Running/Pending/Active columns reflect
    current load and an idle user shows all zeros rather than disappearing.
    """
    rows = users.live_user_stats()
    return [
        UserStats(
            user=row.user,
            task_state_counts=row.task_state_counts,
            job_state_counts=row.job_state_counts,
        )
        for row in rows
    ]


def _attempts_for_worker(
    workers: WorkerOperations, worker_id: WorkerId, limit: int = 50
) -> list[controller_pb2.Controller.WorkerTaskAttempt]:
    """Return per-attempt history for ``worker_id``, newest first.

    Indexed scan of ``task_attempts`` via ``idx_task_attempts_worker_task``;
    each retry of the same task is its own row so the dashboard can render
    independent state/duration per attempt rather than inheriting from the
    parent task (which produced bogus duplicate-RUNNING rows).
    """
    raw_rows, resources_by_job = workers.recent_worker_attempts(worker_id, limit)
    out: list[controller_pb2.Controller.WorkerTaskAttempt] = []
    for row in raw_rows:
        proto_attempt = job_pb2.TaskAttempt(
            attempt_id=row.attempt_id,
            worker_id=str(row.worker_id) if row.worker_id else "",
            state=row.state,
            exit_code=row.exit_code or 0,
            error=row.error or "",
            is_worker_failure=attempt_is_worker_failure(row.state),
            attempt_uid=row.attempt_uid,
            pod_name=row.pod_name or "",
            pod_uid=row.pod_uid or "",
            node_name=row.node_name or "",
            terminal_reason=row.terminal_reason or "",
        )
        if row.started_at_ms is not None:
            proto_attempt.started_at.CopyFrom(timestamp_to_proto(row.started_at_ms))
        if row.finished_at_ms is not None:
            proto_attempt.finished_at.CopyFrom(timestamp_to_proto(row.finished_at_ms))
        out.append(
            controller_pb2.Controller.WorkerTaskAttempt(
                task_id=row.task_id.to_wire(),
                attempt=proto_attempt,
                resources=(
                    resource_spec_to_proto(resources_by_job[row.task_id.parent])
                    if row.task_id.parent in resources_by_job
                    else None
                ),
            )
        )
    return out


class ControllerRuntimeProtocol(Protocol):
    """Protocol for controller operations used by LegacyControllerService."""

    def wake(self) -> None: ...

    def request_worker_eviction(self, worker_ids: Sequence[WorkerId]) -> None: ...

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None: ...

    def begin_checkpoint(self) -> tuple[str, Any]: ...

    @property
    def last_scheduling_context(self) -> SchedulingContext | None: ...

    @property
    def provider(self) -> Any: ...

    @property
    def backends(self) -> dict[str, TaskBackend]: ...

    @property
    def federation(self) -> FederationManager: ...

    @property
    def capabilities(self) -> frozenset[BackendCapability]: ...

    def backend_id_for_scale_group(self, scale_group: str) -> str: ...

    def all_liveness(self) -> dict[WorkerId, WorkerLiveness]: ...

    def liveness_for_worker(self, worker_id: WorkerId) -> WorkerLiveness: ...

    @property
    def last_unroutable_jobs(self) -> dict[str, str]: ...

    @property
    def scale_group_to_backend(self) -> dict[str, str]: ...


class LegacyControllerService:
    """Serve the retired ControllerService and operational RPC contract."""

    def __init__(
        self,
        runtime: ControllerRuntimeProtocol,
        bundle_store: BundleStore,
        log_client: LogClient,
        *,
        operations: OperationalServices,
        endpoint_service: EndpointServiceImpl,
        controller: Controller,
        auth: ControllerAuth | None = None,
        user_budget_defaults: UserBudgetDefaults | None = None,
    ):
        self._database_operations = operations.database
        self._federation_operations = operations.federation
        self._worker_operations = operations.workers
        self._task_operations = operations.tasks
        self._user_operations = operations.users
        self._scheduling_operations = operations.scheduling
        # The leased registry owns endpoint logic; the legacy
        # ControllerService.{Register,Unregister,List}Endpoint RPCs delegate here.
        self._endpoint_service = endpoint_service
        self._controller = controller
        self._runtime = runtime
        self._bundle_store = bundle_store
        self._log_client = log_client
        self._timer = Timer()
        self._auth = auth or ControllerAuth()
        self._user_budget_defaults = user_budget_defaults or UserBudgetDefaults()
        self._profile_table = self._log_client.get_table(PROFILE_NAMESPACE, IrisProfile)
        self._database_operations.attach_task_event_table(
            self._log_client.get_table(
                TASK_EVENT_NAMESPACE,
                TaskEventRow,
                storage_policy=TASK_EVENT_STORAGE_POLICY,
            )
        )

    def bundle_zip(self, bundle_id: str) -> bytes:
        return self._bundle_store.get(bundle_id)

    def blob_data(self, blob_id: str) -> bytes:
        return self._bundle_store.get(blob_id)

    def probe_database(self) -> int | None:
        """Return checkpoint ancestry after verifying controller state is readable."""
        return self._database_operations.probe_database()

    def _authorize_job_owner(self, job_id: JobName) -> None:
        """Raise PERMISSION_DENIED if the authenticated user doesn't own this job.

        Skipped when no auth provider is configured (null-auth mode).
        """
        if not self._auth.provider:
            return
        authorize_resource_owner(job_id.user)

    def _authorize_job_actor(self, job_id: JobName) -> None:
        """Authorize the caller to act on ``job_id`` (e.g. route a cancel).

        The job owner or an admin passes, as with :meth:`_authorize_job_owner`. A
        federation peer additionally passes for a job it federated here — its verified
        requester matches the job's received handle — so the parent can route a cancel
        for a handed-off job whose local owner it is not.
        """
        if not self._auth.provider:
            return
        identity = get_verified_identity()
        if identity is not None and identity.role == FEDERATION_PEER_ROLE:
            if self._federation_operations.received_requester(job_id) == identity.user_id:
                return
            raise ConnectError(Code.PERMISSION_DENIED, f"Peer {identity.user_id!r} did not federate job {job_id}")
        authorize_resource_owner(job_id.user)

    def _authorize_federated_debug_target(self, root_job: JobName) -> None:
        """Scope a federation peer's on-demand debug RPC to a job it federated here.

        ``authorize_method`` admits ProfileTask/ExecInContainer/GetProcessStatus for a
        ``FEDERATION_PEER_ROLE`` identity; this confirms ``root_job`` is one the peer
        actually handed off (matching its received handle), so a peer cannot profile,
        exec into, or inspect this cluster's own tasks. Non-peer callers pass through
        untouched — their access stays governed by ``authorize_method``'s role
        allowlist, so the read-only dashboard keeps reading any task's process status.
        """
        if not self._auth.provider:
            return
        identity = get_verified_identity()
        if identity is None or identity.role != FEDERATION_PEER_ROLE:
            return
        if self._federation_operations.received_requester(root_job) == identity.user_id:
            return
        raise ConnectError(Code.PERMISSION_DENIED, f"Peer {identity.user_id!r} did not federate job {root_job}")

    def launch_job(
        self,
        request: controller_pb2.Controller.LaunchJobRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.LaunchJobResponse:
        """Submit a Job and return its authority-selected ID."""
        try:
            spec = job_spec_from_legacy_request(request)
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        if request.HasField("federation"):
            handoff = request.federation
            identity = resource_call(
                lambda: self._controller.submit_federated_job(
                    spec,
                    request.bundle_blob,
                    FederationSubmission(
                        requester_id=handoff.requester_id,
                        owner_principal=handoff.owner_principal,
                        submitting_user=handoff.submitting_user,
                        handoff_nonce=handoff.handoff_nonce,
                    ),
                )
            )
        else:
            identity = resource_call(
                lambda: self._controller.submit_job(
                    spec,
                    request.bundle_blob,
                    enforce_client_freshness=ctx is not None,
                )
            )
        return controller_pb2.Controller.LaunchJobResponse(job_id=identity.key.resource_id)

    def get_job_status(
        self,
        request: controller_pb2.Controller.GetJobStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetJobStatusResponse:
        """Return current Job detail and its submitted request."""
        del ctx
        try:
            key = self._resource_job_summary(request.job_id).identity.key
            detail = self._controller.describe_job(key)
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found") from exc
        (observation,) = self._controller.observe_jobs((detail.summary,))
        return controller_pb2.Controller.GetJobStatusResponse(
            job=self._legacy_job_status(observation),
            request=redact_request_env_vars(job_spec_to_legacy_request(detail.spec)),
        )

    def _resource_tasks_for_job(self, key: ResourceKey) -> tuple:
        items = []
        page_token = None
        while True:
            page = self._controller.list_tasks(
                TaskQuery(job=key, page_size=_LEGACY_RESOURCE_PAGE_SIZE, page_token=page_token)
            )
            items.extend(page.items)
            page_token = page.next_page_token
            if page_token is None:
                return tuple(items)

    def _resource_job_summary(self, wire_id: str) -> JobSummary:
        page = self._controller.list_jobs(JobQuery(resource_id=wire_id, page_size=1))
        if not page.items:
            raise ResourceNotFound(wire_id)
        return page.items[0]

    def _resource_task_summary(self, wire_id: str) -> TaskSummary:
        job_id, _ = JobName.from_wire(wire_id).require_task()
        page_token = None
        while True:
            page = self._controller.list_tasks(
                TaskQuery(
                    job_id_prefix=job_id.to_wire(),
                    page_size=_LEGACY_RESOURCE_PAGE_SIZE,
                    page_token=page_token,
                )
            )
            for summary in page.items:
                if summary.identity.key.resource_id == wire_id:
                    return summary
            page_token = page.next_page_token
            if page_token is None:
                raise ResourceNotFound(wire_id)

    def _legacy_job_status(self, observation: JobObservation) -> job_pb2.JobStatus:
        summary = observation.summary
        status = job_status_to_legacy(
            summary,
            observation.tasks,
            has_children=observation.has_children,
            local_cluster_id=self._controller.cluster_id,
        )
        if observation.federation_posture is not FederationPosture.LOCAL:
            peer_id = "" if summary.execution_cluster_id == self._controller.cluster_id else summary.execution_cluster_id
            status.peer_status = _peer_status(observation.federation_posture, observation.tasks.task_count > 0)
            if summary.state == job_pb2.JOB_STATE_PENDING:
                status.pending_reason = _federated_pending_reason(
                    peer_id,
                    observation.federation_posture,
                    status.peer_status,
                )
        return status

    def get_job_state(
        self,
        request: controller_pb2.Controller.GetJobStateRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetJobStateResponse:
        """Return states for the requested Jobs that still exist."""
        del ctx
        try:
            states = self._controller.job_states(request.job_ids)
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        return controller_pb2.Controller.GetJobStateResponse(states=states)

    def terminate_job(
        self,
        request: controller_pb2.Controller.TerminateJobRequest,
        ctx: Any,
    ) -> job_pb2.Empty:
        """Cancel the current incarnation of the requested Job."""
        del ctx
        try:
            identity = self._resource_job_summary(request.job_id).identity
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found") from exc
        self._authorize_job_actor(JobName.from_wire(request.job_id))
        resource_call(
            lambda: self._controller.cancel_job(
                identity,
                idempotency_key=f"legacy-terminate:{identity.job_uid}",
                principal_id=JobName.from_wire(identity.key.resource_id).user,
            )
        )
        return job_pb2.Empty()

    def list_jobs(
        self,
        request: controller_pb2.Controller.ListJobsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListJobsResponse:
        """Return an offset-paged Job list matching the request filters."""
        del ctx
        query = _query_from_list_jobs_request(request)
        state_ids = _resolve_state_filter(query.state_filter)
        if state_ids is None:
            return controller_pb2.Controller.ListJobsResponse()
        parent_resource_id = None
        if query.scope == controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN:
            if not query.parent_job_id:
                raise ConnectError(Code.INVALID_ARGUMENT, "parent_job_id is required for child scope")
            parent_resource_id = query.parent_job_id
        execution_cluster_id = None
        if query.cluster:
            execution_cluster_id = self._controller.cluster_id if query.cluster == LOCAL_CLUSTER else query.cluster
        sort_field = _JOB_SORT_FROM_LEGACY.get(query.sort_field, JobSortField.DATE)
        if query.sort_direction == controller_pb2.Controller.SORT_DIRECTION_UNSPECIFIED:
            sort_direction = SortDirection.DESCENDING if sort_field is JobSortField.DATE else SortDirection.ASCENDING
        elif query.sort_direction == controller_pb2.Controller.SORT_DIRECTION_DESC:
            sort_direction = SortDirection.DESCENDING
        else:
            sort_direction = SortDirection.ASCENDING
        page = self._controller.list_job_inventory(
            JobInventoryQuery(
                scope=_JOB_SCOPE_FROM_LEGACY.get(query.scope, JobListScope.ALL),
                parent_resource_id=parent_resource_id,
                name_contains=query.name_filter or None,
                states=frozenset(JobState(state) for state in state_ids),
                sort_field=sort_field,
                sort_direction=sort_direction,
                offset=query.offset,
                limit=query.limit,
                job_id_prefix=query.job_id_prefix or None,
                backend_id=query.backend_id or None,
                execution_cluster_id=execution_cluster_id,
            )
        )
        statuses = [self._legacy_job_status(observation) for observation in page.items]
        return controller_pb2.Controller.ListJobsResponse(
            jobs=statuses,
            total_count=page.total_count,
            has_more=query.offset + len(statuses) < page.total_count,
        )

    def get_task_status(
        self,
        request: controller_pb2.Controller.GetTaskStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetTaskStatusResponse:
        """Return Task detail and its owning Job's resource request."""
        del ctx
        try:
            key = self._resource_task_summary(request.task_id).identity.key
            detail = self._controller.describe_task(key)
            job = self._controller.describe_job(detail.summary.job.key)
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found") from exc
        return controller_pb2.Controller.GetTaskStatusResponse(
            task=task_detail_to_legacy(
                detail,
                local_cluster_id=self._controller.cluster_id,
            ),
            job_resources=resource_spec_to_proto(job.spec.resources),
            root_cause_highlights=detail.root_cause_highlights,
        )

    def list_tasks(
        self,
        request: controller_pb2.Controller.ListTasksRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListTasksResponse:
        """Return every current Task for the requested Job."""
        del ctx
        if not request.job_id:
            raise ConnectError(Code.INVALID_ARGUMENT, "job_id is required")
        key = self._resource_job_summary(request.job_id).identity.key
        summaries = self._resource_tasks_for_job(key)
        details = tuple(
            detail
            for chunk in batched(summaries, _LEGACY_RESOURCE_PAGE_SIZE)
            for detail in self._controller.describe_tasks(tuple(summary.identity.key for summary in chunk))
        )
        tasks = [
            task_detail_to_legacy(
                detail,
                local_cluster_id=self._controller.cluster_id,
            )
            for detail in details
        ]
        return controller_pb2.Controller.ListTasksResponse(tasks=tasks)

    # --- Worker Management ---

    def register(
        self,
        request: controller_pb2.Controller.RegisterRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.RegisterResponse:
        """One-shot worker registration. Returns worker_id.

        Worker registers once, then waits for heartbeats from the controller.
        """
        if self._auth.provider is not None:
            authorize(AuthzAction.ACT_AS_WORKER)

        if not request.worker_id:
            logger.error("Worker at %s registered without worker_id", request.address)
            return controller_pb2.Controller.RegisterResponse(
                worker_id="",
                accepted=False,
            )
        worker_id = WorkerId(request.worker_id)

        backend = self._backend_for_id(self._runtime.backend_id_for_scale_group(request.scale_group))
        health = backend.health
        assert health is not None, f"worker {worker_id} registered into a scale group with no liveness tracker"
        self._worker_operations.register_worker(
            worker_id=worker_id,
            address=request.address,
            metadata=worker_metadata_from_proto(request.metadata),
            timestamp=Timestamp.now(),
            health=health,
            slice_id=request.slice_id,
            scale_group=request.scale_group,
        )
        self._request_recycled_address_eviction(worker_id, request.address)
        logger.info("Worker registered: %s at %s", worker_id, request.address)
        return controller_pb2.Controller.RegisterResponse(
            worker_id=str(worker_id),
            accepted=True,
        )

    def _request_recycled_address_eviction(self, worker_id: WorkerId, address: str) -> None:
        """Hand any stale prior owner of ``address`` to the controller for teardown.

        Detects a recycled internal IP and defers its stale owner to the controller.
        and defers the reap to :meth:`Controller.request_worker_eviction`.
        """
        stale = self._worker_operations.stale_workers_at_address(worker_id, address)
        if not stale:
            return
        logger.warning(
            "Worker %s registered at %s held by %d stale row(s) (recycled IP); evicting: %s",
            worker_id,
            address,
            len(stale),
            [str(wid) for wid in stale],
        )
        self._runtime.request_worker_eviction(stale)

    def list_workers(
        self,
        request: controller_pb2.Controller.ListWorkersRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListWorkersResponse:
        """List worker-daemon Nodes matching the Worker query."""
        del ctx
        query = request.query if request.HasField("query") else controller_pb2.Controller.WorkerQuery()
        nodes = []
        details = {}
        page_token = None
        while True:
            page, page_details = self._controller.list_nodes_with_details(
                NodeQuery(
                    backend_id=query.backend_id or None,
                    contains=None,
                    page_size=100,
                    page_token=page_token,
                )
            )
            nodes.extend(page.items)
            details.update(page_details)
            page_token = page.next_page_token
            if page_token is None:
                break
        nodes = [
            node
            for node in nodes
            if BackendCapability.WORKER_DAEMON in self._runtime.backends[node.identity.backend_id].capabilities
        ]
        if query.contains:
            needle = query.contains.casefold()
            nodes = [
                node
                for node in nodes
                if needle in node.identity.key.resource_id.casefold()
                or needle in (details[(node.identity.backend_id, node.identity.node_uid)].address or "").casefold()
            ]
        descending = query.sort_direction == controller_pb2.Controller.SORT_DIRECTION_DESC
        nodes.sort(key=lambda node: node.identity.key.resource_id, reverse=descending)
        total_count = len(nodes)
        offset = max(query.offset, 0)
        limit = min(max(query.limit, 0), MAX_LIST_WORKERS_LIMIT)
        selected = nodes[offset : offset + limit] if limit else nodes[offset:]
        workers = []
        for node in selected:
            detail = details[(node.identity.backend_id, node.identity.node_uid)]
            metadata = job_pb2.WorkerMetadata(
                hostname=node.identity.key.resource_id,
                ip_address=detail.address or "",
                cpu_count=node.capacity.cpu_millicores // 1_000,
                memory_bytes=node.capacity.memory_bytes,
                disk_bytes=node.capacity.disk_bytes,
            )
            if node.capacity.accelerator_kind == "gpu":
                metadata.device.gpu.variant = node.capacity.accelerator_variant
                metadata.device.gpu.count = node.capacity.accelerator_count
            elif node.capacity.accelerator_kind == "tpu":
                metadata.device.tpu.variant = node.capacity.accelerator_variant
                metadata.device.tpu.count = node.capacity.accelerator_count
            for attribute in detail.attributes:
                if attribute.kind is NodeAttributeKind.STRING:
                    metadata.attributes[attribute.key].string_value = attribute.string_value or ""
                elif attribute.kind is NodeAttributeKind.INTEGER:
                    metadata.attributes[attribute.key].int_value = attribute.integer_value or 0
                else:
                    metadata.attributes[attribute.key].float_value = attribute.float_value or 0.0
            workers.append(
                controller_pb2.Controller.WorkerHealthStatus(
                    worker_id=node.identity.key.resource_id,
                    healthy=node.health is NodeHealth.READY,
                    address=detail.address or "",
                    metadata=metadata,
                    status_message=node.health.value,
                    backend_id=node.identity.backend_id,
                    scale_group=node.scaling_group_id or "",
                    last_heartbeat=timestamp_to_proto(node.observed_at),
                )
            )
        return controller_pb2.Controller.ListWorkersResponse(
            workers=workers,
            total_count=total_count,
            has_more=bool(limit and offset + len(selected) < total_count),
        )

    @property
    def provider(self) -> TaskBackend:
        """The live execution backend (read-only handle for dashboard descriptors)."""
        return self._runtime.provider

    @property
    def backends(self) -> dict[str, TaskBackend]:
        """The controller's full backend collection (for the union capabilities descriptor)."""
        return self._runtime.backends

    def _backend_for_id(self, backend_id: str) -> TaskBackend:
        """Resolve a backend by id for per-task/-worker dispatch (profile, exec,
        process status), falling back to the representative backend when the id is
        empty or unknown — the single-backend case and any pre-routing rows."""
        return self._runtime.backends.get(backend_id) or self._runtime.provider

    def _federated_handle_for_task(self, task_id: JobName) -> FederatedRoute | None:
        """The SENT federated handle owning ``task_id``'s root job, or ``None`` if
        that root job runs locally."""
        return self._federation_operations.federated_handle(task_id.root_job)

    def _proxy_if_federated(self, task_id: JobName, call: Callable[[FederationPeer], _T]) -> _T | None:
        """Forward an on-demand RPC to its owning peer if ``task_id`` is federated.

        ``call`` invokes the matching typed method on the peer connection; the peer is
        authoritative, so its ``NOT_FOUND`` for a moved or finished task propagates
        back. Returns the peer's response, or ``None`` when the root job runs locally —
        the caller then resolves it against the local backend. The proxied responses
        are unary messages, never ``None``, so callers dispatch on ``is not None``.
        """
        handle = self._federated_handle_for_task(task_id)
        if handle is None:
            return None
        return peer_rpc_call(lambda: self._runtime.federation.proxy_to_peer(handle.peer_id, call))

    def _resolve_task_target(self, task: TaskWithAttempts, attempt_id: int, *, wire_name: str) -> TaskTarget:
        """Resolve a running task to a :class:`TaskTarget` for on-demand worker RPCs.

        CLUSTER_VIEW backends (K8s) route by task id with no worker; worker-daemon
        backends resolve and liveness-check the owning worker. Raises
        ``FAILED_PRECONDITION`` if the task is not yet placed and ``UNAVAILABLE`` if
        its worker is gone or unhealthy. Shared by ``profile_task`` and
        ``exec_in_container``.
        """
        # The K8s backend rebuilds the pod name from (task_id, attempt_id, uid);
        # the uid rides on the attempt rows already attached to ``task``.
        attempt_uid = next((a.attempt_uid for a in task.attempts if a.attempt_id == attempt_id), "")
        task_worker_id = _task_worker_id(task)
        if not task_worker_id:
            if BackendCapability.CLUSTER_VIEW not in self._runtime.capabilities:
                raise ConnectError(Code.FAILED_PRECONDITION, f"Task {wire_name} not yet assigned to a worker")
            return TaskTarget(
                task_id=task.task_id.to_wire(),
                attempt_id=attempt_id,
                worker_id=None,
                address=None,
                attempt_uid=attempt_uid,
            )
        worker = _read_worker(self._worker_operations, task_worker_id)
        if not worker or not self._runtime.liveness_for_worker(task_worker_id).healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {task_worker_id} is unavailable")
        return TaskTarget(
            task_id=task.task_id.to_wire(),
            attempt_id=attempt_id,
            worker_id=task_worker_id,
            address=worker.address,
            attempt_uid=attempt_uid,
        )

    @property
    def endpoint_service(self) -> EndpointServiceImpl:
        """The leased endpoint registry these RPCs delegate to (shared with the dashboard)."""
        return self._endpoint_service

    # --- Autoscaler ---

    def get_autoscaler_status(
        self,
        request: controller_pb2.Controller.GetAutoscalerStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetAutoscalerStatusResponse:
        """Get autoscaler status, merged across every backend's autoscaler.

        When ``request.backend_id`` is set, restricts the view to that one
        backend's autoscaler; empty merges all.
        """
        if BackendCapability.IRIS_AUTOSCALER not in self._runtime.capabilities:
            return controller_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())

        status = self._merge_autoscaler_status(only_backend_id=request.backend_id)
        return controller_pb2.Controller.GetAutoscalerStatusResponse(status=status)

    def _merge_autoscaler_status(self, only_backend_id: str = "") -> vm_pb2.AutoscalerStatus:
        """Merge each backend's authored autoscaler status into one.

        Merges every backend by default; ``only_backend_id`` restricts the view to
        a single backend. Each backend authors its own status — groups already
        tagged with its backend_id and every VM overlaid with usability/running-task
        counts — so this only concatenates; a backend with no autoscaler authors an
        empty status that contributes nothing. Each backend owns a disjoint set of
        scale groups (the single scale-group->backend key space), so group-keyed
        fields (``current_demand``, ``recent_actions``) need no further
        disambiguation. ``recent_actions`` are re-sorted newest-first and capped;
        each backend's ``last_routing_decision`` folds into one merged decision
        (disjoint groups, so the per-group fields concatenate).
        """
        merged = vm_pb2.AutoscalerStatus()
        last_evaluation = 0
        for backend_id, backend in self._runtime.backends.items():
            if only_backend_id and backend_id != only_backend_id:
                continue
            sub = autoscaler_status_to_proto(backend.autoscaler_status())
            merged.groups.extend(sub.groups)
            for key, value in sub.current_demand.items():
                merged.current_demand[key] = value
            merged.recent_actions.extend(sub.recent_actions)
            last_evaluation = max(last_evaluation, sub.last_evaluation.epoch_ms)
            if sub.HasField("last_routing_decision"):
                _accumulate_routing_decision(merged.last_routing_decision, sub.last_routing_decision)
        merged.recent_actions.sort(key=lambda action: action.timestamp.epoch_ms, reverse=True)
        del merged.recent_actions[_MERGED_AUTOSCALER_ACTIONS:]
        if last_evaluation:
            merged.last_evaluation.epoch_ms = last_evaluation
        return merged

    # --- Kubernetes Cluster Status ---

    def get_kubernetes_cluster_status(
        self,
        request: controller_pb2.Controller.GetKubernetesClusterStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Get Kubernetes cluster status: node counts, capacity, and recent pod statuses.

        Routes to the ``CLUSTER_VIEW`` backend named by ``request.backend_id``, or
        the sole such backend if there is exactly one; raises ``INVALID_ARGUMENT``
        when the choice is ambiguous.
        """
        cluster_view_backends = [
            (bid, backend)
            for bid, backend in sorted(self._runtime.backends.items())
            if BackendCapability.CLUSTER_VIEW in backend.capabilities
        ]

        if request.backend_id:
            for bid, backend in cluster_view_backends:
                if bid == request.backend_id:
                    status = backend.status().kubernetes
                    assert status is not None
                    return kubernetes_status_to_proto(status)
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"Backend {request.backend_id!r} does not exist or has no cluster view",
            )

        if len(cluster_view_backends) > 1:
            ids = ", ".join(bid for bid, _ in cluster_view_backends)
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"Multiple cluster-view backends ({ids}); specify backend_id in the request",
            )

        if cluster_view_backends:
            status = cluster_view_backends[0][1].status().kubernetes
            assert status is not None
            return kubernetes_status_to_proto(status)

        return controller_pb2.Controller.GetKubernetesClusterStatusResponse()

    # --- VM Logs ---

    # --- Profiling ---

    def profile_task(
        self,
        request: job_pb2.ProfileTaskRequest,
        ctx: RequestContext,
    ) -> job_pb2.ProfileTaskResponse:
        """Profile the controller, a Worker, or the current Task Attempt."""
        del ctx
        if not request.HasField("profile_type"):
            raise ConnectError(Code.INVALID_ARGUMENT, "profile_type is required")
        if request.target in ("/system/controller", "/system/process"):
            try:
                duration = request.duration_seconds or 10
                profile = profile_configuration_from_proto(request.profile_type)
                data = profile_local_process(duration, profile)
                if self._profile_table is not None:
                    self._profile_table.write(
                        [
                            build_profile_row(
                                source="/system/controller",
                                attempt_id=None,
                                vm_id="controller-self",
                                duration_seconds=duration,
                                profile=profile,
                                profile_data=data,
                            )
                        ]
                    )
                return job_pb2.ProfileTaskResponse(profile_data=data)
            except Exception as exc:
                return job_pb2.ProfileTaskResponse(error=str(exc))
        worker_id_text = _parse_worker_target(request.target)
        if worker_id_text is not None:
            worker = _read_worker(self._worker_operations, WorkerId(worker_id_text))
            if worker is None:
                raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id_text} not found")
            if not self._runtime.liveness_for_worker(worker.worker_id).healthy:
                raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id_text} is unavailable")
            backend = self._backend_for_id(self._runtime.backend_id_for_scale_group(str(worker.scale_group or "")))
            result = backend.profile_task(
                TaskTarget(task_id="", attempt_id=0, worker_id=worker.worker_id, address=worker.address),
                ProfileRequest(
                    attempt=None,
                    profile=profile_configuration_from_proto(request.profile_type),
                    duration=Duration.from_seconds(request.duration_seconds) if request.duration_seconds else None,
                ),
            )
            return job_pb2.ProfileTaskResponse(profile_data=result.profile_data, error=result.error_message)
        try:
            target = TaskAttempt.from_wire(request.target)
            target.task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        self._authorize_federated_debug_target(target.task_id.root_job)
        try:
            attempt = self._controller.describe_attempt(
                AttemptLocator(
                    self._resource_task_summary(target.task_id.to_wire()).identity.key,
                    target.attempt_id,
                )
            )
            result = peer_rpc_call(
                lambda: self._controller.profile_attempt(
                    attempt.summary.identity,
                    profile_configuration_from_proto(request.profile_type),
                    Duration.from_seconds(request.duration_seconds) if request.duration_seconds else None,
                )
            )
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.target} not found") from exc
        return job_pb2.ProfileTaskResponse(profile_data=result.profile_data, error=result.error_message)

    def list_users(
        self,
        request: controller_pb2.Controller.ListUsersRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListUsersResponse:
        """Return live per-user aggregate counts for the dashboard.

        The user set is derived from observed owners (jobs + budgets), never a
        ``users`` table; each user's role is resolved from the in-memory,
        config-derived :class:`RolePolicy` (no DB projection).
        """
        del request, ctx
        role_policy = self._auth.role_policy
        users = sorted(
            _live_user_stats(self._user_operations),
            key=lambda entry: (
                -_active_job_count(entry.job_state_counts),
                -(entry.task_state_counts.get(job_pb2.TASK_STATE_RUNNING, 0)),
                entry.user,
            ),
        )
        return controller_pb2.Controller.ListUsersResponse(
            users=[
                controller_pb2.Controller.UserSummary(
                    user=entry.user,
                    task_state_counts=_task_state_counts_for_summary(entry.task_state_counts),
                    job_state_counts=_job_state_counts_for_summary(entry.job_state_counts),
                    role=role_policy.role_for(entry.user) if role_policy else "",
                )
                for entry in users
            ]
        )

    # --- Worker Detail ---

    def get_worker_status(
        self,
        request: controller_pb2.Controller.GetWorkerStatusRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetWorkerStatusResponse:
        """Return detail for a single worker, keyed by worker ID.

        Workers and VMs are independent: the worker detail page shows only
        worker state (health, tasks, logs). VM status lives on the Autoscaler
        tab.
        """
        if BackendCapability.WORKER_DAEMON not in self._runtime.capabilities:
            raise ConnectError(Code.UNIMPLEMENTED, "Direct provider mode: no workers")
        if not request.id:
            raise ConnectError(Code.INVALID_ARGUMENT, "id is required")

        detail = _read_worker_detail(self._worker_operations, WorkerId(str(request.id)))
        if not detail:
            raise ConnectError(Code.NOT_FOUND, f"No worker found for '{request.id}'")

        worker = detail.worker
        liveness = self._runtime.liveness_for_worker(worker.worker_id)
        scale_group = str(worker.scale_group or "")
        worker_health = controller_pb2.Controller.WorkerHealthStatus(
            worker_id=worker.worker_id,
            healthy=liveness.healthy,
            consecutive_failures=liveness.consecutive_failures,
            last_heartbeat=timestamp_to_proto(Timestamp.from_ms(liveness.last_heartbeat_ms)),
            running_job_ids=[tid.to_wire() for tid in detail.running_tasks],
            address=worker.address,
            metadata=worker_metadata_to_proto(detail.metadata),
            status_message=worker_status_message(liveness),
            scale_group=scale_group,
            backend_id=self._runtime.backend_id_for_scale_group(scale_group),
        )

        # Worker daemon logs are NOT inlined here — when the worker is
        # unreachable the LogService proxy blocks for its full timeout
        # (~10s) and stalls the worker page render. The dashboard fetches
        # them in parallel via LogService.FetchLogs with
        # source=/system/worker/<worker_id>.
        recent_attempts = _attempts_for_worker(self._worker_operations, worker.worker_id, limit=50)

        resp = controller_pb2.Controller.GetWorkerStatusResponse(
            recent_attempts=recent_attempts,
        )
        resp.worker.CopyFrom(worker_health)
        return resp

    def begin_checkpoint(
        self,
        request: controller_pb2.Controller.BeginCheckpointRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.BeginCheckpointResponse:
        path, result = self._runtime.begin_checkpoint()
        resp = controller_pb2.Controller.BeginCheckpointResponse(
            checkpoint_path=path,
            job_count=result.job_count,
            task_count=result.task_count,
            worker_count=result.worker_count,
        )
        resp.created_at.CopyFrom(timestamp_to_proto(result.created_at))
        return resp

    def get_process_status(
        self,
        request: job_pb2.GetProcessStatusRequest,
        ctx: Any,
    ) -> job_pb2.GetProcessStatusResponse:
        """Return process info (no logs — use FetchLogs instead).

        Target routing (same convention as ProfileTask):
        - empty or /system/process: the controller process itself
        - /system/worker/<worker_id>: proxy to a specific worker
        - /job/.../task/N: the process serving that task — proxied to the owning
          peer for a federated task, else resolved against the local backend.
        """
        target = request.target
        if not target or target == "/system/process":
            return job_pb2.GetProcessStatusResponse(process_info=process_info_to_proto(get_process_status(self._timer)))

        # Parse /system/worker/<worker_id>
        worker_id = _parse_worker_target(target)
        if worker_id is None:
            return self._task_process_status(target)

        worker = _read_worker(self._worker_operations, WorkerId(worker_id))
        if not worker:
            raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id} not found")
        if not self._runtime.liveness_for_worker(worker.worker_id).healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")

        process_target = TaskTarget(
            task_id="",
            attempt_id=0,
            worker_id=WorkerId(worker_id),
            address=worker.address,
        )
        try:
            worker_backend = self._backend_for_id(
                self._runtime.backend_id_for_scale_group(str(worker.scale_group or ""))
            )
            process_info = worker_backend.get_process_status(process_target)
            return job_pb2.GetProcessStatusResponse(process_info=process_info_to_proto(process_info))
        except ProviderError as exc:
            raise ConnectError(Code.UNAVAILABLE, str(exc)) from exc

    def _task_process_status(self, target: str) -> job_pb2.GetProcessStatusResponse:
        """Process status for a ``/job/.../task/N`` target.

        A federated task's subtree runs on a peer, so the request is proxied through
        the peer controller before any local resolution. Otherwise the owning backend
        reports it: a worker-daemon backend returns the worker hosting the task; the
        K8s backend reads the task pod's PID 1.
        """
        try:
            task_id = JobName.from_wire(target)
            task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, f"Invalid target: {target}") from exc

        self._authorize_federated_debug_target(task_id.root_job)
        task = _read_task_with_attempts(self._task_operations, task_id)
        if not task:
            raise ConnectError(Code.NOT_FOUND, f"Task {target} not found")

        proxied = self._proxy_if_federated(task_id, lambda peer: peer.get_process_status(target))
        if proxied is not None:
            return job_pb2.GetProcessStatusResponse(process_info=process_info_to_proto(proxied))

        task_target = self._resolve_task_target(task, task.current_attempt_id, wire_name=target)
        try:
            process_info = self._backend_for_id(str(task.backend_id or "")).get_process_status(task_target)
            return job_pb2.GetProcessStatusResponse(process_info=process_info_to_proto(process_info))
        except ProviderError as exc:
            raise ConnectError(Code.UNAVAILABLE, str(exc)) from exc

    # ── Auth RPCs ────────────────────────────────────────────────────────

    def mint_endpoint_token(
        self,
        request: controller_pb2.Controller.MintEndpointTokenRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.MintEndpointTokenResponse:
        """Mint a capability token for the named Endpoint."""
        del ctx
        matches = self._controller.list_endpoints(EndpointQuery(name_prefix=request.endpoint_name, page_size=100))
        endpoint = next((item for item in matches.items if item.name == request.endpoint_name), None)
        if endpoint is None:
            raise ConnectError(Code.NOT_FOUND, f"No endpoint {request.endpoint_name!r}")
        token = resource_call(
            lambda: self._controller.mint_endpoint_token(
                endpoint.key,
                duration_from_proto(request.ttl) if request.HasField("ttl") else None,
            )
        )
        return controller_pb2.Controller.MintEndpointTokenResponse(
            token=token.token,
            expires_at=timestamp_to_proto(token.expires_at),
            capability_url=token.capability_url,
        )

    def get_current_user(
        self,
        request: job_pb2.GetCurrentUserRequest,
        ctx: Any,
    ) -> job_pb2.GetCurrentUserResponse:
        identity = get_verified_identity()
        if identity is None:
            return job_pb2.GetCurrentUserResponse(user_id="anonymous", role="")
        return job_pb2.GetCurrentUserResponse(
            user_id=identity.user_id,
            role=identity.role,
        )

    def exec_in_container(
        self,
        request: controller_pb2.Controller.ExecInContainerRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        """Run a command in the current Attempt for the requested Task."""
        del ctx
        try:
            task_id = JobName.from_wire(request.task_id)
            task_id.require_task()
        except ValueError as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        self._authorize_federated_debug_target(task_id.root_job)
        try:
            task = self._resource_task_summary(request.task_id)
            if task.current_attempt is None:
                raise ResourceNotFound(request.task_id)
            result = peer_rpc_call(
                lambda: self._controller.exec_attempt(
                    task.current_attempt,
                    tuple(request.command),
                    Duration.from_seconds(request.timeout_seconds) if request.timeout_seconds else None,
                )
            )
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found") from exc
        return controller_pb2.Controller.ExecInContainerResponse(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error_message,
        )

    def execute_raw_query(
        self,
        request: query_pb2.RawQueryRequest,
        ctx: Any,
    ) -> query_pb2.RawQueryResponse:
        identity = require_identity()
        if identity.role != "admin":
            raise ConnectError(Code.PERMISSION_DENIED, "admin role required for raw queries")

        # The read snapshot connection sets ``PRAGMA query_only = ON``, but a
        # query of the form ``PRAGMA query_only = OFF; UPDATE ...`` flips it
        # back before the snapshot rejects anything. Reject up front: only
        # statements whose first token is ``SELECT`` are permitted.
        if request.sql.lstrip()[:6].upper() != "SELECT":
            raise ConnectError(Code.INVALID_ARGUMENT, "only SELECT statements are allowed")

        result = self._database_operations.raw_query(request.sql)
        columns = [query_pb2.ColumnMeta(name=name, type="unknown") for name in result.columns]
        rows = [json.dumps([_encode_query_cell(value) for value in row]) for row in result.rows]

        return query_pb2.RawQueryResponse(
            columns=columns,
            rows=rows,
        )

    def set_user_budget(
        self,
        request: controller_pb2.Controller.SetUserBudgetRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.SetUserBudgetResponse:
        """Set budget limit and max band for a user. Admin-only."""
        authorize(AuthzAction.MANAGE_BUDGETS)
        if not request.user_id:
            raise ConnectError(Code.INVALID_ARGUMENT, "user_id is required")
        max_band = request.max_band or job_pb2.PRIORITY_BAND_INTERACTIVE
        if max_band not in (
            job_pb2.PRIORITY_BAND_PRODUCTION,
            job_pb2.PRIORITY_BAND_INTERACTIVE,
            job_pb2.PRIORITY_BAND_BATCH,
        ):
            raise ConnectError(Code.INVALID_ARGUMENT, f"Invalid max_band: {request.max_band}")
        self._user_operations.set_user_budget(request.user_id, request.budget_limit, max_band, Timestamp.now())
        return controller_pb2.Controller.SetUserBudgetResponse()

    def get_user_budget(
        self,
        request: controller_pb2.Controller.GetUserBudgetRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetUserBudgetResponse:
        """Get budget config and current spend for a user."""
        require_identity()
        if not request.user_id:
            raise ConnectError(Code.INVALID_ARGUMENT, "user_id is required")
        budget = self._user_operations.user_budget(request.user_id)
        if budget is None:
            raise ConnectError(Code.NOT_FOUND, f"No budget found for user {request.user_id}")
        return controller_pb2.Controller.GetUserBudgetResponse(
            user_id=budget.user_id,
            budget_limit=budget.budget_limit,
            budget_spent=budget.budget_spent,
            max_band=budget.max_band,
        )

    def list_user_budgets(
        self,
        request: controller_pb2.Controller.ListUserBudgetsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListUserBudgetsResponse:
        """List all user budgets with current spend."""
        require_identity()
        budgets = self._user_operations.user_budgets()
        users = []
        for b in budgets:
            users.append(
                controller_pb2.Controller.GetUserBudgetResponse(
                    user_id=b.user_id,
                    budget_limit=b.budget_limit,
                    budget_spent=b.budget_spent,
                    max_band=b.max_band,
                )
            )
        return controller_pb2.Controller.ListUserBudgetsResponse(users=users)

    def get_scheduler_state(
        self,
        request: controller_pb2.Controller.GetSchedulerStateRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.GetSchedulerStateResponse:
        """Return aggregated scheduler state for the dashboard.

        The dashboard SchedulerTab + AutoscalerTab consume rolled-up counts:
        per-(band, user, job) for pending and per-(band, user, worker, job)
        for running. Aggregation runs server-side and emits one proto entry
        per bucket rather than per task.
        """
        require_identity()

        inputs = self._scheduling_operations.scheduler_state_inputs()
        budgets = inputs.budgets
        budget_limits: dict[str, int] = {budget.user_id: budget.budget_limit for budget in budgets}
        user_spend = inputs.user_spend
        pending_rows = inputs.pending_rows
        pending_requested_bands = inputs.pending_requested_bands
        running_rows = inputs.running_rows

        # Aggregate pending into (band, user, job, backend_id) → count buckets.
        pending_counts: dict[tuple[int, str, str, str], int] = {}
        total_pending = 0
        for row in pending_rows:
            user_id = row.task_id.user
            eff_band = compute_effective_band(
                pending_requested_bands.get(row.job_id, row.priority_band),
                user_id,
                user_spend,
                budget_limits,
                self._user_budget_defaults,
            )
            job_id = (row.task_id.parent or row.task_id).to_wire()
            backend_id = str(row.backend_id or "")
            key = (eff_band, user_id, job_id, backend_id)
            pending_counts[key] = pending_counts.get(key, 0) + 1
            total_pending += 1

        # Aggregate running into (band, user, worker, job, backend_id) → count buckets.
        # Use the stamped ``tasks.priority_band`` directly: the scheduler stamps the
        # effective band at assign time (see ``_commit_assignments``), so re-running
        # ``compute_effective_band`` here against current spend would double-demote.
        running_counts: dict[tuple[int, str, str, str, str], int] = {}
        total_running = 0
        for row in running_rows:
            user_id = row.task_id.user
            job_id = (row.task_id.parent or row.task_id).to_wire()
            backend_id = str(row.backend_id or "")
            key = (row.priority_band, user_id, str(row.worker_id), job_id, backend_id)
            running_counts[key] = running_counts.get(key, 0) + 1
            total_running += 1

        # Synthesize budget rows for users with active spend but no explicit
        # user_budgets entry; the dashboard renders their utilization from
        # UserBudgetDefaults instead of '-'.
        budget_protos: list[controller_pb2.Controller.SchedulerUserBudget] = []
        defaults = self._user_budget_defaults
        seen_users = {b.user_id for b in budgets}
        budget_rows: list[tuple[str, int, int]] = [(b.user_id, b.budget_limit, b.max_band) for b in budgets]
        for uid in user_spend:
            if uid not in seen_users:
                budget_rows.append((uid, defaults.budget_limit, defaults.max_band))
        for user_id, budget_limit, max_band in budget_rows:
            spent = user_spend.get(user_id, 0)
            utilization = (spent / budget_limit * 100.0) if budget_limit > 0 else 0.0
            # Probe with INTERACTIVE so the dashboard sees whether this user is
            # currently downgraded.
            eff = compute_effective_band(
                job_pb2.PRIORITY_BAND_INTERACTIVE,
                user_id,
                user_spend,
                budget_limits,
                self._user_budget_defaults,
            )
            budget_protos.append(
                controller_pb2.Controller.SchedulerUserBudget(
                    user_id=user_id,
                    budget_limit=budget_limit,
                    budget_spent=spent,
                    max_band=max_band,
                    effective_band=eff,
                    utilization_percent=utilization,
                )
            )

        pending_buckets = [
            controller_pb2.Controller.PendingTaskBucket(
                band=band,
                user_id=user_id,
                job_id=job_id,
                backend_id=backend_id,
                count=count,
            )
            for (band, user_id, job_id, backend_id), count in pending_counts.items()
        ]
        running_buckets = [
            controller_pb2.Controller.RunningTaskBucket(
                band=band,
                user_id=user_id,
                worker_id=worker_id,
                job_id=job_id,
                backend_id=backend_id,
                count=count,
            )
            for (band, user_id, worker_id, job_id, backend_id), count in running_counts.items()
        ]

        return controller_pb2.Controller.GetSchedulerStateResponse(
            user_budgets=budget_protos,
            total_pending=total_pending,
            total_running=total_running,
            pending_buckets=pending_buckets,
            running_buckets=running_buckets,
        )

    def list_backends(
        self,
        request: controller_pb2.Controller.ListBackendsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListBackendsResponse:
        """List all backends with aggregate task/worker statistics.

        Counts come from grouped SQL queries joined in Python; capacity health is
        read from the in-memory autoscaler snapshot, not the DB.
        """
        require_identity()

        backends = self._runtime.backends
        sg_to_backend = self._runtime.scale_group_to_backend

        # Invert sg_to_backend: backend_id → list[scale_group]
        backend_to_sgs: dict[str, list[str]] = {bid: [] for bid in backends}
        for sg, bid in sg_to_backend.items():
            if bid in backend_to_sgs:
                backend_to_sgs[bid].append(sg)

        counts = self._scheduling_operations.backend_counts(job_pb2.TASK_STATE_PENDING, job_pb2.TASK_STATE_RUNNING)
        pending_by_backend = counts.pending_by_backend
        running_by_backend = counts.running_by_backend
        worker_counts_by_scale_group = counts.workers_by_scale_group

        worker_count_by_backend: dict[str, int] = {bid: 0 for bid in backends}
        for scale_group, count in worker_counts_by_scale_group.items():
            bid = self._runtime.backend_id_for_scale_group(scale_group)
            worker_count_by_backend[bid] = worker_count_by_backend.get(bid, 0) + count

        summaries: list[controller_pb2.Controller.BackendSummary] = []
        for backend_id, backend in sorted(backends.items()):
            caps = backend.capabilities
            if BackendCapability.CLUSTER_VIEW in caps:
                kind = "kubernetes"
            elif BackendCapability.WORKER_DAEMON in caps:
                kind = "worker-daemon"
            else:
                kind = "unknown"

            adv: dict[str, set[str]] = backend.advertised_attributes()

            # Each backend authors its own expanded status variant in full: a
            # worker-daemon backend reads its own liveness tracker and running-task
            # rows to stamp health counts, per-VM usability, and the backend_id on its
            # autoscaler groups. The controller renders the result verbatim — the
            # Backends tab's detail panel shows whichever variant the backend selected,
            # and the per-group capacity-health tally is read off the same authored view.
            backend_status = backend.status()
            variant = "kubernetes" if backend_status.kubernetes is not None else "worker"

            cap_health: dict[str, int] = {}
            if variant == "worker":
                for group in backend_status.worker.autoscaler.groups:
                    st = group.availability_status or "unknown"
                    cap_health[st] = cap_health.get(st, 0) + 1

            summary = controller_pb2.Controller.BackendSummary(
                backend_id=backend_id,
                name=backend.name,
                kind=kind,
                capabilities=sorted(c.value for c in caps),
                scale_groups=sorted(backend_to_sgs.get(backend_id, [])),
                worker_count=worker_count_by_backend.get(backend_id, 0),
                pending_task_count=pending_by_backend.get(backend_id, 0),
                running_task_count=running_by_backend.get(backend_id, 0),
                has_autoscaler=backend.autoscaler is not None,
                capacity_health=cap_health,
            )
            # advertised_attributes is a proto map<string, StringList> (message
            # values), which doesn't support dict-style assignment/update; populate
            # each entry's repeated field in place.
            for key, values in adv.items():
                summary.advertised_attributes[key].values.extend(sorted(values))

            # Capacity metric for federation queueing and the dashboard. A backend
            # that supplies it fills availability even when empty (authoritative
            # zero); one that returns None leaves it UNSET so a peer falls back to
            # shape-only federation. observation_epoch_ms is the generation the
            # parent's reservation ledger keys on.
            capacity = backend.resource_capacity()
            if capacity is not None:
                summary.availability.version = AVAILABILITY_METRIC_VERSION
                summary.availability.observation_epoch_ms = Timestamp.now().epoch_ms()
                held_by_band: dict[int, dict[str, int]] = {}
                for token, device_capacity in capacity.items():
                    summary.availability.amounts[token] = device_capacity.free
                    summary.availability.total_amounts[token] = device_capacity.total
                    for band, amount in device_capacity.held_by_band.items():
                        held_by_band.setdefault(band, {})[token] = amount
                for band, amounts in sorted(held_by_band.items()):
                    summary.availability.held_by_band.add(band=band, amounts=amounts)

            encoded_status = backend_status_to_proto(backend_status)
            if variant == "kubernetes":
                summary.detail.kubernetes.CopyFrom(encoded_status.kubernetes)
            else:
                summary.detail.worker.CopyFrom(encoded_status.worker)

            summaries.append(summary)

        unroutable = self._runtime.last_unroutable_jobs
        sample = [
            controller_pb2.Controller.UnroutableJob(job_id=jid, reason=reason)
            for jid, reason in list(unroutable.items())[:_UNROUTABLE_SAMPLE_SIZE]
        ]

        return controller_pb2.Controller.ListBackendsResponse(
            backends=summaries,
            unroutable_job_count=len(unroutable),
            unroutable_sample=sample,
        )

    def list_peers(
        self,
        request: controller_pb2.Controller.ListPeersRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListPeersResponse:
        """List federation peers this controller may delegate whole jobs to.

        Each summary carries the peer's identity, controller/dashboard addresses,
        and its last capability-heartbeat result: reachability plus the peer's
        forwarded backends.
        """
        require_identity()
        return controller_pb2.Controller.ListPeersResponse(
            peers=[peer_observation_to_legacy(peer) for peer in self._runtime.federation.peer_summaries()]
        )

    def federation_sync(
        self,
        request: controller_pb2.Controller.FederationSyncRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.FederationSyncResponse:
        """Peer side: report jobs ``requester_id`` handed off, changed since its cursor.

        On first contact (empty cursor) or a cursor below the retained changelog
        window, returns the requester's full active set with ``cursor_stale`` so the
        parent set-replaces. Otherwise returns the incremental set: one delta per job
        whose changelog rows advanced past the cursor — a tombstone for a pruned job,
        else the job's summary plus its changed tasks. Assembled in one snapshot.
        """
        # Requester binding: a federation peer may sync only the jobs IT handed off —
        # its verified identity is the requester. A local admin (loopback/trusted) may
        # sync on any requester's behalf; any other identity is denied, so an ordinary
        # authenticated user cannot read another requester's federated set.
        identity = require_identity()
        requester_id = request.requester_id
        if identity.role == FEDERATION_PEER_ROLE:
            if requester_id != identity.user_id:
                raise ConnectError(
                    Code.PERMISSION_DENIED,
                    f"Peer {identity.user_id!r} may not sync jobs for requester {requester_id!r}",
                )
        elif identity.role != "admin":
            raise ConnectError(Code.PERMISSION_DENIED, "federation_sync requires a federation-peer or admin identity")
        batch = self._federation_operations.federation_batch(
            requester_id,
            request.cursor,
            backend_ids=tuple(self._runtime.backends),
        )
        return federation_batch_to_legacy(batch)
