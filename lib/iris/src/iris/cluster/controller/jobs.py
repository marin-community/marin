# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operations for job admission, observation, and lifecycle."""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.server_auth import ANONYMOUS_ADMIN, VerifiedIdentity, get_verified_identity
from rigging.timing import Duration, ExponentialBackoff, Timestamp
from sqlalchemy import bindparam, func, select

from iris.cluster.bundle import MAX_BUNDLE_SIZE_BYTES, BundleStore
from iris.cluster.config import user_admitted
from iris.cluster.constraints import (
    Constraint,
    cluster_directive,
    constraints_from_resources,
    merge_constraints,
    validate_tpu_request,
)
from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.auth import ControllerAuth, authorize_owner_if_configured
from iris.cluster.controller.autoscaler.status import PendingHint
from iris.cluster.controller.backend import BackendCapability, BackendObservation, JobFeasibilityRequest, TaskBackend
from iris.cluster.controller.budget import budget_user_id
from iris.cluster.controller.codec import reconstruct_launch_job_request, resource_spec_from_job_row
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.reads import TaskJobSummary
from iris.cluster.controller.reconcile.policy import MAX_ACTIVE_TASKS_PER_USER
from iris.cluster.controller.schema import jobs_table, tasks_table
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.router import RoutingRequest, SubmitDisposition, SubmitPlan
from iris.cluster.federation.store import HandoffState
from iris.cluster.redaction import redact_request_env_vars
from iris.cluster.types import (
    LOCAL_ADMIN_SUBMITTER,
    TERMINAL_JOB_STATES,
    USER_JOB_STATES,
    JobName,
    UserBudgetDefaults,
    is_federated,
    is_job_finished,
)
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE, AuthzAction, authorize, authorize_resource_owner
from iris.rpc.proto_display import (
    ADMIN_PRIORITY_BAND_VALUES,
    PRIORITY_BAND_VALUES,
    job_state_friendly,
    priority_band_name,
    priority_band_rank,
    resolve_container_profile,
    task_state_friendly,
)
from iris.time_proto import timestamp_to_proto
from iris.version import client_revision_date

logger = logging.getLogger(__name__)

WORKDIR_FILE_OFFLOAD_THRESHOLD = 10 * 1024
_JOB_REPLACEMENT_DRAIN_WAIT = Duration.from_seconds(120)
_LOCAL_ADMIN_FEDERATION_DENIED = (
    "A local_admin (CIDR/loopback) identity cannot submit a federated job. "
    "Federating to a remote cluster requires an authenticated user — log in via "
    "IAP or present a user token so the submission carries your identity."
)
_SUBMITTABLE_PRIORITY_BANDS = frozenset((job_pb2.PRIORITY_BAND_INHERIT, *PRIORITY_BAND_VALUES))
FRESHNESS_WINDOW = timedelta(days=14)
MAX_LIST_JOBS_LIMIT = 500
MAX_LIST_JOBS_OFFSET = 5000
_PROTO_INT32_MAX = (1 << 31) - 1
_EMPTY_TASK_SUMMARY = TaskJobSummary(job_id=JobName.from_wire("/_/_empty"))


class JobRuntime(Protocol):
    @property
    def backend(self) -> TaskBackend: ...

    @property
    def backend_observation(self) -> BackendObservation: ...

    @property
    def federation(self) -> FederationManager: ...

    def wake(self) -> None: ...

    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class JobDependencies:
    db: ControllerDB
    runtime: JobRuntime
    bundles: BundleStore
    auth: ControllerAuth
    user_budget_defaults: UserBudgetDefaults


def submitting_user_for_root(
    identity: VerifiedIdentity | None, request: controller_pb2.Controller.LaunchJobRequest
) -> str:
    """The authenticated principal to attribute a *root* submission to.

    A received handoff carries the submitter as a signed claim the receiving peer
    already re-checked against the presented token, so it is authoritative here. An
    IAP/JWT caller is its verified email; a CIDR/loopback caller authenticates as the
    anonymous admin (a machine, not a person) and is attributed to ``local_admin``.
    Child jobs do not call this — they inherit their root's value at insert time.
    """
    if request.HasField("federation"):
        return request.federation.submitting_user
    if identity is None or identity.user_id == ANONYMOUS_ADMIN.user_id:
        return LOCAL_ADMIN_SUBMITTER
    return identity.user_id


def _child_federation_refusal(job_id: JobName, peer_id: str) -> str:
    """The message refusing to federate child ``job_id`` to ``peer_id``, naming the remedy."""
    return (
        f"Job {job_id} requests a shape no local backend provides, and peer {peer_id!r} advertises it, "
        "but only whole root jobs are federated to a peer — a child job stays on the cluster that "
        f"runs its parent. Submit the root job to {peer_id!r} instead, so its whole tree runs there: "
        f"iris job run --target-cluster {peer_id} -- <command>"
    )


def _check_client_freshness(client_date_str: str, controller_date_str: str, today: date) -> None:
    """Reject root LaunchJob submissions built long before the controller's own marin-iris.

    Both dates come from `iris.version.client_revision_date`: the stamp baked into
    an image or wheel, the last commit touching the iris tree in a source checkout.
    Comparing the two anchors staleness to the code the cluster actually runs, so a
    quiet week in the iris tree cannot strand a client that is current with it.

    A controller that cannot identify its own build measures from today instead,
    which is the rule that predates the stamp. Every image built before the stamp
    existed lands here, so holding the old behavior for them keeps the gate
    enforced across the rollout rather than silently opening it.

    An empty client date means that build cannot identify itself, which leaves no
    basis for a verdict, so the gate does not apply to it.
    """
    if not client_date_str:
        return
    try:
        client_date = date.fromisoformat(client_date_str)
    except ValueError as err:
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            f"client_revision_date must be ISO YYYY-MM-DD, got {client_date_str!r}",
        ) from err
    reference = date.fromisoformat(controller_date_str) if controller_date_str else today
    floor = reference - FRESHNESS_WINDOW
    if client_date < floor:
        measured = (
            f"this controller runs {controller_date_str}" if controller_date_str else "today is " + today.isoformat()
        )
        raise ConnectError(
            Code.FAILED_PRECONDITION,
            f"marin-iris client is too old: your build is {client_date.isoformat()}, "
            f"{measured}, and the oldest client accepted is {floor.isoformat()} "
            f"({FRESHNESS_WINDOW.days} days). A build date is the last commit touching "
            f"lib/iris. From a checkout, merge or rebase onto a newer main; from an "
            f"installed marin-iris, upgrade it and re-run `uv sync`.",
        )


def _clamp_int32(value: int, *, job_id: JobName, field: str) -> int:
    if value > _PROTO_INT32_MAX:
        logger.warning(
            "JobStatus.%s for %s overflowed int32 (%d > %d); clamping. "
            "Investigate the upstream counter — this usually means a task row "
            "has a corrupted failure_count/preemption_count.",
            field,
            job_id.to_wire(),
            value,
            _PROTO_INT32_MAX,
        )
        return _PROTO_INT32_MAX
    return value


def job_status_counts(
    summary: TaskJobSummary | None, job_id: JobName, *, pre_sync_task_count: int = 0
) -> dict[str, Any]:
    """Return the clamped int32 counter fields for a ``JobStatus``.

    Spread into ``JobStatus(...)`` as ``**job_status_counts(summary, job_id)``.
    A ``None`` summary collapses to all-zero counters (no log noise); a real
    summary runs each field through ``_clamp_int32`` so 64-bit aggregates
    never trip the proto encoder.

    ``pre_sync_task_count`` is the requested replica count of a federated job
    (``jobs.num_tasks``). A handed-off job has no local task rows until the first
    FederationSync mirrors the peer's set, so with a ``None`` summary it is
    surfaced as ``task_count`` — the job reads "N tasks, awaiting the peer"
    instead of an unexplained empty table.
    """
    s = summary or _EMPTY_TASK_SUMMARY
    task_count = s.task_count if summary is not None else pre_sync_task_count
    return {
        "failure_count": _clamp_int32(s.failure_count, job_id=job_id, field="failure_count"),
        "preemption_count": _clamp_int32(s.preemption_count, job_id=job_id, field="preemption_count"),
        "task_count": _clamp_int32(task_count, job_id=job_id, field="task_count"),
        "completed_count": _clamp_int32(s.completed_count, job_id=job_id, field="completed_count"),
        "task_state_counts": {
            task_state_friendly(state): _clamp_int32(count, job_id=job_id, field=f"task_state_counts[{state}]")
            for state, count in s.task_state_counts.items()
        },
    }


def peer_status(cluster: str, handoff_state: int | None, has_reported_tasks: bool) -> int:
    """The ``PeerStatus`` for a job, derived from its cluster coordinate, its
    ``federated_jobs.handoff_state``, and whether the peer has mirrored any task
    rows back yet.

    Branch order matters: a peer that has reported tasks is ``SYNCED`` even if
    the local handle still reads ``PENDING_HANDOFF``. The sync loop can mirror a
    job's state before a transient RPC failure lets ``mark_handed_off`` run, so
    the presence of mirrored task rows is the more current signal — checking it
    before the handle avoids labelling a running, populated job "awaiting the
    peer's acceptance". A rejected handoff never reports tasks, so ``REJECTED``
    is checked first.

    For a terminal job this is the last posture observed (e.g. a handoff
    cancelled before delivery stays ``PENDING_SCHEDULING``). ``handoff_state`` is
    ``None`` only if the SENT handle is gone; the handle and jobs row are created
    and CASCADE-deleted together, so for a live federated job that is a
    can't-happen fallback, treated as handed off.
    """
    if not is_federated(cluster):
        return job_pb2.PEER_STATUS_NONE
    if handoff_state == int(HandoffState.HANDOFF_REJECTED):
        return job_pb2.PEER_STATUS_REJECTED
    if has_reported_tasks:
        return job_pb2.PEER_STATUS_SYNCED
    # QUEUED (awaiting a peer with capacity) and PENDING (awaiting the peer's ack) are
    # both pre-registration on the peer, so both read as PENDING_SCHEDULING.
    if handoff_state in (int(HandoffState.PENDING_HANDOFF), int(HandoffState.QUEUED_HANDOFF)):
        return job_pb2.PEER_STATUS_PENDING_SCHEDULING
    return job_pb2.PEER_STATUS_ASSIGNED


def _federated_pending_reason(cluster: str, handoff_state: int | None, peer_status: int) -> str:
    """Pending reason for a federated job, which the local scheduler never sees.

    A handed-off job's tasks live on the peer and are excluded from the local
    fold, so the local scheduling diagnostic is meaningless for it. Derive the
    message from the handoff posture (single source of truth): waiting in the
    federation queue for a peer with free capacity, awaiting the peer's acceptance,
    awaiting its first status report, or pending on the peer once it has reported
    tasks. A queued job names a peer only when it is pinned to one.
    """
    if handoff_state == int(HandoffState.QUEUED_HANDOFF):
        if not cluster:
            return "Queued for a federation peer to report free capacity"
        return f"Queued for peer {cluster} to report free capacity"
    if peer_status == job_pb2.PEER_STATUS_PENDING_SCHEDULING:
        return f"Awaiting acceptance by peer {cluster}"
    if peer_status == job_pb2.PEER_STATUS_ASSIGNED:
        return f"Handed off to peer {cluster}; awaiting first status report"
    return f"Pending on peer {cluster}"


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


def _query_jobs(
    tx,
    query: controller_pb2.Controller.JobQuery,
    state_ids: tuple[int, ...],
) -> tuple[list, int]:
    """Execute a ``JobQuery`` and return ``(rows, total_count)``.

    ``state_ids`` is the pre-resolved state filter (always non-empty); the
    caller owns "unknown state -> empty page" handling so that a bad filter
    never reaches SQL. The caller also owns the read snapshot — list_jobs
    chains the SELECT, COUNT, and downstream summary/parent queries on a
    single snapshot to keep the per-connection page cache hot.
    """
    if query.scope == controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN and not query.parent_job_id:
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            "query.parent_job_id is required for JOB_QUERY_SCOPE_CHILDREN",
        )
    return reads.list_jobs(tx, query, state_ids)


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


def _inject_resource_constraints(
    request: controller_pb2.Controller.LaunchJobRequest,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Merge auto-generated device constraints into a job submission request.

    Constraints derived from ResourceSpecProto.device (device-type, device-variant)
    are merged with any explicit user constraints on the request.  For canonical
    keys the user's explicit constraints replace auto-generated ones, so e.g.
    a user-provided multi-variant IN constraint overrides the single-variant
    EQ constraint from the resource spec.
    """
    auto = constraints_from_resources(request.resources)
    if not auto:
        return request

    user = [Constraint.from_proto(c) for c in request.constraints]
    merged = merge_constraints(auto, user)

    new_request = controller_pb2.Controller.LaunchJobRequest()
    new_request.CopyFrom(request)
    del new_request.constraints[:]
    for c in merged:
        new_request.constraints.append(c.to_proto())
    return new_request


def _get_autoscaler_pending_hints(dependencies: JobDependencies) -> dict[str, PendingHint]:
    """Build the backend autoscaler's cached pending hints keyed by job id."""
    return dependencies.runtime.backend_observation.pending_hints


def _job_state(db: ControllerDB, job_id: JobName) -> int | None:
    with db.read_snapshot() as tx:
        row = tx.execute(select(jobs_table.c.state).where(jobs_table.c.job_id == job_id)).first()
    return int(row.state) if row else None


def _profile_is_elevated(profile: int) -> bool:
    return resolve_container_profile(profile) in (
        job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS,
        job_pb2.CONTAINER_PROFILE_PRIVILEGED,
    )


def _authorize_job_owner(dependencies: JobDependencies, job_id: JobName) -> None:
    """Raise PERMISSION_DENIED if the authenticated user doesn't own this job.

    Skipped when no auth provider is configured (null-auth mode).
    """
    authorize_owner_if_configured(dependencies.auth, job_id.user)


def _authorize_federation_handoff(
    dependencies: JobDependencies,
    identity: VerifiedIdentity | None,
    request: controller_pb2.Controller.LaunchJobRequest,
    job_id: JobName,
) -> None:
    """Authorize an inbound federation handoff (the ``federation`` field is set).

    Only reached with an auth provider configured. A verified federation peer (its
    token yields the ``federation-peer`` role) is admitted only under its own signed
    requester id and only for a submitter this cluster's allowlist permits; a local
    admin (loopback/trusted) is also honored. Any other caller setting ``federation``
    is forging a handoff to run a job as another user. ``local_admin`` is never a
    valid federation submitter, regardless of the allowlist.
    """
    if request.federation.submitting_user == LOCAL_ADMIN_SUBMITTER:
        raise ConnectError(
            Code.PERMISSION_DENIED,
            "A federated job cannot be submitted as local_admin — a CIDR/loopback identity is "
            "never a valid federation submitter. The submitting user must be an authenticated "
            "principal (an IAP or JWT user).",
        )
    if identity is not None and identity.role == FEDERATION_PEER_ROLE:
        if request.federation.requester_id != identity.user_id:
            raise ConnectError(
                Code.PERMISSION_DENIED,
                f"Federation requester {request.federation.requester_id!r} does not match "
                f"the authenticated peer {identity.user_id!r}",
            )
        if not user_admitted(dependencies.auth.allowed_submitters, request.federation.submitting_user):
            raise ConnectError(
                Code.PERMISSION_DENIED,
                f"Submitter {request.federation.submitting_user!r} is not admitted for federation to this cluster",
            )
    elif identity is None or identity.role != "admin":
        raise ConnectError(Code.PERMISSION_DENIED, "The federation handoff field may only be set by a trusted peer.")
    if not job_id.is_root:
        raise ConnectError(Code.INVALID_ARGUMENT, "A federation handoff must be a root job.")


def _authorize_job_actor(dependencies: JobDependencies, job_id: JobName) -> None:
    """Authorize the caller to act on ``job_id`` (e.g. route a cancel).

    The job owner or an admin passes, as with :meth:`_authorize_job_owner`. A
    federation peer additionally passes for a job it federated here — its verified
    requester matches the job's received handle — so the parent can route a cancel
    for a handed-off job whose local owner it is not.
    """
    if not dependencies.auth.provider:
        return
    identity = get_verified_identity()
    if identity is not None and identity.role == FEDERATION_PEER_ROLE:
        with dependencies.db.read_snapshot() as snap:
            handoff = reads.received_handoff(snap, job_id)
            if handoff is not None and handoff.requester_id == identity.user_id:
                return
        raise ConnectError(Code.PERMISSION_DENIED, f"Peer {identity.user_id!r} did not federate job {job_id}")
    authorize_resource_owner(job_id.user)


def _wait_until_job_drained(dependencies: JobDependencies, job_id: JobName, wait: Duration) -> bool:
    """Wait up to ``wait`` for ``job_id`` to have no unfinished worker-bound
    attempts. Returns ``True`` if drained, ``False`` if the wait elapsed.

    Polls the snapshot DB; the reconcile-observation path landing terminal
    updates is what flips the predicate. Caller decides whether to reap the
    predecessor when the wait elapses — a stuck worker must not block
    the new submission forever.
    """

    def drained() -> bool:
        with dependencies.db.read_snapshot() as tx:
            return not reads.has_unfinished_worker_attempts(tx, job_id)

    return ExponentialBackoff(initial=1.0, maximum=10.0, factor=2).wait_until(drained, timeout=wait)


def _replace_finished_job(
    cur: Tx,
    job_id: JobName,
    *,
    record_tombstone: bool = True,
) -> bool:
    """Attempt to replace a terminal job; signal whether a drain is needed.

    CASCADE-deleting a job's tasks while its attempts are still worker-
    bound destroys the rows the reconcile-observation path needs to find when it
    stamps ``finished_at_ms``. Returns ``True`` when the caller must wait
    for worker-bound attempts to finalize before retrying (the job rows
    are left in place), ``False`` when removal completed in this
    transaction. Every replacement path in ``launch_job`` funnels through
    here so the contract is uniform.
    """
    if reads.has_unfinished_worker_attempts(cur, job_id):
        return True
    ops.job.remove_finished(cur, job_id, record_tombstone=record_tombstone)
    return False


def _admit_federated_resubmit(
    cur: Tx,
    job_id: JobName,
    request: controller_pb2.Controller.LaunchJobRequest,
) -> controller_pb2.Controller.LaunchJobResponse | None:
    """Federation-aware admission for a handoff whose job id already exists.

    A delivery repeating the stored nonce from the same requester is an
    idempotent replay: return the existing job and re-report its state, so a
    parent whose sync cursor already consumed this job's deltas still
    converges. The same requester with a new nonce is a fresh incarnation
    (the parent replaced its finished job and resubmitted): return ``None``
    and the caller applies the generic ``existing_job_policy``. Anything
    else — a local job, a different requester's job, or a SENT handle — is a
    collision: raise ``ALREADY_EXISTS``.
    """
    handoff = reads.received_handoff(cur, job_id)
    if handoff is None or handoff.requester_id != request.federation.requester_id:
        raise ConnectError(
            Code.ALREADY_EXISTS,
            f"Job {job_id} already exists and was not handed off by {request.federation.requester_id!r}",
        )
    if handoff.handoff_nonce != request.federation.handoff_nonce:
        return None
    writes.record_federation_change(cur, job_id)
    return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())


def _queue_federated_job(
    dependencies: JobDependencies,
    job_id: JobName,
    request: controller_pb2.Controller.LaunchJobRequest,
    pinned_peer_id: str,
    submitting_user: str,
) -> controller_pb2.Controller.LaunchJobResponse:
    """Admit a root job to the federation queue and return the parent's job id.

    The caller has established that ``job_id`` is a root: the peer runs it under the
    same, cluster-invariant job id, so queueing a non-root job would clash with the
    job's own tree on the peer. The manager persists the handle in ``QUEUED_HANDOFF``
    (no peer chosen unless ``pinned_peer_id`` is set); the control tick later assigns
    it to a peer with room and delivers it. Peer allowlist rejection surfaces later as
    a failed job rather than synchronously here.
    """
    assert job_id.is_root, f"only whole root jobs may be federated to a peer; got {job_id}"
    dependencies.runtime.federation.queue_federated(
        local_job_id=job_id,
        request=request,
        pinned_peer_id=pinned_peer_id,
        owner_principal=job_id.user,
        submitting_user=submitting_user,
    )
    return controller_pb2.Controller.LaunchJobResponse(job_id=job_id.to_wire())


@dataclass(frozen=True, slots=True)
class LaunchIdentity:
    job_id: JobName
    received_handoff: bool
    submitting_user: str
    budget_user: str


def _validate_launch_request(request: controller_pb2.Controller.LaunchJobRequest) -> JobName:
    if not request.name:
        raise ConnectError(Code.INVALID_ARGUMENT, "Job name is required")
    if request.HasField("coscheduling") and not request.coscheduling.group_by:
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            "coscheduling requires a non-empty group_by (the topology level to gang on)",
        )
    return JobName.from_wire(request.name)


def _launch_identity(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
    context: Any,
    requested_job_id: JobName,
) -> LaunchIdentity:
    """Authorize the caller and resolve the owner and budget identities."""
    identity = get_verified_identity()
    received_handoff = request.HasField("federation")
    if received_handoff and dependencies.auth.provider:
        _authorize_federation_handoff(dependencies, identity, request, requested_job_id)

    if requested_job_id.is_root and context is not None and not received_handoff:
        _check_client_freshness(request.client_revision_date, client_revision_date(), date.today())

    job_id = requested_job_id
    if (
        dependencies.auth.provider
        and identity is not None
        and job_id.is_root
        and identity.role != "admin"
        and not received_handoff
    ):
        job_id = JobName.root(identity.user_id, job_id.name)
    if dependencies.auth.provider and identity is not None and not job_id.is_root:
        _authorize_job_owner(dependencies, job_id)

    submitting_user = submitting_user_for_root(identity, request)
    if job_id.parent is not None:
        with dependencies.db.read_snapshot() as snapshot:
            root_submitting_user = reads.get_job_submitting_user(snapshot, job_id.root_job)
        if root_submitting_user is not None:
            submitting_user = root_submitting_user
    return LaunchIdentity(
        job_id=job_id,
        received_handoff=received_handoff,
        submitting_user=submitting_user,
        budget_user=budget_user_id(job_id, submitting_user),
    )


def _resolve_launch_priority(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
    launch: LaunchIdentity,
) -> int:
    """Resolve INHERIT and enforce the caller's priority ceiling."""
    if request.priority_band not in _SUBMITTABLE_PRIORITY_BANDS:
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            f"Unknown priority_band {int(request.priority_band)}; "
            f"expected one of {sorted(_SUBMITTABLE_PRIORITY_BANDS)}",
        )
    inherited_band: int | None = None
    if request.priority_band == job_pb2.PRIORITY_BAND_INHERIT and launch.job_id.parent is not None:
        with dependencies.db.read_snapshot() as snapshot:
            inherited_band = reads.get_priority_bands(snapshot, [launch.job_id.parent])[launch.job_id.parent]
    band = ops.job.resolve_priority_band(int(request.priority_band), inherited_band)
    request.priority_band = band
    if launch.received_handoff:
        return band
    if band in ADMIN_PRIORITY_BAND_VALUES and dependencies.auth.provider:
        authorize(AuthzAction.MANAGE_BUDGETS)
        return band
    with dependencies.db.read_snapshot() as snapshot:
        user_budget = reads.get_user_budget(snapshot, launch.budget_user)
    max_band = user_budget.max_band if user_budget is not None else dependencies.user_budget_defaults.max_band
    if priority_band_rank(band) < priority_band_rank(max_band):
        raise ConnectError(
            Code.PERMISSION_DENIED,
            f"Budget identity {launch.budget_user} cannot submit {priority_band_name(band)} jobs "
            f"(max band: {priority_band_name(max_band)}). "
            f"Resubmit with `--priority {priority_band_name(max_band).lower()}` "
            f"(e.g. `--priority batch`) to launch opportunistically, or ping @Helw150 "
            f"to request a higher band for {launch.budget_user}.",
        )
    return band


def _validate_launch_profile(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
    launch: LaunchIdentity,
) -> None:
    """Authorize elevated profiles and reject unsupported runtime combinations."""
    if _profile_is_elevated(request.container_profile):
        if dependencies.auth.provider and not launch.received_handoff:
            authorize(AuthzAction.SET_CONTAINER_PROFILE)
        logger.info(
            "Job %s using elevated container profile %s",
            launch.job_id.to_wire(),
            job_pb2.ContainerProfile.Name(request.container_profile),
        )
    if (
        resolve_container_profile(request.container_profile) == job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS
        and BackendCapability.WORKER_FLEET not in dependencies.runtime.backend.descriptor.capabilities
    ):
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            "Container profile docker_access requires the docker worker backend (it mounts the "
            "host docker socket); this cluster's backend does not support it. Use a privileged "
            "profile with an in-pod runtime, or submit to a docker-worker cluster.",
        )
    if resolve_container_profile(
        request.container_profile
    ) == job_pb2.CONTAINER_PROFILE_GVISOR and request.resources.device.WhichOneof("device") in ("gpu", "tpu"):
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            "Container profile gvisor is CPU-only: the runsc runtime cannot pass a GPU or TPU "
            "through to the sandboxed guest. Use the default or privileged profile for "
            "accelerator tasks.",
        )


def _validate_launch_capacity(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
    launch: LaunchIdentity,
) -> None:
    """Enforce the active-task cap and require a live parent for child jobs."""
    incoming_tasks = int(request.replicas)
    if incoming_tasks > 0:
        with dependencies.db.read_snapshot() as snapshot:
            active_tasks = reads.count_active_tasks_for_budget_user(snapshot, launch.budget_user)
        if active_tasks + incoming_tasks > MAX_ACTIVE_TASKS_PER_USER:
            raise ConnectError(
                Code.RESOURCE_EXHAUSTED,
                f"Budget identity {launch.budget_user} has {active_tasks} active task(s); submitting "
                f"{incoming_tasks} more would exceed the per-user cap of "
                f"{MAX_ACTIVE_TASKS_PER_USER}. Wait for running tasks to finish, or "
                f"structure the work as a launcher job that admits tasks gradually.",
            )

    if launch.job_id.parent is None:
        return
    parent_state = _job_state(dependencies.db, launch.job_id.parent)
    if parent_state is None:
        raise ConnectError(
            Code.FAILED_PRECONDITION,
            f"Cannot submit job: parent job {launch.job_id.parent} is absent from the database",
        )
    if parent_state in TERMINAL_JOB_STATES:
        raise ConnectError(
            Code.FAILED_PRECONDITION,
            f"Cannot submit job: parent job {launch.job_id.parent} has terminated "
            f"(state={job_pb2.JobState.Name(parent_state)})",
        )


def _prepare_job_slot(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
    launch: LaunchIdentity,
) -> controller_pb2.Controller.LaunchJobResponse | None:
    """Apply the existing-job policy and drain a predecessor when necessary."""
    needs_drain = False
    record_tombstone = not launch.received_handoff
    with dependencies.db.transaction() as cur:
        existing_state = reads.get_job_state(cur, launch.job_id)
        if existing_state is None:
            return None
        if launch.received_handoff:
            replay = _admit_federated_resubmit(cur, launch.job_id, request)
            if replay is not None:
                return replay
        policy = request.existing_job_policy
        if policy == job_pb2.EXISTING_JOB_POLICY_ERROR:
            raise ConnectError(
                Code.ALREADY_EXISTS,
                f"Job {launch.job_id} already exists (state={job_pb2.JobState.Name(existing_state)})",
            )
        if policy == job_pb2.EXISTING_JOB_POLICY_KEEP:
            if not is_job_finished(existing_state):
                return controller_pb2.Controller.LaunchJobResponse(job_id=launch.job_id.to_wire())
            needs_drain = _replace_finished_job(cur, launch.job_id, record_tombstone=record_tombstone)
        elif policy == job_pb2.EXISTING_JOB_POLICY_RECREATE:
            if not is_job_finished(existing_state):
                ops.job.cancel(cur, job_id=launch.job_id, reason="Replaced by new submission")
                needs_drain = True
            else:
                needs_drain = _replace_finished_job(cur, launch.job_id, record_tombstone=record_tombstone)
        elif is_job_finished(existing_state):
            logger.info(
                "Replacing finished job %s (state=%s) with new submission",
                launch.job_id,
                job_pb2.JobState.Name(existing_state),
            )
            needs_drain = _replace_finished_job(cur, launch.job_id, record_tombstone=record_tombstone)
        else:
            raise ConnectError(Code.ALREADY_EXISTS, f"Job {launch.job_id} already exists and is still running")

    if not needs_drain:
        return None
    dependencies.runtime.wake()
    if not _wait_until_job_drained(dependencies, launch.job_id, _JOB_REPLACEMENT_DRAIN_WAIT):
        logger.warning(
            "Job %s did not drain within %ss; force-reaping predecessor and proceeding",
            launch.job_id,
            _JOB_REPLACEMENT_DRAIN_WAIT.to_seconds(),
        )
    with dependencies.db.transaction() as cur:
        ops.job.remove_finished(cur, launch.job_id, record_tombstone=record_tombstone)
    return None


def _store_launch_payload(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Store large bundle and workdir payloads and return the normalized request."""
    if request.bundle_blob:
        bundle_size = len(request.bundle_blob)
        if bundle_size > MAX_BUNDLE_SIZE_BYTES:
            bundle_size_mb = bundle_size / (1024 * 1024)
            max_size_mb = MAX_BUNDLE_SIZE_BYTES / (1024 * 1024)
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                f"Bundle size {bundle_size_mb:.1f}MB exceeds maximum {max_size_mb:.0f}MB",
            )
        normalized = controller_pb2.Controller.LaunchJobRequest()
        normalized.CopyFrom(request)
        normalized.ClearField("bundle_blob")
        normalized.bundle_id = dependencies.bundles.write(request.bundle_blob)
        request = normalized

    large_files = {
        name: data
        for name, data in request.entrypoint.workdir_files.items()
        if len(data) > WORKDIR_FILE_OFFLOAD_THRESHOLD
    }
    if not large_files:
        return request
    normalized = controller_pb2.Controller.LaunchJobRequest()
    normalized.CopyFrom(request)
    for name, data in large_files.items():
        blob_id = dependencies.bundles.write(data)
        del normalized.entrypoint.workdir_files[name]
        normalized.entrypoint.workdir_file_refs[name] = blob_id
        logger.info("Externalized workdir file %s (%d bytes) as blob %s", name, len(data), blob_id[:12])
    return normalized


def _route_launch(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
    launch: LaunchIdentity,
) -> controller_pb2.Controller.LaunchJobResponse | None:
    """Validate placement and either queue, reject, or select local execution."""
    constraints = [Constraint.from_proto(constraint) for constraint in request.constraints]
    tpu_error = validate_tpu_request(request.resources, constraints)
    if tpu_error:
        raise ConnectError(Code.INVALID_ARGUMENT, tpu_error)

    cluster_pin = cluster_directive(constraints)
    if cluster_pin is not None and not dependencies.runtime.federation.has_peer(cluster_pin):
        raise ConnectError(
            Code.INVALID_ARGUMENT,
            f"Job {launch.job_id} pins cluster {cluster_pin!r}, which is not a configured federation peer.",
        )
    error = dependencies.runtime.backend.job_feasibility(
        JobFeasibilityRequest(
            constraints=constraints,
            replicas=request.replicas if request.HasField("coscheduling") else None,
            resources=request.resources,
        )
    )
    if launch.received_handoff:
        plan = SubmitPlan(SubmitDisposition.LOCAL)
    else:
        plan = dependencies.runtime.federation.classify_submit(
            RoutingRequest(
                constraints=constraints,
                local_feasible=error is None,
                cluster_pin=cluster_pin or "",
            )
        )

    if plan.disposition == SubmitDisposition.QUEUE:
        if not launch.job_id.is_root:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                _child_federation_refusal(launch.job_id, plan.pinned_peer_id),
            )
        if dependencies.auth.provider and launch.submitting_user == LOCAL_ADMIN_SUBMITTER:
            raise ConnectError(Code.PERMISSION_DENIED, _LOCAL_ADMIN_FEDERATION_DENIED)
        return _queue_federated_job(
            dependencies,
            launch.job_id,
            request,
            plan.pinned_peer_id,
            launch.submitting_user,
        )
    if plan.disposition == SubmitDisposition.REJECT:
        reason = error or "no local backend or peer can host it"
        raise ConnectError(
            Code.FAILED_PRECONDITION,
            f"Job {launch.job_id} is unschedulable: {reason} (constraints: {constraints})",
        )
    return None


def _insert_local_job(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
    launch: LaunchIdentity,
    priority_band: int,
) -> controller_pb2.Controller.LaunchJobResponse:
    """Insert the admitted local job, fencing a concurrent submission."""
    with dependencies.db.transaction() as cur:
        if reads.get_job_state(cur, launch.job_id) is not None:
            if launch.received_handoff:
                replay = _admit_federated_resubmit(cur, launch.job_id, request)
                if replay is not None:
                    return replay
                raise ConnectError(
                    Code.ALREADY_EXISTS,
                    f"Job {launch.job_id} already exists (concurrent submission)",
                )
            if request.existing_job_policy == job_pb2.EXISTING_JOB_POLICY_KEEP:
                return controller_pb2.Controller.LaunchJobResponse(job_id=launch.job_id.to_wire())
            raise ConnectError(
                Code.ALREADY_EXISTS,
                f"Job {launch.job_id} already exists (concurrent submission)",
            )
        ops.job.submit(
            cur,
            job_id=launch.job_id,
            request=request,
            ts=Timestamp.now(),
            priority_band=priority_band,
            submitting_user=launch.submitting_user,
        )
    dependencies.runtime.wake()

    with dependencies.db.read_snapshot() as snapshot:
        num_tasks = snapshot.execute(select(func.count()).where(tasks_table.c.job_id == launch.job_id)).scalar() or 0
    logger.info("Job %s submitted with %d task(s)", launch.job_id, num_tasks)
    return controller_pb2.Controller.LaunchJobResponse(job_id=launch.job_id.to_wire())


def launch_job(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.LaunchJobRequest,
    ctx: Any,
) -> controller_pb2.Controller.LaunchJobResponse:
    """Submit a new job to the controller.

    The job is expanded into tasks based on the replicas field
    (defaulting to 1). Each task has ID "/job/.../index".
    """
    requested_job_id = _validate_launch_request(request)
    launch = _launch_identity(dependencies, request, ctx, requested_job_id)
    band = _resolve_launch_priority(dependencies, request, launch)
    _validate_launch_profile(dependencies, request, launch)
    _validate_launch_capacity(dependencies, request, launch)

    existing = _prepare_job_slot(dependencies, request, launch)
    if existing is not None:
        return existing

    request = _store_launch_payload(dependencies, request)
    request = _inject_resource_constraints(request)
    routed = _route_launch(dependencies, request, launch)
    if routed is not None:
        return routed
    return _insert_local_job(dependencies, request, launch, band)


def get_job_status(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.GetJobStatusRequest,
    ctx: Any,
) -> controller_pb2.Controller.GetJobStatusResponse:
    """Get job-level status with aggregated task counts.

    Per-task detail (attempts, worker addresses) is NOT included — callers
    that need it should use ListTasks instead.  This keeps GetJobStatus
    cheap: one job row read + one GROUP BY query vs loading every task,
    attempt, and worker address.
    """
    with dependencies.db.read_snapshot() as q:
        job = reads.get_job_detail(q, JobName.from_wire(request.job_id))
        if not job:
            raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")
        # Aggregate task counts via a single GROUP BY query; failure/preemption
        # totals come from the attempt-derived cache.
        summaries = reads.task_summaries_for_jobs(
            q, {job.job_id}, attempt_counts=q.caches[AttemptCountsProjection].get_jobs(q, [job.job_id])
        )
        has_children = bool(reads.parent_ids_with_children(q, [job.job_id]))
        # A federated job's subtree lives on the peer; load the handle to surface
        # its handoff posture in the pending reason.
        handle = reads.federated_handle(q, job.job_id) if is_federated(job.cluster) else None
    summary = summaries.get(job.job_id)

    # Get scheduling diagnostics for pending jobs from cache
    # (populated each scheduling cycle by the controller). The autoscaler
    # hint dict is cached once per evaluate cycle, so the lookup here
    # is a single dict get — we only attach this job's hint, never the
    # full routing decision.
    handoff_state = handle.handoff_state if handle else None
    current_peer_status = peer_status(job.cluster, handoff_state, summary is not None)
    pending_reason = ""
    if job.state == job_pb2.JOB_STATE_PENDING and is_federated(job.cluster):
        pending_reason = _federated_pending_reason(job.cluster, handoff_state, current_peer_status)
    elif job.state == job_pb2.JOB_STATE_PENDING:
        sched_reason = dependencies.runtime.get_job_scheduling_diagnostics(job.job_id.to_wire())
        pending_reason = sched_reason or "Pending scheduler feedback"
        hint = _get_autoscaler_pending_hints(dependencies).get(job.job_id.to_wire())
        if hint is not None:
            scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
            pending_reason = f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

    resources = resource_spec_from_job_row(job)

    proto_job_status = job_pb2.JobStatus(
        job_id=job.job_id.to_wire(),
        state=job.state,
        error=job.error or "",
        exit_code=job.exit_code or 0,
        name=job.name,
        pending_reason=pending_reason,
        resources=resources,
        has_children=has_children,
        parent_job_id=job.parent_job_id.to_wire() if job.parent_job_id else "",
        backend_id=job.backend_id or "",
        cluster=job.cluster,
        peer_status=current_peer_status,
        **job_status_counts(summary, job.job_id, pre_sync_task_count=job.num_tasks if is_federated(job.cluster) else 0),
    )
    if job.started_at_ms:
        proto_job_status.started_at.CopyFrom(timestamp_to_proto(job.started_at_ms))
    if job.finished_at_ms:
        proto_job_status.finished_at.CopyFrom(timestamp_to_proto(job.finished_at_ms))
    if job.submitted_at_ms:
        proto_job_status.submitted_at.CopyFrom(timestamp_to_proto(job.submitted_at_ms))

    # Status describes the job's shape; the workdir file bytes are payload no
    # client of this RPC reads, so they stay out of the response.
    reconstructed_request = reconstruct_launch_job_request(job, workdir_files={})
    return controller_pb2.Controller.GetJobStatusResponse(
        job=proto_job_status,
        request=redact_request_env_vars(reconstructed_request),
    )


def get_job_state(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.GetJobStateRequest,
    ctx: Any,
) -> controller_pb2.Controller.GetJobStateResponse:
    """Lightweight batch job state query.

    Returns only the state enum for each requested job, avoiding the cost
    of loading tasks, attempts, and worker addresses.
    """
    wire_ids = list(request.job_ids)
    if not wire_ids:
        return controller_pb2.Controller.GetJobStateResponse()

    with dependencies.db.read_snapshot() as tx:
        rows = tx.execute(
            select(jobs_table.c.job_id, jobs_table.c.state).where(
                jobs_table.c.job_id.in_(bindparam("job_ids", expanding=True))
            ),
            {"job_ids": wire_ids},
        ).all()

    states = {row.job_id.to_wire(): int(row.state) for row in rows}
    return controller_pb2.Controller.GetJobStateResponse(states=states)


def terminate_job(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.TerminateJobRequest,
    ctx: Any,
) -> job_pb2.Empty:
    """Terminate a running job and all its children.

    Cascade termination is performed depth-first: all children are
    terminated before the parent. All tasks within each job are killed.
    """
    job_id = JobName.from_wire(request.job_id)
    state = _job_state(dependencies.db, job_id)
    if state is None:
        raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

    # Owner, admin, or the peer that federated this job here (a routed cancel).
    _authorize_job_actor(dependencies, job_id)

    # A federated handle owns no local tasks — its subtree lives on the peer.
    # Route a versioned, idempotent cancel there; the next sync mirrors the
    # peer's terminal state (and eventually its tombstone) back.
    with dependencies.db.read_snapshot() as snap:
        has_federated_handle = reads.federated_handle(snap, job_id) is not None
    if has_federated_handle:
        dependencies.runtime.federation.cancel_federated(job_id)
        return job_pb2.Empty()

    # cancel_job uses a recursive CTE to walk the full subtree in a single
    # transaction, so there is no need to recurse manually.
    with dependencies.db.transaction() as cur:
        ops.job.cancel(
            cur,
            job_id=job_id,
            reason="Terminated by user",
        )
        # Re-report the job's state to its requester (a no-op unless this
        # root was received via handoff). A routed cancel of an already-
        # terminal job changes nothing, and this re-report is what converges
        # the parent's stale mirror and stops its cancel re-drive.
        writes.record_federation_change(cur, job_id)
    # The next polling tick reconciles each affected worker; the
    # cancellation appears in the desired-set diff so the worker stops
    # the attempt within one tick rather than waiting on the next backoff.
    dependencies.runtime.wake()
    return job_pb2.Empty()


def complete_job(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.CompleteJobRequest,
    _ctx: RequestContext | None,
) -> job_pb2.Empty:
    """Complete a running job successfully and stop its unfinished tasks."""
    job_id = JobName.from_wire(request.job_id)
    state = _job_state(dependencies.db, job_id)
    if state is None:
        raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")

    _authorize_job_actor(dependencies, job_id)

    with dependencies.db.read_snapshot() as snap:
        if reads.federated_handle(snap, job_id) is not None:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                "A federated job must be completed on the cluster running its tasks",
            )

    with dependencies.db.transaction() as cur:
        ops.job.complete(cur, job_id=job_id)
    dependencies.runtime.wake()
    return job_pb2.Empty()


def _job_to_proto(
    dependencies: JobDependencies,
    j: Any,
    task_summary: TaskJobSummary | None,
    autoscaler_pending_hints: dict[str, PendingHint],
    *,
    has_children: bool = False,
    handoff_state: int | None = None,
) -> job_pb2.JobStatus:
    """Convert a job row and its task summary into a JobStatus proto."""
    current_peer_status = peer_status(j.cluster, handoff_state, task_summary is not None)
    pending_reason = j.error or ""
    if j.state == job_pb2.JOB_STATE_PENDING and is_federated(j.cluster):
        pending_reason = _federated_pending_reason(j.cluster, handoff_state, current_peer_status)
    elif j.state == job_pb2.JOB_STATE_PENDING:
        sched_reason = dependencies.runtime.get_job_scheduling_diagnostics(j.job_id.to_wire())
        pending_reason = sched_reason or "Pending scheduler feedback"
        hint = autoscaler_pending_hints.get(j.job_id.to_wire())
        if hint is not None:
            scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
            pending_reason = f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

    proto_job = job_pb2.JobStatus(
        job_id=j.job_id.to_wire(),
        state=j.state,
        error=j.error or "",
        exit_code=j.exit_code or 0,
        name=j.name,
        pending_reason=pending_reason,
        has_children=has_children,
        backend_id=j.backend_id or "",
        cluster=j.cluster,
        peer_status=current_peer_status,
        **job_status_counts(task_summary, j.job_id, pre_sync_task_count=j.num_tasks if is_federated(j.cluster) else 0),
    )
    if j.started_at_ms:
        proto_job.started_at.CopyFrom(timestamp_to_proto(j.started_at_ms))
    if j.finished_at_ms:
        proto_job.finished_at.CopyFrom(timestamp_to_proto(j.finished_at_ms))
    if j.submitted_at_ms:
        proto_job.submitted_at.CopyFrom(timestamp_to_proto(j.submitted_at_ms))
    return proto_job


def _jobs_to_protos(
    dependencies: JobDependencies,
    jobs: list,
    task_summaries: dict[JobName, TaskJobSummary],
    autoscaler_pending_hints: dict[str, PendingHint],
    has_children: set[JobName] | None = None,
    handoff_states: dict[JobName, int] | None = None,
) -> list[job_pb2.JobStatus]:
    child_parent_ids = has_children or set()
    handoffs = handoff_states or {}
    return [
        _job_to_proto(
            dependencies,
            j,
            task_summaries.get(j.job_id),
            autoscaler_pending_hints,
            has_children=j.job_id in child_parent_ids,
            handoff_state=handoffs.get(j.job_id),
        )
        for j in jobs
    ]


def list_jobs(
    dependencies: JobDependencies,
    request: controller_pb2.Controller.ListJobsRequest,
    ctx: Any,
) -> controller_pb2.Controller.ListJobsResponse:
    """List jobs with filtering, sorting, and pagination.

    Served directly from indexed SQL via ``_query_jobs``. Per-page task
    summaries and parent->child flags are looked up against the same read
    snapshot so the whole RPC observes a single transactionally-consistent
    view.
    """
    query = _query_from_list_jobs_request(request)

    state_ids = _resolve_state_filter(query.state_filter)
    if state_ids is None:
        return controller_pb2.Controller.ListJobsResponse(jobs=[], total_count=0, has_more=False)

    with dependencies.db.read_snapshot() as q:
        page, total_count = _query_jobs(q, query, state_ids)
        page_ids = [j.job_id for j in page]
        summaries = (
            reads.task_summaries_for_jobs(
                q, set(page_ids), attempt_counts=q.caches[AttemptCountsProjection].get_jobs(q, page_ids)
            )
            if page_ids
            else {}
        )
        children = reads.parent_ids_with_children(q, page_ids) if page_ids else set()
        # Batch-load the SENT handoff state for the federated jobs on this page
        # so each JobStatus carries its handoff posture without a per-job read.
        federated_ids = [j.job_id for j in page if is_federated(j.cluster)]
        handoffs = reads.handoff_states(q, federated_ids)

    has_pending = any(j.state == job_pb2.JOB_STATE_PENDING for j in page)
    autoscaler_pending_hints = _get_autoscaler_pending_hints(dependencies) if has_pending else {}
    all_jobs = _jobs_to_protos(
        dependencies, page, summaries, autoscaler_pending_hints, has_children=children, handoff_states=handoffs
    )
    limit = query.limit
    offset = query.offset
    has_more = limit > 0 and offset + limit < total_count
    return controller_pb2.Controller.ListJobsResponse(
        jobs=all_jobs,
        total_count=total_count,
        has_more=has_more,
    )
