# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource-native Job admission."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import Enum, auto
from typing import Protocol

from rigging.server_auth import ANONYMOUS_ADMIN, VerifiedIdentity, get_verified_identity
from rigging.timing import Duration, ExponentialBackoff, Timestamp

from iris.backends.protocol import BackendCapability, TaskBackend
from iris.cluster.authorization import FEDERATION_PEER_ROLE, AuthzAction, authorize, authorize_resource_owner
from iris.cluster.bundle import MAX_BUNDLE_SIZE_BYTES, BundleStore
from iris.cluster.config import user_admitted
from iris.cluster.constraints import (
    backend_directive,
    cluster_directive,
    constraints_from_resources,
    merge_constraints,
    validate_tpu_request,
)
from iris.cluster.controller.action import (
    _action_payload_hash,
    _completed_action,
    _CompletedCancel,
    _duplicate_action,
    _RemoteActionContext,
    _require_idempotency_key,
)
from iris.cluster.controller.auth import (
    ControllerAuth,
)
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.pagination import (
    _decode_page_token,
    _encode_page_token,
    _escaped_prefix,
    _page_size,
    _query_fingerprint,
    _require_kind,
    _stored_cluster,
)
from iris.cluster.controller.persistence import action as action_persistence
from iris.cluster.controller.persistence import operations as ops
from iris.cluster.controller.persistence import reads, writes
from iris.cluster.controller.persistence.database import ControllerDB, Tx
from iris.cluster.controller.persistence.json_codec import (
    reconstruct_job_spec,
    resource_spec_from_scalars,
)
from iris.cluster.controller.persistence.operations import job as job_ops
from iris.cluster.controller.reconcile.policy import MAX_ACTIVE_TASKS_PER_USER
from iris.cluster.controller.resource_identity import (
    _execution_cluster,
    _job_uid,
)
from iris.cluster.controller.source_status import resource_source_statuses
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.protocol import CancelTarget, FederationDirection, HandoffState
from iris.cluster.federation.router import RoutingRequest, SubmitDisposition, SubmitPlan
from iris.cluster.types import (
    LOCAL_ADMIN_SUBMITTER,
    TERMINAL_JOB_STATES,
    UserBudgetDefaults,
    is_job_finished,
)
from iris.resources.action import ActionKind, ActionReceipt, ActionResult
from iris.resources.errors import (
    InvalidResourceRequest,
    ResourceConflict,
    ResourceExhausted,
    ResourceNotFound,
    ResourcePermissionDenied,
    ResourcePreconditionFailed,
    ResourceReplaced,
)
from iris.resources.execution import GpuDevice, TpuDevice
from iris.resources.identity import (
    JobIdentity,
    ResourceKey,
    ResourceKind,
)
from iris.resources.job import (
    ContainerProfile,
    ExistingJobPolicy,
    FederationPosture,
    JobDetail,
    JobInventoryPage,
    JobInventoryQuery,
    JobListScope,
    JobObservation,
    JobQuery,
    JobSpec,
    JobSummary,
    JobTaskAggregate,
    PriorityBand,
    TaskStateCount,
)
from iris.resources.names import JobName
from iris.resources.source import (
    Page,
)
from iris.resources.state import JobState, TaskState

logger = logging.getLogger(__name__)

WORKDIR_FILE_OFFLOAD_THRESHOLD = 10 * 1024
_MAX_JOB_PAGE = 500
_MAX_JOB_STATE_BATCH = 32_767
_JOB_REPLACEMENT_DRAIN_WAIT = Duration.from_seconds(120)
CLIENT_FRESHNESS_WINDOW = timedelta(days=14)
_FEATURE_INTRODUCTION_DATE = date(2026, 4, 22)
_SUBMITTABLE_PRIORITY_BANDS = frozenset(
    {
        PriorityBand.INHERIT,
        PriorityBand.PRODUCTION,
        PriorityBand.INTERACTIVE,
        PriorityBand.BATCH,
    }
)
_LOCAL_ADMIN_FEDERATION_DENIED = (
    "A local_admin (CIDR/loopback) identity cannot submit a federated job. "
    "Federating to a remote cluster requires an authenticated user."
)


@dataclass(frozen=True, slots=True)
class FederationSubmission:
    requester_id: str
    owner_principal: str
    submitting_user: str
    handoff_nonce: str


class _SubmissionPreparation(Enum):
    READY = auto()
    REUSE_EXISTING = auto()
    WAIT_FOR_DRAIN = auto()


class JobRuntime(Protocol):
    @property
    def backends(self) -> Mapping[str, TaskBackend]: ...

    @property
    def capabilities(self) -> frozenset[BackendCapability]: ...

    @property
    def federation(self) -> FederationManager: ...

    def wake(self) -> None: ...


class _JobAdmission:
    """Validate, route, persist, and replace submitted Jobs."""

    def __init__(
        self,
        *,
        db: ControllerDB,
        runtime: JobRuntime,
        bundle_store: BundleStore,
        auth: ControllerAuth,
        user_budget_defaults: UserBudgetDefaults,
    ) -> None:
        self._db = db
        self._runtime = runtime
        self._bundle_store = bundle_store
        self._auth = auth
        self._user_budget_defaults = user_budget_defaults

    def submit(
        self,
        spec: JobSpec,
        bundle_blob: bytes = b"",
        *,
        federation: FederationSubmission | None = None,
        enforce_client_freshness: bool = True,
    ) -> JobName:
        if not spec.name:
            raise InvalidResourceRequest("Job name is required")
        if spec.coscheduling is not None and not spec.coscheduling.group_by:
            raise InvalidResourceRequest("coscheduling requires a non-empty group_by")

        job_id = JobName.from_wire(spec.name)
        identity = get_verified_identity()
        if federation is not None and self._auth.provider:
            self._authorize_federation_handoff(identity, federation, job_id)
        if job_id.is_root and enforce_client_freshness and federation is None:
            _check_client_freshness(spec.client_revision_date, date.today())
        if (
            self._auth.provider
            and identity is not None
            and job_id.is_root
            and identity.role != "admin"
            and federation is None
        ):
            job_id = JobName.root(identity.user_id, job_id.name)
            spec = replace(spec, name=job_id.to_wire())
        if self._auth.provider and identity is not None and not job_id.is_root:
            authorize_resource_owner(job_id.user)

        submitting_user = _submitting_user(identity, federation)
        spec = self._validate_submission(spec, job_id, federation)
        record_tombstone = federation is None
        preparation = self._prepare_replacement(job_id, spec, federation, record_tombstone)
        if preparation is _SubmissionPreparation.REUSE_EXISTING:
            return job_id
        if preparation is _SubmissionPreparation.WAIT_FOR_DRAIN:
            self._runtime.wake()
            if not self._wait_until_drained(job_id):
                logger.warning("Job %s did not drain before replacement; force-reaping", job_id)
            with self._db.transaction() as tx:
                ops.job.remove_finished(tx, job_id, record_tombstone=record_tombstone)

        spec = self._store_payloads(spec, bundle_blob)
        spec = replace(
            spec,
            constraints=tuple(merge_constraints(constraints_from_resources(spec.resources), list(spec.constraints))),
        )
        tpu_error = validate_tpu_request(spec.resources, list(spec.constraints))
        if tpu_error:
            raise InvalidResourceRequest(tpu_error)

        plan, rejection_reason = self._submission_plan(job_id, spec, federation)
        if plan.disposition is SubmitDisposition.QUEUE:
            if not job_id.is_root:
                raise InvalidResourceRequest(_child_federation_refusal(job_id, plan.pinned_peer_id))
            if self._auth.provider and submitting_user == LOCAL_ADMIN_SUBMITTER:
                raise ResourcePermissionDenied(_LOCAL_ADMIN_FEDERATION_DENIED)
            self._runtime.federation.queue_federated(
                local_job_id=job_id,
                spec=spec,
                pinned_peer_id=plan.pinned_peer_id,
                owner_principal=job_id.user,
                submitting_user=submitting_user,
            )
            return job_id
        if plan.disposition is SubmitDisposition.REJECT:
            raise ResourcePreconditionFailed(f"Job {job_id} is unschedulable: {rejection_reason}")

        with self._db.transaction() as tx:
            if reads.get_job_state(tx, job_id) is not None:
                if federation is not None and self._federated_replay(tx, job_id, federation):
                    return job_id
                if spec.existing_job_policy is ExistingJobPolicy.KEEP:
                    return job_id
                raise ResourceConflict(f"Job {job_id} already exists (concurrent submission)")
            ops.job.submit(
                tx,
                job_id=job_id,
                spec=spec,
                ts=Timestamp.now(),
                priority_band=int(spec.priority_band),
                submitting_user=submitting_user,
                received_handoff=(
                    ops.job.ReceivedHandoff(
                        requester_id=federation.requester_id,
                        owner_principal=federation.owner_principal,
                        handoff_nonce=federation.handoff_nonce,
                    )
                    if federation is not None
                    else None
                ),
            )
        self._runtime.wake()
        with self._db.read_snapshot() as tx:
            num_tasks = reads.task_count(tx, job_id)
        logger.info("Job %s submitted with %d task(s)", job_id, num_tasks)
        return job_id

    def _validate_submission(
        self,
        spec: JobSpec,
        job_id: JobName,
        federation: FederationSubmission | None,
    ) -> JobSpec:
        if spec.priority_band not in _SUBMITTABLE_PRIORITY_BANDS:
            raise InvalidResourceRequest(f"Unknown priority_band {int(spec.priority_band)}")
        inherited_band = None
        if spec.priority_band is PriorityBand.INHERIT and job_id.parent is not None:
            with self._db.read_snapshot() as tx:
                inherited_band = reads.get_priority_bands(tx, [job_id.parent])[job_id.parent]
        band = ops.job.resolve_priority_band(int(spec.priority_band), inherited_band)
        resolved_band = PriorityBand(int(band))
        spec = replace(spec, priority_band=resolved_band)
        if federation is None:
            if resolved_band is PriorityBand.PRODUCTION and self._auth.provider:
                authorize(AuthzAction.MANAGE_BUDGETS)
            else:
                with self._db.read_snapshot() as tx:
                    user_budget = reads.get_user_budget(tx, job_id.user)
                max_band = user_budget.max_band if user_budget is not None else self._user_budget_defaults.max_band
                if band < max_band:
                    raise ResourcePermissionDenied(
                        f"User {job_id.user} cannot submit {resolved_band.name.lower()} jobs "
                        f"(max band: {PriorityBand(int(max_band)).name.lower()})",
                    )

        profile = (
            ContainerProfile.DEFAULT
            if spec.container_profile is ContainerProfile.UNSPECIFIED
            else spec.container_profile
        )
        if profile in (
            ContainerProfile.DOCKER_ACCESS,
            ContainerProfile.PRIVILEGED,
        ):
            if self._auth.provider and federation is None:
                authorize(AuthzAction.SET_CONTAINER_PROFILE)
        if (
            profile is ContainerProfile.DOCKER_ACCESS
            and BackendCapability.WORKER_DAEMON not in self._runtime.capabilities
        ):
            raise InvalidResourceRequest("docker_access requires a worker-daemon backend")
        device = spec.resources.device
        if profile is ContainerProfile.GVISOR and isinstance(device, (GpuDevice, TpuDevice)):
            raise InvalidResourceRequest("gvisor is CPU-only")

        if spec.replicas > 0:
            with self._db.read_snapshot() as tx:
                active_tasks = reads.count_active_tasks_for_user(tx, job_id.user)
            if active_tasks + spec.replicas > MAX_ACTIVE_TASKS_PER_USER:
                raise ResourceExhausted(
                    f"User {job_id.user} would exceed the active Task cap of {MAX_ACTIVE_TASKS_PER_USER}",
                )
        if job_id.parent is not None:
            with self._db.read_snapshot() as tx:
                parent_state = reads.get_job_state(tx, job_id.parent)
            if parent_state is None:
                raise ResourcePreconditionFailed(f"Parent Job {job_id.parent} is absent")
            if parent_state in TERMINAL_JOB_STATES:
                raise ResourcePreconditionFailed(f"Parent Job {job_id.parent} has terminated")
        return spec

    def _prepare_replacement(
        self,
        job_id: JobName,
        spec: JobSpec,
        federation: FederationSubmission | None,
        record_tombstone: bool,
    ) -> _SubmissionPreparation:
        with self._db.transaction() as tx:
            state = reads.get_job_state(tx, job_id)
            if state is None:
                return _SubmissionPreparation.READY
            if federation is not None and self._federated_replay(tx, job_id, federation):
                return _SubmissionPreparation.REUSE_EXISTING
            policy = spec.existing_job_policy
            if policy is ExistingJobPolicy.ERROR:
                raise ResourceConflict(f"Job {job_id} already exists")
            if policy is ExistingJobPolicy.KEEP:
                if not is_job_finished(state):
                    return _SubmissionPreparation.REUSE_EXISTING
                return self._replace_finished(tx, job_id, record_tombstone)
            elif policy is ExistingJobPolicy.RECREATE:
                if is_job_finished(state):
                    return self._replace_finished(tx, job_id, record_tombstone)
                else:
                    ops.job.cancel(tx, job_id=job_id, reason="Replaced by new submission")
                    return _SubmissionPreparation.WAIT_FOR_DRAIN
            elif is_job_finished(state):
                return self._replace_finished(tx, job_id, record_tombstone)
            else:
                raise ResourceConflict(f"Job {job_id} already exists and is still running")

    @staticmethod
    def _replace_finished(
        tx: Tx,
        job_id: JobName,
        record_tombstone: bool,
    ) -> _SubmissionPreparation:
        if reads.has_unfinished_worker_attempts(tx, job_id):
            return _SubmissionPreparation.WAIT_FOR_DRAIN
        ops.job.remove_finished(tx, job_id, record_tombstone=record_tombstone)
        return _SubmissionPreparation.READY

    def _wait_until_drained(self, job_id: JobName) -> bool:
        def drained() -> bool:
            with self._db.read_snapshot() as tx:
                return not reads.has_unfinished_worker_attempts(tx, job_id)

        return ExponentialBackoff(initial=1.0, maximum=10.0, factor=2).wait_until(
            drained,
            timeout=_JOB_REPLACEMENT_DRAIN_WAIT,
        )

    def _store_payloads(self, spec: JobSpec, bundle_blob: bytes) -> JobSpec:
        if bundle_blob:
            if len(bundle_blob) > MAX_BUNDLE_SIZE_BYTES:
                raise InvalidResourceRequest("Job bundle exceeds the 25 MiB limit")
            spec = replace(spec, bundle_id=self._bundle_store.write(bundle_blob))
        large_files = {
            name: data
            for name, data in spec.entrypoint.workdir_files.items()
            if len(data) > WORKDIR_FILE_OFFLOAD_THRESHOLD
        }
        if not large_files:
            return spec
        files = dict(spec.entrypoint.workdir_files)
        refs = dict(spec.entrypoint.workdir_file_refs)
        for name, data in large_files.items():
            del files[name]
            refs[name] = self._bundle_store.write(data)
        return replace(
            spec,
            entrypoint=replace(spec.entrypoint, workdir_files=files, workdir_file_refs=refs),
        )

    def _submission_plan(
        self,
        job_id: JobName,
        spec: JobSpec,
        federation: FederationSubmission | None,
    ) -> tuple[SubmitPlan, str]:
        local_backend = backend_directive(spec.constraints)
        cluster_pin = cluster_directive(spec.constraints)
        if local_backend is not None and cluster_pin is not None:
            raise InvalidResourceRequest(f"Job {job_id} pins both a backend and a cluster")
        if cluster_pin is not None and not self._runtime.federation.has_peer(cluster_pin):
            raise InvalidResourceRequest(f"Cluster {cluster_pin!r} is not a configured peer")
        candidates = (
            [self._runtime.backends[local_backend]]
            if local_backend is not None and local_backend in self._runtime.backends
            else list(self._runtime.backends.values()) if local_backend is None else []
        )
        errors: list[str] = []
        feasible = False
        for backend in candidates:
            if backend.autoscaler is None:
                feasible = True
                break
            error = backend.autoscaler.job_feasibility(
                constraints=list(spec.constraints),
                replicas=spec.replicas if spec.coscheduling is not None else None,
                resources=spec.resources,
            )
            if error is None:
                feasible = True
                break
            errors.append(error)
        if federation is not None:
            return SubmitPlan(SubmitDisposition.LOCAL), ""
        plan = self._runtime.federation.classify_submit(
            RoutingRequest(
                constraints=list(spec.constraints),
                local_feasible=feasible,
                cluster_pin=cluster_pin or "",
            )
        )
        return plan, errors[0] if errors else "no backend or peer can host the Job"

    def _federated_replay(self, tx: Tx, job_id: JobName, submission: FederationSubmission) -> bool:
        handoff = reads.received_handoff(tx, job_id)
        if handoff is None or handoff.requester_id != submission.requester_id:
            raise ResourceConflict(f"Job {job_id} belongs to another authority")
        if handoff.handoff_nonce != submission.handoff_nonce:
            return False
        writes.record_federation_change(tx, job_id)
        return True

    def _authorize_federation_handoff(
        self,
        identity: VerifiedIdentity | None,
        submission: FederationSubmission,
        job_id: JobName,
    ) -> None:
        if submission.submitting_user == LOCAL_ADMIN_SUBMITTER:
            raise ResourcePermissionDenied("Federated Jobs require an authenticated submitter")
        if identity is not None and identity.role == FEDERATION_PEER_ROLE:
            if submission.requester_id != identity.user_id:
                raise ResourcePermissionDenied("Federation requester does not match its token")
            if not user_admitted(self._auth.allowed_submitters, submission.submitting_user):
                raise ResourcePermissionDenied("Federated submitter is not admitted")
        elif identity is None or identity.role != "admin":
            raise ResourcePermissionDenied("Only a trusted peer may hand off a Job")
        if not job_id.is_root:
            raise InvalidResourceRequest("A federation handoff must be a root Job")


def _submitting_user(
    identity: VerifiedIdentity | None,
    federation: FederationSubmission | None,
) -> str:
    if federation is not None:
        return federation.submitting_user
    if identity is None or identity.user_id == ANONYMOUS_ADMIN.user_id:
        return LOCAL_ADMIN_SUBMITTER
    return identity.user_id


def _check_client_freshness(client_date_text: str, today: date) -> None:
    try:
        client_date = date.fromisoformat(client_date_text) if client_date_text else _FEATURE_INTRODUCTION_DATE
    except ValueError as exc:
        raise InvalidResourceRequest("client_revision_date must be ISO YYYY-MM-DD") from exc
    floor = today - CLIENT_FRESHNESS_WINDOW
    if client_date < floor:
        raise ResourcePreconditionFailed(
            f"marin-iris client is too old (build {client_date.isoformat()}; minimum {floor.isoformat()})",
        )


def _child_federation_refusal(job_id: JobName, peer_id: str) -> str:
    return f"Only a root Job can federate to {peer_id!r}; {job_id} must stay with its parent"


class JobService:
    """Submission, observation, and cancellation for Job resources."""

    def __init__(
        self,
        dependencies: ResourceDependencies,
        *,
        bundle_store: BundleStore,
        auth: ControllerAuth,
        user_budget_defaults: UserBudgetDefaults,
    ) -> None:
        self._dependencies = dependencies
        self._admission = _JobAdmission(
            db=dependencies.db,
            runtime=dependencies.runtime,
            bundle_store=bundle_store,
            auth=auth,
            user_budget_defaults=user_budget_defaults,
        )

    def received_job_from_peer(self, root_job: JobName, peer_id: str) -> bool:
        """Whether ``root_job`` is a handoff received from ``peer_id``."""
        with self._dependencies.db.read_snapshot() as tx:
            return reads.has_received_job_from_peer(tx, peer_id, root_job)

    def submit_job(
        self,
        spec: JobSpec,
        bundle_blob: bytes = b"",
        *,
        enforce_client_freshness: bool = True,
    ) -> JobIdentity:
        job_id = self._admission.submit(
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
        job_id = self._admission.submit(
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
                "resource_id": query.resource_id,
                "owner_id": query.owner_id,
                "parent": query.parent.resource_id if query.parent else None,
                "job_id_prefix": query.job_id_prefix,
                "states": sorted(int(state) for state in query.states),
                "backend_id": query.backend_id,
                "execution_cluster_id": query.execution_cluster_id,
                "top_level_only": query.top_level_only,
                "page_size": page_size,
            },
        )
        position = _decode_page_token(query.page_token, fingerprint)
        offset = int(position["offset"]) if position is not None else 0
        parent_job_id = None
        resource_id = JobName.from_wire(query.resource_id) if query.resource_id is not None else None
        if query.parent is not None:
            _require_kind(query.parent, ResourceKind.JOB)
            parent_job_id = JobName.from_wire(query.parent.resource_id)
        with self._dependencies.db.read_snapshot() as tx:
            rows = reads.list_resource_jobs(
                tx,
                owner_id=query.owner_id,
                resource_id=resource_id,
                job_id_prefix=_escaped_prefix(query.job_id_prefix) if query.job_id_prefix is not None else None,
                parent_job_id=parent_job_id,
                states=[int(state) for state in query.states],
                backend_id=query.backend_id,
                execution_cluster=(
                    _stored_cluster(self._dependencies.cluster_id, query.execution_cluster_id)
                    if query.execution_cluster_id is not None
                    else None
                ),
                top_level_only=query.top_level_only,
                offset=offset,
                limit=page_size + 1,
            )
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
            source_statuses=resource_source_statuses(self._dependencies),
        )

    def describe_job(self, key: ResourceKey) -> JobDetail:
        _require_kind(key, ResourceKind.JOB)
        job_id = JobName.from_wire(key.resource_id)
        with self._dependencies.db.read_snapshot() as tx:
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
        with self._dependencies.db.read_snapshot() as tx:
            states = reads.resource_job_states(tx, normalized)
        return {resource_id: JobState(state) for resource_id, state in states.items()}

    def job_state(self, identity: JobIdentity) -> JobState:
        """Return the state of one exact Job without loading its specification."""
        _require_kind(identity.key, ResourceKind.JOB)
        job_id = JobName.from_wire(identity.key.resource_id)
        with self._dependencies.db.read_snapshot() as tx:
            coordinates = reads.job_coordinates(tx, {job_id}).get(job_id)
        if coordinates is None:
            raise ResourceNotFound(identity.key.resource_id)
        if self._job_identity(coordinates) != identity:
            raise ResourceReplaced(identity.key.resource_id)
        return JobState(coordinates.state)

    def observe_jobs(self, summaries: Sequence[JobSummary]) -> tuple[JobObservation, ...]:
        """Read bounded Task, child, and federation aggregates for Jobs in one snapshot."""
        if len(summaries) > _MAX_JOB_PAGE:
            raise ValueError(f"Job observation batch cannot exceed {_MAX_JOB_PAGE} items")
        if not summaries:
            return ()
        job_ids = [JobName.from_wire(summary.identity.key.resource_id) for summary in summaries]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("Job observation keys must be unique")
        with self._dependencies.db.read_snapshot() as tx:
            attempt_counts = reads.attempt_counts_for_jobs(tx, job_ids)
            task_aggregates = reads.task_summaries_for_jobs(tx, job_ids, attempt_counts=attempt_counts)
            parents = reads.parent_ids_with_children(tx, job_ids)
            handoff_states = reads.handoff_states(tx, job_ids)
        return self._build_job_observations(summaries, task_aggregates, parents, handoff_states)

    def list_job_inventory(self, query: JobInventoryQuery) -> JobInventoryPage:
        """Return one bounded, sorted Job inventory page in a single snapshot."""
        if query.limit <= 0 or query.limit > _MAX_JOB_PAGE:
            raise ValueError(f"Job inventory limit must be between 1 and {_MAX_JOB_PAGE}")
        if query.offset < 0:
            raise ValueError("Job inventory offset cannot be negative")
        parent_job_id = None
        if query.scope is JobListScope.CHILDREN:
            if query.parent_resource_id is None:
                raise ValueError("parent_resource_id is required for child scope")
            parent_job_id = JobName.from_wire(query.parent_resource_id)
        with self._dependencies.db.read_snapshot() as tx:
            rows, total_count = reads.list_inventory_jobs(
                tx,
                scope=query.scope,
                parent_job_id=parent_job_id,
                states=[int(state) for state in query.states],
                name_contains=query.name_contains,
                job_id_prefix=_escaped_prefix(query.job_id_prefix) if query.job_id_prefix is not None else None,
                backend_id=query.backend_id,
                execution_cluster=(
                    _stored_cluster(self._dependencies.cluster_id, query.execution_cluster_id)
                    if query.execution_cluster_id is not None
                    else None
                ),
                sort_field=query.sort_field,
                sort_direction=query.sort_direction,
                offset=query.offset,
                limit=query.limit,
            )
            job_ids = [row.job_id for row in rows]
            parent_coordinates = self._job_coordinates_in_snapshot(
                tx,
                {row.parent_job_id for row in rows if row.parent_job_id is not None},
            )
            attempt_counts = reads.attempt_counts_for_jobs(tx, job_ids)
            task_aggregates = reads.task_summaries_for_jobs(tx, job_ids, attempt_counts=attempt_counts)
            parents = reads.parent_ids_with_children(tx, job_ids)
            handoff_states = reads.handoff_states(tx, job_ids)
        summaries = tuple(self._job_summary_from_row(row, parent_coordinates=parent_coordinates) for row in rows)
        return JobInventoryPage(
            self._build_job_observations(summaries, task_aggregates, parents, handoff_states),
            total_count,
        )

    def _build_job_observations(
        self,
        summaries: Sequence[JobSummary],
        task_aggregates: Mapping[JobName, reads.TaskJobSummary],
        parents: set[JobName],
        handoff_states: Mapping[JobName, int],
    ) -> tuple[JobObservation, ...]:
        observations = []
        for summary in summaries:
            job_id = JobName.from_wire(summary.identity.key.resource_id)
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
        if summary.execution_cluster_id == self._dependencies.cluster_id:
            return FederationPosture.LOCAL
        return FederationPosture.ACCEPTED

    def cancel_job(
        self,
        identity: JobIdentity,
        *,
        idempotency_key: str,
        reason: str = "Cancelled through the resource API",
        principal_id: str = LOCAL_ADMIN_SUBMITTER,
    ) -> ActionReceipt:
        payload_hash = _action_payload_hash(ActionKind.CANCEL_JOB, identity.job_uid, None, reason)
        preparation = self._prepare_cancel_job(
            identity,
            principal_id=principal_id,
            idempotency_key=idempotency_key,
            reason=reason,
            payload_hash=payload_hash,
        )
        if isinstance(preparation, _RemoteActionContext):
            receipt = self._dependencies.runtime.federation.proxy_to_peer(
                preparation.peer_id,
                lambda peer: peer.cancel_job(identity, idempotency_key=idempotency_key, reason=reason),
            )
            return self._persist_remote_action(
                receipt,
                preparation,
                principal_id=principal_id,
                kind=ActionKind.CANCEL_JOB,
                idempotency_key=idempotency_key,
                payload_hash=payload_hash,
            )

        if preparation.cancel_target is not None:
            self._dependencies.runtime.federation.deliver_cancel(preparation.cancel_target)
        else:
            self._dependencies.runtime.wake()
        return preparation.receipt

    def _prepare_cancel_job(
        self,
        identity: JobIdentity,
        *,
        principal_id: str,
        idempotency_key: str,
        reason: str,
        payload_hash: str,
    ) -> _CompletedCancel | _RemoteActionContext:
        cancel_target: CancelTarget | None = None
        with self._dependencies.db.transaction() as tx:
            duplicate = _duplicate_action(
                tx,
                principal_id=principal_id,
                kind=ActionKind.CANCEL_JOB,
                idempotency_key=idempotency_key,
                payload_hash=payload_hash,
            )
            if duplicate is not None:
                return _CompletedCancel(duplicate, None)
            row = reads.get_job_detail(tx, JobName.from_wire(identity.key.resource_id))
            if row is None:
                raise ResourceNotFound(identity.key.resource_id)
            coordinates = self._job_rows(tx, {row.job_id})[row.job_id]
            authority = self._authority_cluster(coordinates)
            expected = self._job_identity(coordinates).job_uid
            if identity.key.cluster_id != authority or identity.job_uid != expected:
                raise ResourceReplaced(identity.key.resource_id)
            execution_cluster_id = _execution_cluster(self._dependencies.cluster_id, row.cluster)
            handle = reads.federated_handle(tx, row.job_id.root_job)
            if handle is None:
                job_ops.cancel(tx, job_id=row.job_id, reason=reason)
                writes.record_federation_change(tx, row.job_id)
            elif row.job_id != row.job_id.root_job:
                return _RemoteActionContext(handle.peer_id, authority, "", execution_cluster_id)
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
                        error=reason,
                    )
                if handle.handoff_state != int(HandoffState.QUEUED_HANDOFF):
                    cancel_target = CancelTarget(row.job_id, handle.peer_id)
            receipt = _completed_action(
                kind=ActionKind.CANCEL_JOB,
                target=identity.key,
                expected_target_uid=identity.job_uid,
                expected_attempt_uid=None,
                expected_attempt_number=None,
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
        return _CompletedCancel(receipt, cancel_target)

    def _job_summary_from_row(
        self,
        row: reads.JobRecord,
        *,
        coordinates: reads.JobCoordinates | None = None,
        parent_coordinates: Mapping[JobName, reads.JobCoordinates] | None = None,
    ) -> JobSummary:
        job_id = row.job_id
        authority = self._authority_cluster(coordinates or row)
        execution = _execution_cluster(self._dependencies.cluster_id, str(row.cluster))
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
                    handoff_nonce=str((coordinates or row).handoff_nonce or ""),
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
            exit_code=int(row.exit_code) if row.exit_code is not None else None,
            resources=resource_spec_from_scalars(
                row.res_cpu_millicores,
                row.res_memory_bytes,
                row.res_disk_bytes,
                row.res_device_json,
            ),
        )

    def _job_pending_reason(
        self,
        row: reads.JobRecord,
        coordinates: reads.JobCoordinates | reads.JobRecord,
    ) -> str:
        if JobState(row.state) is not JobState.PENDING:
            return ""
        if coordinates.direction == int(FederationDirection.SENT):
            peer_id = str(coordinates.peer_id or "")
            handoff_state = coordinates.handoff_state
            if handoff_state == int(HandoffState.QUEUED_HANDOFF):
                if peer_id:
                    return f"Queued for peer {peer_id} to report free capacity"
                return "Queued for a federation peer to report free capacity"
            if handoff_state == int(HandoffState.PENDING_HANDOFF):
                return f"Awaiting acceptance by peer {peer_id}"
            return f"Pending on peer {peer_id}"

        scheduler_reason = self._dependencies.runtime.get_job_scheduling_diagnostics(row.job_id.to_wire())
        pending_reason = scheduler_reason or "Pending scheduler feedback"
        hint = None
        for backend in self._dependencies.backends.values():
            if backend.autoscaler is not None:
                hint = backend.autoscaler.get_pending_hints().get(row.job_id.to_wire())
                if hint is not None:
                    break
        if hint is None:
            return pending_reason
        scaling_prefix = "(scaling up) " if hint.is_scaling_up else ""
        return f"Scheduler: {pending_reason}\n\nAutoscaler: {scaling_prefix}{hint.message}"

    def _job_identity(self, row: reads.JobCoordinates | reads.JobRecord) -> JobIdentity:
        authority = self._authority_cluster(row)
        return JobIdentity(
            ResourceKey(authority, ResourceKind.JOB, row.job_id.to_wire()),
            _job_uid(
                authority,
                row.job_id,
                row.submitted_at_ms,
                handoff_nonce=str(row.handoff_nonce or ""),
            ),
        )

    def _authority_cluster(self, row: reads.JobCoordinates | reads.JobRecord) -> str:
        if row.direction == int(FederationDirection.RECEIVED):
            return str(row.peer_id)
        return self._dependencies.cluster_id

    def _known_backend_id(self, stored: str, execution_cluster_id: str) -> str:
        if stored or execution_cluster_id != self._dependencies.cluster_id:
            return stored
        if len(self._dependencies.backends) == 1:
            return next(iter(self._dependencies.backends))
        return ""

    def _job_rows(self, tx: Tx, job_ids: set[JobName]) -> dict[JobName, reads.JobCoordinates]:
        return reads.job_coordinates(tx, job_ids)

    def _job_authorities(self, job_ids: Iterable[JobName]) -> dict[JobName, str]:
        ids = set(job_ids)
        with self._dependencies.db.read_snapshot() as tx:
            return {job_id: self._authority_cluster(row) for job_id, row in self._job_rows(tx, ids).items()}

    @staticmethod
    def _job_coordinates_in_snapshot(
        tx: Tx,
        job_ids: set[JobName],
    ) -> dict[JobName, reads.JobCoordinates]:
        return reads.job_coordinates(tx, job_ids)

    def _persist_remote_action(
        self,
        receipt: ActionReceipt,
        remote: _RemoteActionContext,
        *,
        principal_id: str,
        kind: ActionKind,
        idempotency_key: str,
        payload_hash: str,
    ) -> ActionReceipt:
        with self._dependencies.db.transaction() as tx:
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
                authority_cluster_id=remote.authority_cluster_id,
                authority_action_id=receipt.action_id,
                backend_id=remote.backend_id,
                execution_cluster_id=remote.execution_cluster_id,
                principal_id=principal_id,
                idempotency_key=_require_idempotency_key(idempotency_key),
                payload_hash=payload_hash,
            )
        return receipt
