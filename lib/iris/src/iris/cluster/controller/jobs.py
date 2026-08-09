# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource-native Job admission."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import Enum, auto
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import ANONYMOUS_ADMIN, VerifiedIdentity, get_verified_identity
from rigging.timing import Duration, ExponentialBackoff, Timestamp

from iris.cluster.authorization import authorize_resource_owner
from iris.cluster.bundle import MAX_BUNDLE_SIZE_BYTES, BundleStore
from iris.cluster.config import user_admitted
from iris.cluster.constraints import (
    backend_directive,
    cluster_directive,
    constraints_from_resources,
    merge_constraints,
    validate_tpu_request,
)
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.backend import BackendCapability, TaskBackend
from iris.cluster.controller.persistence import operations as ops
from iris.cluster.controller.persistence import reads, writes
from iris.cluster.controller.persistence.database import ControllerDB, Tx
from iris.cluster.controller.reconcile.policy import MAX_ACTIVE_TASKS_PER_USER
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.router import RoutingRequest, SubmitDisposition, SubmitPlan
from iris.cluster.types import (
    LOCAL_ADMIN_SUBMITTER,
    TERMINAL_JOB_STATES,
    JobName,
    UserBudgetDefaults,
    is_job_finished,
)
from iris.resources.execution import GpuDevice, TpuDevice
from iris.resources.job import ContainerProfile, ExistingJobPolicy, JobSpec, PriorityBand
from iris.rpc.auth import FEDERATION_PEER_ROLE, AuthzAction, authorize

logger = logging.getLogger(__name__)

WORKDIR_FILE_OFFLOAD_THRESHOLD = 10 * 1024
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


class JobResources:
    """Admit Jobs from typed specifications."""

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
            raise ConnectError(Code.INVALID_ARGUMENT, "Job name is required")
        if spec.coscheduling is not None and not spec.coscheduling.group_by:
            raise ConnectError(Code.INVALID_ARGUMENT, "coscheduling requires a non-empty group_by")

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
            raise ConnectError(Code.INVALID_ARGUMENT, tpu_error)

        plan, rejection_reason = self._submission_plan(job_id, spec, federation)
        if plan.disposition is SubmitDisposition.QUEUE:
            if not job_id.is_root:
                raise ConnectError(Code.INVALID_ARGUMENT, _child_federation_refusal(job_id, plan.pinned_peer_id))
            if self._auth.provider and submitting_user == LOCAL_ADMIN_SUBMITTER:
                raise ConnectError(Code.PERMISSION_DENIED, _LOCAL_ADMIN_FEDERATION_DENIED)
            self._runtime.federation.queue_federated(
                local_job_id=job_id,
                spec=spec,
                pinned_peer_id=plan.pinned_peer_id,
                owner_principal=job_id.user,
                submitting_user=submitting_user,
            )
            return job_id
        if plan.disposition is SubmitDisposition.REJECT:
            raise ConnectError(Code.FAILED_PRECONDITION, f"Job {job_id} is unschedulable: {rejection_reason}")

        with self._db.transaction() as tx:
            if reads.get_job_state(tx, job_id) is not None:
                if federation is not None and self._federated_replay(tx, job_id, federation):
                    return job_id
                if spec.existing_job_policy is ExistingJobPolicy.KEEP:
                    return job_id
                raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists (concurrent submission)")
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
            raise ConnectError(Code.INVALID_ARGUMENT, f"Unknown priority_band {int(spec.priority_band)}")
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
                    raise ConnectError(
                        Code.PERMISSION_DENIED,
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
            raise ConnectError(Code.INVALID_ARGUMENT, "docker_access requires a worker-daemon backend")
        device = spec.resources.device
        if profile is ContainerProfile.GVISOR and isinstance(device, (GpuDevice, TpuDevice)):
            raise ConnectError(Code.INVALID_ARGUMENT, "gvisor is CPU-only")

        if spec.replicas > 0:
            with self._db.read_snapshot() as tx:
                active_tasks = reads.count_active_tasks_for_user(tx, job_id.user)
            if active_tasks + spec.replicas > MAX_ACTIVE_TASKS_PER_USER:
                raise ConnectError(
                    Code.RESOURCE_EXHAUSTED,
                    f"User {job_id.user} would exceed the active Task cap of {MAX_ACTIVE_TASKS_PER_USER}",
                )
        if job_id.parent is not None:
            with self._db.read_snapshot() as tx:
                parent_state = reads.get_job_state(tx, job_id.parent)
            if parent_state is None:
                raise ConnectError(Code.FAILED_PRECONDITION, f"Parent Job {job_id.parent} is absent")
            if parent_state in TERMINAL_JOB_STATES:
                raise ConnectError(Code.FAILED_PRECONDITION, f"Parent Job {job_id.parent} has terminated")
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
                raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists")
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
                raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} already exists and is still running")

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
                raise ConnectError(Code.INVALID_ARGUMENT, "Job bundle exceeds the 25 MiB limit")
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
            raise ConnectError(Code.INVALID_ARGUMENT, f"Job {job_id} pins both a backend and a cluster")
        if cluster_pin is not None and not self._runtime.federation.has_peer(cluster_pin):
            raise ConnectError(Code.INVALID_ARGUMENT, f"Cluster {cluster_pin!r} is not a configured peer")
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
            raise ConnectError(Code.ALREADY_EXISTS, f"Job {job_id} belongs to another authority")
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
            raise ConnectError(Code.PERMISSION_DENIED, "Federated Jobs require an authenticated submitter")
        if identity is not None and identity.role == FEDERATION_PEER_ROLE:
            if submission.requester_id != identity.user_id:
                raise ConnectError(Code.PERMISSION_DENIED, "Federation requester does not match its token")
            if not user_admitted(self._auth.allowed_submitters, submission.submitting_user):
                raise ConnectError(Code.PERMISSION_DENIED, "Federated submitter is not admitted")
        elif identity is None or identity.role != "admin":
            raise ConnectError(Code.PERMISSION_DENIED, "Only a trusted peer may hand off a Job")
        if not job_id.is_root:
            raise ConnectError(Code.INVALID_ARGUMENT, "A federation handoff must be a root Job")


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
        raise ConnectError(Code.INVALID_ARGUMENT, "client_revision_date must be ISO YYYY-MM-DD") from exc
    floor = today - CLIENT_FRESHNESS_WINDOW
    if client_date < floor:
        raise ConnectError(
            Code.FAILED_PRECONDITION,
            f"marin-iris client is too old (build {client_date.isoformat()}; minimum {floor.isoformat()})",
        )


def _child_federation_refusal(job_id: JobName, peer_id: str) -> str:
    return f"Only a root Job can federate to {peer_id!r}; {job_id} must stay with its parent"
