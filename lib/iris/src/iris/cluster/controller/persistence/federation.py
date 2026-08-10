# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The controller-side :class:`FederationStore` implementation.

Backs the federation manager's durable operations against the controller's own
tables: it admits and persists handoff handles, re-drives pending handoffs, mirrors
a peer's synced state into the local ``jobs``/``tasks`` rows, and routes cancel
intent. Keeping this on the controller side lets the federation module depend only
on the ``FederationStore`` protocol.
"""

import logging
import uuid

from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import (
    peer_availability_gate,
    routing_constraints,
    strip_backend_constraints,
    strip_cluster_constraints,
)
from iris.cluster.controller.persistence import operations as ops
from iris.cluster.controller.persistence import reads, writes
from iris.cluster.controller.persistence.database import ControllerDB, Tx
from iris.cluster.controller.persistence.json_codec import reconstruct_job_spec
from iris.cluster.controller.persistence.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.controller.persistence.projections.endpoints import EndpointRow, EndpointsProjection
from iris.cluster.controller.persistence.projections.run_templates import RunTemplatesProjection
from iris.cluster.federation.availability import QueuedCandidate
from iris.cluster.federation.protocol import (
    CancelTarget,
    FederationJobDelta,
    HandoffAdmission,
    HandoffSpec,
    HandoffState,
    SyncedEndpoint,
    SyncedJob,
    SyncedTask,
)
from iris.cluster.types import TERMINAL_JOB_STATES
from iris.resources.names import JobName

logger = logging.getLogger(__name__)

# Cap on a mirrored federated endpoint's local lease. The peer re-reports its full
# endpoint set every sync (~3s), refreshing this, so the cap only bounds how long a
# stale mirror keeps forwarding after the peer goes silent — short enough to stop
# routing to a lost peer, long enough to ride out sync jitter.
FEDERATED_ENDPOINT_MIRROR_TTL = Duration.from_minutes(5)


def build_queued_candidates(tx: Tx) -> list[QueuedCandidate]:
    """Read the queued federated jobs into candidates for the tick's federation pass.

    Each candidate carries its shape (routing constraints) and its
    ``ge(available:<token>, amount)`` availability gate, derived from the job's stored
    request. Ordered by priority band ascending (lower band = higher priority), then
    oldest submission first — the order the assignment pass consumes them in.
    """
    candidates: list[QueuedCandidate] = []
    for handle in reads.queued_handoff_handles(tx):
        job = reads.get_job_detail(tx, handle.job_id)
        # Skip a job the tick already terminalized this pass (scheduling timeout, cancel,
        # or a parent cascade): its handle still reads QUEUED_HANDOFF but it must not be
        # promoted — the promotion CAS would reject it anyway.
        if job is None or job.state in TERMINAL_JOB_STATES:
            continue
        # Shape only — this pass reads constraints and resources, never the payload.
        spec = reconstruct_job_spec(job, workdir_files={})
        shape = routing_constraints(strip_cluster_constraints(strip_backend_constraints(list(spec.constraints))))
        candidates.append(
            QueuedCandidate(
                job_id=handle.job_id,
                pinned_peer_id=handle.peer_id,
                priority_band=job.priority_band,
                submitted_at_ms=job.submitted_at_ms.epoch_ms() if job.submitted_at_ms is not None else 0,
                shape_constraints=shape,
                availability_gate=peer_availability_gate(spec.resources.device, spec.replicas),
            )
        )
    candidates.sort(key=lambda c: (c.priority_band, c.submitted_at_ms))
    return candidates


def _epoch_ms(value: Timestamp | None) -> int | None:
    return value.epoch_ms() if value is not None else None


def _endpoint_rows(peer_id: str, endpoints: tuple[SyncedEndpoint, ...], now: Timestamp) -> list[EndpointRow]:
    """Build mirror ``EndpointRow``s from a peer's reported endpoint snapshot.

    The lease deadline is ``now`` plus the reported remaining time, capped at
    :data:`FEDERATED_ENDPOINT_MIRROR_TTL` so a mirror never outlives contact with
    the peer. A snapshot without ``lease_remaining`` still expires under the cap.
    """
    rows: list[EndpointRow] = []
    for endpoint in endpoints:
        remaining = endpoint.lease_remaining
        remaining_ms = FEDERATED_ENDPOINT_MIRROR_TTL.to_ms() if remaining is None else remaining.to_ms()
        remaining_ms = min(remaining_ms, FEDERATED_ENDPOINT_MIRROR_TTL.to_ms())
        rows.append(
            EndpointRow(
                endpoint_id=endpoint.endpoint_id,
                name=endpoint.name,
                address=endpoint.address,
                task_id=endpoint.task_id,
                metadata=dict(endpoint.metadata),
                registered_at=now,
                lease_deadline=now.add_ms(remaining_ms),
                access=endpoint.access,
                peer_id=peer_id,
            )
        )
    return rows


class ControllerFederationStore:
    """A :class:`~iris.cluster.federation.protocol.FederationStore` over ``ControllerDB``."""

    def __init__(
        self,
        db: ControllerDB,
    ):
        self._db = db

    # -- handoff -------------------------------------------------------------

    def admit_and_persist_queued(self, spec: HandoffSpec) -> HandoffAdmission:
        now = Timestamp.now()
        with self._db.transaction() as cur:
            if reads.get_job_state(cur, spec.local_job_id) is not None:
                return HandoffAdmission.ALREADY_EXISTS

            # A pinned candidate names its peer as the cluster coordinate up front (so
            # status reads surface "queued for peer X"); an unpinned one has no peer yet,
            # so cluster is "" until the tick's promotion stamps the chosen peer. Both are
            # is_federated (!= LOCAL_CLUSTER), so a queued job is never folded into local
            # scheduling — it owns no task rows and waits for a peer.
            ops.job.insert_job_and_config(
                cur,
                job_id=spec.local_job_id,
                spec=spec.spec,
                ts=now,
                priority_band=int(spec.spec.priority_band),
                cluster=spec.peer_id,
                submitting_user=spec.submitting_user,
            )
            writes.insert_federated_handle(
                cur,
                job_id=spec.local_job_id,
                peer_id=spec.peer_id,
                owner_principal=spec.owner_principal,
                handoff_state=int(HandoffState.QUEUED_HANDOFF),
                # Each admission is a new incarnation; every re-drive repeats its nonce.
                handoff_nonce=uuid.uuid4().hex,
            )
        return HandoffAdmission.ADMITTED

    def mark_handed_off(self, local_job_id: JobName) -> None:
        with self._db.transaction() as cur:
            writes.set_handoff_state(cur, local_job_id, int(HandoffState.HANDED_OFF))

    def mark_handoff_rejected(self, local_job_id: JobName, *, reason: str) -> None:
        with self._db.transaction() as cur:
            writes.set_handoff_state(cur, local_job_id, int(HandoffState.HANDOFF_REJECTED))
            writes.mark_federated_job_killed(cur, local_job_id, now_ms=Timestamp.now().epoch_ms(), error=reason)

    def pending_handoffs(self) -> list[HandoffSpec]:
        """The handles awaiting delivery, each with the full request the peer will run.

        The peer executes this request, so it carries the job's inline workdir files
        (the small ones the offload threshold kept out of the blob store, e.g. a
        ``from_callable`` job's ``_callable_runner.py``) alongside the stored config.
        """
        with self._db.read_snapshot() as tx:
            handles = reads.pending_handoff_handles(tx)
            pending = []
            for handle in handles:
                job = reads.get_job_detail(tx, handle.job_id)
                if job is None:
                    continue
                pending.append(
                    HandoffSpec(
                        local_job_id=handle.job_id,
                        peer_id=handle.peer_id,
                        owner_principal=handle.owner_principal,
                        submitting_user=job.submitting_user,
                        spec=reconstruct_job_spec(job, workdir_files=reads.get_workdir_files(tx, handle.job_id)),
                        handoff_nonce=handle.handoff_nonce,
                    )
                )
        return pending

    # -- cancel --------------------------------------------------------------

    def bump_cancel_intent(self, local_job_id: JobName) -> CancelTarget | None:
        """Bump a SENT handle's cancel intent and return the peer to cancel.

        Returns ``None`` when ``local_job_id`` is not a SENT handle. A handle the
        peer never received (still ``PENDING_HANDOFF``) is terminated locally here —
        the re-drive now skips it, so no sync will ever mirror it terminal. A
        delivered handle keeps its synced state: the routed cancel drives it terminal
        on the peer and the next sync reflects it.
        """
        with self._db.transaction() as cur:
            handle = reads.federated_handle(cur, local_job_id)
            if handle is None:
                return None
            writes.bump_cancel_intent(cur, local_job_id)
            # A handle the peer has not yet accepted — QUEUED_HANDOFF (before the tick
            # promotes it) or PENDING_HANDOFF (before the peer acks) — owns no peer-side
            # job, so it is terminated locally; the bumped intent makes the promotion CAS
            # / re-drive skip it, so no sync ever mirrors it terminal.
            not_yet_delivered = (int(HandoffState.QUEUED_HANDOFF), int(HandoffState.PENDING_HANDOFF))
            if handle.handoff_state in not_yet_delivered:
                writes.mark_federated_job_killed(
                    cur, local_job_id, now_ms=Timestamp.now().epoch_ms(), error="Cancelled before handoff"
                )
            # A QUEUED handle has no peer yet (peer_id may be ""), so there is nothing to
            # route a TerminateJob to. A PENDING or delivered handle names its peer, so a
            # routed cancel drives it terminal on the peer as a best effort against a race.
            if handle.handoff_state == int(HandoffState.QUEUED_HANDOFF):
                return None
            return CancelTarget(local_job_id=local_job_id, peer_id=handle.peer_id)

    def pending_cancels(self) -> list[CancelTarget]:
        with self._db.read_snapshot() as tx:
            return [CancelTarget(local_job_id=h.job_id, peer_id=h.peer_id) for h in reads.pending_cancel_handles(tx)]

    def mark_cancel_satisfied(self, local_job_id: JobName, *, now_ms: int) -> None:
        with self._db.transaction() as cur:
            writes.mark_federated_job_killed(
                cur, local_job_id, now_ms=now_ms, error="Cancelled (peer reported the job gone)"
            )

    # -- sync ----------------------------------------------------------------

    def read_cursor(self, peer_id: str) -> str:
        with self._db.read_snapshot() as tx:
            return reads.read_sync_cursor(tx, peer_id)

    def active_federated_job_count(self, peer_id: str) -> int:
        with self._db.read_snapshot() as tx:
            return reads.active_federated_job_count(tx, peer_id)

    def apply_sync_batch(
        self,
        peer_id: str,
        deltas: tuple[FederationJobDelta, ...],
        *,
        next_cursor: str,
        cursor_stale: bool,
        endpoints: tuple[SyncedEndpoint, ...] = (),
    ) -> None:
        with self._db.transaction() as cur:
            for delta in deltas:
                # Job ids are cluster-invariant: the peer reports the same id the
                # parent handed it. Guard on a SENT handle for the delta's *root* —
                # a job whose root this parent handed to this peer is legitimately
                # part of that subtree, whether it is the root itself or a child the
                # peer spawned under it. A delta whose root was never handed here is
                # a disagreement, not normal traffic, so log and ignore it.
                local_job_id = delta.job_id
                if reads.federated_sent_job(cur, peer_id, local_job_id.root_job) is None:
                    logger.warning("peer %s reported job %s it was not handed; ignoring", peer_id, local_job_id)
                    continue
                if delta.tombstone:
                    # delete_job drops the job's mirrored endpoints (cache + DB) too.
                    writes.delete_job(cur, local_job_id)
                    continue
                self._mirror_delta(cur, peer_id, local_job_id, delta)

            if cursor_stale:
                self._set_replace(cur, peer_id, deltas)

            # Set-replace this peer's mirrored endpoints from its full reported set
            # (applied after the job/task deltas above create the FK targets).
            cur.caches[EndpointsProjection].replace_remote_for_peer(
                cur, peer_id, _endpoint_rows(peer_id, endpoints, Timestamp.now())
            )

            writes.upsert_sync_cursor(cur, peer_id, next_cursor)

    def _mirror_delta(self, cur: Tx, peer_id: str, local_job_id: JobName, delta: FederationJobDelta) -> None:
        summary = delta.summary
        if summary is None:
            raise ValueError(f"federation delta for {local_job_id} has no summary")
        existing = reads.get_job_detail(cur, local_job_id)
        backend_id = _canonical_delta_backend(existing, summary, delta.changed_tasks)
        first_backend_observation = existing is not None and not str(existing.backend_id or "") and bool(backend_id)
        # The root's mirror row is created at handoff; a child the peer spawned under
        # it has none until its first delta, so create it before mirroring state (and
        # before its tasks, which FK to the job row).
        if existing is None and not self._insert_child_mirror(cur, peer_id, local_job_id, summary):
            return
        writes.mirror_federated_job(
            cur,
            job_id=local_job_id,
            state=summary.state,
            error=summary.error_message or None,
            exit_code=summary.exit_code,
            started_at_ms=_epoch_ms(summary.started_at),
            finished_at_ms=_epoch_ms(summary.finished_at),
            num_tasks=summary.task_count,
            backend_id=backend_id,
        )
        if first_backend_observation:
            writes.adopt_federated_backend(cur, job_id=local_job_id, backend_id=backend_id)
        for task in delta.changed_tasks:
            peer_task_id = task.task_id
            index = peer_task_id.task_index
            if index is None:
                continue
            local_task_id = local_job_id.task(index)
            writes.mirror_federated_task(
                cur,
                task_id=local_task_id,
                job_id=local_job_id,
                task_index=index,
                peer_id=peer_id,
                state=task.state,
                error=task.error_message or None,
                exit_code=task.exit_code,
                submitted_at_ms=_epoch_ms(task.submitted_at),
                started_at_ms=_epoch_ms(task.started_at),
                finished_at_ms=_epoch_ms(task.finished_at),
                current_attempt_id=task.current_attempt_id,
                worker_address=task.worker_address,
                peer_worker_label=task.worker_label,
                status_message=task.status_message or None,
                backend_id=backend_id,
            )
            writes.mirror_federated_attempts(
                cur,
                task_id=local_task_id,
                attempts=task.attempts,
                backend_id=backend_id,
            )
            # The parent derives the federated task's counts from these mirrored
            # attempts, so drop the job's cached totals via the cursor's memo.
            cur.caches[AttemptCountsProjection].invalidate_for_tasks(cur, [local_task_id])

    def _insert_child_mirror(self, cur: Tx, peer_id: str, local_job_id: JobName, summary: SyncedJob) -> bool:
        """Create the local mirror rows for a child a peer spawned under a received root.

        The whole federated subtree shares the root's submitter, root submit time, and
        priority band, read from the (already-present) parent; the row is stamped with the peer
        cluster so it folds out of local scheduling and renders as federated. Returns
        ``False`` (and skips) if the parent is not mirrored yet — deltas arrive in
        changelog order, so the parent's creation precedes the child's and this is
        only a defensive guard; the child is re-created on its next delta.

        Writes the ``job_config`` companion the dashboard reads join against, so the
        child renders like any other job. Only the peer-reported resources are known
        here — the parent never authored a request for this job and never runs it —
        so the rest of the config keeps its column defaults.
        """
        parent = local_job_id.parent
        if parent is None:
            return False
        seed = reads.parent_mirror_seed(cur, parent)
        if seed is None:
            logger.warning("peer %s reported child %s before its parent; will retry", peer_id, local_job_id)
            return False
        root_submitted_ms = seed.root_submitted_at_ms.epoch_ms()
        writes.insert_job(
            cur,
            job_id=local_job_id,
            user_id=local_job_id.user,
            submitting_user=seed.submitting_user,
            parent_job_id=parent,
            root_job_id=local_job_id.root_job.to_wire(),
            depth=local_job_id.depth,
            state=summary.state,
            submitted_at_ms=_epoch_ms(summary.submitted_at) or root_submitted_ms,
            root_submitted_at_ms=root_submitted_ms,
            started_at_ms=_epoch_ms(summary.started_at),
            finished_at_ms=_epoch_ms(summary.finished_at),
            scheduling_deadline_epoch_ms=None,
            error=summary.error_message or None,
            exit_code=summary.exit_code,
            num_tasks=summary.task_count,
            name=local_job_id.name,
            cluster=peer_id,
        )
        writes.insert_mirrored_job_config(
            cur,
            job_id=local_job_id,
            name=local_job_id.name,
            resources=summary.resources,
            priority_band=int(seed.priority_band),
        )
        # job_config backs RunTemplatesProjection; invalidate post-commit per its
        # watch contract, as ops.job.submit does for a locally-submitted job.
        cur.caches[RunTemplatesProjection].invalidate_for_job(cur, local_job_id)
        return True

    def _set_replace(self, cur: Tx, peer_id: str, deltas: tuple[FederationJobDelta, ...]) -> None:
        """Full-resync set-replacement: drop any local handle for ``peer_id``
        absent from the peer's active set, reclaiming a job the parent never saw
        tombstoned."""
        active = {delta.job_id for delta in deltas if not delta.tombstone}
        for local_job_id in reads.federated_handles_for_peer(cur, peer_id):
            if local_job_id not in active:
                writes.delete_job(cur, local_job_id)


def _canonical_delta_backend(existing, summary: SyncedJob, tasks: tuple[SyncedTask, ...]) -> str:
    stored = str(existing.backend_id or "") if existing is not None else ""
    reported = {str(value) for value in (summary.backend_id, *(task.backend_id for task in tasks)) if value}
    if len(reported) > 1:
        raise ValueError(f"federation delta reports conflicting backend coordinates: {sorted(reported)}")
    asserted = next(iter(reported), "")
    if stored and asserted and stored != asserted:
        raise ValueError(f"federation delta changed backend coordinate from {stored!r} to {asserted!r}")
    return stored or asserted
