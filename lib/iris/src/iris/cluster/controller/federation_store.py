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

from rigging.timing import Timestamp

from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.codec import reconstruct_launch_job_request
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection
from iris.cluster.federation.store import (
    CancelTarget,
    HandoffAdmission,
    HandoffSpec,
    HandoffState,
)
from iris.cluster.types import JobName
from iris.time_proto import timestamp_from_proto

logger = logging.getLogger(__name__)


def _proto_ms(has: bool, ts) -> int | None:
    """Epoch ms of a proto ``Timestamp`` field, or ``None`` when unset."""
    return timestamp_from_proto(ts).epoch_ms() if has else None


class ControllerFederationStore:
    """A :class:`~iris.cluster.federation.store.FederationStore` over ``ControllerDB``."""

    def __init__(
        self,
        db: ControllerDB,
    ):
        self._db = db

    # -- handoff -------------------------------------------------------------

    def admit_and_persist_handoff(self, spec: HandoffSpec) -> HandoffAdmission:
        now = Timestamp.now()
        with self._db.transaction() as cur:
            if reads.get_job_state(cur, spec.local_job_id) is not None:
                # A handle already exists — a retried/idempotent resubmit.
                return HandoffAdmission.ALREADY_EXISTS

            ops.job.insert_job_and_config(
                cur,
                job_id=spec.local_job_id,
                request=spec.request,
                ts=now,
                cluster=spec.peer_id,
            )
            writes.insert_federated_handle(
                cur,
                job_id=spec.local_job_id,
                peer_id=spec.peer_id,
                owner_principal=spec.owner_principal,
                handoff_state=int(HandoffState.PENDING_HANDOFF),
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
                        request=reconstruct_launch_job_request(job),
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
            if handle.handoff_state == int(HandoffState.PENDING_HANDOFF):
                writes.mark_federated_job_killed(
                    cur, local_job_id, now_ms=Timestamp.now().epoch_ms(), error="Cancelled before handoff"
                )
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
        deltas,
        *,
        next_cursor: str,
        cursor_stale: bool,
    ) -> None:
        with self._db.transaction() as cur:
            for delta in deltas:
                # Job ids are cluster-invariant: the peer reports the same id the
                # parent handed it. Guard on a SENT handle for (peer, id) — a peer
                # reporting an id it was never handed is a disagreement, not normal
                # traffic, so log and ignore it.
                local_job_id = JobName.from_wire(delta.job_id)
                if reads.federated_sent_job(cur, peer_id, local_job_id) is None:
                    logger.warning("peer %s reported job %s it was not handed; ignoring", peer_id, local_job_id)
                    continue
                if delta.tombstone:
                    writes.delete_job(cur, local_job_id)
                    continue
                self._mirror_delta(cur, peer_id, local_job_id, delta)

            if cursor_stale:
                self._set_replace(cur, peer_id, deltas)

            writes.upsert_sync_cursor(cur, peer_id, next_cursor)

    def _mirror_delta(self, cur: Tx, peer_id: str, local_job_id: JobName, delta) -> None:
        summary = delta.summary
        writes.mirror_federated_job(
            cur,
            job_id=local_job_id,
            state=summary.state,
            error=summary.error or None,
            exit_code=summary.exit_code or None,
            started_at_ms=_proto_ms(summary.HasField("started_at"), summary.started_at),
            finished_at_ms=_proto_ms(summary.HasField("finished_at"), summary.finished_at),
            num_tasks=summary.task_count,
        )
        for task in delta.changed_tasks:
            peer_task_id = JobName.from_wire(task.task_id)
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
                error=task.error or None,
                exit_code=task.exit_code or None,
                submitted_at_ms=_proto_ms(task.HasField("submitted_at"), task.submitted_at),
                started_at_ms=_proto_ms(task.HasField("started_at"), task.started_at),
                finished_at_ms=_proto_ms(task.HasField("finished_at"), task.finished_at),
                current_attempt_id=task.current_attempt_id,
                worker_address=task.worker_address,
                peer_worker_label=task.worker_id or task.worker_address,
            )
            writes.mirror_federated_attempts(cur, task_id=local_task_id, attempts=task.attempts)
            # The parent derives the federated task's counts from these mirrored
            # attempts, so drop the job's cached totals via the cursor's memo.
            cur.caches[AttemptCountsProjection].invalidate_for_tasks(cur, [local_task_id])

    def _set_replace(self, cur: Tx, peer_id: str, deltas) -> None:
        """Full-resync set-replacement: drop any local handle for ``peer_id``
        absent from the peer's active set, reclaiming a job the parent never saw
        tombstoned."""
        active = {delta.job_id for delta in deltas if not delta.tombstone}
        for local_job_id in reads.federated_handles_for_peer(cur, peer_id):
            if local_job_id.to_wire() not in active:
                writes.delete_job(cur, local_job_id)
