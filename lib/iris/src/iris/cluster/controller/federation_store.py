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
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.run_template import RunTemplateCache
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
        *,
        run_template_cache: RunTemplateCache,
    ):
        self._db = db
        self._run_template_cache = run_template_cache

    # -- handoff -------------------------------------------------------------

    def admit_and_persist_handoff(self, spec: HandoffSpec) -> HandoffAdmission:
        now = Timestamp.now()
        with self._db.transaction() as cur:
            if reads.get_job_state(cur, spec.parent_job_id) is not None:
                # A handle already exists — a retried/idempotent resubmit.
                return HandoffAdmission.ALREADY_EXISTS

            ops.job.insert_job_and_config(
                cur,
                job_id=spec.parent_job_id,
                request=spec.request,
                ts=now,
                run_template_cache=self._run_template_cache,
                child_cluster=spec.peer_id,
            )
            writes.insert_federated_handle(
                cur,
                job_id=spec.parent_job_id,
                peer_id=spec.peer_id,
                remote_job_id=spec.remote_job_id,
                owner_principal=spec.owner_principal,
                handoff_state=int(HandoffState.PENDING_HANDOFF),
                now_ms=now.epoch_ms(),
            )
        return HandoffAdmission.ADMITTED

    def mark_handed_off(self, parent_job_id: JobName, *, now_ms: int) -> None:
        with self._db.transaction() as cur:
            writes.set_handoff_state(cur, parent_job_id, int(HandoffState.HANDED_OFF), now_ms=now_ms)

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
                        parent_job_id=handle.job_id,
                        remote_job_id=handle.remote_job_id,
                        peer_id=handle.peer_id,
                        owner_principal=handle.owner_principal,
                        request=reconstruct_launch_job_request(job),
                    )
                )
        return pending

    # -- cancel --------------------------------------------------------------

    def bump_cancel_intent(self, parent_job_id: JobName) -> CancelTarget | None:
        """Bump a SENT handle's cancel intent and return the peer/remote-id to cancel.

        Returns ``None`` when ``parent_job_id`` is not a SENT handle. A handle the
        peer never received (still ``PENDING_HANDOFF``) is terminated locally here —
        the re-drive now skips it, so no sync will ever mirror it terminal. A
        delivered handle keeps its synced state: the routed cancel drives it terminal
        on the peer and the next sync reflects it.
        """
        with self._db.transaction() as cur:
            handle = reads.federated_handle(cur, parent_job_id)
            if handle is None:
                return None
            writes.bump_cancel_intent(cur, parent_job_id)
            if handle.handoff_state == int(HandoffState.PENDING_HANDOFF):
                writes.mark_federated_job_killed(
                    cur, parent_job_id, now_ms=Timestamp.now().epoch_ms(), error="Cancelled before handoff"
                )
            return CancelTarget(parent_job_id=parent_job_id, peer_id=handle.peer_id, remote_job_id=handle.remote_job_id)

    def pending_cancels(self) -> list[CancelTarget]:
        with self._db.read_snapshot() as tx:
            return [
                CancelTarget(parent_job_id=h.job_id, peer_id=h.peer_id, remote_job_id=h.remote_job_id)
                for h in reads.pending_cancel_handles(tx)
            ]

    def mark_cancel_satisfied(self, parent_job_id: JobName, *, now_ms: int) -> None:
        with self._db.transaction() as cur:
            writes.mark_federated_job_killed(
                cur, parent_job_id, now_ms=now_ms, error="Cancelled (peer reported the job gone)"
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
        now_ms = Timestamp.now().epoch_ms()
        with self._db.transaction() as cur:
            for delta in deltas:
                local_job_id = reads.federated_job_for_remote_id(cur, peer_id, delta.remote_job_id)
                if local_job_id is None:
                    continue
                if delta.tombstone:
                    writes.delete_job(cur, local_job_id)
                    continue
                self._mirror_delta(cur, peer_id, local_job_id, delta)

            if cursor_stale:
                self._set_replace(cur, peer_id, deltas)

            writes.upsert_sync_cursor(cur, peer_id, next_cursor, now_ms=now_ms)

    def _mirror_delta(self, cur, peer_id: str, local_job_id: JobName, delta) -> None:
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
            writes.mirror_federated_task(
                cur,
                task_id=local_job_id.task(index),
                job_id=local_job_id,
                task_index=index,
                peer_id=peer_id,
                state=task.state,
                error=task.error or None,
                exit_code=task.exit_code or None,
                started_at_ms=_proto_ms(task.HasField("started_at"), task.started_at),
                finished_at_ms=_proto_ms(task.HasField("finished_at"), task.finished_at),
                failure_count=task.failure_count,
                current_attempt_id=task.current_attempt_id,
                worker_address=task.worker_address,
                peer_worker_label=task.worker_id or task.worker_address,
            )

    def _set_replace(self, cur, peer_id: str, deltas) -> None:
        """Full-resync set-replacement: drop any local handle for ``peer_id``
        absent from the peer's active set, reclaiming a job the parent never saw
        tombstoned."""
        active = {delta.remote_job_id for delta in deltas if not delta.tombstone}
        for remote_id, local_job_id in reads.federated_handles_for_peer(cur, peer_id).items():
            if remote_id not in active:
                writes.delete_job(cur, local_job_id)
