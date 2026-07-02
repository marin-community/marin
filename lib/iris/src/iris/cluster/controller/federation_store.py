# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The controller-side :class:`FederationStore` implementation.

Backs the federation manager's durable operations against the controller's own
tables: it admits and persists handoff handles (with the federated budget check),
re-drives pending handoffs, mirrors a peer's synced state into the local
``jobs``/``tasks`` rows, and routes cancel intent. Keeping this on the controller
side lets the federation module depend only on the ``FederationStore`` protocol.
"""

import logging

from rigging.timing import Timestamp

from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.budget import compute_user_spend, resource_value
from iris.cluster.controller.codec import (
    device_counts_from_json,
    proto_to_json,
    reconstruct_launch_job_request,
)
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.run_template import RunTemplateCache
from iris.cluster.federation.store import (
    CancelTarget,
    HandoffAdmission,
    HandoffOutcome,
    HandoffSpec,
    HandoffState,
    PendingHandoff,
)
from iris.cluster.types import JobName, UserBudgetDefaults
from iris.rpc import controller_pb2
from iris.time_proto import timestamp_from_proto

logger = logging.getLogger(__name__)


def _proto_ms(has: bool, ts) -> int | None:
    """Epoch ms of a proto ``Timestamp`` field, or ``None`` when unset."""
    return timestamp_from_proto(ts).epoch_ms() if has else None


def _reservation(request: controller_pb2.Controller.LaunchJobRequest) -> int:
    """The budget charge for a federated job: its shape's value times replicas.

    Uses the same ``resource_value`` scalar as local spend so the admission check
    compares like with like.
    """
    res = request.resources if request.HasField("resources") else None
    if res is None:
        return 0
    device_json = proto_to_json(res.device) if res.HasField("device") else None
    counts = device_counts_from_json(device_json)
    replicas = max(int(request.replicas), 1)
    return resource_value(int(res.cpu_millicores), int(res.memory_bytes), counts.gpu + counts.tpu) * replicas


class ControllerFederationStore:
    """A :class:`~iris.cluster.federation.store.FederationStore` over ``ControllerDB``."""

    def __init__(
        self,
        db: ControllerDB,
        *,
        run_template_cache: RunTemplateCache,
        user_budget_defaults: UserBudgetDefaults,
    ):
        self._db = db
        self._run_template_cache = run_template_cache
        self._user_budget_defaults = user_budget_defaults

    # -- handoff -------------------------------------------------------------

    def admit_and_persist_handoff(self, spec: HandoffSpec) -> HandoffOutcome:
        now = Timestamp.now()
        with self._db.transaction() as cur:
            if reads.get_job_state(cur, spec.parent_job_id) is not None:
                # A handle already exists — a retried/idempotent resubmit.
                return HandoffOutcome(admission=HandoffAdmission.ALREADY_EXISTS)

            reservation = _reservation(spec.request)
            reject = self._admission_rejection(cur, spec.parent_job_id.user, reservation)
            if reject is not None:
                return HandoffOutcome(admission=HandoffAdmission.REJECTED, reject_reason=reject)

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
                spend_snapshot=reservation,
                now_ms=now.epoch_ms(),
            )
        return HandoffOutcome(admission=HandoffAdmission.ADMITTED)

    def _admission_rejection(self, cur, user_id: str, reservation: int) -> str | None:
        """The federated budget check: local + cached-federated spend + this
        job's reservation against the user cap. Returns a rejection message when
        over cap, else ``None``. A cap of 0 means unlimited."""
        budget = reads.get_user_budget(cur, user_id)
        limit = budget.budget_limit if budget is not None else self._user_budget_defaults.budget_limit
        if limit <= 0:
            return None
        local_spend = compute_user_spend(cur).get(user_id, 0)
        federated_spend = reads.federated_spend_for_user(cur, user_id)
        projected = local_spend + federated_spend + reservation
        if projected > limit:
            return (
                f"User {user_id} would exceed their budget with this federated job "
                f"(local {local_spend} + federated {federated_spend} + reservation {reservation} "
                f"= {projected} > limit {limit})"
            )
        return None

    def mark_handed_off(self, parent_job_id: JobName, *, now_ms: int) -> None:
        with self._db.transaction() as cur:
            writes.set_handoff_state(cur, parent_job_id, int(HandoffState.HANDED_OFF), now_ms=now_ms)

    def mark_handoff_failed(self, parent_job_id: JobName, error: str) -> None:
        with self._db.transaction() as cur:
            writes.set_handoff_state(
                cur, parent_job_id, int(HandoffState.HANDOFF_FAILED), now_ms=Timestamp.now().epoch_ms(), error=error
            )

    def pending_handoffs(self) -> list[PendingHandoff]:
        with self._db.read_snapshot() as tx:
            handles = reads.pending_handoff_handles(tx)
            pending = []
            for handle in handles:
                job = reads.get_job_detail(tx, handle.job_id)
                if job is None:
                    continue
                pending.append(
                    PendingHandoff(
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
        with self._db.transaction() as cur:
            handle = reads.federated_handle(cur, parent_job_id)
            if handle is None:
                return None
            writes.bump_cancel_intent(cur, parent_job_id)
            # A handle the peer never received (the re-drive now skips it) will
            # never be mirrored terminal by a sync, so terminate its local job now.
            # A delivered job keeps its synced state — the routed cancel drives it
            # terminal on the peer and the next sync reflects it.
            if handle.handoff_state == int(HandoffState.PENDING_HANDOFF):
                writes.mark_federated_job_killed(
                    cur, parent_job_id, now_ms=Timestamp.now().epoch_ms(), error="Cancelled before handoff"
                )
            return CancelTarget(peer_id=handle.peer_id, remote_job_id=handle.remote_job_id)

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
