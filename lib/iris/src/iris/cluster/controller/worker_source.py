# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DbWorkerSource: a worker-daemon backend's live-worker read surface.

Backs a backend's :class:`~iris.cluster.controller.backend.WorkerSource` with a
scale-group-scoped read of the controller database, so the backend reads its own
workers, placement and worker-status without the controller partitioning a global
snapshot. The ``owns_scale_group`` predicate is the backend's worker-ownership
test (the default backend also claims workers whose scale group is unmapped,
matching the controller's scale-group→backend resolution). When live worker and
placement state move into the backend's own store, only this class is replaced —
the backend contract does not change.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller import reads
from iris.cluster.controller.backend import BackendSchedulingInputs
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.reads import ControlSnapshot, ReconcileRow, WorkerAttrsSource
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile.snapshot import TransitionSnapshot
from iris.cluster.controller.run_template import RunTemplateCache
from iris.cluster.controller.scheduling.policy import build_scheduling_context
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import (
    AttemptUid,
    JobName,
    UserBudgetDefaults,
    WorkerId,
    WorkerStatus,
    WorkerStatusMap,
    WorkerUsability,
)
from iris.rpc import job_pb2


def load_transition_snapshot(
    db: ControllerDB,
    *,
    now: Timestamp,
    seed_worker_ids: Iterable[WorkerId] = (),
    observation_uids: Iterable[AttemptUid] = (),
    seed_task_ids: Iterable[JobName] = (),
    extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
) -> TransitionSnapshot:
    """Open a control read snapshot and close it over the seeded entities.

    The DB-backed read surface a backend authors its task projection through: it
    runs the same loader the commit-side glue runs, but against a read-only
    snapshot rather than the tick's write transaction, so the backend never
    touches the controller database directly.
    """
    with db.control_read_snapshot() as snap:
        return load_closed_snapshot(
            snap,
            now=now,
            seed_worker_ids=seed_worker_ids,
            observation_uids=observation_uids,
            seed_task_ids=seed_task_ids,
            extra_attempt_keys=extra_attempt_keys,
        )


@dataclass(frozen=True)
class DbTransitionReader:
    """A controller-DB-backed :class:`~...reconcile.loader.TransitionReader`.

    Gives a placement-owning backend (one with no :class:`WorkerSource`) a read
    snapshot to author its dispatch effects from, without handing it the DB.
    """

    db: ControllerDB

    def transition_snapshot(
        self,
        *,
        now: Timestamp,
        seed_worker_ids: Iterable[WorkerId] = (),
        observation_uids: Iterable[AttemptUid] = (),
        seed_task_ids: Iterable[JobName] = (),
        extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
    ) -> TransitionSnapshot:
        return load_transition_snapshot(
            self.db,
            now=now,
            seed_worker_ids=seed_worker_ids,
            observation_uids=observation_uids,
            seed_task_ids=seed_task_ids,
            extra_attempt_keys=extra_attempt_keys,
        )


@dataclass(frozen=True)
class DbWorkerSource:
    """A worker-daemon backend's worker source backed by the controller DB.

    Each method opens its own scoped read; the backend never receives a global
    worker snapshot from the controller.
    """

    db: ControllerDB
    owns_scale_group: Callable[[str], bool]
    health: WorkerHealthTracker
    worker_attrs: WorkerAttrsSource
    run_template_cache: RunTemplateCache
    defaults: UserBudgetDefaults

    def transition_snapshot(
        self,
        *,
        now: Timestamp,
        seed_worker_ids: Iterable[WorkerId] = (),
        observation_uids: Iterable[AttemptUid] = (),
        seed_task_ids: Iterable[JobName] = (),
        extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
    ) -> TransitionSnapshot:
        return load_transition_snapshot(
            self.db,
            now=now,
            seed_worker_ids=seed_worker_ids,
            observation_uids=observation_uids,
            seed_task_ids=seed_task_ids,
            extra_attempt_keys=extra_attempt_keys,
        )

    def scheduling_inputs(self) -> BackendSchedulingInputs:
        with self.db.control_read_snapshot() as snap:
            ctx = build_scheduling_context(snap, self.health, self.worker_attrs, self.defaults)
            owned = self._owned_worker_ids(snap)
        workers = [w for w in ctx.workers if w.worker_id in owned]
        building_counts = {wid: count for wid, count in ctx.building_counts.items() if wid in owned}
        running = [r for r in ctx.running_for_preemption if r.worker_id in owned]
        return BackendSchedulingInputs(
            workers=workers,
            building_counts=building_counts,
            running_for_preemption=running,
            max_building_tasks=ctx.max_building_tasks,
            max_assignments_per_worker=ctx.max_assignments_per_worker,
        )

    def reconcile_snapshot(self) -> ControlSnapshot:
        with self.db.control_read_snapshot() as snap:
            control = reads.load_control_snapshot(snap, self.health, scan_timeouts=False)
            owned = self._owned_worker_ids(snap)
            worker_addresses = {wid: addr for wid, addr in control.worker_addresses.items() if wid in owned}
            reconcile_rows = [r for r in control.reconcile_rows if r.worker_id in owned]
            job_specs = self._run_templates(snap, reconcile_rows)
        return ControlSnapshot(
            worker_addresses=worker_addresses,
            reconcile_rows=reconcile_rows,
            timeout_rows=[],
            job_specs=job_specs,
        )

    def worker_status(self) -> WorkerStatusMap:
        usability = {wid: live.usability for wid, live in self.health.all().items()}
        worker_ids = {wid for wid, use in usability.items() if use is not WorkerUsability.DEAD}
        with self.db.control_read_snapshot() as snap:
            running_by_worker = reads.running_tasks_by_worker(snap, worker_ids)
            wid_to_scale_group = reads.worker_scale_groups(snap)
        result: WorkerStatusMap = {}
        for wid in worker_ids:
            if not self.owns_scale_group(wid_to_scale_group.get(wid, "")):
                continue
            result[wid] = WorkerStatus(
                worker_id=wid,
                running_task_ids=frozenset(tid.to_wire() for tid in running_by_worker.get(wid, set())),
                usability=usability[wid],
            )
        return result

    def _owned_worker_ids(self, snap: Tx) -> set[WorkerId]:
        """The workers this backend owns, by scale group, in the read ``snap``."""
        groups = reads.worker_scale_groups(snap)
        return {wid for wid, scale_group in groups.items() if self.owns_scale_group(scale_group)}

    def _run_templates(self, snap: Tx, reconcile_rows: Sequence[ReconcileRow]) -> dict[JobName, job_pb2.RunTaskRequest]:
        """Per-job ``RunTaskRequest`` templates for the ASSIGNED rows, dropping uncached jobs."""
        templates: dict[JobName, job_pb2.RunTaskRequest | None] = {}
        for row in reconcile_rows:
            if row.task_state != job_pb2.TASK_STATE_ASSIGNED:
                continue
            if row.job_id not in templates:
                templates[row.job_id] = dispatch.run_request_template(self.run_template_cache, snap, row.job_id)
        return {job_id: spec for job_id, spec in templates.items() if spec is not None}
