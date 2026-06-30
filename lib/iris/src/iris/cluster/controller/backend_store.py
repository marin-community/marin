# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`BackendWorkerStore` interface and its controller-DB implementation.

A worker-daemon backend uses a store to read its workers, build the snapshots it
schedules and reconciles from, resolve a worker's address, and reap dead workers.
:class:`DbBackendWorkerStore` implements the interface against the controller database.
"""

import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rigging.timing import Timestamp

from iris.cluster.controller import reads
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.autoscaler.persistence import persist_autoscaler_state
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendSchedulingInputs,
    RegisterOutcome,
    WorkerRegistration,
)
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.ops.worker import fail as fail_workers
from iris.cluster.controller.ops.worker import register as register_worker_row
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.reads import ControlSnapshot, ReconcileRow
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.loader import TransitionReader
from iris.cluster.controller.reconcile.snapshot import TransitionSnapshot
from iris.cluster.controller.run_template import RunTemplateCache
from iris.cluster.controller.scheduling.policy import build_scheduling_context
from iris.cluster.controller.transition_reader import load_transition_snapshot
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

# Failure reason stamped on a healthy slice sibling reaped alongside a dead worker.
_SLICE_SIBLING_TEARDOWN_REASON = "unhealthy worker failed, slice terminated"

# Failure reason stamped on a stale worker evicted because a new worker registered
# at its address (a recycled internal IP).
_RECYCLED_IP_EVICTION_REASON = "address reused by newly-registered worker (recycled IP)"


class BackendWorkerStore(TransitionReader, Protocol):
    """The worker-state operations a worker-daemon backend depends on."""

    def owned_worker_ids(self) -> set[WorkerId]:
        """The worker IDs this backend owns, by scale group."""
        ...

    def scheduling_inputs(self) -> BackendSchedulingInputs:
        """This backend's live workers, their building counts, and preemptible running attempts."""
        ...

    def reconcile_snapshot(self) -> ControlSnapshot:
        """This backend's worker addresses, reconcile rows, and per-job run-task templates."""
        ...

    def worker_status(self) -> WorkerStatusMap:
        """Each owned worker's idle/running status."""
        ...

    def worker_address(self, worker_id: WorkerId) -> str | None:
        """The worker's address, or ``None`` if it has none."""
        ...

    def register_worker(self, registration: WorkerRegistration) -> RegisterOutcome:
        """Persist a registering worker and queue any recycled-address eviction.

        Runs on the Register RPC thread. Writes the worker row, seeds its liveness,
        and queues any stale prior owner of the same address (a recycled IP) for
        :meth:`drain_pending_evictions` to reap on the next control tick."""
        ...

    def drain_pending_evictions(self) -> list[WorkerId]:
        """Reap the recycled-address workers queued by :meth:`register_worker`.

        Runs on the control thread once per tick. Returns every worker removed."""
        ...

    def reap_workers(self, worker_ids: list[WorkerId], *, reason: str) -> list[WorkerId]:
        """Fail ``worker_ids``, terminate their slices and healthy siblings, and forget
        them. Returns every worker removed (the failed workers plus reaped siblings)."""
        ...


@dataclass(frozen=True)
class DbBackendWorkerStore:
    """:class:`BackendWorkerStore` backed by the controller database.

    Built per backend with the controller DB plus the backend's own liveness tracker
    and ``autoscale`` callback, which ``reap_workers`` uses to fail workers and
    terminate their slices.
    """

    db: ControllerDB
    owns_scale_group: Callable[[str], bool]
    health: WorkerHealthTracker
    worker_attrs: WorkerAttrsProjection
    endpoints: EndpointsProjection
    run_template_cache: RunTemplateCache
    defaults: UserBudgetDefaults
    autoscale: Callable[[AutoscaleRequest], AutoscaleResult]
    # Stale recycled-address owners queued by register_worker (RPC thread) for
    # drain_pending_evictions (control thread); guarded by its lock.
    _pending_evictions: set[WorkerId] = field(default_factory=set)
    _pending_evictions_lock: threading.Lock = field(default_factory=threading.Lock)

    def register_worker(self, registration: WorkerRegistration) -> RegisterOutcome:
        """Persist a registering worker and queue any recycled-address eviction.

        Writes the worker row and seeds its liveness, then detects a stale prior
        owner of the same address (a recycled internal IP) and queues it for
        :meth:`drain_pending_evictions`. Returns the queued stale owners, if any."""
        now = Timestamp.now()
        with self.db.transaction() as cur:
            register_worker_row(
                cur,
                worker_id=registration.worker_id,
                address=registration.address,
                metadata=registration.metadata,
                ts=now,
                health=self.health,
                worker_attrs=self.worker_attrs,
                slice_id=registration.slice_id,
                scale_group=registration.scale_group,
            )
        with self.db.read_snapshot() as snap:
            stale = reads.worker_ids_at_address(snap, registration.address, exclude=registration.worker_id)
        if stale:
            with self._pending_evictions_lock:
                self._pending_evictions.update(stale)
        return RegisterOutcome(worker_id=registration.worker_id, queued_eviction=stale)

    def drain_pending_evictions(self) -> list[WorkerId]:
        """Reap the recycled-address workers queued by :meth:`register_worker`."""
        with self._pending_evictions_lock:
            if not self._pending_evictions:
                return []
            drained = sorted(self._pending_evictions)
            self._pending_evictions.clear()
        return self.reap_workers(drained, reason=_RECYCLED_IP_EVICTION_REASON)

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

    def owned_worker_ids(self) -> set[WorkerId]:
        with self.db.control_read_snapshot() as snap:
            return self._owned_worker_ids(snap)

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

    def worker_address(self, worker_id: WorkerId) -> str | None:
        with self.db.control_read_snapshot() as snap:
            return reads.bulk_get_worker_addresses(snap, [worker_id]).get(worker_id)

    def reap_workers(self, worker_ids: list[WorkerId], *, reason: str) -> list[WorkerId]:
        """Fail ``worker_ids``, terminate their slices and healthy siblings, and forget
        them from the liveness tracker. Returns every worker removed."""
        if not worker_ids:
            return []
        for wid in worker_ids:
            log_event("worker_failing", str(wid), trigger=reason)
        failure_result = fail_workers(
            self.db,
            worker_ids=[str(wid) for wid in worker_ids],
            reason=reason,
            health=self.health,
            endpoints=self.endpoints,
            worker_attrs=self.worker_attrs,
        )
        removed_ids = [wid for wid, _ in failure_result.removed_workers]
        if not removed_ids:
            # A concurrent reaper already failed every candidate (or they had no
            # address). Nothing was removed, so skip autoscale entirely: calling it
            # with no dead workers would run a full provisioning cycle on the control
            # thread (probe_health + update_slice_activity) racing the autoscaler thread.
            return []

        removed_set = set(removed_ids)
        auto = self.autoscale(AutoscaleRequest(dead_workers=removed_ids))
        if auto.autoscaler_state is not None:
            with self.db.transaction() as cur:
                persist_autoscaler_state(cur, auto.autoscaler_state)
        siblings = [wid for wid in auto.removed_workers if wid not in removed_set]
        if siblings:
            for wid in siblings:
                log_event("worker_failing", str(wid), trigger=_SLICE_SIBLING_TEARDOWN_REASON)
            fail_workers(
                self.db,
                worker_ids=[str(wid) for wid in siblings],
                reason=_SLICE_SIBLING_TEARDOWN_REASON,
                health=self.health,
                endpoints=self.endpoints,
                worker_attrs=self.worker_attrs,
            )
        self.health.forget_many(removed_set | set(siblings))
        return removed_ids + siblings

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
