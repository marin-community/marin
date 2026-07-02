# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`BackendWorkerStore` interface and its controller-DB implementation.

A worker-daemon backend uses a store to read its workers, build the snapshots it
schedules and reconciles from, resolve a worker's address, and reap dead workers.
:class:`DbBackendWorkerStore` implements the interface against the controller database.
"""

import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from rigging.timing import Timestamp

from iris.cluster.controller import reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.autoscaler.persistence import persist_autoscaler_state
from iris.cluster.controller.autoscaler.status import overlay_worker_usability
from iris.cluster.controller.backend import AutoscaleRequest, AutoscaleResult, BackendSchedulingInputs
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.ops.worker import fail as fail_workers
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
from iris.rpc import controller_pb2, job_pb2, vm_pb2

# Failure reason stamped on a healthy slice sibling reaped alongside a dead worker.
_SLICE_SIBLING_TEARDOWN_REASON = "unhealthy worker failed, slice terminated"


def _find_prunable_worker(health: WorkerHealthTracker, before_ms: int) -> WorkerId | None:
    """Return one tracker-known worker that is DEAD with a heartbeat older than ``before_ms``.

    Every persisted ``workers`` row has a tracker entry by construction (seeded at
    boot/restore, registered on commit of ``upsert``, removed on commit of
    ``remove``), so scanning the tracker is sufficient.
    """
    for worker_id, liveness in health.all().items():
        if liveness.usability is WorkerUsability.DEAD and liveness.last_heartbeat_ms < before_ms:
            return worker_id
    return None


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

    def overlaid_autoscaler_status(self, autoscaler_status: vm_pb2.AutoscalerStatus) -> vm_pb2.AutoscalerStatus:
        """Tag ``autoscaler_status``'s groups with the backend id and overlay each VM
        with the usability/running-task verdict from this backend's own state."""
        ...

    def worker_fleet_detail(
        self, autoscaler_status: vm_pb2.AutoscalerStatus
    ) -> controller_pb2.Controller.WorkerFleetDetail:
        """Author the backend's ``worker`` status variant: health counts plus the
        overlaid ``autoscaler_status``."""
        ...

    def worker_address(self, worker_id: WorkerId) -> str | None:
        """The worker's address, or ``None`` if it has none."""
        ...

    def reap_workers(self, worker_ids: list[WorkerId], *, reason: str) -> list[WorkerId]:
        """Fail ``worker_ids``, terminate their slices and healthy siblings, and forget
        them. Returns every worker removed (the failed workers plus reaped siblings)."""
        ...

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        """Delete this backend's DEAD workers whose last heartbeat predates ``cutoff_ms``.

        Removes one worker (and its attributes) per transaction, sleeping ``pause``
        between deletes so heartbeat and scheduling traffic interleave. Returns the
        number of workers removed; stops early once ``stop_event`` is set."""
        ...


@dataclass(frozen=True)
class DbBackendWorkerStore:
    """:class:`BackendWorkerStore` backed by the controller database.

    Built per backend with the controller DB plus the backend's own liveness tracker
    and ``autoscale`` callback, which ``reap_workers`` uses to fail workers and
    terminate their slices.
    """

    backend_id: str
    db: ControllerDB
    owns_scale_group: Callable[[str], bool]
    health: WorkerHealthTracker
    worker_attrs: WorkerAttrsProjection
    endpoints: EndpointsProjection
    run_template_cache: RunTemplateCache
    defaults: UserBudgetDefaults
    autoscale: Callable[[AutoscaleRequest], AutoscaleResult]

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

    def overlaid_autoscaler_status(self, autoscaler_status: vm_pb2.AutoscalerStatus) -> vm_pb2.AutoscalerStatus:
        """Author this backend's autoscaler status from the state it owns.

        Stamps each group with this backend's id, then overlays every VM named in
        the status with the usability / running-task-count / capacity verdict
        derived from this backend's own liveness tracker (which holds exactly its
        persisted workers) plus the running-task rows for those VMs.
        ``autoscaler_status`` is mutated in place and returned.
        """
        usability_by_id = {str(wid): live.usability for wid, live in self.health.all().items()}
        for group in autoscaler_status.groups:
            group.backend_id = self.backend_id
        vm_ids = {
            WorkerId(vm.vm_id)
            for group in autoscaler_status.groups
            for slice_info in group.slices
            for vm in slice_info.vms
            if vm.vm_id
        }
        if vm_ids:
            with self.db.control_read_snapshot() as snap:
                running = reads.running_tasks_by_worker(snap, vm_ids)
        else:
            running = {}
        overlay_worker_usability(autoscaler_status, usability_by_id, running)
        return autoscaler_status

    def worker_fleet_detail(
        self, autoscaler_status: vm_pb2.AutoscalerStatus
    ) -> controller_pb2.Controller.WorkerFleetDetail:
        """Author this backend's ``worker`` status variant from the state it owns:
        the health counts from its own liveness tracker, wrapping the overlaid
        ``autoscaler_status``.
        """
        liveness = self.health.all()
        return controller_pb2.Controller.WorkerFleetDetail(
            autoscaler=self.overlaid_autoscaler_status(autoscaler_status),
            total_worker_count=len(liveness),
            healthy_worker_count=sum(1 for live in liveness.values() if live.healthy),
        )

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

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        """Delete this backend's DEAD workers whose last heartbeat predates ``cutoff_ms``.

        Removes one worker (and its attributes) per transaction, sleeping ``pause``
        between deletes so heartbeat and scheduling traffic interleave. Returns the
        number of workers removed; stops early once ``stop_event`` is set.
        """
        deleted = 0
        while stop_event is None or not stop_event.is_set():
            worker_id = _find_prunable_worker(self.health, cutoff_ms)
            if worker_id is None:
                break
            with self.db.transaction() as cur:
                writes.remove_worker(cur, worker_id, health=self.health, worker_attrs=self.worker_attrs)
            log_event("worker_pruned", str(worker_id))
            deleted += 1
            time.sleep(pause)
        return deleted

    def _owned_worker_ids(self, snap: Tx) -> set[WorkerId]:
        """The workers this backend owns, by scale group, in the read ``snap``."""
        return reads.owned_worker_ids(snap, self.owns_scale_group)

    def _run_templates(self, snap: Tx, reconcile_rows: Sequence[ReconcileRow]) -> dict[JobName, job_pb2.RunTaskRequest]:
        """Per-job ``RunTaskRequest`` templates for the ASSIGNED rows, dropping uncached jobs."""
        templates: dict[JobName, job_pb2.RunTaskRequest | None] = {}
        for row in reconcile_rows:
            if row.task_state != job_pb2.TASK_STATE_ASSIGNED:
                continue
            if row.job_id not in templates:
                templates[row.job_id] = dispatch.run_request_template(self.run_template_cache, snap, row.job_id)
        return {job_id: spec for job_id, spec in templates.items() if spec is not None}
