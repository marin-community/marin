# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler manages scaling across scale groups.

The autoscaler coordinates scaling decisions across multiple scale groups,
delegating slice ownership to ScalingGroup.

Key design principles:
- Autoscaler does NOT track slices directly - that's ScalingGroup's job
- Scale-up decisions come from Autoscaler, scale-down is delegated to ScalingGroup
- ScalingGroup owns per-slice idle tracking and decides which slices to scale down
- Bootstrap is handled internally by each provider implementation, not by the autoscaler

The run_once() flow splits into two phases:
- refresh(): state-read phase — scale down idle slices from tracked state
- update(): CPU phase — evaluate demand and execute scale-up decisions
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence

from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import Constraint
from iris.cluster.controller.autoscaler.models import (
    DemandEntry,
    ScalingAction,
    ScalingDecision,
)
from iris.cluster.controller.autoscaler.operations import (
    restart_worker as restart_worker_operation,
)
from iris.cluster.controller.autoscaler.operations import (
    terminate_slices_for_workers as terminate_slices_for_workers_operation,
)
from iris.cluster.controller.autoscaler.planning import ScalePlan, build_scale_plan
from iris.cluster.controller.autoscaler.recovery import (
    load_autoscaler_checkpoint,
    restore_autoscaler_state,
)
from iris.cluster.controller.autoscaler.routing import job_feasibility, route_demand
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup, build_worker_config_for_group
from iris.cluster.controller.autoscaler.status import PendingHint, build_job_pending_hints, routing_decision_to_proto
from iris.cluster.controller.autoscaler.worker_registry import TrackedWorker, WorkerRegistry
from iris.cluster.controller.db import ControllerDB
from iris.cluster.providers.protocols import WorkerInfraProvider
from iris.cluster.providers.types import (
    CloudSliceState,
    QuotaExhaustedError,
    RemoteWorkerHandle,
    SliceHandle,
)
from iris.cluster.types import WorkerStatusMap
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import config_pb2, vm_pb2
from iris.time_proto import duration_from_proto, timestamp_to_proto

logger = logging.getLogger(__name__)


# Slices that die within this time of creation trigger backoff (preemption detection)
SHORT_LIVED_SLICE_THRESHOLD = Duration.from_minutes(5)

# After this long in UNKNOWN state, treat the slice as FAILED (quota timeout is 5 min, so this is conservative)
DEFAULT_UNRESOLVABLE_TIMEOUT = Duration.from_minutes(15)


class Autoscaler:
    """Manages scaling across scale groups.

    The autoscaler:
    - Receives demand from a DemandSource
    - Evaluates scaling decisions based on demand vs capacity
    - Executes decisions by calling ScalingGroup.scale_up/scale_down

    It does NOT:
    - Track VM groups (ScalingGroup does that)
    - Know about controller internals (DemandSource abstracts that)
    """

    def __init__(
        self,
        scale_groups: dict[str, ScalingGroup],
        evaluation_interval: Duration,
        platform: WorkerInfraProvider,
        threads: ThreadContainer | None = None,
        base_worker_config: config_pb2.WorkerConfig | None = None,
        db: ControllerDB | None = None,
        unresolvable_timeout: Duration = DEFAULT_UNRESOLVABLE_TIMEOUT,
    ):
        """Create autoscaler with explicit parameters.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            evaluation_interval: How often to evaluate scaling decisions
            platform: WorkerInfraProvider instance for shutdown lifecycle
            threads: Optional thread container for testing
            base_worker_config: Base worker config merged with per-group overrides
                and passed to platform.create_slice(). None disables bootstrap (test/local mode).
            db: Optional DB handle for write-through persistence of tracked workers.
            unresolvable_timeout: How long a slice can remain UNKNOWN before being treated as FAILED.
        """
        self._groups = scale_groups
        self._platform = platform
        self._db = db
        self.evaluation_interval = evaluation_interval
        self._base_worker_config = base_worker_config
        self._unresolvable_timeout = unresolvable_timeout

        # Centralized per-worker state indexed by worker_id
        self._worker_registry = WorkerRegistry()

        # Bounded log of recent autoscaler actions for dashboard/debugging
        self._action_log: deque[vm_pb2.AutoscalerAction] = deque(maxlen=100)

        # Most recent routing decision (for status API)
        self._last_scale_plan: ScalePlan | None = None
        self._last_evaluation: Timestamp = Timestamp.from_ms(0)

        # Derived views of _last_scale_plan, built lazily and invalidated by
        # evaluate(). Dashboard polls (GetJobStatus, ListJobs) hit these on
        # every pending job; building them per request was the bottleneck
        # described in #4844.
        self._last_routing_decision_proto: vm_pb2.RoutingDecision | None = None
        self._last_pending_hints: dict[str, PendingHint] | None = None

        # Thread management
        self._threads = threads if threads is not None else get_thread_container()

    @classmethod
    def from_config(
        cls,
        scale_groups: dict[str, ScalingGroup],
        config: config_pb2.AutoscalerConfig,
        platform: WorkerInfraProvider,
        threads: ThreadContainer | None = None,
        base_worker_config: config_pb2.WorkerConfig | None = None,
        db: ControllerDB | None = None,
    ) -> Autoscaler:
        """Create autoscaler from proto config.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            config: Autoscaler configuration proto (with defaults already applied)
            platform: WorkerInfraProvider instance for shutdown lifecycle
            threads: Optional thread container for testing
            base_worker_config: Base worker config merged with per-group overrides
            db: Optional DB handle for write-through persistence.

        Returns:
            Configured Autoscaler instance
        """
        return cls(
            scale_groups=scale_groups,
            evaluation_interval=duration_from_proto(config.evaluation_interval),
            platform=platform,
            threads=threads,
            base_worker_config=base_worker_config,
            db=db,
        )

    def _wait_for_inflight(self) -> None:
        """Wait for in-flight scale-ups to complete without terminating anything.

        Test-only: Waits for all scale-up threads to complete.
        """
        self._threads.wait()

    def shutdown(self) -> None:
        """Shutdown the autoscaler, terminate all VM groups, and clean up platform.

        Shutdown ordering:
        1. Stop all threads in the autoscaler's ThreadContainer. This signals
           stop_events for both in-flight scale-up threads AND worker lifecycle
           threads (via child containers), then joins with timeout.
        2. Terminate all VM groups — calls Worker.stop() for final cleanup
           of any workers that didn't exit in step 1.
        3. Shutdown platform — clears local tracking state.
        """
        # Stop all threads (scale-ups + workers) via ThreadContainer.
        # Using stop() rather than wait() because wait() doesn't signal
        # stop_events and would block forever on worker-lifecycle threads.
        self._threads.stop()

        # Step 2: Terminate VMs and cleanup (idempotent with step 1)
        for group in self._groups.values():
            group.terminate_all()

        # Step 3: Shutdown platform (cleanup remaining threads)
        self._platform.shutdown()

    def __enter__(self) -> Autoscaler:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def _log_action(
        self,
        action_type: str,
        scale_group: str,
        slice_id: str = "",
        reason: str = "",
        status: str = "completed",
    ) -> vm_pb2.AutoscalerAction:
        """Log an autoscaler action to the bounded action log.

        Args:
            action_type: Type of action (scale_up, scale_down, etc.)
            scale_group: Name of the scale group
            slice_id: ID of the slice (if applicable)
            reason: Human-readable reason for the action
            status: Action status ("pending", "completed", "failed")

        Returns:
            The action object. The caller may mutate this object to update
            status after execution (e.g., from "pending" to "completed").
            This works because the deque holds references to the proto objects.
        """

        action = vm_pb2.AutoscalerAction(
            timestamp=timestamp_to_proto(Timestamp.now()),
            action_type=action_type,
            scale_group=scale_group,
            slice_id=slice_id,
            reason=reason,
            status=status,
        )
        self._action_log.append(action)
        logger.info(
            "event=autoscaler action=%s entity=%s trigger=- group=%s status=%s reason=%s",
            action_type,
            slice_id or scale_group,
            scale_group,
            status,
            reason,
        )
        return action

    def evaluate(
        self,
        demand_entries: list[DemandEntry],
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """Compute scaling decisions based on demand.

        Routes demand to groups based on accelerator_type requirements and
        priority. Higher-priority groups (lower priority number) receive
        demand first; overflow routes to lower-priority groups.

        Args:
            demand_entries: List of demand entries with requirements and counts.
            timestamp: Optional timestamp for testing. If None, uses Timestamp.now().

        Returns:
            List of scaling decisions to execute.
        """
        ts = timestamp or Timestamp.now()

        routing_decision = route_demand(list(self._groups.values()), demand_entries, ts)
        scale_plan = build_scale_plan(self._groups, routing_decision, ts)
        self._last_scale_plan = scale_plan
        # Build cached views eagerly here so dashboard/service RPCs never pay
        # the conversion cost on the hot path (#4844).
        self._last_routing_decision_proto = routing_decision_to_proto(
            routing_decision,
            group_to_launch=scale_plan.launch_counts(),
        )
        self._last_pending_hints = build_job_pending_hints(self._last_routing_decision_proto)

        if routing_decision.unmet_entries:
            logger.debug(
                "Unmet demand: %d entries cannot be satisfied (visible in dashboard)",
                len(routing_decision.unmet_entries),
            )

        for plan in scale_plan.group_plans.values():
            group = self._groups[plan.group]
            group.update_demand(plan.required_slices)
            logger.debug(
                "Evaluating group %s: total=%d, ready=%d, pending=%d, required_slices=%d, buffer=%d, target=%d, max=%d",
                group.name,
                plan.counts.total,
                plan.counts.ready,
                plan.counts.pending,
                plan.required_slices,
                group.buffer_slices,
                plan.target_slices,
                group.max_slices,
            )
            if plan.scale_up_blocked:
                logger.debug("Scale group %s: scale up blocked", group.name)

        return scale_plan.decisions()

    def execute(
        self,
        decisions: list[ScalingDecision],
        timestamp: Timestamp,
    ) -> None:
        """Execute scale-up decisions.

        Args:
            decisions: List of scaling decisions to execute.
            timestamp: Current timestamp.
        """
        for decision in decisions:
            group = self._groups.get(decision.scale_group)
            if not group:
                logger.warning("Unknown scale group in decision: %s", decision.scale_group)
                continue

            if decision.action == ScalingAction.SCALE_UP:
                if not group.acquire_scale_up_token(timestamp):
                    logger.info("Rate-limited scale-up for %s: %s", decision.scale_group, decision.reason)
                    self._log_action(
                        "rate_limited",
                        decision.scale_group,
                        reason=decision.reason,
                    )
                    continue
                self._execute_scale_up(group, timestamp, reason=decision.reason)

    def _execute_scale_up(self, group: ScalingGroup, ts: Timestamp, reason: str = "") -> None:
        """Initiate async scale-up for a scale group.

        Increments the group's pending scale-up counter and spawns a background
        thread for the actual scale-up work. The counter is included in
        slice_count(), preventing double scale-up.
        """
        group.begin_scale_up(timestamp=ts)

        def _scale_up_wrapper(stop_event):
            self._do_scale_up(group, ts, reason)

        self._threads.spawn(
            target=_scale_up_wrapper,
            name=f"scale-up-{group.name}",
        )

    def _do_scale_up(self, group: ScalingGroup, ts: Timestamp, reason: str = "") -> bool:
        """Execute the actual blocking scale-up work.

        This runs in a background thread and should not be called directly.
        Use _execute_scale_up instead. Bootstrap is handled internally by the
        platform when cluster_config is provided.

        Returns:
            True if scale-up succeeded, False otherwise.
        """
        action = self._log_action("scale_up", group.name, reason=reason, status="pending")

        slice_obj = None

        try:
            logger.info("Scaling up %s: %s", group.name, reason)
            wc = self._per_group_worker_config(group)
            slice_obj = group.scale_up(worker_config=wc, timestamp=ts)
            group.complete_scale_up(slice_obj, ts)
            logger.info("Created slice %s for group %s", slice_obj.slice_id, group.name)
            action.slice_id = slice_obj.slice_id
            action.status = "completed"
            return True
        except QuotaExhaustedError as e:
            group.cancel_scale_up()
            group.record_quota_exceeded(str(e), ts)
            logger.warning("Quota exceeded for %s: %s", group.name, e)
            action.action_type = "quota_exceeded"
            action.status = "failed"
            action.reason = str(e)
            return False
        except Exception as e:
            group.cancel_scale_up()
            logger.exception("Failed to create slice for %s: %s", group.name, e)
            action.status = "failed"
            action.reason = f"{reason} - error: {e}"
            group.record_failure(ts)
            return False

    def _per_group_worker_config(self, group: ScalingGroup) -> config_pb2.WorkerConfig | None:
        """Build per-group WorkerConfig by merging base config with scale group overrides."""
        return build_worker_config_for_group(self._base_worker_config, group.config)

    def _register_slice_workers(
        self,
        workers: list[RemoteWorkerHandle],
        slice_id: str,
        scale_group: str,
    ) -> None:
        """Register all workers from a slice into the in-memory handle cache."""

        self._worker_registry.register_slice_workers(workers, slice_id, scale_group)

    def _unregister_slice_workers(self, slice_id: str, worker_ids: Sequence[str] | None = None) -> None:
        """Remove tracked workers belonging to a slice from the handle cache."""

        self._worker_registry.unregister_slice_workers(slice_id, worker_ids)

    def refresh(self, worker_status_map: WorkerStatusMap, timestamp: Timestamp | None = None) -> None:
        """State-read phase: scale down idle slices from currently tracked state."""
        timestamp = timestamp or Timestamp.now()

        for group in self._groups.values():
            for slice_id, handle in group.non_ready_slice_handles():
                try:
                    status = handle.describe()
                except Exception as e:
                    logger.warning("Failed to poll slice %s: %s", slice_id, e)
                    continue

                if status.state == CloudSliceState.READY:
                    worker_ids = [w.worker_id for w in status.workers]
                    group.mark_slice_ready(slice_id, worker_ids)
                    self._register_slice_workers(status.workers, slice_id, group.name)
                    self._log_action(
                        "slice_ready",
                        group.name,
                        slice_id,
                        reason=f"bootstrap completed ({len(worker_ids)} workers)",
                    )
                elif status.state == CloudSliceState.FAILED:
                    group.mark_slice_failed(slice_id, error_message=status.error_message)
                    group.scale_down(slice_id)
                    self._unregister_slice_workers(slice_id)
                    group.record_failure()
                    reason = status.error_message if status.error_message else "bootstrap failed"
                    self._log_action(
                        "slice_failed",
                        group.name,
                        slice_id,
                        reason=reason,
                        status="failed",
                    )
                elif status.state == CloudSliceState.UNKNOWN:
                    age = Duration.from_ms(timestamp.epoch_ms() - handle.created_at.epoch_ms())
                    if age >= self._unresolvable_timeout:
                        group.mark_slice_failed(slice_id, error_message="unresolvable after timeout")
                        group.scale_down(slice_id)
                        self._unregister_slice_workers(slice_id)
                        group.record_failure()
                        self._log_action(
                            "slice_failed",
                            group.name,
                            slice_id,
                            reason=f"TPU unresolvable for {age}",
                            status="failed",
                        )
                    else:
                        logger.debug(
                            "Slice %s UNKNOWN (age %s < timeout %s); will retry",
                            slice_id,
                            age,
                            self._unresolvable_timeout,
                        )

        for group in self._groups.values():
            target_capacity = min(group.current_demand + group.buffer_slices, group.max_slices)
            ready_before = group.ready_slice_count()
            scaled_down_handles = group.scale_down_if_idle(worker_status_map, target_capacity, timestamp)
            for handle in scaled_down_handles:
                self._unregister_slice_workers(handle.slice_id)
                self._log_action(
                    "scale_down",
                    group.name,
                    slice_id=handle.slice_id,
                    reason=f"idle slice (target={target_capacity}, ready={ready_before})",
                )

    def update(
        self,
        demand_entries: list[DemandEntry],
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """CPU phase: evaluate demand and execute scale-up decisions."""
        timestamp = timestamp or Timestamp.now()
        self._last_evaluation = timestamp

        decisions = self.evaluate(demand_entries, timestamp)
        if decisions:
            logger.info("Autoscaler decisions: %s", [(d.scale_group, d.action.value, d.reason) for d in decisions])
        self.execute(decisions, timestamp)
        return decisions

    def run_once(
        self,
        demand_entries: list[DemandEntry],
        worker_status_map: WorkerStatusMap,
        timestamp: Timestamp | None = None,
    ) -> list[ScalingDecision]:
        """Full cycle: refresh + update. Preserved for tests."""
        timestamp = timestamp or Timestamp.now()
        logger.debug("Autoscaler run_once: demand_entries=%s", demand_entries)
        self.refresh(worker_status_map, timestamp)
        return self.update(demand_entries, timestamp)

    def get_tracked_worker(self, worker_id: str) -> TrackedWorker | None:
        """Look up a tracked worker by ID."""
        return self._worker_registry.tracked_worker(worker_id)

    def restart_worker(self, worker_id: str) -> None:
        """Restart a worker with a fresh bootstrap script using the latest image."""

        restart_worker_operation(self._groups, self._db, worker_id, self._per_group_worker_config)

    def restore_tracked_workers(self, workers: dict[str, TrackedWorker]) -> None:
        """Restore tracked worker state from a snapshot. Called before loops start."""
        self._worker_registry.restore(workers)

    def restore_from_db(self, db: ControllerDB, platform: WorkerInfraProvider) -> None:
        """Reconcile DB-checkpointed autoscaler state against live cloud.

        Reads scaling group and slice rows from proper DB tables,
        reconciles each group against the cloud in parallel, and restores
        tracked workers. Call at startup before loops begin.
        """
        checkpoint = load_autoscaler_checkpoint(db)
        restored_workers = restore_autoscaler_state(self._groups, checkpoint, platform)
        self.restore_tracked_workers(restored_workers)
        logger.info("Restored %d tracked workers", len(restored_workers))

    def get_vm(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Get VM info by platform worker ID from the centralized worker registry."""
        return self._worker_registry.vm_info(vm_id)

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get bootstrap log for a VM by platform worker ID."""
        return self._worker_registry.init_log(vm_id, tail)

    def job_feasibility(
        self,
        constraints: list[Constraint],
        *,
        replicas: int | None = None,
    ) -> str | None:
        """Gate LaunchJob: can this job shape ever be scheduled?

        Returns None if some scaling group can, in principle, host the job
        (autoscaler may still need to scale up); otherwise a human-readable
        reason suitable for returning to the caller.

        `replicas` applies only to coscheduled jobs — None skips the
        num_vms-divisibility check.
        """
        result = job_feasibility(self._groups.values(), constraints, replicas=replicas)
        return result.reason

    def get_last_routing_decision_proto(self) -> vm_pb2.RoutingDecision | None:
        """Return the last routing decision as a proto.

        Populated by evaluate() so dashboard/service callers (GetJobStatus,
        ListJobs) never pay the per-entry conversion cost on the hot path
        (#4844). Returns None before the first evaluate() cycle.
        """
        return self._last_routing_decision_proto

    def get_pending_hints(self) -> dict[str, PendingHint]:
        """Return autoscaler pending hints keyed by job id.

        Populated by evaluate(); the service never triggers a live rebuild.
        Returns an empty dict before the first evaluate() cycle or if no
        hints are cached yet (#4844).
        """
        if self._last_pending_hints is None:
            return {}
        return self._last_pending_hints

    def get_status(self) -> vm_pb2.AutoscalerStatus:
        """Build status for the status API."""
        status = vm_pb2.AutoscalerStatus(
            groups=[g.to_status() for g in self._groups.values()],
            current_demand={g.name: g.current_demand for g in self._groups.values()},
            last_evaluation=timestamp_to_proto(self._last_evaluation),
            recent_actions=list(self._action_log),
        )
        routing_proto = self.get_last_routing_decision_proto()
        if routing_proto is not None:
            status.last_routing_decision.CopyFrom(routing_proto)
        return status

    def get_group(self, name: str) -> ScalingGroup | None:
        """Get a scale group by name."""
        return self._groups.get(name)

    @property
    def groups(self) -> dict[str, ScalingGroup]:
        """All scale groups."""
        return self._groups

    def terminate_slices_for_workers(self, worker_ids: Sequence[str]) -> list[str]:
        """Terminate the unique slices containing the given workers.

        Returns sibling worker IDs that should be failed immediately because
        their slices are being torn down.
        """
        result = terminate_slices_for_workers_operation(
            groups=self._groups,
            worker_ids=worker_ids,
            unregister_slice_workers=self._unregister_slice_workers,
            log_action=self._log_action,
            timestamp=Timestamp.now(),
            short_lived_slice_threshold=SHORT_LIVED_SLICE_THRESHOLD,
        )
        for request in result.termination_requests:

            def _terminate(
                stop_event,
                target_group: ScalingGroup = request.group,
                target_slice_id: str = request.slice_id,
                target_handle: SliceHandle = request.handle,
            ) -> None:
                del stop_event
                self._terminate_slice_handle(target_group, target_slice_id, target_handle)

            self._threads.spawn(
                target=_terminate,
                name=f"slice-terminate-{request.slice_id}",
            )

        return result.sibling_worker_ids

    def _terminate_slice_handle(self, group: ScalingGroup, slice_id: str, handle: SliceHandle) -> None:
        group._terminate_slice_handle(handle, context="cleaning up anyway")
