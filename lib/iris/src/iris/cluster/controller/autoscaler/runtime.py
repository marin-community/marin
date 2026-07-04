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

import logging
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

from finelog.client.log_client import Table
from rigging.timing import Duration, Timestamp, TokenBucket

from iris.cluster.config import AutoscalerConfig, WorkerConfig
from iris.cluster.constraints import Constraint
from iris.cluster.controller.autoscaler.models import (
    DemandEntry,
    ScalingAction,
    ScalingDecision,
)
from iris.cluster.controller.autoscaler.operations import (
    terminate_slices_for_workers as terminate_slices_for_workers_operation,
)
from iris.cluster.controller.autoscaler.planning import build_scale_plan
from iris.cluster.controller.autoscaler.provisioning import (
    ERROR_MESSAGE_MAX_LEN,
    IrisProvisioning,
    ProvisioningOutcome,
    classify_create_failure,
)
from iris.cluster.controller.autoscaler.recovery import (
    load_autoscaler_checkpoint,
    restore_autoscaler_state,
)
from iris.cluster.controller.autoscaler.routing import (
    availability_probe_entries,
    empirical_zone_capabilities,
    job_feasibility,
    route_demand,
)
from iris.cluster.controller.autoscaler.scaling_group import (
    ScalingGroup,
    build_worker_config_for_group,
)
from iris.cluster.controller.autoscaler.state import AutoscalerState, GroupPersist, SlicePersist
from iris.cluster.controller.autoscaler.status import PendingHint, build_job_pending_hints, routing_decision_to_proto
from iris.cluster.controller.autoscaler.worker_registry import TrackedWorker, WorkerRegistry
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.worker_health import CONSECUTIVE_FAILURE_THRESHOLD
from iris.cluster.platforms.protocols import WorkerInfraProvider
from iris.cluster.platforms.types import (
    CloudSliceState,
    QuotaExhaustedError,
    RemoteWorkerHandle,
    SliceHandle,
    SliceStatus,
)
from iris.cluster.types import WorkerStatusMap
from iris.rpc import job_pb2, vm_pb2
from iris.time_proto import timestamp_to_proto

logger = logging.getLogger(__name__)


# After this long in UNKNOWN state, treat the slice as FAILED (quota timeout is 5 min, so this is conservative)
DEFAULT_UNRESOLVABLE_TIMEOUT = Duration.from_minutes(15)

# Project-wide cap on slice creates per minute, shared across all scale groups.
# Per-group limits don't coordinate, so a fan-out across groups in one cycle can
# burst past GCP's per-project create quota (tpu.googleapis.com/qps/create, ~91/min).
DEFAULT_CREATE_RATE_LIMIT = 60

# How long the autoscaler waits for a worker /health response per probe.
_HEALTH_PROBE_TIMEOUT_SECONDS = 3.0

# Bypass any HTTP_PROXY env var: worker addresses are private cluster IPs,
# never reachable via an upstream proxy.
_HEALTH_PROBE_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Cap concurrent /health probes. ~1000 VMs in production; serializing 3 s
# timeouts would blow past evaluation_interval (10 s default).
_HEALTH_PROBE_MAX_WORKERS = 64

# Cap concurrent describe() calls in refresh(). Each is a blocking GCP round-trip
# (two for a still-queued reserved slice), so a large reserved backlog would
# otherwise serialize into tens of seconds and starve the shared control loop.
_REFRESH_DESCRIBE_MAX_WORKERS = 64

# Cap concurrent slice create/terminate cloud requests issued in one phase. The
# fan-out keeps a burst (cold-start scale-up, mass-preemption teardown) within
# the phase budget instead of serializing one bounded HTTP round-trip at a time.
_CLOUD_OP_MAX_WORKERS = 16

_T = TypeVar("_T")
_R = TypeVar("_R")


def _run_io_batch(
    items: Sequence[_T],
    issue: Callable[[_T], _R],
    *,
    max_workers: int,
    thread_name_prefix: str,
) -> list[_R]:
    """Run a pure-I/O ``issue`` over ``items`` on a bounded, joined thread pool.

    The pool only performs cloud/network I/O and *returns* its results; callers
    fold those results into autoscaler state serially afterward. Because no
    shared state is touched inside the pool, the fan-out is race-free regardless
    of its width, and it joins before returning so the phase stays bounded.
    """
    if not items:
        return []
    workers = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix) as pool:
        return list(pool.map(issue, items))


def _probe_worker_health(worker_url: str) -> bool:
    """Probe a worker's /health endpoint. ``worker_url`` is an ``http://host:port`` base URL.

    Returns True iff the response is 2xx.
    """
    try:
        resp = _HEALTH_PROBE_OPENER.open(
            f"{worker_url}/health",
            timeout=_HEALTH_PROBE_TIMEOUT_SECONDS,
        )
        return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


def _safe_describe(slice_id: str, handle: SliceHandle) -> SliceStatus | None:
    """Describe a slice on the fan-out pool, returning None (logged) if it raises."""
    try:
        return handle.describe()
    except Exception as e:
        # A failed poll is transient: skip this slice this tick and retry. A slice
        # that stays unresolvable is failed via the UNKNOWN/unresolvable-timeout path.
        logger.warning("Failed to poll slice %s: %s", slice_id, e)
        return None


@dataclass
class _ScaleUpRequest:
    """One scale-up to issue this phase: a group, its reason, and the pending action."""

    group: ScalingGroup
    reason: str
    action: vm_pb2.AutoscalerAction


@dataclass
class _ScaleUpOutcome:
    """Result of issuing one create: the slice handle on success, else the error.

    Captured as plain data so the issuing fan-out touches no shared state; the
    outcome is classified and folded into autoscaler state serially.
    """

    request: _ScaleUpRequest
    handle: SliceHandle | None = None
    error: Exception | None = None


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
        base_worker_config: WorkerConfig | None = None,
        unresolvable_timeout: Duration = DEFAULT_UNRESOLVABLE_TIMEOUT,
        create_rate_limit: int = DEFAULT_CREATE_RATE_LIMIT,
        make_draining_group: Callable[[str], ScalingGroup] | None = None,
        provisioning_table: Table | None = None,
    ):
        """Create autoscaler with explicit parameters.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            evaluation_interval: How often to evaluate scaling decisions
            platform: WorkerInfraProvider instance for shutdown lifecycle
            base_worker_config: Base worker config merged with per-group overrides
                and passed to platform.create_slice(). None disables bootstrap (test/local mode).
            unresolvable_timeout: How long a slice can remain UNKNOWN before being treated as FAILED.
            create_rate_limit: Project-wide ceiling on slice-creation requests per minute,
                shared across all scale groups. See ``DEFAULT_CREATE_RATE_LIMIT``.
            make_draining_group: Builds a scale-to-zero ``ScalingGroup`` for a scale group that has
                left config but still has live VMs (see ``restore_autoscaler_state``). None disables
                drain adoption (test/local mode); the factory always wires it in prod.
            provisioning_table: finelog ``iris.provisioning`` table for slice-provisioning
                outcomes. None disables emission (test/local mode without finelog).
        """
        self._groups = scale_groups
        self._make_draining_group = make_draining_group
        self._platform = platform
        self.evaluation_interval = evaluation_interval
        self._base_worker_config = base_worker_config
        self._unresolvable_timeout = unresolvable_timeout

        # Project-wide slice-creation throttle, shared across all groups. Deferred
        # scale-ups are retried on the next evaluation cycle.
        self._create_rate_limit = create_rate_limit
        self._create_bucket = TokenBucket(capacity=create_rate_limit, refill_period=Duration.from_minutes(1))

        # Centralized per-worker state indexed by worker_id
        self._worker_registry = WorkerRegistry()

        # Bounded log of recent autoscaler actions for dashboard/debugging
        self._action_log: deque[vm_pb2.AutoscalerAction] = deque(maxlen=100)

        # finelog table for the iris.provisioning namespace, built from the
        # controller's log client before construction. None in test/local mode
        # disables emission.
        self._provisioning_table: Table | None = provisioning_table

        self._last_evaluation: Timestamp = Timestamp.from_ms(0)

        # Most recent routing decision, materialized as status protos. Dashboard
        # polls (GetJobStatus, ListJobs) hit these on every pending job; building
        # them per request was the bottleneck described in #4844.
        self._last_routing_decision_proto: vm_pb2.RoutingDecision | None = None
        self._last_pending_hints: dict[str, PendingHint] | None = None

    @classmethod
    def from_config(
        cls,
        scale_groups: dict[str, ScalingGroup],
        config: AutoscalerConfig,
        platform: WorkerInfraProvider,
        base_worker_config: WorkerConfig | None = None,
        make_draining_group: Callable[[str], ScalingGroup] | None = None,
        provisioning_table: Table | None = None,
    ) -> "Autoscaler":
        """Create autoscaler from config.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            config: Autoscaler configuration (with defaults already applied)
            platform: WorkerInfraProvider instance for shutdown lifecycle
            base_worker_config: Base worker config merged with per-group overrides
            make_draining_group: Builds a scale-to-zero group for a retired-but-live scale group
                (see ``Autoscaler.__init__`` / ``restore_autoscaler_state``).
            provisioning_table: finelog ``iris.provisioning`` table for slice-provisioning outcomes.

        Returns:
            Configured Autoscaler instance
        """
        return cls(
            scale_groups=scale_groups,
            evaluation_interval=config.evaluation_interval,
            platform=platform,
            base_worker_config=base_worker_config,
            make_draining_group=make_draining_group,
            provisioning_table=provisioning_table,
        )

    def shutdown(self) -> None:
        """Shutdown the autoscaler, terminate all VM groups, and clean up platform.

        Cloud operations (scale-up, terminate) run synchronously inside the
        autoscale phase — there are no background threads to join — so shutdown
        is just: terminate all tracked slices, then release platform resources.
        """
        for group in self._groups.values():
            group.terminate_all()
        self._platform.shutdown()

    def __enter__(self) -> "Autoscaler":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def _record_provisioning_outcome(
        self,
        group: ScalingGroup,
        outcome: ProvisioningOutcome,
        *,
        created_at: Timestamp | None = None,
        error_message: str = "",
        worker_count: int = 0,
    ) -> None:
        """Write one ``iris.provisioning`` row; no-op until the sink is injected."""
        table = self._provisioning_table
        if table is None:
            return
        latency_ms = 0
        if outcome == ProvisioningOutcome.READY and created_at is not None:
            latency_ms = max(0, Timestamp.now().epoch_ms() - created_at.epoch_ms())
        table.write(
            [
                IrisProvisioning(
                    ts=Timestamp.now().as_naive_utc(),
                    resource_type=group.device_type.value,
                    scale_group=group.name,
                    zone=group.zone or "",
                    accelerator_variant=group.accelerator_variant,
                    outcome=outcome.value,
                    error_message=error_message[:ERROR_MESSAGE_MAX_LEN],
                    worker_count=worker_count,
                    provision_latency_ms=latency_ms,
                )
            ]
        )

    def _log_action(
        self,
        action_type: str,
        scale_group: str,
        slice_id: str = "",
        reason: str = "",
        status: str = "completed",
        *,
        group: ScalingGroup | None = None,
        outcome: ProvisioningOutcome | None = None,
        created_at: Timestamp | None = None,
        worker_count: int = 0,
    ) -> vm_pb2.AutoscalerAction:
        """Append an autoscaler action to the bounded action log and return it.

        The returned action is mutable so callers can update its status after
        execution (e.g. pending → completed). When ``outcome`` is given, the same
        call also writes the structured :class:`IrisProvisioning` row — the action
        log line and the provisioning row are two views of one event, sharing
        ``reason`` as the row's ``error_message`` — so an outcome site logs once.
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
        if outcome is not None:
            assert group is not None, "provisioning outcome requires the ScalingGroup"
            self._record_provisioning_outcome(
                group, outcome, created_at=created_at, error_message=reason, worker_count=worker_count
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

        # Empirical availability: a variant is "available" in a region only once a
        # scale-up there has succeeded. Convert each unmet availability:<variant>
        # constraint into a probe scale-up of that accelerator group so capacity can
        # be discovered/established; the demand subsides once the constrained job
        # places (see availability_probe_entries).
        caps = self.zone_capabilities(ts)
        available_variants = frozenset(variant for variants in caps.values() for variant in variants)
        probes = availability_probe_entries(list(self._groups.values()), demand_entries, available_variants)
        if probes:
            demand_entries = list(demand_entries) + probes

        routing_decision = route_demand(list(self._groups.values()), demand_entries, ts, zone_capabilities=caps)
        scale_plan = build_scale_plan(self._groups, routing_decision, ts)
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

    def zone_capabilities(self, timestamp: Timestamp | None = None) -> dict[str, frozenset[str]]:
        """Map zone -> {device_variant} the cluster has EMPIRICALLY confirmed available.

        A variant counts for a zone only after a scale-up there succeeded (≥1 live
        ``READY`` slice, not erroring) — see :func:`empirical_zone_capabilities`.
        """
        ts = timestamp or Timestamp.now()
        return empirical_zone_capabilities(self._groups.values(), ts)

    def execute(
        self,
        decisions: list[ScalingDecision],
        timestamp: Timestamp,
    ) -> None:
        """Execute scale-up decisions.

        Runs in three steps so the cloud I/O never races autoscaler state:
        plan (serial: apply rate limits, mark pending, build the issue list),
        issue (bounded parallel: submit the creates, capturing handle-or-error
        as data), then fold (serial: record each outcome into group state).

        Args:
            decisions: List of scaling decisions to execute.
            timestamp: Current timestamp.
        """
        # Aggregate rate-limited decisions per group so we emit a single summary
        # line per group per cycle instead of one per deferred slice (#5580).
        rate_limited: dict[str, list[ScalingDecision]] = {}
        # Scale-ups deferred by the project-wide create budget (not a per-group
        # limit), summarized once per cycle so a global throttle doesn't spam a
        # line per group.
        create_throttled = 0
        # Plan: gate each decision and mark it pending before any I/O.
        to_issue: list[_ScaleUpRequest] = []
        for decision in decisions:
            group = self._groups.get(decision.scale_group)
            if not group:
                logger.warning("Unknown scale group in decision: %s", decision.scale_group)
                continue

            if decision.action == ScalingAction.SCALE_UP:
                # Per-group gate first: it honors the group's health/quota block,
                # so we never spend the scarce global create budget on a group
                # that couldn't scale anyway.
                if not group.try_acquire_scale_up(timestamp):
                    rate_limited.setdefault(decision.scale_group, []).append(decision)
                    continue
                if not self._create_bucket.try_acquire(now=timestamp):
                    create_throttled += 1
                    continue
                group.begin_scale_up(timestamp=timestamp)
                action = self._log_action("scale_up", group.name, reason=decision.reason, status="pending")
                to_issue.append(_ScaleUpRequest(group=group, reason=decision.reason, action=action))

        # Issue the creates in a bounded parallel fan-out (pure I/O), then fold
        # each outcome into state serially.
        outcomes = _run_io_batch(
            to_issue,
            lambda req: self._issue_scale_up(req, timestamp),
            max_workers=_CLOUD_OP_MAX_WORKERS,
            thread_name_prefix="scale-up",
        )
        for outcome in outcomes:
            self._fold_scale_up(outcome, timestamp)

        if create_throttled:
            logger.info(
                "Throttled %d scale-up(s) to stay under the global create limit (%d/min)",
                create_throttled,
                self._create_rate_limit,
            )
            self._log_action(
                "create_throttled",
                "",
                reason=f"deferred={create_throttled} limit={self._create_rate_limit}/min",
            )

        for scale_group, deferred in rate_limited.items():
            # All decisions in a cycle share the same target/demand snapshot;
            # the first reason is representative of the whole batch.
            sample_reason = deferred[0].reason
            summary = f"deferred={len(deferred)} sample_reason={sample_reason}"
            logger.info(
                "Rate-limited scale-up for %s: deferred %d slice(s) (sample reason: %s)",
                scale_group,
                len(deferred),
                sample_reason,
            )
            self._log_action(
                "rate_limited",
                scale_group,
                reason=summary,
            )

    def _issue_scale_up(self, request: _ScaleUpRequest, ts: Timestamp) -> _ScaleUpOutcome:
        """Submit one create and return the handle-or-error as data. Pure I/O.

        Runs inside the bounded issue fan-out, so it must not touch shared
        autoscaler/group state: ``group.scale_up`` only submits the create LRO
        (bounded; it does not wait) and returns a handle. The exception is
        captured rather than raised so a single failed create can't abort the
        whole batch; classification and state changes happen in
        :meth:`_fold_scale_up`.
        """
        group = request.group
        try:
            logger.info("Scaling up %s: %s", group.name, request.reason)
            wc = self._per_group_worker_config(group)
            handle = group.scale_up(worker_config=wc, timestamp=ts)
            return _ScaleUpOutcome(request=request, handle=handle)
        except Exception as e:
            # Captured as data (not swallowed): _fold_scale_up classifies it,
            # records the failure on the group, and logs with a stack trace.
            return _ScaleUpOutcome(request=request, error=e)

    def _fold_scale_up(self, outcome: _ScaleUpOutcome, ts: Timestamp) -> None:
        """Fold one create outcome into group state. Serial; mutates shared state.

        Success records the slice as BOOTING (it then advances BOOTING→READY/
        FAILED via the describe() poll in :meth:`refresh`). Quota/error failures
        clear the pending scale-up and feed the group's detector so the freed
        demand re-plans on the next cycle.
        """
        request = outcome.request
        group, action = request.group, request.action

        if outcome.error is None:
            handle = outcome.handle
            assert handle is not None, "a successful scale-up must carry a slice handle"
            group.complete_scale_up(handle, ts)
            logger.info("Created slice %s for group %s", handle.slice_id, group.name)
            action.slice_id = handle.slice_id
            action.status = "completed"
            return

        error = outcome.error
        if isinstance(error, QuotaExhaustedError):
            group.cancel_scale_up()
            group.record_quota_exceeded(str(error), ts)
            logger.warning("Quota exceeded for %s: %s", group.name, error)
            action.action_type = "quota_exceeded"
            action.status = "failed"
            action.reason = str(error)
            # A create that failed at submit (no slice handle, so no later
            # describe() outcome) — record it here or it's invisible to the
            # success rate. Quota/stockout both surface as QuotaExhaustedError.
            self._record_provisioning_outcome(group, classify_create_failure(str(error)), error_message=str(error))
            return

        group.cancel_scale_up()
        logger.error("Failed to create slice for %s: %s", group.name, error, exc_info=error)
        action.status = "failed"
        action.reason = f"{request.reason} - error: {error}"
        group.record_create_failed(ts)
        self._record_provisioning_outcome(group, ProvisioningOutcome.ERROR, error_message=str(error))

    def _per_group_worker_config(self, group: ScalingGroup) -> WorkerConfig | None:
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
        """Poll non-READY slices and scale down idle ones.

        Reading a slice's cloud state is a blocking GCP round-trip (two for a
        still-queued reserved slice). A large reserved backlog would serialize
        into tens of seconds and, since this runs inline on the shared control
        loop, starve reconcile — so the reads are fanned out over a bounded pool
        and folded serially (see the phase comments below).
        """
        timestamp = timestamp or Timestamp.now()

        # Phase 1: snapshot every slice that needs a describe() — not-yet-ready
        # slices, plus READY slices the autoscaler tracks no workers for (so an
        # adopted READY-but-empty slice repopulates instead of staying DEGRADED).
        targets = [
            (group, slice_id, handle)
            for group in self._groups.values()
            for slice_id, handle in group.slices_needing_describe()
        ]

        # Phase 2: fan out the blocking describe() over a bounded pool.
        statuses = _run_io_batch(
            targets,
            lambda t: _safe_describe(t[1], t[2]),
            max_workers=_REFRESH_DESCRIBE_MAX_WORKERS,
            thread_name_prefix="slice-describe",
        )

        # Phase 3: fold describe results into group state serially.
        for (group, slice_id, handle), status in zip(targets, statuses, strict=True):
            if status is None:
                continue
            if status.state == CloudSliceState.READY:
                worker_ids = [w.worker_id for w in status.workers]
                worker_urls = self._worker_urls(status.workers)
                group.mark_slice_ready(slice_id, worker_ids, worker_urls=worker_urls)
                self._register_slice_workers(status.workers, slice_id, group.name)
                self._log_action(
                    "slice_ready",
                    group.name,
                    slice_id,
                    reason=f"bootstrap completed ({len(worker_ids)} workers)",
                    group=group,
                    outcome=ProvisioningOutcome.READY,
                    created_at=handle.created_at,
                    worker_count=len(worker_ids),
                )
            elif status.state == CloudSliceState.FAILED:
                group.mark_slice_failed(slice_id, error_message=status.error_message)
                group.scale_down(slice_id)
                self._unregister_slice_workers(slice_id)
                group.record_slice_boot_failed(slice_id, timestamp)
                reason = status.error_message if status.error_message else "bootstrap failed"
                self._log_action(
                    "slice_failed",
                    group.name,
                    slice_id,
                    reason=reason,
                    status="failed",
                    group=group,
                    outcome=classify_create_failure(reason),
                )
            elif status.state == CloudSliceState.UNKNOWN:
                # Measure how long the slice has been CONTINUOUSLY UNKNOWN, not its
                # age since creation: a single transient UNKNOWN describe must not
                # terminate a long-running or freshly-adopted (drained) slice that
                # still has running tasks.
                unknown_for = group.note_slice_unknown(slice_id, timestamp)
                if unknown_for >= self._unresolvable_timeout:
                    group.mark_slice_failed(slice_id, error_message="unresolvable after timeout")
                    group.scale_down(slice_id)
                    self._unregister_slice_workers(slice_id)
                    group.record_slice_boot_failed(slice_id, timestamp)
                    self._log_action(
                        "slice_failed",
                        group.name,
                        slice_id,
                        reason=f"unresolvable for {unknown_for}",
                        status="failed",
                        group=group,
                        outcome=ProvisioningOutcome.ERROR,
                    )
                else:
                    logger.debug(
                        "Slice %s UNKNOWN (unknown for %s < timeout %s); will retry",
                        slice_id,
                        unknown_for,
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

    def probe_health(self, timestamp: Timestamp | None = None) -> None:
        """Probe each READY slice's worker /health endpoint and terminate dead slices.

        Catches zombie slices whose VM is still up in the cloud but whose
        worker process is dead — these would otherwise be invisible to the
        heartbeat path (no worker row → no heartbeat to time out) and pin the
        scale group at max_slices indefinitely. Also catches READY slices whose
        backing allocation vanished entirely (e.g. a preempted TPU after a
        controller restart, when no worker URLs were cached): describe() resolves
        zero workers, so there is nothing to probe and no heartbeat row to expire,
        and the slice is reaped via a per-slice no-worker counter instead.

        Per-worker counters live on SliceState. CONSECUTIVE_FAILURE_THRESHOLD
        consecutive failures (~100s at the default 10s evaluation interval) trip
        termination — the same constant the controller's worker-liveness
        tracker uses for its reconcile-failure threshold. Worker URLs
        come from the slice handle, refreshed lazily via ``handle.describe()``
        when SliceState has none cached (e.g. after a controller restart).
        Probes are fanned out over a bounded thread pool so a partitioned AZ
        doesn't burn the entire evaluation interval at the 3s per-probe timeout.
        The pool returns results; slice teardown is folded in serially after.
        """
        timestamp = timestamp or Timestamp.now()

        # Phase 1: collect every (group, slice_id, worker_id, worker_url) probe
        # target. A READY slice that resolves to zero workers (cloud allocation
        # gone) has nothing to probe; it's tracked via a per-slice counter and
        # torn down after CONSECUTIVE_FAILURE_THRESHOLD sustained empty observations,
        # which the per-worker counters never catch.
        probes: list[tuple[ScalingGroup, str, str, str]] = []
        tripped: dict[str, tuple[ScalingGroup, str]] = {}  # slice_id -> (group, reason)
        for group in self._groups.values():
            for slice_id, handle, worker_urls in group.ready_slice_probe_targets():
                if not worker_urls:
                    refreshed = self._refresh_slice_worker_urls(group, slice_id, handle)
                    if refreshed is None:
                        continue  # describe() failed or still booting; retry next tick
                    worker_urls = refreshed
                if not worker_urls:
                    count = group.record_slice_no_workers(slice_id)
                    if count >= CONSECUTIVE_FAILURE_THRESHOLD and slice_id not in tripped:
                        reason = f"cloud allocation reports no workers ({count}x)"
                        logger.warning("Slice %s: %s; terminating", slice_id, reason)
                        tripped[slice_id] = (group, reason)
                    continue
                for worker_id, worker_url in worker_urls.items():
                    probes.append((group, slice_id, worker_id, worker_url))

        # Phase 2: fan out probes over a bounded thread pool, returning results.
        if probes:
            results = _run_io_batch(
                probes,
                lambda p: _probe_worker_health(p[3]),
                max_workers=_HEALTH_PROBE_MAX_WORKERS,
                thread_name_prefix="health-probe",
            )

            # Phase 3: record results and collect slices that tripped the threshold.
            for (group, slice_id, worker_id, _url), healthy in zip(probes, results, strict=True):
                count = group.record_health_probe_result(slice_id, worker_id, healthy)
                if count >= CONSECUTIVE_FAILURE_THRESHOLD and slice_id not in tripped:
                    reason = f"worker {worker_id} failed /health {count}x"
                    logger.warning("Slice %s: %s; terminating", slice_id, reason)
                    tripped[slice_id] = (group, reason)

        # Phase 4: terminate tripped slices. These booted cleanly and only died
        # at runtime, so record PREEMPTED (excluded from the create success rate),
        # not a boot failure — the BackoffDetector's scale-up budget shouldn't decay.
        for slice_id, (group, reason) in tripped.items():
            group.mark_slice_failed(slice_id, error_message=reason)
            group.scale_down(slice_id, timestamp)
            self._unregister_slice_workers(slice_id)
            self._log_action(
                "slice_failed",
                group.name,
                slice_id,
                reason=reason,
                status="failed",
                group=group,
                outcome=ProvisioningOutcome.PREEMPTED,
            )

    def _worker_urls(self, workers: Sequence[RemoteWorkerHandle]) -> dict[str, str]:
        """Map worker_id to the worker's reachable ``http://host:port`` URL.

        Workers with no internal address yet (mid-boot) report an empty
        ``worker_url`` and are skipped.
        """
        return {w.worker_id: w.worker_url for w in workers if w.worker_url}

    def _refresh_slice_worker_urls(
        self, group: ScalingGroup, slice_id: str, handle: SliceHandle
    ) -> dict[str, str] | None:
        """Resolve worker URLs for a slice by calling handle.describe().

        Returns:
            - The resolved ``worker_id -> url`` map when describe() succeeds and
              every reported worker has published a URL. The map is *empty* when
              the cloud reports zero workers (the backing allocation is gone) --
              the caller treats that as a missing-worker signal, not a no-op.
            - ``None`` when the result is inconclusive and the slice should be
              retried next tick without penalty: describe() raised, or only some
              workers have published an IP (the slice is still booting). A
              partial set is never cached -- doing so would permanently exclude
              the missing workers from probing, and probing the incomplete set
              risks terminating a slice for a worker that simply hasn't
              published its IP yet.
        """
        try:
            status = handle.describe()
        except Exception as e:
            logger.warning("Failed to describe slice %s for health probe: %s", slice_id, e)
            return None
        worker_urls = self._worker_urls(status.workers)
        if status.workers and len(worker_urls) != len(status.workers):
            return None
        group.set_worker_urls(slice_id, worker_urls)
        return worker_urls

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
        """Full cycle: refresh + probe_health + update. Preserved for tests."""
        timestamp = timestamp or Timestamp.now()
        logger.debug("Autoscaler run_once: demand_entries=%s", demand_entries)
        self.refresh(worker_status_map, timestamp)
        self.probe_health(timestamp)
        return self.update(demand_entries, timestamp)

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
        restored_workers = restore_autoscaler_state(self._groups, checkpoint, platform, self._make_draining_group)
        self.restore_tracked_workers(restored_workers)
        logger.info("Restored %d tracked workers", len(restored_workers))

    def persistable_state(self) -> AutoscalerState:
        """Snapshot all tracked slices and groups for the controller to persist.

        The in-memory ``ScalingGroup`` state is authoritative during operation;
        the controller mirrors this snapshot into the ``slices`` /
        ``scaling_groups`` tables after each capacity call so a restarted
        controller can recover.
        """
        all_slices: list[SlicePersist] = []
        groups: list[GroupPersist] = []
        for group in self._groups.values():
            group_slices, group_row = group.persistable_state()
            all_slices.extend(group_slices)
            groups.append(group_row)
        return AutoscalerState(slices=all_slices, groups=groups)

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
        resources: job_pb2.ResourceSpecProto | None = None,
    ) -> str | None:
        """Gate LaunchJob: can this job shape ever be scheduled?

        Returns None if some scaling group can, in principle, host the job
        (autoscaler may still need to scale up); otherwise a human-readable
        reason suitable for returning to the caller.

        `replicas` applies only to coscheduled jobs — None skips the
        num_vms-divisibility check. `resources`, when given, additionally
        rejects requests whose per-VM capacity (cpu/memory/disk/device count)
        no matching group can satisfy.
        """
        result = job_feasibility(self._groups.values(), constraints, replicas=replicas, resources=resources)
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

    @property
    def groups(self) -> dict[str, ScalingGroup]:
        """All scale groups."""
        return self._groups

    def terminate_slices_for_workers(self, worker_ids: Sequence[str]) -> list[str]:
        """Terminate the unique slices containing the given workers.

        Returns sibling worker IDs that should be failed immediately because
        their slices are being torn down. The operation detaches the slices from
        tracking serially (the state mutation), then issues the deletes in a
        bounded parallel fan-out: each ``terminate_slice_handle`` is a bounded
        cloud request that catches and logs its own errors and touches no shared
        state, so the fan-out stays within the phase budget under a mass
        preemption without racing.
        """
        result = terminate_slices_for_workers_operation(
            groups=self._groups,
            worker_ids=worker_ids,
            unregister_slice_workers=self._unregister_slice_workers,
            log_action=self._log_action,
            timestamp=Timestamp.now(),
        )
        # These slices had registered workers (so they reached READY) and lost
        # them at runtime — the primary preemption path, which the probe_health
        # backstop never sees. Record PREEMPTED so they don't pollute the create
        # success rate (mirrors the probe_health termination path).
        for req in result.termination_requests:
            self._record_provisioning_outcome(
                req.group, ProvisioningOutcome.PREEMPTED, error_message="worker liveness lost"
            )
        _run_io_batch(
            result.termination_requests,
            lambda req: req.group.terminate_slice_handle(req.handle, context="cleaning up anyway"),
            max_workers=_CLOUD_OP_MAX_WORKERS,
            thread_name_prefix="slice-terminate",
        )
        return result.sibling_worker_ids
