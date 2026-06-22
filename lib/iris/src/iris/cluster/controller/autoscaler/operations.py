# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operational helpers for autoscaler worker and slice actions."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Timestamp

from iris.cluster.backends.types import SliceHandle
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.rpc import vm_pb2

logger = logging.getLogger(__name__)


class SliceDrainCause(StrEnum):
    """Why a slice's workers are being drained.

    Selects telemetry and whether the teardown counts against the group's health
    and backoff signals: a lost-liveness failure is charged to the group as churn,
    a deliberate cross-variant preemption leaves those signals untouched.
    """

    WORKER_FAILED = "worker_failed"
    PREEMPTED = "preempted"


@dataclass(frozen=True)
class DrainTelemetry:
    """Per-cause action label, reason prefix, log verb, and terminate context."""

    action_type: str
    reason_prefix: str
    log_verb: str
    terminate_context: str


DRAIN_TELEMETRY: dict[SliceDrainCause, DrainTelemetry] = {
    SliceDrainCause.WORKER_FAILED: DrainTelemetry(
        action_type="worker_failed",
        reason_prefix="workers failed",
        log_verb="termination",
        terminate_context="cleaning up anyway",
    ),
    SliceDrainCause.PREEMPTED: DrainTelemetry(
        action_type="slice_drained",
        reason_prefix="drained for preemption",
        log_verb="drain",
        terminate_context="draining for preemption",
    ),
}


@dataclass(frozen=True)
class SliceTerminationRequest:
    """A DRAINING slice handle whose VMs the caller must terminate."""

    slice_id: str
    group: ScalingGroup
    handle: SliceHandle


@dataclass(frozen=True)
class SliceTerminationResult:
    """Drain work derived from a set of workers leaving their slices."""

    sibling_worker_ids: list[str]
    termination_requests: list[SliceTerminationRequest]


def drain_slices_for_workers(
    *,
    groups: dict[str, ScalingGroup],
    worker_ids: Sequence[str],
    unregister_slice_workers: Callable[[str, Sequence[str] | None], None],
    log_action: Callable[..., vm_pb2.AutoscalerAction],
    timestamp: Timestamp,
    cause: SliceDrainCause,
) -> SliceTerminationResult:
    """Mark every slice holding ``worker_ids`` DRAINING and collect its handle.

    Workers are grouped by physical slice because the teardown removes a whole
    slice: every task on it goes together and its chips free once. Each slice is
    drained at most once. The slice stays tracked — counted against its reservation
    pool as DRAINING — until ``refresh`` observes its VMs are gone and reaps it; the
    caller terminates the returned handles to start that deletion.

    ``cause`` selects telemetry and, for ``WORKER_FAILED``, records a PREEMPTED fate
    on the group's churn detector. Returns the drained slices' sibling worker ids
    for the caller to fail immediately.
    """
    if not worker_ids:
        return SliceTerminationResult(sibling_worker_ids=[], termination_requests=[])

    telemetry = DRAIN_TELEMETRY[cause]
    primary_workers = set(worker_ids)
    sibling_worker_ids: set[str] = set()
    termination_requests: list[SliceTerminationRequest] = []
    slices_seen: set[str] = set()

    for worker_id in primary_workers:
        slice_id, group = find_slice_for_worker(groups, worker_id)
        if not slice_id or group is None:
            logger.debug("Worker %s not found in any managed slice", worker_id)
            continue
        if slice_id in slices_seen:
            continue
        slices_seen.add(slice_id)

        slice_worker_ids = group.get_slice_worker_ids(slice_id)
        sibling_worker_ids.update(wid for wid in slice_worker_ids if wid not in primary_workers)
        affected_workers = sorted(primary_workers & set(slice_worker_ids))

        logger.info("Workers %s triggered slice %s for %s", affected_workers, telemetry.log_verb, slice_id)
        log_action(
            telemetry.action_type,
            group.name,
            slice_id=slice_id,
            reason=f"{telemetry.reason_prefix}: {', '.join(affected_workers)}",
        )
        if cause is SliceDrainCause.WORKER_FAILED:
            group.record_slice_preempted(slice_id, timestamp)
        handle = group.drain_slice(slice_id, timestamp)
        unregister_slice_workers(slice_id, slice_worker_ids)
        if handle is not None:
            termination_requests.append(SliceTerminationRequest(slice_id=slice_id, group=group, handle=handle))

    return SliceTerminationResult(
        sibling_worker_ids=sorted(sibling_worker_ids),
        termination_requests=termination_requests,
    )


def find_slice_for_worker(
    groups: dict[str, ScalingGroup],
    worker_id: str,
) -> tuple[str | None, ScalingGroup | None]:
    """Find the slice and group containing a worker by worker ID."""

    for group in groups.values():
        slice_id = group.find_slice_for_worker(worker_id)
        if slice_id is not None:
            return slice_id, group
    return None, None
