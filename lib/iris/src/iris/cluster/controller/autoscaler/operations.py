# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operational helpers for autoscaler worker and slice actions."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.platforms.types import SliceHandle
from iris.rpc import vm_pb2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SliceTerminationRequest:
    """A detached slice handle scheduled for termination."""

    slice_id: str
    group: ScalingGroup
    handle: SliceHandle


@dataclass(frozen=True)
class SliceTerminationResult:
    """Batch termination work derived from a set of failed workers."""

    sibling_worker_ids: list[str]
    termination_requests: list[SliceTerminationRequest]


def terminate_slices_for_workers(
    groups: dict[str, ScalingGroup],
    worker_ids: Sequence[str],
    unregister_slice_workers: Callable[[str, Sequence[str] | None], None],
    log_action: Callable[..., vm_pb2.AutoscalerAction],
    timestamp: Timestamp,
) -> SliceTerminationResult:
    """Detach and schedule slice termination for the given failed workers.

    Every observed slice termination is reported to the group's churn detector
    as :class:`~iris.cluster.controller.autoscaler.backoff_detector.SliceFate.PREEMPTED`.
    The detector classifies internally based on slice age — short-lived deaths
    move churn rate; long-lived deaths count as positive samples.
    """

    if not worker_ids:
        return SliceTerminationResult(sibling_worker_ids=[], termination_requests=[])

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
        failed_workers = sorted(primary_workers & set(slice_worker_ids))

        logger.info("Workers %s triggered slice termination for %s", failed_workers, slice_id)
        log_action(
            "worker_failed",
            group.name,
            slice_id=slice_id,
            reason=f"workers failed: {', '.join(failed_workers)}",
        )
        group.record_slice_preempted(slice_id, timestamp)
        handle = group.detach_slice(slice_id)
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
