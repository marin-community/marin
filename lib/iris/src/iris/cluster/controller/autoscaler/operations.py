# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operational helpers for autoscaler worker and slice actions."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rigging.timing import Duration, Timestamp

from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.autoscaler.slice_lifecycle import SliceEvent, SliceSideEffectKind
from iris.cluster.controller.db import ControllerDB
from iris.cluster.providers.gcp.bootstrap import build_worker_bootstrap_script
from iris.cluster.providers.types import SliceHandle
from iris.rpc import config_pb2, vm_pb2

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


def restart_worker(
    groups: dict[str, ScalingGroup],
    db: ControllerDB | None,
    worker_id: str,
    build_worker_config: Callable[[ScalingGroup], config_pb2.WorkerConfig | None],
) -> None:
    """Restart a worker with a fresh bootstrap script using the latest image."""

    if db is None:
        raise ValueError("No DB configured — cannot look up worker")

    with db.read_snapshot() as snapshot:
        rows = snapshot.raw(
            "SELECT slice_id, scale_group FROM workers WHERE worker_id = ? AND slice_id != ''",
            params=(worker_id,),
        )
    if not rows:
        raise ValueError(f"Worker {worker_id} not found in workers table (or has no slice_id)")
    row = rows[0]

    group = groups.get(row.scale_group)
    if group is None:
        raise ValueError(f"Scale group {row.scale_group} not found for worker {worker_id}")

    slice_handle = group.get_slice(row.slice_id)
    if slice_handle is None:
        raise ValueError(f"Slice {row.slice_id} not found in group {row.scale_group}")

    workers = slice_handle.describe().workers
    handle = next((worker for worker in workers if worker.worker_id == worker_id), None)
    if handle is None:
        raise ValueError(f"Worker {worker_id} not found in slice {row.slice_id}")

    worker_config = build_worker_config(group)
    if worker_config is None:
        raise ValueError("No base worker config — cannot build bootstrap script")

    worker_config.worker_id = worker_id
    worker_config.slice_id = row.slice_id
    handle.restart_worker(build_worker_bootstrap_script(worker_config))


def terminate_slices_for_workers(
    groups: dict[str, ScalingGroup],
    worker_ids: Sequence[str],
    unregister_slice_workers: Callable[[str, Sequence[str] | None], None],
    log_action: Callable[..., vm_pb2.AutoscalerAction],
    timestamp: Timestamp,
    short_lived_slice_threshold: Duration,
) -> SliceTerminationResult:
    """Detach and schedule slice termination for the given failed workers."""

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

        # Determine if short-lived for backoff tracking
        slice_handle = group.get_slice(slice_id)
        is_short_lived = False
        if slice_handle is not None:
            age_ms = timestamp.epoch_ms() - slice_handle.created_at.epoch_ms()
            is_short_lived = Duration.from_ms(age_ms) < short_lived_slice_threshold

        result = group.dispatch(
            slice_id,
            SliceEvent.WORKER_FAILURE_REPORTED,
            {"failed_workers": failed_workers, "is_short_lived": is_short_lived},
            now=timestamp,
        )

        # Execute side effects
        if result is not None:
            for effect in result.side_effects:
                if effect.kind == SliceSideEffectKind.RECORD_GROUP_FAILURE:
                    group.record_failure(timestamp)
                    log_action(
                        "backoff_triggered",
                        group.name,
                        slice_id=slice_id,
                        reason=f"short-lived slice (age={age_ms}ms)",
                    )

        # Detach and schedule termination regardless of dispatch result
        handle = group.detach_slice(slice_id)
        unregister_slice_workers(slice_id, worker_ids=slice_worker_ids)
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
