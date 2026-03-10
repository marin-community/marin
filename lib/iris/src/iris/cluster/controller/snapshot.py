# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint restore helpers for autoscaler state.

The controller runtime state is checkpointed as a SQLite DB copy. This module
only contains autoscaler reconciliation helpers used when restoring from that DB.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from iris.cluster.controller.autoscaler import TrackedWorker
from iris.cluster.controller.scaling_group import SliceLifecycleState, SliceState, _zones_from_config
from iris.cluster.platform.base import CloudWorkerState, CommandResult, Labels, Platform, SliceHandle, WorkerStatus
from iris.rpc import config_pb2, snapshot_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)


class _RestoredWorkerHandle:
    """Minimal handle placeholder used for restored tracked workers."""

    def __init__(self, worker_id: str, internal_address: str) -> None:
        self._worker_id = worker_id
        self._internal_address = internal_address

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def vm_id(self) -> str:
        return self._worker_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    @property
    def bootstrap_log(self) -> str:
        return ""

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        raise NotImplementedError("RestoredWorkerHandle does not support run_command")

    def reboot(self) -> None:
        raise NotImplementedError("RestoredWorkerHandle does not support reboot")


@dataclass
class ScalingGroupRestoreResult:
    """Result of restoring a single scaling group from checkpoint metadata."""

    slices: dict[str, SliceState] = field(default_factory=dict)
    consecutive_failures: int = 0
    backoff_active: bool = False
    quota_exceeded_active: bool = False
    quota_reason: str = ""
    discarded_count: int = 0
    adopted_count: int = 0
    last_scale_up: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    last_scale_down: Timestamp = field(default_factory=lambda: Timestamp.from_ms(0))
    backoff_until: Deadline | None = None
    quota_exceeded_until: Deadline | None = None


def _restore_tracked_worker(snap: snapshot_pb2.TrackedWorkerSnapshot) -> TrackedWorker:
    handle = _RestoredWorkerHandle(worker_id=snap.worker_id, internal_address=snap.internal_address)
    return TrackedWorker(
        worker_id=snap.worker_id,
        slice_id=snap.slice_id,
        scale_group=snap.scale_group,
        handle=handle,
    )


def restore_tracked_workers(proto: snapshot_pb2.ControllerSnapshot) -> dict[str, TrackedWorker]:
    """Restore tracked workers from checkpoint metadata."""
    workers: dict[str, TrackedWorker] = {}
    for tw_snap in proto.tracked_workers:
        tw = _restore_tracked_worker(tw_snap)
        workers[tw.worker_id] = tw
    return workers


def _wall_clock_to_deadline(wall_clock_ts: Timestamp) -> Deadline | None:
    """Convert a wall-clock timestamp from checkpoint into a deadline."""
    if wall_clock_ts.epoch_ms() == 0:
        return None
    return Deadline.after(wall_clock_ts, Duration.from_ms(0))


def restore_scaling_group(
    group_snapshot: snapshot_pb2.ScalingGroupSnapshot,
    platform: Platform,
    config: config_pb2.ScaleGroupConfig,
    label_prefix: str,
) -> ScalingGroupRestoreResult:
    """Reconcile checkpointed group slices against live cloud slices."""
    labels = Labels(label_prefix)
    filter_labels = {labels.iris_scale_group: group_snapshot.name}

    zones = _zones_from_config(config)
    cloud_handles = platform.list_slices(zones=zones, labels=filter_labels)
    cloud_by_id: dict[str, SliceHandle] = {h.slice_id: h for h in cloud_handles}
    checkpoint_slices = {s.slice_id: s for s in group_snapshot.slices}

    result = ScalingGroupRestoreResult()
    result.consecutive_failures = group_snapshot.consecutive_failures

    for slice_id, slice_snap in checkpoint_slices.items():
        cloud_handle = cloud_by_id.get(slice_id)
        if cloud_handle is None:
            logger.info("Scaling group %s: discarding slice %s (missing from cloud)", group_snapshot.name, slice_id)
            result.discarded_count += 1
            continue

        try:
            lifecycle = SliceLifecycleState(slice_snap.lifecycle)
        except ValueError:
            logger.warning(
                "Scaling group %s: unknown lifecycle %r for slice %s, defaulting to BOOTING",
                group_snapshot.name,
                slice_snap.lifecycle,
                slice_id,
            )
            lifecycle = SliceLifecycleState.BOOTING

        result.slices[slice_id] = SliceState(
            handle=cloud_handle,
            lifecycle=lifecycle,
            vm_addresses=list(slice_snap.vm_addresses),
            last_active=Timestamp.from_proto(slice_snap.last_active),
            error_message=slice_snap.error_message,
        )

    for slice_id, cloud_handle in cloud_by_id.items():
        if slice_id in checkpoint_slices:
            continue
        logger.info("Scaling group %s: adopting unknown cloud slice %s as BOOTING", group_snapshot.name, slice_id)
        result.slices[slice_id] = SliceState(handle=cloud_handle, lifecycle=SliceLifecycleState.BOOTING)
        result.adopted_count += 1

    if group_snapshot.HasField("backoff_until") and group_snapshot.backoff_until.epoch_ms > 0:
        backoff_ts = Timestamp.from_proto(group_snapshot.backoff_until)
        result.backoff_until = _wall_clock_to_deadline(backoff_ts)
        result.backoff_active = result.backoff_until is not None and not result.backoff_until.expired()

    if group_snapshot.HasField("quota_exceeded_until") and group_snapshot.quota_exceeded_until.epoch_ms > 0:
        quota_ts = Timestamp.from_proto(group_snapshot.quota_exceeded_until)
        result.quota_exceeded_until = _wall_clock_to_deadline(quota_ts)
        result.quota_exceeded_active = (
            result.quota_exceeded_until is not None and not result.quota_exceeded_until.expired()
        )
        result.quota_reason = group_snapshot.quota_reason

    if group_snapshot.HasField("last_scale_up") and group_snapshot.last_scale_up.epoch_ms > 0:
        result.last_scale_up = Timestamp.from_proto(group_snapshot.last_scale_up)
    if group_snapshot.HasField("last_scale_down") and group_snapshot.last_scale_down.epoch_ms > 0:
        result.last_scale_down = Timestamp.from_proto(group_snapshot.last_scale_down)

    logger.info(
        "Restored scaling group %s: %d slices (%d discarded, %d adopted), consecutive_failures=%d, "
        "backoff_active=%s, quota_exceeded=%s",
        group_snapshot.name,
        len(result.slices),
        result.discarded_count,
        result.adopted_count,
        result.consecutive_failures,
        result.backoff_active,
        result.quota_exceeded_active,
    )
    return result
