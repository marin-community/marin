# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler checkpoint restore helpers."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from sqlalchemy import select

from iris.cluster.controller.autoscaler.scaling_group import (
    GroupSnapshot,
    ScalingGroup,
    SliceSnapshot,
    restore_scaling_group,
)
from iris.cluster.controller.autoscaler.worker_registry import TrackedWorker, TrackedWorkerRow, restore_tracked_workers
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import scaling_groups_table, slices_table, workers_table
from iris.cluster.providers.protocols import WorkerInfraProvider
from iris.cluster.providers.types import CloudSliceState, SliceHandle

_LIVE_CLOUD_STATES = frozenset({CloudSliceState.CREATING, CloudSliceState.READY, CloudSliceState.REPAIRING})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoscalerCheckpoint:
    """Checkpointed autoscaler state loaded from the controller DB."""

    group_snapshots: dict[str, GroupSnapshot]
    tracked_worker_rows: list[TrackedWorkerRow]


def load_autoscaler_checkpoint(db: ControllerDB) -> AutoscalerCheckpoint:
    """Load autoscaler state from the controller DB.

    Backoff/churn state is no longer persisted (it lives in the in-memory
    :class:`BackoffDetector`), so we only restore slice membership and
    informational scale timestamps from the DB.
    """
    with db.read_snapshot() as tx:
        scaling_rows = tx.execute(
            select(
                scaling_groups_table.c.name,
                scaling_groups_table.c.last_scale_up_ms,
                scaling_groups_table.c.last_scale_down_ms,
            )
        ).all()

        slice_rows = tx.execute(
            select(
                slices_table.c.slice_id,
                slices_table.c.scale_group,
                slices_table.c.lifecycle,
                slices_table.c.worker_ids,
                slices_table.c.created_at_ms,
                slices_table.c.error_message,
            )
        ).all()

        # Failed workers have their DB row deleted (writes.workers.remove_worker), so
        # surviving rows with a slice are by definition the live tracked set.
        tracked_rows = tx.execute(
            select(
                workers_table.c.worker_id,
                workers_table.c.slice_id,
                workers_table.c.scale_group,
                workers_table.c.address,
            ).where(workers_table.c.slice_id != "")
        ).all()

    slices_by_group: dict[str, list[SliceSnapshot]] = {}
    for row in slice_rows:
        slices_by_group.setdefault(row.scale_group, []).append(
            SliceSnapshot(
                slice_id=row.slice_id,
                scale_group=row.scale_group,
                lifecycle=row.lifecycle,
                worker_ids=row.worker_ids,
                created_at_ms=int(row.created_at_ms),
                error_message=row.error_message,
            )
        )

    group_snapshots: dict[str, GroupSnapshot] = {}
    for row in scaling_rows:
        group_snapshots[row.name] = GroupSnapshot(
            name=row.name,
            slices=slices_by_group.get(row.name, []),
            last_scale_up_ms=int(row.last_scale_up_ms),
            last_scale_down_ms=int(row.last_scale_down_ms),
        )

    tracked_worker_rows = [
        TrackedWorkerRow(
            worker_id=str(row.worker_id),
            slice_id=row.slice_id,
            scale_group=row.scale_group,
            address=str(row.address),
        )
        for row in tracked_rows
    ]
    return AutoscalerCheckpoint(group_snapshots=group_snapshots, tracked_worker_rows=tracked_worker_rows)


def restore_autoscaler_state(
    groups: dict[str, ScalingGroup],
    checkpoint: AutoscalerCheckpoint,
    platform: WorkerInfraProvider,
) -> dict[str, TrackedWorker]:
    """Restore scaling groups and tracked workers from a checkpoint."""

    cloud_by_group: dict[str, list[SliceHandle]] = {}
    for listed in platform.list_all_slices():
        if listed.state not in _LIVE_CLOUD_STATES:
            _reclaim_dead_slice(listed.handle, listed.state)
            continue
        cloud_by_group.setdefault(listed.handle.scale_group, []).append(listed.handle)

    for group_snapshot in checkpoint.group_snapshots.values():
        group = groups.get(group_snapshot.name)
        if group is None:
            logger.warning(
                "Checkpoint references scaling group %s which does not exist in config, skipping",
                group_snapshot.name,
            )
            continue
        restore_result = restore_scaling_group(
            group_snapshot=group_snapshot,
            cloud_handles=cloud_by_group.get(group_snapshot.name, []),
            label_prefix=group.label_prefix,
        )
        group.restore_from_snapshot(
            slices=restore_result.slices,
            last_scale_up=restore_result.last_scale_up,
            last_scale_down=restore_result.last_scale_down,
        )
        group.purge_persisted_slice_rows(restore_result.discarded_slice_ids)

    return restore_tracked_workers(checkpoint.tracked_worker_rows)


def _reclaim_dead_slice(handle: SliceHandle, state: CloudSliceState) -> None:
    """Best-effort terminate of a dead slice in a daemon thread.

    Boot recovery must not block on or fail because of a stale cloud resource:
    terminate() can hit transient API errors and is not guaranteed to be fast.
    Errors are logged; on the next restart the slice will surface again.
    """
    logger.info("Reclaiming dead slice %s (state=%s, zone=%s)", handle.slice_id, state, handle.zone)

    def _run() -> None:
        try:
            handle.terminate()
        except Exception as e:
            logger.warning("Failed to terminate dead slice %s: %s", handle.slice_id, e)

    threading.Thread(target=_run, name=f"reclaim-{handle.slice_id}", daemon=True).start()
