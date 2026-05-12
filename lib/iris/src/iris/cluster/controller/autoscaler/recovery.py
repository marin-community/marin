# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler checkpoint restore helpers."""

from __future__ import annotations

import json
import logging
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
from iris.cluster.providers.types import SliceHandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoscalerCheckpoint:
    """Checkpointed autoscaler state loaded from the controller DB."""

    group_snapshots: dict[str, GroupSnapshot]
    tracked_worker_rows: list[TrackedWorkerRow]


def load_autoscaler_checkpoint(db: ControllerDB) -> AutoscalerCheckpoint:
    """Load autoscaler state from the controller DB."""
    with db.read_snapshot() as tx:
        scaling_rows = tx.execute(
            select(
                scaling_groups_table.c.name,
                scaling_groups_table.c.consecutive_failures,
                scaling_groups_table.c.backoff_until_ms,
                scaling_groups_table.c.last_scale_up_ms,
                scaling_groups_table.c.last_scale_down_ms,
                scaling_groups_table.c.quota_exceeded_until_ms,
                scaling_groups_table.c.quota_reason,
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
                worker_ids=json.loads(row.worker_ids) if row.worker_ids else [],
                created_at_ms=int(row.created_at_ms),
                error_message=row.error_message,
            )
        )

    group_snapshots: dict[str, GroupSnapshot] = {}
    for row in scaling_rows:
        group_snapshots[row.name] = GroupSnapshot(
            name=row.name,
            slices=slices_by_group.get(row.name, []),
            consecutive_failures=int(row.consecutive_failures),
            backoff_until_ms=int(row.backoff_until_ms),
            last_scale_up_ms=int(row.last_scale_up_ms),
            last_scale_down_ms=int(row.last_scale_down_ms),
            quota_exceeded_until_ms=int(row.quota_exceeded_until_ms),
            quota_reason=row.quota_reason,
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

    all_cloud_slices = platform.list_all_slices()
    cloud_by_group: dict[str, list[SliceHandle]] = {}
    for handle in all_cloud_slices:
        cloud_by_group.setdefault(handle.scale_group, []).append(handle)

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
            consecutive_failures=restore_result.consecutive_failures,
            last_scale_up=restore_result.last_scale_up,
            last_scale_down=restore_result.last_scale_down,
            backoff_until=restore_result.backoff_until,
            quota_exceeded_until=restore_result.quota_exceeded_until,
            quota_reason=restore_result.quota_reason,
        )

    return restore_tracked_workers(checkpoint.tracked_worker_rows)
