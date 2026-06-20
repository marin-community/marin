# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror in-memory autoscaler state into the controller DB.

The autoscaler tracks slices and groups entirely in memory. After each capacity
call the controller calls :func:`persist_autoscaler_state` to sync the
``slices`` / ``scaling_groups`` tables to match the returned
:class:`~iris.cluster.controller.autoscaler.state.AutoscalerState`: upsert every
present row and delete any slice row no longer tracked. Slice counts are small
and the cadence is ~10s, so a wholesale upsert is cheap and avoids threading
incremental deltas through every autoscaler mutation.
"""

from rigging.timing import Timestamp
from sqlalchemy import delete
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.controller.autoscaler.state import AutoscalerState
from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import scaling_groups_table, slices_table


def persist_autoscaler_state(cur: Tx, state: AutoscalerState) -> None:
    """Sync the ``slices`` / ``scaling_groups`` tables to match ``state``.

    Upserts every group and slice row in ``state`` and deletes slice rows for
    the tracked groups whose slices are no longer present. Deletion is scoped to
    the groups in ``state``: an empty state (a backend with no autoscaler) is a
    no-op rather than wiping every row, and rows belonging to *abandoned* groups
    — dropped from config with their VMs already gone, so never re-adopted here —
    are left for the slice pruner to garbage-collect.

    Runs in the caller's write transaction (``cur``) so the control tick can
    fold this persistence into its single end-of-tick commit.
    """
    if not state.groups:
        return

    now_ms = Timestamp.now().epoch_ms()
    group_names = [g.name for g in state.groups]
    present_slice_ids = [s.slice_id for s in state.slices]

    for group in state.groups:
        cur.execute(
            sqlite_insert(scaling_groups_table)
            .values(
                name=group.name,
                last_scale_up_ms=group.last_scale_up_ms,
                last_scale_down_ms=group.last_scale_down_ms,
                updated_at_ms=now_ms,
            )
            .on_conflict_do_update(
                index_elements=["name"],
                set_={
                    "last_scale_up_ms": group.last_scale_up_ms,
                    "last_scale_down_ms": group.last_scale_down_ms,
                    "updated_at_ms": now_ms,
                },
            )
        )

    for slice_row in state.slices:
        cur.execute(
            sqlite_insert(slices_table)
            .values(
                slice_id=slice_row.slice_id,
                scale_group=slice_row.scale_group,
                lifecycle=slice_row.lifecycle,
                worker_ids=list(slice_row.worker_ids),
                created_at_ms=slice_row.created_at_ms,
                error_message=slice_row.error_message,
            )
            .on_conflict_do_update(
                index_elements=["slice_id"],
                set_={
                    "scale_group": slice_row.scale_group,
                    "lifecycle": slice_row.lifecycle,
                    "worker_ids": list(slice_row.worker_ids),
                    "created_at_ms": slice_row.created_at_ms,
                    "error_message": slice_row.error_message,
                },
            )
        )

    # Drop tracked-group slice rows that are no longer present. Scoped to
    # group_names, so abandoned-group rows fall to the pruner, not here.
    cur.execute(
        delete(slices_table).where(
            slices_table.c.scale_group.in_(group_names),
            slices_table.c.slice_id.notin_(present_slice_ids),
        )
    )
