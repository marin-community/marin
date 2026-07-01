# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller-owned job-DAG fold over every backend's per-tick apply pass.

Each backend's ``reconcile()`` authors only its own workers' direct task/attempt
transitions (:class:`DirectTransitionResult`); :func:`fold_direct_results` is the
controller-side counterpart that recomputes/finalizes/cascades once per tick, over
the union of every backend's result, mirroring ``ops.job.cancel``'s controller-side
subtree kill.
"""

from collections.abc import Iterable

from rigging.timing import Timestamp

from iris.cluster.controller.db import Tx
from iris.cluster.controller.reconcile import ControllerEffects, DirectTransitionResult, ReconcileState
from iris.cluster.controller.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile.overlay import Overlay
from iris.cluster.types import JobName


def fold_direct_results(
    cur: Tx,
    results: Iterable[DirectTransitionResult],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Merge every backend's direct transitions into one ``ControllerEffects``.

    Loads a single closed snapshot over ``cur``, seeded by the union of every
    result's ``touched_jobs`` (which expands to their full descendant subtrees —
    the same closure ``ops.job.cancel`` loads for a subtree kill), builds one
    shared :class:`Overlay`, merges each backend's row deltas and pass-through
    effect categories into it, then runs the recompute pass once over the union.

    In the production control tick, ``cur`` is a read snapshot taken BEFORE the
    tick's write transaction opens (alongside each backend's own reconcile
    snapshot), so the fold never sees this tick's own schedule/assign writes —
    preserving today's per-backend reconcile timing.
    """
    results = list(results)
    if all(r.is_empty for r in results):
        return ControllerEffects()

    touched: list[JobName] = []
    touched_seen: set[JobName] = set()
    pending_child_cascades: dict[JobName, str] = {}
    for result in results:
        for job_id in result.touched_jobs:
            if job_id not in touched_seen:
                touched_seen.add(job_id)
                touched.append(job_id)
        for job_id, reason in result.pending_child_cascades.items():
            pending_child_cascades.setdefault(job_id, reason)

    snapshot = load_closed_snapshot(cur, now=now, seed_job_ids=touched)
    overlay = Overlay(snapshot)
    for result in results:
        for delta in result.tasks.values():
            overlay.merge_task(delta)
        for delta in result.attempts.values():
            overlay.merge_attempt(delta)
        overlay.effects.endpoint_deletions.extend(result.endpoint_deletions)
        overlay.effects.log_events.extend(result.log_events)

    ReconcileState.fold(overlay, touched, pending_child_cascades)
    return overlay.effects
