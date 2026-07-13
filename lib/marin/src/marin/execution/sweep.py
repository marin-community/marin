# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Race-to-claim sweep execution backed by the executor's ``step_lock``.

A *sweep* is a flat collection of independent targets — typically
hyper-parameter combinations — that need to be evaluated exactly once across a
pool of workers. N independent jobs run this library concurrently: every
worker walks the same target list in the same order and uses ``step_lock`` to
claim each target. Workers are otherwise uncoordinated; ``step_lock``'s
heartbeat handles dead claimants and its STATUS_SUCCESS check makes
already-completed targets no-ops.

**Multi-host gangs.** On a gang-scheduled TPU slice (``v5p-32`` …) fray runs
this entrypoint on *every* host, yet the run is a single ``jax.distributed``
gang that needs all hosts to enter ``run_fn`` together. Letting every host race
the same ``step_lock`` would hand the target to one host while the rest spin in
the lock loop, and the gang would never assemble. So within a gang only the
**leader** (Iris task 0) touches the lock and *decides* which target the gang
runs; **followers** (tasks 1..N-1) block on the leader's decision and run
``run_fn`` alongside it (``run_fn`` is itself the cross-host barrier). The
leader/follower split lives in ``sweep_coordination``. A single-host (or
non-Iris) worker simply leads a gang of one — it runs the loop alone, and its
announcements are inert because no followers exist.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from rigging.filesystem import prefix_join

from marin.execution.step_status import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    StepAlreadyDone,
    step_lock,
)
from marin.execution.sweep_coordination import GangRole, SweepCoordinator, gang_coordinator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepTarget:
    """One unit of work in a sweep.

    Attributes:
        target_id: Unique identifier within the sweep. Used as the per-target
            output directory name (under ``sweep_root``) and as the human
            readable lock label. Must be filesystem-safe.
        config: Opaque payload passed to ``run_fn``. The sweep library does not
            introspect this — it is up to ``run_fn`` to interpret it.
    """

    target_id: str
    config: Any


def claim_and_run(
    sweep_root: str,
    targets: Iterable[SweepTarget],
    run_fn: Callable[[SweepTarget], None],
    *,
    coordinator: SweepCoordinator | None = None,
) -> None:
    """Run a sweep worker: claim targets and run them, gang-aware.

    Per-target output path is ``f"{sweep_root}/{target.target_id}"``. The
    behaviour depends on this process's :class:`GangRole` (resolved from Iris
    job metadata via ``coordinator``):

    - **Leader** (task 0, or any single-host worker) — iterate ``targets`` and,
      for each, enter ``step_lock(target_path, target.target_id)``:

      - If a peer already wrote ``STATUS_SUCCESS``, ``step_lock`` raises
        ``StepAlreadyDone`` and we silently move on (and, as leader, publish
        nothing — followers never see this target).
      - Otherwise we hold the lock (with heartbeat refresh) while we call
        ``run_fn(target)`` and write ``STATUS_SUCCESS`` on completion. As
        leader we first announce the claimed target so followers run it too.
      - If ``run_fn`` raises, we write ``STATUS_FAILED`` and let the exception
        propagate. The worker stops iterating; remaining unclaimed targets are
        left for peers. Because ``step_lock`` defaults to
        ``force_run_failed=True``, a peer that later reaches the failed target
        retries it.

    - **Follower** — never touches the lock. Blocks on the leader's announced
      rounds and calls ``run_fn`` for each, until the leader signals stop.

    All leaders iterate ``targets`` in the same order. Across gangs,
    coordination is pure-races on the lock file (one leader per gang); with N
    leaders and M targets each grabs roughly M/N targets.

    Args:
        sweep_root: Directory (local or fsspec URL) under which per-target
            status/lock files live.
        targets: Iterable of ``SweepTarget``s. Workers must be given the same
            list in the same order.
        run_fn: Called once per claimed/announced target. Must be idempotent
            enough that a retry after a crashed peer does not corrupt prior
            state.
        coordinator: Leader/follower channel for the gang. Defaults to one
            derived from Iris job metadata; injected in tests.
    """
    targets = list(targets)
    coordinator = coordinator or gang_coordinator()
    try:
        if coordinator.role is GangRole.FOLLOWER:
            _follow(targets, run_fn, coordinator)
        else:
            _lead(sweep_root, targets, run_fn, coordinator)
    finally:
        coordinator.close()


def _lead(
    sweep_root: str,
    targets: list[SweepTarget],
    run_fn: Callable[[SweepTarget], None],
    coordinator: SweepCoordinator,
) -> None:
    """Claim targets via ``step_lock`` and announce each claimed round.

    ``coordinator.publish`` is inert when there are no followers (a single-host
    gang), so this is the original claim loop plus a per-round announcement for
    any gang followers. After the final stop round we wait for followers to
    consume it, so the leader's actor is not torn down out from under them.
    """
    seq = 0
    for target in targets:
        target_path = prefix_join(sweep_root, target.target_id)
        try:
            with step_lock(target_path, target.target_id) as status_file:
                # Announce before training so the gang enters run_fn together;
                # run_fn is the cross-host barrier between rounds.
                coordinator.publish(seq, target.target_id)
                seq += 1
                try:
                    run_fn(target)
                except Exception:
                    status_file.write_status(STATUS_FAILED)
                    raise
                status_file.write_status(STATUS_SUCCESS)
        except StepAlreadyDone:
            logger.info("Skipping %s: already completed by a peer worker", target.target_id)
    coordinator.publish(seq, None)
    coordinator.wait_for_followers(seq)


def _follow(
    targets: list[SweepTarget],
    run_fn: Callable[[SweepTarget], None],
    coordinator: SweepCoordinator,
) -> None:
    """Mirror the leader: run each announced target until the stop signal.

    Followers never touch ``step_lock`` — they act only on rounds the leader
    actually claimed, so already-completed targets (which the leader skips) are
    never run here.
    """
    targets_by_id = {target.target_id: target for target in targets}
    seq = 0
    while True:
        target_id = coordinator.receive(seq)
        if target_id is None:
            return
        seq += 1
        run_fn(targets_by_id[target_id])
