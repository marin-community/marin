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
"""

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from marin.execution.executor_step_status import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    StepAlreadyDone,
    step_lock,
)

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
) -> None:
    """Iterate ``targets``; race peers for each target's lock and run if claimed.

    Per-target output path is ``f"{sweep_root}/{target.target_id}"``. For each
    target we enter ``step_lock(target_path, target.target_id)``:

    - If a peer already wrote ``STATUS_SUCCESS`` for that target, ``step_lock``
      raises ``StepAlreadyDone`` and we silently move on.
    - Otherwise we hold the lock (with heartbeat refresh) while we call
      ``run_fn(target)`` and write ``STATUS_SUCCESS`` on completion.
    - If ``run_fn`` raises, we write ``STATUS_FAILED`` and let the exception
      propagate out of ``claim_and_run``. The current worker stops iterating;
      remaining unclaimed targets are left for peer workers. Because
      ``step_lock`` defaults to ``force_run_failed=True``, a peer that later
      reaches the failed target will retry it.

    All workers iterate ``targets`` in the same order. Coordination is
    pure-races on the lock file; with N workers and M targets each worker
    grabs roughly M/N targets.

    Args:
        sweep_root: Directory (local or fsspec URL) under which per-target
            status/lock files live.
        targets: Iterable of ``SweepTarget``s. Workers must be given the same
            list in the same order.
        run_fn: Called once per claimed target. Must be idempotent enough that
            a retry after a crashed peer does not corrupt prior state.
    """
    for target in targets:
        target_path = os.path.join(sweep_root, target.target_id)
        try:
            with step_lock(target_path, target.target_id) as status_file:
                try:
                    run_fn(target)
                except Exception:
                    status_file.write_status(STATUS_FAILED)
                    raise
                status_file.write_status(STATUS_SUCCESS)
        except StepAlreadyDone:
            logger.info("Skipping %s: already completed by a peer worker", target.target_id)
