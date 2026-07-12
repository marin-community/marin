# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loading a :class:`TransitionSnapshot` from the controller database.

:func:`load_transition_snapshot` is the function form; :class:`DbTransitionReader`
adapts it to the :class:`~iris.cluster.controller.reconcile.loader.TransitionReader`
interface for a backend that has no worker store of its own.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile.snapshot import TransitionSnapshot
from iris.cluster.types import AttemptUid, JobName, WorkerId


def load_transition_snapshot(
    db: ControllerDB,
    *,
    now: Timestamp,
    seed_worker_ids: Iterable[WorkerId] = (),
    observation_uids: Iterable[AttemptUid] = (),
    seed_task_ids: Iterable[JobName] = (),
    extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
) -> TransitionSnapshot:
    """Load a :class:`TransitionSnapshot` over a closed control read snapshot.

    The ``seed_*``, ``observation_uids``, and ``extra_attempt_keys`` arguments pull the
    named workers, tasks, attempts, and observed attempt-uids into the snapshot.
    """
    with db.control_read_snapshot() as snap:
        return load_closed_snapshot(
            snap,
            now=now,
            seed_worker_ids=seed_worker_ids,
            observation_uids=observation_uids,
            seed_task_ids=seed_task_ids,
            extra_attempt_keys=extra_attempt_keys,
        )


@dataclass(frozen=True)
class DbTransitionReader:
    """A :class:`~iris.cluster.controller.reconcile.loader.TransitionReader` that loads
    its snapshot from the controller database."""

    db: ControllerDB

    def transition_snapshot(
        self,
        *,
        now: Timestamp,
        seed_worker_ids: Iterable[WorkerId] = (),
        observation_uids: Iterable[AttemptUid] = (),
        seed_task_ids: Iterable[JobName] = (),
        extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
    ) -> TransitionSnapshot:
        return load_transition_snapshot(
            self.db,
            now=now,
            seed_worker_ids=seed_worker_ids,
            observation_uids=observation_uids,
            seed_task_ids=seed_task_ids,
            extra_attempt_keys=extra_attempt_keys,
        )
