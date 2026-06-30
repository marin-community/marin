# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller-DB transition-reader helpers shared by the backends.

:func:`load_transition_snapshot` opens a closed control read snapshot a backend
authors its task projection from without touching the controller database directly;
:class:`DbTransitionReader` wraps it for a placement-owning backend (one with no
worker store) so it reads its dispatch drain through the same seam.
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
    """Open a control read snapshot and close it over the seeded entities.

    The read-only surface a backend authors its task projection through: a closed
    snapshot rather than the tick's write transaction, so the backend never touches
    the controller database directly.
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
    """A controller-DB-backed :class:`~...reconcile.loader.TransitionReader`.

    Gives a placement-owning backend (one with no worker store) a read snapshot to
    author its dispatch effects from, without handing it the DB.
    """

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
