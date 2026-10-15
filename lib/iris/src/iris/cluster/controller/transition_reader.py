# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loading a :class:`TransitionSnapshot` from an open write transaction."""

from collections.abc import Iterable
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.db import Tx
from iris.cluster.controller.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile.snapshot import TransitionSnapshot
from iris.cluster.types import AttemptUid, JobName, WorkerId


@dataclass(frozen=True)
class TxTransitionReader:
    """A :class:`~iris.cluster.controller.reconcile.loader.TransitionReader` that loads
    its snapshot from an open write transaction.

    Effects authored from it commit into the same transaction they were read from,
    so no concurrent write can land between the read and the flush.
    """

    cur: Tx

    def transition_snapshot(
        self,
        *,
        now: Timestamp,
        seed_worker_ids: Iterable[WorkerId] = (),
        observation_uids: Iterable[AttemptUid] = (),
        seed_task_ids: Iterable[JobName] = (),
        extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
    ) -> TransitionSnapshot:
        return load_closed_snapshot(
            self.cur,
            now=now,
            seed_worker_ids=seed_worker_ids,
            observation_uids=observation_uids,
            seed_task_ids=seed_task_ids,
            extra_attempt_keys=extra_attempt_keys,
        )
