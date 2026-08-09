# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Database-free snapshot reader contract for reconciliation backends."""

from collections.abc import Iterable
from typing import Protocol

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile.snapshot import TransitionSnapshot
from iris.cluster.types import AttemptUid, JobName, WorkerId


class TransitionReader(Protocol):
    """Yield one closed reconciliation snapshot from implementation-owned state."""

    def transition_snapshot(
        self,
        *,
        now: Timestamp,
        seed_worker_ids: Iterable[WorkerId] = (),
        observation_uids: Iterable[AttemptUid] = (),
        seed_task_ids: Iterable[JobName] = (),
        extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
    ) -> TransitionSnapshot:
        """Load a closed snapshot stamped with ``now`` and seeded by the named entities."""
        ...
