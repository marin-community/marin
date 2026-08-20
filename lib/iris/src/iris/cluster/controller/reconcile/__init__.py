# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan canonical state transitions from a closed controller snapshot.

``ReconcileState`` consumes a ``TransitionSnapshot`` plus task, worker, and peer
observations. Each operation returns ``ControllerEffects`` for the persistence
layer to commit atomically. Snapshot loading and effect commits live in
``controller.persistence.reconcile``; this package owns only transition policy.
"""

from iris.cluster.controller.reconcile.batches import ReconcileState
from iris.cluster.controller.reconcile.effects import ControllerEffects
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind

__all__ = [
    "ControllerEffects",
    "ReconcileState",
    "TaskUpdate",
    "TerminalDecision",
    "TerminalKind",
]
