# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile kernel: a pure state machine over a closed snapshot, plus thin I/O.

The package surface is the **pure** kernel API — importing it pulls in no
``db`` / ``schema`` / ``projections``, so a pure submodule (``task``, ``job``,
``peers``, ``overlay``) stays import-clean even though Python evaluates this
``__init__`` first:

* :func:`load_closed_snapshot`'s output, :class:`TransitionSnapshot`, is built
  by the loader; the neutral inputs callers construct are :class:`TaskUpdate`,
  :class:`TerminalDecision`, :class:`TerminalKind`.
* :class:`ReconcileState` is the batch facade; ``.open(snapshot).<verb>(...)``
  runs one operation and returns a :class:`ControllerEffects`.

The two I/O boundary helpers are intentionally NOT re-exported here so this
surface stays pure — the I/O command layer imports them from their own modules:

* ``load_closed_snapshot`` from :mod:`iris.cluster.controller.reconcile.loader`
* ``commit_effects`` from :mod:`iris.cluster.controller.reconcile.commit`
"""

from iris.cluster.controller.reconcile.batches import ReconcileState
from iris.cluster.controller.reconcile.effects import ControllerEffects, DirectTransitionResult
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind

__all__ = [
    "ControllerEffects",
    "DirectTransitionResult",
    "ReconcileState",
    "TaskUpdate",
    "TerminalDecision",
    "TerminalKind",
]
