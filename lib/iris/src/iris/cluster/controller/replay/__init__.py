# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Event-replay testing system for ControllerTransitions.

This package defines a frozen ``IrisEvent`` union — one variant per public
mutation method on ``ControllerTransitions`` — together with a dispatcher,
a SQLite trace hook, a deterministic DB dump, and a curated set of
scenarios. Used both as a pytest golden suite and as a CLI for diffing
DB state across branches/checkpoints.
"""

from iris.cluster.controller.replay.db_dump import deterministic_dump
from iris.cluster.controller.replay.dispatcher import apply_event
from iris.cluster.controller.replay.events import (
    AddEndpoint,
    ApplyDirectProviderUpdates,
    ApplyHeartbeatsBatch,
    ApplyTaskUpdates,
    BufferDirectKill,
    CancelJob,
    CancelTasksForTimeout,
    DrainForDirectProvider,
    IrisEvent,
    MarkTaskUnschedulable,
    PreemptTask,
    QueueAssignments,
    RegisterOrRefreshWorker,
    RemoveEndpoint,
    RemoveFinishedJob,
    RemoveWorker,
    ReplaceReservationClaims,
    SubmitJob,
    UpdateWorkerPings,
)
from iris.cluster.controller.replay.scenarios import (
    SCENARIO_NAMES,
    SCENARIOS,
    run_scenario,
)
from iris.cluster.controller.replay.sql_trace import sql_tracing

__all__ = [
    "SCENARIOS",
    "SCENARIO_NAMES",
    "AddEndpoint",
    "ApplyDirectProviderUpdates",
    "ApplyHeartbeatsBatch",
    "ApplyTaskUpdates",
    "BufferDirectKill",
    "CancelJob",
    "CancelTasksForTimeout",
    "DrainForDirectProvider",
    "IrisEvent",
    "MarkTaskUnschedulable",
    "PreemptTask",
    "QueueAssignments",
    "RegisterOrRefreshWorker",
    "RemoveEndpoint",
    "RemoveFinishedJob",
    "RemoveWorker",
    "ReplaceReservationClaims",
    "SubmitJob",
    "UpdateWorkerPings",
    "apply_event",
    "deterministic_dump",
    "run_scenario",
    "sql_tracing",
]
