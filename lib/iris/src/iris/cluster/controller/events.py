# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Event types for controller state transitions.

All state changes flow through ControllerState.handle_event(), which dispatches
to handlers based on event type and logs actions to a transaction log for debugging.
"""

from dataclasses import dataclass, field
from typing import Any

from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms

# =============================================================================
# Typed Event Classes
# =============================================================================


@dataclass(frozen=True)
class Event:
    """Base class for controller state change events."""

    pass


@dataclass(frozen=True)
class WorkerRegisteredEvent(Event):
    """Worker registration or heartbeat update."""

    worker_id: WorkerId
    address: str
    metadata: cluster_pb2.WorkerMetadata
    timestamp_ms: int


@dataclass(frozen=True)
class WorkerHeartbeatEvent(Event):
    """Worker heartbeat timestamp update."""

    worker_id: WorkerId
    timestamp_ms: int


@dataclass(frozen=True)
class WorkerHeartbeatFailedEvent(Event):
    """Single heartbeat failure for a worker. State layer tracks consecutive count."""

    worker_id: WorkerId
    error: str


@dataclass(frozen=True)
class WorkerFailedEvent(Event):
    """Worker marked as failed."""

    worker_id: WorkerId
    error: str | None = None


@dataclass(frozen=True)
class JobSubmittedEvent(Event):
    """New job submission."""

    job_id: JobId
    request: cluster_pb2.Controller.LaunchJobRequest
    timestamp_ms: int


@dataclass(frozen=True)
class JobCancelledEvent(Event):
    """Job termination request."""

    job_id: JobId
    reason: str


@dataclass(frozen=True)
class TaskAssignedEvent(Event):
    """Task assigned to worker (creates attempt, commits resources)."""

    task_id: TaskId
    worker_id: WorkerId


@dataclass(frozen=True)
class TaskStateChangedEvent(Event):
    """Task state transition."""

    task_id: TaskId
    new_state: int  # cluster_pb2.TaskState
    attempt_id: int
    error: str | None = None
    exit_code: int | None = None


# =============================================================================
# Transaction Logging
# =============================================================================


@dataclass
class Action:
    """Single action taken during event handling."""

    timestamp_ms: int
    action: str
    entity_id: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransactionLog:
    """Records actions from handling one event.

    Calls to handle_event() produce a TransactionLog that captures all
    the actions taken during that event's processing. This includes both
    the direct effects of the event and any cascading effects (e.g., worker
    failure causing task failures).
    """

    event: Event
    timestamp_ms: int = field(default_factory=now_ms)
    actions: list[Action] = field(default_factory=list)
    tasks_to_kill: set[TaskId] = field(default_factory=set)

    def log(self, action: str, entity_id: str, **details: Any) -> None:
        """Record an action taken during event handling."""
        self.actions.append(
            Action(
                timestamp_ms=now_ms(),
                action=action,
                entity_id=str(entity_id),
                details=details,
            )
        )
