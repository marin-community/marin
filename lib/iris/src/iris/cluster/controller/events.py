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
to handlers and logs actions to a transaction log for debugging.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms


class EventType(Enum):
    """Discriminator for controller state change events."""

    # Worker lifecycle
    WORKER_REGISTERED = auto()
    WORKER_HEARTBEAT = auto()
    WORKER_FAILED = auto()

    # Job lifecycle
    JOB_SUBMITTED = auto()
    JOB_CANCELLED = auto()

    # Task lifecycle
    TASK_ASSIGNED = auto()
    TASK_RUNNING = auto()
    TASK_SUCCEEDED = auto()
    TASK_FAILED = auto()
    TASK_KILLED = auto()
    TASK_WORKER_FAILED = auto()


@dataclass(frozen=True)
class Event:
    """All state change events. Fields are optional based on event_type.

    This is a single event class with optional fields rather than a hierarchy
    of event subclasses. The event_type discriminator indicates which fields
    are relevant for each event.

    Examples:
        Event(EventType.WORKER_FAILED, worker_id=worker_id, error="Connection lost")
        Event(EventType.TASK_SUCCEEDED, task_id=task_id, exit_code=0)
        Event(EventType.TASK_WORKER_FAILED, task_id=task_id, worker_id=worker_id, error="Worker died")
    """

    event_type: EventType

    # Entity IDs (use whichever are relevant)
    task_id: TaskId | None = None
    worker_id: WorkerId | None = None
    job_id: JobId | None = None

    # Event data
    error: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    timestamp_ms: int | None = None

    # For WORKER_REGISTERED
    address: str | None = None
    metadata: cluster_pb2.WorkerMetadata | None = None

    # For JOB_SUBMITTED
    request: cluster_pb2.Controller.LaunchJobRequest | None = None


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

    Each call to handle_event() creates a TransactionLog that captures all
    the actions taken during that event's processing. This includes both
    the direct effects of the event and any cascading effects (e.g., worker
    failure causing task failures).
    """

    event: Event
    timestamp_ms: int = field(default_factory=now_ms)
    actions: list[Action] = field(default_factory=list)

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
