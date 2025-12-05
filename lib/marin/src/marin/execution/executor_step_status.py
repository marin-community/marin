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

"""
Each `ExecutorStep` produces an `output_path`.
We associate each `output_path` with a `output_path/executor_status` file that contains a
list of events corresponding to that step.  For example:

    {"date": "2024-09-28T13:29:20.780705", "status": "WAITING", "message": null}
    {"date": "2024-09-28T13:29:21.091470", "status": "RUNNING", "message": null}
    {"date": "2024-09-28T13:29:47.559614", "status": "SUCCESS", "message": null}

This allows us to track both the status of each step as well as the time spent
on each step.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import fsspec

from marin.utils import fsspec_exists

# Old plain-text status values that may exist in legacy files
OLD_STATUS_VALUES = {"RUNNING", "FAILED", "SUCCESS", "WAITING", "DEP_FAILED", "UNKNOWN", "CANCELLED"}

# Heartbeat configuration for distributed locking
HEARTBEAT_INTERVAL = 30  # seconds between heartbeat updates
HEARTBEAT_TIMEOUT = 90  # seconds before considering a heartbeat stale

STATUS_WAITING = "WAITING"  # Waiting for dependencies to finish
STATUS_RUNNING = "RUNNING"
STATUS_FAILED = "FAILED"
STATUS_SUCCESS = "SUCCESS"
STATUS_DEP_FAILED = "DEP_FAILED"  # Dependency failed
STATUS_UNKNOWN = "UNKNOWN"  # Unknown status, Ray failed to return the status
STATUS_CANCELLED = "CANCELLED"  # Job was cancelled by user


@dataclass(frozen=True)
class ExecutorStepEvent:
    """Represents a change in the status of an `ExecutorStep`."""

    date: str
    """When the `status` changed."""

    status: str
    """Represents the `status` of the job."""

    message: str | None = None
    """An optional message to provide more context (especially for errors)."""

    task_id: str | None = None
    """The task ID associated with executing this step."""


def get_status_path(output_path: str) -> str:
    """Return the `path` of the status file associated with `output_path`, which contains a list of events."""
    return os.path.join(output_path, ".executor_status")


def read_events(path: str) -> list[ExecutorStepEvent]:
    """Reads the status events from `path`.

    Handles both old plain-text status files (e.g., just "RUNNING") and new JSON-lines format.
    """
    events = []
    if fsspec_exists(path):
        with fsspec.open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("{"):
                    # New JSON format
                    data = json.loads(line)
                    # Handle old ray_task_id field
                    if "ray_task_id" in data and "task_id" not in data:
                        data["task_id"] = data.pop("ray_task_id")
                    events.append(ExecutorStepEvent(**data))
                elif line in OLD_STATUS_VALUES:
                    # Old plain-text format - treat as stale (epoch timestamp)
                    events.append(
                        ExecutorStepEvent(
                            date="1970-01-01T00:00:00+00:00",
                            status=line,
                        )
                    )
    return events


def get_current_status(events: list[ExecutorStepEvent]) -> str | None:
    """Get the most recent status (last event)."""
    return events[-1].status if len(events) > 0 else None


def is_failure(status: str):
    return status in [STATUS_FAILED, STATUS_DEP_FAILED]


def is_running_or_waiting(status: str):
    return status in [STATUS_WAITING, STATUS_RUNNING]


def _is_timestamp_stale(date_str: str) -> bool:
    """Check if timestamp is older than HEARTBEAT_TIMEOUT seconds.

    Always uses UTC. Treats naive timestamps as UTC.
    """
    try:
        event_time = datetime.fromisoformat(date_str)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - event_time).total_seconds() > HEARTBEAT_TIMEOUT
    except (ValueError, TypeError):
        return True


class StatusFile:
    """Manages an executor step status file with atomic updates and heartbeat support.

    Handles both old plain-text status files (e.g., just "RUNNING") and new JSON-lines format.
    All timestamps are written in UTC.
    """

    def __init__(self, output_path: str, worker_id: str):
        self.path = get_status_path(output_path)
        self.worker_id = worker_id

    def read(self) -> list[ExecutorStepEvent]:
        """Read events, handling both old plain-text and new JSON formats."""
        return read_events(self.path)

    @property
    def status(self) -> str | None:
        """Get the current status (last event's status)."""
        events = self.read()
        return events[-1].status if events else None

    @property
    def last_event(self) -> ExecutorStepEvent | None:
        """Get the last event."""
        events = self.read()
        return events[-1] if events else None

    def is_stale(self) -> bool:
        """Check if last event's heartbeat is stale (>HEARTBEAT_TIMEOUT seconds old)."""
        last = self.last_event
        if last is None:
            return True
        return _is_timestamp_stale(last.date)

    def write(self, status: str, message: str | None = None) -> None:
        """Atomically write a new status event."""
        events = self.read()
        events.append(
            ExecutorStepEvent(
                date=datetime.now(timezone.utc).isoformat(),
                status=status,
                message=message,
                task_id=self.worker_id,
            )
        )
        self._atomic_write(events)

    def ping(self, message: str = "heartbeat") -> None:
        """Write a heartbeat RUNNING event."""
        self.write(STATUS_RUNNING, message=message)

    def _atomic_write(self, events: list[ExecutorStepEvent]) -> None:
        """Write events atomically using temp file + rename."""
        temp_path = f"{self.path}.tmp.{self.worker_id}"
        fs = fsspec.core.url_to_fs(self.path)[0]

        with fsspec.open(temp_path, "w") as f:
            for event in events:
                print(json.dumps(asdict(event)), file=f)

        fs.mv(temp_path, self.path)
