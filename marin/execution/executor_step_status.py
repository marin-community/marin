"""
Each `ExecutorStep` produces an `output_path`.
We associate each `output_path` with a `output_path.STATUS` file that contains a
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
from datetime import datetime

import fsspec

from marin.utils import fsspec_exists

STATUS_WAITING = "WAITING"  # Waiting for dependencies to finish
STATUS_RUNNING = "RUNNING"
STATUS_FAILED = "FAILED"
STATUS_SUCCESS = "SUCCESS"


@dataclass(frozen=True)
class ExecutorStepEvent:
    """Represents a change in the status of an `ExecutorStep`."""

    date: str
    """When the `status` changed."""

    status: str
    """Represents the `status` of the job."""

    message: str | None = None
    """An optional message to provide more context (especially for errors)."""


def get_status_path(output_path: str) -> str:
    """Return the `path` of the status file associated with `output_path`, which contains a list of events."""
    return os.path.join(output_path, ".executor_status")


def read_events(path: str) -> list[ExecutorStepEvent]:
    """Reads the status of a step from `path`."""
    events = []
    if fsspec_exists(path):
        with fsspec.open(path, "r") as f:
            for line in f:
                events.append(ExecutorStepEvent(**json.loads(line)))
    return events


def get_current_status(events: list[ExecutorStepEvent]) -> str | None:
    """Get the most recent status (last event)."""
    return events[-1].status if len(events) > 0 else None


def append_status(path: str, status: str, message: str | None = None):
    """Append a new event with `status` to the file at `path`."""
    events = read_events(path)

    date = datetime.now().isoformat()
    event = ExecutorStepEvent(date=date, status=status, message=message)
    events.append(event)

    # Note: gcs files are immutable so can't append, so have to read everything.
    with fsspec.open(path, "w") as f:
        for event in events:
            print(json.dumps(asdict(event)), file=f)
