# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The store of each workpiece's event log, the one record a relaunch resumes from.

``run_workpiece`` loads a workpiece's events from a ``WorkpieceEventStore``, folds them into the workpiece's state
and saves the whole log after every event it appends, so the store always holds the last checkpoint and a relaunch
replays it. ``JsonEventStore`` writes ``workpieces/<workpiece_id>/events.json`` under its root with ``write_events``
and reads it back with ``load_events``, which refuses a file written under another ``EVENT_SCHEMA``.
"""

from pathlib import Path
from typing import Protocol, TypedDict

from pydantic import TypeAdapter
from rigging.filesystem.atomic import write_bytes_atomic

from taskforge.ledger.records import check_workpiece_id
from taskforge.stage_coordinator.workpiece_machine import WorkpieceEvent

EVENT_SCHEMA = "1"
EVENTS_FILE = "events.json"
WORKPIECES_DIR = "workpieces"


class WorkpieceEventStore(Protocol):
    def load(self, workpiece_id: str) -> tuple[WorkpieceEvent, ...] | None:
        """Read the events saved for ``workpiece_id``, or ``None`` when the store holds none."""
        ...

    def save(self, workpiece_id: str, events: tuple[WorkpieceEvent, ...]) -> None:
        """Replace the events saved for ``workpiece_id`` with ``events``."""
        ...


class EventFile(TypedDict):
    schema: str
    events: tuple[WorkpieceEvent, ...]


def write_events(path: Path, events: tuple[WorkpieceEvent, ...]) -> None:
    """Write ``events`` to ``path`` as JSON tagged with ``EVENT_SCHEMA``, through a temporary sibling."""
    write_bytes_atomic(str(path), _EVENT_FILE.dump_json(EventFile(schema=EVENT_SCHEMA, events=events), indent=2))


def load_events(path: Path) -> tuple[WorkpieceEvent, ...]:
    """Read the events ``write_events`` wrote to ``path``, raising ``ValueError`` on another schema."""
    stored = _EVENT_FILE.validate_json(path.read_bytes())
    if stored["schema"] != EVENT_SCHEMA:
        raise ValueError(f"{path}: event schema {stored['schema']!r}, expected {EVENT_SCHEMA!r}")
    return stored["events"]


class JsonEventStore:
    """Events of each workpiece in ``<root>/workpieces/<workpiece_id>/events.json``."""

    def __init__(self, root: Path):
        self.root = root

    def path(self, workpiece_id: str) -> Path:
        check_workpiece_id(workpiece_id)
        return self.root / WORKPIECES_DIR / workpiece_id / EVENTS_FILE

    def load(self, workpiece_id: str) -> tuple[WorkpieceEvent, ...] | None:
        path = self.path(workpiece_id)
        return load_events(path) if path.exists() else None

    def save(self, workpiece_id: str, events: tuple[WorkpieceEvent, ...]) -> None:
        write_events(self.path(workpiece_id), events)


_EVENT_FILE: TypeAdapter[EventFile] = TypeAdapter(EventFile)
