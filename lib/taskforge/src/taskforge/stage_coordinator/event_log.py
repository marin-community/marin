# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An item's event log while it runs, and the store for its events once it reaches a terminal.

While ``run_item`` carries an item, it appends each event to an ``ItemEventLog``. The log numbers
the event and stamps it with the time. ``InMemoryEventLog`` holds the events in a list, so they
last only as long as the process.

When the item reaches a terminal, ``run_item`` saves the whole log to an ``ItemEventStore``, and at
the next start it restores the log from the store. ``JsonEventStore`` writes the events to
``items/<item_id>/events.json`` under its root with ``write_events`` and reads them back with
``load_events``, which refuses a file written under another ``EVENT_SCHEMA``.
"""

import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol, TypedDict

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename

from taskforge.stage40_build.step_store import check_item_id
from taskforge.stage_coordinator.item_machine import ItemEvent

EVENT_SCHEMA = "1"
EVENTS_FILE = "events.json"
ITEMS_DIR = "items"


class ItemEventLog(Protocol):
    def events(self) -> tuple[ItemEvent, ...]: ...

    def append(self, event: ItemEvent) -> ItemEvent:
        """Record ``event`` as the next event, setting ``seq`` to its position from 1 and ``at`` to now."""
        ...

    def restore(self, events: tuple[ItemEvent, ...]) -> None:
        """Fill an empty log with ``events`` as an ``ItemEventStore`` returned them, ``seq`` and ``at`` intact."""
        ...


class InMemoryEventLog:
    def __init__(self) -> None:
        self._events: list[ItemEvent] = []

    def events(self) -> tuple[ItemEvent, ...]:
        return tuple(self._events)

    def append(self, event: ItemEvent) -> ItemEvent:
        numbered = replace(event, seq=len(self._events) + 1, at=time.time())
        self._events.append(numbered)
        return numbered

    def restore(self, events: tuple[ItemEvent, ...]) -> None:
        if self._events:
            raise ValueError(f"a log holding {len(self._events)} events cannot be restored")
        self._events = list(events)


class ItemEventStore(Protocol):
    def load(self, item_id: str) -> tuple[ItemEvent, ...] | None:
        """Read the events saved for ``item_id``, or ``None`` when the store holds none."""
        ...

    def save(self, item_id: str, events: tuple[ItemEvent, ...]) -> None:
        """Replace the events saved for ``item_id`` with ``events``."""
        ...


class EventFile(TypedDict):
    schema: str
    events: tuple[ItemEvent, ...]


def write_events(path: Path, events: tuple[ItemEvent, ...]) -> None:
    """Write ``events`` to ``path`` as JSON tagged with ``EVENT_SCHEMA``, through a temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_rename(str(path)) as temp:
        Path(temp).write_bytes(_EVENT_FILE.dump_json(EventFile(schema=EVENT_SCHEMA, events=events), indent=2))


def load_events(path: Path) -> tuple[ItemEvent, ...]:
    """Read the events ``write_events`` wrote to ``path``, raising ``ValueError`` on another schema."""
    stored = _EVENT_FILE.validate_json(path.read_bytes())
    if stored["schema"] != EVENT_SCHEMA:
        raise ValueError(f"{path}: event schema {stored['schema']!r}, expected {EVENT_SCHEMA!r}")
    return stored["events"]


class JsonEventStore:
    """Events of each item in ``<root>/items/<item_id>/events.json``."""

    def __init__(self, root: Path):
        self.root = root

    def path(self, item_id: str) -> Path:
        check_item_id(item_id)
        return self.root / ITEMS_DIR / item_id / EVENTS_FILE

    def load(self, item_id: str) -> tuple[ItemEvent, ...] | None:
        path = self.path(item_id)
        return load_events(path) if path.exists() else None

    def save(self, item_id: str, events: tuple[ItemEvent, ...]) -> None:
        write_events(self.path(item_id), events)


_EVENT_FILE: TypeAdapter[EventFile] = TypeAdapter(EventFile)
