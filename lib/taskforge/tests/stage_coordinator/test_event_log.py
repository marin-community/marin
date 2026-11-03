# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The JSON store round-trips a workpiece's events and refuses another schema.

Invariants: none (pins the store's round trip and schema refusal).
"""

import json

import pytest

from taskforge.stage_coordinator.event_log import JsonEventStore
from taskforge.stage_coordinator.workpiece_machine import EXIT, EventKind, WorkpieceEvent

WORKPIECE = "toy-0"
EVENTS = (
    WorkpieceEvent(EventKind.OPENED, 0, "p1", {"budget_digest": "b1"}, seq=1, at=1.0),
    WorkpieceEvent(EventKind.TRIAGED, 0, "p1", {EXIT: "accept"}, seq=2, at=2.0),
)


def test_the_store_round_trips_a_workpieces_events(tmp_path):
    store = JsonEventStore(tmp_path)

    assert store.load(WORKPIECE) is None
    store.save(WORKPIECE, EVENTS)

    assert store.load(WORKPIECE) == EVENTS
    assert (tmp_path / "workpieces" / WORKPIECE / "events.json").is_file()


def test_a_file_written_under_another_schema_is_refused(tmp_path):
    store = JsonEventStore(tmp_path)
    store.save(WORKPIECE, EVENTS[:1])
    path = store.path(WORKPIECE)
    path.write_text(json.dumps({**json.loads(path.read_text()), "schema": "0"}))

    with pytest.raises(ValueError, match="event schema '0'"):
        store.load(WORKPIECE)
