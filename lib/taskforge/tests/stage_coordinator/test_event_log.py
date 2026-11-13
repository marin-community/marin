# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The in-memory log numbers and stamps events, and the JSON store round-trips them."""

import json

import pytest

from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.workpiece_machine import EXIT, EventKind, WorkpieceEvent

WORKPIECE = "toy-0"
OPENED = WorkpieceEvent(EventKind.OPENED, 0, "p1", {"budget_digest": "b1"})
TRIAGED = WorkpieceEvent(EventKind.TRIAGED, 0, "p1", {EXIT: "accept"})


def test_append_numbers_events_from_one_and_the_store_round_trips_them(tmp_path):
    log = InMemoryEventLog()
    first, second = log.append(OPENED), log.append(TRIAGED)
    store = JsonEventStore(tmp_path)

    assert store.load(WORKPIECE) is None
    store.save(WORKPIECE, log.events())
    restored = InMemoryEventLog()
    restored.restore(store.load(WORKPIECE) or ())

    assert (first.seq, second.seq) == (1, 2)
    assert first.at <= second.at
    assert restored.events() == log.events() == (first, second)
    assert (tmp_path / "workpieces" / WORKPIECE / "events.json").is_file()
    with pytest.raises(ValueError, match="cannot be restored"):
        restored.restore(log.events())


def test_a_file_written_under_another_schema_is_refused(tmp_path):
    store = JsonEventStore(tmp_path)
    log = InMemoryEventLog()
    log.append(OPENED)
    store.save(WORKPIECE, log.events())
    path = store.path(WORKPIECE)
    path.write_text(json.dumps({**json.loads(path.read_text()), "schema": "0"}))

    with pytest.raises(ValueError, match="event schema '0'"):
        store.load(WORKPIECE)
