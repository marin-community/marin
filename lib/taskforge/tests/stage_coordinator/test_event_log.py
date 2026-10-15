# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The in-memory log numbers and stamps events, and the JSON store round-trips them."""

import json

import pytest

from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.item_machine import EXIT, EventKind, ItemEvent

ITEM = "toy-0"
OPENED = ItemEvent(EventKind.OPENED, 0, "p1", {"budget_digest": "b1"})
TRIAGED = ItemEvent(EventKind.TRIAGED, 0, "p1", {EXIT: "accept"})


def test_append_numbers_events_from_one_and_the_store_round_trips_them(tmp_path):
    log = InMemoryEventLog()
    first, second = log.append(OPENED), log.append(TRIAGED)
    store = JsonEventStore(tmp_path)

    assert store.load(ITEM) is None
    store.save(ITEM, log.events())
    restored = InMemoryEventLog()
    restored.restore(store.load(ITEM) or ())

    assert (first.seq, second.seq) == (1, 2)
    assert first.at <= second.at
    assert restored.events() == log.events() == (first, second)
    assert (tmp_path / "items" / ITEM / "events.json").is_file()
    with pytest.raises(ValueError, match="cannot be restored"):
        restored.restore(log.events())


def test_a_file_written_under_another_schema_is_refused(tmp_path):
    store = JsonEventStore(tmp_path)
    log = InMemoryEventLog()
    log.append(OPENED)
    store.save(ITEM, log.events())
    path = store.path(ITEM)
    path.write_text(json.dumps({**json.loads(path.read_text()), "schema": "0"}))

    with pytest.raises(ValueError, match="event schema '0'"):
        store.load(ITEM)
