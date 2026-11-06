# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from forge.effects.events import StageCompleted, digest, event_digest


def test_event_digest_canonicalizes_mapping_order_and_distinguishes_content_and_type() -> None:
    first = StageCompleted("author", "config", "submitted", {"task": "a", "round": "1"})
    reordered = StageCompleted("author", "config", "submitted", {"round": "1", "task": "a"})
    changed = StageCompleted("author", "config", "submitted", {"round": "2", "task": "a"})

    @dataclass(frozen=True, slots=True)
    class OtherStage(StageCompleted):
        pass

    assert event_digest(first) == event_digest(reordered)
    assert event_digest(first) != event_digest(changed)
    assert event_digest(first) != event_digest(OtherStage("author", "config", "submitted", first.refs))


def test_binary_digest_distinguishes_bytes_from_text_and_marker_shaped_mappings() -> None:
    assert digest(b"\xff") != digest("/w==")
    assert digest(b"\xff") != digest({"__forge_bytes__": "/w=="})


def test_stage_event_snapshots_refs_before_the_caller_reuses_the_mapping() -> None:
    refs = {"round": "1", "terminal": "HELD"}
    event = StageCompleted("author", "config", "held", refs)
    digest = event_digest(event)
    refs["round"] = "2"
    refs["terminal"] = "ACCEPTED"
    assert event_digest(event) == digest
    assert event.refs == {"round": "1", "terminal": "HELD"}
