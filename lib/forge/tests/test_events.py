# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from forge.effects.events import StageCompleted, canonical_digest, event_digest


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
    assert canonical_digest(b"\xff") != canonical_digest("/w==")
    assert canonical_digest(b"\xff") != canonical_digest({"__forge_bytes__": "/w=="})
