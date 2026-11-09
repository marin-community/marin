# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from taskforge.effects.events import StageCompleted, digest, event_digest
from taskforge.model.stage import Stage


def test_event_digest_distinguishes_content_and_type() -> None:
    first = StageCompleted(Stage.AUTHOR, "config", "submitted", 1)
    reordered = StageCompleted(Stage.AUTHOR, "config", "submitted", 1)
    changed = StageCompleted(Stage.AUTHOR, "config", "submitted", 2)

    @dataclass(frozen=True, slots=True)
    class OtherStage(StageCompleted):
        pass

    assert event_digest(first) == event_digest(reordered)
    assert event_digest(first) != event_digest(changed)
    assert event_digest(first) != event_digest(OtherStage(Stage.AUTHOR, "config", "submitted", first.round))


def test_binary_digest_distinguishes_bytes_from_text_and_marker_shaped_mappings() -> None:
    assert digest(b"\xff") != digest("/w==")
    assert digest(b"\xff") != digest({"__forge_bytes__": "/w=="})
