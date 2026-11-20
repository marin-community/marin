# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import pytest
from taskcompendium.identity import canonical_sha256

from taskforge import wiring
from taskforge.effects.events import (
    EVENT_TYPES,
    Event,
    Fingerprint,
    Note,
    ReplayPolicy,
    Resolved,
    Retrying,
    ScopeEntered,
    ScopeExited,
    StageCompleted,
    digest,
    event_digest,
)
from taskforge.effects.log import decode
from taskforge.harness.events import ToolInvoked, ToolReturned
from taskforge.llm.policy import LLMPolicy
from taskforge.model.content_hash import json_value
from taskforge.model.llm import Message, Role
from taskforge.model.stage import Stage
from taskforge.resources.env import ReadFile, RunCommand, WriteFile, WriteFiles
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.ids import NewId
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import SolverRollout
from taskforge.resources.web import WebFetch, WebSearch

KINDS: dict[str, type[Event[object]]] = {
    "effect.note": Note,
    "effect.resolved": Resolved,
    "effect.retrying": Retrying,
    "env.read_file": ReadFile,
    "env.run_command": RunCommand,
    "env.write_file": WriteFile,
    "env.write_files": WriteFiles,
    "grading.grade_candidate": GradeCandidate,
    "ids.new_id": NewId,
    "inference.request": InferenceRequest,
    "rollouts.solver_rollout": SolverRollout,
    "scope.entered": ScopeEntered,
    "scope.exited": ScopeExited,
    "stage.completed": StageCompleted,
    "tool.invoked": ToolInvoked,
    "tool.returned": ToolReturned,
    "web.fetch": WebFetch,
    "web.search": WebSearch,
}
"""Recorded journals name events by these kinds; a class may move or be renamed, but its kind may not change."""

RECORDED = b"""\
{"event": {"payload": {"name": "author"}, "type": "scope.entered"}, "sequence": 1, "wall_time": 0.0}
{"event": {"payload": {"msg": "hello"}, "type": "effect.note"}, "sequence": 2, "wall_time": 0.0}
{"event": {"payload": {"fingerprints": [{"digest": "uuid4-v1", "replay": "cache", "resource": "ids"}], \
"replayed": false, "request": {"__forge_event__": {"payload": {"key": "author"}, "type": "ids.new_id"}}, \
"value": "abc"}, "type": "effect.resolved"}, "sequence": 3, "wall_time": 0.0}
{"event": {"payload": {"config_digest": "config", "exit": "Submitted", "round": 1, "stage": "author", \
"terminal": null}, "type": "stage.completed"}, "sequence": 4, "wall_time": 0.0}
"""
"""Journal lines written by this format, frozen."""


def test_recorded_kinds_decoders_and_replay_digests_are_pinned() -> None:
    assert wiring.install  # importing wiring registers every library event
    assert {kind: cls for kind, cls in EVENT_TYPES.items() if cls.__module__.startswith("taskforge.")} == KINDS
    ids = (Fingerprint("ids", "uuid4-v1", ReplayPolicy.CACHE),)
    resolved = Resolved(NewId("author"), ids, "abc")
    stage = StageCompleted(Stage.AUTHOR, "config", "Submitted", 1)
    assert [record.event for record in decode(RECORDED)] == [ScopeEntered("author"), Note("hello"), resolved, stage]
    request = InferenceRequest((Message(Role.USER, "hi"),), (), LLMPolicy())
    assert {
        "stage": event_digest(StageCompleted(Stage.AUTHOR, "config", "submitted", 1)),
        "command": event_digest(RunCommand("true", 30.0)),
        "inference": event_digest(request),
        "resolved": event_digest(resolved),
    } == {
        "stage": "230238f1053d6ebfb3a54954281e736c301dcb21323bbc456f77d5e39500585d",
        "command": "227b4531c6a0a3a00d43fa741cd010bd246b32d81c1237126231830b797279ba",
        "inference": "2709c75295b758602f82a1e114cfbf401cdba869c2d6620caaa851973f9887aa",
        "resolved": "052985a3615a19c16965da77e0bcd07d7741bf5b9f6f1a65ca4d8bd946ffc5e0",
    }


def test_event_classes_need_one_unique_kind() -> None:
    with pytest.raises(TypeError, match="must declare a recorded kind"):

        class Unnamed(Event[None]):
            pass

    with pytest.raises(TypeError, match=r"'env\.run_command' is declared by both RunCommand and .*Shadow"):

        class Shadow(Event[None]):
            kind = "env.run_command"

    assert EVENT_TYPES["env.run_command"] is RunCommand

    with pytest.raises(TypeError, match=r"'test\.nested' is declared by both .*First\.Nested and .*Second\.Nested"):

        class First:
            @dataclass(frozen=True, slots=True)
            class Nested(Event[None]):
                kind = "test.nested"

        class Second:
            @dataclass(frozen=True, slots=True)
            class Nested(Event[None]):
                kind = "test.nested"


def test_event_digest_distinguishes_content_and_type() -> None:
    first = StageCompleted(Stage.AUTHOR, "config", "submitted", 1)
    reordered = StageCompleted(Stage.AUTHOR, "config", "submitted", 1)
    changed = StageCompleted(Stage.AUTHOR, "config", "submitted", 2)

    @dataclass(frozen=True, slots=True)
    class OtherStage(StageCompleted):
        kind = "test.other_stage"

    assert event_digest(first) == event_digest(reordered)
    assert event_digest(first) != event_digest(changed)
    assert event_digest(first) != event_digest(OtherStage(Stage.AUTHOR, "config", "submitted", first.round))


def test_binary_digest_distinguishes_bytes_from_text_and_marker_shaped_mappings() -> None:
    assert digest(b"\xff") != digest("/w==")
    assert digest(b"\xff") != digest({"__forge_bytes__": "/w=="})


def test_digest_matches_the_taskcompendium_hash_and_also_accepts_nan() -> None:
    value = {"text": "é ✓", "items": [1, 2.5, None, True], "nested": {"b": 1, "a": "x"}}
    assert digest(value) == canonical_sha256({"value": json_value(value)})
    with pytest.raises(ValueError):
        canonical_sha256({"value": math.nan})
    assert digest(math.nan) != digest(math.inf)
