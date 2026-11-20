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
    ReplayPolicy,
    Resolved,
    StageCompleted,
    digest,
    event_digest,
)
from taskforge.llm.policy import LLMPolicy
from taskforge.model.content_hash import json_value
from taskforge.model.llm import Message, Role
from taskforge.model.stage import Stage
from taskforge.resources.env import RunCommand
from taskforge.resources.ids import NewId
from taskforge.resources.inference import InferenceRequest

KINDS = {
    "effect.note": "Note",
    "effect.resolved": "Resolved",
    "effect.retrying": "Retrying",
    "env.read_file": "ReadFile",
    "env.run_command": "RunCommand",
    "env.write_file": "WriteFile",
    "env.write_files": "WriteFiles",
    "grading.grade_candidate": "GradeCandidate",
    "ids.new_id": "NewId",
    "inference.request": "InferenceRequest",
    "rollouts.solver_rollout": "SolverRollout",
    "scope.entered": "ScopeEntered",
    "scope.exited": "ScopeExited",
    "stage.completed": "StageCompleted",
    "tool.invoked": "ToolInvoked",
    "tool.returned": "ToolReturned",
    "web.fetch": "WebFetch",
    "web.search": "WebSearch",
}
"""Recorded journals name events by these kinds; changing one breaks every journal that recorded it."""


def test_recorded_kinds_and_replay_digests_are_pinned() -> None:
    assert wiring.install  # importing wiring registers every library event
    library = {kind: cls.__qualname__ for kind, cls in EVENT_TYPES.items() if cls.__module__.startswith("taskforge.")}
    assert library == KINDS
    request = InferenceRequest((Message(Role.USER, "hi"),), (), LLMPolicy())
    resolved = Resolved(NewId("author"), (Fingerprint("ids", "uuid4-v1", ReplayPolicy.CACHE),), "abc")
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
