# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from pathlib import PurePosixPath

import pytest
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.identity import canonical_sha256

from taskforge import wiring
from taskforge.effects.events import (
    EVENT_TYPES,
    Committed,
    Event,
    Failed,
    Fingerprint,
    Invalidated,
    Note,
    ReplayPolicy,
    Resolved,
    Retrying,
    ScopeEntered,
    ScopeExited,
    StageBoundary,
    StageCompleted,
    digest,
    event_digest,
)
from taskforge.effects.log import BufferSink, Record, decode
from taskforge.harness.events import ToolInvoked, ToolReturned
from taskforge.llm.policy import LLMPolicy
from taskforge.model.content_hash import json_value
from taskforge.model.llm import Message, Role
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import TrialResult, ValidationResult
from taskforge.model.stage import Stage
from taskforge.model.workpiece import ENTRY, Workpiece
from taskforge.resources.decision_model import DecisionModelRead
from taskforge.resources.env import ReadFile, RunCommand, WriteFile, WriteFiles
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.ids import NewId
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import SolverGeneration, SolverRollout
from taskforge.resources.web import WebFetch, WebSearch
from taskforge.stages.author import AuthorReference

KINDS: dict[str, type[Event[object]]] = {
    "decision_model.read": DecisionModelRead,
    "effect.failed": Failed,
    "effect.invalidated": Invalidated,
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
    "rollouts.solver_generation": SolverGeneration,
    "rollouts.solver_rollout": SolverRollout,
    "schema.author_reference": AuthorReference,
    "scope.entered": ScopeEntered,
    "scope.exited": ScopeExited,
    "stage.boundary": StageBoundary,
    "stage.completed": StageCompleted,
    "tool.invoked": ToolInvoked,
    "tool.returned": ToolReturned,
    "web.fetch": WebFetch,
    "web.search": WebSearch,
}
"""Recorded journals name events by these kinds; a class may move or be renamed, but its kind may not change."""

RECORDED = b"""\
{"event": {"payload": {"name": "author"}, "type": "scope.entered"}, "opened": {"definition": "image", \
"owner": "author/0"}, "sequence": 1, "wall_time": 0.0}
{"event": {"payload": {"msg": "hello"}, "type": "effect.note"}, "sequence": 2, "wall_time": 0.0}
{"event": {"payload": {"fingerprints": [{"digest": "uuid4-v1", "replay": "cache", "resource": "ids"}], \
"replayed": false, "request": {"__forge_event__": {"payload": {"key": "author"}, "type": "ids.new_id"}}, \
"value": "abc"}, "type": "effect.resolved"}, "sequence": 3, "wall_time": 0.0}
{"event": {"payload": {"config_digest": "config", "exit": "Submitted", "round": 1, "stage": "author", \
"terminal": null}, "type": "stage.completed"}, "sequence": 4, "wall_time": 0.0}
{"attempt": 1, "event": {"payload": {"limit": null, "offset": 0, "path": "missing"}, "type": "env.read_file"}, \
"sequence": 5, "wall_time": 0.0}
{"answers": 5, "event": {"payload": {"category": "ENOENT", "message": "[Errno 2] No such file or directory: \
'missing'"}, "type": "effect.failed"}, "sequence": 6, "wall_time": 0.0}
"""
"""Journal lines written by this format, frozen."""


COMMITTED = Committed("visit", ValidationResult("task", (), (), ()))


# Regression: an unconditional ``committed`` field moved every marker's digest, so a format-12 root's repeated
# markers, stored as digest references, no longer decoded. A marker without a commitment keeps its format-12 digest.
def test_recorded_kinds_decoders_and_replay_digests_are_pinned() -> None:
    assert wiring.install  # importing wiring registers every library event
    assert {kind: cls for kind, cls in EVENT_TYPES.items() if cls.__module__.startswith("taskforge.")} == KINDS
    ids = (Fingerprint("ids", "uuid4-v1", ReplayPolicy.CACHE),)
    resolved = Resolved(NewId("author"), ids, "abc")
    stage = StageCompleted(Stage.AUTHOR, "config", "Submitted", 1)
    failed = Failed("ENOENT", "[Errno 2] No such file or directory: 'missing'")
    records = decode(RECORDED)
    assert [record.event for record in records] == [
        ScopeEntered("author"),
        Note("hello"),
        resolved,
        stage,
        ReadFile(PurePosixPath("missing")),
        failed,
    ]
    assert [record.answers for record in records] == [None] * 5 + [5]  # an older format's open and attempt are dropped
    request = InferenceRequest((Message(Role.USER, "hi"),), (), LLMPolicy())
    assert {
        "stage": event_digest(StageCompleted(Stage.AUTHOR, "config", "submitted", 1)),
        "committed": event_digest(StageCompleted(Stage.REVIEW, "config", "Decided", 0, None, COMMITTED)),
        "command": event_digest(RunCommand("true", 30.0)),
        "inference": event_digest(request),
        "resolved": event_digest(resolved),
    } == {
        "stage": "230238f1053d6ebfb3a54954281e736c301dcb21323bbc456f77d5e39500585d",
        "committed": "c43a00a68030212385966fb9d52d9f1f0ff5bb7d511ae6f122ac132195195878",
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


@pytest.mark.parametrize("reward", [math.nan, math.inf, -math.inf])
async def test_a_boundary_workpiece_round_trips_a_nonfinite_reward_and_its_dict_order(
    proposal: TaskProposal, reward: float
) -> None:
    """Regression: the event codec sorted every dict, so a resumed stage rendered a task's grader parameters in
    another order and its first model request missed the cache. A nonfinite reward survives as a JSON constant."""
    trial = TrialResult(0, GradeResult(Outcome.GRADED, reward, diagnostics={"zeta": 1, "alpha": 0.0}), receipt=3)
    wp = Workpiece("id", proposal, evidence=(ValidationResult("task", (trial,), (), ()),))
    sink = BufferSink()
    await sink.write([Record(1, 0.0, StageBoundary("plan", Stage.VALIDATE, ENTRY.dump_json((wp,)).decode()))])
    (record,) = decode(bytes(sink.data))
    boundary = record.event
    assert isinstance(boundary, StageBoundary)
    grade = boundary.workpiece.evidence[0].solver[0].grade
    assert grade is not None and grade.reward is not None and math.isnan(reward) == math.isnan(grade.reward)
    assert math.isnan(reward) or grade.reward == reward
    assert list(grade.diagnostics.items()) == [("zeta", 1), ("alpha", 0.0)]
    assert isinstance(grade.diagnostics["alpha"], float)
