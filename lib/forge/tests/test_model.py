# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from pydantic import TypeAdapter
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Message, Role, ToolCall, ToolSchema
from forge.model.proposal import (
    Environment,
    Grounding,
    ProposalHeader,
    ResearchNeed,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
)
from forge.model.review import (
    Accept,
    AdversaryResult,
    BandOutcome,
    BandRules,
    Cause,
    ControlResult,
    Decision,
    Incomplete,
    Modify,
    Reject,
    RejectKind,
    Retry,
    TrialResult,
    ValidationResult,
)
from forge.model.workpiece import WORKPIECE, Conversation, Submission, Terminal, Workpiece
from forge.policy import Policy, Prompt


@pytest.fixture
def workpiece() -> Workpiece:
    task = TaskSpec(
        id="sum",
        context=ConversationInput(events=(TextMessage(role="user", content="1 + 1?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="serialization fixture"),
        source=Source(dataset="forge", revision="v1", row="1", importer_revision="v1"),
    )
    proposal = TaskProposal(
        ProposalHeader(
            id="sum",
            source=SourceRef(SourceKind.CAPABILITY, "arithmetic", "abc"),
            environment=Environment.REASONING,
            verification=Verification.CODE,
            composite=(),
            grounding=Grounding.UNVERIFIED,
            research_need=ResearchNeed.NOT_NEEDED,
            research_reason="synthetic",
            research=(),
            build=(),
            resources=(),
            null_reason="fixture",
        ),
        "fixture\n",
    )
    controls = (Control(Candidate("2"), RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),)
    evidence = ValidationResult(
        "task-digest",
        (TrialResult(0, GradeResult(Outcome.GRADED, 1.0)), TrialResult(1, None, Cause.AGENT_TIMEOUT)),
        (ControlResult(0, GradeResult(Outcome.GRADED, 1.0), True),),
        (AdversaryResult("shortcut", True, ()),),
    )
    session = Conversation(
        (
            Message(Role.USER, "make a task"),
            Message(Role.ASSISTANT, "", (ToolCall("1", "submit", "{}"),), reasoning="done"),
        ),
        (ToolSchema("submit", "Submit the task", {"type": "object", "properties": {"task": {"type": "string"}}}),),
    )
    return Workpiece(
        "w1",
        proposal,
        session=session,
        submission=Submission(task, controls, "authored"),
        evidence=(evidence,),
        decisions=(Modify(("harder",)), Incomplete(("retry later",))),
        terminal=Terminal.HELD,
    )


def test_saved_workpiece_round_trip_keeps_conversation_evidence_and_decisions(workpiece: Workpiece) -> None:
    restored = WORKPIECE.validate_json(WORKPIECE.dump_json(workpiece))
    assert restored == workpiece
    assert restored.evidence[0].solver[1].cause is Cause.AGENT_TIMEOUT
    assert isinstance(restored.decisions[0], Modify)
    assert isinstance(restored.decisions[1], Incomplete)
    assert restored.session is not None
    assert restored.session.messages[-1].reasoning == "done"


@pytest.mark.parametrize(
    "decision",
    [
        Accept(BandOutcome.IN_BAND),
        Reject(RejectKind.TASK, ("ambiguous",)),
        Modify(("revise",)),
        Retry(Cause.NO_GRADE, 2),
        Incomplete(("machine gone",)),
    ],
)
def test_decision_round_trip_preserves_variant(decision: Decision) -> None:
    restored = TypeAdapter(Accept | Reject | Modify | Retry | Incomplete).validate_json(
        TypeAdapter(Accept | Reject | Modify | Retry | Incomplete).dump_json(decision)
    )
    assert restored == decision
    assert type(restored) is type(decision)


def test_workpiece_replace_does_not_change_previous_round(workpiece: Workpiece) -> None:
    assert workpiece.submission is not None
    saved_round = WORKPIECE.dump_json(workpiece)
    edited_task = workpiece.submission.task.model_copy(deep=True)
    edited_task.labels["revision"] = "1"
    revised = workpiece.with_(
        round=1,
        submission=Submission(edited_task, workpiece.submission.controls, "revised"),
        session=None,
        decisions=(*workpiece.decisions, Accept(BandOutcome.IN_BAND)),
        terminal=Terminal.ACCEPTED,
    )
    assert WORKPIECE.dump_json(workpiece) == saved_round
    assert revised.submission is not None
    assert revised.submission.task.labels["revision"] == "1"
    assert revised.submission.digest != workpiece.submission.digest
    assert WORKPIECE.validate_json(WORKPIECE.dump_json(revised)) == revised


def test_submission_digest_tracks_task_and_controls_but_excludes_notes(workpiece: Workpiece) -> None:
    submission = workpiece.submission
    assert submission is not None
    assert replace(submission, notes="new note").digest == submission.digest
    assert replace(submission, controls=()).digest != submission.digest
    assert replace(submission, task=submission.task.model_copy(update={"id": "different"})).digest != submission.digest


@pytest.mark.parametrize(
    ("rate", "placement"),
    [
        (0.1, BandOutcome.TOO_HARD),
        (0.2, BandOutcome.IN_BAND),
        (0.5, BandOutcome.IN_BAND),
        (0.8, BandOutcome.IN_BAND),
        (0.9, BandOutcome.TOO_EASY),
    ],
)
def test_band_placement_includes_both_boundaries(rate: float, placement: BandOutcome) -> None:
    band = BandRules(0.2, 0.8, frozenset({BandOutcome.IN_BAND}))
    assert band.place(rate) is placement


def test_expected_zero_accepts_submission_failure_but_reference_does_not() -> None:
    failure = GradeResult(Outcome.SUBMISSION_FAILURE, None)
    assert RewardBand(Outcome.GRADED, reward_max=0).met_by(failure)
    assert not RewardBand(Outcome.GRADED, reward_min=1).met_by(failure)


def test_policy_overlay_changes_digest_and_preserves_original() -> None:
    policy = Policy({"author.system": Prompt("Hello {name}"), "author.turns": 2, "author.tags": ("a",)})
    updated = policy.overlaid({"author.system": "Welcome {name}", "author.tags": ["b"]})
    assert updated.prompt("author.system").render(name="Ada") == "Welcome Ada"
    assert updated.stage_digest("author") != policy.stage_digest("author")
    assert policy.prompt("author.system").render(name="Ada") == "Hello Ada"
    assert updated.setting("author.tags", tuple) == ("b",)
