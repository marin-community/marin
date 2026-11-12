# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome

from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Message, Role, ToolCall, ToolSchema
from forge.model.proposal import TaskProposal
from forge.model.review import (
    Abandon,
    Accept,
    AdversaryResult,
    BandOutcome,
    Cause,
    ControlResult,
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

pytest_plugins = ("lib.forge.tests.test_proposal",)


@pytest.fixture
def workpiece(proposal: TaskProposal) -> Workpiece:
    task = arithmetic_task()
    controls = (Control(Candidate("12"), RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),)
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
        decisions=(
            Modify(("harder",)),
            Incomplete(("retry later",)),
            Accept(BandOutcome.IN_BAND),
            Reject(RejectKind.TASK, ("ambiguous",)),
            Retry(Cause.NO_GRADE, 2),
            Abandon(("author stopped",)),
        ),
        terminal=Terminal.HELD,
    )


def test_saved_workpiece_round_trip_keeps_conversation_evidence_and_decisions(workpiece: Workpiece) -> None:
    restored = WORKPIECE.validate_json(WORKPIECE.dump_json(workpiece))
    assert restored == workpiece
    assert restored.evidence[0].solver[1].cause is Cause.AGENT_TIMEOUT
    assert tuple(type(d) for d in restored.decisions) == (Modify, Incomplete, Accept, Reject, Retry, Abandon)
    assert restored.session is not None
    assert restored.session.messages[-1].reasoning == "done"


def test_submission_digest_tracks_task_and_controls_but_excludes_notes(workpiece: Workpiece) -> None:
    submission = workpiece.submission
    assert submission is not None
    assert replace(submission, notes="new note").digest == submission.digest
    assert replace(submission, controls=()).digest != submission.digest
    assert replace(submission, task=submission.task.model_copy(update={"id": "different"})).digest != submission.digest


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


# Pilots 20261106i / 20261107j: abandoned records had no reason; reject both construction and persisted malformed data.
@pytest.mark.parametrize("reasons", [(), ("",), (" ",)])
def test_abandon_requires_a_reason(workpiece: Workpiece, reasons: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="non-empty reasons"):
        Abandon(reasons)
    saved = WORKPIECE.dump_python(workpiece, mode="json")
    saved.update(terminal="abandoned", decisions=[{"kind": "abandon", "reasons": reasons}])
    with pytest.raises(ValueError, match=r"decisions.0.abandon"):
        WORKPIECE.validate_python(saved)


def test_old_format_workpiece_fails_loudly(workpiece: Workpiece) -> None:
    saved = WORKPIECE.dump_python(workpiece, mode="json")
    saved["too_easy_baseline"] = None
    with pytest.raises(ValueError, match="too_easy_baseline"):
        WORKPIECE.validate_python(saved)
