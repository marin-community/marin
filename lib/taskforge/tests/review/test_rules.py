# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from dataclasses import replace

import pytest
from taskcompendium.grading_result import Outcome

from taskforge.build.step import StepRole
from taskforge.review.decision import Accept, Reject, RejectKind, Repair, Retry
from taskforge.review.rules import (
    NEW_CONTROLS_HEADER,
    NOTES_HEADER,
    STAGED_BRIEF,
    ItemHistory,
    decide,
    render_brief,
    staged_repair,
    steps_for,
)
from taskforge.spec.controls import (
    REJECTION_CEILING,
    Control,
    ControlCategory,
    ControlConcern,
    ControlKind,
    Expectation,
    Transcript,
    parse_controls,
    reply,
)
from taskforge.validate.calibration import FINDING_ROLES, Finding, FindingKind
from taskforge.validate.outcome import Cause

FRESH = ItemHistory(repairs_used=0, max_repairs=2, prior_band_findings=frozenset())
SPENT = ItemHistory(repairs_used=2, max_repairs=2, prior_band_findings=frozenset())

SHORTCUT_CONTROL = Control(
    id="adv-shortcut-0",
    kind=ControlKind.NEGATIVE,
    category=ControlCategory.REWARD_HACK,
    concern=ControlConcern.SHORTCUT,
    author="adversary/shortcut/0",
    payload=Transcript((reply("ANSWER = 394\nANSWER = 395\nANSWER = 396"),)),
    expect=Expectation(Outcome.GRADED, reward_max=REJECTION_CEILING),
)


def finding(kind: FindingKind, detail: str = "", new_controls: tuple[Control, ...] = ()) -> Finding:
    return Finding(kind=kind, detail=detail or f"{kind} detail", roles=FINDING_ROLES[kind], new_controls=new_controls)


NOTES = (
    finding(
        FindingKind.SHORTCUT_PASSED,
        "The shortcut adversary 0 was graded as passing. Tier noted (row 7): solved against orders: read "
        '["/workspace/numbers.txt"] and submitted the answer; a role violation, not a task defect',
    ),
    finding(
        FindingKind.AMBIGUOUS,
        "The ambiguity adversary 1 was graded as passing. Tier noted (row 7): passed under a stated reading; "
        "the submission cannot be compared for a machine-state answer",
    ),
)


def test_decisive_finding_repairs_even_when_trials_are_ungraded(draft, summary):
    shortcut = finding(FindingKind.SHORTCUT_PASSED, "listed three answers and passed", (SHORTCUT_CONTROL,))
    decision = decide(draft, summary((shortcut,), causes=Counter({Cause.MODEL_UNAVAILABLE: 3})), FRESH)

    assert isinstance(decision, Repair)
    assert decision.program_digest == draft.provenance.program_digest
    assert decision.invalidate == ("grader", "controls")
    assert decision.brief.findings == (shortcut,)
    assert "listed three answers and passed" in decision.brief.failure


def test_decisive_findings_come_before_a_host_refusal(draft, summary):
    defect = Finding(kind=FindingKind.TASK_DEFECT, detail="setup exited 1", roles=(StepRole.ENVIRONMENT,))
    causes = Counter({Cause.MACHINE_UNSUPPORTED: 1, Cause.TASK_SETUP: 7})
    decision = decide(draft, summary((defect,), causes=causes), FRESH)

    assert isinstance(decision, Repair)
    assert decision.invalidate == ("machine",)


def test_host_refusal_rejects_for_the_host_before_any_retry(draft, summary):
    causes = Counter({Cause.MODEL_UNAVAILABLE: 5, Cause.SUBMISSION_UNSUPPORTED: 2})
    decision = decide(draft, summary(causes=causes), FRESH)

    assert isinstance(decision, Reject)
    assert decision.kind == RejectKind.HOST
    assert decision.reasons == ("submission_unsupported: 2 trials ungraded",)


def test_rerunnable_incomplete_evidence_retries_its_most_common_cause(draft, summary):
    causes = Counter({Cause.TOKEN_CONTRACT: 1, Cause.MODEL_UNAVAILABLE: 3, Cause.MACHINE_START: 3})
    assert decide(draft, summary(causes=causes), FRESH) == Retry(cause=Cause.MACHINE_START, count=3)


def test_retry_is_not_bounded_by_the_repair_budget(draft, summary):
    decision = decide(draft, summary(causes=Counter({Cause.MODEL_UNAVAILABLE: 8}), solved=0), SPENT)
    assert decision == Retry(cause=Cause.MODEL_UNAVAILABLE, count=8)


def test_band_finding_is_repaired_once_then_rejects_the_task(draft, summary):
    too_hard = finding(FindingKind.TOO_HARD, "0 of 8 solved; 6 timed out")
    first = decide(draft, summary((too_hard,), solved=0), FRESH)

    assert isinstance(first, Repair)
    assert first.invalidate == ("fixtures", "instruction")

    repaired = ItemHistory(repairs_used=1, max_repairs=2, prior_band_findings=frozenset({FindingKind.TOO_HARD}))
    again = decide(draft, summary((too_hard,), solved=0), repaired)

    assert isinstance(again, Reject)
    assert again.kind == RejectKind.TASK
    assert again.reasons == ("too_hard: 0 of 8 solved; 6 timed out (again after a repair)",)


def test_a_different_band_finding_still_gets_its_one_repair(draft, summary):
    history = ItemHistory(repairs_used=1, max_repairs=2, prior_band_findings=frozenset({FindingKind.TOO_HARD}))
    decision = decide(draft, summary((finding(FindingKind.TOO_EASY),), solved=8), history)
    assert isinstance(decision, Repair)


def test_clean_complete_evidence_accepts(draft, summary):
    clean = summary()
    assert decide(draft, clean, FRESH) == Accept(summary=clean)


@pytest.mark.parametrize(
    "findings",
    [(finding(FindingKind.CONTROL_VIOLATED, "neg-0 scored 1.0"),), (finding(FindingKind.TOO_EASY, "8 of 8 solved"),)],
)
def test_a_repair_past_the_budget_rejects_for_budget(draft, summary, findings):
    evidence = summary(findings)
    decision = decide(draft, evidence, SPENT)

    assert isinstance(decision, Reject)
    assert decision.kind == RejectKind.BUDGET
    assert decision.summary == evidence
    assert decision.reasons[-1] == "repairs exhausted: 2 of 2"


def test_a_repair_whose_roles_ran_no_step_still_revises_the_program(draft, summary):
    ambiguous = Finding(kind=FindingKind.AMBIGUOUS, detail="two readings", roles=(StepRole.INSTRUCTIONS,))
    bare = replace(draft, provenance=replace(draft.provenance, steps=draft.provenance.steps[3:5]))
    decision = decide(bare, summary((ambiguous,)), FRESH)

    assert isinstance(decision, Repair)
    assert decision.invalidate == ()


def test_staged_draft_gets_the_fixed_repair_not_validation(staged_draft, summary):
    with pytest.raises(ValueError, match="staged"):
        decide(staged_draft, summary(), FRESH)

    decision = staged_repair(staged_draft, FRESH)
    assert isinstance(decision, Repair)
    assert decision.brief.failure == STAGED_BRIEF
    assert decision.invalidate == ()

    spent = staged_repair(staged_draft, SPENT)
    assert isinstance(spent, Reject)
    assert spent.kind == RejectKind.BUDGET
    assert spent.summary is None


def test_brief_carries_new_controls_the_author_can_copy_verbatim():
    leak = Control(
        id="adv-leak-1",
        kind=ControlKind.NEGATIVE,
        category=ControlCategory.REWARD_HACK,
        concern=ControlConcern.SHORTCUT,
        author="adversary/leak/1",
        payload=Transcript((reply("395"),)),
        expect=Expectation(Outcome.GRADED, reward_max=REJECTION_CEILING),
    )
    findings = (
        finding(FindingKind.SHORTCUT_PASSED, new_controls=(SHORTCUT_CONTROL,)),
        finding(FindingKind.LEAK_PASSED, new_controls=(leak,)),
    )
    brief = render_brief(findings, ())

    blocks = re.findall(r"```json\n(.*?)\n```", brief.failure, re.DOTALL)
    assert len(blocks) == 1
    assert parse_controls(blocks[0]) == (SHORTCUT_CONTROL, leak)
    assert (
        "Finding 2: leak_passed (revise the fixtures, environment, instructions, grader, controls steps)"
        in brief.failure
    )


def test_steps_for_names_each_step_once_in_build_order(draft):
    steps = (*draft.provenance.steps, draft.provenance.steps[3])
    assert steps_for([StepRole.CONTROLS, StepRole.GRADER], steps) == ("grader", "controls")


def test_notes_never_change_the_decision(draft, summary):
    accepted = summary(notes=NOTES)
    assert decide(draft, accepted, FRESH) == Accept(summary=accepted)

    shortcut = finding(FindingKind.SHORTCUT_PASSED, "listed three answers and passed", (SHORTCUT_CONTROL,))
    decision = decide(draft, summary((shortcut,), notes=NOTES), FRESH)

    assert isinstance(decision, Repair)
    assert decision.brief.findings == (shortcut,)
    assert decision.brief.notes == NOTES
    assert decision.invalidate == ("grader", "controls")
    failure = decision.brief.failure
    assert failure.index(shortcut.detail) < failure.index(NOTES_HEADER) < failure.index(NEW_CONTROLS_HEADER)
    assert f"Note 2: ambiguous (instructions)\n{NOTES[1].detail}" in failure
    blocks = re.findall(r"```json\n(.*?)\n```", failure, re.DOTALL)
    assert [parse_controls(block) for block in blocks] == [(SHORTCUT_CONTROL,)]


def test_notes_ride_along_a_band_repair(draft, summary):
    too_easy = finding(FindingKind.TOO_EASY, "8 of 8 solved")
    decision = decide(draft, summary((too_easy,), solved=8, notes=NOTES), FRESH)

    assert isinstance(decision, Repair)
    assert decision.brief.findings == (too_easy,)
    assert decision.brief.notes == NOTES
    assert NOTES[0].detail in decision.brief.failure
    assert "```json" not in decision.brief.failure


def test_a_rejection_names_findings_not_notes(draft, summary):
    violated = finding(FindingKind.CONTROL_VIOLATED, "neg-0 scored 1.0")
    decision = decide(draft, summary((violated,), notes=NOTES), SPENT)

    assert isinstance(decision, Reject)
    assert decision.reasons == ("control_violated: neg-0 scored 1.0", "repairs exhausted: 2 of 2")


def test_render_brief_needs_a_finding():
    with pytest.raises(ValueError, match="at least one finding"):
        render_brief((), NOTES)
