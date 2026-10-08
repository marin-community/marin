# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from collections import Counter
from dataclasses import replace

import pytest
from taskcompendium.grading_result import Outcome

from taskforge.builder.step import StepRole
from taskforge.review.decision import Accept, BandOutcome, Reject, RejectKind, Repair, Retry
from taskforge.review.rules import (
    NEW_CONTROLS_HEADER,
    NOTES_HEADER,
    BandChoice,
    BandRule,
    BandRules,
    ItemHistory,
    decide,
    render_brief,
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


def history(repairs_used: int, max_repairs: int, band_repairs: dict[FindingKind, int] | None = None) -> ItemHistory:
    return ItemHistory(repairs_used=repairs_used, max_repairs=max_repairs, band_repairs=band_repairs or {})


FRESH = history(0, 2)
SPENT = history(2, 2)

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
        "The shortcut adversary 0 found an accepted submission. Tier noted (row 7): an accepted shortcut after "
        "6 verifier calls, above the repair threshold 3; the adversary reports: any integer line passes",
    ),
    finding(
        FindingKind.SHORTCUT_PASSED,
        "The shortcut adversary 1 found an accepted submission. Tier noted (row 8): 1 of 4 submissions accepted "
        "and no verdict given (stop max_turns)",
    ),
)


def test_decisive_finding_repairs_even_when_trials_are_ungraded(draft, summary, rules):
    shortcut = finding(FindingKind.SHORTCUT_PASSED, "listed three answers and passed", (SHORTCUT_CONTROL,))
    decision = decide(draft, summary((shortcut,), causes=Counter({Cause.MODEL_UNAVAILABLE: 3})), FRESH, rules)

    assert isinstance(decision, Repair)
    assert decision.program_digest == draft.provenance.program_digest
    assert decision.invalidate == ("grader", "controls")
    assert decision.brief.findings == (shortcut,)
    assert "listed three answers and passed" in decision.brief.failure


def test_decisive_findings_come_before_a_host_refusal(draft, summary, rules):
    defect = Finding(kind=FindingKind.TASK_DEFECT, detail="setup exited 1", roles=(StepRole.ENVIRONMENT,))
    causes = Counter({Cause.MACHINE_UNSUPPORTED: 1, Cause.TASK_SETUP: 7})
    decision = decide(draft, summary((defect,), causes=causes), FRESH, rules)

    assert isinstance(decision, Repair)
    assert decision.invalidate == ("machine",)


def test_host_refusal_rejects_for_the_host_before_any_retry(draft, summary, rules):
    causes = Counter({Cause.MODEL_UNAVAILABLE: 5, Cause.SUBMISSION_UNSUPPORTED: 2})
    decision = decide(draft, summary(causes=causes), FRESH, rules)

    assert isinstance(decision, Reject)
    assert decision.kind == RejectKind.HOST
    assert decision.reasons == ("submission_unsupported: 2 trials ungraded",)


def test_rerunnable_incomplete_evidence_retries_its_most_common_cause(draft, summary, rules):
    causes = Counter({Cause.TOKEN_CONTRACT: 1, Cause.MODEL_UNAVAILABLE: 3, Cause.MACHINE_START: 3})
    assert decide(draft, summary(causes=causes), FRESH, rules) == Retry(cause=Cause.MACHINE_START, count=3)


def test_retry_is_not_bounded_by_the_repair_budget(draft, summary, rules):
    decision = decide(draft, summary(causes=Counter({Cause.MODEL_UNAVAILABLE: 8}), solved=0), SPENT, rules)
    assert decision == Retry(cause=Cause.MODEL_UNAVAILABLE, count=8)


@pytest.mark.parametrize("repairs", [0, 1, 2])
def test_a_band_finding_is_repaired_while_its_kind_has_repairs_left(draft, summary, repairs):
    rules = BandRules(too_easy=BandRule(1, BandChoice.REJECT), too_hard=BandRule(repairs, BandChoice.REJECT))
    too_hard = finding(FindingKind.TOO_HARD, "0 of 8 solved; 6 timed out")
    evidence = summary((too_hard,), solved=0)
    decisions = [
        decide(draft, evidence, history(used, 5, {FindingKind.TOO_HARD: used}), rules) for used in range(repairs + 1)
    ]

    for repair in decisions[:-1]:
        assert isinstance(repair, Repair)
        assert repair.brief.findings == (too_hard,)
        assert repair.invalidate == ("fixtures", "instruction")
    assert decisions[-1] == Reject(
        kind=RejectKind.TASK,
        reasons=(f"too_hard: 0 of 8 solved; 6 timed out (repairs spent: {repairs} of {repairs} for too_hard)",),
        summary=evidence,
    )


@pytest.mark.parametrize("choice", list(BandChoice))
def test_a_band_finding_past_its_kind_repairs_takes_the_consumers_choice(draft, summary, choice):
    rules = BandRules(too_easy=BandRule(1, choice), too_hard=BandRule(1, BandChoice.REJECT))
    evidence = summary((finding(FindingKind.TOO_EASY, "8 of 8 solved"),), solved=8)
    decision = decide(draft, evidence, history(1, 2, {FindingKind.TOO_EASY: 1}), rules)

    if choice is BandChoice.ACCEPT:
        assert decision == Accept(summary=evidence, band=BandOutcome.TOO_EASY)
    else:
        assert decision == Reject(
            kind=RejectKind.TASK,
            reasons=("too_easy: 8 of 8 solved (repairs spent: 1 of 1 for too_easy)",),
            summary=evidence,
        )


def test_the_choice_also_applies_when_the_repair_budget_is_spent(draft, summary):
    rules = BandRules(too_easy=BandRule(1, BandChoice.ACCEPT), too_hard=BandRule(1, BandChoice.REJECT))
    evidence = summary((finding(FindingKind.TOO_EASY, "8 of 8 solved"),), solved=8)

    assert decide(draft, evidence, history(0, 0), rules) == Accept(summary=evidence, band=BandOutcome.TOO_EASY)


def test_a_band_finding_of_the_other_kind_still_gets_its_repair(draft, summary, rules):
    decision = decide(
        draft, summary((finding(FindingKind.TOO_EASY),), solved=8), history(1, 2, {FindingKind.TOO_HARD: 1}), rules
    )
    assert isinstance(decision, Repair)


def test_decisive_findings_still_outrank_the_band_choice(draft, summary):
    rules = BandRules(too_easy=BandRule(0, BandChoice.ACCEPT), too_hard=BandRule(0, BandChoice.ACCEPT))
    violated = finding(FindingKind.CONTROL_VIOLATED, "neg-0 scored 1.0")
    evidence = summary((violated, finding(FindingKind.TOO_EASY, "8 of 8 solved")), solved=8)

    repair = decide(draft, evidence, FRESH, rules)
    assert isinstance(repair, Repair)
    assert repair.brief.findings == (violated,)
    assert decide(draft, evidence, SPENT, rules) == Reject(
        kind=RejectKind.BUDGET,
        reasons=("control_violated: neg-0 scored 1.0", "repairs exhausted: 2 of 2"),
        summary=evidence,
    )


def test_clean_complete_evidence_accepts(draft, summary, rules):
    clean = summary()
    assert decide(draft, clean, FRESH, rules) == Accept(summary=clean, band=BandOutcome.IN_BAND)


@pytest.mark.parametrize(
    "findings",
    [(finding(FindingKind.CONTROL_VIOLATED, "neg-0 scored 1.0"),), (finding(FindingKind.TOO_EASY, "8 of 8 solved"),)],
)
def test_a_repair_past_the_budget_rejects_for_budget(draft, summary, rules, findings):
    evidence = summary(findings)
    decision = decide(draft, evidence, SPENT, rules)

    assert isinstance(decision, Reject)
    assert decision.kind == RejectKind.BUDGET
    assert decision.summary == evidence
    assert decision.reasons[-1] == "repairs exhausted: 2 of 2"


def test_a_repair_whose_roles_ran_no_step_still_revises_the_program(draft, summary, rules):
    defect = Finding(kind=FindingKind.TASK_DEFECT, detail="setup exited 1", roles=(StepRole.ENVIRONMENT,))
    bare = replace(draft, provenance=replace(draft.provenance, steps=draft.provenance.steps[3:5]))
    decision = decide(bare, summary((defect,)), FRESH, rules)

    assert isinstance(decision, Repair)
    assert decision.invalidate == ()


def test_brief_carries_new_controls_the_author_can_copy_verbatim():
    second = Control(
        id="adv-shortcut-1-2",
        kind=ControlKind.NEGATIVE,
        category=ControlCategory.REWARD_HACK,
        concern=ControlConcern.SHORTCUT,
        author="adversary/shortcut/1#2",
        payload=Transcript((reply("395"),)),
        expect=Expectation(Outcome.GRADED, reward_max=REJECTION_CEILING),
    )
    findings = (
        finding(FindingKind.SHORTCUT_PASSED, new_controls=(SHORTCUT_CONTROL,)),
        finding(FindingKind.SHORTCUT_PASSED, "passed with no workspace files", new_controls=(second,)),
    )
    brief = render_brief(findings, ())

    blocks = re.findall(r"```json\n(.*?)\n```", brief.failure, re.DOTALL)
    assert len(blocks) == 1
    assert parse_controls(blocks[0]) == (SHORTCUT_CONTROL, second)
    assert (
        "Finding 2: shortcut_passed (revise the grader, controls steps)\npassed with no workspace files" in brief.failure
    )


def test_steps_for_names_each_step_once_in_build_order(draft):
    steps = (*draft.provenance.steps, draft.provenance.steps[3])
    assert steps_for([StepRole.CONTROLS, StepRole.GRADER], steps) == ("grader", "controls")


def test_notes_never_change_the_decision(draft, summary, rules):
    accepted = summary(notes=NOTES)
    assert decide(draft, accepted, FRESH, rules) == Accept(summary=accepted, band=BandOutcome.IN_BAND)

    shortcut = finding(FindingKind.SHORTCUT_PASSED, "listed three answers and passed", (SHORTCUT_CONTROL,))
    decision = decide(draft, summary((shortcut,), notes=NOTES), FRESH, rules)

    assert isinstance(decision, Repair)
    assert decision.brief.findings == (shortcut,)
    assert decision.brief.notes == NOTES
    assert decision.invalidate == ("grader", "controls")
    failure = decision.brief.failure
    assert failure.index(shortcut.detail) < failure.index(NOTES_HEADER) < failure.index(NEW_CONTROLS_HEADER)
    assert f"Note 2: shortcut_passed (grader, controls)\n{NOTES[1].detail}" in failure
    blocks = re.findall(r"```json\n(.*?)\n```", failure, re.DOTALL)
    assert [parse_controls(block) for block in blocks] == [(SHORTCUT_CONTROL,)]


def test_notes_ride_along_a_band_repair(draft, summary, rules):
    too_easy = finding(FindingKind.TOO_EASY, "8 of 8 solved")
    decision = decide(draft, summary((too_easy,), solved=8, notes=NOTES), FRESH, rules)

    assert isinstance(decision, Repair)
    assert decision.brief.findings == (too_easy,)
    assert decision.brief.notes == NOTES
    assert NOTES[0].detail in decision.brief.failure
    assert "```json" not in decision.brief.failure


def test_a_rejection_names_findings_not_notes(draft, summary, rules):
    violated = finding(FindingKind.CONTROL_VIOLATED, "neg-0 scored 1.0")
    decision = decide(draft, summary((violated,), notes=NOTES), SPENT, rules)

    assert isinstance(decision, Reject)
    assert decision.reasons == ("control_violated: neg-0 scored 1.0", "repairs exhausted: 2 of 2")


def test_render_brief_needs_a_finding():
    with pytest.raises(ValueError, match="at least one finding"):
        render_brief((), NOTES)
