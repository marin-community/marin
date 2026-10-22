# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""default_review: one test per row of the rule table, and the order between the rows."""

from collections import Counter
from collections.abc import Sequence

import pytest
from taskcompendium.grading_result import Outcome as GradeStatus

from taskforge.sandbox.factories import MachineRole, Refusal, RefusalReason
from taskforge.spec.controls import ControlKind
from taskforge.stage50_validate.adversary import AdversaryMode, Claim, ClaimKind
from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.evidence import RewardStats
from taskforge.stage50_validate.outcome import Cause
from taskforge.stage50_validate.round import (
    AdversaryResult,
    ControlResult,
    Lowering,
    LoweringStatus,
    SolverEvidence,
    SubmissionResult,
    TrialResult,
    TrialStatus,
    ValidationResult,
)
from taskforge.stage60_review.decision import Accept, BandOutcome, Modify, Reject, RejectKind, Retry
from taskforge.stage60_review.rules import BandRules, ReviewContext, default_review

BAND = BandRules(low=0.25, high=0.75, accepted=frozenset({BandOutcome.IN_BAND, BandOutcome.TOO_HARD}))
LOWERED = Lowering(LoweringStatus.LOWERED, "", (), None, "lowered-task")
REFUSAL = Refusal(RefusalReason.NO_FACTORY, MachineRole.TASK, "this host has no docker factory")


def solver(solved: int, failed: int, ungraded: Sequence[Cause] = ()) -> SolverEvidence:
    """Solver evidence of ``solved`` passing and ``failed`` failing graded trials, then one ungraded trial per cause."""
    rewards = [1.0] * solved + [0.0] * failed
    trials = [
        TrialResult(str(index), TrialStatus.GRADED, reward, None, f"solver/{index}/attempt-0.json")
        for index, reward in enumerate(rewards)
    ]
    trials += [
        TrialResult(str(index), TrialStatus.UNGRADED, None, cause, f"solver/{index}/attempt-1.json")
        for index, cause in enumerate(ungraded, start=len(rewards))
    ]
    graded = solved + failed
    stats = RewardStats(graded, solved / graded if graded else None, solved, 0, 0)
    return SolverEvidence(tuple(trials), stats, dict(Counter(ungraded)))


def control(control_id: str, verdict: ControlVerdict, cause: Cause | None = None) -> ControlResult:
    """A negative control: met at reward 0, violated at reward 1, or ungraded for ``cause``."""
    reward = {ControlVerdict.MET: 0.0, ControlVerdict.VIOLATED: 1.0}.get(verdict)
    return ControlResult(
        control_id, ControlKind.NEGATIVE, verdict, reward, cause, f"control/{control_id}/attempt-0.json"
    )


MET = control("N1-shortcut", ControlVerdict.MET)
VIOLATED = control("N2-off-by-one", ControlVerdict.VIOLATED)


IN_BAND = solver(2, 2)


def result(
    lowering: Lowering = LOWERED,
    evidence: SolverEvidence | None = IN_BAND,
    controls: tuple[ControlResult, ...] = (MET,),
    adversary: tuple[AdversaryResult, ...] = (),
) -> ValidationResult:
    return ValidationResult("workpiece", 0, lowering, evidence, controls, adversary)


def context(modify_rounds_left: int = 1) -> ReviewContext:
    return ReviewContext(round=0, modify_rounds_left=modify_rounds_left, band=BAND)


@pytest.mark.parametrize(
    "rejected",
    [
        result(
            Lowering(LoweringStatus.REFUSED, "", (REFUSAL,), None, "lowered-task"),
            solver(0, 0, [Cause.MACHINE_UNSUPPORTED] * 4),
            (VIOLATED, control("N3", ControlVerdict.UNGRADED, Cause.MACHINE_UNSUPPORTED)),
        ),
        result(Lowering(LoweringStatus.REJECTED, "the verifier machine has no factory", (), None, None), None, ()),
    ],
    ids=["refused", "rejected"],
)
async def test_a_refused_or_rejected_lowering_rejects_on_the_host_before_any_other_rule(rejected):
    decision = await default_review(rejected, context())

    assert isinstance(decision, Reject) and decision.kind is RejectKind.HOST


@pytest.mark.parametrize(
    ("modify_rounds_left", "decided"),
    [(1, Modify), (0, Reject)],
    ids=["a modify round left", "no modify round left"],
)
async def test_a_violated_control_asks_for_a_modify_round_while_one_remains_and_rejects_the_task_after(
    modify_rounds_left, decided
):
    decision = await default_review(result(controls=(MET, VIOLATED)), context(modify_rounds_left))

    assert isinstance(decision, decided)
    findings = decision.findings if isinstance(decision, Modify) else decision.reasons
    assert len(findings) == 1 and VIOLATED.id in findings[0]
    if isinstance(decision, Reject):
        assert decision.kind is RejectKind.TASK


async def test_a_violated_control_decides_before_incomplete_evidence():
    incomplete = result(
        evidence=solver(1, 0, [Cause.MODEL_TIMEOUT]),
        controls=(VIOLATED, control("N3", ControlVerdict.UNGRADED, Cause.GRADER_TIMEOUT)),
    )

    decision = await default_review(incomplete, context())

    assert isinstance(decision, Modify) and len(decision.findings) == 1


async def test_incomplete_evidence_retries_for_the_cause_that_left_the_most_solver_trials_and_controls_ungraded():
    incomplete = result(
        evidence=solver(2, 0, [Cause.MODEL_TIMEOUT, Cause.GRADER_TIMEOUT]),
        controls=(MET, control("N3", ControlVerdict.UNGRADED, Cause.GRADER_TIMEOUT)),
    )

    assert await default_review(incomplete, context()) == Retry(Cause.GRADER_TIMEOUT, 2)


async def test_a_solve_rate_in_a_place_the_band_rules_refuse_rejects_the_task():
    decision = await default_review(result(evidence=solver(4, 0)), context())

    assert isinstance(decision, Reject) and decision.kind is RejectKind.TASK


@pytest.mark.parametrize(
    ("solved", "band"),
    [(2, BandOutcome.IN_BAND), (1, BandOutcome.IN_BAND), (0, BandOutcome.TOO_HARD)],
    ids=["in band", "at the low edge", "too hard and accepted"],
)
async def test_complete_evidence_is_accepted_in_the_place_the_callers_band_gives_it(solved, band):
    assert await default_review(result(evidence=solver(solved, 4 - solved)), context()) == Accept(band)


async def test_an_adversary_shortcut_claim_decides_nothing():
    shortcut = AdversaryResult(
        trial="0",
        mode=AdversaryMode.WHITEBOX,
        status=TrialStatus.GRADED,
        cause=None,
        claim=Claim(ClaimKind.SHORTCUT, "the grader accepts 6 * 7", "expected 42"),
        submissions=(SubmissionResult(1, GradeStatus.GRADED, 1.0, True, "adversary/0/submissions/0/control/s1"),),
        attempt_file="adversary/0/attempt-0.json",
    )

    assert await default_review(result(adversary=(shortcut,)), context()) == Accept(BandOutcome.IN_BAND)
