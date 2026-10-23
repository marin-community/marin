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
    EarlyDecision,
    Lowering,
    LoweringStatus,
    SolverEvidence,
    SubmissionResult,
    TrialResult,
    TrialStatus,
    ValidationResult,
)
from taskforge.stage60_review.decision import Accept, BandOutcome, Hold, Modify
from taskforge.stage60_review.rules import BandRules, default_review

BAND = BandRules(low=0.25, high=0.75, accepted=frozenset({BandOutcome.IN_BAND}))
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
    early: EarlyDecision | None = None,
) -> ValidationResult:
    return ValidationResult("workpiece", 0, lowering, evidence, controls, adversary, early)


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
async def test_a_refused_or_rejected_lowering_holds_before_any_task_decision(rejected):
    decision = await default_review(rejected, band=BAND)

    assert isinstance(decision, Hold)


async def test_a_violated_control_asks_for_a_modify_round():
    decision = await default_review(result(controls=(MET, VIOLATED)), band=BAND)

    assert isinstance(decision, Modify)
    assert len(decision.findings) == 1 and VIOLATED.id in decision.findings[0]


async def test_infrastructure_holds_even_with_a_known_control_defect():
    incomplete = result(
        evidence=solver(1, 0, [Cause.MODEL_TIMEOUT]),
        controls=(VIOLATED, control("N3", ControlVerdict.UNGRADED, Cause.GRADER_TIMEOUT)),
    )

    decision = await default_review(incomplete, band=BAND)

    assert isinstance(decision, Hold)


async def test_a_round_decided_early_is_decided_by_its_violated_control_never_as_incomplete_evidence():
    """The round cancelled three of four solver trials, so the solver evidence alone is no measurement."""
    early = result(
        evidence=solver(1, 0),
        controls=(MET, VIOLATED),
        early=EarlyDecision(VIOLATED.id, ("solver/1", "solver/2", "solver/3", "adversary/0")),
    )

    assert isinstance(await default_review(early, band=BAND), Modify)


async def test_incomplete_evidence_holds_with_all_unavailable_causes():
    incomplete = result(
        evidence=solver(2, 0, [Cause.MODEL_TIMEOUT, Cause.GRADER_TIMEOUT]),
        controls=(MET, control("N3", ControlVerdict.UNGRADED, Cause.GRADER_TIMEOUT)),
    )

    assert await default_review(incomplete, band=BAND) == Hold(
        ("2 ungraded trial(s): grader_timeout", "1 ungraded trial(s): model_timeout")
    )


async def test_a_solve_rate_outside_the_band_requests_repair():
    decision = await default_review(result(evidence=solver(4, 0)), band=BAND)

    assert isinstance(decision, Modify)


@pytest.mark.parametrize(
    ("solved", "band"),
    [(2, BandOutcome.IN_BAND), (1, BandOutcome.IN_BAND)],
    ids=["in band", "at the low edge"],
)
async def test_complete_evidence_is_accepted_in_the_place_the_callers_band_gives_it(solved, band):
    assert await default_review(result(evidence=solver(solved, 4 - solved)), band=BAND) == Accept(band)


@pytest.mark.parametrize(
    "cause,expected",
    [
        (Cause.GRADER_REWARD_RANGE, Modify),
        (Cause.MODEL_REJECTED, Hold),
        (Cause.MACHINE_UNSUPPORTED, Hold),
        (Cause.GRADER_INFRA, Hold),
        (Cause.GENERATION_LIMIT, Hold),
        (Cause.UNCLASSIFIED, Hold),
    ],
)
@pytest.mark.parametrize("retryable", [(), (Cause.MODEL_TIMEOUT,) * 3], ids=["permanent-only", "mixed"])
async def test_ungraded_causes_separate_authored_defects_from_missing_evidence(cause, expected, retryable):
    decision = await default_review(result(evidence=solver(2, 0, [cause, *retryable])), band=BAND)

    assert isinstance(decision, Hold if retryable else expected)


async def test_a_control_defect_does_not_turn_a_solver_outage_into_rejection():
    incomplete = result(
        evidence=solver(2, 0, [Cause.MODEL_TIMEOUT]),
        controls=(control("N3", ControlVerdict.UNGRADED, Cause.GRADER_INVALID_REWARD),),
    )
    decision = await default_review(incomplete, band=BAND)

    assert decision == Hold(("1 ungraded trial(s): model_timeout",))


@pytest.mark.parametrize("cause", [Cause.MODEL_UNAVAILABLE, Cause.GENERATION_LIMIT, None])
async def test_an_adversary_outage_cannot_be_erased_by_other_passing_evidence(cause):
    adversary = AdversaryResult(
        "0",
        AdversaryMode.WHITEBOX,
        TrialStatus.UNGRADED if cause else TrialStatus.GRADED,
        cause,
        Claim(ClaimKind.NO_SHORTCUT, "", ""),
        (
            SubmissionResult(0, GradeStatus.INFRA_ERROR, None, False, "adversary/0/failed.json"),
            SubmissionResult(1, GradeStatus.GRADED, 1.0, True, "adversary/0/passed.json"),
        ),
        "adversary/0/attempt-0.json",
    )
    decision = await default_review(result(adversary=(adversary,)), band=BAND)
    assert isinstance(decision, Hold)
