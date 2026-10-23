# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""default_review: one test per row of the rule table, and the order between the rows."""

from collections import Counter
from collections.abc import Sequence

import pytest

from taskforge.sandbox.factories import MachineRole, Refusal, RefusalReason
from taskforge.spec.controls import ControlKind
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
    TrialResult,
    TrialStatus,
    ValidationResult,
)
from taskforge.stage60_review.decision import Accept, BandOutcome, Modify, Reject, RejectKind, Retry
from taskforge.stage60_review.rules import BandRules, default_review

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
async def test_a_refused_or_rejected_lowering_rejects_on_the_host_before_any_other_rule(rejected):
    decision = await default_review(rejected, band=BAND)

    assert isinstance(decision, Reject) and decision.kind is RejectKind.HOST


async def test_a_violated_control_asks_for_a_modify_round():
    decision = await default_review(result(controls=(MET, VIOLATED)), band=BAND)

    assert isinstance(decision, Modify)
    assert len(decision.findings) == 1 and VIOLATED.id in decision.findings[0]


async def test_a_violated_control_decides_before_incomplete_evidence():
    incomplete = result(
        evidence=solver(1, 0, [Cause.MODEL_TIMEOUT]),
        controls=(VIOLATED, control("N3", ControlVerdict.UNGRADED, Cause.GRADER_TIMEOUT)),
    )

    decision = await default_review(incomplete, band=BAND)

    assert isinstance(decision, Modify) and len(decision.findings) == 1


async def test_a_round_decided_early_is_decided_by_its_violated_control_never_as_incomplete_evidence():
    """The round cancelled three of four solver trials, so the solver evidence alone is no measurement."""
    early = result(
        evidence=solver(1, 0),
        controls=(MET, VIOLATED),
        early=EarlyDecision(VIOLATED.id, ("solver/1", "solver/2", "solver/3", "adversary/0")),
    )

    assert isinstance(await default_review(early, band=BAND), Modify)


async def test_incomplete_evidence_retries_for_the_cause_that_left_the_most_solver_trials_and_controls_ungraded():
    incomplete = result(
        evidence=solver(2, 0, [Cause.MODEL_TIMEOUT, Cause.GRADER_TIMEOUT]),
        controls=(MET, control("N3", ControlVerdict.UNGRADED, Cause.GRADER_TIMEOUT)),
    )

    assert await default_review(incomplete, band=BAND) == Retry(Cause.GRADER_TIMEOUT, 2)


async def test_a_solve_rate_in_a_place_the_band_rules_refuse_rejects_the_task():
    decision = await default_review(result(evidence=solver(4, 0)), band=BAND)

    assert isinstance(decision, Reject) and decision.kind is RejectKind.TASK


@pytest.mark.parametrize(
    ("solved", "band"),
    [(2, BandOutcome.IN_BAND), (1, BandOutcome.IN_BAND), (0, BandOutcome.TOO_HARD)],
    ids=["in band", "at the low edge", "too hard and accepted"],
)
async def test_complete_evidence_is_accepted_in_the_place_the_callers_band_gives_it(solved, band):
    assert await default_review(result(evidence=solver(solved, 4 - solved)), band=BAND) == Accept(band)


@pytest.mark.parametrize(
    "cause,kind",
    [
        (Cause.GRADER_REWARD_RANGE, RejectKind.TASK),
        (Cause.MODEL_REJECTED, RejectKind.TASK),
        (Cause.MACHINE_UNSUPPORTED, RejectKind.HOST),
    ],
)
@pytest.mark.parametrize("retryable", [(), (Cause.MODEL_TIMEOUT,) * 3], ids=["permanent-only", "mixed"])
async def test_a_permanent_ungraded_cause_rejects_without_spending_retries(cause, kind, retryable):
    decision = await default_review(result(evidence=solver(2, 0, [cause, *retryable])), band=BAND)

    assert decision == Reject(kind, (f"1 ungraded trial(s): {cause}",))


async def test_a_permanent_control_failure_rejects_before_retryable_solver_failures():
    incomplete = result(
        evidence=solver(2, 0, [Cause.MODEL_TIMEOUT]),
        controls=(control("N3", ControlVerdict.UNGRADED, Cause.GRADER_INVALID_REWARD),),
    )
    decision = await default_review(incomplete, band=BAND)

    assert decision == Reject(RejectKind.TASK, (f"1 ungraded trial(s): {Cause.GRADER_INVALID_REWARD}",))
