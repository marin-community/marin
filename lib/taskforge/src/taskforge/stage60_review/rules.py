# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review complete validation evidence before accepting or requesting author repair.

A refused host, failed solver/control/adversary, or incomplete adversary submission produces ``Hold``.
Unavailable execution takes precedence over known task defects and preserves the author revision budget.
Known authored defects and violated controls produce ``Modify``. Complete solver evidence in an accepted
band produces ``Accept``; another band requests repair. Exhausted retry/revision
budgets abandon the workpiece.
The adversary's shortcut claim still needs separate adjudication; availability is checked here.
"""

import math
from collections import Counter
from dataclasses import dataclass

from taskcompendium.grading_result import Outcome as GradeStatus

from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.outcome import GRADED_STATUSES, TASK_DEFECTS, Cause
from taskforge.stage50_validate.round import ControlResult, LoweringStatus, TrialStatus, ValidationResult
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Hold, Modify, Reject, RejectKind


@dataclass(frozen=True)
class BandRules:
    """Where a solve rate falls against the calibration band, and which places review accepts.

    Attributes:
        low: The lowest solve rate in the band; a lower one is ``TOO_HARD``.
        high: The highest solve rate in the band; a higher one is ``TOO_EASY``.
        accepted: The places review accepts a task in; a task anywhere else needs repair.
    """

    low: float
    high: float
    accepted: frozenset[BandOutcome]

    def __post_init__(self) -> None:
        if not 0 <= self.low <= self.high <= 1:
            raise ValueError("the solve-rate band must satisfy 0 <= low <= high <= 1")
        if self.accepted - {BandOutcome.IN_BAND}:
            raise ValueError("only completed in-band evidence can be accepted")

    def place(self, solve_rate: float) -> BandOutcome:
        if solve_rate < self.low:
            return BandOutcome.TOO_HARD
        if solve_rate > self.high:
            return BandOutcome.TOO_EASY
        return BandOutcome.IN_BAND


def evidence_gaps(result: ValidationResult) -> tuple[str, ...]:
    """Why the evidence cannot support a terminal accept/reject, including inconsistent summaries.

    Authored defects can still receive repair feedback. They cannot become conclusive
    task verdicts while any required evaluation remains unavailable or incomplete.
    """
    gaps: list[str] = []
    if result.lowering.status is not LoweringStatus.LOWERED:
        gaps.append(f"lowering is {result.lowering.status}: {result.lowering.detail}")
    if result.early is not None:
        gaps.append(f"validation stopped early at control {result.early.control}")
    solver = result.solver
    if solver is None:
        gaps.append("solver evidence is absent")
    else:
        stats = solver.stats
        if stats.graded == 0:
            gaps.append("no solver trial has a usable grade")
        if stats.graded != len(solver.trials) or solver.ungraded:
            gaps.append("solver trials are incomplete or disagree with their summary")
        if not 0 <= stats.solved <= stats.graded or stats.timed_out:
            gaps.append("solver totals contain an invalid count or incomplete timeout")
        if stats.mean_reward is None or not math.isfinite(stats.mean_reward):
            gaps.append("solver mean reward is unavailable or non-finite")
        for trial in solver.trials:
            if (
                trial.status is not TrialStatus.GRADED
                or trial.cause is not None
                or trial.reward is None
                or not math.isfinite(trial.reward)
            ):
                gaps.append(f"solver trial {trial.trial} has no completed numeric evaluation")
    if not result.controls:
        gaps.append("required control evidence is absent")
    for control in result.controls:
        if (
            control.verdict is ControlVerdict.UNGRADED
            or control.cause is not None
            or control.reward is None
            or not math.isfinite(control.reward)
        ):
            gaps.append(f"control {control.id} has no completed numeric evaluation")
    for trial in result.adversary:
        if trial.status is not TrialStatus.GRADED or trial.cause is not None or not trial.submissions:
            gaps.append(f"adversary trial {trial.trial} is incomplete")
        for submission in trial.submissions:
            if (
                submission.status not in GRADED_STATUSES
                or (submission.status is GradeStatus.GRADED and submission.reward is None)
                or (submission.reward is not None and not math.isfinite(submission.reward))
            ):
                gaps.append(f"adversary trial {trial.trial} submission {submission.ordinal} is incomplete")
    return tuple(gaps)


async def default_review(result: ValidationResult, *, band: BandRules) -> Decision:
    """Hold unavailable evidence, request repair for authored defects, or accept a complete in-band round."""
    lowering = result.lowering
    if lowering.status is LoweringStatus.REFUSED:
        refusals = tuple(f"the {r.where} machine was refused ({r.reason}): {r.detail}" for r in lowering.refusals)
        return Hold(refusals)
    if lowering.status is LoweringStatus.REJECTED:
        return Hold((f"RolloutEngine rejected the lowered task: {lowering.detail}",))
    solver = result.solver
    if solver is None:
        return Hold(("solver evidence is absent",))
    ungraded = Counter(solver.ungraded)
    ungraded.update(control.cause for control in result.controls if control.cause is not None)
    ungraded.update(trial.cause for trial in result.adversary if trial.cause is not None)
    # An adversary may have a usable final answer after an earlier verifier outage. Every required
    # submission must still be accounted for; a passing answer cannot erase the unavailable evidence.
    for trial in result.adversary:
        if trial.cause is None:
            ungraded.update(
                Cause.GRADER_INFRA for submission in trial.submissions if submission.status not in GRADED_STATUSES
            )
    if Cause.BUDGET_EXHAUSTED in ungraded:
        return Reject(RejectKind.BUDGET, ("an explicitly configured trial budget was exhausted",))
    unavailable = {cause: count for cause, count in ungraded.items() if cause not in TASK_DEFECTS}
    if unavailable:
        return Hold(tuple(f"{count} ungraded trial(s): {cause}" for cause, count in sorted(unavailable.items())))
    findings = tuple(_finding(control) for control in result.controls if control.verdict is ControlVerdict.VIOLATED)
    if findings:
        return Modify(findings)
    if result.early is not None:
        return Hold((f"validation stopped early at control {result.early.control} without a violated control",))
    if ungraded:
        return Modify(tuple(f"{count} ungraded trial(s): {cause}" for cause, count in sorted(ungraded.items())))
    gaps = evidence_gaps(result)
    if gaps:
        return Hold(gaps)
    stats = solver.stats
    place = band.place(stats.solved / stats.graded)
    if place in band.accepted:
        return Accept(place)
    return Modify((f"{stats.solved} of {stats.graded} solver trials solved the task: {place}",))


def _finding(control: ControlResult) -> str:
    return (
        f"the {control.kind} control {control.id} earned reward {control.reward}, which its expectation rejects "
        f"({control.attempt_file})"
    )
