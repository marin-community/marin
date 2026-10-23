# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The default review: a rule table from a validation round's result to a ``Decision``.

``default_review`` applies the rules in order, and the first that matches decides:

1. Host. A lowering a machine factory refused (``LoweringStatus.REFUSED``) or RolloutEngine rejected
   (``LoweringStatus.REJECTED``) says this host cannot run the task: ``Reject(HOST)``.
2. Violated control. The grader gave a control a grade outside its expectation, a defect the author
   can fix: ``Modify`` with one finding per violated control. The coordinator enforces its MODIFY budget.
3. Incomplete evidence. A permanent ungraded cause rejects the task (or the host for
   ``MACHINE_UNSUPPORTED``). Otherwise ``Retry`` names the most frequent retryable cause and its count.
4. Band. The solve rate of the solver trials, placed against the caller's ``BandRules``:
   ``Accept(band)`` for a place the rules accept, else ``Reject(TASK)``.

A round its violated control decided early (``ValidationResult.early``) left trials cancelled, and rule 2
decides it: its cancelled trials are never read as incomplete evidence, nor its partial solver trials as a
solve rate.

Adversary trials decide nothing here: whether a passing submission is a shortcut is the arbiter's
judgement, which a later rule will read.
"""

from collections import Counter
from dataclasses import dataclass

from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.outcome import RETRYABLE, Cause
from taskforge.stage50_validate.round import ControlResult, LoweringStatus, ValidationResult
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Modify, Reject, RejectKind, Retry


@dataclass(frozen=True)
class BandRules:
    """Where a solve rate falls against the calibration band, and which places review accepts.

    Attributes:
        low: The lowest solve rate in the band; a lower one is ``TOO_HARD``.
        high: The highest solve rate in the band; a higher one is ``TOO_EASY``.
        accepted: The places review accepts a task in; a task anywhere else is rejected.
    """

    low: float
    high: float
    accepted: frozenset[BandOutcome]

    def place(self, solve_rate: float) -> BandOutcome:
        if solve_rate < self.low:
            return BandOutcome.TOO_HARD
        if solve_rate > self.high:
            return BandOutcome.TOO_EASY
        return BandOutcome.IN_BAND


async def default_review(result: ValidationResult, *, band: BandRules) -> Decision:
    """Decide on ``result`` by the module's rule table.

    Raises:
        ValueError: the result was decided early on a control it does not hold as violated, or the evidence is
            complete and no solver trial was graded, so no solve rate exists.
    """
    lowering = result.lowering
    if lowering.status is LoweringStatus.REFUSED:
        refusals = tuple(f"the {r.where} machine was refused ({r.reason}): {r.detail}" for r in lowering.refusals)
        return Reject(RejectKind.HOST, refusals)
    if lowering.status is LoweringStatus.REJECTED:
        return Reject(RejectKind.HOST, (f"RolloutEngine rejected the lowered task: {lowering.detail}",))
    findings = tuple(_finding(control) for control in result.controls if control.verdict is ControlVerdict.VIOLATED)
    if findings:
        return Modify(findings)
    if result.early is not None:
        raise ValueError(
            f"{result.workpiece_id} round {result.round} was decided early on control {result.early.control}, "
            "which it does not hold as violated"
        )
    solver = result.solver
    assert solver is not None, "a lowered round has solver evidence"
    ungraded = Counter(solver.ungraded)
    ungraded.update(control.cause for control in result.controls if control.cause is not None)
    permanent = {cause: count for cause, count in ungraded.items() if cause not in RETRYABLE}
    if permanent:
        kind = RejectKind.HOST if Cause.MACHINE_UNSUPPORTED in permanent else RejectKind.TASK
        reasons = tuple(f"{count} ungraded trial(s): {cause}" for cause, count in sorted(permanent.items()))
        return Reject(kind, reasons)
    if ungraded:
        cause, count = min(ungraded.items(), key=lambda item: (-item[1], list(Cause).index(item[0])))
        return Retry(cause, count)
    stats = solver.stats
    if stats.graded == 0:
        raise ValueError(f"{result.workpiece_id} round {result.round} graded no solver trial to place in the band")
    place = band.place(stats.solved / stats.graded)
    if place in band.accepted:
        return Accept(place)
    return Reject(RejectKind.TASK, (f"{stats.solved} of {stats.graded} solver trials solved the task: {place}",))


def _finding(control: ControlResult) -> str:
    return (
        f"the {control.kind} control {control.id} earned reward {control.reward}, which its expectation rejects "
        f"({control.attempt_file})"
    )
