# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Review complete evidence before accepting or requesting author repair."""

import math
from collections import Counter

from taskcompendium.grading_result import GradeResult, Outcome

from forge.model.review import (
    Accept,
    BandRules,
    Cause,
    Decision,
    Incomplete,
    Modify,
    Reject,
    RejectKind,
    ValidationResult,
)
from forge.stages.classify import TASK_DEFECTS, grade_cause


def passed(grade: GradeResult) -> bool:
    if grade.status is Outcome.SUBMISSION_FAILURE:
        return False
    if grade.passed is not None:
        return grade.passed
    reward, score_max = grade.reward or 0.0, grade.score_max
    return reward >= score_max or math.isclose(reward, score_max, rel_tol=1e-9, abs_tol=1e-9)


def default_review(result: ValidationResult, band: BandRules) -> Decision:
    if result.lowering_error is not None:
        return Incomplete((f"Lowering failed: {result.lowering_error}",))
    if not result.solver or not result.controls:
        return Incomplete(("Required solver or control evidence is absent",))
    causes = Counter(
        trial.cause or (Cause.NO_GRADE if trial.grade is None else grade_cause(trial.grade))
        for trial in (*result.solver, *result.controls)
    )
    causes.update(
        finding.cause or (Cause.NO_GRADE if finding.grade is None else grade_cause(finding.grade))
        for adversary in result.adversary
        for finding in adversary.findings
    )
    unavailable = tuple(
        f"{count} ungraded trial(s): {cause}"
        for cause, count in sorted((cause, count) for cause, count in causes.items() if cause is not None)
        if cause not in TASK_DEFECTS and cause is not Cause.BUDGET_EXHAUSTED
    )
    unavailable += tuple(
        f"{trial.name}: {reason}"
        for trial in result.adversary
        if not trial.completed
        for reason in trial.reasons or ("Adversary search is incomplete (reason unavailable)",)
    )
    if unavailable:
        return Incomplete(unavailable)
    if causes[Cause.BUDGET_EXHAUSTED]:
        return Reject(RejectKind.BUDGET, ("An explicitly configured trial budget was exhausted",))
    findings = tuple(
        f"Control {control.index} violates its expected reward" for control in result.controls if not control.met
    )
    findings += tuple(f"{count} ungraded trial(s): {cause}" for cause, count in causes.items() if cause in TASK_DEFECTS)
    findings += tuple(
        f"{trial.name}: {finding.claim}" for trial in result.adversary for finding in trial.findings if finding.met
    )
    if findings:
        return Modify(findings)
    solved = sum(passed(trial.grade) for trial in result.solver if trial.grade is not None)
    place = band.place(solved / len(result.solver))
    return (
        Accept(place)
        if place in band.accepted
        else Modify((f"{solved} of {len(result.solver)} trials solved: {place}",))
    )
