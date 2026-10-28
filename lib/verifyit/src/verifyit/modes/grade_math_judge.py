# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic math equivalence followed by a semantic judge for unresolved answers."""

from pathlib import Path

from verifyit.grade import InvalidTask, Reward, Status, empty_output_policy, read_output, scored
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate, validate_judge_spec
from verifyit.modes.grade_math import grade_math_text_candidate, validate_math
from verifyit.spec import JudgeRuntimeSource, MathJudgeSpec, MathProfile


def validate_math_judge(spec: MathJudgeSpec) -> None:
    """Validate both stages before grading or making a remote request."""
    empty_output_policy(spec)
    validate_math(spec.math)
    validate_judge_spec(spec.judge)
    if spec.math.profile is not MathProfile.EVALCHEMY_OLYMPIAD:
        raise InvalidTask("math-judge requires the Olympiad extraction profile")


def grade_math_judge_candidate(
    spec: MathJudgeSpec,
    candidate: str,
    *,
    connection: JudgeConnection | None = None,
    runtime: JudgeRuntimeSource | None = JudgeRuntimeSource.ENVIRONMENT,
    context: str = "",
) -> Reward:
    """Preserve deterministic successes and extraction failures before consulting the judge."""
    validate_math_judge(spec)
    verdict = grade_math_text_candidate(spec.math, candidate)
    if verdict.status is not Status.SCORED or verdict.reward == 1:
        return verdict
    answer = verdict.detail["extracted"]
    if not answer.strip():
        return verdict
    result = grade_judge_candidate(spec.judge, answer, connection=connection, runtime=runtime, context=context)
    return Reward(result.reward, result.status, {**result.detail, "extracted": answer, "method": "llm_judge"})


def grade(spec: MathJudgeSpec, tests_dir: Path, workspace: Path) -> Reward:
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    context = (tests_dir / spec.judge.context).read_text() if spec.judge.context else ""
    return grade_math_judge_candidate(spec, text, context=context)
