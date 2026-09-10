# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidates that must score 1 and 0 for a spec, used to verify a task's grader before release.

Only output-file modes have probes: their whole grading behaviour is a function of the spec and
the candidate text. Execution modes need the task image and are exercised by converter authors
against samples instead.
"""

from tasktrove_verify.spec import ExactSpec, JudgeSpec, MathSpec, McqSpec, NumericSpec, Spec


def positive_candidate(spec: Spec) -> str | None:
    """A candidate the grader must score 1.0, or ``None`` when the mode has no probe."""
    if isinstance(spec, McqSpec):
        return f"Answer: {spec.expected}"
    if isinstance(spec, MathSpec):
        return f"\\boxed{{{spec.expected}}}"
    if isinstance(spec, NumericSpec):
        return f"\\boxed{{{spec.expected}}}"
    if isinstance(spec, ExactSpec):
        return "\n".join(spec.expected)
    if isinstance(spec, JudgeSpec):
        gated = spec.rubric == "reference" and spec.exact_gate and spec.references and not spec.constraints
        return spec.references[0] if gated else None
    return None


def negative_candidate(spec: Spec) -> str | None:
    """A candidate the grader must score 0.0, or ``None`` when no safe perturbation exists."""
    if isinstance(spec, McqSpec):
        other = "B" if spec.expected.upper() != "B" else "A"
        return f"Answer: {other}"
    if isinstance(spec, NumericSpec):
        return f"\\boxed{{{spec.expected + 1.0}}}"
    if isinstance(spec, ExactSpec) and len(spec.expected) > 1 and spec.ordered:
        return "\n".join(reversed(spec.expected))
    return None
