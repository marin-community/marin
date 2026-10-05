# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure candidate grading using the standard verifier specifications.

Callers own submission extraction. These graders neither read files nor inspect a harness trace.
"""

from typing import Any

from verifyit.grade import Reward, numeric_tolerance
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_math import grade_numeric_candidate
from verifyit.modes.grade_mcq import grade_mcq_candidate
from verifyit.modes.grade_predicted_action import validate_predicted_action
from verifyit.spec import ExactSpec, McqSpec, Mode, NumericSpec, PredictedActionSpec, spec_from_table

TextSpec = ExactSpec | NumericSpec | McqSpec
CandidateSpec = TextSpec | PredictedActionSpec


def supports_candidate_mode(mode: str) -> bool:
    return mode in (Mode.EXACT, Mode.NUMERIC, Mode.MCQ, Mode.PREDICTED_ACTION)


def candidate_spec(mode: str, parameters: dict[str, Any]) -> CandidateSpec:
    """Validate standard private configuration for an already-extracted candidate."""
    if not supports_candidate_mode(mode):
        raise NotImplementedError(f"No pure candidate grader for mode {mode!r}")
    if "mode" in parameters:
        raise ValueError("Candidate parameters must not override the verifier mode")
    spec = spec_from_table({"mode": mode, **parameters})
    if isinstance(spec, ExactSpec):
        grade_exact_candidate(spec, "")
        return spec
    if isinstance(spec, NumericSpec):
        numeric_tolerance(spec)
        return spec
    if isinstance(spec, McqSpec):
        grade_mcq_candidate(spec, spec.expected)
        return spec
    assert isinstance(spec, PredictedActionSpec)
    validate_predicted_action(spec)
    return spec


def grade_text_candidate(spec: TextSpec, candidate: str) -> Reward:
    """Score text whose presentation has already been removed by the caller."""
    if isinstance(spec, ExactSpec):
        return grade_exact_candidate(spec, candidate)
    if isinstance(spec, McqSpec):
        return grade_mcq_candidate(spec, candidate)
    try:
        value = float(candidate.strip())
    except ValueError:
        value = float("nan")
    return grade_numeric_candidate(spec, value)
