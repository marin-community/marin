# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JEEBench source preparation composed with Exact and Numeric primitives."""

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from tasktrove_verify.grade import InvalidTask, Reward, invalid_task
from tasktrove_verify.modes.grade_exact import grade_collection_subset, grade_exact_candidate
from tasktrove_verify.modes.grade_math import grade_numeric_candidate
from tasktrove_verify.spec import ExactSpec, NumericSpec

LETTERS = "ABCD"


class JEEPolicy(StrEnum):
    SOURCE = "source_case_sensitive_ad_membership_python_float_abs_001_v1"


@dataclass(frozen=True)
class JEEInputs:
    expected: object
    candidate: object
    question_type: str


@dataclass(frozen=True)
class PreparedJEE:
    raw: JEEInputs
    expected: tuple[str, ...] | float
    candidate: tuple[str, ...] | float | None
    policy: JEEPolicy


def capture_jee_input(expected: object, candidate: object, question_type: str) -> JEEInputs:
    """Capture the primitive input before source membership or float preparation."""
    return JEEInputs(expected, candidate, question_type)


def prepare_jee_input(raw: JEEInputs, policy: JEEPolicy) -> PreparedJEE:
    """Apply named source rules after validating the complete trusted reference."""
    if not isinstance(policy, JEEPolicy):
        raise InvalidTask("unknown JEEBench preparation policy")
    if not isinstance(raw.question_type, str) or raw.question_type not in {"MCQ", "MCQ(multiple)", "Integer", "Numeric"}:
        raise InvalidTask("unknown JEEBench question type")
    if raw.question_type in {"MCQ", "MCQ(multiple)"}:
        if not isinstance(raw.expected, str):
            raise InvalidTask("JEEBench choice reference must be text")
        reference = tuple(letter for letter in LETTERS if letter in raw.expected)
        if not reference or any(letter in raw.expected for letter in "EFGHIJKLMNOPQRSTUVWXYZ"):
            raise InvalidTask("JEEBench choice reference must define valid A-D options")
        if raw.question_type == "MCQ" and len(reference) != 1:
            raise InvalidTask("JEEBench single choice must define exactly one option")
        candidate = (
            tuple(letter for letter in LETTERS if letter in raw.candidate) if isinstance(raw.candidate, str) else ()
        )
        return PreparedJEE(raw, reference, candidate, policy)
    if isinstance(raw.expected, bool) or not isinstance(raw.expected, str | int | float):
        raise InvalidTask("JEEBench numeric reference must be text or a nonboolean number")
    try:
        reference_value = float(raw.expected)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidTask("JEEBench numeric reference is not a valid number") from error
    if not math.isfinite(reference_value):
        raise InvalidTask("JEEBench numeric reference must be finite")
    candidate_value = None
    if isinstance(raw.candidate, str):
        try:
            candidate_value = float(raw.candidate)
        except (ValueError, OverflowError):
            pass
    return PreparedJEE(raw, reference_value, candidate_value, policy)


def grade_prepared_jee(prepared: PreparedJEE) -> Reward:
    """Let Exact or Numeric determine reward for prepared source answers."""
    if isinstance(prepared.expected, tuple):
        assert isinstance(prepared.candidate, tuple)
        if prepared.raw.question_type == "MCQ(multiple)":
            result = grade_collection_subset(prepared.expected, prepared.candidate, item_credit=0.25)
        else:
            result = grade_exact_candidate(
                ExactSpec(prepared.expected, ignore_case=False, ignore_whitespace=False),
                ",".join(prepared.candidate),
            )
    else:
        assert prepared.candidate is None or isinstance(prepared.candidate, float)
        value = math.nan if prepared.candidate is None else prepared.candidate
        result = grade_numeric_candidate(NumericSpec(prepared.expected, tolerance_abs=0.01, tolerance_rel=0), value)
    return replace(result, detail={**result.detail, "policy": prepared.policy.value})


def grade_jee_answer(
    expected: object, candidate: object, question_type: str, *, policy: JEEPolicy = JEEPolicy.SOURCE
) -> Reward:
    """Score a source-extracted answer with typed preparation failures."""
    raw = capture_jee_input(expected, candidate, question_type)
    try:
        prepared = prepare_jee_input(raw, policy)
    except InvalidTask as error:
        return invalid_task(str(error))
    return grade_prepared_jee(prepared)
