# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed private verifier for NeMo predicted function-call submissions."""

import json

import msgspec

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, VerifierHandler
from taskcompendium.models import FunctionCall, ToolCallComparatorConfig, VerifierSpec
from taskcompendium.predicted_action import compare, decode_action, parse_arguments
from taskcompendium.rendering import AnswerFormat

KIND = "nemo_predicted_action"


class FunctionCallPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    arguments: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Expected function calls require a name")
        parse_arguments(self.arguments)


class PredictedActionPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    expected_calls: tuple[FunctionCallPayload, ...]
    numeric_tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.expected_calls:
            raise ValueError("Predicted-action verifier requires expected function calls")
        ToolCallComparatorConfig(self.numeric_tolerance)


def predicted_action_verifier(expected_calls: tuple[FunctionCall, ...]) -> VerifierSpec:
    """Construct a private strict function-call verifier descriptor."""
    payload = PredictedActionPayload(tuple(FunctionCallPayload(call.name, call.arguments) for call in expected_calls))
    return VerifierSpec(KIND, msgspec.to_builtins(payload))


def _grade_action(payload: PredictedActionPayload, attempt: GradingAttempt) -> GradeResult:
    if attempt.rendering.answer_format != AnswerFormat.FINAL_ACTION:
        return GradeResult(Outcome.INFRA_ERROR, None, "Incompatible final-action rendering")
    try:
        if attempt.response is None:
            raise ValueError("Final action is missing")
        message = json.loads(attempt.response)
        if not isinstance(message, dict):
            raise ValueError("Final action must be an object")
        actual = decode_action(message)
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    expected = tuple(FunctionCall(call.name, call.arguments) for call in payload.expected_calls)
    reward = compare(expected, actual, ToolCallComparatorConfig(payload.numeric_tolerance))
    return GradeResult(Outcome.GRADED, reward)


def nemo_predicted_action_handler() -> VerifierHandler[PredictedActionPayload]:
    return VerifierHandler(PredictedActionPayload, _grade_action)
