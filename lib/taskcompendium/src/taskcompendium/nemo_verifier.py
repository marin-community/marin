# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed private verifier for NeMo predicted function-call submissions."""

import json

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, VerifierHandler
from taskcompendium.models import FunctionCall, ToolCallComparatorConfig, VerifierSpec
from taskcompendium.predicted_action import compare, decode_action, parse_arguments
from taskcompendium.rendering import AnswerFormat

KIND = "nemo_predicted_action"


class FunctionCallPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    arguments: str

    @model_validator(mode="after")
    def validate_call(self) -> "FunctionCallPayload":
        if not self.name:
            raise ValueError("Expected function calls require a name")
        parse_arguments(self.arguments)
        return self


class PredictedActionPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_calls: tuple[FunctionCallPayload, ...]
    numeric_tolerance: float | None = None

    @field_validator("numeric_tolerance", mode="before")
    @classmethod
    def validate_numeric_type(cls, value: object) -> object:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Numeric tolerance must be a number")
        return value

    @model_validator(mode="after")
    def validate_action(self) -> "PredictedActionPayload":
        if not self.expected_calls:
            raise ValueError("Predicted-action verifier requires expected function calls")
        ToolCallComparatorConfig(self.numeric_tolerance)
        return self


def predicted_action_verifier(expected_calls: tuple[FunctionCall, ...]) -> VerifierSpec:
    """Construct a private strict function-call verifier descriptor."""
    payload = PredictedActionPayload(
        expected_calls=tuple(FunctionCallPayload(name=call.name, arguments=call.arguments) for call in expected_calls)
    )
    return VerifierSpec(kind=KIND, parameters=payload.model_dump(mode="json"))


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
