# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade typed final function calls without depending on a harness wire protocol."""

from pydantic import JsonValue, field_validator, model_validator

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import (
    AssistantMessage,
    AssistantToolCalls,
    ConversationToolCall,
    FunctionCall,
    TextMessage,
    ToolCallComparatorConfig,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.submission import AnswerFormat


def _arguments_match(expected: JsonValue, actual: JsonValue, config: ToolCallComparatorConfig) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _arguments_match(value, actual[key], config) for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _arguments_match(left, right, config) for left, right in zip(expected, actual, strict=True)
        )
    if isinstance(expected, float) and isinstance(actual, float) and config.numeric_tolerance is not None:
        return abs(expected - actual) <= config.numeric_tolerance
    return expected == actual


def _call_matches(expected: FunctionCall, actual: ConversationToolCall, config: ToolCallComparatorConfig) -> bool:
    if expected.name != actual.name:
        return False
    return _arguments_match(expected.arguments, actual.arguments, config)


def _matching_count(
    expected: tuple[FunctionCall, ...], actual: tuple[ConversationToolCall, ...], config: ToolCallComparatorConfig
) -> int:
    candidates = [
        [index for index, right in enumerate(actual) if _call_matches(left, right, config)] for left in expected
    ]
    matching: dict[int, int] = {}

    def augment(expected_index: int, visited: set[int]) -> bool:
        for actual_index in candidates[expected_index]:
            if actual_index in visited:
                continue
            visited.add(actual_index)
            if actual_index not in matching or augment(matching[actual_index], visited):
                matching[actual_index] = expected_index
                return True
        return False

    for index in sorted(range(len(expected)), key=lambda candidate: len(candidates[candidate])):
        augment(index, set())
    return len(matching)


def compare(expected: tuple[FunctionCall, ...], actual: AssistantMessage, config: ToolCallComparatorConfig) -> float:
    """Require the same number of calls with exact JSON arguments by default."""
    if not expected:
        raise ValueError("Expected function calls are required")
    if isinstance(actual, TextMessage):
        return 0.0
    if len(expected) != len(actual.calls):
        return 0.0
    matched = _matching_count(expected, actual.calls, config)
    return float(matched == len(expected))


class PredictedActionVerifier(Verifier):
    """Compare a final action with the source function calls."""

    expected_calls: tuple[FunctionCall, ...]
    numeric_tolerance: float | None = None

    @field_validator("numeric_tolerance", mode="before")
    @classmethod
    def validate_numeric_type(cls, value: object) -> object:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Numeric tolerance must be a number")
        return value

    @model_validator(mode="after")
    def validate_action(self) -> "PredictedActionVerifier":
        if not self.expected_calls:
            raise ValueError("Predicted-action verifier requires expected function calls")
        ToolCallComparatorConfig(self.numeric_tolerance)
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        if attempt.convention.answer_format != AnswerFormat.FINAL_ACTION:
            return GradeResult(Outcome.INFRA_ERROR, None, "Incompatible final-action convention")
        actual = attempt.conversation[-1]
        if not isinstance(actual, (TextMessage, AssistantToolCalls)):
            return GradeResult(Outcome.INFRA_ERROR, None, "Missing final assistant message")
        reward = compare(self.expected_calls, actual, ToolCallComparatorConfig(self.numeric_tolerance))
        return GradeResult(Outcome.GRADED, reward)


def predicted_action_verifier(expected_calls: tuple[FunctionCall, ...]) -> VerifierSpec:
    """Construct a private strict function-call verifier descriptor."""
    verifier = PredictedActionVerifier(expected_calls=expected_calls)
    return VerifierSpec(kind=VerifierKind.PREDICTED_ACTION, parameters_json=verifier.model_dump_json())
