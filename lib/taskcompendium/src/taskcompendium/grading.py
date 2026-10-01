# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade submissions with typed private verifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator
from tasktrove_verify.grade import InvalidTask, numeric_tolerance
from tasktrove_verify.json_comparison import json_values_equal
from tasktrove_verify.modes.extract import extract_boxed
from tasktrove_verify.modes.grade_exact import grade_exact_candidate
from tasktrove_verify.modes.grade_math import grade_numeric_candidate
from tasktrove_verify.spec import ExactSpec, NumericSpec, SchemaFormat

from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import (
    GradingAttempt,
    StateSubmission,
    Submission,
    TextSubmission,
)


class Outcome(StrEnum):
    GRADED = "graded"
    SUBMISSION_FAILURE = "submission_failure"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class GradeResult:
    status: Outcome
    reward: float | None
    error: str | None = None


class Verifier(BaseModel, ABC):
    """Validated private configuration that grades one submission."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    @abstractmethod
    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        """Grade a submission using this verifier's configuration."""


class ExactAnswerOrder(StrEnum):
    """Whether multiple accepted exact values must retain their source order."""

    ORDERED = "ordered"
    UNORDERED = "unordered"


class ExactAnswerVerifier(Verifier):
    """Compare one or more text values using exact-mode normalization."""

    expected: tuple[str, ...]
    ignore_case: bool = True
    collapse_whitespace: bool = True
    ordering: ExactAnswerOrder = ExactAnswerOrder.ORDERED

    @model_validator(mode="after")
    def validate_expected(self) -> "ExactAnswerVerifier":
        if not self.expected or any(not item.strip() for item in self.expected):
            raise ValueError("An exact answer requires nonempty expected values")
        return self

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(submission, (TextSubmission, StateSubmission)) or not isinstance(submission.value, str):
            raise TypeError("Exact-answer verifier requires a string value")
        contract = ExactSpec(
            expected=self.expected,
            ignore_case=self.ignore_case,
            ignore_whitespace=self.collapse_whitespace,
            ordered=self.ordering is ExactAnswerOrder.ORDERED,
        )
        boxed = extract_boxed(submission.value)
        if boxed is not None:
            result = grade_exact_candidate(contract, boxed)
            if result.reward:
                return GradeResult(Outcome.GRADED, result.reward)
        return GradeResult(Outcome.GRADED, grade_exact_candidate(contract, submission.value).reward)


class JsonSchemaVerifier(Verifier):
    """Accept a text document validated by a private JSON Schema."""

    schema_definition: dict[str, JsonValue]
    format: SchemaFormat

    @model_validator(mode="after")
    def validate_schema(self) -> "JsonSchemaVerifier":
        from jsonschema.validators import validator_for  # noqa: PLC0415

        validator_for(self.schema_definition).check_schema(self.schema_definition)
        return self

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("JSON-schema verifier requires a text submission")
        import yaml  # noqa: PLC0415
        from jsonschema.validators import validator_for  # noqa: PLC0415
        from tasktrove_verify.modes.extract import unwrap_fence  # noqa: PLC0415
        from tasktrove_verify.modes.grade_json_schema import parse_candidate  # noqa: PLC0415

        try:
            candidate = parse_candidate(unwrap_fence(submission.value), self.format)
        except (ValueError, TypeError, yaml.YAMLError):
            return GradeResult(Outcome.GRADED, 0.0)
        validator = validator_for(self.schema_definition)(self.schema_definition)
        return GradeResult(Outcome.GRADED, float(validator.is_valid(candidate)))


class NumericAnswerVerifier(Verifier):
    """Compare a submitted number with explicit absolute and relative tolerances."""

    expected: float
    tolerance_abs: float
    tolerance_rel: float

    @model_validator(mode="after")
    def validate_contract(self) -> "NumericAnswerVerifier":
        contract = NumericSpec(
            expected=self.expected, tolerance_abs=self.tolerance_abs, tolerance_rel=self.tolerance_rel
        )
        try:
            numeric_tolerance(contract)
        except InvalidTask as error:
            raise ValueError(f"Invalid numeric verifier contract: {error}") from error
        return self

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(submission, TextSubmission):
            raise TypeError("Numeric verifier requires a text submission")
        try:
            value = float(submission.value.strip())
        except ValueError:
            return GradeResult(Outcome.GRADED, 0.0)
        contract = NumericSpec(
            expected=self.expected, tolerance_abs=self.tolerance_abs, tolerance_rel=self.tolerance_rel
        )
        return GradeResult(Outcome.GRADED, grade_numeric_candidate(contract, value).reward)


class StructuredExactVerifier(Verifier):
    """Compare a JSON-compatible submission with a private expected object."""

    expected: JsonValue

    async def grade(self, submission: Submission, *, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(submission, StateSubmission):
            raise TypeError("Structured exact verifier requires a state submission")
        return GradeResult(Outcome.GRADED, float(json_values_equal(self.expected, submission.value)))


def structured_exact(expected: JsonValue) -> VerifierSpec:
    verifier = StructuredExactVerifier(expected=expected)
    return VerifierSpec(kind=VerifierKind.STRUCTURED_EXACT, parameters_json=verifier.model_dump_json())


def exact_answer(
    expected: tuple[str, ...],
    ignore_case: bool = True,
    collapse_whitespace: bool = True,
    ordering: ExactAnswerOrder = ExactAnswerOrder.ORDERED,
) -> VerifierSpec:
    """Construct an exact verifier; one expected value matches the whole candidate."""
    verifier = ExactAnswerVerifier(
        expected=expected,
        ignore_case=ignore_case,
        collapse_whitespace=collapse_whitespace,
        ordering=ordering,
    )
    return VerifierSpec(kind=VerifierKind.EXACT_ANSWER, parameters_json=verifier.model_dump_json())


def json_schema_answer(schema: dict[str, JsonValue], schema_format: SchemaFormat) -> VerifierSpec:
    """Construct a private verifier that accepts instances satisfying the schema."""
    verifier = JsonSchemaVerifier(schema_definition=schema, format=schema_format)
    return VerifierSpec(kind=VerifierKind.JSON_SCHEMA, parameters_json=verifier.model_dump_json())


def numeric_answer(expected: float, tolerance_abs: float, tolerance_rel: float) -> VerifierSpec:
    verifier = NumericAnswerVerifier(expected=expected, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel)
    return VerifierSpec(kind=VerifierKind.NUMERIC_ANSWER, parameters_json=verifier.model_dump_json())
