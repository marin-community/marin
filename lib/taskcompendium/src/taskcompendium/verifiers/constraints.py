# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct-chat adapters for TaskTrove's instruction and JSON Schema checks."""

import json
import re
from typing import Any

import yaml
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, JsonValue, model_validator
from verifyit.grade import InvalidTask
from verifyit.modes.extract import unwrap_fence
from verifyit.modes.grade_ifeval import resolve_checks
from verifyit.modes.grade_json_schema import parse_candidate
from verifyit.spec import Constraint, SchemaFormat

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.submission import extract_answer


def required_object_conflicts(schema: dict[str, Any], path: str = "$") -> list[str]:
    """Find forbidden required keys in mandatory object branches.

    This proves specific contradictions; an empty result is not a general
    satisfiability proof. Optional branches and union types are not followed.
    """
    if schema.get("type") != "object":
        return []
    properties = schema.get("properties", {})
    patterns = schema.get("patternProperties", {})
    conflicts = []
    for name in schema.get("required", []):
        if (
            schema.get("additionalProperties") is False
            and name not in properties
            and not any(re.search(pattern, name) for pattern in patterns)
        ):
            conflicts.append(f"{path}.{name}: required but forbidden by additionalProperties=false")
        child = properties.get(name)
        if isinstance(child, dict):
            conflicts.extend(required_object_conflicts(child, f"{path}.{name}"))
    return conflicts


class InstructionConstraint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    parameters: dict[str, JsonValue]


class IfevalVerifier(Verifier):
    constraints: tuple[InstructionConstraint, ...]

    @model_validator(mode="after")
    def validate_constraints(self) -> "IfevalVerifier":
        try:
            resolve_checks(tuple(Constraint(c.name, c.parameters) for c in self.constraints))
        except InvalidTask as error:
            raise ValueError(str(error)) from error
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            text = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        checks = resolve_checks(tuple(Constraint(c.name, c.parameters) for c in self.constraints))
        try:
            results = [check(text, constraint.params)[0] for constraint, check in checks]
        except (KeyError, TypeError, ValueError) as error:
            return GradeResult(Outcome.INFRA_ERROR, None, f"Invalid constraint parameters: {error}")
        return GradeResult(Outcome.GRADED, float(all(results)))


class JsonSchemaVerifier(Verifier):
    document_schema_json: str
    schema_format: SchemaFormat

    @model_validator(mode="after")
    def validate_schema(self) -> "JsonSchemaVerifier":
        schema = json.loads(self.document_schema_json)
        if not isinstance(schema, dict):
            raise ValueError("The verifier requires a JSON Schema object")
        try:
            validator_for(schema).check_schema(schema)
        except SchemaError as error:
            raise ValueError(error.message) from error
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            text = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        try:
            instance = parse_candidate(unwrap_fence(text), self.schema_format)
        except (ValueError, yaml.YAMLError):
            return GradeResult(Outcome.GRADED, 0.0)
        schema = json.loads(self.document_schema_json)
        # pyrefly: ignore[bad-instantiation, missing-argument]  # jsonschema types the concrete class as a protocol.
        validator = validator_for(schema)(schema)
        return GradeResult(Outcome.GRADED, float(validator.is_valid(instance)))
