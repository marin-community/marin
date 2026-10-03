# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct-chat adapters for TaskTrove's instruction and JSON Schema checks."""

import json
import re
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, JsonValue, model_validator
from verifyit.grade import InvalidTask, scored
from verifyit.modes.grade_ifeval import grade_ifeval_chat_candidate, resolve_checks
from verifyit.modes.grade_json_schema import grade_json_document
from verifyit.spec import Constraint, SchemaFormat

from taskcompendium.grading import GradeResult
from taskcompendium.verifiers.base import GradingAttempt, Verifier, grade_extracted, grade_result


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
        return grade_extracted(
            attempt,
            lambda text: grade_result(
                grade_ifeval_chat_candidate(tuple(Constraint(c.name, c.parameters) for c in self.constraints), text)
            ),
        )


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
        return grade_extracted(
            attempt,
            lambda text: grade_result(
                scored(grade_json_document(json.loads(self.document_schema_json), self.schema_format, text).reward)
            ),
        )
