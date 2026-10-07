# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve TaskTrove's explicit any-valid-instance contract and JSON Schema."""

import base64
import json
import re
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from verifyit.spec import JsonSchemaSpec, SchemaFormat

from taskcompendium.grader import grader_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import resolve_verifier


def required_object_conflicts(schema: dict[str, Any], path: str = "$") -> list[str]:
    """Identify mandatory object properties forbidden by their own schema."""
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


DELIVERY = "Write your final JSON to `/app/answer.txt`."
RUBRIC = ReviewRubric(
    id="structured-output-contract",
    version="2",
    criteria=(
        "The opening Evaluation contract is authoritative: any schema-valid instance is acceptable, and unstated "
        "values may be chosen. Do not misclassify this as extraction of one hidden reference document.",
        "Flag contradictory lower-priority extraction or grounding instructions as ambiguity when they can confuse "
        "a solver. Remove them in a separate rewrite, preserving the authoritative contract and all supplied facts.",
        "The public JSON Schema must match the private schema. Never relax a required field, type, enum, or bound.",
        "Compare complete object structure, including root type, required, additionalProperties, and definitions. "
        "Matching nested field definitions is insufficient when the public schema is only a property map and the "
        "private verifier adds hidden root requirements. Cite a specific omitted constraint rather than claiming "
        "the schemas are identical without checking them.",
        "Check satisfiability: required properties forbidden by additionalProperties=false make a mandatory "
        "object impossible. Check mandatory nested objects too. Meta-schema validity does not prove that an "
        "instance exists. Misplaced keywords may be ignored rather than making the schema unsatisfiable.",
        "Require meaningful fields, not only formally valid strings: a contact phone pattern excluding all digits "
        "cannot express the described phone number. The any-valid-instance contract does not make that defect "
        "disappear. Do not invent units or facts when a numeric field cannot represent stated content.",
        "Flag quote-all-values boilerplate when it contradicts required numeric or boolean types. This can need "
        "a wording repair while preserving the authoritative contract and the exact schema. Ordinary restructuring "
        "language subordinate to a clear generation contract is not by itself a defect.",
        "A schema-valid instance demonstrates structural feasibility only. The grader does not enforce grounding "
        "or format annotations; report a mismatch if the prompt requires checks the grader does not implement.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_input",
            detail="Instruction and verifier_data are required",
        )
    if data.get("schema_type") != "json" or not isinstance(data.get("schema"), dict):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_schema",
            detail="Expected an explicit JSON Schema object",
        )
    try:
        validator_for(data["schema"]).check_schema(data["schema"])
    except SchemaError as error:
        return ImportRejection(kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_schema", detail=str(error))
    package = grader_package(
        JsonSchemaSpec(schema="schema.json", format=SchemaFormat.JSON),
        (inline_resource("schema.json", json.dumps(data["schema"]).encode()),),
    )
    return TaskSpec(
        id=row.id,
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content=instruction.replace(DELIVERY, "Return your final JSON in the assistant response."),
                ),
            )
        ),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=package.verifier,
        resources=ResourceGroups(verifier=package.resources),
        source=row.source,
    )


def policy() -> TaskPolicy:
    """Build the source normalization and review policy."""
    return TaskPolicy(
        normalize=normalize,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="json-schema-contract-and-controls",
            revision="1",
            parameters={},
            run=verification_report,
        ),
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    spec = resolve_verifier(task.verifier)
    assert isinstance(spec, JsonSchemaSpec)
    schema_resource = next(resource for resource in task.resources.verifier if resource.path == spec.schema)
    conflicts = required_object_conflicts(json.loads(base64.b64decode(schema_resource.source.content_base64)))
    checks = [
        CheckResult(check="required_object_contract", status=CheckStatus.FAIL, detail=conflict) for conflict in conflicts
    ]
    return VerificationReport(checks=[*checks, *verify_task(task)])
