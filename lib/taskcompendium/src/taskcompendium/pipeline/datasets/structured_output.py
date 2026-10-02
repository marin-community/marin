# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve TaskTrove's explicit any-valid-instance contract and JSON Schema."""

import json

from pydantic import ValidationError
from verifyit.spec import SchemaFormat

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.source_definitions import TASKTROVE_DATASET, tasktrove_inputs
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.verifiers.constraints import JsonSchemaVerifier, required_object_conflicts

CONFIG = "laion__nemotron-gym-instruction-following-structured-v3"
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
        return ImportRejection(reason="missing_input", detail="Instruction and verifier_data are required")
    if data.get("schema_type") != "json" or not isinstance(data.get("schema"), dict):
        return ImportRejection(reason="unsupported_schema", detail="Expected an explicit JSON Schema object")
    try:
        verifier = JsonSchemaVerifier(document_schema_json=json.dumps(data["schema"]), schema_format=SchemaFormat.JSON)
    except (ValidationError, ValueError) as error:
        return ImportRejection(reason="invalid_schema", detail=str(error))
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
        verifier=VerifierSpec(kind=VerifierKind.JSON_SCHEMA, parameters_json=verifier.model_dump_json()),
        source=row.source,
    )


def recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="tasktrove-structured",
        version="tasktrove-structured-v1",
        source=HFSource(TASKTROVE_DATASET, REVISION, CONFIG, "train"),
        inputs=tasktrove_inputs(CONFIG, REVISION),
        normalize=normalize,
        rubric=RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="json-schema-contract-and-controls",
            revision="1",
            parameters={},
            run=verification_report,
        ),
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    verifier = JsonSchemaVerifier.model_validate_json(task.verifier.parameters_json)
    conflicts = required_object_conflicts(json.loads(verifier.document_schema_json))
    checks = [
        CheckResult(check="required_object_contract", status=CheckStatus.FAIL, detail=conflict) for conflict in conflicts
    ]
    return VerificationReport(checks=[*checks, *verify_task(task)])
