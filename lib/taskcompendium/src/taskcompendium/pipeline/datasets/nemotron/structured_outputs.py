# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron schema-generation tasks, distinct from instruction-following tasks."""

import base64
import csv
import io

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
from taskcompendium.pipeline.datasets.raw_conversion import RawConverter, with_raw_converter
from taskcompendium.pipeline.datasets.source_definitions import tasktrove_inputs
from taskcompendium.pipeline.datasets.structured_output import verification_report as schema_verification_report
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_witness
from taskcompendium.verifiers.constraints import JsonSchemaVerifier
from taskcompendium.verifiers.structured_fields import NamedFieldsVerifier

CONFIG = "laion__nemotron-gym-structured-outputs-v4"
RUBRIC = ReviewRubric(
    id="structured-outputs-answerability",
    version="3",
    criteria=(
        "A request to read a supplied document and produce structured output from it implies grounding "
        "in that document. Required factual fields must have evidence or a missing-data policy even when "
        "the checker validates only schema. Treat any-valid-instance generation as authorized only when "
        "the public request explicitly allows arbitrary or synthetic values.",
        "For extraction requests, trace each required factual field to the supplied document or an "
        "explicit missing-data policy. A schema does not authorize inventing a user identity, numeric "
        "measurements, or dates absent from the document.",
        "When the document contains several entities but the schema accepts one, check the selection "
        "rule. An absent required numeric value cannot be filled using outside knowledge unless the "
        "public request allows it.",
        "A task asking only for any schema-valid instance can be coherent without source facts. Apply "
        "extraction requirements only when the public task asks to parse, extract, or populate facts "
        "from a document.",
        "Compare the requested serialization format and complete public schema against the private grader.",
        "Check schema satisfiability, required fields, bounds, types, enums, and additionalProperties constraints.",
        "The checker enforces schema structure; identify unsupported semantic or factual requirements in the request.",
        "Converter repairs are review evidence: reject a repair that changes the public contract "
        "rather than its syntax.",
        "This is the structured-outputs source, not the instruction-following structured source; do not assume an "
        "authoritative any-valid-instance preamble when the supplied request has none.",
    ),
)


def normalize(row: RawRow) -> NormalizedTask | ImportRejection:
    """Import a borrowed converter result without discarding its schema repairs."""
    rejection = row.data.get("conversion_rejection")
    if isinstance(rejection, dict):
        return ImportRejection.model_validate(rejection)
    converted = row.data.get("converted")
    if not isinstance(converted, dict):
        return ImportRejection(reason="missing_conversion", detail="Run the structured-outputs converter binding first")
    spec = converted["grader_spec"]
    verifier: JsonSchemaVerifier | NamedFieldsVerifier
    try:
        if spec["mode"] == "json-schema":
            schema_path = "tests/" + spec["schema"]
            schema = base64.b64decode(converted["data_files"][schema_path], validate=True).decode()
            verifier = JsonSchemaVerifier(document_schema_json=schema, schema_format=SchemaFormat(spec["format"]))
            kind = VerifierKind.JSON_SCHEMA
        elif spec["mode"] in ("xml-elements", "csv-columns"):
            verifier = NamedFieldsVerifier(
                mode=spec["mode"], required=tuple(spec["required"]), any_of=tuple(spec["any_of"])
            )
            kind = VerifierKind.STRUCTURED_FIELDS
        else:
            return ImportRejection(reason="unsupported_structured_mode", detail=f"Unknown mode: {spec['mode']}")
    except (ValidationError, ValueError) as error:
        return ImportRejection(reason="invalid_structured_contract", detail=str(error))
    original = converted["instruction"]
    instruction = original.replace(
        "Write your final answer to `/app/answer.txt`.", "Return your final answer in the assistant response."
    ).replace("Write your final JSON to `/app/answer.txt`.", "Return your final JSON in the assistant response.")
    footer = "\n## Submitting your answer (IMPORTANT)\n"
    if footer in instruction:
        body, _, submission = instruction.partition(footer)
        if "Your chat reply is NOT graded" in submission and "/app/answer.txt" in submission:
            instruction = body.rstrip()
    task = TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=kind, parameters_json=verifier.model_dump_json()),
    )
    changes = (
        ()
        if instruction == original
        else (
            NormalizationChange(
                field="instruction",
                reason="Adapt terminal file delivery to direct assistant response",
                original=original,
                replacement=instruction,
            ),
        )
    )
    return NormalizedTask(task, changes)


def recipe(*, converter: RawConverter, converter_revision: str) -> DatasetRecipe:
    source_recipe = DatasetRecipe(
        name="tasktrove-structured_outputs",
        version="tasktrove-structured_outputs-v2",
        source=HFSource("open-thoughts/TaskTrove", REVISION, CONFIG, "train"),
        normalize=normalize,
        rubric=RUBRIC,
        intended_use=IntendedUse.TRAIN,
        inputs=tasktrove_inputs(CONFIG, REVISION),
        check_suite=CheckSuite(
            id="structured-format-contract-and-controls", revision="2", parameters={}, run=verification_report
        ),
    )
    return with_raw_converter(source_recipe, converter, converter_revision)


def verification_report(task: TaskSpec) -> VerificationReport:
    """Check schema contradictions or the preserved named-fields runtime contract."""
    if task.verifier.kind == VerifierKind.JSON_SCHEMA:
        return schema_verification_report(task)
    verifier = NamedFieldsVerifier.model_validate_json(task.verifier.parameters_json)
    names = (*verifier.required, *(verifier.any_of[:1]))
    if verifier.mode == "xml-elements":
        witness = "<control>" + "".join(f"<{name}/>" for name in names) + "</control>"
        negative = "<control>"
    else:
        document = io.StringIO()
        writer = csv.writer(document)
        writer.writerow(names)
        negative = document.getvalue()
        writer.writerow("control" for _ in names)
        witness = document.getvalue()
    return VerificationReport(checks=verify_witness(task, witness, negative))
