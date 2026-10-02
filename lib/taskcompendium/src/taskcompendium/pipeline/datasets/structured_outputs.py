# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron schema-generation tasks, distinct from instruction-following tasks."""

import base64
from pathlib import Path

from pydantic import ValidationError

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.pipeline.datasets.structured_output import verification_report
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    SnapshotSource,
)
from taskcompendium.verifiers.constraints import JsonSchemaVerifier

CONFIG = "laion__nemotron-gym-structured-outputs-v4"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
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
    if spec["mode"] != "json-schema" or spec.get("format", "json") != "json":
        return ImportRejection(
            reason="unsupported_format",
            detail=f"Direct-chat runtime currently supports JSON schemas, not {spec['mode']}:{spec.get('format')}",
        )
    schema_path = "tests/" + spec["schema"]
    schema = base64.b64decode(converted["data_files"][schema_path], validate=True).decode()
    try:
        verifier = JsonSchemaVerifier(document_schema_json=schema)
    except (ValidationError, ValueError) as error:
        return ImportRejection(reason="invalid_schema", detail=str(error))
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
        verifier=VerifierSpec(kind=VerifierKind.JSON_SCHEMA, parameters_json=verifier.model_dump_json()),
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


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="tasktrove-structured_outputs",
        version="tasktrove-structured_outputs-v1",
        source=SnapshotSource("open-thoughts/TaskTrove", REVISION, CONFIG, "train", str(snapshot)),
        normalize=normalize,
        rubric=RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="json-schema-contract-and-controls", revision="1", parameters={}, run=verification_report
        ),
    )
