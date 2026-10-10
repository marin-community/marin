# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove structured extraction, graded for format and grounding in its source document."""

import json

from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.convert.json_schema import required_object_conflicts
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizedTask, RawRow
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import archive_conversion
from experiments.post_training.task_curation.datasets.tasktrove.conversion.structured_outputs import (
    MISSING_INSTRUCTION,
    convert_nemotron_structured_outputs,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, ShellSim, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

CONFIG = "laion__nemotron-gym-structured-outputs-v4"
FILE_DELIVERY = (
    ("Write your final answer to `/app/answer.txt`.", "Return your final answer in the assistant response."),
    ("Write your final JSON to `/app/answer.txt`.", "Return your final JSON in the assistant response."),
)
SUBMISSION_FOOTER = "\n## Submitting your answer (IMPORTANT)\n"
REWRITE_REASON = "Adapt terminal file delivery to direct assistant response"

STRUCTURED_OUTPUTS_RUBRIC = """
A request to read a supplied document and produce structured output from it implies grounding in that
document. Required factual fields must have evidence or a missing-data rule even when the checker validates
only schema. Treat any-valid-instance generation as authorized only when the public request explicitly allows
arbitrary or synthetic values.

For extraction requests, trace each required factual field to the supplied document or an explicit
missing-data rule. A schema does not authorize inventing a user identity, numeric measurements, or dates
absent from the document.

When the document contains several entities but the schema accepts one, check the selection rule. An absent
required numeric value cannot be filled using outside knowledge unless the public request allows it.

A task asking only for any schema-valid instance can be coherent without source facts. Apply extraction
requirements only when the public task asks to parse, extract, or populate facts from a document.

Compare the requested serialization format and complete public schema against the hidden grader.

Check schema satisfiability, required fields, bounds, types, enums, and additionalProperties constraints.

The checker enforces format first, then a judge checks grounded values and the missing-data convention.

Converter repairs are review evidence: reject a repair that changes the public contract rather than its syntax.

This is the structured-outputs source, not the instruction-following structured source; do not assume an
authoritative any-valid-instance preamble when the supplied request has none.
"""


def reply_instruction(instruction: str) -> str:
    """Ask for the answer in the reply instead of the terminal answer file."""
    instruction = replace_phrases(instruction, FILE_DELIVERY)
    if SUBMISSION_FOOTER in instruction:
        body, _, submission = instruction.partition(SUBMISSION_FOOTER)
        if "Your chat reply is NOT graded" in submission and "/app/answer.txt" in submission:
            return body.rstrip()
    return instruction


def convert_structured_outputs(row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    converted = archive_conversion(row.data, convert_nemotron_structured_outputs)
    if isinstance(converted, ImportRejection):
        return converted
    schema = converted.data_files.get("tests/format/schema.json")
    if schema is not None:
        conflicts = required_object_conflicts(json.loads(schema))
        if conflicts:
            return source_defect("unsatisfiable_schema", "; ".join(conflicts))
    original = converted.instruction
    instruction = reply_instruction(original.removesuffix(MISSING_INSTRUCTION)) + MISSING_INSTRUCTION
    package = verifyit_package(
        converted.spec,
        tuple(inline_resource(path.removeprefix("tests/"), data) for path, data in converted.data_files.items()),
        environment=required_grader_environment(context),
    )
    task = TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
        tags=converted.tags,
    )
    return rewritten_task(task, original=original, reason=REWRITE_REASON)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-structured-outputs-v4",
                title="laion/nemotron-gym-structured-outputs-v4",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn"),
                count=53870,
                notes="All formats require both a format check and a source-grounding judge.",
            ),
            config=CurationRecipe(
                name="tasktrove-structured_outputs",
                source=tasktrove_source(CONFIG),
                convert=TaskTroveConverter(CONFIG, convert_structured_outputs),
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=STRUCTURED_OUTPUTS_RUBRIC,
                grader=GRADER_PACKAGES,
            ),
        )
    ]
