# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Nemotron structured-outputs tasks, graded in process on the assistant's reply.

JSON, YAML and TOML answers are validated against the source schema after the converter's repairs.
XML and CSV answers are checked for the schema's top-level field names. The archived instruction asks a
terminal agent to write ``/app/answer.txt``; the task asks for the answer in the reply instead.
"""

import csv
import io
import json

from taskcompendium.convert.answers import answer_task, json_schema_task, source_defect
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.convert.json_schema import required_object_conflicts
from taskcompendium.convert.tasktrove_converted_task import archive_conversion
from taskcompendium.convert.tasktrove_nemotron_structured_outputs import convert_nemotron_structured_outputs
from taskcompendium.models import TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.models import (
    Controls,
    ImportFailureKind,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    RawRow,
    Reply,
)
from taskcompendium.pipeline.verification import MALFORMED_JSON
from verifyit.spec import CsvColumnsSpec, JsonSchemaSpec, Spec, XmlElementsSpec

from experiments.post_training.task_curation.datasets.tasktrove import tasktrove_source
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim

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

The checker enforces schema structure; identify unsupported semantic or factual requirements in the request.

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


def invalid_contract(detail: str) -> ImportRejection:
    return ImportRejection(kind=ImportFailureKind.CONVERTER_ERROR, reason="invalid_structured_contract", detail=detail)


def convert_structured_outputs(row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
    converted = archive_conversion(row.data, convert_nemotron_structured_outputs)
    if isinstance(converted, ImportRejection):
        return converted
    original = converted.instruction
    instruction = reply_instruction(original)
    spec = converted.spec
    if isinstance(spec, JsonSchemaSpec):
        schema = converted.data_files[f"tests/{spec.schema}"].decode()
        conflicts = required_object_conflicts(json.loads(schema))
        if conflicts:
            return source_defect("unsatisfiable_schema", "; ".join(conflicts))
        task = json_schema_task(row, prompt=instruction, schema=schema, schema_format=spec.format)
        if isinstance(task, ImportRejection):
            return task
    else:
        assert isinstance(spec, XmlElementsSpec | CsvColumnsSpec)
        if any(not name for name in (*spec.required, *spec.any_of)):
            return invalid_contract("Required and alternative names must be nonempty strings")
        task = answer_task(row, prompt=instruction, spec=spec)
    return rewritten_task(task, original=original, reason=REWRITE_REASON)


def _names(spec: XmlElementsSpec | CsvColumnsSpec) -> tuple[str, ...]:
    return (*spec.required, *spec.any_of[:1])


def _csv(rows: list[tuple[str, ...]]) -> str:
    document = io.StringIO()
    csv.writer(document).writerows(rows)
    return document.getvalue()


def _grader_spec(task: TaskSpec) -> Spec:
    assert isinstance(task.grader, VerifyitGrader)
    return verifyit_spec(task.grader)


def structured_witness(task: TaskSpec) -> Reply | None:
    """An XML element or CSV table naming every required field; JSON schemas supply no instance."""
    spec = _grader_spec(task)
    if isinstance(spec, XmlElementsSpec):
        return answer_reply(task, "<control>" + "".join(f"<{name}/>" for name in _names(spec)) + "</control>")
    if isinstance(spec, CsvColumnsSpec):
        return answer_reply(task, _csv([_names(spec), tuple("control" for _ in _names(spec))]))
    return None


def structured_negative(task: TaskSpec) -> Reply:
    """Malformed JSON or XML, or a CSV header without a row."""
    spec = _grader_spec(task)
    if isinstance(spec, XmlElementsSpec):
        return answer_reply(task, "<control>")
    if isinstance(spec, CsvColumnsSpec):
        return answer_reply(task, _csv([_names(spec)]))
    return answer_reply(task, MALFORMED_JSON)


def pipelines() -> list[RlDataPipeline]:
    return [
        RlDataPipeline(
            name="tasktrove-structured_outputs",
            source=tasktrove_source(CONFIG),
            convert=convert_structured_outputs,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=STRUCTURED_OUTPUTS_RUBRIC,
            controls=Controls(golden=structured_witness, negative=structured_negative),
            atlas_id=f"Task Trove:{CONFIG}",
        )
    ]
