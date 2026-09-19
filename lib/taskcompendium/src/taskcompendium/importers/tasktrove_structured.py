# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import TaskTrove structured-output JSON and XML tasks."""

import json
import re
import tomllib
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from tasktrove_verify.spec import Mode

from taskcompendium.importers.tasktrove import TaskArchive, semantic_verifier
from taskcompendium.models import (
    AnswerRequirements,
    Embedded,
    Rejected,
    RejectionReason,
    Resource,
    ResourceRole,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
)

_FAMILY = "other"
_CONVERTER = "nemotron_structured_outputs"
_JSON_SCHEMA_MODE = Mode.JSON_SCHEMA
_XML_MODE = Mode.XML_ELEMENTS
_SUPPORTED_TYPES = {"json": (_JSON_SCHEMA_MODE, "json"), "xml": (_XML_MODE, "xml")}
_INSTRUCTION_PREFIX = "You will produce a structured response. Write your final answer to `/app/answer.txt`.\n"
_SUBMISSION_HEADER = "\n## Submitting your answer (IMPORTANT)"
_XML_SCHEMA_WORDING = (
    "Emit a single well-formed XML document representing data that follows the JSON Schema in the task."
)
_XML_SCHEMA_REPLACEMENT = "Emit a single well-formed XML document representing data that follows the schema in the task."
_EVALUATION_NOTES = (
    (
        " The verifier parses your answer (optionally unwrapping a ```json fence) and validates it "
        "with `jsonschema` Draft 2020-12.",
        "",
    ),
    (
        "The verifier checks the answer is well-formed XML and that every top-level required field "
        "from the schema appears as an element (or attribute) in the document.",
        "Include every top-level required field from the schema as an element or attribute in the document.",
    ),
)
_ALL_VALUES_QUOTED = re.compile(r"all\s+values?\s+for\s+attributes?\s+(?:should|must)\s+be\s+in\s+quotes", re.IGNORECASE)


def _metadata(archive: TaskArchive) -> dict[str, Any]:
    raw = archive.files.get("task.toml")
    if raw is None:
        raise ValueError("missing task.toml")
    value = tomllib.loads(raw.decode())
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _clean_instructions(instructions: str) -> str:
    """Retain task requirements and schema text without delivery or evaluation commentary."""
    if not instructions.startswith(_INSTRUCTION_PREFIX):
        raise ValueError("unsupported structured-output instruction template")
    body = instructions[len(_INSTRUCTION_PREFIX) :].lstrip("\n")
    if _SUBMISSION_HEADER not in body:
        raise ValueError("structured-output instruction has no submission boundary")
    cleaned = body.split(_SUBMISSION_HEADER, 1)[0].strip()
    if cleaned.startswith(_XML_SCHEMA_WORDING):
        cleaned = _XML_SCHEMA_REPLACEMENT + cleaned[len(_XML_SCHEMA_WORDING) :]
    for source_text, task_requirement in _EVALUATION_NOTES:
        cleaned = cleaned.replace(source_text, task_requirement)
    if not cleaned:
        raise ValueError("structured-output instruction has no semantic task text")
    return cleaned


def _schema_resource(archive: TaskArchive, schema_type: str) -> tuple[Resource | None, Any | None]:
    if schema_type != "json":
        return None, None
    raw = archive.files.get("tests/schema.json")
    if raw is None:
        raise ValueError("JSON structured-output task is missing tests/schema.json")
    try:
        schema = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON schema: {error}") from error
    if not isinstance(schema, dict):
        raise ValueError("JSON schema must be an object")
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as error:
        raise ValueError(f"schema fails Draft 2020-12 metaschema check: {error.message}") from error
    return Resource("schema.json", (ResourceRole.VERIFIER,), Embedded(raw)), schema


def _schema_types(schema: Any) -> set[str]:
    """Collect declared JSON primitive types without changing the source schema."""
    if isinstance(schema, dict):
        result = set()
        declared = schema.get("type")
        if isinstance(declared, str) and declared in {"number", "integer", "boolean", "null"}:
            result.add(declared)
        for value in schema.values():
            result.update(_schema_types(value))
        return result
    if isinstance(schema, list):
        result = set()
        for value in schema:
            result.update(_schema_types(value))
        return result
    return set()


def _is_trivial_schema(schema: dict[str, Any]) -> bool:
    declared = schema.get("type")
    if declared in (None, "object"):
        return not schema.get("properties") and not schema.get("required")
    return declared == "array" and not schema.get("items")


def import_task(archive: TaskArchive) -> TaskSpec | Rejected:
    """Convert a deterministic TaskTrove structured-output archive."""
    source = archive.source
    if archive.family != _FAMILY:
        return Rejected(
            source,
            RejectionReason.UNSUPPORTED_VERIFIER,
            f"unsupported structured family {archive.family!r}",
        )
    try:
        metadata = _metadata(archive)
        instructions = _clean_instructions(archive.instructions)
        verifier = semantic_verifier(archive.verifier, implementation_revision=archive.release.verifier_revision)
    except (KeyError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as error:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, str(error))
    if metadata.get("converter") != _CONVERTER:
        return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, "unsupported structured-output converter")
    schema_type = metadata.get("schema_type")
    if not isinstance(schema_type, str) or schema_type not in _SUPPORTED_TYPES:
        return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, f"unsupported schema_type {schema_type!r}")
    expected_mode, _ = _SUPPORTED_TYPES[schema_type]
    if verifier.mode is not expected_mode:
        return Rejected(
            source,
            RejectionReason.UNSUPPORTED_VERIFIER,
            f"schema_type {schema_type!r} does not match verifier mode {verifier.mode.value!r}",
        )
    try:
        resource, schema = _schema_resource(archive, schema_type)
    except ValueError as error:
        return Rejected(source, RejectionReason.BROKEN_GRADER, str(error))
    if schema is not None:
        if not schema or _is_trivial_schema(schema):
            return Rejected(
                source, RejectionReason.NULL_ANSWER_PASSES, "schema has no properties or required fields to check"
            )
        format_value = verifier.parameters.get("format")
        if getattr(format_value, "value", format_value) != "json":
            return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, "JSON schema verifier format is not json")
    else:
        required = verifier.parameters.get("required", ())
        any_of = verifier.parameters.get("any_of", ())
        if not required and not any_of:
            return Rejected(
                source, RejectionReason.NULL_ANSWER_PASSES, "XML verifier has no required or any_of element names"
            )
    if schema is not None and _ALL_VALUES_QUOTED.search(archive.instructions):
        non_string_types = _schema_types(schema) - {"string"}
        if non_string_types:
            return Rejected(
                source,
                RejectionReason.UNRECOVERABLE_SOURCE,
                "source instructions require quoted values but the retained JSON schema declares "
                + ", ".join(sorted(non_string_types)),
            )
    tags = metadata.get("tags", ())
    competencies = tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()
    return TaskSpec(
        id=f"tasktrove-{source.row}",
        requirements=TaskRequirements(),
        resources=(resource,) if resource is not None else (),
        metadata=TaskMetadata(source=source, competencies=competencies, task_shape="answer"),
        steps=(
            StepSpecification(
                instructions=instructions,
                verifier=verifier,
                answer_requirements=AnswerRequirements("json" if schema_type == "json" else "xml"),
            ),
        ),
    )
