# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron structured-outputs schema tasks.

``tests/verifier_data.json`` carries ``{"schema": <json-schema-like dict>, "schema_type": "json" |
"yaml" | "toml" | "xml" | "csv"}``, and the instruction, which quotes the schema in full, says
which format the answer must be written in. A ``json``, ``yaml`` or ``toml`` answer is parsed and
validated against the schema (mode ``json-schema``). XML and CSV cannot carry the schema's nesting,
so those answers are graded on their structure: the schema's top-level required field names must
appear as element or attribute names (mode ``xml-elements``) or as column headers (mode
``csv-columns``), and a schema that requires nothing falls back to any one of its top-level
property names.

The dataset's schemas are LLM-generated and often not quite valid JSON Schema; ``normalize_schema``
repairs the shapes worth fixing (see ``json_schemas``) and a metaschema check after normalizing
rejects the rest. A schema every answer in its format satisfies is a null grader: an empty TOML
table that validates, an XML or CSV schema with no name to look for. So is an array-typed schema
under ``toml``, which no TOML document can satisfy.
"""

import json
from enum import StrEnum

from jsonschema.validators import validator_for
from tasktrove_verify.spec import CsvColumnsSpec, JsonSchemaSpec, SchemaFormat, Spec, XmlElementsSpec, mode_of

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.json_schemas import normalize_schema, usable_schema
from experiments.post_training.tasktrove.converters.nemotron_data import verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

SCHEMA_NAME = "schema.json"
SCHEMA_FILE = f"tests/{SCHEMA_NAME}"


class SchemaType(StrEnum):
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    XML = "xml"
    CSV = "csv"


SCHEMA_FORMATS = {
    SchemaType.JSON: SchemaFormat.JSON,
    SchemaType.YAML: SchemaFormat.YAML,
    SchemaType.TOML: SchemaFormat.TOML,
}
NESTED_TYPES = frozenset({"object", "array"})
"""Property types a CSV cell cannot carry, so the old grader never asked for their column."""


def top_level_names(schema: dict) -> tuple[list[str], dict]:
    """The schema's required field names and its property map, both after normalization."""
    required = [name for name in schema.get("required") or () if isinstance(name, str)]
    properties = schema.get("properties")
    return required, properties if isinstance(properties, dict) else {}


def is_nested(subschema: object) -> bool:
    declared = subschema.get("type") if isinstance(subschema, dict) else None
    types = declared if isinstance(declared, list) else [declared]
    return any(declared_type in NESTED_TYPES for declared_type in types)


def validates_empty_table(schema: dict) -> bool:
    """Whether ``{}`` satisfies the schema, which is what an empty TOML document parses to."""
    validator_class = validator_for(schema)
    # pyrefly: ignore[bad-instantiation, missing-argument]  # validator_for returns a concrete
    # validator class; jsonschema types it as the Validator protocol.
    return validator_class(schema).is_valid({})


def toml_rejection(schema: dict) -> Rejected | None:
    """Why no TOML answer, or every TOML answer, would satisfy ``schema``."""
    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared]
    if declared is not None and "object" not in types:
        return Rejected(ConvertStatus.NULL_GRADER, f"schema type {declared!r} is unreachable: TOML parses to a table")
    if validates_empty_table(schema):
        return Rejected(ConvertStatus.NULL_GRADER, "an empty TOML document satisfies the schema")
    return None


def xml_spec(schema: dict) -> XmlElementsSpec | Rejected:
    required, properties = top_level_names(schema)
    if required:
        return XmlElementsSpec(required=tuple(required))
    if properties:
        return XmlElementsSpec(any_of=tuple(properties))
    return Rejected(ConvertStatus.NULL_GRADER, "schema names no top-level field: any well-formed XML would score 1")


def csv_spec(schema: dict) -> CsvColumnsSpec | Rejected:
    required, properties = top_level_names(schema)
    columns = [name for name in required if not is_nested(properties.get(name))]
    if columns:
        return CsvColumnsSpec(required=tuple(columns))
    if properties:
        return CsvColumnsSpec(any_of=tuple(properties))
    return Rejected(ConvertStatus.NULL_GRADER, "schema names no top-level scalar field: any CSV table would score 1")


def graded_by(schema_type: SchemaType, schema: object) -> tuple[Spec, dict[str, bytes]] | Rejected:
    """The mode for one schema type and the files it ships, or why the task cannot be graded."""
    if schema_type in SCHEMA_FORMATS:
        validated = usable_schema(schema)
        if isinstance(validated, Rejected):
            return validated
        if schema_type is SchemaType.TOML:
            rejected = toml_rejection(validated)
            if rejected is not None:
                return rejected
        spec = JsonSchemaSpec(schema=SCHEMA_NAME, format=SCHEMA_FORMATS[schema_type])
        return spec, {SCHEMA_FILE: json.dumps(validated, indent=2).encode()}

    if not isinstance(schema, dict) or not schema:
        return Rejected(ConvertStatus.NULL_GRADER, f"schema missing or not an object: {type(schema).__name__}")
    normalized = normalize_schema(schema)
    assert isinstance(normalized, dict)
    structural = xml_spec(normalized) if schema_type is SchemaType.XML else csv_spec(normalized)
    return structural if isinstance(structural, Rejected) else (structural, {})


def convert_nemotron_structured_outputs(task: TaskFiles) -> ConvertedTask | Rejected:
    """Structured-output schema tasks: ``{"schema": ..., "schema_type": "json" | "xml" | ...}``."""
    data = verifier_data(task)
    raw_type = data.get("schema_type")
    try:
        schema_type = SchemaType(raw_type)
    except ValueError:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"unknown schema_type {raw_type!r}")
    graded = graded_by(schema_type, data.get("schema"))
    if isinstance(graded, Rejected):
        return graded
    spec, data_files = graded
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=spec,
        dockerfile=task.text(DOCKERFILE),
        tags=("structured-outputs", mode_of(spec).value, "nemotron", schema_type.value),
        data_files=data_files,
    )


CONVERTER = Converter(
    name="nemotron_structured_outputs",
    keys=(ConverterKey("other", frozenset({"tests/test.sh", "tests/verifier.py"})),),
    convert=convert_nemotron_structured_outputs,
)
