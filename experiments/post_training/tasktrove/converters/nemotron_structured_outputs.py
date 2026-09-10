# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron structured-outputs schema tasks.

``tests/verifier_data.json`` carries ``{"schema": <json-schema-like dict>, "schema_type": "json" |
"yaml" | "toml" | "xml" | "csv"}``. The old grader parsed the candidate per ``schema_type`` and
either validated it against the schema with ``jsonschema`` (``json``/``yaml``/``toml``) or ran a
weaker structural check -- well-formed plus required keys present as tags/columns -- for
``xml``/``csv``. ``tasktrove_verify``'s ``json-schema`` mode only parses candidates as JSON or
YAML (:class:`SchemaFormat`), so it reproduces the old grader exactly for ``schema_type in
("json", "yaml")`` and cannot express the other three: ``toml`` has no format, and the ``xml``/
``csv`` structural checks are a different, weaker grading semantics the mode does not implement.

The dataset's schemas are LLM-generated and often not quite valid JSON Schema; ``normalize_schema`` repairs the
shapes worth fixing (see ``json_schemas``) and a metaschema check after normalizing rejects the rest.
"""

import json

from tasktrove_verify.spec import JsonSchemaSpec, SchemaFormat

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.json_schemas import is_trivial, normalize_schema, schema_error
from experiments.post_training.tasktrove.converters.nemotron_data import metadata, verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

SCHEMA_NAME = "schema.json"
SCHEMA_FILE = f"tests/{SCHEMA_NAME}"

_SCHEMA_FORMAT_BY_TYPE = {"json": SchemaFormat.JSON, "yaml": SchemaFormat.YAML}


def _clean_metadata(task: TaskFiles) -> dict:
    """``metadata(task)`` with ``None`` values dropped.

    ``schema_fields_count`` is JSON ``null`` on ~40% of this source's rows (the adapter's field
    count came back empty); ``render_task_toml`` hands the whole dict to ``tomlkit``, which has no
    TOML representation for ``None`` and raises ``ConvertError``.
    """
    return {key: value for key, value in metadata(task).items() if value is not None}


def convert_nemotron_structured_outputs(task: TaskFiles) -> ConvertedTask | Rejected:
    """Structured-output schema tasks: ``{"schema": ..., "schema_type": "json" | "yaml" | ...}``."""
    data = verifier_data(task)
    schema_type = data.get("schema_type")
    schema_format = _SCHEMA_FORMAT_BY_TYPE.get(schema_type)
    if schema_format is None:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"schema_type {schema_type!r} has no json-schema mode format")
    schema = data.get("schema")
    if not isinstance(schema, dict):
        return Rejected(ConvertStatus.NULL_GRADER, f"schema is not a JSON object: {type(schema).__name__}")
    normalized = normalize_schema(schema)
    if is_trivial(normalized):
        return Rejected(ConvertStatus.NULL_GRADER, "schema has no properties or required fields to check")
    error = schema_error(normalized)
    if error is not None:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"schema fails Draft 2020-12 metaschema check: {error}")
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=JsonSchemaSpec(schema=SCHEMA_NAME, format=schema_format),
        dockerfile=task.text(DOCKERFILE),
        tags=("structured-outputs", "json-schema", "nemotron", schema_type),
        data_files={SCHEMA_FILE: json.dumps(normalized, indent=2).encode()},
        metadata=_clean_metadata(task),
    )


CONVERTER = Converter(
    name="nemotron_structured_outputs",
    keys=(ConverterKey("other", frozenset({"tests/test.sh", "tests/verifier.py"})),),
    convert=convert_nemotron_structured_outputs,
)
