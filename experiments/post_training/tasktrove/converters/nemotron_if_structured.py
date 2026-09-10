# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron instruction-following tasks keyed under the ``instruction-following`` family whose old
grader is ``tests/verifier.py`` plus ``tests/validate_verifier_data.py``.

Two sources share this template but ship different ``tests/verifier_data.json`` shapes:

- ``...-structured-v3``: ``{"schema": <JSON Schema>, "schema_type": "json"}``. The old grader
  validated the answer file against the schema with ``jsonschema`` -- :class:`JsonSchemaSpec`.
- ``...-calendar-v3``: ``{"expected_events": {...}}``, the same scheduling grader as the
  agent-calendar source, so those rows go through that converter's checker.
"""

import json

from tasktrove_verify.spec import JsonSchemaSpec, SchemaFormat

from experiments.post_training.tasktrove.converters.agent_calendar import convert_agent_calendar
from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    Rejected,
)
from experiments.post_training.tasktrove.converters.json_schemas import usable_schema
from experiments.post_training.tasktrove.converters.nemotron_data import metadata, verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

SCHEMA_FILE = "schema.json"


def convert_nemotron_if_structured(task: TaskFiles) -> ConvertedTask | Rejected:
    data = verifier_data(task)
    if "expected_events" in data:
        return convert_agent_calendar(task)
    normalized = usable_schema(data.get("schema"))
    if isinstance(normalized, Rejected):
        return normalized
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=JsonSchemaSpec(schema=SCHEMA_FILE, format=SchemaFormat.JSON),
        dockerfile=task.text(DOCKERFILE),
        tags=("instruction-following", "structured-output", "json-schema", "nemotron"),
        data_files={f"tests/{SCHEMA_FILE}": json.dumps(normalized, indent=2).encode()},
        metadata=metadata(task),
    )


CONVERTER = Converter(
    name="nemotron_if_structured",
    keys=(
        ConverterKey(
            "instruction-following", frozenset({"tests/test.sh", "tests/validate_verifier_data.py", "tests/verifier.py"})
        ),
    ),
    convert=convert_nemotron_if_structured,
)
