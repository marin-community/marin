# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import TaskTrove JSON-schema tasks as direct text-answer tasks."""

import json
import re
from types import MappingProxyType

from tasktrove_verify.spec import JsonSchemaSpec, SchemaFormat, parse_spec

from taskcompendium.grading import json_schema_answer
from taskcompendium.importers.tasktrove.convert import VERIFIER_FILE, import_metadata, task_id
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage

CONVERTERS = MappingProxyType(
    {
        "nemotron_structured_outputs": "laion__nemotron-gym-structured-outputs-v4",
        "nemotron_if_structured": "laion__nemotron-gym-instruction-following-structured-v3",
    }
)


def _clean_instruction(instruction: str) -> str:
    """Drop source file-writing directions and preserve the underlying task prompt."""
    instruction = instruction.split("\n## Submitting your answer", maxsplit=1)[0]
    instruction = re.sub(r"Write your final [^\n.]+ to `[^`]+`\.\s*", "", instruction, flags=re.IGNORECASE)
    if "/app/" in instruction or "terminal agent" in instruction.lower():
        raise ValueError("Unsupported file-oriented JSON-schema instruction")
    return instruction.strip()


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import a JSON-schema task and expose its schema and answer format to the model."""
    try:
        metadata = import_metadata(archive)
        if (
            metadata.mode != "json-schema"
            or CONVERTERS.get(metadata.converter) != metadata.source
            or archive.source.row.partition(":")[0] != metadata.source
        ):
            raise ValueError("Unsupported TaskTrove JSON-schema source")
        contract = parse_spec(archive.files[VERIFIER_FILE].decode())
        if not isinstance(contract, JsonSchemaSpec) or contract.schema != "schema.json":
            raise ValueError("JSON-schema tasks must use tests/schema.json")
        schema = json.loads(archive.files["tests/schema.json"])
        if not isinstance(schema, dict):
            raise ValueError("TaskTrove JSON Schema must be an object")
        instruction = _clean_instruction(archive.files["instruction.md"].decode())
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove JSON-schema archive: {error}") from error
    schema_format = SchemaFormat(contract.format)
    schema_block = json.dumps(schema, indent=2, sort_keys=True)
    instruction = (
        f"{instruction}\n\nReturn only a {schema_format.value.upper()} document "
        f"matching this JSON Schema:\n```json\n{schema_block}\n```"
    )
    return TaskSpec(
        id=task_id(archive),
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=json_schema_answer(schema, schema_format),
        source=archive.source,
        tags=metadata.tags,
    )
