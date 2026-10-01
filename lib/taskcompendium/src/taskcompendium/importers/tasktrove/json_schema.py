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

_SUBMISSION_FOOTER = """ (IMPORTANT)
You are a terminal agent. Your chat reply is NOT graded — the grader only reads the file
`/app/answer.txt` inside the sandbox. You MUST write your {format} document to `/app/answer.txt`
by RUNNING A SHELL COMMAND, e.g. a heredoc:

    cat > /app/answer.txt <<'EOF'
    <your {format} document here>
    EOF

Then confirm it with `cat /app/answer.txt`. An empty or missing `/app/answer.txt` scores 0
regardless of what you wrote in your reply.
"""
_SCHEMA_HARNESS = re.compile(
    r"The verifier parses your answer \(optionally unwrapping a ```json fence\) and "
    r"validates (?:it with|against the schema with) `jsonschema` Draft 2020-12\.",
    re.IGNORECASE,
)

CONVERTERS = MappingProxyType(
    {
        "nemotron_structured_outputs": "laion__nemotron-gym-structured-outputs-v4",
        "nemotron_if_structured": "laion__nemotron-gym-instruction-following-structured-v3",
    }
)


def _clean_instruction(instruction: str, schema_format: SchemaFormat) -> str:
    """Drop source file-writing directions and preserve the underlying task prompt."""
    instruction, separator, footer = instruction.partition("\n## Submitting your answer")
    if separator and " ".join(footer.split()) != " ".join(
        _SUBMISSION_FOOTER.format(format=schema_format.value.upper()).split()
    ):
        raise ValueError("Unsupported JSON-schema submission footer")
    instruction = re.sub(r"Write your final [^\n.]+ to `[^`]+`\.\s*", "", instruction, flags=re.IGNORECASE)
    instruction = _SCHEMA_HARNESS.sub("", instruction)
    instruction = instruction.replace("# Evaluation contract", "# Answer requirements")
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
        instruction = _clean_instruction(archive.files["instruction.md"].decode(), SchemaFormat(contract.format))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove JSON-schema archive: {error}") from error
    schema_format = SchemaFormat(contract.format)
    schema_block = json.dumps(schema, indent=2, sort_keys=True)
    instruction = (
        f"{instruction}\n\nProvide a {schema_format.value.upper()} document "
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
