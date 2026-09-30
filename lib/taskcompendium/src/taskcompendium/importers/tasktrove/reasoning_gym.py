# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import cleaned TaskTrove Reasoning Gym tasks for direct chat."""

import hashlib
import json
import tomllib

from tasktrove_verify.spec import ReasoningGymSpec, parse_spec

from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.verifiers.reasoning_gym import reasoning_gym_answer

FAMILY = "other"
CONVERTER = "nemotron_reasoning"
IMPORTER_REVISION = "taskcompendium-tasktrove-reasoning-gym-v0.1"
ANSWER_PATH = "/app/answer.txt"
_INSTRUCTION_PREFIX = (
    "You are solving a procedurally-generated reasoning task from Reasoning Gym. Read the problem below and write "
    f"your final answer to `{ANSWER_PATH}`. The verifier will try the upstream Reasoning Gym scorer first, then fall "
    "back to normalized exact-match.\n\n---\n\n"
)


def _plain_text_instruction(instructions: str) -> str:
    """Replace the known answer-file preamble with a direct text-answer request."""
    if not instructions.startswith(_INSTRUCTION_PREFIX):
        raise ValueError("Unsupported Reasoning Gym instruction template")
    question = instructions.removeprefix(_INSTRUCTION_PREFIX).strip()
    if not question:
        raise ValueError("Reasoning Gym instruction has no question")
    return f"{question}\n\nGive your answer in plain text."


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import a Reasoning Gym archive with its generated entry kept in the verifier."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if (
            metadata.get("family") != FAMILY
            or metadata.get("converter") != CONVERTER
            or metadata.get("mode") != "reasoning-gym"
        ):
            raise ValueError("Unsupported TaskTrove Reasoning Gym source")
        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("TaskTrove tags must be an ordered list of strings")
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, ReasoningGymSpec) or contract.entry != "entry.json":
            raise ValueError("TaskTrove archive must declare a Reasoning Gym verifier with entry.json")
        if contract.output != ANSWER_PATH:
            raise ValueError("TaskTrove Reasoning Gym archive has an unsupported answer path")
        entry = json.loads(archive.files[f"tests/{contract.entry}"])
        if not isinstance(entry, dict):
            raise ValueError("Reasoning Gym entry must be an object")
        instructions = _plain_text_instruction(archive.files["instruction.md"].decode())
        verifier = reasoning_gym_answer(contract.dataset, entry)
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove Reasoning Gym archive: {error}") from error
    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=instructions),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=verifier,
        source=archive.source.model_copy(update={"importer_revision": IMPORTER_REVISION}),
        tags=tuple(tags),
    )
