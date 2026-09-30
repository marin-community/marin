# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import answer-only TaskTrove Clean tasks graded by exact mode."""

import hashlib
import json
import tomllib

from tasktrove_verify.spec import ExactSpec, parse_spec

from taskcompendium.grading import exact_list_answer
from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage

FAMILY = "math-answer"
CONVERTER = "all_puzzles"


def _clean_instruction(instruction: str) -> str:
    """Keep the puzzle content while removing the source's file-based harness contract."""
    title, separator, remainder = instruction.partition("\n")
    _, puzzle_separator, content = remainder.partition("## Puzzle Type")
    problem, task_separator, _ = content.partition("## Task")
    if not separator or not puzzle_separator or not task_separator or not problem.strip():
        raise ValueError("Unsupported exact-mode puzzle instruction")
    return f"{title.strip()}\n\n## Puzzle Type{problem.rstrip()}\n\nReturn only the requested answer."


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import a source exact-mode puzzle as a private text-answer task."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if metadata.get("family") != FAMILY or metadata.get("converter") != CONVERTER or metadata.get("mode") != "exact":
            raise ValueError("Unsupported TaskTrove exact-mode source")
        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("TaskTrove tags must be an ordered list of strings")
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, ExactSpec):
            raise ValueError("TaskTrove archive must declare an exact verifier")
        instruction = _clean_instruction(archive.files["instruction.md"].decode())
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove exact-mode archive: {error}") from error
    verifier = exact_list_answer(
        contract.expected,
        ignore_case=contract.ignore_case,
        collapse_whitespace=contract.ignore_whitespace,
        ordered=contract.ordered,
    )
    identity = json.dumps((archive.source.dataset, archive.source.revision, archive.source.row), separators=(",", ":"))
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=verifier,
        source=archive.source,
        tags=tuple(tags),
    )
