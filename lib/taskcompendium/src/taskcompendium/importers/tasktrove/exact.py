# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import answer-only TaskTrove Clean tasks graded by exact mode."""

from tasktrove_verify.spec import ExactSpec, parse_spec

from taskcompendium.grading import exact_list_answer
from taskcompendium.importers.tasktrove.convert import import_metadata, task_id
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
        metadata = import_metadata(archive)
        if metadata.family != FAMILY or metadata.converter != CONVERTER or metadata.mode != "exact":
            raise ValueError("Unsupported TaskTrove exact-mode source")
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, ExactSpec):
            raise ValueError("TaskTrove archive must declare an exact verifier")
        instruction = _clean_instruction(archive.files["instruction.md"].decode())
    except (KeyError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove exact-mode archive: {error}") from error
    verifier = exact_list_answer(
        contract.expected,
        ignore_case=contract.ignore_case,
        collapse_whitespace=contract.ignore_whitespace,
        ordered=contract.ordered,
    )
    return TaskSpec(
        id=task_id(archive),
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=verifier,
        source=archive.source,
        tags=metadata.tags,
    )
