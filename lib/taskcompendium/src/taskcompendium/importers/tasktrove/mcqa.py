# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the supported cleaned TaskTrove MCQA answer format."""

import hashlib
import json
import tomllib

from tasktrove_verify.spec import McqSpec, parse_spec

from taskcompendium.importers.tasktrove.convert import INSTRUCTION_FILE, METADATA_TABLE, TASK_MANIFEST, VERIFIER_FILE
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer

FAMILY = "qa-short-answer"
CONVERTER = "nemotron_mcqa"
_PREFIX = (
    "You are answering a multiple-choice question. Read the question below and write your final "
    "answer to `/app/answer.txt`.\n\n"
    "The verifier extracts a single letter (A/B/C/...) from your answer file using a regex pattern; "
    "the simplest valid output is a file containing exactly\n`Answer: X` (where X is your chosen letter).\n\n"
    "---\n\n"
)
_FORMAT_PREFIX = (
    "Answer the following multiple choice question. The last line of your response "
    "should be in the following format: "
)


def _clean_instructions(instructions: str, options: int) -> str:
    """Replace a recognized answer-file template with a one-letter request."""
    if not instructions.startswith(_PREFIX):
        raise ValueError("Unsupported MCQA instruction template")
    prompt = instructions.removeprefix(_PREFIX)
    first_line, separator, question = prompt.partition("\n\n")
    letters = tuple(chr(ord("A") + index) for index in range(options))
    option_list = "/".join(letters)
    valid_formats = set()
    for wrapper in ("{}", "\\boxed{{{}}}"):
        expected_format = f"{_FORMAT_PREFIX}'Answer: {wrapper.format(option_list)}'"
        for example in letters:
            valid_formats.add(f"{expected_format} (e.g. 'Answer: {wrapper.format(example)}').")
    if not separator or first_line not in valid_formats or not question.strip():
        raise ValueError("Unsupported MCQA instruction format")
    return f"{question.strip()}\n\nChoose one option letter from A through {letters[-1]}."


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import a cleaned MCQA archive as a text-answer task."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if metadata.get("family") != FAMILY or metadata.get("converter") != CONVERTER or metadata.get("mode") != "mcq":
            raise ValueError("Unsupported TaskTrove MCQA source")
        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("TaskTrove tags must be an ordered list of strings")
        contract = parse_spec(archive.files[VERIFIER_FILE].decode())
        if not isinstance(contract, McqSpec):
            raise ValueError("TaskTrove MCQA archive must declare an MCQ verifier")
        instructions = _clean_instructions(archive.files[INSTRUCTION_FILE].decode(), contract.options)
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove MCQA archive: {error}") from error
    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=instructions),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=multiple_choice_answer(contract.expected, contract.options),
        source=archive.source,
        tags=tuple(tags),
    )
