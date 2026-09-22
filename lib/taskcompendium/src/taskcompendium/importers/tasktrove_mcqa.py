# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the supported cleaned TaskTrove MCQA answer format."""

import hashlib
import json
import tomllib

from tasktrove_verify.spec import McqSpec, parse_spec

from taskcompendium.importers.tasktrove import TaskArchive
from taskcompendium.models import AnswerFormat, AnswerKind, TaskRequirements, TaskSpec
from taskcompendium.verifiers.tasktrove_mcqa import tasktrove_mcqa

FAMILY = "qa-short-answer"
CONVERTER = "nemotron_mcqa"
_PREFIX = (
    "You are answering a multiple-choice question. Read the question below and write your final "
    "answer to `/app/answer.txt`.\n\n"
    "The verifier extracts a single letter (A/B/C/...) from your answer file using a regex pattern; "
    "the simplest valid output is a file containing exactly\n`Answer: X` (where X is your chosen letter).\n\n"
    "---\n\n"
)
_FORMAT_PREFIX = "Answer the following multiple choice question. The last line of your response should be"


def _clean_instructions(instructions: str) -> str:
    if not instructions.startswith(_PREFIX):
        raise ValueError("Unsupported MCQA instruction template")
    prompt = instructions.removeprefix(_PREFIX)
    first_line, separator, question = prompt.partition("\n\n")
    if not separator or not first_line.startswith(_FORMAT_PREFIX) or not question.strip():
        raise ValueError("MCQA instruction template has no question")
    return question.strip()


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import a cleaned Nemotron MCQA archive with its original verifier contract."""
    try:
        metadata = tomllib.loads(archive.files["task.toml"].decode())["metadata"]
        if metadata.get("family") != FAMILY or metadata.get("converter") != CONVERTER:
            raise ValueError("Unsupported TaskTrove MCQA source")
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, McqSpec):
            raise ValueError("TaskTrove MCQA archive must declare an MCQ verifier")
        instructions = _clean_instructions(archive.files["instruction.md"].decode())
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove MCQA archive: {error}") from error
    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        instructions=instructions,
        verifier=tasktrove_mcqa(contract.expected, contract.options),
        source=archive.source,
        requirements=TaskRequirements(),
        answer_kind=AnswerKind.OPTION_LETTER,
        permitted_answer_formats=(AnswerFormat.PLAIN, AnswerFormat.JSON),
    )
