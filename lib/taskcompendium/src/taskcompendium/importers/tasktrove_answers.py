# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import TaskTrove answer-only MCQA and exact-answer tasks."""

import re
import tomllib
from typing import Any

from tasktrove_verify.spec import Mode, mode_of

from taskcompendium.importers.tasktrove import TaskArchive, puzzle_instructions, semantic_verifier
from taskcompendium.models import (
    AnswerRequirements,
    NoEnvironment,
    PythonRuntime,
    Rejected,
    RejectionReason,
    TaskMetadata,
    TaskSpecification,
)

_MCQ_FAMILY = "qa-short-answer"
_EXACT_FAMILY = "math-answer"
_SUPPORTED_FAMILIES = frozenset({_MCQ_FAMILY, _EXACT_FAMILY})
_WORD_LIST = re.compile(r"return them as a comma-separated list:\s*(?P<words>[^\n]+)", re.IGNORECASE)
_MCQ_PREFIX = (
    "You are answering a multiple-choice question. Read the question below and write your final "
    "answer to `/app/answer.txt`.\n\n"
    "The verifier extracts a single letter (A/B/C/...) from your answer file using a regex pattern; "
    "the simplest valid output is a file containing exactly\n`Answer: X` (where X is your chosen letter).\n\n"
    "---\n\n"
)
_MCQ_FORMAT_LINE = re.compile(
    r"^Answer the following multiple choice question\. The last line of your response should be in "
    r"the following format: 'Answer: \\boxed\{A/B/C/D/E/F/G/H/I/J\}' "
    r"\(e\.g\. 'Answer: \\boxed\{E\}'\)\.\n*",
    re.MULTILINE,
)
_MCQ_GOLD_ERROR = "MCQ verifier must have one expected option letter"
_EXACT_GOLD_ERROR = "exact verifier must have nonempty string references"
_DESCENDING_GOLD_ERROR = "descending-order instruction has ascending expected output"


def _metadata(archive: TaskArchive) -> dict[str, Any]:
    raw = archive.files.get("task.toml")
    if raw is None:
        raise ValueError("missing task.toml")
    value = tomllib.loads(raw.decode())
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _clean_mcq(instructions: str) -> str:
    # These strings are emitted by the nemotron_mcqa converter itself. Keep the
    # question and options byte-for-byte, while removing delivery conventions.
    if not instructions.startswith(_MCQ_PREFIX):
        return ""
    instructions = instructions[len(_MCQ_PREFIX) :]
    instructions = _MCQ_FORMAT_LINE.sub("", instructions, count=1)
    return instructions.strip()


def _defective_descending_gold(instructions: str, expected: tuple[str, ...]) -> bool:
    """Reject the known all-puzzles rows with an ascending gold for descending text."""
    if "descending order" not in instructions.lower():
        return False
    match = _WORD_LIST.search(instructions)
    if match is None:
        return False
    words = tuple(word.strip() for word in match.group("words").split(","))
    return (
        len(words) == len(expected)
        and expected == tuple(sorted(words))
        and expected != tuple(sorted(words, reverse=True))
    )


def import_task(archive: TaskArchive) -> TaskSpecification | Rejected:
    """Convert a cleaned TaskTrove answer-only archive."""
    source = archive.source
    if archive.family not in _SUPPORTED_FAMILIES:
        detail = f"unsupported answer family {archive.family!r}"
        return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, detail)
    try:
        instructions = archive.instructions
        verifier = archive.verifier
    except (KeyError, UnicodeDecodeError, ValueError) as error:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, str(error))
    verifier_mode = mode_of(verifier)
    if verifier_mode not in {Mode.MCQ, Mode.EXACT}:
        return Rejected(
            source, RejectionReason.UNSUPPORTED_VERIFIER, f"unsupported verifier mode {verifier_mode.value!r}"
        )
    semantic = semantic_verifier(verifier)
    parameters = semantic.parameters
    expected = parameters.get("expected")
    if archive.family == _MCQ_FAMILY:
        if (
            verifier_mode != Mode.MCQ
            or not isinstance(expected, str)
            or len(expected.strip()) != 1
            or not "A" <= expected.strip().upper() <= "Z"
            or not isinstance(parameters.get("options"), int)
            or not 1 <= parameters["options"] <= 26
            or ord(expected.strip().upper()) - ord("A") >= parameters["options"]
        ):
            return Rejected(source, RejectionReason.BROKEN_GRADER, _MCQ_GOLD_ERROR)
        cleaned = _clean_mcq(instructions)
        requirements = AnswerRequirements("value")
    else:
        if (
            verifier_mode != Mode.EXACT
            or not isinstance(expected, tuple)
            or not expected
            or any(not isinstance(item, str) or not item for item in expected)
        ):
            return Rejected(source, RejectionReason.BROKEN_GRADER, _EXACT_GOLD_ERROR)
        if _defective_descending_gold(instructions, expected):
            return Rejected(
                source,
                RejectionReason.BROKEN_GRADER,
                _DESCENDING_GOLD_ERROR,
            )
        try:
            cleaned = puzzle_instructions(instructions)
        except ValueError as error:
            return Rejected(source, RejectionReason.UNDERSPECIFIED, str(error))
        requirements = AnswerRequirements("value")
    if not cleaned:
        return Rejected(source, RejectionReason.UNDERSPECIFIED, "source instruction has no semantic task text")
    try:
        metadata = _metadata(archive)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError) as error:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, f"invalid task.toml: {error}")
    tags = metadata.get("tags", ())
    competencies = tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()
    return TaskSpecification(
        id=f"tasktrove-{source.row}",
        instructions=cleaned,
        environment=NoEnvironment(),
        resources=(),
        verifier=semantic,
        verifier_runtime=PythonRuntime(),
        metadata=TaskMetadata(source=source, competencies=competencies, task_shape="answer"),
        answer_requirements=requirements,
    )
