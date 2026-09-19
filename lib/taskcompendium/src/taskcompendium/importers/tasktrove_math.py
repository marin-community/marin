# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the supported TaskTrove math-answer source wrappers."""

import tomllib
from typing import Any

from tasktrove_verify.spec import MathType, Mode, mode_of

from taskcompendium.importers.tasktrove import TaskArchive, puzzle_instructions, semantic_verifier
from taskcompendium.models import (
    AnswerRequirements,
    Rejected,
    RejectionReason,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
)

_FAMILY = "math-answer"
_ALL_PUZZLES = "all_puzzles"
_SUPPORTED_CONVERTERS = frozenset({_ALL_PUZZLES})


def _metadata(archive: TaskArchive) -> dict[str, Any]:
    raw = archive.files.get("task.toml")
    if raw is None:
        raise ValueError("missing task.toml")
    value = tomllib.loads(raw.decode())
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _converter(metadata: dict[str, Any]) -> str:
    value = metadata.get("converter")
    return value if isinstance(value, str) else ""


def import_task(archive: TaskArchive) -> TaskSpec | Rejected:
    """Convert a supported deterministic TaskTrove math archive."""
    source = archive.source
    if archive.family != _FAMILY:
        return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, f"unsupported math family {archive.family!r}")
    try:
        instructions = archive.instructions
        verifier = archive.verifier
        metadata = _metadata(archive)
    except (KeyError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as error:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, str(error))
    converter = _converter(metadata)
    if converter not in _SUPPORTED_CONVERTERS:
        return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, f"unsupported math converter {converter!r}")
    if metadata.get("answer_type") not in {"number", "coords"}:
        return Rejected(
            source, RejectionReason.BROKEN_GRADER, "all-puzzles math requires a number or coordinates answer"
        )
    if mode_of(verifier) is not Mode.MATH:
        return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, "math family does not have a math verifier")
    semantic = semantic_verifier(verifier, implementation_revision=archive.release.verifier_revision)
    expected = semantic.parameters.get("expected")
    math_type = semantic.parameters.get("math_type")
    if not isinstance(expected, str) or not expected.strip():
        return Rejected(source, RejectionReason.BROKEN_GRADER, "math verifier must have nonempty expected answer")
    if not isinstance(math_type, MathType):
        try:
            MathType(math_type)
        except (TypeError, ValueError):
            return Rejected(source, RejectionReason.BROKEN_GRADER, "math verifier has unknown math type")
    try:
        cleaned = puzzle_instructions(instructions)
    except ValueError as error:
        return Rejected(source, RejectionReason.UNDERSPECIFIED, str(error))
    tags = metadata.get("tags", ())
    competencies = tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()
    return TaskSpec(
        id=f"tasktrove-{source.row}",
        requirements=TaskRequirements(),
        resources=(),
        metadata=TaskMetadata(source=source, competencies=competencies, task_shape="answer"),
        steps=(
            StepSpecification(
                instructions=cleaned,
                verifier=semantic,
                answer_requirements=AnswerRequirements("text"),
            ),
        ),
    )
