# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import TaskTrove Clean tasks with a supported mathematical answer contract."""

import hashlib
import json
import re
import tomllib

from tasktrove_verify.modes.grade_math import is_finite_real_scalar
from tasktrove_verify.spec import MathSpec, parse_spec

from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST
from taskcompendium.importers.tasktrove.models import (
    TaskArchive,
    TaskTroveImportResult,
    TaskTroveSourceEvidence,
)
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.verifiers.mathematical import mathematical_answer

FAMILY = "math-answer"
MODE = "math"
_SUPPORTED_FILES = frozenset(
    {TASK_MANIFEST, "instruction.md", "environment/Dockerfile", "tests/test.sh", "tests/verifier.toml"}
)
_FILE_SUBMISSION = re.compile(
    r"^\s*(?:please\s+)?(?:write|save|put|submit|store|place)\s+(?:your\s+)?(?:final\s+)?"
    r"(?:answer|response)\s+(?:to|in|at)\s+[`']?/app/(?:answer|solution)\.(?:txt|json)[`']?[.!]?\s*$",
    re.IGNORECASE,
)
_TOOL_REQUIREMENT = re.compile(
    r"\b(?:run|execute|install|download|open|read)\s+(?:the\s+)?(?:provided\s+)?"
    r"(?:[\w-]+\s+){0,3}(?:script|program|code|file|tool|calculator|notebook)\b|"
    r"\buse\s+(?:python|a\s+calculator)\b",
    re.IGNORECASE,
)
_UNKNOWN_FILE_REFERENCE = re.compile(r"\b(?:answer|solution)\.txt\b|/app/(?:answer|solution)\.", re.IGNORECASE)
_REPEATED_BLANK_LINES = re.compile(r"\n{3,}")
_NEMOTRON_OUTPUT_PATH = re.compile(r"Write the final answer to `/app/answer\.txt`\.\s*", re.IGNORECASE)
_ALL_PUZZLES_DELIVERABLE = re.compile(
    r"## Deliverable \(REQUIRED\)\s*\n\s*"
    r"Write ONLY your final answer to \*\*`/app/answer\.txt`\*\* \(a single line, no\s*\n"
    r"explanation\)\. The verifier reads that file and compares it to the gold answer\s*\n"
    r"using the format described in the problem statement:\s*\n",
    re.IGNORECASE,
)


def _metadata(archive: TaskArchive) -> tuple[tuple[str, ...], TaskTroveSourceEvidence]:
    manifest = tomllib.loads(archive.files[TASK_MANIFEST].decode())
    metadata = manifest[METADATA_TABLE]
    if metadata.get("family") != FAMILY or metadata.get("mode") != MODE:
        raise ValueError("Unsupported TaskTrove mathematical source")
    raw_tags = metadata.get("tags")
    if not isinstance(raw_tags, list) or any(not isinstance(tag, str) for tag in raw_tags):
        raise ValueError("TaskTrove mathematical source must declare ordered string tags")
    evidence = TaskTroveSourceEvidence(
        source=archive.upstream_subset,
        path=archive.archive_path,
        family=metadata["family"],
        converter=metadata["converter"],
        template_id=metadata["template_id"],
        mode=metadata["mode"],
        archive_sha256=archive.archive_sha256,
    )
    return tuple(raw_tags), evidence


def _instructions(archive: TaskArchive, converter: str) -> str:
    extras = set(archive.files) - _SUPPORTED_FILES
    if extras:
        raise ValueError(f"Mathematical task requires unsupported task resources: {sorted(extras)}")
    instruction = archive.files["instruction.md"].decode()
    if converter == "nemotron_math":
        instruction = instruction.replace("Provide your answer in the file answer.txt", "")
        instruction = instruction.replace("## Submitting the answer", "")
        instruction = _NEMOTRON_OUTPUT_PATH.sub("", instruction)
    elif converter == "all_puzzles":
        instruction = instruction.replace("<!-- laion v2 puzzles deliverable: answer.txt -->", "")
        instruction, count = _ALL_PUZZLES_DELIVERABLE.subn(
            "## Answer format\n\nWrite only your final answer on one line, without explanation. "
            "Use the required answer format:\n",
            instruction,
        )
        if count != 1:
            raise ValueError("Unsupported all-puzzles answer-delivery template")
    else:
        raise ValueError(f"Unsupported TaskTrove mathematical converter {converter!r}")
    if _TOOL_REQUIREMENT.search(instruction):
        raise ValueError("Mathematical task requires tool execution or an external resource")
    retained_lines = [line for line in instruction.splitlines() if not _FILE_SUBMISSION.fullmatch(line)]
    cleaned = _REPEATED_BLANK_LINES.sub("\n\n", "\n".join(retained_lines)).strip()
    if not cleaned:
        raise ValueError("Mathematical task has no question after removing submission scaffolding")
    if _UNKNOWN_FILE_REFERENCE.search(cleaned):
        raise ValueError("Unsupported TaskTrove mathematical submission instruction")
    return cleaned


def import_task(archive: TaskArchive) -> TaskTroveImportResult:
    """Convert a math-mode TaskTrove Clean archive to a text or numeric task."""
    try:
        tags, evidence = _metadata(archive)
        instructions = _instructions(archive, evidence.converter)
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, MathSpec):
            raise ValueError("TaskTrove mathematical archive must declare a MathSpec")
        verifier = mathematical_answer(contract.expected, contract.math_type)
        answer_type = AnswerType.NUMBER if is_finite_real_scalar(contract) else AnswerType.TEXT
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove mathematical archive: {error}") from error

    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    specification = TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=instructions),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=answer_type,
        tags=tags,
        verifier=verifier,
        source=archive.source,
    )
    return TaskTroveImportResult(specification=specification, tags=tags, source_evidence=evidence)
