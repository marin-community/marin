# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import TaskTrove's reference-answer LLM-judge tasks."""

import dataclasses
import re
import tomllib
from typing import Any

from tasktrove_verify.spec import RUBRIC_REFERENCE, JudgeSpec, Mode, mode_of

from taskcompendium.importers.tasktrove import TaskArchive
from taskcompendium.models import (
    AnswerRequirements,
    JudgeConfig,
    NoEnvironment,
    PythonRuntime,
    Rejected,
    RejectionReason,
    TaskMetadata,
    TaskSpecification,
    VerifierSpec,
)

_FAMILY = "qa-short-answer"
_CONVERTER = "nemotron_openqa"
_DELIVERY_PREFIX = (
    "You are answering an open-ended question. Write your concise final answer to "
    "`/app/response.txt`. An LLM judge will compare your response to the reference answer(s) "
    "and score for semantic equivalence (paraphrases and alternative phrasings are acceptable "
    "as long as the substantive answer matches).\n\n---\n\n"
)
_DELIVERY_SUFFIX = re.compile(r"\nRemember to put your answer inside \\boxed\{\}\.\s*\Z")


def _metadata(archive: TaskArchive) -> dict[str, Any]:
    raw = archive.files.get("task.toml")
    if raw is None:
        raise ValueError("missing task.toml")
    value = tomllib.loads(raw.decode())
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _clean_instructions(instructions: str) -> str:
    """Remove the known response-file wrapper while retaining the source question."""
    if not instructions.startswith(_DELIVERY_PREFIX):
        raise ValueError("unsupported judge instruction template")
    cleaned = _DELIVERY_SUFFIX.sub("", instructions[len(_DELIVERY_PREFIX) :]).strip()
    if not cleaned:
        raise ValueError("judge instruction has no semantic task text")
    return cleaned


def _judge_verifier(archive: TaskArchive, judge: JudgeConfig) -> VerifierSpec:
    contract = archive.verifier
    if not isinstance(contract, JudgeSpec) or mode_of(contract) is not Mode.JUDGE:
        raise ValueError("judge family does not have a judge verifier")
    mode = mode_of(contract)
    parameters = dataclasses.asdict(contract)
    parameters.pop("output", None)
    parameters.pop("workspace", None)
    if mode is not Mode.JUDGE:
        raise ValueError("judge family does not have a judge verifier")
    if parameters.get("rubric") != RUBRIC_REFERENCE:
        raise ValueError("openqa judge must use the reference rubric")
    references = parameters.get("references")
    if not isinstance(references, tuple) or not references or any(not item.strip() for item in references):
        raise ValueError("judge verifier must have non-empty reference answers")
    if not parameters.get("exact_gate"):
        raise ValueError("openqa judge must retain its exact gate")
    return VerifierSpec(mode, parameters, judge)


def import_task(archive: TaskArchive, judge: JudgeConfig) -> TaskSpecification | Rejected:
    """Convert a cleaned TaskTrove reference-answer judge archive.

    The model endpoint is deliberately supplied by the caller; source archives never carry
    credentials or an implicit judge policy.
    """
    source = archive.source
    if archive.family != _FAMILY:
        return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, f"unsupported judge family {archive.family!r}")
    try:
        metadata = _metadata(archive)
        if metadata.get("converter") != _CONVERTER:
            return Rejected(source, RejectionReason.UNSUPPORTED_VERIFIER, "unsupported judge converter")
        instructions = _clean_instructions(archive.instructions)
        verifier = _judge_verifier(archive, judge)
    except (KeyError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as error:
        return Rejected(source, RejectionReason.UNRECOVERABLE_SOURCE, str(error))
    tags = metadata.get("tags", ())
    competencies = tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()
    return TaskSpecification(
        id=f"tasktrove-{source.row}",
        instructions=instructions,
        environment=NoEnvironment(),
        resources=(),
        verifier=verifier,
        verifier_runtime=PythonRuntime(),
        metadata=TaskMetadata(source=source, competencies=competencies, task_shape="answer"),
        answer_requirements=AnswerRequirements("value"),
    )
