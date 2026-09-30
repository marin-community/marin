# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import supported TaskTrove Clean free-form judge task templates."""

import hashlib
import json
import tomllib
from pathlib import PurePosixPath

from tasktrove_verify.spec import JudgeSpec, parse_spec

from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.verifiers.judge import MAX_CONTEXT_CHARS, judge_answer

IMPORTER_REVISION = "taskcompendium-tasktrove-judge-v0.1"
SUPPORTED_TEMPLATES = frozenset({("qa-short-answer", "nemotron_openqa"), ("llm-judge-freeform", "judge_rubric")})


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import a supported TaskTrove judge archive into a private TaskSpec."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        family = metadata.get("family")
        converter = metadata.get("converter")
        if not isinstance(family, str) or not isinstance(converter, str):
            raise ValueError("TaskTrove judge archive requires a family and converter")
        template = (family, converter)
        if metadata.get("mode") != "judge" or template not in SUPPORTED_TEMPLATES:
            raise ValueError(f"Unsupported TaskTrove judge template: {template!r}")
        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("TaskTrove tags must be an ordered list of strings")
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, JudgeSpec):
            raise ValueError("TaskTrove judge archive must declare a judge verifier")
        instructions = archive.files["instruction.md"].decode()
        prompt = contract.question.strip()
        if not prompt or prompt not in instructions:
            raise ValueError("TaskTrove judge question is not present in its instruction template")
        context = _context(archive, contract)
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove judge archive: {error}") from error

    identity = json.dumps((archive.source.dataset, archive.source.revision, archive.source.row), separators=(",", ":"))
    source = archive.source.model_copy(update={"importer_revision": IMPORTER_REVISION})
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=judge_answer(contract, context=context),
        source=source,
        tags=tuple(tags),
    )


def _context(archive: TaskArchive, spec: JudgeSpec) -> str | None:
    if not spec.context:
        return None
    path = PurePosixPath(spec.context)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Judge context path must be relative to the private tests directory")
    context_path = f"tests/{path.as_posix()}"
    content = archive.files.get(context_path)
    if content is None:
        raise ValueError(f"TaskTrove judge context file is missing: {context_path}")
    context = content.decode()
    if len(context) > MAX_CONTEXT_CHARS:
        raise ValueError(f"TaskTrove judge context exceeds {MAX_CONTEXT_CHARS} characters")
    return context
