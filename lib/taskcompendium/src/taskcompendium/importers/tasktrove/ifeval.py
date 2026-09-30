# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import supported TaskTrove Clean IFEval tasks."""

import hashlib
import json
import tomllib

from tasktrove_verify.spec import IfevalSpec, parse_spec

from taskcompendium.importers.tasktrove.convert import INSTRUCTION_FILE, METADATA_TABLE, TASK_MANIFEST, VERIFIER_FILE
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.verifiers.ifeval import IFEvalConstraint, ifeval_answer

FAMILY = "instruction-following"
CONVERTER = "nemotron_ifeval"
IFEVAL_MODE = "ifeval"
_SUBMISSION_DIVIDER = "\n---\n\n"
_SUBMISSION_MARKERS = (
    "The verifier reads ONLY `/app/answer.txt`",
    "Your answer.txt must be the raw answer prose itself",
    "cat > /app/answer.txt",
    "Verify the file exists and contains your answer",
)


def _clean_instructions(instructions: str) -> str:
    """Remove the one recognized shell submission wrapper and preserve the task verbatim."""
    scaffold, separator, prompt = instructions.partition(_SUBMISSION_DIVIDER)
    if (
        not separator
        or not scaffold.startswith("You are running in a shell-based sandbox.")
        or any(marker not in scaffold for marker in _SUBMISSION_MARKERS)
    ):
        raise ValueError("Unsupported IFEval submission scaffold")
    if not prompt.strip():
        raise ValueError("IFEval task prompt is empty")
    return prompt


def _constraints(contract: IfevalSpec) -> tuple[IFEvalConstraint, ...]:
    return tuple(IFEvalConstraint(constraint.name, constraint.params) for constraint in contract.constraints)


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import one cleaned IFEval archive as a text-answer task."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if (
            metadata.get("family") != FAMILY
            or metadata.get("converter") != CONVERTER
            or metadata.get("mode") != IFEVAL_MODE
        ):
            raise ValueError("Unsupported TaskTrove IFEval source")
        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("TaskTrove tags must be an ordered list of strings")
        contract = parse_spec(archive.files[VERIFIER_FILE].decode())
        if not isinstance(contract, IfevalSpec):
            raise ValueError("TaskTrove IFEval archive must declare an IFEval verifier")
        prompt = _clean_instructions(archive.files[INSTRUCTION_FILE].decode())
        verifier = ifeval_answer(_constraints(contract))
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove IFEval archive: {error}") from error
    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=verifier,
        source=archive.source,
        tags=tuple(tags),
    )
