# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import self-contained TaskTrove tasks with a generic numeric contract."""

import hashlib
import json
import tomllib

from tasktrove_verify.spec import NumericSpec, parse_spec

from taskcompendium.grading import numeric_answer
from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage

_ALLOWED_MEMBERS = frozenset({TASK_MANIFEST, "instruction.md", "tests/verifier.toml", "environment/Dockerfile"})


def import_task(archive: TaskArchive) -> TaskSpec:
    """Import a plain numeric question while retaining its parsed grading tolerances."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if metadata.get("mode") != "numeric":
            raise ValueError("TaskTrove numeric archive must declare mode=numeric")
        for field in ("family", "converter", "template_id"):
            if not isinstance(metadata.get(field), str) or not metadata[field]:
                raise ValueError(f"TaskTrove numeric archive requires {field} metadata")
        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("TaskTrove tags must be an ordered list of strings")
        unexpected_members = sorted(set(archive.files) - _ALLOWED_MEMBERS)
        if unexpected_members:
            raise ValueError(f"TaskTrove numeric task has unsupported supporting files: {unexpected_members}")
        instruction = archive.files["instruction.md"].decode()
        if not instruction.strip():
            raise ValueError("TaskTrove numeric instruction is empty")
        if "/app/" in instruction:
            raise ValueError("TaskTrove numeric instruction requires an unsupported file-based environment")
        contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(contract, NumericSpec):
            raise ValueError("TaskTrove numeric archive must declare a numeric verifier")
        verifier = numeric_answer(contract.expected, contract.tolerance_abs, contract.tolerance_rel)
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove numeric archive: {error}") from error

    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=verifier,
        source=archive.source,
        tags=tuple(tags),
    )
