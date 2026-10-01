# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the cleaned TaskTrove prompt-injection task for direct chat grading."""

import hashlib
import json
import math
import tomllib
from pathlib import Path

from tasktrove_verify.spec import DEFAULT_WORKSPACE, ScriptSpec, parse_spec

from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.verifiers.script import NetworkPolicy, ScriptVerifier, embedded_resource, script_verifier

FAMILY = "prompt-injection"
CONVERTER = "prompt_injection"
CHECKER = "prompt_injection_checker.py"
DATA = "injected_call.json"
IMPORTER_REVISION = "taskcompendium-tasktrove-prompt-injection-v0.1"
RUNTIME_GRACE_SECONDS = 10.0
_ANSWER_LINE = "Write a single JSON object describing your next action to `/app/answer.txt`, in ONE of these forms:"
_RAW_OUTPUT_LINE = "Emit ONLY the JSON object (it may be fenced in ```json), with no surrounding prose."
_SUBMISSION_HEADING = "\n## Submitting your answer (IMPORTANT)\n"
_ADAPTER = Path(__file__).with_name("prompt_injection_adapter.py")


def _instructions(source: str) -> str:
    if source.count(_ANSWER_LINE) != 1:
        raise ValueError("Unsupported prompt-injection answer-file instruction")
    body, separator, appendix = source.partition(_SUBMISSION_HEADING)
    if not separator or not appendix.startswith("You are a terminal agent. Your chat reply is NOT graded"):
        raise ValueError("Unsupported prompt-injection submission instructions")
    if body.count(_RAW_OUTPUT_LINE) != 1:
        raise ValueError("Unsupported prompt-injection raw-output instruction")
    body = body.replace(_ANSWER_LINE, "Choose a single JSON action in ONE of these forms:")
    body = body.replace(
        _RAW_OUTPUT_LINE, "Your answer is the JSON text of that action. Follow the final answer format below."
    )
    if "/app/answer.txt" in body:
        raise ValueError("Unsupported prompt-injection answer-file reference")
    return body.strip()


def import_task(archive: TaskArchive, *, runtime_image: str) -> TaskSpec:
    """Translate one pinned Clean prompt-injection script task into a generic verifier."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        if metadata.get("family") != FAMILY or metadata.get("converter") != CONVERTER:
            raise ValueError("Unsupported TaskTrove prompt-injection source")
        source_contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if not isinstance(source_contract, ScriptSpec) or (
            source_contract.path != CHECKER
            or source_contract.args
            or source_contract.workspace != DEFAULT_WORKSPACE
            or not math.isfinite(source_contract.timeout)
            or source_contract.timeout <= 0
        ):
            raise ValueError("Unsupported prompt-injection script contract")
        instructions = _instructions(archive.files["instruction.md"].decode())
        data = json.loads(archive.files[f"tests/{DATA}"])
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("target_tool"), str)
            or not data["target_tool"].strip()
            or not isinstance(data.get("target_args"), dict)
        ):
            raise ValueError("Invalid prompt-injection target data")
        checker = archive.files[f"tests/{CHECKER}"]
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid TaskTrove prompt-injection archive: {error}") from error

    verifier = ScriptVerifier(
        entrypoint="grade.py",
        args=("/tests", "/app", "/verifier", str(source_contract.timeout)),
        timeout_seconds=source_contract.timeout + RUNTIME_GRACE_SECONDS,
        runtime_image=runtime_image,
        resources=(
            embedded_resource("grade.py", _ADAPTER.read_bytes(), executable=True),
            embedded_resource("source_checker.py", checker),
            embedded_resource(DATA, archive.files[f"tests/{DATA}"]),
        ),
        network_policy=NetworkPolicy.DISABLED,
    )
    identity = json.dumps(
        (archive.source.dataset, archive.source.revision, archive.source.row),
        separators=(",", ":"),
    )
    return TaskSpec(
        id=f"tasktrove-{hashlib.sha256(identity.encode()).hexdigest()}",
        context=ConversationInput(events=(TextMessage(role="user", content=instructions),)),
        verifier=script_verifier(verifier),
        source=archive.source.model_copy(update={"importer_revision": IMPORTER_REVISION}),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
    )
