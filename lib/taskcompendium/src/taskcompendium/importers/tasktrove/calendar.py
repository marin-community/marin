# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the TaskTrove Clean calendar scheduling scripts."""

import hashlib
import json
import math
import tomllib
from pathlib import Path
from types import MappingProxyType

from tasktrove_verify.spec import DEFAULT_WORKSPACE, ScriptSpec, parse_spec

from taskcompendium.importers.tasktrove.convert import METADATA_TABLE, TASK_MANIFEST
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TaskSpec, TextMessage
from taskcompendium.verifiers.script import NetworkPolicy, ScriptVerifier, embedded_resource, script_verifier

CHECKER = "agent_calendar_checker.py"
DATA = "expected_events.json"
IMPORTER_REVISION = "taskcompendium-tasktrove-calendar-v0.1"
RUNTIME_GRACE_SECONDS = 10.0
SOURCES = MappingProxyType(
    {
        "laion__nemotron-gym-agent-calendar-v2": ("tool-use", "agent_calendar"),
        "laion__nemotron-gym-instruction-following-calendar-v3": ("instruction-following", "nemotron_if_structured"),
    }
)
_OUTPUT_INSTRUCTION = (
    "You are scheduling events on a calendar. Read the conversation below and "
    "write your final calendar as a JSON list to `/app/answer.txt`. Each event "
    "must include `event_id` (int), `event_name` (str), `start_time` "
    '("HH:MM"), and `duration` (minutes). Events must not overlap. The verifier '
    "checks the exact event set, duration, time window, declared constraints, "
    "and pairwise overlap.\n\n---\n\n"
)
_TASK_INSTRUCTION = (
    "You are scheduling events on a calendar. Read the conversation below and "
    "provide the calendar as a JSON list. Each event must include `event_id` "
    '(int), `event_name` (str), `start_time` ("HH:MM"), and `duration` (minutes). '
    "Include exactly the requested events with their specified durations. "
    "Satisfy every event's time window and declared constraints. Events must not overlap.\n\n---\n\n"
)
_ADAPTER = Path(__file__).with_name("calendar_adapter.py")


def _valid_expected_events(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("expected_events"), dict):
        return False
    events = value["expected_events"]
    if not events:
        return False
    for key, event in events.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key or not isinstance(event, dict):
            return False
        if (
            not isinstance(event.get("event_name"), str)
            or not event["event_name"].strip()
            or isinstance(event.get("duration"), bool)
            or not isinstance(event.get("duration"), int)
            or event["duration"] <= 0
            or not isinstance(event.get("min_time"), str)
            or not isinstance(event.get("max_time"), str)
            or (event.get("constraint") is not None and not isinstance(event.get("constraint"), str))
        ):
            return False
    return True


def import_task(archive: TaskArchive, *, runtime_image: str) -> TaskSpec:
    """Translate one pinned Clean calendar task into a private script verifier."""
    try:
        metadata = tomllib.loads(archive.files[TASK_MANIFEST].decode())[METADATA_TABLE]
        source = archive.upstream_subset
        family, converter = SOURCES[source]
        tags = metadata.get("tags")
        if (
            metadata.get("family") != family
            or metadata.get("converter") != converter
            or metadata.get("mode") != "script"
            or not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag for tag in tags)
        ):
            raise ValueError("Unsupported TaskTrove calendar source metadata")
        source_contract = parse_spec(archive.files["tests/verifier.toml"].decode())
        if (
            not isinstance(source_contract, ScriptSpec)
            or source_contract.path != CHECKER
            or source_contract.args
            or source_contract.workspace != DEFAULT_WORKSPACE
            or not math.isfinite(source_contract.timeout)
            or source_contract.timeout <= 0
        ):
            raise ValueError("Unsupported TaskTrove calendar script contract")
        instructions = archive.files["instruction.md"].decode().strip()
        if not instructions.startswith(_OUTPUT_INSTRUCTION):
            raise ValueError("Unsupported TaskTrove calendar instruction shape")
        # The recognized harness header also contains task requirements; retain
        # those while leaving answer delivery to the submission convention.
        instructions = _TASK_INSTRUCTION + instructions[len(_OUTPUT_INSTRUCTION) :]
        checker = archive.files[f"tests/{CHECKER}"]
        expected_events = archive.files[f"tests/{DATA}"]
        if not instructions or not checker or not _valid_expected_events(json.loads(expected_events)):
            raise ValueError("Invalid TaskTrove calendar task data")
    except (
        KeyError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise ValueError(f"Invalid TaskTrove calendar archive: {error}") from error

    verifier = ScriptVerifier(
        entrypoint="grade.py",
        args=("/tests", "/app", "/verifier", str(source_contract.timeout)),
        timeout_seconds=source_contract.timeout + RUNTIME_GRACE_SECONDS,
        runtime_image=runtime_image,
        resources=(
            embedded_resource("grade.py", _ADAPTER.read_bytes(), executable=True),
            embedded_resource("source_checker.py", checker),
            embedded_resource(DATA, expected_events),
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
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=script_verifier(verifier),
        source=archive.source.model_copy(update={"importer_revision": IMPORTER_REVISION}),
        tags=tuple(tags),
    )
