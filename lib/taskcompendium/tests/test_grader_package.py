# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private grader packages and their observable verdicts."""

from verifyit.spec import JsonSchemaSpec

from taskcompendium.environment import EnvironmentFile, EnvironmentKind, EnvironmentSpec
from taskcompendium.grader import grader_package, script_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import PlainText

PLAIN = PlainText(id="plain")


def _task(package, answer_type=AnswerType.TEXT):
    return TaskSpec(
        id="private-grader",
        context=ConversationInput(events=(TextMessage(role="user", content="Submit an answer"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=answer_type,
        verifier=package,
        source=Source(dataset="test", revision="1", row="1", importer_revision="1"),
    )


def _conversation(task, answer):
    return ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))


def test_json_schema_grader_reads_private_schema_and_scores_document():
    package = grader_package(
        JsonSchemaSpec(schema="schema.json"),
        (inline_resource("schema.json", b'{"type":"object","required":["value"]}'),),
    )
    task = _task(package)
    task = TaskSpec.model_validate_json(task.model_dump_json())

    good = grade_task(task, PLAIN, _conversation(task, '{"value": 1}'))
    bad = grade_task(task, PLAIN, _conversation(task, "{}"))

    assert (good.status, good.reward) == (Outcome.GRADED, 1.0)
    assert (bad.status, bad.reward) == (Outcome.GRADED, 0.0)


def test_public_files_cannot_replace_the_private_schema():
    package = grader_package(
        JsonSchemaSpec(schema="schema.json"),
        (inline_resource("schema.json", b'{"type":"object","required":["value"]}'),),
    )
    task = _task(package).model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                files=(EnvironmentFile(path="/tests/schema.json", content=b"{}"),),
            )
        }
    )
    assert grade_task(task, PLAIN, _conversation(task, "{}")).reward == 0.0
    assert grade_task(task, PLAIN, _conversation(task, '{"value":1}')).reward == 1.0


def test_script_grader_reads_private_configuration_and_captured_state():
    script = b"""import json
import os
from pathlib import Path

tests = Path(os.environ['VERIFYIT_TESTS_DIR'])
workspace = Path(os.environ['VERIFYIT_WORKSPACE'])
logs = Path(os.environ['VERIFYIT_LOGS_DIR'])
expected = json.loads((tests / 'config.json').read_text())['expected']
fixture = (tests / 'config.json').stat()
assert fixture.st_mode & 0o777 == 0o600
assert fixture.st_mtime_ns == 1234567890
actual = json.loads((workspace / 'state.json').read_text())['value']
reward = float(actual == expected)
status = 'infra_error' if actual == -2 else 'invalid_task' if actual < 0 else 'scored'
verdict = {
    'status': status,
    'reward': 0 if actual < 0 else reward,
    'detail': {'error': 'runner failed' if actual == -2 else 'bad reference'} if actual < 0 else {},
}
(logs / 'verdict.json').write_text(json.dumps(verdict))
"""
    package = script_package(script, {"expected": 3})
    config_resource = package.files[1].model_copy(update={"mode": 0o600, "mtime_ns": 1_234_567_890})
    package = package.model_copy(update={"files": (package.files[0], config_resource)})
    original = _task(package, AnswerType.STATE)
    task = TaskSpec.model_validate_json(original.model_dump_json())

    good = grade_task(task, PLAIN, _conversation(task, "Done"), RuntimeEvidence({}, '{"value":3}'))
    bad = grade_task(task, PLAIN, _conversation(task, "Done"), RuntimeEvidence({}, '{"value":4}'))
    invalid = grade_task(task, PLAIN, _conversation(task, "Done"), RuntimeEvidence({}, '{"value":-1}'))
    infrastructure = grade_task(task, PLAIN, _conversation(task, "Done"), RuntimeEvidence({}, '{"value":-2}'))

    assert (good.status, good.reward) == (Outcome.GRADED, 1.0)
    assert (bad.status, bad.reward) == (Outcome.GRADED, 0.0)
    assert (invalid.status, invalid.reward, invalid.error) == (Outcome.INVALID_TASK, None, "bad reference")
    assert (infrastructure.status, infrastructure.reward, infrastructure.error) == (
        Outcome.INFRA_ERROR,
        None,
        "runner failed",
    )
