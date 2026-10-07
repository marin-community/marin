# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private grader packages and their observable verdicts."""

import json

import pytest
from shellbox.machine import Backend, DockerImage, MachineSpec
from verifyit.spec import JsonSchemaSpec, ScriptSpec

from taskcompendium.grader import GraderPackage, grader_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionDefinition,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import FinalAction, PlainText

from .test_executable_ingestion import GradingMachines

PLAIN = PlainText(id="plain")


def _task(package, answer_type=AnswerType.TEXT):
    return TaskSpec(
        id="private-grader",
        context=ConversationInput(events=(TextMessage(role="user", content="Submit an answer"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=answer_type,
        verifier=package.verifier,
        resources=ResourceGroups(verifier=package.resources),
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
    'detail': {'error': 'runner failed' if actual == -2 else 'bad reference'} if actual < 0
        else {'reason': 'invalid_numeric_candidate'},
}
(logs / 'verdict.json').write_text(json.dumps(verdict))
"""
    package = grader_package(
        ScriptSpec(path="grader.py", verdict_file="verdict.json", timeout=60),
        (inline_resource("grader.py", script), inline_resource("config.json", b'{"expected": 3}')),
    )
    config_resource = package.resources[1].model_copy(update={"mode": "0600", "mtime_ns": 1_234_567_890})
    package = GraderPackage(package.verifier, (package.resources[0], config_resource))
    original = _task(package, AnswerType.STATE)
    task = TaskSpec.model_validate_json(original.model_dump_json())

    good = grade_task(task, PLAIN, _conversation(task, "Done"), RuntimeEvidence({}, '{"value":3}'))
    bad = grade_task(task, PLAIN, _conversation(task, "Done"), RuntimeEvidence({}, '{"value":4}'))
    invalid = grade_task(task, PLAIN, _conversation(task, "Done"), RuntimeEvidence({}, '{"value":-1}'))
    infrastructure = grade_task(task, PLAIN, _conversation(task, "Done"), RuntimeEvidence({}, '{"value":-2}'))

    assert (good.status, good.reward) == (Outcome.GRADED, 1.0)
    assert (bad.status, bad.reward) == (Outcome.GRADED, 0.0)
    assert good.detail["reason"] == bad.detail["reason"] == "invalid_numeric_candidate"
    assert (invalid.status, invalid.reward, invalid.error) == (Outcome.INVALID_TASK, None, "bad reference")
    assert (infrastructure.status, infrastructure.reward, infrastructure.error) == (
        Outcome.INFRA_ERROR,
        None,
        "runner failed",
    )


@pytest.fixture
def native_script_task():
    script = b"""import json
import os
from pathlib import Path

workspace = Path(os.environ['VERIFYIT_WORKSPACE'])
logs = Path(os.environ['VERIFYIT_LOGS_DIR'])
event = json.loads((workspace / 'answer.txt').read_text())
reward = float(event['type'] == 'assistant_tool_calls'
    and len(event['calls']) == 1
    and event['calls'][0]['name'] == 'lookup'
    and event['calls'][0]['arguments'] == {'index': 3})
(logs / 'verdict.json').write_text(json.dumps({'status': 'scored', 'reward': reward, 'detail': {}}))
"""
    task = _task(
        grader_package(
            ScriptSpec(path="grader.py", verdict_file="verdict.json", timeout=60),
            (inline_resource("grader.py", script),),
        )
    ).model_copy(
        update={
            "answer_type": AnswerType.NATIVE_ACTION,
            "final_tools": (FunctionDefinition(name="lookup", parameters={"type": "object"}),),
        }
    )
    return TaskSpec.model_validate_json(task.model_dump_json())


@pytest.mark.parametrize(
    "event, expected_reward",
    [
        (
            AssistantToolCalls(calls=(ConversationToolCall(call_id="call-1", name="lookup", arguments={"index": 3}),)),
            1.0,
        ),
        (
            AssistantToolCalls(calls=(ConversationToolCall(call_id="call-1", name="lookup", arguments={"index": "3"}),)),
            0.0,
        ),
        (TextMessage(role="assistant", content="lookup(index=3)"), 0.0),
    ],
)
def test_native_script_grader_receives_terminal_call_and_argument_types(native_script_task, event, expected_reward):
    trace = ConversationTrace(events=(*native_script_task.context.events, event))
    result = grade_task(native_script_task, FinalAction(id="native"), trace)
    assert (result.status, result.reward) == (Outcome.GRADED, expected_reward)


def test_isolated_native_script_receives_terminal_event_in_submission_archive(native_script_task):
    image = "test@sha256:" + "a" * 64
    task = native_script_task.model_copy(
        update={
            "verifier": native_script_task.verifier.model_copy(
                update={
                    "environment_requirements": EnvironmentRequirements(
                        docker_image=image, compatible_backends=(Backend.DOCKER,)
                    )
                }
            )
        }
    )
    event = AssistantToolCalls(
        calls=(ConversationToolCall(call_id="call-1", name="lookup", arguments={"index": 3}),),
        content="A native action",
    )
    machines = GradingMachines()
    result = grade_task(
        task,
        FinalAction(id="native"),
        ConversationTrace(events=(*task.context.events, event)),
        RuntimeEvidence({}, "{}"),
        machine_factory=machines,
        machine_spec=MachineSpec(DockerImage(image)),
    )
    assert result.status == Outcome.GRADED
    assert json.loads(machines.machines[0].files["/app/answer.txt"]) == {
        "type": "assistant_tool_calls",
        "calls": [{"call_id": "call-1", "name": "lookup", "arguments": {"index": 3}}],
        "content": "A native action",
    }
    assert machines.machines[0].closed
