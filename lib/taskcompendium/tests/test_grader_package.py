# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader packages and their observable verdicts."""

import json
from dataclasses import replace

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineSpec, RegistryImage, ShellSimBuiltins
from verifyit.spec import JsonSchemaSpec, NumericSpec, ScriptSpec

from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.grading import parse_grade_result
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    FileReward,
    FinalAction,
    FunctionDefinition,
    GradingAttempt,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    StateSubmission,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import grade_task

IMAGE = "fixture@sha256:" + "0" * 64
GRADING_ENVIRONMENT = EnvironmentRequirements(docker_image=IMAGE, compatible_backends=(Backend.DOCKER,))
GRADING_MACHINE = MachineSpec(RegistryImage(IMAGE))


class FixtureImageFactory:
    """Run the pinned fixture image's commands on the ShellSim built-ins."""

    backend = Backend.DOCKER

    async def create(self, spec):
        return await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins()))


def _task(package, answer_type=AnswerType.TEXT, answer_format=PlainText(), final_tools=()):
    task = TaskSpec(
        id="private-grader",
        context=ConversationInput(events=(TextMessage(role="user", content="Submit an answer"),)),
        environment_requirements=EnvironmentRequirements(),
        final_tools=final_tools,
        answer_type=answer_type,
        answer_format=answer_format,
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
        source=Source(dataset="test", revision="1", row="1", importer_revision="1"),
    )
    return TaskSpec.model_validate_json(task.model_dump_json())


def _attempt(task, final, state=None):
    return GradingAttempt(ConversationTrace(events=(*task.context.events, final)), state=state)


def test_json_schema_grader_reads_private_schema_and_scores_document():
    package = verifyit_package(
        JsonSchemaSpec(schema="schema.json"),
        (inline_resource("schema.json", b'{"type":"object","required":["value"]}'),),
    )
    task = _task(package)

    good = grade_task(task, _attempt(task, TextMessage(role="assistant", content='{"value": 1}')))
    bad = grade_task(task, _attempt(task, TextMessage(role="assistant", content="{}")))

    assert (good.status, good.reward) == (Outcome.GRADED, 1.0)
    assert (bad.status, bad.reward) == (Outcome.GRADED, 0.0)


STATE_GRADER = b"""import json
import os

config = '/tests/config.json'
assert os.stat(config).st_mode & 0o777 == 0o600
expected = json.load(open(config))['expected']
actual = json.load(open('/app/state.json'))['value']
with open('/logs/verifier/reward.json', 'w') as reward:
    json.dump({'reward': float(actual == expected), 'detail': {'actual': actual}}, reward)
"""


@pytest.mark.parametrize("value,reward,passed", [(3, 1.0, True), (4, 0.0, False)])
def test_script_grader_reads_private_configuration_and_captured_state(value, reward, passed):
    grader = ScriptGrader(
        argv=("python3", "/tests/grade.py"),
        environment=GRADING_ENVIRONMENT,
        answer_path=None,
        reward=FileReward(
            files=(RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),), pass_above=0.5
        ),
    )
    config = inline_resource("config.json", b'{"expected": 3}').model_copy(update={"mode": "0600"})
    task = _task(GraderPackage(grader, (inline_resource("grade.py", STATE_GRADER), config)), AnswerType.STATE)
    result = grade_task(
        task,
        _attempt(task, TextMessage(role="assistant", content="Done"), StateSubmission({"value": value})),
        machine_factory=FixtureImageFactory(),
        machine_spec=GRADING_MACHINE,
    )
    assert (result.status, result.reward, result.passed, result.detail) == (
        Outcome.GRADED,
        reward,
        passed,
        {"actual": value},
    )


@pytest.mark.parametrize(
    "spec,verdict,expected",
    [
        (
            ScriptSpec(path="grade.py"),
            {"status": "scored", "reward": 1, "detail": {"reason": "invalid_numeric_candidate"}},
            (Outcome.GRADED, 1.0, None, None),
        ),
        (
            NumericSpec("12", tolerance_abs=0.0, tolerance_rel=0.0),
            {"status": "scored", "reward": 0, "detail": {"reason": "invalid_numeric_candidate", "error": "no number"}},
            (Outcome.SUBMISSION_FAILURE, 0.0, "no number", None),
        ),
        (
            ScriptSpec(path="grade.py"),
            {"status": "invalid_task", "reward": 0, "detail": {"error": "bad reference"}},
            (Outcome.INVALID_TASK, None, "bad reference", None),
        ),
        (
            ScriptSpec(path="grade.py"),
            {"status": "infra_error", "reward": 0, "detail": {"error": "runner failed"}},
            (Outcome.INFRA_ERROR, None, "runner failed", None),
        ),
        (
            ScriptSpec(path="grade.py"),
            {"status": "scored", "reward": 2, "detail": {}},
            (Outcome.INFRA_ERROR, None, "Invalid verifier verdict", GradingFailure.INVALID_REWARD),
        ),
        (
            ScriptSpec(path="grade.py"),
            {"status": "invalid_task", "reward": 1, "detail": {}},
            (Outcome.INFRA_ERROR, None, "Invalid verifier verdict", GradingFailure.INVALID_REWARD),
        ),
    ],
)
def test_verifyit_verdict_file_maps_to_grade_outcome(spec, verdict, expected):
    result = parse_grade_result(spec, json.dumps(verdict).encode())
    assert (result.status, result.reward, result.error, result.failure) == expected


NATIVE_ACTION_GRADER = b"""import json

expected = {
    'type': 'assistant_tool_calls',
    'calls': [{'call_id': 'call-1', 'name': 'lookup', 'arguments': {'index': 3}}],
    'content': 'A native action',
}
print(float(json.load(open('/app/answer.txt')) == expected))
"""


def _lookup(index, content="A native action"):
    return AssistantToolCalls(
        calls=(ConversationToolCall(call_id="call-1", name="lookup", arguments={"index": index}),), content=content
    )


@pytest.mark.parametrize(
    "answer_format,final,expected",
    [
        (FinalAction(), _lookup(3), (Outcome.GRADED, 1.0)),
        (FinalAction(), _lookup("3"), (Outcome.GRADED, 0.0)),
        (FinalAction(), _lookup(3, content=None), (Outcome.GRADED, 0.0)),
        (FinalAction(), TextMessage(role="assistant", content="lookup(index=3)"), (Outcome.GRADED, 0.0)),
        (
            FinalAction(require_call=True),
            TextMessage(role="assistant", content="lookup(index=3)"),
            (Outcome.SUBMISSION_FAILURE, 0.0),
        ),
    ],
)
def test_native_action_script_grader_receives_the_final_message(answer_format, final, expected):
    grader = ScriptGrader(argv=("python3", "/tests/grade.py"), environment=GRADING_ENVIRONMENT)
    task = _task(
        GraderPackage(grader, (inline_resource("grade.py", NATIVE_ACTION_GRADER),)),
        AnswerType.NATIVE_ACTION,
        answer_format,
        final_tools=(FunctionDefinition(name="lookup", parameters={"type": "object"}),),
    )
    result = grade_task(task, _attempt(task, final), machine_factory=FixtureImageFactory(), machine_spec=GRADING_MACHINE)
    assert (result.status, result.reward) == expected
