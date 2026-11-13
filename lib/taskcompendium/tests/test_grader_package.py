# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader packages and their observable verdicts."""

import json
from dataclasses import replace

import pytest
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, MachineSpec, PackageSpec, RegistryImage
from verifyit.spec import JsonSchemaSpec, NumericSpec, ScriptSpec

from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.grading import parse_grade_result
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    CommandSemantics,
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
GRADING_ENVIRONMENT = EnvironmentRequirements(docker_image=IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS)
GRADING_MACHINE = MachineSpec(RegistryImage(IMAGE))


class FixtureImageFactory(DockerMachineFactory):
    """Run the pinned fixture image's commands on the ShellSim built-ins."""

    backend = Backend.DOCKER

    async def create(self, spec):
        return await ShellSimMachineFactory().create(replace(spec, source=PackageSpec()))


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


STATE_GRADER = b"""expected=$(jq .expected /tests/config.json)
jq --argjson expected "$expected" '{
    reward: (if .value == $expected then 1 else 0 end), detail: {actual: .value}
}' /app/state.json > /logs/verifier/reward.json
"""


def _state_task(environment=GRADING_ENVIRONMENT):
    """A state-graded task whose script grader reads a private config and the captured state."""
    grader = ScriptGrader(
        argv=("sh", "/tests/grade.sh"),
        environment=environment,
        answer_path=None,
        reward=FileReward(
            files=(RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),), pass_above=0.5
        ),
    )
    config = inline_resource("config.json", b'{"expected": 3}').model_copy(update={"mode": "0600"})
    return _task(GraderPackage(grader, (inline_resource("grade.sh", STATE_GRADER), config)), AnswerType.STATE)


class RecordingImageFactory(FixtureImageFactory):
    """The fixture factory, keeping every spec it was asked to create."""

    def __init__(self):
        self.specs: list[MachineSpec] = []

    async def create(self, spec):
        self.specs.append(spec)
        return await super().create(spec)


@pytest.mark.parametrize(
    "environment,workdir",
    [
        (GRADING_ENVIRONMENT, "/app"),
        (GRADING_ENVIRONMENT.model_copy(update={"working_directory": "/grading"}), "/grading"),
    ],
    ids=["grader-workspace", "named-by-the-environment"],
)
def test_grading_machine_starts_in_the_grader_workspace(environment, workdir):
    """The grading machine starts in the grader's workspace unless its environment names a directory."""
    task = _state_task(environment)
    factory = RecordingImageFactory()
    result = grade_task(
        task,
        _attempt(task, TextMessage(role="assistant", content="Done"), StateSubmission({"value": 3})),
        machine_factory=factory,
        machine_spec=replace(GRADING_MACHINE, workdir=""),
    )
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    assert [spec.workdir for spec in factory.specs] == [workdir]


@pytest.mark.parametrize("value,reward,passed", [(3, 1.0, True), (4, 0.0, False)])
def test_script_grader_reads_private_configuration_and_captured_state(value, reward, passed):
    task = _state_task()
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


NATIVE_ACTION_GRADER = b"""jq 'if . == {
    type: "assistant_tool_calls",
    calls: [{call_id: "call-1", name: "lookup", arguments: {index: 3}}],
    content: "A native action"
} then 1 else 0 end' /app/answer.txt
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
    grader = ScriptGrader(argv=("sh", "/tests/grade.sh"), environment=GRADING_ENVIRONMENT)
    task = _task(
        GraderPackage(grader, (inline_resource("grade.sh", NATIVE_ACTION_GRADER),)),
        AnswerType.NATIVE_ACTION,
        answer_format,
        final_tools=(FunctionDefinition(name="lookup", parameters={"type": "object"}),),
    )
    result = grade_task(task, _attempt(task, final), machine_factory=FixtureImageFactory(), machine_spec=GRADING_MACHINE)
    assert (result.status, result.reward) == expected
