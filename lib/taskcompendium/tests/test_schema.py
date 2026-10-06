# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public schema loading and runtime boundaries preserve private contracts."""

import json

import pytest
from pydantic import ValidationError

from taskcompendium.environment import EnvironmentFile, EnvironmentKind, EnvironmentSpec, ShellVerifierSpec
from taskcompendium.grading import exact_answer, grade_answer
from taskcompendium.grading_contract import GradingAttempt
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import PlainText, chat_request


@pytest.fixture
def specification():
    return TaskSpec(
        id="schema-example",
        context=ConversationInput(events=(TextMessage(role="user", content="Repair the project."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=exact_answer("done"),
        source=Source(dataset="org/project", revision="pinned-revision", row="0", importer_revision="1"),
    )


@pytest.mark.parametrize(
    "update",
    [
        {"environment_requirements": EnvironmentRequirements(capabilities=("browser",))},
        {"environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM)},
        {"environment": EnvironmentSpec(kind=EnvironmentKind.NULL, interaction="company")},
        {"answer_type": AnswerType.FILE},
        {"answer_type": AnswerType.STATE},
        {"answer_type": AnswerType.WORKSPACE_STATE},
        {
            "verifier": VerifierSpec(
                kind="exact",
                parameters_json='{"expected":"done"}',
                environment_requirements=EnvironmentRequirements(capabilities=("process",)),
            ),
        },
    ],
)
def test_direct_chat_rejects_semantics_it_cannot_preserve_before_request(tmp_path, specification, update):
    record = TaskSpec.model_validate({**specification.model_dump(), **update})
    path = tmp_path / "specification.json"
    path.write_text(record.model_dump_json())
    task = TaskSpec.model_validate_json(path.read_text())
    with pytest.raises(NotImplementedError):
        chat_request(task, PlainText(id="plain"))


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_private_verifier_config_rejects_nested_nonfinite_json_numbers(tmp_path, specification, number):
    path = tmp_path / "specification.json"
    wire = specification.model_dump(mode="json")
    valid_parameters = ' {"checks": [{"tolerance": 0.125}], "label": "NaN"} '
    wire["verifier"] = {"kind": "future_grader", "parameters_json": valid_parameters}
    path.write_text(json.dumps(wire))
    assert TaskSpec.model_validate_json(path.read_text()).verifier.parameters_json == valid_parameters
    wire["verifier"]["parameters_json"] = '{"checks": [{"tolerance": ' + number + "}]} "
    path.write_text(json.dumps(wire))
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(path.read_text())


def test_private_verifier_config_rejects_duplicate_keys(specification):
    wire = specification.model_dump(mode="json")
    wire["verifier"] = {"kind": "exact", "parameters_json": '{"expected": ["done"], "expected": ["other"]}'}
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(wire)


def test_pure_grading_cannot_ignore_a_private_verifier_environment(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["verifier"]["environment_requirements"] = {"capabilities": ["process"]}
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    task = TaskSpec.model_validate_json(path.read_text())
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done")))
    # This correct answer must not earn credit without the required private runtime.
    with pytest.raises(NotImplementedError):
        grade_answer(task, PlainText(id="plain"), GradingAttempt(conversation))
    result = grade_task(task, PlainText(id="plain"), conversation)
    assert (result.status, result.reward) == ("invalid_task", None)


@pytest.mark.parametrize("kind", ["llm_judge", "private_script"])
def test_schema_only_verifiers_cannot_grade(tmp_path, specification, kind):
    specification = specification.model_copy(update={"verifier": VerifierSpec(kind=kind, parameters_json="{}")})
    path = tmp_path / "specification.json"
    path.write_text(specification.model_dump_json())
    task = TaskSpec.model_validate_json(path.read_text())
    convention = PlainText(id="plain")
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done")))
    with pytest.raises(NotImplementedError):
        grade_answer(task, convention, GradingAttempt(conversation))


@pytest.mark.parametrize("candidate,reward", [("done", 1.0), ("incorrect", 0.0)])
def test_pure_per_attempt_grading_accepts_answers_acquired_in_a_worker_workspace(specification, candidate, reward):
    task = specification.model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(capabilities=("shell", "filesystem")),
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                workdir="/app",
                files=(EnvironmentFile(path="/app/project.txt", content=b"worker input"),),
            ),
        }
    )
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=candidate)))
    result = grade_answer(task, PlainText(id="plain"), GradingAttempt(conversation))
    assert (result.status, result.reward) == ("graded", reward)


@pytest.mark.parametrize("payload", [b"UTF-8 text: \xe2\x98\x83\n", b"\x00\xff\x80\n"])
def test_reader_preserves_public_and_private_files_through_json_roundtrip(tmp_path, specification, payload):
    task = specification.model_copy(
        update={
            "environment": EnvironmentSpec(
                kind=EnvironmentKind.SHELLSIM,
                workdir="/app",
                env={"TASK_MODE": "repair"},
                files=(EnvironmentFile(path="/app/input.dat", content=payload, mode=0o500),),
            ),
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=ShellVerifierSpec(
                    argv=("/checks/grade",),
                    timeout=5,
                ).model_dump_json(),
                files=(EnvironmentFile(path="/checks/grade", content=b"private checks", mode=0o755),),
            ),
        }
    )
    path = tmp_path / "specification.json"
    path.write_text(task.model_dump_json())
    restored = TaskSpec.model_validate_json(path.read_text())
    assert restored == task
    with pytest.raises(NotImplementedError):
        chat_request(restored, PlainText(id="plain"))


@pytest.mark.parametrize(
    "legacy_field,value",
    [
        ("environment_requirements", {"working_directory": "/app"}),
        ("resources", {"worker": [{"path": "input.dat", "source": {"kind": "inline_file", "content_base64": "eA=="}}]}),
        ("schema_version", "0.24"),
    ],
)
def test_reader_rejects_obsolete_task_records(tmp_path, specification, legacy_field, value):
    wire = specification.model_dump(mode="json")
    wire[legacy_field] = value
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError):
        TaskSpec.model_validate_json(path.read_text())


def test_task_json_rejects_old_schema_and_verifier_shape(specification):
    payload = json.loads(specification.model_dump_json())
    payload["schema_version"] = "0.1"
    payload["verifier"] = {"expected": "done", "ignore_case": True, "ignore_whitespace": True}
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(json.dumps(payload))
