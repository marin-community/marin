# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public schema loading and runtime boundaries preserve private contracts."""

import json

import pytest
from pydantic import ValidationError

from taskcompendium.environment import EnvironmentFile, EnvironmentKind, EnvironmentSpec, ShellVerifierSpec
from taskcompendium.grading import exact_answer, grade_answer
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.lowering import HarborEnvironmentConfig, compatible_lowerings, lower_to_harbor, read_specification
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
from taskcompendium.submission import AnswerFormat, SubmissionConvention, chat_request


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
def test_direct_chat_rejects_semantics_it_cannot_preserve_before_export(tmp_path, specification, update):
    record = TaskSpec.model_validate({**specification.model_dump(), **update})
    path = tmp_path / "specification.json"
    path.write_text(record.model_dump_json())
    task = read_specification(path)
    convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)
    assert compatible_lowerings(task, (convention,), (HarborEnvironmentConfig(),)) == ()
    with pytest.raises(NotImplementedError):
        chat_request(task, convention)
    destination = tmp_path / "task"
    with pytest.raises(NotImplementedError):
        lower_to_harbor(task, convention, HarborEnvironmentConfig(), destination)
    assert not destination.exists()


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_private_verifier_config_rejects_nested_nonfinite_json_numbers(tmp_path, specification, number):
    path = tmp_path / "specification.json"
    wire = specification.model_dump(mode="json")
    valid_parameters = ' {"checks": [{"tolerance": 0.125}], "label": "NaN"} '
    wire["verifier"] = {"kind": "future_grader", "parameters_json": valid_parameters}
    path.write_text(json.dumps(wire))
    assert read_specification(path).verifier.parameters_json == valid_parameters
    wire["verifier"]["parameters_json"] = '{"checks": [{"tolerance": ' + number + "}]} "
    path.write_text(json.dumps(wire))
    with pytest.raises(ValidationError):
        read_specification(path)


def test_pure_grading_cannot_ignore_a_private_verifier_environment(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["verifier"]["environment_requirements"] = {"capabilities": ["process"]}
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    task = read_specification(path)
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done")))
    # This correct answer must not earn credit without the required private runtime.
    result = grade_answer(task, SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN), conversation)
    assert (result.status, result.reward) == ("invalid_task", None)


@pytest.mark.parametrize("kind", ["llm_judge", "structured_exact"])
def test_schema_only_verifiers_cannot_export_or_grade(tmp_path, specification, kind):
    specification = specification.model_copy(update={"verifier": VerifierSpec(kind=kind, parameters_json="{}")})
    path = tmp_path / "specification.json"
    path.write_text(specification.model_dump_json())
    task = read_specification(path)
    convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)
    assert compatible_lowerings(task, (convention,), (HarborEnvironmentConfig(),)) == ()
    with pytest.raises(NotImplementedError):
        lower_to_harbor(task, convention, HarborEnvironmentConfig(), tmp_path / "export")
    assert not (tmp_path / "export").exists()
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done")))
    with pytest.raises(ValueError):
        grade_answer(task, convention, conversation)


@pytest.mark.parametrize("kind", ["llm_judge", "structured_exact"])
async def test_launch_rejects_schema_only_verifier_before_starting_a_trial(tmp_path, specification, kind):
    convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    unsupported = specification.model_copy(update={"verifier": VerifierSpec(kind=kind, parameters_json="{}")})
    (task / "specification.json").write_text(unsupported.model_dump_json())
    with pytest.raises(NotImplementedError):
        await run_trial(
            task,
            HarborEnvironmentConfig(),
            ChatLaunch(model="unused", api_base="https://example.invalid"),
            tmp_path / "trials",
            "unsupported",
        )
    assert not (tmp_path / "trials").exists()


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
    result = grade_answer(task, SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN), conversation)
    assert (result.status, result.reward) == ("graded", reward)


@pytest.mark.parametrize("payload", [b"UTF-8 text: \xe2\x98\x83\n", b"\x00\xff\x80\n"])
def test_reader_preserves_public_and_private_files_before_unsupported_export_is_rejected(
    tmp_path, specification, payload
):
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
                    files=(EnvironmentFile(path="/checks/grade", content=b"private checks", mode=0o755),),
                    timeout=5,
                ).model_dump_json(),
            ),
        }
    )
    path = tmp_path / "specification.json"
    path.write_text(task.model_dump_json())
    restored = read_specification(path)
    assert restored == task
    with pytest.raises(NotImplementedError):
        lower_to_harbor(
            restored,
            SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
            HarborEnvironmentConfig(),
            tmp_path / "export",
        )
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize(
    "legacy_field,value",
    [
        ("environment_requirements", {"working_directory": "/app"}),
        ("resources", {"worker": [{"path": "input.dat", "source": {"kind": "inline_file", "content_base64": "eA=="}}]}),
        ("schema_version", "0.22"),
    ],
)
def test_reader_rejects_obsolete_task_records(tmp_path, specification, legacy_field, value):
    wire = specification.model_dump(mode="json")
    wire[legacy_field] = value
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError):
        read_specification(path)
