# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public schema loading and runtime boundaries preserve private contracts."""

import base64
import json

import pytest
from pydantic import ValidationError

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
    VerifierSpec,
)
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
        {"environment_requirements": EnvironmentRequirements(docker_image="org/image@sha256:" + "a" * 64)},
        {"environment_requirements": EnvironmentRequirements(working_directory="/app")},
        {"environment_requirements": EnvironmentRequirements(setup_commands=("initialize",))},
        {"environment_requirements": EnvironmentRequirements(environment_variables={"TASK_MODE": "repair"})},
        {
            "environment_requirements": EnvironmentRequirements.model_validate(
                {
                    "tool_providers": {
                        "company": {
                            "action_interface": "workplace:v1",
                            "initial_state": {"inbox": [], "company": "example"},
                        }
                    },
                }
            )
        },
    ]
    + [
        {"resources": {role: [{"path": "input.txt", "source": {"kind": "inline_file", "content_base64": "eA=="}}]}}
        for role in ("all", "worker", "oracle", "verifier")
    ]
    + [
        {
            "resources": {
                "worker": [
                    {
                        "path": "project/input.txt",
                        "source": {"kind": "inline_file", "content_base64": "cHVibGljIGlucHV0"},
                        "mode": "0755",
                    }
                ]
            }
        },
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
        {
            "resources": {
                "verifier": [
                    {
                        "path": "checks/grade.py",
                        "source": {"kind": "inline_file", "content_base64": "cHJpdmF0ZSBjaGVja3M="},
                        "mode": "0755",
                    }
                ]
            }
        },
    ],
)
def test_direct_chat_rejects_semantics_it_cannot_preserve_before_request(tmp_path, specification, update):
    record = TaskSpec.model_validate({**specification.model_dump(), **update})
    path = tmp_path / "specification.json"
    path.write_text(record.model_dump_json())
    task = TaskSpec.model_validate_json(path.read_text())
    convention = PlainText(id="plain")
    with pytest.raises(NotImplementedError):
        chat_request(task, convention)


@pytest.mark.parametrize("second_path", ["data", "DATA", "data/input.txt"])
@pytest.mark.parametrize("role", ["worker", "oracle", "verifier"])
def test_shared_resource_destinations_cannot_overwrite_role_mounts(specification, second_path, role):
    wire = specification.model_dump()
    wire["resources"] = {
        "all": [{"path": "data", "source": {"kind": "inline_file", "content_base64": "c2hhcmVk"}}],
        role: [{"path": second_path, "source": {"kind": "inline_file", "content_base64": "cHJpdmF0ZQ=="}}],
    }
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(wire)


def test_private_role_mounts_reuse_paths_without_becoming_worker_visible(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["resources"] = {
        role: [
            {
                "path": "fixture.txt",
                "source": {"kind": "inline_file", "content_base64": base64.b64encode(content.encode()).decode("ascii")},
            }
        ]
        for role, content in (("worker", "public"), ("oracle", "gold"), ("verifier", "hidden test"))
    }
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    resources = TaskSpec.model_validate_json(path.read_text()).model_dump(mode="json")["resources"]
    assert resources["all"] == []
    assert base64.b64decode(resources["worker"][0]["source"]["content_base64"]) == b"public"
    assert base64.b64decode(resources["oracle"][0]["source"]["content_base64"]) == b"gold"
    assert base64.b64decode(resources["verifier"][0]["source"]["content_base64"]) == b"hidden test"


@pytest.mark.parametrize("initial_state", [None, "company-snapshot", {"inbox": [], "counter": 3}])
def test_reader_keeps_literal_provider_state_but_direct_chat_cannot_request_it(tmp_path, specification, initial_state):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"]["tool_providers"] = {
        "company": {"action_interface": "workplace:v1", "initial_state": initial_state}
    }
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    task = TaskSpec.model_validate_json(path.read_text())
    assert task.environment_requirements.tool_providers["company"].initial_state == initial_state
    with pytest.raises(NotImplementedError):
        chat_request(task, PlainText(id="plain"))


def test_reader_rejects_nested_nonfinite_provider_state(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"]["tool_providers"] = {
        "company": {"action_interface": "workplace:v1", "initial_state": {"counters": [float("nan")]}}
    }
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(path.read_text())


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_private_verifier_config_rejects_nested_nonfinite_json_numbers(tmp_path, specification, number):
    # JsonValue previously allowed nonfinite values despite allow_inf_nan=False.
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


async def test_pure_grading_cannot_ignore_a_private_verifier_environment(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["verifier"]["environment_requirements"] = {"docker_image": "private/grader@sha256:" + "a" * 64}
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    task = TaskSpec.model_validate_json(path.read_text())
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done")))
    # This correct answer must not earn credit without the required private runtime.
    with pytest.raises(NotImplementedError):
        await grade_answer(task, PlainText(id="plain"), GradingAttempt(conversation, object()))


@pytest.mark.parametrize("kind", ["llm_judge", "private_script"])
async def test_schema_only_verifiers_cannot_grade(tmp_path, specification, kind):
    specification = specification.model_copy(update={"verifier": VerifierSpec(kind=kind, parameters_json="{}")})
    path = tmp_path / "specification.json"
    path.write_text(specification.model_dump_json())
    task = TaskSpec.model_validate_json(path.read_text())
    convention = PlainText(id="plain")
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done")))
    with pytest.raises(NotImplementedError):
        await grade_answer(task, convention, GradingAttempt(conversation, object()))


@pytest.mark.parametrize(
    "base_path,alias", [("foo", "foo."), ("foo", "foo "), ("inputs/answer", "inputs/answer:backup")]
)
def test_resource_groups_reject_portable_path_aliases_before_mounts_can_overwrite_inputs(
    specification, base_path, alias
):
    wire = specification.model_dump(mode="json")
    wire["resources"] = {
        "all": [{"path": base_path, "source": {"kind": "inline_file", "content_base64": "cHVibGlj"}}],
        "worker": [{"path": alias, "source": {"kind": "inline_file", "content_base64": "b3ZlcndyaXRl"}}],
    }
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(wire)


@pytest.mark.parametrize("candidate,reward", [("done", 1.0), ("incorrect", 0.0)])
async def test_pure_per_attempt_grading_accepts_answers_acquired_in_a_worker_workspace(specification, candidate, reward):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"] = {"capabilities": ["shell", "filesystem"], "working_directory": "/app"}
    wire["resources"] = {
        "worker": [{"path": "project.txt", "source": {"kind": "inline_file", "content_base64": "d29ya2VyIGlucHV0"}}]
    }
    task = TaskSpec.model_validate(wire)
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=candidate)))
    result = await grade_answer(task, PlainText(id="plain"), GradingAttempt(conversation, object()))
    assert (result.status, result.reward) == ("graded", reward)


def test_reader_preserves_private_schema_contracts_through_json_roundtrip(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"] = {"environment_variables": {"TASK_MODE": "repair"}}
    wire["verifier"] = {
        "kind": "private_script",
        "parameters_json": '{"entrypoint":"checks/grade.py"}',
        "environment_requirements": {"environment_variables": {"CHECK_MODE": "strict"}},
    }
    wire["resources"] = {
        "worker": [
            {"path": "project/input.txt", "source": {"kind": "inline_file", "content_base64": "cHVibGljIGlucHV0"}}
        ],
        "verifier": [
            {"path": "checks/grade.py", "source": {"kind": "inline_file", "content_base64": "cHJpdmF0ZSBjaGVja3M="}}
        ],
    }
    task = TaskSpec.model_validate(wire)
    path = tmp_path / "specification.json"
    path.write_text(task.model_dump_json())
    restored = TaskSpec.model_validate_json(path.read_text())
    assert restored == task
    with pytest.raises(NotImplementedError):
        chat_request(restored, PlainText(id="plain"))


@pytest.mark.parametrize("payload", [b"UTF-8 text: \xe2\x98\x83\n", b"\x00\xff\x80\n"])
def test_inline_file_bytes_and_metadata_survive_json_reader(tmp_path, specification, payload):
    wire = specification.model_dump(mode="json")
    wire["resources"] = {
        "worker": [
            {
                "path": "input.dat",
                "source": {"kind": "inline_file", "content_base64": base64.b64encode(payload).decode("ascii")},
                "mode": "0500",
                "mtime_ns": 1_725_555_600_123_456_789,
            }
        ]
    }
    task = TaskSpec.model_validate(wire)
    path = tmp_path / "specification.json"
    path.write_text(task.model_dump_json())
    restored = TaskSpec.model_validate_json(path.read_text()).resources.worker[0]
    assert base64.b64decode(restored.source.content_base64) == payload
    assert restored.mode == "0500"
    assert restored.mtime_ns == 1_725_555_600_123_456_789


def test_task_json_rejects_old_schema_and_verifier_shape(specification):
    payload = json.loads(specification.model_dump_json())
    payload["schema_version"] = "0.1"
    payload["verifier"] = {"expected": "done", "ignore_case": True, "ignore_whitespace": True}
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(json.dumps(payload))
