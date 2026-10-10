# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public schema loading and runtime boundaries preserve private contracts."""

import base64
import json

import pytest
from pydantic import ValidationError
from verifyit.spec import ExactSpec

from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_answer
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    ConversationTrace,
    DockerBuildContext,
    EnvironmentRequirements,
    GradingAttempt,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.submission import chat_request

IMAGE = "private/grader@sha256:" + "a" * 64
GRADING_ENVIRONMENT = EnvironmentRequirements(
    docker_image=IMAGE,
    command_semantics=CommandSemantics.LINUX_PROCESS,
    setup_commands=("pip install --no-index /tests/wheels/*.whl",),
    environment_variables={"CHECK_MODE": "strict"},
)
EXACT_DONE = verifyit_package(ExactSpec(expected=("done",))).grader
EXACT_DONE_IN_ENVIRONMENT = verifyit_package(ExactSpec(expected=("done",)), environment=GRADING_ENVIRONMENT).grader


@pytest.fixture
def specification():
    return TaskSpec(
        id="schema-example",
        context=ConversationInput(events=(TextMessage(role="user", content="Repair the project."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=EXACT_DONE,
        source=Source(dataset="org/project", revision="pinned-revision", row="0", importer_revision="1"),
    )


@pytest.mark.parametrize(
    "update",
    [
        {"environment_requirements": EnvironmentRequirements(capabilities=("browser",))},
        {"environment_requirements": EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR)},
        {
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="org/image@sha256:" + "a" * 64
            )
        },
        {
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM mutable:latest\n"),)),
            )
        },
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
        for role in ("all", "worker", "oracle")
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
        {"answer_type": AnswerType.FILE, "output_paths": ("/app/answer.txt",), "grader": EXACT_DONE_IN_ENVIRONMENT},
        {"answer_type": AnswerType.STATE},
        {"answer_type": AnswerType.WORKSPACE_STATE, "grader": EXACT_DONE_IN_ENVIRONMENT},
        {"grader": EXACT_DONE_IN_ENVIRONMENT},
    ],
)
def test_direct_chat_rejects_semantics_it_cannot_preserve_before_request(specification, update):
    record = TaskSpec.model_validate({**specification.model_dump(), **update})
    task = TaskSpec.model_validate_json(record.model_dump_json())
    with pytest.raises(NotImplementedError):
        chat_request(task)


@pytest.mark.parametrize("second_path", ["data", "data/input.txt"])
@pytest.mark.parametrize("role", ["worker", "oracle", "verifier"])
def test_shared_resource_destinations_cannot_overwrite_role_mounts(specification, second_path, role):
    wire = specification.model_dump()
    wire["resources"] = {
        "all": [{"path": "data", "source": {"kind": "inline_file", "content_base64": "c2hhcmVk"}}],
        role: [{"path": second_path, "source": {"kind": "inline_file", "content_base64": "cHJpdmF0ZQ=="}}],
    }
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(wire)


def test_private_role_mounts_reuse_paths_without_becoming_worker_visible(specification):
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
    resources = TaskSpec.model_validate_json(json.dumps(wire)).model_dump(mode="json")["resources"]
    assert resources["all"] == []
    assert base64.b64decode(resources["worker"][0]["source"]["content_base64"]) == b"public"
    assert base64.b64decode(resources["oracle"][0]["source"]["content_base64"]) == b"gold"
    assert base64.b64decode(resources["verifier"][0]["source"]["content_base64"]) == b"hidden test"


def test_build_context_roundtrip_keeps_bytes_metadata_and_role_boundaries(specification):
    dockerfile = inline_resource("Dockerfile", b"FROM mutable:latest\nCOPY payload.bin /input\n")
    payload = inline_resource("payload.bin", b"\x00\xff\x80").model_copy(update={"mode": "0755", "mtime_ns": 123456789})
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_build=DockerBuildContext(files=(dockerfile, payload))
    )
    wire = specification.model_dump()
    wire["environment_requirements"] = environment.model_dump()
    wire["grader"] = ScriptGrader(argv=("true",), environment=environment).model_dump()
    task = TaskSpec.model_validate_json(json.dumps(wire))
    restored = task.model_dump(mode="json")
    for requirements in (restored["environment_requirements"], restored["grader"]["environment"]):
        files = {resource["path"]: resource for resource in requirements["docker_build"]["files"]}
        assert (
            base64.b64decode(files["Dockerfile"]["source"]["content_base64"])
            == b"FROM mutable:latest\nCOPY payload.bin /input\n"
        )
        assert base64.b64decode(files["payload.bin"]["source"]["content_base64"]) == b"\x00\xff\x80"
        assert (files["payload.bin"]["mode"], files["payload.bin"]["mtime_ns"]) == ("0755", 123456789)
    assert task.resources == ResourceGroups()


@pytest.mark.parametrize("initial_state", [None, "company-snapshot", {"inbox": [], "counter": 3}])
def test_reader_keeps_literal_provider_state_but_direct_chat_cannot_request_it(specification, initial_state):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"]["tool_providers"] = {
        "company": {"action_interface": "workplace:v1", "initial_state": initial_state}
    }
    task = TaskSpec.model_validate_json(json.dumps(wire))
    assert task.environment_requirements.tool_providers["company"].initial_state == initial_state
    with pytest.raises(NotImplementedError):
        chat_request(task)


def test_reader_rejects_nested_nonfinite_provider_state(specification):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"]["tool_providers"] = {
        "company": {"action_interface": "workplace:v1", "initial_state": {"counters": [float("nan")]}}
    }
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(json.dumps(wire))


GRADER_CONFIGURATION = {"checks": [{"tolerance": 0.125}], "label": "NaN"}


@pytest.mark.parametrize(
    "grader",
    [
        {
            "kind": "verifyit",
            "mode": "structured_exact",
            "parameters": {"expected": GRADER_CONFIGURATION},
            "environment": None,
        },
        {"kind": "none", "reason": "Source evaluator is unavailable", "contract": GRADER_CONFIGURATION},
    ],
)
@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_grader_configuration_rejects_nested_nonfinite_json_numbers(specification, grader, number):
    # JsonValue alone accepts nonfinite numbers despite allow_inf_nan=False.
    wire = {**specification.model_dump(mode="json"), "answer_type": "json", "answer_format": {"kind": "json_value"}}
    text = json.dumps({**wire, "grader": grader})
    assert TaskSpec.model_validate_json(text).model_dump(mode="json")["grader"] == grader
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(text.replace("0.125", number))


@pytest.mark.parametrize(
    "base_path,alias",
    [("foo", "foo."), ("foo", "foo "), ("inputs/answer", "inputs/answer:backup"), ("foo", "FOO"), ("a/b", "a\\b")],
)
def test_resource_groups_preserve_distinct_linux_files(tmp_path, specification, base_path, alias):
    wire = specification.model_dump(mode="json")
    wire["resources"] = {
        "all": [{"path": base_path, "source": {"kind": "inline_file", "content_base64": "cHVibGlj"}}],
        "worker": [{"path": alias, "source": {"kind": "inline_file", "content_base64": "b3ZlcndyaXRl"}}],
    }
    task = TaskSpec.model_validate(wire)
    for resource in task.resources.all + task.resources.worker:
        target = tmp_path / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(resource.source.content_base64))
    assert (tmp_path / base_path).read_bytes() == b"public"
    assert (tmp_path / alias).read_bytes() == b"overwrite"


@pytest.mark.parametrize("candidate,reward", [("done", 1.0), ("incorrect", 0.0)])
def test_in_process_grading_accepts_answers_acquired_in_a_worker_workspace(specification, candidate, reward):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"] = {"capabilities": ["shell", "filesystem"], "working_directory": "/app"}
    wire["resources"] = {
        "worker": [{"path": "project.txt", "source": {"kind": "inline_file", "content_base64": "d29ya2VyIGlucHV0"}}]
    }
    task = TaskSpec.model_validate(wire)
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=candidate)))
    result = grade_answer(task, GradingAttempt(conversation))
    assert (result.status, result.reward) == ("graded", reward)


@pytest.mark.parametrize("payload", [b"UTF-8 text: \xe2\x98\x83\n", b"\x00\xff\x80\n"])
def test_inline_file_bytes_and_metadata_survive_json_reader(specification, payload):
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
    restored = TaskSpec.model_validate_json(task.model_dump_json()).resources.worker[0]
    assert base64.b64decode(restored.source.content_base64) == payload
    assert restored.mode == "0500"
    assert restored.mtime_ns == 1_725_555_600_123_456_789


SCRIPT = {
    "kind": "script",
    "argv": ["python3", "/tests/grade.py"],
    "environment": {"docker_image": IMAGE, "command_semantics": "linux_process"},
}
VERIFYIT_ENVIRONMENT = {"kind": "verifyit", "environment": {"docker_image": IMAGE, "command_semantics": "linux_process"}}


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(
            {"answer_type": "file", "output_paths": ["/tests/answer.txt"], "grader": SCRIPT | {"answer_path": None}},
            id="output-path-under-tests",
        ),
        pytest.param(
            {
                "answer_type": "file",
                "environment_requirements": {"capabilities": ["python3"]},
                "output_directories": [{"root": "/logs/verifier", "patterns": ["*"], "max_files": 1, "max_bytes": 1}],
                "grader": SCRIPT | {"answer_path": None},
            },
            id="output-directory-under-verifier-logs",
        ),
        pytest.param(
            {"answer_type": "file", "output_paths": ["/app/answer.txt"], "grader": SCRIPT},
            id="script-answer-path-for-file-answer",
        ),
        pytest.param({"grader": SCRIPT | {"answer_path": "/tests/answer.txt"}}, id="script-answer-path-under-tests"),
        pytest.param(
            {
                "grader": SCRIPT | {"conversation_path": "/tests/grade.py"},
                "resources": {
                    "verifier": [{"path": "grade.py", "source": {"kind": "inline_file", "content_base64": "eA=="}}]
                },
            },
            id="script-conversation-path-replaces-resource",
        ),
        pytest.param(
            {
                "grader": (
                    VERIFYIT_ENVIRONMENT
                    | {"mode": "exact", "parameters": {"expected": ["done"], "output": "/tests/answer.txt"}}
                )
            },
            id="verifyit-answer-file-under-tests",
        ),
    ],
)
def test_task_validation_rejects_invalid_grading_contracts(specification, update):
    wire = specification.model_dump(mode="json")
    with pytest.raises(ValidationError):
        TaskSpec.model_validate({**wire, **update})


def test_task_json_rejects_prior_schema_version(specification):
    payload = json.loads(specification.model_dump_json())
    payload["schema_version"] = "0.25"
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(json.dumps(payload))
