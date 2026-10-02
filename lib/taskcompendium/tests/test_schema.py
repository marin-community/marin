# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public schema loading and runtime boundaries preserve private contracts."""

import json

import pytest
from pydantic import ValidationError

from taskcompendium.grading import exact_answer
from taskcompendium.lowering import HarborEnvironmentConfig, compatible_lowerings, lower_to_harbor, read_specification
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
from taskcompendium.submission import GradingAttempt, PlainText, chat_request
from taskcompendium.verifier_registry import grade_answer


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
        {"resources": {role: [{"path": "input.txt", "source": {"kind": "inline_text", "content": "x"}}]}}
        for role in ("all", "worker", "oracle", "verifier")
    ]
    + [
        {
            "resources": {
                "worker": [
                    {
                        "path": "project",
                        "source": {"kind": "dataset_path", "path": "vendored/project"},
                        "mode": "0755",
                    }
                ]
            }
        },
        {"answer_type": AnswerType.FILE},
        {"answer_type": AnswerType.STATE},
        {"answer_type": AnswerType.WORKSPACE_STATE},
        {"verifier": VerifierSpec(kind="llm_judge", parameters_json='{"rubric":"private"}')},
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
                        "path": "checks",
                        "source": {"kind": "dataset_path", "path": "vendored/checks.tar.gz"},
                        "format": "tar_gz",
                        "mode": "0755",
                    }
                ]
            }
        },
    ],
)
def test_direct_chat_rejects_semantics_it_cannot_preserve_before_export(tmp_path, specification, update):
    record = TaskSpec.model_validate({**specification.model_dump(), **update})
    path = tmp_path / "specification.json"
    path.write_text(record.model_dump_json())
    task = read_specification(path)
    convention = PlainText(id="plain")
    assert compatible_lowerings(task, (convention,), (HarborEnvironmentConfig(),)) == ()
    with pytest.raises(NotImplementedError):
        chat_request(task, convention)
    destination = tmp_path / "task"
    with pytest.raises(NotImplementedError):
        lower_to_harbor(task, convention, HarborEnvironmentConfig(), destination)
    assert not destination.exists()


@pytest.mark.parametrize("path", ["../private.txt", "/private.txt", "https://example.com/private.txt", "a/../b"])
@pytest.mark.parametrize("digest", [None, "a" * 64])
def test_vendored_resource_references_cannot_escape_the_dataset_reader_root(specification, path, digest):
    wire = specification.model_dump()
    source = {"kind": "dataset_path", "path": path, "sha256": digest}
    wire["resources"] = {
        "worker": [
            {
                "path": "input",
                "source": source,
            }
        ]
    }
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(wire)


@pytest.mark.parametrize("second_path", ["data", "DATA", "data/input.txt"])
@pytest.mark.parametrize("role", ["worker", "oracle", "verifier"])
def test_shared_resource_destinations_cannot_overwrite_role_mounts(specification, second_path, role):
    wire = specification.model_dump()
    wire["resources"] = {
        "all": [{"path": "data", "source": {"kind": "inline_text", "content": "shared"}}],
        role: [{"path": second_path, "source": {"kind": "inline_text", "content": "private"}}],
    }
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(wire)


def test_private_role_mounts_reuse_paths_without_becoming_worker_visible(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["resources"] = {
        role: [{"path": "fixture.txt", "source": {"kind": "inline_text", "content": content}}]
        for role, content in (("worker", "public"), ("oracle", "gold"), ("verifier", "hidden test"))
    }
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    resources = read_specification(path).model_dump(mode="json")["resources"]
    assert resources["all"] == []
    assert resources["worker"][0]["source"]["content"] == "public"
    assert resources["oracle"][0]["source"]["content"] == "gold"
    assert resources["verifier"][0]["source"]["content"] == "hidden test"


@pytest.mark.parametrize("initial_state", [None, "company-snapshot", {"inbox": [], "counter": 3}])
def test_reader_keeps_literal_provider_state_but_direct_chat_cannot_export_it(tmp_path, specification, initial_state):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"]["tool_providers"] = {
        "company": {"action_interface": "workplace:v1", "initial_state": initial_state}
    }
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    task = read_specification(path)
    with pytest.raises(NotImplementedError):
        lower_to_harbor(task, PlainText(id="plain"), HarborEnvironmentConfig(), tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_reader_rejects_nested_nonfinite_provider_state(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"]["tool_providers"] = {
        "company": {"action_interface": "workplace:v1", "initial_state": {"counters": [float("nan")]}}
    }
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    with pytest.raises(ValidationError):
        read_specification(path)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_private_verifier_config_rejects_nested_nonfinite_json_numbers(tmp_path, specification, number):
    # JsonValue previously allowed nonfinite values despite allow_inf_nan=False.
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


async def test_pure_grading_cannot_ignore_a_private_verifier_environment(tmp_path, specification):
    wire = specification.model_dump(mode="json")
    wire["verifier"]["environment_requirements"] = {"docker_image": "private/grader@sha256:" + "a" * 64}
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    task = read_specification(path)
    attempt = GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="done"))),
        workspace=object(),
    )
    # This correct answer must not earn credit without the required private runtime.
    with pytest.raises(NotImplementedError):
        await grade_answer(task, PlainText(id="plain"), attempt)
