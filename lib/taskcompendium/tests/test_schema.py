# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete semantic records round-trip independently of runtime support."""

import json

import pytest
from pydantic import ValidationError

from taskcompendium.grading import exact_answer
from taskcompendium.lowering import HarborEnvironmentConfig, compatible_lowerings, lower_to_harbor, read_specification
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.submission import PlainText, chat_request
from taskcompendium.verifier_registry import resolve_verifier


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


def test_complete_private_schema_round_trips_without_execution(tmp_path, specification):
    wire = json.loads(specification.model_dump_json())
    wire.update(
        answer_type="workspace_state",
        tags=["", "difficulty:7", "custom label", "custom label"],
        environment_requirements={
            "capabilities": ["shell", "filesystem", "process", "browser", "network"],
            "docker_image": "org/project@sha256:" + "a" * 64,
            "working_directory": "/app",
            "setup_commands": ["git checkout pinned-commit"],
            "additional_workspace_roots": ["/data"],
            "tool_providers": {"company": {"action_interface": "workplace:v1", "seed_sha256": "b" * 64}},
        },
        verifier={"kind": "script", "parameters_json": '{"entrypoint":"private/check.py","image":"private/image"}'},
        resources={
            "all": [{"path": "readme.txt", "source": {"kind": "inline_text", "content": "Shared inputs."}}],
            "worker": [
                {"path": "input.txt", "source": {"kind": "inline_text", "content": "café\n"}},
                {
                    "path": "project",
                    "source": {"kind": "dataset_file", "path": "archives/project.tar.gz", "sha256": "c" * 64},
                    "format": "tar_gz",
                },
                {
                    "path": "fixtures",
                    "source": {"kind": "dataset_directory", "path": "projects/example/fixtures"},
                    "format": "directory",
                },
            ],
            "oracle": [{"path": "input.txt", "source": {"kind": "inline_binary", "content_base64": "AP8="}}],
            "verifier": [
                {
                    "path": "private/check.py",
                    "source": {"kind": "inline_text", "content": "raise SystemExit(0)\n"},
                    "executable": True,
                },
                {"path": "input.txt", "source": {"kind": "inline_text", "content": "private evaluator fixture"}},
            ],
        },
    )
    path = tmp_path / "specification.json"
    path.write_text(json.dumps(wire))
    loaded = read_specification(path)
    persisted = json.loads(loaded.model_dump_json())
    assert set(persisted) == {
        "id",
        "context",
        "environment_requirements",
        "final_tools",
        "answer_type",
        "source",
        "verifier",
        "schema_version",
        "resources",
        "tags",
    }
    assert persisted["tags"] == ["", "difficulty:7", "custom label", "custom label"]
    assert persisted["environment_requirements"] == wire["environment_requirements"]
    assert persisted["verifier"] == wire["verifier"]
    assert persisted["resources"]["oracle"][0]["source"] == {"kind": "inline_binary", "content_base64": "AP8="}
    assert persisted["resources"]["worker"][1]["format"] == "tar_gz"
    assert persisted["resources"]["worker"][2] == {
        "path": "fixtures",
        "source": {"kind": "dataset_directory", "path": "projects/example/fixtures"},
        "format": "directory",
        "executable": False,
    }
    assert set(persisted["resources"]) == {"all", "worker", "oracle", "verifier"}
    path.write_text(loaded.model_dump_json())
    assert read_specification(path).model_dump(mode="json") == persisted
    with pytest.raises(NotImplementedError):
        resolve_verifier(loaded.verifier)


@pytest.mark.parametrize(
    "update",
    [
        {"environment_requirements": EnvironmentRequirements(capabilities=("browser",))},
        {"environment_requirements": EnvironmentRequirements(docker_image="org/image@sha256:" + "a" * 64)},
        {"environment_requirements": EnvironmentRequirements(working_directory="/app")},
        {"environment_requirements": EnvironmentRequirements(setup_commands=("initialize",))},
        {"environment_requirements": EnvironmentRequirements(additional_workspace_roots=("/data",))},
        {
            "environment_requirements": EnvironmentRequirements.model_validate(
                {
                    "tool_providers": {"company": {"action_interface": "workplace:v1", "seed_sha256": "b" * 64}},
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
                        "source": {"kind": "dataset_directory", "path": "tasks/example/project"},
                        "format": "directory",
                    }
                ]
            }
        },
        {"answer_type": AnswerType.FILE},
        {"answer_type": AnswerType.STATE},
        {"answer_type": AnswerType.WORKSPACE_STATE},
        {"verifier": VerifierSpec(kind="llm_judge", parameters_json='{"rubric":"private"}')},
    ],
)
def test_direct_chat_rejects_semantics_it_cannot_preserve_before_export(tmp_path, specification, update):
    task = TaskSpec.model_validate({**specification.model_dump(), **update})
    convention = PlainText(id="plain")
    assert compatible_lowerings(task, (convention,), (HarborEnvironmentConfig(),)) == ()
    with pytest.raises(NotImplementedError):
        chat_request(task, convention)
    destination = tmp_path / "task"
    with pytest.raises(NotImplementedError):
        lower_to_harbor(task, convention, HarborEnvironmentConfig(), destination)
    assert not destination.exists()


@pytest.mark.parametrize("path", ["../private.txt", "/private.txt", "https://example.com/private.txt", "a/../b"])
@pytest.mark.parametrize("source_kind", ["dataset_file", "dataset_directory"])
def test_dataset_resource_references_cannot_escape_the_pinned_dataset(specification, path, source_kind):
    wire = specification.model_dump()
    source = {"kind": source_kind, "path": path}
    if source_kind == "dataset_file":
        source["sha256"] = "a" * 64
    wire["resources"] = {
        "worker": [
            {
                "path": "input",
                "source": source,
                "format": "directory" if source_kind == "dataset_directory" else "file",
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
