# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readable reviewer evidence retains fixture visibility and bounds large files."""

import hashlib
import json

from taskcompendium.models import (
    CommandSemantics,
    DockerBuildContext,
    EnvironmentRequirements,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.review import completion_body
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from .pipeline_stages import SVAMP_RUBRIC, svamp_row_task


def test_review_can_inspect_verifier_text_without_changing_task_bytes():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(
        RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"})
    )
    assert isinstance(task, TaskSpec)
    resource = inline_resource("tests/cases.json", b'{"input":"two","output":"2"}')
    task = task.model_copy(update={"resources": ResourceGroups(verifier=(resource,))})
    payload = json.loads(completion_body(task, SVAMP_RUBRIC, "reviewer", 100)["messages"][1]["content"])
    assert payload["resources"][0]["text"] == '{"input":"two","output":"2"}'
    assert payload["resources"][0]["role"] == "verifier"
    assert resource_bytes(task.resources.verifier[0]) == resource_bytes(resource)


def test_review_bounds_private_and_build_files_without_changing_the_task():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(
        RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"})
    )
    assert isinstance(task, TaskSpec)
    large = inline_resource("large", b"x" * 100_000)
    build = DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM mutable:latest\n"), large))
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_build=build)
    task = task.model_copy(
        update={
            "environment_requirements": environment,
            "grader": ScriptGrader(environment=environment, argv=("true",)),
            "resources": ResourceGroups(verifier=(large,)),
        }
    )
    original = task.model_dump_json()
    payload = json.loads(completion_body(task, SVAMP_RUBRIC, "reviewer", 100)["messages"][1]["content"])
    assert "content_base64" not in json.dumps(payload)
    previews = {resource["role"]: resource for resource in payload["resources"] if resource["path"] == "large"}
    assert previews.keys() == {"verifier", "actor_build", "grader_build"}
    for preview in previews.values():
        assert preview["truncated"] and preview["byte_count"] == 100_000
        assert 0 < len(preview["text"]) < preview["byte_count"]
    assert payload["resource_manifest"]["total_count"] == 5
    assert task.model_dump_json() == original


def test_fixture_heavy_review_keeps_public_inputs_and_oracle_visible():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(
        RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"})
    )
    assert isinstance(task, TaskSpec)
    resources = ResourceGroups(
        verifier=tuple(inline_resource(f"tests/case-{index}.txt", b"case") for index in range(300)),
        worker=(inline_resource("input.txt", b"Public input"),),
        oracle=(inline_resource("solution/solve.sh", b"Private oracle"),),
    )
    task = task.model_copy(update={"resources": resources})
    body = completion_body(task, SVAMP_RUBRIC, "reviewer", 100)
    payload = json.loads(body["messages"][1]["content"])
    previews = {resource["path"]: resource["text"] for resource in payload["resources"]}
    assert previews["input.txt"] == "Public input"
    assert previews["solution/solve.sh"] == "Private oracle"
    assert payload["resource_manifest"]["omitted_count"] > 0
    assert task.resources == resources
    changed_resources = resources.model_copy(
        update={"verifier": (*resources.verifier[:-1], inline_resource("tests/case-299.txt", b"edit"))}
    )
    changed_task = task.model_copy(update={"resources": changed_resources})
    changed_body = completion_body(changed_task, SVAMP_RUBRIC, "reviewer", 100)
    changed_payload = json.loads(changed_body["messages"][1]["content"])
    assert changed_payload["resources"] == payload["resources"]
    assert changed_payload["resource_manifest"]["sha256"] != payload["resource_manifest"]["sha256"]
    assert changed_body != body


def test_review_exposes_late_small_cases_that_can_violate_the_public_domain():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(
        RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"})
    )
    assert isinstance(task, TaskSpec)
    resources = ResourceGroups(
        verifier=tuple(
            resource
            for index in range(100)
            for resource in (
                inline_resource(f"tests/input_{index}.txt", b"0" if index == 80 else b"123"),
                inline_resource(f"tests/output_{index}.txt", b"1"),
            )
        )
    )
    task = task.model_copy(update={"resources": resources})
    payload = json.loads(completion_body(task, SVAMP_RUBRIC, "reviewer", 100)["messages"][1]["content"])
    previews = {resource["path"]: resource["text"] for resource in payload["resources"]}
    assert previews["tests/input_80.txt"] == "0"
    assert previews["tests/output_80.txt"] == "1"


def test_large_public_files_do_not_hide_later_oracle_and_test_prefixes():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(
        RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"})
    )
    assert isinstance(task, TaskSpec)
    oracle = "é" * 1_000
    resources = ResourceGroups(
        worker=tuple(inline_resource(f"large-{index}.txt", b"a" * 100_000) for index in range(4)),
        oracle=(inline_resource("solution/solve.sh", oracle.encode()),),
        verifier=(
            *(inline_resource(f"tests/case-{index}.txt", b"x" * 1_000) for index in range(250)),
            inline_resource("tests/binary.bin", b"\xff\x00"),
        ),
    )
    task = task.model_copy(update={"resources": resources})
    payload = json.loads(completion_body(task, SVAMP_RUBRIC, "reviewer", 100)["messages"][1]["content"])
    previews = {resource["path"]: resource for resource in payload["resources"]}
    text_previews = [resource for resource in previews.values() if resource["encoding"] == "utf-8"]
    assert all(len(resource["text"]) >= 100 for resource in text_previews)
    assert sum(len(resource["text"]) for resource in text_previews) <= 32_768
    assert previews["solution/solve.sh"]["text"] == oracle[:100]
    assert previews["solution/solve.sh"]["byte_count"] == len(oracle.encode())
    assert previews["solution/solve.sh"]["sha256"] == hashlib.sha256(oracle.encode()).hexdigest()
    assert previews["tests/case-249.txt"]["text"] == "x" * 100
    assert previews["tests/case-249.txt"]["truncated"]
    assert previews["tests/binary.bin"]["text"] is None
    assert previews["tests/binary.bin"]["byte_count"] == 2
    assert task.resources == resources
