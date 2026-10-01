# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readable reviewer evidence retains fixture visibility and bounds large files."""

import json

from taskcompendium.models import ResourceVisibility, Source, TaskSpec, task_resource
from taskcompendium.pipeline.datasets.svamp import normalize, recipe
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.review import completion_body


def test_review_can_inspect_private_text_without_changing_task_bytes():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = normalize(RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"}))
    assert isinstance(task, TaskSpec)
    resource = task_resource("/tests/cases.json", b'{"input":"two","output":"2"}', ResourceVisibility.VERIFIER)
    task = task.model_copy(update={"resources": (resource,)})
    payload = json.loads(completion_body(task, recipe.rubric, "reviewer", 100)["messages"][1]["content"])
    assert payload["resources"][0]["text"] == '{"input":"two","output":"2"}'
    assert payload["resources"][0]["visibility"] == "verifier"
    assert task.resources[0].data() == resource.data()


def test_review_marks_omitted_fixture_content_and_retains_full_task():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = normalize(RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"}))
    assert isinstance(task, TaskSpec)
    resource = task_resource("/tests/large.txt", b"a" * 100_000, ResourceVisibility.VERIFIER)
    task = task.model_copy(update={"resources": (resource,)})
    payload = json.loads(completion_body(task, recipe.rubric, "reviewer", 100)["messages"][1]["content"])
    preview = payload["resources"][0]
    assert preview["truncated"] and preview["byte_count"] == 100_000
    assert 0 < len(preview["text"]) < preview["byte_count"]
    assert task.resources[0].data() == b"a" * 100_000


def test_fixture_heavy_review_keeps_public_inputs_and_oracle_visible():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = normalize(RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"}))
    assert isinstance(task, TaskSpec)
    resources = (
        *(task_resource(f"/tests/case-{index}.txt", b"case", ResourceVisibility.VERIFIER) for index in range(300)),
        task_resource("/input.txt", b"Public input", ResourceVisibility.AGENT),
        task_resource("/solution/solve.sh", b"Private oracle", ResourceVisibility.CONTROL),
    )
    task = task.model_copy(update={"resources": resources})
    body = completion_body(task, recipe.rubric, "reviewer", 100)
    payload = json.loads(body["messages"][1]["content"])
    previews = {resource["path"]: resource["text"] for resource in payload["resources"]}
    assert previews["/input.txt"] == "Public input"
    assert previews["/solution/solve.sh"] == "Private oracle"
    assert payload["resource_manifest"]["omitted_count"] > 0
    assert task.resources == resources


def test_review_exposes_late_small_cases_that_can_violate_the_public_domain():
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = normalize(RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"}))
    assert isinstance(task, TaskSpec)
    resources = tuple(
        resource
        for index in range(100)
        for resource in (
            task_resource(f"/tests/input_{index}.txt", b"0" if index == 80 else b"123", ResourceVisibility.VERIFIER),
            task_resource(f"/tests/output_{index}.txt", b"1", ResourceVisibility.VERIFIER),
        )
    )
    task = task.model_copy(update={"resources": resources})
    payload = json.loads(completion_body(task, recipe.rubric, "reviewer", 100)["messages"][1]["content"])
    previews = {resource["path"]: resource["text"] for resource in payload["resources"]}
    assert previews["/tests/input_80.txt"] == "0"
    assert previews["/tests/output_80.txt"] == "1"
