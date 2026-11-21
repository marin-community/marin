# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The task's file inventory and signature, shared by the task review and other readers of a task's files."""

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
from taskcompendium.pipeline.review import review_payload
from taskcompendium.runtime.resources import inline_resource, resource_bytes, resource_inventory, resource_signature

from .pipeline_stages import svamp_row_task


def task_with_files() -> TaskSpec:
    """A task with a file in every group, each with its own mode or mtime, and actor and grader build inputs."""
    source = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    task = svamp_row_task(
        RawRow("fixture", source, {"Body": "I have 2 apples.", "Question": "How many?", "Answer": "2"})
    )
    assert isinstance(task, TaskSpec)
    actor = DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM actor:1\n"),))
    grader = DockerBuildContext(
        files=(inline_resource("Dockerfile", b"FROM grader:1\n"), inline_resource("setup.sh", b"true\n"))
    )
    groups = ResourceGroups(
        all=(inline_resource("data.json", b"[1, 2]"),),
        worker=(inline_resource("notes.txt", b"hi").model_copy(update={"mode": "0600"}),),
        oracle=(inline_resource("answer.txt", b"2"),),
        verifier=(inline_resource("check.sh", b"exit 0\n").model_copy(update={"mode": "0755", "mtime_ns": 7}),),
    )
    semantics = CommandSemantics.LINUX_PROCESS
    return task.model_copy(
        update={
            "resources": groups,
            "environment_requirements": EnvironmentRequirements(command_semantics=semantics, docker_build=actor),
            "grader": ScriptGrader(
                environment=EnvironmentRequirements(command_semantics=semantics, docker_build=grader), argv=("true",)
            ),
        }
    )


def test_the_inventory_lists_each_group_then_the_actor_and_grader_build_inputs_in_order() -> None:
    inventory = [(role, resource.path) for role, resource in resource_inventory(task_with_files())]
    assert inventory == [
        ("all", "data.json"),
        ("worker", "notes.txt"),
        ("oracle", "answer.txt"),
        ("verifier", "check.sh"),
        ("actor_build", "Dockerfile"),
        ("grader_build", "Dockerfile"),
        ("grader_build", "setup.sh"),
    ]


def test_a_signature_names_role_path_mode_mtime_size_and_hash() -> None:
    signatures = [
        resource_signature(role, resource, resource_bytes(resource))
        for role, resource in resource_inventory(task_with_files())
    ]
    assert signatures[3] == {
        "role": "verifier",
        "path": "check.sh",
        "mode": "0755",
        "mtime_ns": 7,
        "sha256": hashlib.sha256(b"exit 0\n").hexdigest(),
        "byte_count": 7,
    }
    assert signatures[1]["mode"] == "0600" and signatures[1]["mtime_ns"] is None


def test_the_task_review_payload_is_unchanged_by_the_shared_inventory() -> None:
    """Pinned at origin/main cb991a5a5c, before the inventory moved out of the review."""
    payload = json.dumps(review_payload(task_with_files()), sort_keys=True).encode()
    assert hashlib.sha256(payload).hexdigest() == "c3128d91d9ebcfefb82343db4c4092060af1e910637e21ab3fcba6f12c3ae948"
