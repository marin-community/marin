# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC bindings retain source files and declare the source runtime."""

import base64
import json
from functools import partial

from taskcompendium.datasets import atlas_arc_injection
from taskcompendium.models import Source, TaskSpec
from taskcompendium.native_grader import NativeCommandSpec
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.arc import binding as arc_binding

CASES = [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": [[3, 4]], "output": [[4, 3]]}]
IMAGE = "example.invalid/arc@sha256:" + "a" * 64


def tasktrove_row(files: dict[str, bytes], *, name: str = "arc_inductive") -> RawRow:
    return RawRow(
        "arc-fixture",
        Source(
            dataset=arc_binding.TASKTROVE_PIN[0],
            revision=arc_binding.TASKTROVE_PIN[1],
            row="fixture",
            importer_revision="test",
        ),
        {
            "instruction": "Write transform to /app/solution.py.",
            "verifier_data": {"test_cases": CASES} if name == "arc_inductive" else {"expected_output": [[2, 1]]},
            "archive_sha256": "fixture-archive-hash",
            "path": "fixture/arc",
            "files": {path: base64.b64encode(content).decode() for path, content in files.items()},
            "file_metadata": {
                path: {"mode": "0755" if path.endswith(".sh") else "0644", "mtime_ns": 1} for path in files
            },
            "archive_links": {},
        },
    )


def test_tasktrove_missing_native_command_remains_neutrally_unsupported():
    row = tasktrove_row({"tests/verifier.py": b"# original grader without its command"})
    result = arc_binding.normalize_isolated(
        row,
        normalize_task=partial(atlas_arc_injection.normalize, name="arc_inductive"),
        image=IMAGE,
        source=arc_binding.ArcSource.TASKTROVE,
    )
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.UNSUPPORTED, "missing_original_arc_command")


def test_tasktrove_runtime_preserves_archived_command_and_private_files():
    files = {
        "tests/test.sh": b"#!/bin/bash\npython3 /tests/verifier.py\n",
        "tests/verifier.py": b"# source verifier bytes\n",
        "task.toml": b"[verifier]\ntimeout_sec = 600\n",
        "solution/solve.sh": b"# source golden script",
    }
    task = arc_binding.normalize_isolated(
        tasktrove_row(files),
        normalize_task=partial(atlas_arc_injection.normalize, name="arc_inductive"),
        image=IMAGE,
        source=arc_binding.ArcSource.TASKTROVE,
    )
    assert isinstance(task, TaskSpec)
    assert [resource.path for resource in task.resources.oracle] == ["task.toml", "solution/solve.sh"]
    assert task.resources.worker == ()
    command = NativeCommandSpec.model_validate_json(task.verifier.parameters_json)
    assert command.argv == ("bash", "/tests/test.sh")
    assert command.result_path == "/logs/verifier/reward.txt"
    mounted = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    assert mounted["test.sh"] == files["tests/test.sh"]
    assert mounted["verifier.py"] == files["tests/verifier.py"]
    assert "config.json" not in mounted


def test_tasktrove_transductive_binds_original_command_without_solution_program():
    files = {
        "tests/test.sh": b"#!/bin/bash\npython3 /tests/verifier.py\n",
        "tests/verifier.py": b"# source transductive verifier bytes\n",
        "task.toml": b"[verifier]\ntimeout_sec = 600\n",
    }
    task = arc_binding.normalize_isolated(
        tasktrove_row(files, name="arc_transductive"),
        normalize_task=partial(atlas_arc_injection.normalize, name="arc_transductive"),
        image=IMAGE,
        source=arc_binding.ArcSource.TASKTROVE,
    )
    assert isinstance(task, TaskSpec)
    assert task.output_paths == ("/app/answer.txt",)
    assert {resource.path: resource_bytes(resource) for resource in task.resources.verifier}["verifier.py"] == files[
        "tests/verifier.py"
    ]
    assert NativeCommandSpec.model_validate_json(task.verifier.parameters_json).argv == ("bash", "/tests/test.sh")


def test_ultra_declares_image_scorer_and_private_contract():
    contract = {"agent_ref": {"name": arc_binding.AGENTS[0]}, "test_input": [[1, 2]], "expected_output": [[2, 1]]}
    package = arc_binding.original_package({"evaluator": "ultra", "contract": contract})
    command = NativeCommandSpec.model_validate_json(package.verifier.parameters_json)
    assert command.argv == ("python3", "/tests/grade.py")
    assert command.result_format == "score_json"
    resources = {resource.path: resource_bytes(resource) for resource in package.resources}
    assert set(resources) == {"grade.py", "arc_contract.json"}
    assert json.loads(resources["arc_contract.json"])["contract"] == contract


def test_ultra_transductive_uses_shared_original_callable_transport():
    contract = {"agent_ref": {"name": arc_binding.AGENTS[1]}, "test_input": [[1, 2]], "expected_output": [[2, 1]]}
    package = arc_binding.original_package({"evaluator": "ultra", "contract": contract})
    command = NativeCommandSpec.model_validate_json(package.verifier.parameters_json)
    assert command.argv[1] == "/tests/source_callable.py"
    resources = {resource.path: resource_bytes(resource) for resource in package.resources}
    assert json.loads(resources["invocation.json"]) == {
        "function": "skyrl_gym.envs.nemotron_ultra.nvarc:grade_transductive_arc",
        "args": ["answer", "contract"],
    }
    assert json.loads(resources["arc_contract.json"])["contract"] == contract
