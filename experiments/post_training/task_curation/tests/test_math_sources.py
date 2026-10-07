# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain both original scorer contracts while actors submit ordinary text."""

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from taskcompendium.models import AnswerType, EnvironmentRequirements, Source, TaskSpec
from taskcompendium.native_grader import NativeCommandSpec
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.pipeline import SourceRuntime, SourceRuntimeConfig
from experiments.post_training.task_curation.sources import rl_data_pipelines
from experiments.post_training.tasktrove.taskbinary import TaskFiles, read_task_binary

SOURCES = {source.source_key: source for source in rl_data_pipelines().values()}


@dataclass(frozen=True)
class MathTask:
    original: TaskFiles
    task: TaskSpec


@pytest.fixture
def math_task(tmp_path):
    def load(name):
        # Complete files from pinned TaskTrove row zero, repacked without changes.
        original = read_task_binary((Path(__file__).parent / f"fixtures/{name}.tar.gz").read_bytes())
        recipe = SOURCES[
            {"math_gym": "Task Trove:laion__nemotron-gym-math-v5", "math_prism": "Task Trove:laion__nemo-prism-math-v3"}[
                name
            ]
        ].recipe(
            SourceRuntimeConfig(
                images={
                    {
                        "math_gym": "Task Trove:laion__nemotron-gym-math-v5",
                        "math_prism": "Task Trove:laion__nemo-prism-math-v3",
                    }[name]: SourceRuntime(
                        backend="qemu",
                        image="example.org/grader@sha256:" + "a" * 64,
                        worker_image="example.org/worker@sha256:" + "b" * 64,
                        qemu_bundle=str(tmp_path / "bundle"),
                    )
                },
                controller_url=None,
            )
        )
        source = Source(dataset=recipe.source.dataset, revision=recipe.source.revision, row="0", importer_revision="1")
        result = recipe.policy.normalize(
            RawRow(
                name,
                source,
                {
                    "instruction": original.text("instruction.md"),
                    "verifier_data": json.loads(original.text("tests/verifier_data.json")),
                    "files": {path: base64.b64encode(content).decode() for path, content in original.files.items()},
                },
            )
        )
        assert isinstance(result, NormalizedTask)
        return MathTask(original, TaskSpec.model_validate_json(result.task.model_dump_json()))

    return load


@pytest.mark.parametrize("name", ["math_gym", "math_prism"])
def test_math_binding_keeps_text_actor_original_private_scorer_and_actual_goldens(math_task, name):
    fixture = math_task(name)
    original, task = fixture.original, fixture.task
    assert task.answer_type == AnswerType.TEXT
    assert task.environment_requirements == EnvironmentRequirements()
    assert task.output_paths == ("/app/answer.txt",)
    assert not task.resources.all and not task.resources.worker
    private = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    for path, content in original.files.items():
        if not path.startswith("solution/"):
            assert private["source/" + path] == content
    oracle = {resource.path: resource_bytes(resource) for resource in task.resources.oracle}
    assert oracle == {path: content for path, content in original.files.items() if path.startswith("solution/")}
    assert bool(oracle) == (name == "math_prism")


@pytest.mark.parametrize("name", ["math_gym", "math_prism"])
def test_math_original_runner_is_bound_without_rewriting(math_task, name):
    fixture = math_task(name)
    task = fixture.task
    spec = NativeCommandSpec.model_validate_json(task.verifier.parameters_json)
    assert spec.argv == ("bash", "-c", "python3 /tests/runtime_check.py && bash /tests/test.sh")
    assert spec.result_path == "/logs/verifier/reward.txt"
    private = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    assert private["verifier.py"] == fixture.original.files["tests/verifier.py"]
    assert private["test.sh"] == fixture.original.files["tests/test.sh"]
    assert private["verifier_data.json"] == fixture.original.files["tests/verifier_data.json"]
    assert b"Original math scorer requires Python 3.11" in private["runtime_check.py"]
