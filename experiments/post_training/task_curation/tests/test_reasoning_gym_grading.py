# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym binds the original package scorer and source archive."""

import dataclasses
import json
from fractions import Fraction
from functools import partial
from pathlib import Path

import pytest
import reasoning_gym
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.datasets import reasoning_tasks
from taskcompendium.datasets.nemotron_ultra import normalization
from taskcompendium.datasets.reasoning_gym.source import _json_value
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.models import Source, TaskSpec
from taskcompendium.native_grader import NativeCommandSpec
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.reasoning_gym import binding as native
from experiments.post_training.task_curation.datasets.reasoning_gym import runner as reasoning_gym_runner

IMAGE = "fixture@sha256:" + "1" * 64


def ultra_row():
    return RawRow(
        "ultra",
        Source(dataset="fixture", revision="pinned", row="1", importer_revision="1"),
        {
            "dataset": "fixture",
            "agent_ref": {"name": "reasoning_gym_simple_agent"},
            "responses_create_params": {"input": [{"role": "user", "content": "Solve x+8=50."}]},
            "question": "Solve x+8=50.",
            "answer": "42",
            "metadata": {"source_dataset": "simple_equations"},
        },
    )


def tasktrove_row():
    fixture = Path(__file__).parents[4] / "experiments/post_training/tasktrove/fixtures/nemotron_reasoning.tar.gz"
    data = unpack_task_binary({"task_binary": fixture.read_bytes(), "path": "fixture/reasoning"}, StoragePath("/tmp"))
    return RawRow(
        "trove",
        Source(dataset="open-thoughts/TaskTrove", revision=native.TASKTROVE_REVISION, row="0", importer_revision="2"),
        data,
    )


def test_ultra_binding_declares_image_scorer_and_preserves_source_contract():
    row = ultra_row()
    result = native.normalize_isolated(
        row,
        image=IMAGE,
        contract=native.ReasoningContract.ULTRA,
        package_path="/opt/reasoning-gym-ultra",
        normalize_task=partial(normalization.normalize, selector="fixture", family="reasoning-gym"),
    )
    assert isinstance(result, NormalizedTask)
    task = result.task
    command = NativeCommandSpec.model_validate_json(task.verifier.parameters_json)
    assert command.argv == ("python3", "/tests/grade.py")
    assert command.result_format == "score_json"
    assert task.output_paths == ("/app/answer.txt",)
    private = {item.path: resource_bytes(item) for item in task.resources.verifier}
    assert set(private) == {"grade.py", "reasoning_contract.json"}
    assert json.loads(private["reasoning_contract.json"])["contract"]["metadata"] == row.data["metadata"]
    assert not task.resources.worker and not task.resources.all


def test_tasktrove_binding_runs_original_archive_command_without_synthetic_config():
    row = tasktrove_row()
    task = native.normalize_isolated(
        row,
        image=IMAGE,
        contract=native.ReasoningContract.TASKTROVE,
        package_path="/opt/reasoning-gym-tasktrove",
        normalize_task=reasoning_tasks.normalize_reasoning,
    )
    assert isinstance(task, TaskSpec)
    command = NativeCommandSpec.model_validate_json(task.verifier.parameters_json)
    assert command.argv == ("bash", "/tests/test.sh")
    assert command.env == {"PYTHONPATH": "/opt/reasoning-gym-tasktrove"}
    assert task.output_paths == ("/app/answer.txt",)
    private = {item.path: resource_bytes(item) for item in task.resources.verifier}
    assert "config.json" not in private
    assert private["test.sh"] == reasoning_tasks.snapshot_file(row, "tests/test.sh")
    assert private["verifier.py"] == reasoning_tasks.snapshot_file(row, "tests/verifier.py")
    assert private["verifier_data.json"] == reasoning_tasks.snapshot_file(row, "tests/verifier_data.json")


@pytest.mark.parametrize("family,index", [("arc_agi", 0), ("gsm_symbolic", 23)])
def test_generated_regeneration_recovers_native_types_and_rejects_changed_transport(family, index):
    dataset = reasoning_gym.create_dataset(
        family, size=1000, seed=42 + sorted(reasoning_gym.factory.DATASETS).index(family)
    )
    original = dataset[index]
    contract = {
        "entry": json.loads(json.dumps(original, default=_json_value)),
        "generation": {
            "task": family,
            "seed": dataset.config.seed,
            "index": index,
            "config": json.loads(json.dumps(dataclasses.asdict(dataset.config), default=_json_value)),
            "python_hash_seed": 0,
        },
    }
    inputs = reasoning_gym_runner.native_input(
        reasoning_gym, "generated", contract, "work Answer: " + original["answer"]
    )
    assert reasoning_gym.get_score_answer_fn(inputs.task_name)(inputs.candidate, inputs.entry) == 1.0
    if family == "arc_agi":
        assert isinstance(inputs.entry["metadata"]["output"], tuple)
        assert reasoning_gym.get_score_answer_fn(inputs.task_name)(inputs.candidate, contract["entry"]) < 1.0
    else:
        assert isinstance(inputs.entry["metadata"]["variables"]["serving_fraction"], Fraction)
    contract["entry"]["question"] += " Altered problem"
    with pytest.raises(ValueError):
        reasoning_gym_runner.native_input(reasoning_gym, "generated", contract, inputs.candidate)


def test_runner_transport_preserves_codeio_symbolic_integer_reward():
    dataset = reasoning_gym.create_dataset("codeio", size=1000, seed=59)
    entry = dataset[676]
    transported = json.loads(json.dumps(entry, default=reasoning_gym_runner.json_value))
    assert type(transported["metadata"]["output_data"]) is int
    assert dataset.score_answer(entry["answer"], transported) == dataset.score_answer(entry["answer"], entry) == 1.0
