# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym tasks are graded by their own task scorer, on regenerated entries where generated."""

import json
import tomllib
import zipfile
from fractions import Fraction
from typing import cast

import pytest
import reasoning_gym
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import grader_config
from taskcompendium.models import TaskSpec, TextMessage, verifyit_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, NormalizedTask, Reply
from taskcompendium.runtime.resources import resource_bytes
from verifyit.grade import Status, grade

from experiments.post_training.task_curation.datasets.environments import VERIFYIT_PACKAGE
from experiments.post_training.task_curation.datasets.reasoning_gym import generate
from experiments.post_training.task_curation.datasets.reasoning_gym import tasks as declarations
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.tasktrove.harbor_export import harbor_payload
from experiments.post_training.task_curation.tests.conversion import (
    convert_row,
    converted_task,
    fixture_context,
    tasktrove_row,
)

RECIPES = {source.name: cast(CurationRecipe, source.config) for source in declarations.sources()}
EXCLUDED = (("composite", "Requires explicit component configuration"),)
TASKTROVE_INSTRUCTION = "Solve x + 8 = 50. Write ONLY your final answer to **`/app/answer.txt`**"
TASKTROVE_ENTRY = {"question": "Solve x + 8 = 50.", "answer": "42", "metadata": {"source_dataset": "simple_equations"}}
GRADER_FILES = {
    "task.toml": b"[verifier]\ntimeout_sec = 600.0\n",
    "tests/test.sh": b"#!/bin/bash\npython3 /tests/verifier.py\n",
    "tests/verifier.py": b"print(1)\n",
}


def tasktrove_archive(instruction: str = TASKTROVE_INSTRUCTION, entry: dict | None = None, **files: bytes) -> dict:
    return tasktrove_row(
        {
            "instruction.md": instruction.encode(),
            "tests/verifier_data.json": json.dumps(TASKTROVE_ENTRY if entry is None else entry).encode(),
            **GRADER_FILES,
            **files,
        }
    )


GENERATED_ROW = {
    "entry": TASKTROVE_ENTRY,
    "reproducible": True,
    "generation": {
        "task": "simple_equations",
        "seed": 128,
        "index": 0,
        "config": {"size": 1000, "seed": 128},
        "python_hash_seed": 0,
    },
    "recorded_pinned_generator_controls": {
        "generator_version": declarations.GENERATOR_VERSION,
        "positive": {"candidate": "42", "reward": 1.0},
        "negative": {"candidate": "definitely wrong", "reward": 0.0},
        "execution": "Pinned reasoning-gym scorer",
    },
}
ROWS: dict[str, dict] = {"reasoning_gym_generated": GENERATED_ROW, "tasktrove-reasoning-gym": tasktrove_archive()}


@pytest.mark.parametrize(
    ("row", "kind", "reason"),
    [
        (tasktrove_row({"instruction.md": b"Solve.", **GRADER_FILES}), ImportFailureKind.SOURCE_DEFECT, "missing_input"),
        (tasktrove_archive(entry={"answer": "42", "metadata": {}}), ImportFailureKind.UNSUPPORTED, "missing_scorer"),
        (
            tasktrove_archive(entry={**TASKTROVE_ENTRY, "answer": 42}),
            ImportFailureKind.SOURCE_DEFECT,
            "invalid_entry",
        ),
    ],
)
def test_tasktrove_rejects_rows_without_an_entry_or_grader(row, kind, reason):
    result = convert_row(RECIPES["tasktrove-reasoning-gym"], row)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (kind, reason)


def test_generated_row_without_matching_provenance_is_a_converter_error():
    row = {**GENERATED_ROW, "generation": {**GENERATED_ROW["generation"], "task": "basic_arithmetic"}}
    result = convert_row(RECIPES["reasoning_gym_generated"], row)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.CONVERTER_ERROR, "invalid_generated_reasoning_entry")


def test_generated_row_a_fresh_dataset_does_not_reproduce_is_rejected():
    result = convert_row(RECIPES["reasoning_gym_generated"], {**GENERATED_ROW, "reproducible": False})
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.SOURCE_DEFECT, "irreproducible_entry")


def test_generator_transport_keeps_symbolic_integers_exact():
    dataset = reasoning_gym.create_dataset("codeio", size=1000, seed=59)
    entry = dataset[676]
    transported = generate.encoded(entry)
    assert type(transported["metadata"]["output_data"]) is int
    assert dataset.score_answer(entry["answer"], transported) == dataset.score_answer(entry["answer"], entry) == 1.0


@pytest.fixture
def noisy_generator_wheel(tmp_path) -> StoragePath:
    """A generator wheel that prints at import, construction, generation and scoring."""
    factory = (
        "import os\n"
        "DATASETS = {'composite': object(),\n"
        "    os.environ.get('REASONING_TEST_EXAMPLE_FAMILY', 'puzzle'): object(),\n"
        "    **{name: object() for name in os.environ.get('REASONING_TEST_MORE_FAMILIES', '').split()}}\n"
    )
    package = """import dataclasses
import os
from datetime import date, time
from fractions import Fraction
print('import diagnostic')

@dataclasses.dataclass
class Config:
    size: int
    seed: int
    min_time: time = time(1, 2, 3)
    min_date: date = date(1900, 1, 1)

class Dataset:
    built = 0

    def __init__(self, name, size, seed):
        self.name = name
        print('construction diagnostic')
        self.config = Config(size, seed)
        Dataset.built += 1
        self.generation = Dataset.built if os.environ.get('REASONING_TEST_DRIFT') else 0
    def __getitem__(self, index):
        print('.', end='', flush=True)
        if os.environ.get('REASONING_TEST_BAD_WIRE'):
            os.write(1, b'not json\\n')
        metadata = {'source_dataset': self.name, 'output': ((index,),), 'exact_fraction': Fraction(2**60 + 1, 3)}
        if self.name == 'graph_color': metadata['possible_answer'] = {0: 1}
        if self.name == 'propositional_logic': metadata['example_answer'] = 'Example'
        if self.name == 'rubiks_cube': metadata['example_correct_answer'] = 'Example'
        return {'question': f'Puzzle {index}' + (f' (instance {self.generation})' if self.generation else ''),
                'answer': None if os.environ.get('REASONING_TEST_NO_ANSWER') or self.name != 'puzzle' else str(index),
                'metadata': metadata}

def create_dataset(name, **kwargs):
    if name == "composite":
        raise ValueError("composite requires explicit components")
    return Dataset(name, **kwargs)

def get_score_answer_fn(name):
    def score(answer, entry):
        print('score diagnostic')
        if os.environ.get('REASONING_TEST_BAD_NEGATIVE') and answer == 'definitely wrong':
            raise ValueError('Malformed synthetic negative')
        if os.environ.get('REASONING_TEST_BAD_POSITIVE') and answer == entry['answer']:
            raise ValueError('Generated positive failure')
        expected = '{\"0\": 1}' if name == 'graph_color' else 'Example' if name != 'puzzle' else entry['answer']
        return float(answer == expected and isinstance(entry['metadata']['output'], tuple)
                     and isinstance(entry['metadata']['exact_fraction'], Fraction))
    return score
"""
    wheel_path = tmp_path / declarations.GENERATOR_ARCHIVE
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr("reasoning_gym/__init__.py", package)
        wheel.writestr("reasoning_gym/factory.py", factory)
    return StoragePath(str(wheel_path))


def first_rows(wheel: StoragePath, count: int = 1, excluded=EXCLUDED, python_hash_seed: int = 0) -> list[dict]:
    rows = declarations.generated_rows(wheel, "pinned-version", excluded, python_hash_seed, 0, 1, None)
    try:
        return [next(rows)[1] for _ in range(count)]
    finally:
        rows.close()


def test_generated_parts_together_yield_the_rows_of_one_run(noisy_generator_wheel, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_MORE_FAMILIES", "riddle sudoku")

    def rows(parts: int) -> dict[int, dict]:
        return {
            index: row
            for part in range(parts)
            for index, row in declarations.generated_rows(
                noisy_generator_wheel, "pinned-version", EXCLUDED, 0, part, parts, None
            )
        }

    whole = rows(1)
    assert sorted(whole) == list(range(3 * generate.ROWS_PER_TASK))
    # Locators cycle the sorted registry, one entry of each task per round.
    assert [whole[index]["generation"]["task"] for index in range(4)] == ["puzzle", "riddle", "sudoku", "puzzle"]
    assert rows(2) == whole


def test_sampled_rows_are_the_rows_of_a_whole_generation(noisy_generator_wheel, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_MORE_FAMILIES", "riddle sudoku")
    context = ConversionContext({}, None)
    whole = dict(declarations.generated_rows(noisy_generator_wheel, "pinned-version", EXCLUDED, 0, 0, 1, None))
    parts = declarations.GeneratedRows("pinned-version", EXCLUDED, 0, 2)
    assert parts.size(noisy_generator_wheel, context) == len(whole)
    sample = frozenset({1, 2, 1500, len(whole) - 1})
    sampled = {
        index: row for part in range(parts.count) for index, row in parts(noisy_generator_wheel, context, part, sample)
    }
    assert sampled == {index: whole[index] for index in sample}


def test_generated_rows_keep_generator_output_out_of_the_jsonl(noisy_generator_wheel):
    first, second = first_rows(noisy_generator_wheel, 2)
    assert (first["entry"]["question"], first["entry"]["answer"], second["entry"]["answer"]) == ("Puzzle 0", "0", "1")
    assert first["reproducible"] and second["reproducible"]
    assert first["entry"]["metadata"]["output"] == [[0]]
    assert first["generation"]["seed"] == second["generation"]["seed"] == 43
    assert (first["generation"]["index"], second["generation"]["index"]) == (0, 1)
    config = first["generation"]["config"]
    assert config["min_time"] == {"python_type": "datetime.time", "isoformat": "01:02:03"}
    assert config["min_date"] == {"python_type": "datetime.date", "isoformat": "1900-01-01"}
    controls = first["recorded_pinned_generator_controls"]
    assert controls["positive"] == {"candidate": "0", "reward": 1.0}
    assert controls["negative"] == {"candidate": "definitely wrong", "reward": 0.0}
    result = convert_row(RECIPES["reasoning_gym_generated"], first)
    assert isinstance(result, TaskSpec)
    exact = grader_config(TaskSpec.model_validate_json(result.model_dump_json()))["contract"]["entry"]["metadata"]
    fraction = exact["exact_fraction"]
    assert Fraction(fraction["numerator"], fraction["denominator"]) == Fraction(2**60 + 1, 3)


def test_generated_rows_record_entries_a_fresh_dataset_does_not_reproduce(noisy_generator_wheel, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_DRIFT", "1")
    (row,) = first_rows(noisy_generator_wheel)
    assert row["reproducible"] is False
    result = convert_row(RECIPES["reasoning_gym_generated"], row)
    assert isinstance(result, ImportRejection) and result.reason == "irreproducible_entry"


def test_generated_rows_surface_a_malformed_stream(noisy_generator_wheel, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_BAD_WIRE", "1")
    with pytest.raises(json.JSONDecodeError):
        first_rows(noisy_generator_wheel)


def test_generated_rows_surface_a_generator_that_needs_configuration(noisy_generator_wheel):
    with pytest.raises(RuntimeError, match="composite requires explicit components"):
        first_rows(noisy_generator_wheel, excluded=())


def test_generated_rows_fail_on_a_scorer_error_for_the_known_answer(noisy_generator_wheel, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_BAD_POSITIVE", "1")
    with pytest.raises(RuntimeError, match="Generated positive failure"):
        first_rows(noisy_generator_wheel)


def test_generated_rows_record_a_scorer_error_for_the_wrong_answer(noisy_generator_wheel, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_BAD_NEGATIVE", "1")
    (row,) = first_rows(noisy_generator_wheel)
    negative = row["recorded_pinned_generator_controls"]["negative"]
    assert negative["reward"] is None and negative["scoring_error"]["type"] == "ValueError"
    assert isinstance(convert_row(RECIPES["reasoning_gym_generated"], row), TaskSpec)


def test_generated_row_without_a_known_answer_has_no_golden(noisy_generator_wheel, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_NO_ANSWER", "1")
    (row,) = first_rows(noisy_generator_wheel)
    task = convert_row(RECIPES["reasoning_gym_generated"], row)
    assert isinstance(task, TaskSpec)
    assert declarations.generated_golden(task) is None


@pytest.mark.parametrize(
    ("family", "candidate"),
    [("graph_color", '{"0": 1}'), ("propositional_logic", "Example"), ("rubiks_cube", "Example")],
)
def test_generated_rows_use_the_documented_example_when_the_answer_is_unset(
    noisy_generator_wheel, monkeypatch, family, candidate
):
    monkeypatch.setenv("REASONING_TEST_EXAMPLE_FAMILY", family)
    (row,) = first_rows(noisy_generator_wheel)
    assert row["entry"]["answer"] is None
    assert row["recorded_pinned_generator_controls"]["positive"] == {"candidate": candidate, "reward": 1.0}


def test_generated_rows_use_the_declared_hash_seed_not_the_parents(noisy_generator_wheel, monkeypatch):
    observed = []
    for parent_seed in ("1", "2"):
        monkeypatch.setenv("PYTHONHASHSEED", parent_seed)
        observed.extend(first_rows(noisy_generator_wheel, python_hash_seed=6501))
    assert observed[0] == observed[1]
    assert observed[0]["generation"]["python_hash_seed"] == 6501


def test_tasktrove_reasoning_gym_preserves_fractional_reward(tmp_path):
    pipeline = RECIPES["tasktrove-reasoning-gym"]
    converted = cast(NormalizedTask, convert_row(pipeline, tasktrove_archive()))
    task = converted.task
    prompt = task.context.events[0].content
    assert TASKTROVE_ENTRY["question"] in prompt
    assert "/app/answer.txt" not in prompt
    assert converted.changes[0].original == TASKTROVE_INSTRUCTION
    assert task.grader.environment == fixture_context(pipeline).grader_environment
    tests = tmp_path / "tests"
    tests.mkdir()
    for resource in task.resources.verifier:
        (tests / resource.path).write_bytes(resource_bytes(resource))
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("x = 42")
    verdict = grade(verifyit_spec(task.grader), tests, workspace)
    assert (verdict.status, verdict.reward) == (Status.SCORED, 1 / 3)
    assert declarations.tasktrove_golden(task) == Reply(TextMessage(role="assistant", content="42"))


def test_tasktrove_unrecognized_delivery_keeps_instruction_and_discloses_capture_path():
    instruction = "Solve x + 8 = 50. Put x in /app/answer.txt."
    converted = cast(NormalizedTask, convert_row(RECIPES["tasktrove-reasoning-gym"], tasktrove_archive(instruction)))
    prompt = converted.task.context.events[0].content
    assert prompt.startswith(instruction)
    assert "/app/answer.txt" in prompt[len(instruction) :]
    assert converted.changes[0].original == instruction


def test_tasktrove_reasoning_gym_exports_its_source_environment():
    task = converted_task(
        RECIPES["tasktrove-reasoning-gym"],
        tasktrove_archive(**{"environment/Dockerfile": b"FROM python:3.11-slim\nWORKDIR /app\n"}),
    )
    payload = harbor_payload(
        {
            "task_json": task.model_dump_json(),
            "source_row": declarations.TASKTROVE_CONFIG + "/fixture",
            "original_path": "fixture",
        },
        grader_image=None,
        family="reasoning-gym",
        fallback_actor_image="unused",
        verifyit_package_root=VERIFYIT_PACKAGE,
    )
    config = tomllib.loads(payload.files["task.toml"].decode())
    assert config["verifier"]["environment_mode"] == "shared"
    assert payload.files["environment/Dockerfile"].startswith(b"FROM python:3.11-slim\nWORKDIR /app\n")
    assert "tests/Dockerfile" not in payload.files


@pytest.mark.parametrize("dataset", ["arc_agi", "rearc"])
def test_tasktrove_reasoning_gym_grades_json_grid_entries(dataset, tmp_path):
    entry = {
        "question": "Fill the grid.",
        "answer": "1 2\n3 4",
        "metadata": {"source_dataset": dataset, "output": [[1, 2], [3, 4]]},
    }
    task = converted_task(RECIPES["tasktrove-reasoning-gym"], tasktrove_archive(entry=entry))
    tests = tmp_path / "tests"
    tests.mkdir()
    for resource in task.resources.verifier:
        (tests / resource.path).write_bytes(resource_bytes(resource))
    workspace = tmp_path / "app"
    workspace.mkdir()

    for candidate, expected in [(entry["answer"], 1.0), ("1 2\n3 5", 0.05), ("not a grid", 0.0)]:
        (workspace / "answer.txt").write_text(candidate)
        verdict = grade(verifyit_spec(task.grader), tests, workspace)
        assert (verdict.status, verdict.reward) == (Status.SCORED, expected)
