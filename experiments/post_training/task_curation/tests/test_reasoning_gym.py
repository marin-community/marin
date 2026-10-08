# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym tasks are graded by their own task scorer, on regenerated entries where generated."""

import dataclasses
import json
import subprocess
import sys
import tarfile
from fractions import Fraction
from pathlib import Path

import pytest
import reasoning_gym
from reasoning_gym.factory import DATASETS
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import grader_config
from taskcompendium.models import AnswerType, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, NormalizedTask, Reply

from experiments.post_training.task_curation.datasets import reasoning_gym as declarations
from experiments.post_training.task_curation.datasets.reasoning_gym import generate
from experiments.post_training.task_curation.images import REASONING_GYM_IMAGE
from experiments.post_training.task_curation.tests.conversion import convert_row, converted_task, tasktrove_row

PIPELINES = {pipeline.name: pipeline for pipeline in declarations.pipelines()}
GRADE = Path(declarations.__file__).with_name(declarations.GRADE)
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
    "generation": {
        "task": "simple_equations",
        "seed": 128,
        "index": 0,
        "config": {"size": 1000, "seed": 128},
        "python_hash_seed": 0,
    },
    "recorded_pinned_generator_controls": {
        "generator_revision": declarations.GENERATOR_REVISION,
        "positive": {"candidate": "42", "reward": 1.0},
        "negative": {"candidate": "definitely wrong", "reward": 0.0},
        "execution": "Pinned reasoning-gym scorer",
    },
}
ROWS: dict[str, dict] = {"reasoning_gym_generated": GENERATED_ROW, "tasktrove-reasoning-gym": tasktrove_archive()}
GRADERS = {
    "reasoning_gym_generated": (
        (
            "python3",
            "/tests/reasoning_gym_grade.py",
            "/tests/config.json",
            "/app/answer.txt",
            "/logs/verifier/score.json",
        ),
        "/opt/reasoning-gym-generated:/opt/skyrl_gym",
    ),
    "tasktrove-reasoning-gym": (("bash", "/tests/test.sh"), "/opt/reasoning-gym-tasktrove"),
}


@pytest.mark.parametrize("name", sorted(ROWS))
def test_reasoning_gym_task_runs_its_scorer_in_the_grading_image(name):
    task = converted_task(PIPELINES[name], ROWS[name])
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    argv, package = GRADERS[name]
    assert (grader.argv, grader.env["PYTHONPATH"], grader.answer_path) == (argv, package, "/app/answer.txt")
    assert grader.environment.docker_image == REASONING_GYM_IMAGE.reference
    assert task.answer_type == AnswerType.TEXT
    controls = PIPELINES[name].controls
    assert controls is not None and controls.golden is not None
    assert controls.golden(task) == Reply(TextMessage(role="assistant", content="42"))


def test_tasktrove_instruction_asks_for_the_answer_in_the_reply():
    result = convert_row(PIPELINES["tasktrove-reasoning-gym"], tasktrove_archive())
    assert isinstance(result, NormalizedTask)
    assert result.task.context.events == (
        TextMessage(role="user", content="Solve x + 8 = 50. Return ONLY your final answer in the assistant response"),
    )
    assert [change.field for change in result.changes] == ["instruction"]
    unrecognized = convert_row(PIPELINES["tasktrove-reasoning-gym"], tasktrove_archive("Put x in /app/answer.txt."))
    assert isinstance(unrecognized, NormalizedTask)
    message = unrecognized.task.context.events[0]
    assert isinstance(message, TextMessage) and message.content.endswith(declarations.ANSWER_FILE_NOTE)


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
        (
            tasktrove_row(
                {"instruction.md": b"Solve.", "tests/verifier_data.json": json.dumps(TASKTROVE_ENTRY).encode()}
            ),
            ImportFailureKind.UNSUPPORTED,
            "missing_archive_grader",
        ),
    ],
)
def test_tasktrove_rejects_rows_without_an_entry_or_grader(row, kind, reason):
    result = convert_row(PIPELINES["tasktrove-reasoning-gym"], row)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (kind, reason)


def test_generated_row_without_matching_provenance_is_a_converter_error():
    row = {**GENERATED_ROW, "generation": {**GENERATED_ROW["generation"], "task": "basic_arithmetic"}}
    result = convert_row(PIPELINES["reasoning_gym_generated"], row)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.CONVERTER_ERROR, "invalid_generated_reasoning_entry")


def generated_contract(family: str, index: int) -> tuple[dict, dict]:
    """A contract the local reasoning-gym regenerates exactly, and its entry with generator types."""
    seed = generate.GENERATION_SEED + sorted(DATASETS).index(family)
    dataset = reasoning_gym.create_dataset(family, size=generate.ROWS_PER_TASK, seed=seed)
    entry = dataset[index]
    contract = {
        "entry": json.loads(json.dumps(entry, default=generate.json_value)),
        "generation": {
            "task": family,
            "seed": seed,
            "index": index,
            "config": json.loads(json.dumps(dataclasses.asdict(dataset.config), default=generate.json_value)),
            "python_hash_seed": 0,
        },
    }
    return contract, entry


def run_grade(tmp_path: Path, contract: dict, answer: str) -> subprocess.CompletedProcess:
    config, reply, score = tmp_path / "config.json", tmp_path / "answer.txt", tmp_path / "score.json"
    config.write_text(json.dumps({"mode": "generated", "contract": contract}))
    reply.write_text(answer)
    return subprocess.run(
        [sys.executable, str(GRADE), str(config), str(reply), str(score)], capture_output=True, text=True, timeout=300
    )


@pytest.mark.parametrize(("family", "index"), [("arc_agi", 0), ("gsm_symbolic", 23)])
def test_grade_script_scores_against_the_regenerated_entry(tmp_path, family, index):
    contract, entry = generated_contract(family, index)
    passing = run_grade(tmp_path, contract, "Work shown.\nAnswer: " + entry["answer"])
    assert passing.returncode == 0, passing.stderr
    assert json.loads((tmp_path / "score.json").read_text())["reward"] == 1.0
    failing = run_grade(tmp_path, contract, "Answer: definitely wrong")
    assert failing.returncode == 0, failing.stderr
    assert json.loads((tmp_path / "score.json").read_text())["reward"] < 1.0


def test_grade_script_refuses_a_recorded_entry_that_differs_from_the_generator(tmp_path):
    contract, entry = generated_contract("gsm_symbolic", 23)
    contract["entry"]["question"] += " Altered problem"
    result = run_grade(tmp_path, contract, "Answer: " + entry["answer"])
    assert result.returncode != 0
    assert "Regenerated entry differs from the recorded entry" in result.stderr
    assert not (tmp_path / "score.json").exists()


def test_generator_transport_keeps_symbolic_integers_exact():
    dataset = reasoning_gym.create_dataset("codeio", size=1000, seed=59)
    entry = dataset[676]
    transported = json.loads(json.dumps(entry, default=generate.json_value))
    assert type(transported["metadata"]["output_data"]) is int
    assert dataset.score_answer(entry["answer"], transported) == dataset.score_answer(entry["answer"], entry) == 1.0


@pytest.fixture
def noisy_generator_archive(tmp_path) -> StoragePath:
    """A generator that prints at import, construction, generation and scoring."""
    root = tmp_path / "pinned" / "reasoning_gym"
    root.mkdir(parents=True)
    (root / "factory.py").write_text(
        "import os\n"
        "DATASETS = {'composite': object(),\n"
        "    os.environ.get('REASONING_TEST_EXAMPLE_FAMILY', 'puzzle'): object()}\n"
    )
    (root / "__init__.py").write_text(
        """import dataclasses
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
    def __init__(self, name, size, seed):
        self.name = name
        print('construction diagnostic')
        self.config = Config(size, seed)
    def __getitem__(self, index):
        print('.', end='', flush=True)
        if os.environ.get('REASONING_TEST_BAD_WIRE'):
            os.write(1, b'not json\\n')
        metadata = {'source_dataset': self.name, 'output': ((index,),), 'exact_fraction': Fraction(2**60 + 1, 3)}
        if self.name == 'graph_color': metadata['possible_answer'] = {0: 1}
        if self.name == 'propositional_logic': metadata['example_answer'] = 'Example'
        if self.name == 'rubiks_cube': metadata['example_correct_answer'] = 'Example'
        return {'question': f'Puzzle {index}',
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
    )
    archive_path = tmp_path / "generator.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root.parent, arcname="pinned")
    return StoragePath(str(archive_path))


def first_rows(archive: StoragePath, count: int = 1, excluded=EXCLUDED, python_hash_seed: int = 0) -> list[dict]:
    rows = declarations.generated_rows(archive, "pinned-revision", excluded, python_hash_seed)
    try:
        return [next(rows) for _ in range(count)]
    finally:
        rows.close()


def test_generated_rows_keep_generator_output_out_of_the_jsonl(noisy_generator_archive):
    first, second = first_rows(noisy_generator_archive, 2)
    assert (first["entry"]["question"], first["entry"]["answer"], second["entry"]["answer"]) == ("Puzzle 0", "0", "1")
    assert first["entry"]["metadata"]["output"] == [[0]]
    assert first["generation"]["seed"] == second["generation"]["seed"] == 43
    assert (first["generation"]["index"], second["generation"]["index"]) == (0, 1)
    config = first["generation"]["config"]
    assert config["min_time"] == {"python_type": "datetime.time", "isoformat": "01:02:03"}
    assert config["min_date"] == {"python_type": "datetime.date", "isoformat": "1900-01-01"}
    controls = first["recorded_pinned_generator_controls"]
    assert controls["positive"] == {"candidate": "0", "reward": 1.0}
    assert controls["negative"] == {"candidate": "definitely wrong", "reward": 0.0}
    result = convert_row(PIPELINES["reasoning_gym_generated"], first)
    assert isinstance(result, TaskSpec)
    exact = grader_config(TaskSpec.model_validate_json(result.model_dump_json()))["contract"]["entry"]["metadata"]
    fraction = exact["exact_fraction"]
    assert Fraction(fraction["numerator"], fraction["denominator"]) == Fraction(2**60 + 1, 3)


def test_generated_rows_surface_a_malformed_stream(noisy_generator_archive, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_BAD_WIRE", "1")
    with pytest.raises(json.JSONDecodeError):
        first_rows(noisy_generator_archive)


def test_generated_rows_surface_a_generator_that_needs_configuration(noisy_generator_archive):
    with pytest.raises(RuntimeError, match="composite requires explicit components"):
        first_rows(noisy_generator_archive, excluded=())


def test_generated_rows_fail_on_a_scorer_error_for_the_known_answer(noisy_generator_archive, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_BAD_POSITIVE", "1")
    with pytest.raises(RuntimeError, match="Generated positive failure"):
        first_rows(noisy_generator_archive)


def test_generated_rows_record_a_scorer_error_for_the_wrong_answer(noisy_generator_archive, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_BAD_NEGATIVE", "1")
    (row,) = first_rows(noisy_generator_archive)
    negative = row["recorded_pinned_generator_controls"]["negative"]
    assert negative["reward"] is None and negative["scoring_error"]["type"] == "ValueError"
    assert isinstance(convert_row(PIPELINES["reasoning_gym_generated"], row), TaskSpec)


def test_generated_row_without_a_known_answer_has_no_golden(noisy_generator_archive, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_NO_ANSWER", "1")
    (row,) = first_rows(noisy_generator_archive)
    task = convert_row(PIPELINES["reasoning_gym_generated"], row)
    assert isinstance(task, TaskSpec)
    assert declarations.generated_golden(task) is None


@pytest.mark.parametrize(
    ("family", "candidate"),
    [("graph_color", '{"0": 1}'), ("propositional_logic", "Example"), ("rubiks_cube", "Example")],
)
def test_generated_rows_use_the_documented_example_when_the_answer_is_unset(
    noisy_generator_archive, monkeypatch, family, candidate
):
    monkeypatch.setenv("REASONING_TEST_EXAMPLE_FAMILY", family)
    (row,) = first_rows(noisy_generator_archive)
    assert row["entry"]["answer"] is None
    assert row["recorded_pinned_generator_controls"]["positive"] == {"candidate": candidate, "reward": 1.0}


def test_generated_rows_use_the_declared_hash_seed_not_the_parents(noisy_generator_archive, monkeypatch):
    observed = []
    for parent_seed in ("1", "2"):
        monkeypatch.setenv("PYTHONHASHSEED", parent_seed)
        observed.extend(first_rows(noisy_generator_archive, python_hash_seed=6501))
    assert observed[0] == observed[1]
    assert observed[0]["generation"]["python_hash_seed"] == 6501
