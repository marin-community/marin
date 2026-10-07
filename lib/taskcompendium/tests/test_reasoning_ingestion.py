# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import tarfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import reasoning_gym
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.datasets import calendar_tasks, reasoning_tasks
from taskcompendium.datasets.reasoning_gym import generated
from taskcompendium.datasets.reasoning_gym import source as producer
from taskcompendium.datasets.reasoning_gym.generated import generated_rows
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.models import CheckStatus, ImportRejection, RawRow
from taskcompendium.pipeline.verification import verify_task
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import PlainText


@pytest.fixture
def row_source():
    return Source(dataset="test/tasks", revision="1", row="0", importer_revision="1")


def encoded_file(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


def answer_grade(task, answer):
    return grade_task(
        task,
        PlainText(id="plain"),
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer))),
    )


def test_generated_codeio_symbolic_integer_keeps_exact_json_and_native_reward():
    # The pinned49 CodeIO row676 emits a SymPy Integer in private output data.
    dataset = reasoning_gym.create_dataset("codeio", size=1000, seed=59)
    entry = dataset[676]
    transported = json.loads(json.dumps(entry, default=producer._json_value))
    assert type(transported["metadata"]["output_data"]) is int
    assert transported["metadata"]["output_data"] == entry["metadata"]["output_data"]
    assert transported["question"] == entry["question"]
    assert transported["answer"] == entry["answer"]
    assert dataset.score_answer(entry["answer"], entry) == 1.0
    assert dataset.score_answer(entry["answer"], transported) == 1.0


def test_puzzle_ingestion_preserves_order_and_symbolic_coordinates(row_source):
    ordered = reasoning_tasks.normalize_puzzle(
        RawRow(
            "ordered",
            row_source,
            {
                "instruction": "Sort in ASCII order: chair, Defect, donate, Salt. Return a comma-separated list.",
                "files": {
                    "tests/gold.json": encoded_file(
                        {"gold": "Defect, Salt, chair, donate", "answer_type": "ordered_list"}
                    )
                },
            },
        )
    )
    assert isinstance(ordered, TaskSpec)
    assert answer_grade(ordered, "Defect\nSalt\nchair\ndonate").reward == 1.0
    assert answer_grade(ordered, "chair, Defect, donate, Salt").reward == 0.0
    coordinates = reasoning_tasks.normalize_puzzle(
        RawRow(
            "coords",
            row_source,
            {
                "instruction": "Return the point halfway between (1, 2) and (3, 4).",
                "files": {"tests/gold.json": encoded_file({"gold": "(2, 3)", "answer_type": "coords"})},
            },
        )
    )
    assert isinstance(coordinates, TaskSpec)
    assert answer_grade(coordinates, "(2.0, 3.0)").reward == 1.0
    assert answer_grade(coordinates, "(3, 2)").reward == 0.0


def test_reasoning_ingestion_preserves_upstream_contract_without_local_execution(row_source):
    task = reasoning_tasks.normalize_reasoning(
        RawRow(
            "equation",
            row_source,
            {
                "instruction": "Solve x + 8 = 50. Reply with x.",
                "verifier_data": {"answer": "42", "metadata": {"source_dataset": "simple_equations"}},
            },
        )
    )
    assert isinstance(task, TaskSpec)
    assert grader_config(task)["contract"]["entry"]["answer"] == "42"
    result = answer_grade(task, "42")
    assert (result.status, result.reward) == (Outcome.UNAVAILABLE, None)
    assert all(check.status == CheckStatus.UNSUPPORTED for check in reasoning_tasks.reasoning_checks(task).checks)


def test_calendar_ingestion_preserves_original_command_and_private_archive(row_source):
    fixture = Path(__file__).parents[3] / "experiments/post_training/tasktrove/fixtures/agent_calendar.tar.gz"
    decoded = unpack_task_binary(
        {"task_binary": fixture.read_bytes(), "path": "agent_calendar/fixture"}, StoragePath("/tmp")
    )
    row = RawRow("calendar", row_source, decoded)
    task = calendar_tasks.normalize(row)
    assert isinstance(task, TaskSpec)
    command = json.loads(task.verifier.parameters_json)
    assert command["argv"] == ["bash", "/tests/test.sh"]
    assert command["result_format"] == "reward_file"
    assert [(check.check, check.status) for check in verify_task(task)] == [("runtime", CheckStatus.UNSUPPORTED)]
    assert {resource.path for resource in task.resources.verifier} >= {"test.sh", "verifier.py", "verifier_data.json"}
    assert all(resource.path != "solution/answer.json" for resource in task.resources.worker)
    missing_command = calendar_tasks.normalize(replace(row, data={**row.data, "files": {}}))
    assert isinstance(missing_command, ImportRejection)
    assert missing_command.reason == "missing_original_calendar_command"


@pytest.fixture
def noisy_generator_archive(tmp_path):
    """A native generator that prints at import, construction, generation and scoring."""
    root = tmp_path / "pinned" / "reasoning_gym"
    root.mkdir(parents=True)
    (root / "factory.py").write_text(
        "import os\n"
        "DATASETS = {'composite': object(),\n"
        "    os.environ.get('REASONING_TEST_WITNESS_FAMILY', 'native'): object()}\n"
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
        metadata = {'source_dataset': self.name, 'output': ((index,),), 'hash_order': ''.join(set('abcdefghijk')),
                    'exact_fraction': Fraction(2**60 + 1, 3)}
        if self.name == 'graph_color': metadata['possible_answer'] = {0: 1}
        if self.name == 'propositional_logic': metadata['example_answer'] = 'Native witness'
        if self.name == 'rubiks_cube': metadata['example_correct_answer'] = 'Native witness'
        return {'question': f'Puzzle {index}',
                'answer': None if os.environ.get('REASONING_TEST_NO_POSITIVE') or self.name != 'native' else str(index),
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
        expected = '{\"0\": 1}' if name == 'graph_color' else 'Native witness' if name != 'native' else entry['answer']
        return float(answer == expected and isinstance(entry['metadata']['output'], tuple)
                     and isinstance(entry['metadata']['exact_fraction'], Fraction))
    return score
"""
    )
    archive_path = tmp_path / "generator.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(root.parent, arcname="pinned")
    return StoragePath(str(archive_path))


def test_generated_reasoning_rows_keep_native_diagnostics_out_of_jsonl(noisy_generator_archive, row_source):
    rows = generated_rows(
        noisy_generator_archive,
        "pinned-revision",
        excluded_generators=(("composite", "Requires explicit component configuration"),),
        python_hash_seed=0,
    )
    try:
        first, second = next(rows), next(rows)
    finally:
        rows.close()
    assert first["entry"]["question"] == "Puzzle 0"
    assert first["entry"]["answer"] == "0"
    assert first["entry"]["metadata"]["source_dataset"] == "native"
    assert first["entry"]["metadata"]["output"] == [[0]]
    assert second["entry"]["answer"] == "1"
    assert first["generation"]["seed"] == second["generation"]["seed"] == 43
    assert first["generation"]["index"] == 0 and second["generation"]["index"] == 1
    config = first["generation"]["config"]
    assert config["min_time"] == {"python_type": "datetime.time", "isoformat": "01:02:03"}
    assert config["min_date"] == {"python_type": "datetime.date", "isoformat": "1900-01-01"}
    controls = first["recorded_pinned_generator_controls"]
    assert controls["positive"] == {"candidate": "0", "reward": 1.0}
    assert controls["negative"] == {"candidate": "definitely wrong", "reward": 0.0}
    task = generated.normalize(RawRow("generated", row_source, first))
    assert isinstance(task, TaskSpec)
    reloaded = TaskSpec.model_validate_json(task.model_dump_json())
    preserved = grader_config(reloaded)["contract"]["entry"]["metadata"]["exact_fraction"]
    assert preserved["python_type"] == "fractions.Fraction"
    assert Fraction(preserved["numerator"], preserved["denominator"]) == Fraction(2**60 + 1, 3)


def test_generated_reasoning_rows_do_not_hide_malformed_protocol(noisy_generator_archive, monkeypatch):
    monkeypatch.setenv("REASONING_TEST_BAD_WIRE", "1")
    rows = generated_rows(
        noisy_generator_archive,
        "pinned-revision",
        excluded_generators=(("composite", "Requires explicit component configuration"),),
        python_hash_seed=0,
    )
    try:
        with pytest.raises(json.JSONDecodeError):
            next(rows)
    finally:
        rows.close()


def test_generated_reasoning_rows_preserve_unconfigured_native_constructor_failure(noisy_generator_archive):
    rows = generated_rows(noisy_generator_archive, "pinned-revision", excluded_generators=(), python_hash_seed=0)
    try:
        with pytest.raises(RuntimeError, match="composite requires explicit components"):
            next(rows)
    finally:
        rows.close()


@pytest.mark.parametrize("scoring_failure", ["negative", "positive"])
def test_generated_reasoning_distinguishes_synthetic_negative_from_native_positive_failure(
    noisy_generator_archive, monkeypatch, row_source, scoring_failure
):
    monkeypatch.setenv("REASONING_TEST_BAD_" + scoring_failure.upper(), "1")
    rows = generated_rows(
        noisy_generator_archive,
        "pinned-revision",
        excluded_generators=(("composite", "Requires components"),),
        python_hash_seed=0,
    )
    try:
        if scoring_failure == "positive":
            with pytest.raises(RuntimeError, match="Generated positive failure"):
                next(rows)
            return
        row = next(rows)
    finally:
        rows.close()
    controls = row["recorded_pinned_generator_controls"]
    assert controls["positive"] == {"candidate": "0", "reward": 1.0}
    assert controls["negative"]["reward"] is None
    assert controls["negative"]["scoring_error"]["type"] == "ValueError"
    task = generated.normalize(RawRow("generated", row_source.model_copy(update={"revision": "pinned-revision"}), row))
    assert isinstance(task, TaskSpec)
    checks = generated.verification_report(task).checks
    assert (
        next(check.status for check in checks if check.check == "recorded_pinned_generator_controls/negative")
        == CheckStatus.UNSUPPORTED
    )


def test_generated_reasoning_missing_positive_witness_keeps_valid_task_and_negative_coverage(
    noisy_generator_archive, monkeypatch, row_source
):
    monkeypatch.setenv("REASONING_TEST_NO_POSITIVE", "1")
    rows = generated_rows(
        noisy_generator_archive,
        "pinned-revision",
        excluded_generators=(("composite", "Requires components"),),
        python_hash_seed=0,
    )
    try:
        row = next(rows)
    finally:
        rows.close()
    assert row["entry"]["answer"] is None
    controls = row["recorded_pinned_generator_controls"]
    assert controls["positive"] == {"candidate": None, "reward": None}
    assert controls["negative"]["reward"] == 0.0
    task = generated.normalize(
        RawRow("multiple-solutions", row_source.model_copy(update={"revision": "pinned-revision"}), row)
    )
    assert isinstance(task, TaskSpec)
    statuses = {check.check: check.status for check in generated.verification_report(task).checks}
    assert statuses["recorded_pinned_generator_controls/positive"] == CheckStatus.SKIPPED
    assert statuses["recorded_pinned_generator_controls/negative"] == CheckStatus.PASS


@pytest.mark.parametrize(
    "family,candidate",
    [("graph_color", '{"0": 1}'), ("propositional_logic", "Native witness"), ("rubiks_cube", "Native witness")],
)
def test_generated_reasoning_uses_documented_native_witness_without_replacing_answer(
    noisy_generator_archive, monkeypatch, family, candidate
):
    monkeypatch.setenv("REASONING_TEST_WITNESS_FAMILY", family)
    rows = generated_rows(
        noisy_generator_archive,
        "pinned-revision",
        excluded_generators=(("composite", "Requires components"),),
        python_hash_seed=0,
    )
    try:
        row = next(rows)
    finally:
        rows.close()
    assert row["entry"]["answer"] is None
    assert row["entry"]["metadata"]["source_dataset"] == family
    assert row["recorded_pinned_generator_controls"]["positive"] == {"candidate": candidate, "reward": 1.0}


def test_generated_reasoning_explicit_hash_seed_is_independent_of_parent_process_environment(
    noisy_generator_archive, monkeypatch
):
    observed = []
    for parent_seed in ("1", "2"):
        monkeypatch.setenv("PYTHONHASHSEED", parent_seed)
        rows = generated_rows(
            noisy_generator_archive,
            "pinned-revision",
            excluded_generators=(("composite", "Requires components"),),
            python_hash_seed=6501,
        )
        try:
            observed.append(next(rows))
        finally:
            rows.close()
    assert observed[0] == observed[1]
    assert observed[0]["generation"]["python_hash_seed"] == 6501
