# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning Gym graders score replies with the grader image's ``reasoning_gym``.

See ``local_grader`` for building the image these tests run.
"""

import dataclasses
from pathlib import Path

import pytest
import reasoning_gym
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus, Reply
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.reasoning_gym import generate
from experiments.post_training.task_curation.tests.conversion import converted_task
from experiments.post_training.task_curation.tests.local_grader import (
    LocalGraderMachines,
    grade,
    local_grader_machines,
    with_verifier_file,
)
from experiments.post_training.task_curation.tests.test_reasoning_gym import GENERATED_ROW, PIPELINES

pytestmark = [pytest.mark.docker, pytest.mark.timeout(300)]

GRADING_MEMORY_MB = 2048
TASKTROVE_ARCHIVE = Path(__file__).parent / "fixtures" / "reasoning_gym.tar.gz"
"""A TaskTrove ``fraction_simplification`` task, as its ``tasks.parquet`` row stores it."""


@pytest.fixture(scope="module")
def machines() -> LocalGraderMachines:
    return local_grader_machines()


def reply(content: str) -> Reply:
    return Reply(TextMessage(role="assistant", content=content))


def generated_task(task_name: str, index: int, **entry_changes) -> tuple[TaskSpec, str]:
    """A generated task the image's reasoning-gym regenerates, and its entry's answer."""
    seed = generate.task_seed(task_name)
    dataset = reasoning_gym.create_dataset(task_name, size=generate.ROWS_PER_TASK, seed=seed)
    entry = generate.encoded(dataset[index])
    generation = {
        "task": task_name,
        "seed": seed,
        "index": index,
        "config": generate.encoded(dataclasses.asdict(dataset.config)),
        "python_hash_seed": 0,
    }
    row = {**GENERATED_ROW, "entry": {**entry, **entry_changes}, "generation": generation}
    return converted_task(PIPELINES["reasoning_gym_generated"], row), entry["answer"]


@pytest.mark.parametrize(("task_name", "index"), [("arc_agi", 0), ("gsm_symbolic", 23)])
def test_generated_grade_scores_against_the_regenerated_entry(task_name, index, machines):
    task, answer = generated_task(task_name, index)
    passing = grade(task, reply(f"Work shown.\nAnswer: {answer}"), machines, GRADING_MEMORY_MB)
    assert (passing.status, passing.reward) == (Outcome.GRADED, 1.0), passing
    failing = grade(task, reply("Answer: definitely wrong"), machines, GRADING_MEMORY_MB)
    assert (failing.status, failing.reward) == (Outcome.GRADED, 0.0), failing


def test_generated_grade_refuses_a_recorded_entry_the_generator_does_not_produce(machines):
    task, answer = generated_task("gsm_symbolic", 23, question="An altered problem")
    result = grade(task, reply(f"Answer: {answer}"), machines, GRADING_MEMORY_MB)
    assert result.status == Outcome.INFRA_ERROR
    assert result.diagnostics is not None and "Regenerated entry differs" in result.diagnostics["stderr"]


def test_generated_grade_fails_rather_than_scoring_when_generate_cannot_import(machines):
    task, answer = generated_task("gsm_symbolic", 23)
    broken = inline_resource("generate.py", b"import package_missing_from_the_grader_image\n")
    result = grade(with_verifier_file(task, broken), reply(f"Answer: {answer}"), machines, GRADING_MEMORY_MB)
    assert result.status == Outcome.INFRA_ERROR
    assert result.diagnostics is not None and result.diagnostics["exit_code"] != 0


def test_tasktrove_archive_grader_scores_with_the_images_reasoning_gym(machines):
    pipeline = PIPELINES["tasktrove-reasoning-gym"]
    row = {"path": "reasoning-gym-d9f956ecd029.tar.gz", "task_binary": TASKTROVE_ARCHIVE.read_bytes()}
    task = converted_task(pipeline, row)
    assert pipeline.controls is not None
    report = run_controls(task, controls=pipeline.controls, machines=machines)
    assert {check.check: check.status for check in report.checks} == {
        "empty": CheckStatus.PASS,
        "golden": CheckStatus.PASS,
    }, report
    wrong = grade(task, reply("$1/2$"), machines, GRADING_MEMORY_MB)
    assert (wrong.status, wrong.reward) == (Outcome.GRADED, 0.0), wrong
