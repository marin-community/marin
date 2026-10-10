# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC's grade script scores submissions with the vendored NVARC scorer in the grader image.

See ``local_grader`` for building the image these tests run.
"""

import pytest
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus, ImportRejection, WorkspaceFiles
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.tests.conversion import converted_task
from experiments.post_training.task_curation.tests.local_grader import (
    grade,
    with_verifier_file,
)
from experiments.post_training.task_curation.tests.test_arc import GRID, RECIPES, ROWS, ULTRA_CONTEXT, ultra_row

pytestmark = [pytest.mark.docker, pytest.mark.timeout(300)]

CONFIG_READING_TRANSFORM = b"""import json

def transform(grid):
    return json.load(open("/tests/config.json"))["contract"]["expected_output"]
"""
"""A transform that answers from the hidden record, if it can read it."""


def ultra_task(agent: str) -> TaskSpec:
    result = arc.convert_ultra_arc(ultra_row(agent, test_input=GRID, expected_output=GRID), ULTRA_CONTEXT)
    assert not isinstance(result, ImportRejection)
    return result.task


def tasks() -> dict[str, tuple[TaskSpec, arc.Controls]]:
    tasktrove = {name: (converted_task(RECIPES[name], ROWS[name]), RECIPES[name].controls) for name in ROWS}
    ultra = {
        agent: (ultra_task(agent), arc.ULTRA_ARC_CONTROLS) for agent in (arc.INDUCTIVE_AGENT, arc.TRANSDUCTIVE_AGENT)
    }
    return tasktrove | ultra


@pytest.mark.parametrize(
    "name", ["tasktrove-arc_inductive", "tasktrove-arc_transductive", arc.INDUCTIVE_AGENT, arc.TRANSDUCTIVE_AGENT]
)
def test_golden_control_scores_the_reference_one(name, machines):
    task, controls = tasks()[name]
    assert controls is not None
    report = run_controls(task, controls=controls, machines=machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"golden": CheckStatus.PASS}, report


def test_transform_cannot_read_the_hidden_record(machines):
    task = converted_task(RECIPES["tasktrove-arc_inductive"], ROWS["tasktrove-arc_inductive"])
    result = grade(task, WorkspaceFiles({"/app/solution.py": CONFIG_READING_TRANSFORM}), machines, arc.GRADER_MEMORY_MB)
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0), result
    assert result.diagnostics is not None and "PermissionError" in result.diagnostics["stderr"]


def test_grade_script_fails_rather_than_scoring_when_nvarc_cannot_import(machines):
    task = converted_task(RECIPES["tasktrove-arc_inductive"], ROWS["tasktrove-arc_inductive"])
    broken = inline_resource("skyrl_gym/envs/nemotron_ultra/nvarc.py", b"import package_missing_from_the_grader_image\n")
    result = grade(with_verifier_file(task, broken), arc.tasktrove_golden(task), machines, 2048)
    assert result.status == Outcome.INFRA_ERROR
    assert result.diagnostics is not None and result.diagnostics["exit_code"] != 0
