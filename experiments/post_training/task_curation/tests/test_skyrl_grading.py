# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL grade scripts score replies in the grader image.

Each test converts a fixture row and grades it the way a campaign does, in a fresh container of the
locally built grader image. See ``local_grader`` for building the image these tests run.
"""

import json

import pytest
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.controls import answer_reply, run_controls
from taskcompendium.pipeline.models import CheckStatus
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.tests.conversion import converted_task
from experiments.post_training.task_curation.tests.local_grader import (
    LocalGraderMachines,
    with_verifier_file,
)
from experiments.post_training.task_curation.tests.local_grader import grade as grade_submission
from experiments.post_training.task_curation.tests.test_skyrl import RECIPES, ROWS, SUM_SOLUTION

pytestmark = pytest.mark.docker

GRADING_MEMORY_MB = 5120
NOISY_ADD = '```python\ndef add(a, b):\n    print("x" * 20000)\n    return a + b\n```'
"""A correct function that prints more than the runtime keeps of a grader's stdout."""


def grade(task: TaskSpec, reply: str, machines: LocalGraderMachines) -> GradeResult:
    return grade_submission(task, answer_reply(task, reply), machines, GRADING_MEMORY_MB)


@pytest.mark.parametrize(
    ("name", "control"),
    [
        ("apps", "golden"),
        ("eurus2_code", "empty"),
        ("verifiable_code", "golden"),
        ("gretel_text_to_sql", "golden"),
        ("nemotron_if", "empty"),
        ("rlvr_ifeval", "empty"),
    ],
)
def test_declared_control_passes_in_the_grader_image(name, control, machines):
    """The source's known solution scores 1, or an empty reply scores 0 where it has none."""
    pipeline = RECIPES[name]
    assert pipeline.controls is not None
    report = run_controls(converted_task(pipeline, ROWS[name]), controls=pipeline.controls, machines=machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {control: CheckStatus.PASS}, report.checks


@pytest.mark.parametrize(
    ("name", "scorer"),
    [
        ("apps", "apps_testing_util.py"),
        ("verifiable_code", "livecodebench.py"),
        ("gretel_text_to_sql", "text_to_sql_scoring.py"),
    ],
)
def test_grade_script_fails_rather_than_scoring_when_its_scorer_cannot_import(name, scorer, machines):
    broken = inline_resource(scorer, b"import package_missing_from_the_grader_image\n")
    task = with_verifier_file(converted_task(RECIPES[name], ROWS[name]), broken)
    result = grade(task, f"```python\n{SUM_SOLUTION}\n```", machines)
    assert result.status == Outcome.INFRA_ERROR
    assert result.diagnostics is not None and result.diagnostics["exit_code"] != 0


@pytest.mark.parametrize(
    ("reply", "reward"),
    [("a calm haiku about rain.", 1.0), ("a calm haiku, about rain.", 0.5), ("A calm haiku, about rain.", 0.0)],
)
def test_ifeval_rewards_the_fraction_of_constraints_a_reply_meets(reply, reward, machines):
    row = {
        **ROWS["nemotron_if"],
        "args": {
            "instruction_id_list": ["change_case:english_lowercase", "punctuation:no_comma"],
            "instruction_kwargs": [{}, {}],
        },
    }
    result = grade(converted_task(RECIPES["nemotron_if"], row), reply, machines)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize(
    ("reply", "reward"),
    [
        ("First paragraph.\n* One\n* Two\n\n***\n\nSecond paragraph.", 1.0),
        ("First paragraph.\n* One\n* Two\n\nSecond paragraph.", 0.5),
    ],
)
def test_nemotron_paragraphs_follow_the_prompts_markdown_divider(reply, reward, machines):
    """Nemotron prompts separate paragraphs with ``***``; the divider line is no bullet point."""
    row = {
        **ROWS["nemotron_if"],
        "args": {
            "instruction_id_list": ["length_constraints:number_paragraphs", "detectable_format:number_bullet_lists"],
            "instruction_kwargs": [{"num_paragraphs": 2}, {"num_bullets": 2}],
        },
    }
    result = grade(converted_task(RECIPES["nemotron_if"], row), reply, machines)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize(
    ("descriptor", "passing", "failing"),
    [
        ({"func_name": "verify_paragraph_count", "N": 2}, "First.\n***\nSecond.", "First.\n\nSecond."),
        ({"func_name": "verify_bullet_points", "N": 2}, "**Items:**\n* One\n* Two\n* * *", "* One\n* * *"),
        (
            {"func_name": "validate_sections", "N": 2, "section_splitter": "SECTION"},
            "<<Guide to SECTION 1>>\nSECTION 1\nFirst body.\nSECTION 2\nSecond body.",
            "<<Guide to SECTION 1>>\nSECTION 1\nFirst body.",
        ),
        ({"func_name": "validate_highlighted_sections", "N": 2}, "**First** and **second**.", "**Only one**."),
    ],
)
def test_rlvr_ifeval_grader_accepts_reported_format_cases_and_rejects_ignored_instructions(
    descriptor, passing, failing, machines
):
    row = {**ROWS["rlvr_ifeval"], "ground_truth": json.dumps(descriptor)}
    task = converted_task(RECIPES["rlvr_ifeval"], row)
    positive = grade(task, passing, machines)
    negative = grade(task, failing, machines)
    assert (positive.status, positive.reward) == (Outcome.GRADED, 1.0)
    assert (negative.status, negative.reward) == (Outcome.GRADED, 0.0)


@pytest.mark.parametrize(
    ("reply", "reward"),
    [
        (NOISY_ADD, 1.0),
        ("```python\ndef add(a, b):\n    raise SystemExit(0)\n```", 0.0),
    ],
)
def test_apps_reward_survives_a_program_that_floods_stdout_or_exits(reply, reward, machines):
    row = {**ROWS["apps"], "input_output": json.dumps({"inputs": [[1, 2]], "outputs": [3], "fn_name": "add"})}
    result = grade(converted_task(RECIPES["apps"], row), reply, machines)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


def test_lcb_reward_survives_a_program_that_floods_stdout(machines):
    cases = [{"type": "functional", "fn_name": "add", "input": [1, 2], "output": 3}]
    row = {**ROWS["verifiable_code"], "verification_info": {"language": "python", "test_cases": cases}}
    result = grade(converted_task(RECIPES["verifiable_code"], row), NOISY_ADD, machines)
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
