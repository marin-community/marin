# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Control submissions check a grader through the grading path that rollouts use."""

import json

import pytest
from verifyit.spec import SchemaFormat

from taskcompendium.convert.answers import (
    exact_answer_task,
    json_schema_task,
    math_answer_task,
    mcq_task,
    numeric_answer_task,
)
from taskcompendium.models import AnswerType, NoGrader, ResourceGroups, SessionGrader, Source, TaskSpec
from taskcompendium.pipeline.controls import answer_reply, control_suite, reference_reply, wrong_reply
from taskcompendium.pipeline.models import CheckStatus, Controls, OracleCommand, RawRow, WorkspaceFiles
from taskcompendium.runtime.resources import inline_resource

from .pipeline_stages import FixtureGradingMachines, UnavailableImages, script_graded

PASS, FAIL, SKIPPED = CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.SKIPPED
ROW = RawRow("task", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})
REFERENCE_CONTROLS = Controls(golden=reference_reply, negative=wrong_reply)


def checks(task: TaskSpec, controls: Controls, machines: FixtureGradingMachines | None = None) -> dict[str, CheckStatus]:
    return {check.check: check.status for check in control_suite(controls, machines).run(task).checks}


def in_process_task(kind: str) -> TaskSpec:
    if kind == "math":
        task = math_answer_task(ROW, prompt="What is half of one?", answer=r"\boxed{\frac{1}{2}}")
    elif kind == "numeric":
        task = numeric_answer_task(ROW, prompt="What is 3 + 4?", answer="7", tolerance_abs=0, tolerance_rel=0)
    elif kind == "mcq":
        task = mcq_task(ROW, prompt="Which is blue? A. grass B. snow C. sky D. coal", answer="c", options=4)
    elif kind == "exact":
        task = exact_answer_task(ROW, prompt="Name the capital of France.", answers=("Paris",), ignore_case=True)
    else:
        task = json_schema_task(
            ROW, prompt="Return a JSON object.", schema=json.dumps({"type": "object"}), schema_format=SchemaFormat.JSON
        )
    assert isinstance(task, TaskSpec)
    return task


def answer_task() -> TaskSpec:
    """A conversation answer graded in its image by comparing /app/answer.txt with the public input."""
    task = in_process_task("numeric").model_copy(
        update={"resources": ResourceGroups(worker=(inline_resource("data/expected.txt", b"7"),))}
    )
    return script_graded(task, b'test "$(cat answer.txt)" = 7\n')


def file_task() -> TaskSpec:
    """A file answer at /app/solution.txt, with an oracle script that derives it from a worker file."""
    task = in_process_task("numeric").model_copy(
        update={
            "answer_type": AnswerType.FILE,
            "output_paths": ("/app/solution.txt",),
            "resources": ResourceGroups(
                worker=(inline_resource("data/expected.txt", b"7"),),
                oracle=(inline_resource("solution/solve.sh", b"cp /data/expected.txt solution.txt\n"),),
            ),
        }
    )
    return script_graded(task, b'test "$(cat solution.txt)" = 7\n', answer_path=None)


def wrong_file(_task: TaskSpec) -> WorkspaceFiles:
    return WorkspaceFiles({"/app/solution.txt": b"8"})


@pytest.mark.parametrize("kind", ["math", "numeric", "mcq", "exact"])
def test_reference_controls_pass_for_in_process_graders(kind):
    assert checks(in_process_task(kind), REFERENCE_CONTROLS) == {"empty": PASS, "golden": PASS, "negative": PASS}


@pytest.mark.parametrize("kind", ["math", "numeric", "mcq", "exact"])
def test_controls_fail_when_grader_rejects_the_golden_or_accepts_the_negative(kind):
    swapped = Controls(golden=wrong_reply, negative=reference_reply)
    assert checks(in_process_task(kind), swapped) == {"empty": PASS, "golden": FAIL, "negative": FAIL}


@pytest.mark.parametrize("controls", [Controls(negative=wrong_reply), REFERENCE_CONTROLS], ids=["none", "no_reference"])
def test_missing_golden_is_skipped_while_empty_and_negative_controls_run(controls):
    # A JSON schema grader has no reference instance, so reference_reply also has no golden.
    assert checks(in_process_task("schema"), controls) == {"empty": PASS, "golden": SKIPPED, "negative": PASS}


@pytest.mark.parametrize("grader", [NoGrader(reason="Source evaluator unavailable"), SessionGrader()])
def test_graders_without_offline_grading_have_unsupported_controls(grader):
    task = in_process_task("numeric").model_copy(update={"grader": grader})
    report = control_suite(REFERENCE_CONTROLS, FixtureGradingMachines()).run(task)
    assert [check.status for check in report.checks] == [CheckStatus.UNSUPPORTED]


@pytest.mark.parametrize(
    "task,golden,negative",
    [
        (answer_task(), lambda task: answer_reply(task, "7"), wrong_reply),
        (
            answer_task(),
            lambda _: OracleCommand("cp /data/expected.txt answer.out", answer_file="answer.out"),
            wrong_reply,
        ),
        (
            answer_task(),
            lambda _: OracleCommand("printf 7 > /tmp/answer.out", answer_file="/tmp/answer.out"),
            wrong_reply,
        ),
        (file_task(), lambda _: WorkspaceFiles({"/app/solution.txt": b"7"}), wrong_file),
        (file_task(), lambda _: OracleCommand("bash /solution/solve.sh"), wrong_file),
    ],
    ids=["reply", "oracle_relative_answer_file", "oracle_absolute_answer_file", "workspace_files", "oracle_files"],
)
def test_sandbox_controls_grade_each_submission_in_a_fresh_grader_machine(task, golden, negative):
    controls = Controls(golden=golden, negative=negative)
    assert checks(task, controls, FixtureGradingMachines()) == {"empty": PASS, "golden": PASS, "negative": PASS}


@pytest.mark.parametrize(
    "task,oracle",
    [
        (file_task(), OracleCommand("exit 3")),
        (file_task(), OracleCommand("true")),
        (answer_task(), OracleCommand("true", answer_file="answer.out")),
    ],
    ids=["oracle_exits_nonzero", "oracle_writes_no_file", "oracle_writes_no_answer_file"],
)
def test_oracle_without_a_correct_submission_fails_the_golden_control(task, oracle):
    controls = Controls(
        golden=lambda _: oracle, negative=wrong_file if task.answer_type == AnswerType.FILE else wrong_reply
    )
    assert checks(task, controls, FixtureGradingMachines())["golden"] == FAIL


def test_grader_that_accepts_any_submission_fails_empty_and_negative_controls():
    task = script_graded(file_task(), b"true\n", answer_path=None)
    controls = Controls(golden=lambda _: WorkspaceFiles({"/app/solution.txt": b"7"}), negative=wrong_file)
    assert checks(task, controls, FixtureGradingMachines()) == {"empty": FAIL, "golden": PASS, "negative": FAIL}


@pytest.mark.parametrize(
    "golden",
    [lambda _: WorkspaceFiles({"/app/solution.txt": b"7"}), lambda _: OracleCommand("bash /solution/solve.sh")],
    ids=["workspace_files", "oracle"],
)
def test_unavailable_grading_machines_are_infrastructure_errors(golden):
    controls = Controls(golden=golden, negative=wrong_file)
    assert checks(file_task(), controls, FixtureGradingMachines(UnavailableImages())) == {
        "empty": CheckStatus.INFRA_ERROR,
        "golden": CheckStatus.INFRA_ERROR,
        "negative": CheckStatus.INFRA_ERROR,
    }
