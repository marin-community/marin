# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove sandbox graders run on archived source tasks in the locally built grader image.

Each test converts an archive kept under ``fixtures/`` and grades it through the runtime's sandbox
path. See ``local_grader`` for building the image these tests run.
"""

import pytest
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import ScriptGrader, TaskSpec
from taskcompendium.pipeline.controls import answer_reply, run_controls
from taskcompendium.pipeline.models import CheckStatus, WorkspaceFiles
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.tasktrove import calendar, math, python_tests
from experiments.post_training.task_curation.tests.conversion import converted_task, tasktrove_row
from experiments.post_training.task_curation.tests.local_grader import grade
from experiments.post_training.task_curation.tests.test_tasktrove_text import fixture_files

pytestmark = pytest.mark.docker

GRADING_MEMORY_MB = 2048
PIPELINES = {pipeline.name: pipeline for module in (calendar, math, python_tests) for pipeline in module.pipelines()}
ARCHIVE_GRADERS = [
    ("tasktrove-math_gym", "math_gym"),
    ("tasktrove-math_prism", "math_prism"),
    ("tasktrove-calendar", "calendar"),
]
"""A source using each archived scorer revision, and the fixture archive it is graded on."""

TYPER_PACKAGE = {
    "/app/funk_lines/__init__.py": b'__version__ = "0.1.0"\n',
    "/app/funk_lines/__main__.py": (
        b"""import typer

from funk_lines import __version__

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(version: bool = typer.Option(False, "--version", help="Show the version.")) -> None:
    if version:
        typer.echo(f"funk_lines {__version__}")
        raise typer.Exit()
"""
    ),
}
"""An implementation of the ``stack_pytest`` fixture's request; its hidden tests import ``typer``."""
BROKEN_PACKAGE = {"/app/funk_lines/__init__.py": b"raise RuntimeError('__broken_package__')\n"}


def fixture_task(name: str, fixture: str) -> TaskSpec:
    return converted_task(PIPELINES[name], tasktrove_row(fixture_files(fixture)))


def without_module(task: TaskSpec, module: str) -> TaskSpec:
    """The task with ``module`` shadowed on the scorer's import path by a package that fails to import."""
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    broken = inline_resource(f"unimportable/{module}/__init__.py", f"raise ImportError({module!r})\n".encode())
    return task.model_copy(
        update={
            "grader": grader.model_copy(update={"env": {**grader.env, "PYTHONPATH": "/tests/unimportable"}}),
            "resources": task.resources.model_copy(update={"verifier": (*task.resources.verifier, broken)}),
        }
    )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name, fixture", ARCHIVE_GRADERS)
def test_archive_grader_passes_its_golden(machines, name, fixture):
    controls = PIPELINES[name].controls
    assert controls is not None
    report = run_controls(fixture_task(name, fixture), controls=controls, machines=machines)
    statuses = {check.check: check.status for check in report.checks}
    assert statuses == {"golden": CheckStatus.PASS}, report


# The gym scorer is absent: its runner writes reward 0 before scoring and the scorer turns any exception into 0,
# so only its golden control shows that the image runs it.
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "name, fixture, module",
    [("tasktrove-math_prism", "math_prism", "sympy"), ("tasktrove-calendar", "calendar", "json")],
)
def test_archive_grader_reports_an_unimportable_dependency_as_a_grading_failure(machines, name, fixture, module):
    task = without_module(fixture_task(name, fixture), module)
    result = grade(task, answer_reply(task, "__incorrect_answer__"), machines, GRADING_MEMORY_MB)
    assert (result.status, result.failure) == (Outcome.INFRA_ERROR, GradingFailure.MISSING_REWARD), result


@pytest.mark.timeout(120)
@pytest.mark.parametrize("files, reward", [(TYPER_PACKAGE, 1.0), (BROKEN_PACKAGE, 0.0)])
def test_stack_pytest_runs_hidden_tests_that_need_a_third_party_package(machines, files, reward):
    task = fixture_task("tasktrove-stack_pytest", "stack_pytest")
    result = grade(task, WorkspaceFiles(files), machines, GRADING_MEMORY_MB)
    assert (result.status, result.reward) == (Outcome.GRADED, reward), result
