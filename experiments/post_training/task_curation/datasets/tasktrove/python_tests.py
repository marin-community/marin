# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Python implementation tasks graded by one hidden pytest file.

Each source ships one pytest file and, usually, a ``solution/solution.py`` oracle. The agent writes
files under ``/app``; unitsyn's agent writes the fixed solution file instead. The hidden tests run
with the grader packages (``GRADER_PACKAGES``). The sources differ only by
configuration and rubric. Each agent environment replays its archived Dockerfile on its pinned base;
the grader runs separately with ``GRADER_PACKAGES``.
"""

from dataclasses import dataclass, field

from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import Controls, Converter, ImportRejection, IntendedUse, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.code import ANSWERABILITY_CRITERIA
from experiments.post_training.task_curation.datasets.tasktrove.conversion.executable import (
    SOLUTION_PATHS,
    solve_script,
    tasktrove_archive_task,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.python_unit_tests import (
    convert as convert_unit_tests,
)
from experiments.post_training.task_curation.environment_fidelity import archive_environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

PYTHON_TESTS_CONTROLS = Controls(golden=solve_script)

PUBLIC_FIXTURE_CRITERION = """
Oracle solutions and hidden tests must not be shown to the solver; explicitly public setup tests are part
of the contract.
"""

PYTHON_BASIC_CRITERIA = f"""
Check that the public Python API, output filenames, return values, and exceptions agree with hidden tests.

Flag contradictory examples, unstated behavior, missing fixtures, and unavailable dependencies.
{PUBLIC_FIXTURE_CRITERION}
A passing oracle shows compatibility with tests; assess whether those tests cover the public specification.
"""

PYMETHODS_COMMON_CRITERIA = """
Check parameter meanings and essential transition rules against the tests and oracle; an algorithmic
constraint missing from the public request is a defect even when the oracle passes.

Do not equate a quantity, rate, time limit, and lower bound; identify concrete parameter mismatches.

Flag undefined behavior, missing fixtures, unavailable dependencies, and contradictory examples.

Hidden tests and oracle solutions are review evidence and must not be shown to the solving actor.

A passing oracle shows test compatibility, not specification coverage; cite a concrete defect when rejecting.
"""

CURRICULUM_EASY_RUBRIC = f"""
{PYTHON_BASIC_CRITERIA}
Check the Python entry point and each stated beginner-level rule against test cases, including empty input
and boundaries.
"""

CURRICULUM_MEDIUM_RUBRIC = f"""
For every boolean membership assertion in disclosed setup tests, derive the expected value from its literal
group fixture and the public parsing rule before deciding quality. Including assertions in the contract does
not excuse a contradiction with an explicit prose rule. Cite the fixture membership and contradictory
assertion when one exists.

Public setup tests may specify missing API details, but they do not override an explicit prose rule unless
the task states a precedence rule. A membership fixture that marks a listed member false contradicts a rule
that all listed members are true; cite the literal values.

Check that the public Python API, output filenames, return values, and exceptions agree with hidden tests.

The repair note explicitly exposes setup tests as API evidence; assess the request together with these
fixtures and flag contradictions between them.
{PUBLIC_FIXTURE_CRITERION}
A passing oracle shows compatibility with tests; assess whether those tests cover the public specification.

Check every stated algorithmic rule, mutation requirement, and boundary against the hidden tests; difficulty
alone is not a defect.
"""

E2EGIT_RUBRIC = f"""
{PYTHON_BASIC_CRITERIA}
Check calculator, banking, inventory, and library APIs against tests; inspect exact error messages and
whether filename normalization leaves any unstated behavior.
"""

E2EGIT_LARGE_RUBRIC = f"""
{PYTHON_BASIC_CRITERIA}
Check Calculator arithmetic methods and exact zero-division error messages; repeated calculator tasks need
duplicate review, and missing multiplication tests mean incomplete coverage.
"""

MULTIFILE_RUBRIC = f"""
Check that the public Python API, output filenames, return values, and exceptions agree with hidden tests.

The repair note explicitly exposes setup tests as API evidence; assess the request together with these
fixtures and flag contradictions between them.
{PUBLIC_FIXTURE_CRITERION}
A passing oracle shows compatibility with tests; assess whether those tests cover the public specification.

Check that every required file and import is specified and captured by the grading contract; flag tests that
require unavailable sibling modules.
"""

PYMETHODS_RUBRIC = f"""
For partitioning and scheduling problems, check whether contiguity, order, indivisibility, and coverage
restrictions are explicitly supplied. Construct a better valid solution under the public rules before
accepting a narrower hidden optimum.

Check tests against the stated input domain, including zero values and allowed worker counts. Reject a
contradiction in expected behavior; distinguish explicitly described edge cases from a merely abbreviated
constraints list.

Check method signatures, class context, return values, and exceptions against the hidden tests.
{PYMETHODS_COMMON_CRITERIA}"""

PYMETHODS_LARGE_RUBRIC = f"""
Verify that every function or class name and signature required by hidden imports is present in the public
request or public fixtures. A request to follow a provided signature is incomplete when no signature is
supplied; a conventional name is not a public API contract.

Check class context, method signatures, instance state, and dependency requirements against the hidden tests.
{PYMETHODS_COMMON_CRITERIA}"""

STACK_PYTEST_RUBRIC = """
Check that the adapted Stack Overflow request defines the tested API and supplies all relevant context.

Check that the named modules and package files in the public request are captured by the runtime; a
solution.py-only submission cannot implement a different named package.

Missing oracle controls imply verification uncertainty, not an automatically bad problem.

Flag undefined behavior, missing fixtures, unavailable dependencies, and contradictory examples.

Hidden tests and oracle solutions are review evidence and must not be shown to the solving actor.

A passing oracle shows test compatibility, not specification coverage; cite a concrete defect when rejecting.
"""

UNITSYN_RUBRIC = f"""
{ANSWERABILITY_CRITERIA}
Check that the requested Python API, filenames, return values, and exceptions agree with the hidden tests.

Look for tests that invent unstated behavior, incomplete definitions, missing fixtures, or dependencies.

A passing oracle proves compatibility with the supplied tests, not that the tests cover the specification.

A public example that contradicts the written rule is a defect even if the hidden tests follow the rule. Do
not dismiss incorrect example comments, off-by-one boundaries or required unspecified behavior as minor.
"""

UNITSYN_LARGE_RUBRIC = f"""
Check that the public Python API, filenames, return values, and exception behavior agree with the hidden
tests.
{PYMETHODS_COMMON_CRITERIA}"""


def convert_python_tests(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Capture the implementation and assets under the source workspace."""
    return tasktrove_archive_task(
        row,
        convert=convert_unit_tests,
        environment=archive_environment(row.data["files"]),
        grader_environment=required_grader_environment(context),
        output_paths=("/app",),
    )


def convert_unitsyn(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """Capture the fixed solution files, as the source's grader did."""
    return tasktrove_archive_task(
        row,
        convert=convert_unit_tests,
        environment=archive_environment(row.data["files"]),
        grader_environment=required_grader_environment(context),
        output_paths=SOLUTION_PATHS,
    )


@dataclass(frozen=True)
class PythonTestsSource:
    config: str
    convert: Converter
    rubric: str
    info: SourceInfo = field(kw_only=True)
    version: str = "5"


SOURCES = {
    "curriculum_easy": PythonTestsSource(
        "DCAgent__exp_rpt_curriculum-easy",
        convert_python_tests,
        CURRICULUM_EASY_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_curriculum-easy",
            upstream_dataset="DCAgent__exp_rpt_curriculum-easy",
            upstream_name="DCAgent__exp_rpt_curriculum-easy",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=509,
            notes=(
                "Self-contained pytest tasks. In a 10-task sample, empty and trivial submissions "
                "failed every task. Kept with the kata tag."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "curriculum_medium": PythonTestsSource(
        "DCAgent__exp_rpt_curriculum-medium-v2",
        convert_python_tests,
        CURRICULUM_MEDIUM_RUBRIC,
        version="6",
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_curriculum-medium-v2",
            upstream_dataset="DCAgent__exp_rpt_curriculum-medium-v2",
            upstream_name="DCAgent__exp_rpt_curriculum-medium-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=492,
            notes=(
                "Self-contained pytest tasks. In a 10-task sample, empty and trivial submissions "
                "failed every task. Kept with the kata tag."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "e2egit": PythonTestsSource(
        "DCAgent__exp_rpt_e2egit-v2",
        convert_python_tests,
        E2EGIT_RUBRIC,
        version="8",
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_e2egit-v2",
            upstream_dataset="DCAgent__exp_rpt_e2egit-v2",
            upstream_name="DCAgent__exp_rpt_e2egit-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python", "language:javascript"),
            count=487,
            notes=(
                "Self-contained pytest tasks. Empty and trivial submissions failed all 10 sampled "
                "tasks; exact instruction dedup handles overlap with the large cut."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "e2egit_large": PythonTestsSource(
        "DCAgent__exp_rpt_e2egit-large",
        convert_python_tests,
        E2EGIT_LARGE_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_e2egit-large",
            upstream_dataset="DCAgent__exp_rpt_e2egit-large",
            upstream_name="DCAgent__exp_rpt_e2egit-large",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=4998,
            notes=(
                "Self-contained pytest tasks. Empty and trivial submissions failed all 10 sampled "
                "tasks; exact instruction dedup handles repeated katas."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "multifile": PythonTestsSource(
        "DCAgent__exp_rpt_multifile-v3",
        convert_python_tests,
        MULTIFILE_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_multifile-v3",
            upstream_dataset="DCAgent__exp_rpt_multifile-v3",
            upstream_name="DCAgent__exp_rpt_multifile-v3",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=4843,
            notes=(
                "Self-contained multi-file pytest tasks. In a 10-task sample, empty and trivial "
                "submissions failed every task. Kept with the kata tag."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "pymethods": PythonTestsSource(
        "DCAgent__exp_rpt_pymethods2test-v3",
        convert_python_tests,
        PYMETHODS_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_pymethods2test-v3",
            upstream_dataset="DCAgent__exp_rpt_pymethods2test-v3",
            upstream_name="DCAgent__exp_rpt_pymethods2test-v3",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=500,
            notes=(
                "Self-contained pytest katas with shipped Python oracles. All 10 sampled oracles "
                "passed, while empty and trivial submissions failed."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "pymethods_large": PythonTestsSource(
        "DCAgent__exp_rpt_pymethods2test-large-v2",
        convert_python_tests,
        PYMETHODS_LARGE_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_pymethods2test-large-v2",
            upstream_dataset="DCAgent__exp_rpt_pymethods2test-large-v2",
            upstream_name="DCAgent__exp_rpt_pymethods2test-large-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=4991,
            notes=(
                "Self-contained pytest katas with shipped Python oracles. All 10 sampled oracles "
                "passed, while empty and trivial submissions failed."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "unitsyn": PythonTestsSource(
        "DCAgent__exp_rpt_unitsyn-python-v4",
        convert_unitsyn,
        UNITSYN_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_unitsyn-python-v4",
            upstream_dataset="DCAgent__exp_rpt_unitsyn-python-v4",
            upstream_name="DCAgent__exp_rpt_unitsyn-python-v4",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=491,
            notes=(
                "Self-contained pytest katas with shipped Python oracles. The standardized mode "
                "passed all 10 sampled oracles and rejected all 10 empty submissions."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "unitsyn_large": PythonTestsSource(
        "DCAgent__exp_rpt_unitsyn-python-large-v2",
        convert_python_tests,
        UNITSYN_LARGE_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_unitsyn-python-large-v2",
            upstream_dataset="DCAgent__exp_rpt_unitsyn-python-large-v2",
            upstream_name="DCAgent__exp_rpt_unitsyn-python-large-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=4991,
            notes=(
                "Self-contained pytest katas with shipped Python oracles. All 10 sampled oracles "
                "passed, while empty and trivial submissions failed."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
    "stack_pytest": PythonTestsSource(
        "DCAgent__exp_rpt_stack-pytest-v2",
        convert_python_tests,
        STACK_PYTEST_RUBRIC,
        info=SourceInfo(
            id="TaskTrove:DCAgent__exp_rpt_stack-pytest-v2",
            upstream_dataset="DCAgent__exp_rpt_stack-pytest-v2",
            upstream_name="DCAgent__exp_rpt_stack-pytest-v2",
            subset=None,
            build_path="Task Trove",
            origin=SourceOrigin.TASKTROVE,
            family="unit-test-gen",
            tags=("agentic", "multi-turn", "language:python"),
            count=500,
            notes=(
                "Self-contained pytest tasks. In a 10-task sample, empty and trivial submissions "
                "failed every task. Kept with the kata tag."
            ),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    ),
}


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=source.info,
            config=CurationRecipe(
                name=f"tasktrove-{name}",
                source=tasktrove_source(source.config),
                convert=TaskTroveConverter(source.config, source.convert),
                version=source.version,
                intended_use=IntendedUse.TRAIN,
                rubric=source.rubric,
                controls=PYTHON_TESTS_CONTROLS,
                oracle_solutions=True,
                grader=GRADER_PACKAGES,
            ),
        )
        for name, source in SOURCES.items()
    ]
