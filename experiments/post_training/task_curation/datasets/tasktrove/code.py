# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove competitive-programming sources: stdin/stdout programs graded by verifyit ``stdio``.

The agent writes a program in the executable image; the grader compiles it when it is C++ and runs it on
the hidden cases with the grader packages and a C++ toolchain (``COMPILER_GRADER_PACKAGES``). Controls
run the source's ``solution/solve.sh`` oracle when the archive ships one, and otherwise grade an empty
submission.
"""

from taskcompendium.convert.executable import (
    SOLUTION_PATHS,
    solve_script,
    tasktrove_archive_task,
    tasktrove_python_task,
)
from taskcompendium.convert.tasktrove import DOCKERFILE, INSTRUCTION, TaskFiles
from taskcompendium.convert.tasktrove_code_contests import convert_code_contests
from taskcompendium.convert.tasktrove_codeforces import convert_codeforces
from taskcompendium.convert.tasktrove_converted_task import ConvertedTask, ConvertFn, ConvertStatus, Rejected
from taskcompendium.convert.tasktrove_nemotron_data import verifier_data
from taskcompendium.convert.tasktrove_stdio_cases import SOLUTION_COMMAND, case_files
from taskcompendium.convert.tasktrove_taco import convert_taco
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import Controls, Converter, ImportRejection, IntendedUse, NormalizedTask, RawRow
from verifyit.spec import Compare, StdioSpec

from experiments.post_training.task_curation.datasets.environments import COMPILER_GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import tasktrove_source
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import RlDataPipeline, environment_requirements

AGENT_IMAGE = Environment(
    image="ghcr.io/marin-community/iris-task@sha256:d6af0d198b29650fea0eaccb27d01cb9bbfb0aec9e1d5959a74c2ae055b6f305"
)
"""The executable image the agent writes and runs its program in."""

EXECUTABLE_CONTROLS = Controls(golden=solve_script)

ANSWERABILITY_CRITERIA = """
Judge whether the underlying request is comprehensible and answerable using the public inputs.

Hidden tests and oracle solutions are review evidence and must never be shown to the solving actor.

Report a concrete defect rather than penalizing difficulty. Missing oracle controls imply verification
uncertainty, not an automatically bad problem.
"""
"""Criteria shared by TaskTrove executable sources, before each source's own criteria."""

CODE_CONTESTS_RUBRIC = """
Check the full stdin/stdout problem, constraints, examples, and hidden cases for agreement. Absent
diagrams, interactive protocols without an interactor, and contradictory outputs are defects.

Inspect numerical error clauses and special-output semantics. Exact line comparison cannot grade an
arbitrary valid construction unless the task specifies a canonical output.

The source supplies no oracle solution. Assess static coherence from the problem and cases anyway;
missing executable positive controls and inability to solve quickly are not quality defects.
"""

CODEFORCES_RUBRIC = f"""
{ANSWERABILITY_CRITERIA}
Check the complete problem statement, constraints, input/output format, and examples against hidden cases.

A special judge may allow many correct outputs; exact comparison must not replace that semantic contract.

Flag unsupported language promises, absent input, inconsistent numerical tolerances, or sample-only tests.
"""

COMPETITIVE_CODING_RUBRIC = """
Check the complete problem statement, constraints, stdin format, stdout format, and public examples.

Compare hidden case inputs and outputs with the public rules; identify wrong keys and missing drivers.

The source grades exact line output with trailing whitespace normalized and requires all cases to pass.

Flag tasks needing a special judge, numerical tolerance, or multiple valid outputs that exact grading rejects.

Public examples are legitimate cases. A sample-only set limits coverage; it does not by itself show leaked
gold or a defective task. Report coverage separately from content quality.

Missing oracle controls imply verification uncertainty, not an automatically bad programming problem.
"""

TACO_RUBRIC = f"""
{ANSWERABILITY_CRITERIA}
Check whether this is a complete stdin/stdout program problem or a function-call problem without a driver.

Inspect public examples versus hidden case format and expected outputs; flag contradictions or leaked gold.

Check numerical tolerances, multi-case parsing, and that a meaningful set of hidden cases exists.
"""


def stdio_task(row: RawRow, context: ConversionContext, convert: ConvertFn) -> NormalizedTask | ImportRejection:
    return tasktrove_archive_task(
        row,
        convert=convert,
        environment=environment_requirements(AGENT_IMAGE),
        grader_environment=required_grader_environment(context),
        output_paths=SOLUTION_PATHS,
    )


def convert_code_contests_task(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return stdio_task(row, context, convert_code_contests)


def convert_codeforces_task(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return stdio_task(row, context, convert_codeforces)


def convert_taco_task(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return stdio_task(row, context, convert_taco)


def convert_competitive_coding(task: TaskFiles) -> ConvertedTask | Rejected:
    """The source's aligned input/output pairs, run with the solution command and compared exactly."""
    data = verifier_data(task)
    inputs, outputs = data.get("inputs"), data.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs) or not inputs:
        return Rejected(ConvertStatus.NULL_GRADER, "At least one aligned input/output case is required")
    if not all(isinstance(value, str) for value in [*inputs, *outputs]):
        return Rejected(ConvertStatus.NULL_GRADER, "Inputs and outputs must be strings")
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=StdioSpec(command=SOLUTION_COMMAND, compare=Compare.EXACT),
        dockerfile=task.text(DOCKERFILE),
        tags=("code", "competitive-programming", "stdio", "nemotron"),
        language="python",
        data_files=case_files(inputs, outputs),
    )


def convert_competitive_coding_task(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return tasktrove_python_task(
        row,
        convert=convert_competitive_coding,
        environment=environment_requirements(AGENT_IMAGE),
        grader_environment=required_grader_environment(context),
    )


def stdio_pipeline(name: str, config: str, convert: Converter, rubric: str) -> RlDataPipeline:
    return RlDataPipeline(
        name=f"tasktrove-{name}",
        source=tasktrove_source(config),
        convert=convert,
        version="1",
        environment=AGENT_IMAGE,
        intended_use=IntendedUse.TRAIN,
        rubric=rubric,
        controls=EXECUTABLE_CONTROLS,
        atlas_id=f"Task Trove:{config}",
        grader=COMPILER_GRADER_PACKAGES,
    )


def pipelines() -> list[RlDataPipeline]:
    return [
        stdio_pipeline(
            "code_contests", "DCAgent__code-contests-noblock", convert_code_contests_task, CODE_CONTESTS_RUBRIC
        ),
        stdio_pipeline("codeforces", "laion__codeforces-v3", convert_codeforces_task, CODEFORCES_RUBRIC),
        stdio_pipeline(
            "competitive_coding",
            "laion__nemotron-gym-competitive-coding-v2",
            convert_competitive_coding_task,
            COMPETITIVE_CODING_RUBRIC,
        ),
        stdio_pipeline("taco", "laion__exp_rpt_taco-v2", convert_taco_task, TACO_RUBRIC),
    ]
