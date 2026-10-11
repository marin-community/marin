# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove competitive-programming sources: stdin/stdout programs graded by verifyit ``stdio``.

Code Contests and TACO retain their source image recipes for the actor and its independent grader.
The remaining sources use the campaign executable image and compiler grader packages. Controls run
an archive's ``solution/solve.sh`` oracle when available, and otherwise grade an empty submission.
"""


import json

from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import Controls, Converter, ImportRejection, IntendedUse, NormalizedTask, RawRow
from verifyit.spec import Compare, StdioSpec

from experiments.post_training.task_curation.datasets.environments import COMPILER_GRADER_PACKAGES, VERIFYIT_PACKAGE
from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    DOCKERFILE,
    INSTRUCTION,
    VERIFIER_DATA,
    TaskFiles,
    archive_resources,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.code_contests import convert_code_contests
from experiments.post_training.task_curation.datasets.tasktrove.conversion.codeforces import convert_codeforces
from experiments.post_training.task_curation.datasets.tasktrove.conversion.executable import (
    SOLUTION_PATHS,
    converted_workspace_task,
    solve_script,
    tasktrove_archive_task,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import (
    ConvertedTask,
    ConvertFn,
    ConvertStatus,
    Rejected,
    archive_conversion,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.stdio_cases import (
    SOLUTION_COMMAND,
    case_files,
    hidden_case_rejection,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.taco import convert_taco
from experiments.post_training.task_curation.datasets.tasktrove.conversion.verifyit_build import verifyit_build_context
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import (
    CurationRecipe,
    environment_requirements,
    process_rows,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

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

Public examples can supplement hidden cases. Reject sample-only sets: printing the disclosed outputs
must not pass the task.

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


def source_stdio_task(row: RawRow, convert: ConvertFn) -> NormalizedTask | ImportRejection:
    """Keep the source actor image and grade captured submissions in a fresh copy."""
    converted = archive_conversion(row.data, convert)
    if isinstance(converted, ImportRejection):
        return converted
    build = verifyit_build_context(converted.dockerfile, archive_resources(row.data).oracle, package=VERIFYIT_PACKAGE)
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_build=build)
    task = converted_workspace_task(
        row,
        converted,
        instruction=converted.instruction,
        environment=environment,
        grader_environment=environment,
        output_paths=SOLUTION_PATHS,
    )
    return NormalizedTask(task, ())


def convert_code_contests_task(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    return source_stdio_task(row, convert_code_contests)


def convert_codeforces_task(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return stdio_task(row, context, convert_codeforces)


def convert_taco_task(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    return source_stdio_task(row, convert_taco)


def convert_competitive_coding(task: TaskFiles) -> ConvertedTask | Rejected:
    """The source's aligned input/output pairs, run with the solution command and compared exactly."""
    data = json.loads(task.text(VERIFIER_DATA))
    inputs, outputs = data.get("inputs"), data.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs) or not inputs:
        return Rejected(ConvertStatus.NULL_GRADER, "At least one aligned input/output case is required")
    if not all(isinstance(value, str) for value in [*inputs, *outputs]):
        return Rejected(ConvertStatus.NULL_GRADER, "Inputs and outputs must be strings")
    cases = case_files(inputs, outputs)
    instruction = task.text(INSTRUCTION)
    rejection = hidden_case_rejection(cases, instruction)
    if rejection is not None:
        return rejection
    return ConvertedTask(
        instruction=instruction,
        spec=StdioSpec(command=SOLUTION_COMMAND, compare=Compare.EXACT),
        dockerfile=task.text(DOCKERFILE),
        tags=("code", "competitive-programming", "stdio", "nemotron"),
        language="python",
        data_files=cases,
    )


def convert_competitive_coding_task(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    return tasktrove_archive_task(
        row,
        convert=convert_competitive_coding,
        environment=environment_requirements(AGENT_IMAGE),
        grader_environment=required_grader_environment(context),
        output_paths=SOLUTION_PATHS,
    )


def stdio_source(
    name: str,
    config: str,
    convert: Converter,
    rubric: str,
    info: SourceInfo,
    *,
    version: str = "1",
    grader: Environment | None = COMPILER_GRADER_PACKAGES,
) -> RlDataSource[CurationRecipe]:
    return RlDataSource(
        pipeline=process_rows,
        info=info,
        config=CurationRecipe(
            name=f"tasktrove-{name}",
            source=tasktrove_source(config),
            convert=TaskTroveConverter(config, convert),
            version=version,
            intended_use=IntendedUse.TRAIN,
            rubric=rubric,
            controls=EXECUTABLE_CONTROLS,
            grader=grader,
        ),
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        stdio_source(
            "code_contests",
            "DCAgent__code-contests-noblock",
            convert_code_contests_task,
            CODE_CONTESTS_RUBRIC,
            version="2",
            grader=None,
            info=SourceInfo(
                id="Task Trove:DCAgent__code-contests-noblock",
                title="DCAgent/code-contests-noblock",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python"),
                count=8728,
                notes="Hidden test_data.json, only the first case is shown. Exact-line compare; fine.",
            ),
        ),
        stdio_source(
            "codeforces",
            "laion__codeforces-v3",
            convert_codeforces_task,
            CODEFORCES_RUBRIC,
            info=SourceInfo(
                id="Task Trove:laion__codeforces-v3",
                title="laion/codeforces-v3",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python"),
                count=10000,
                notes=(
                    "Keep rows with at least 5 test files; drop the zero-test fallback (currently full "
                    "credit for not crashing)."
                ),
            ),
        ),
        stdio_source(
            "competitive_coding",
            "laion__nemotron-gym-competitive-coding-v2",
            convert_competitive_coding_task,
            COMPETITIVE_CODING_RUBRIC,
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-competitive-coding-v2",
                title="laion/nemotron-gym-competitive-coding-v2",
                origin="Task Trove",
                family="competitive-programming",
                tags=("agentic", "multi-turn", "language:python"),
                count=15713,
                notes="50 hidden stdin/stdout cases per task. Best competitive source in the corpus.",
            ),
        ),
        stdio_source(
            "taco",
            "laion__exp_rpt_taco-v2",
            convert_taco_task,
            TACO_RUBRIC,
            version="2",
            grader=None,
            info=SourceInfo(
                id="Task Trove:laion__exp_rpt_taco-v2",
                title="laion/exp_rpt_taco-v2",
                origin="Task Trove",
                family="stdin-stdout",
                tags=("agentic", "multi-turn", "language:python"),
                count=10000,
                notes=(
                    "Hidden cases plus oracle. Filter to tasks with at least 5 cases and add tolerance-"
                    "aware compare where the prompt promises one."
                ),
            ),
        ),
    ]
