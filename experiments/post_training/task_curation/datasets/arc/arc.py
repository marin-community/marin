# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI grid puzzles: two TaskTrove Nemotron Gym configs and the Nemotron Ultra NVARC converter.

TaskTrove transductive tasks retain the release's whitespace-normalized exact grid comparison.
Other ARC tasks use ``arc_grade.py`` with the vendored NVARC scorer from ``scorers/``, with the
grader packages (``GRADER_PACKAGES``). An inductive submission is a ``transform(grid)`` program,
which the script runs on the hidden test input as an unprivileged user; a transductive submission is
the output grid. TaskTrove tasks ask for a file (``/app/solution.py`` or ``/app/answer.txt``); Ultra
NVARC rows ask for a reply.
"""

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.nemotron_ultra import blend_task, text_request
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.convert.tasktrove import ANSWER_PATH, archive_resources
from taskcompendium.grader import GraderPackage, grader_config, verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    verifyit_spec,
)
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import (
    Controls,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    RawRow,
    Reply,
    WorkspaceFiles,
)
from verifyit.spec import ExactSpec

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.nemotron_ultra.graders import SCORERS as ULTRA_SCORERS
from experiments.post_training.task_curation.datasets.nemotron_ultra.graders import ULTRA_BASE
from experiments.post_training.task_curation.datasets.tasktrove.archives import tasktrove_source
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim, recipe_source
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

INDUCTIVE_CONFIG = "laion__nemotron-gym-arc-agi-python-inductive-v2"
TRANSDUCTIVE_CONFIG = "laion__nemotron-gym-arc-agi-transductive-v3"
INDUCTIVE_AGENT = "nvarc_inductive_simple_agent"
TRANSDUCTIVE_AGENT = "nvarc_transductive_simple_agent"

HERE = Path(__file__).parent
SCORERS = HERE / "scorers"
ARC_SHIPS = (SCORERS, ULTRA_SCORERS)
"""The vendored directories an ARC task ships files from: NVARC here, its answer extractor in Ultra's."""
NVARC_MODULES = "skyrl_gym/envs/nemotron_ultra"
ARC_GRADE = grade_script(
    HERE / "arc_grade.py",
    *ULTRA_BASE,
    *shipped_files(SCORERS, f"{NVARC_MODULES}/nvarc.py", f"{NVARC_MODULES}/sandbox.py", "local_sandbox.py"),
)
SOLUTION_PATH = "/app/solution.py"
# NVARC gives the transform 30 seconds; the rest covers interpreter start-up and imports.
GRADER_TIMEOUT = 45.0
GRADER_MEMORY_MB = 4096


class ArcMode(StrEnum):
    INDUCTIVE = "inductive"
    TRANSDUCTIVE = "transductive"


TASKTROVE_INDUCTIVE_RUBRIC = """
The requested Python transform must be grounded in complete public input-output examples. A small held-out set does
not by itself prove the puzzle is incoherent or unsolvable.

Compare the hidden test case with a transformation supported by all examples where feasible. Missing oracle code is a
readiness gap, not a content defect.

The public instruction promises numpy, scipy, itertools and collections; the grader runs the transform with the grader
image's numpy and scipy.

The NVARC scorer takes the last fenced program in /app/solution.py, else in /app/answer.txt, or the whole file when it
defines transform; runs transform(grid) on the held-out input; and compares the JSON-encoded result exactly with the
held-out output. This is code evaluation, not an exact text match against an oracle program.
"""

TASKTROVE_TRANSDUCTIVE_RUBRIC = """
The public examples and test grid must be complete and readable. Judge the common transformation rule, not whether the
review model can fully solve a difficult ARC puzzle.

Compare the hidden expected grid against the examples and test input when a concrete rule can be established. Do not
invent an alternative key from superficial pattern matching.

The grader reads /app/answer.txt and compares the last boxed answer or the whole answer after whitespace normalization
against space-separated grid rows, as in the TaskTrove release. JSON and extra unboxed prose fail. Record conflicts
between the source's quoted answer format and its file-submission instruction rather than silently rewriting it.
"""


def valid_grid(grid: object) -> bool:
    """NVARC's grid check: a nonempty rectangular list of integer cells 0-9."""
    return (
        isinstance(grid, list)
        and bool(grid)
        and all(isinstance(row, list) and row and len(row) == len(grid[0]) for row in grid)
        and all(type(cell) is int and 0 <= cell <= 9 for row in grid for cell in row)
    )


def reference_rejection(mode: ArcMode, record: Mapping[str, Any]) -> ImportRejection | None:
    """Reject a record whose grids NVARC cannot accept, since no submission could then score."""
    fields = ("test_input", "expected_output") if mode == ArcMode.INDUCTIVE else ("expected_output",)
    missing = [name for name in fields if record.get(name) is None]
    if missing:
        return source_defect("invalid_verifier_data", f"Missing {', '.join(missing)}")
    invalid = [name for name in fields if not valid_grid(record[name])]
    if invalid:
        return source_defect("reference_conflict", f"NVARC rejects {', '.join(invalid)} as an ARC grid")
    return None


def arc_package(
    mode: ArcMode, record: Mapping[str, Any], context: ConversionContext, answer_path: str | None
) -> GraderPackage:
    return script_package(
        ARC_GRADE,
        {"mode": mode, "contract": record},
        environment=required_grader_environment(context),
        timeout=GRADER_TIMEOUT,
        answer_path=answer_path,
    )


def tasktrove_record(mode: ArcMode, data: Mapping[str, Any]) -> dict[str, Any] | ImportRejection:
    """The NVARC record of a TaskTrove ``verifier_data.json``: its one held-out case, or its expected grid."""
    if mode == ArcMode.TRANSDUCTIVE:
        return {"expected_output": data.get("expected_output")}
    cases = data.get("test_cases")
    if not isinstance(cases, list) or not cases or not all(isinstance(case, dict) for case in cases):
        return source_defect("invalid_verifier_data", "test_cases must be a nonempty list of input-output pairs")
    if len(cases) > 1:
        return unsupported("multiple_test_cases", "NVARC scores one held-out test input")
    return {"test_input": cases[0].get("input"), "expected_output": cases[0].get("output")}


def _tasktrove_task(
    row: RawRow, context: ConversionContext, mode: ArcMode, output_paths: tuple[str, ...]
) -> TaskSpec | ImportRejection:
    """A file task graded on the workspace files at ``output_paths``."""
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and verifier_data are required")
    record = tasktrove_record(mode, data)
    if isinstance(record, ImportRejection):
        return record
    rejection = reference_rejection(mode, record)
    if rejection is not None:
        return rejection
    if mode == ArcMode.TRANSDUCTIVE:
        package = verifyit_package(
            ExactSpec(expected=(grid_text(record["expected_output"]),)),
            environment=required_grader_environment(context),
        )
        tags = ("reasoning", "arc-agi", "grid-match", "nemotron")
    else:
        package = arc_package(mode, record, context, answer_path=None)
        tags = ("reasoning", "arc-agi", "grid-transform", "code", "nemotron", "language:python")
    archive = archive_resources(row.data)
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(capabilities=("filesystem", "python")),
        # The archive's own tests are replaced by the NVARC grader, so only its other files are kept.
        resources=ResourceGroups(worker=archive.worker, verifier=package.resources, oracle=archive.oracle),
        output_paths=output_paths,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=package.grader,
        tags=tags,
    )


def convert_tasktrove_inductive(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """The grader reads ``solution.py`` first and ``answer.txt`` as a fallback."""
    return _tasktrove_task(row, context, ArcMode.INDUCTIVE, (SOLUTION_PATH, ANSWER_PATH))


def convert_tasktrove_transductive(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    return _tasktrove_task(row, context, ArcMode.TRANSDUCTIVE, (ANSWER_PATH,))


def convert_ultra_arc(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    """An Ultra NVARC reply: a fenced transform program (inductive) or an output grid (transductive)."""
    request = text_request(row.data, (INDUCTIVE_AGENT, TRANSDUCTIVE_AGENT))
    if isinstance(request, ImportRejection):
        return request
    mode = ArcMode.INDUCTIVE if request.agent == INDUCTIVE_AGENT else ArcMode.TRANSDUCTIVE
    rejection = reference_rejection(mode, request.contract)
    if rejection is not None:
        return rejection
    return blend_task(row, request, arc_package(mode, request.contract, context, answer_path=ANSWER_PATH))


def grid_text(grid: list[list[int]]) -> str:
    """Space-separated rows, the grid layout the NVARC prompts request."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def literal_transform(grid: list[list[int]]) -> str:
    """A transform that returns ``grid`` whatever its input: correct on the one held-out input only."""
    return f"def transform(grid):\n    return {grid!r}\n"


def _record(task: TaskSpec) -> tuple[ArcMode, dict[str, Any]]:
    config = grader_config(task)
    return ArcMode(config["mode"]), config["contract"]


def tasktrove_golden(task: TaskSpec) -> WorkspaceFiles:
    if isinstance(task.grader, VerifyitGrader):
        spec = verifyit_spec(task.grader)
        assert isinstance(spec, ExactSpec)
        return WorkspaceFiles({ANSWER_PATH: (spec.expected[0] + "\n").encode()})
    record = grader_config(task)["contract"]
    return WorkspaceFiles({SOLUTION_PATH: literal_transform(record["expected_output"]).encode()})


def ultra_arc_golden(task: TaskSpec) -> Reply:
    """The expected grid, or a fenced transform returning it."""
    mode, record = _record(task)
    if mode == ArcMode.INDUCTIVE:
        return answer_reply(task, f"```python\n{literal_transform(record['expected_output'])}```")
    return answer_reply(task, grid_text(record["expected_output"]))


TASKTROVE_CONTROLS = Controls(golden=tasktrove_golden, memory_mb=GRADER_MEMORY_MB)
ULTRA_ARC_CONTROLS = Controls(golden=ultra_arc_golden, memory_mb=GRADER_MEMORY_MB)


def sources() -> list[RlDataSource[RlDataPipeline]]:
    return [
        recipe_source(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-arc-agi-python-inductive-v2",
                title="laion/nemotron-gym-arc-agi-python-inductive-v2",
                origin="Task Trove",
                family="arc-agi",
                tags=("agentic", "multi-turn", "language:python"),
                count=10000,
                notes="Agent writes a transform, graded on held-out grids. One of the best sources here.",
            ),
            pipeline=RlDataPipeline(
                name="tasktrove-arc_inductive",
                source=tasktrove_source(INDUCTIVE_CONFIG),
                convert=convert_tasktrove_inductive,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=TASKTROVE_INDUCTIVE_RUBRIC,
                controls=TASKTROVE_CONTROLS,
                grader=GRADER_PACKAGES,
                ships=ARC_SHIPS,
            ),
        ),
        recipe_source(
            info=SourceInfo(
                id="Task Trove:laion__nemotron-gym-arc-agi-transductive-v3",
                title="laion/nemotron-gym-arc-agi-transductive-v3",
                origin="Task Trove",
                family="arc-agi",
                tags=("agentic", "multi-turn"),
                count=10000,
                notes="Direct grid answer against gold. Subsample; the inductive variant is stronger.",
            ),
            pipeline=RlDataPipeline(
                name="tasktrove-arc_transductive",
                source=tasktrove_source(TRANSDUCTIVE_CONFIG),
                convert=convert_tasktrove_transductive,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=TASKTROVE_TRANSDUCTIVE_RUBRIC,
                controls=TASKTROVE_CONTROLS,
                grader=GRADER_PACKAGES,
                ships=ARC_SHIPS,
            ),
        ),
    ]
