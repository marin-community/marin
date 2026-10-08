# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI grid puzzles: two TaskTrove Nemotron Gym configs and the Nemotron Ultra NVARC converter.

TaskTrove tasks ask for a file (``/app/solution.py`` holding ``transform(grid)``, or the output grid
in ``/app/answer.txt``) and keep the archive's ``tests/test.sh`` as their grader. Ultra NVARC rows
ask for a reply: the transductive reply is scored by the image's NVARC scorer directly, and the
inductive reply by ``arc_grade.py``, which runs the transform through the image's sandbox server.
Both run in ``ARC_IMAGE``.
"""

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.nemotron_ultra import blend_task, text_request
from taskcompendium.convert.source_scorer import source_scorer_package
from taskcompendium.convert.tasktrove import archive_file, archive_resources
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    PlainText,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.models import (
    Controls,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    RawRow,
    Reply,
    WorkspaceFiles,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.tasktrove import tasktrove_source
from experiments.post_training.task_curation.images import ARC_IMAGE
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim

INDUCTIVE_CONFIG = "laion__nemotron-gym-arc-agi-python-inductive-v2"
TRANSDUCTIVE_CONFIG = "laion__nemotron-gym-arc-agi-transductive-v3"
INDUCTIVE_AGENT = "nvarc_inductive_simple_agent"
TRANSDUCTIVE_AGENT = "nvarc_transductive_simple_agent"
ARC_GRADE = "arc_grade.py"
ARC_GRADE_BYTES = Path(__file__).with_name(ARC_GRADE).read_bytes()
TRANSDUCTIVE_SCORER = {
    "function": "skyrl_gym.envs.nemotron_ultra.nvarc:grade_transductive_arc",
    "args": ["answer", "contract"],
}
ANSWER_PATH = "/app/answer.txt"
SOLUTION_PATH = "/app/solution.py"
REPLY_REWARD_PATH = "/logs/verifier/reward.json"
ARCHIVE_REWARD_PATH = "/logs/verifier/reward.txt"
REPLY_TIMEOUT = 45.0
GRADER_MEMORY_MB = 4096
ARCHIVE_GRADER_FILES = ("tests/test.sh", "tests/verifier.py", "task.toml")
FAILING_TRANSFORM = "def transform(grid):\n    raise RuntimeError('__negative_control__')\n"

TASKTROVE_INDUCTIVE_RUBRIC = """
The requested Python transform must be grounded in complete public input-output examples. A small held-out set does
not by itself prove the puzzle is incoherent or unsolvable.

Compare hidden test cases with a transformation supported by all examples where feasible. Missing oracle code is a
readiness gap, not a content defect.

Inspect the source Dockerfile against the public dependency promises. The wrapper lists numpy/scipy but the embedded
source additionally promises torch; distinguish that missing source dependency from what the grading image provides.

The source grader executes transform(grid), coerces returned cells with int(), and compares every row to held-out
outputs. It extracts solution.py first and answer.txt as a fallback; this is code evaluation, not an exact text match
against an oracle program.
"""

TASKTROVE_TRANSDUCTIVE_RUBRIC = """
The public examples and test grid must be complete and readable. Judge the common transformation rule, not whether the
review model can fully solve a difficult ARC puzzle.

Compare the hidden expected grid against the examples and test input when a concrete rule can be established. Do not
invent an alternative key from superficial pattern matching.

The source parser compares grid rows and cells, accepts bare digits, JSON or boxed grids, and ignores nonnumeric prose
lines. The wrapper requests plain space-separated rows while its quoted source asks for a boxed output; record this
format conflict rather than silently rewriting it.
"""


def check_grid(grid: object) -> None:
    """Require a nonempty rectangular grid of integer cells 0-9."""
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) and row for row in grid):
        raise ValueError("Expected a nonempty rectangular ARC grid")
    width = len(grid[0])
    if any(len(row) != width or any(type(cell) is not int or not 0 <= cell <= 9 for cell in row) for row in grid):
        raise ValueError("ARC grids require rectangular rows of integer cells 0-9")


def _check_transform_cases(data: dict[str, Any]) -> None:
    cases = data["test_cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("At least one held-out grid pair is required")
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"input", "output"}:
            raise ValueError("ARC transform cases require input and output grids")
        check_grid(case["input"])
        check_grid(case["output"])


def _check_expected_grid(data: dict[str, Any]) -> None:
    check_grid(data["expected_output"])


def _tasktrove_task(
    row: RawRow, output_paths: tuple[str, ...], check_reference: Callable[[dict[str, Any]], None]
) -> TaskSpec | ImportRejection:
    """A file task graded by the archive's own ``tests/test.sh``."""
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and verifier_data are required")
    try:
        check_reference(data)
    except (KeyError, TypeError, ValueError) as error:
        return source_defect("invalid_verifier_data", str(error))
    task_toml = archive_file(row.data, "task.toml")
    if (
        row.data.get("archive_links")
        or task_toml is None
        or any(archive_file(row.data, path) is None for path in ARCHIVE_GRADER_FILES)
    ):
        return unsupported(
            "missing_original_arc_command",
            "The archive needs tests/test.sh, tests/verifier.py, task.toml and no archive links",
        )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(capabilities=("filesystem", "python")),
        resources=archive_resources(row.data),
        output_paths=output_paths,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("bash", "/tests/test.sh"),
            cwd="/",
            environment=ARC_IMAGE.requirements(),
            answer_path=None,
            reward=FileReward(files=(RewardFile(path=ARCHIVE_REWARD_PATH, format=RewardFileFormat.NUMBER),)),
            timeout=float(tomllib.loads(task_toml.decode())["verifier"]["timeout_sec"]),
        ),
    )


def convert_tasktrove_inductive(row: RawRow) -> TaskSpec | ImportRejection:
    """The grader reads ``solution.py`` first and ``answer.txt`` as a fallback."""
    return _tasktrove_task(row, (SOLUTION_PATH, ANSWER_PATH), _check_transform_cases)


def convert_tasktrove_transductive(row: RawRow) -> TaskSpec | ImportRejection:
    return _tasktrove_task(row, (ANSWER_PATH,), _check_expected_grid)


def tasktrove_inductive_negative(_task: TaskSpec) -> WorkspaceFiles:
    return WorkspaceFiles({SOLUTION_PATH: FAILING_TRANSFORM.encode()})


def tasktrove_transductive_negative(_task: TaskSpec) -> WorkspaceFiles:
    return WorkspaceFiles({ANSWER_PATH: b"__incorrect_grid__\n"})


def convert_ultra_arc(row: RawRow) -> NormalizedTask | ImportRejection:
    """An Ultra NVARC reply: a fenced transform program (inductive) or an output grid (transductive)."""
    request = text_request(row.data, (INDUCTIVE_AGENT, TRANSDUCTIVE_AGENT))
    if isinstance(request, ImportRejection):
        return request
    config = {"contract": request.contract}
    if request.agent == TRANSDUCTIVE_AGENT:
        package = source_scorer_package(
            invocation=TRANSDUCTIVE_SCORER, config=config, environment=ARC_IMAGE.requirements(), timeout=REPLY_TIMEOUT
        )
    else:
        package = GraderPackage(
            ScriptGrader(
                argv=("python3", f"/tests/{ARC_GRADE}"),
                cwd="/",
                environment=ARC_IMAGE.requirements(),
                answer_path=ANSWER_PATH,
                reward=FileReward(files=(RewardFile(path=REPLY_REWARD_PATH, format=RewardFileFormat.JSON),)),
                timeout=REPLY_TIMEOUT,
            ),
            (
                inline_resource(ARC_GRADE, ARC_GRADE_BYTES),
                inline_resource("config.json", json.dumps(config, allow_nan=False).encode()),
            ),
        )
    return blend_task(row, request, package)


def grid_text(grid: list[list[int]]) -> str:
    """Space-separated rows, the grid layout the NVARC prompt requests."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def ultra_arc_golden(task: TaskSpec) -> Reply | None:
    """The expected grid; the source supplies no known-correct transform program."""
    contract = grader_config(task)["contract"]
    if contract["agent_ref"]["name"] == INDUCTIVE_AGENT:
        return None
    return answer_reply(task, grid_text(contract["expected_output"]))


def ultra_arc_negative(task: TaskSpec) -> Reply:
    """A raising transform, or the expected grid with its first cell changed."""
    contract = grader_config(task)["contract"]
    if contract["agent_ref"]["name"] == INDUCTIVE_AGENT:
        return answer_reply(task, f"```python\n{FAILING_TRANSFORM}```")
    wrong = [list(row) for row in contract["expected_output"]]
    wrong[0][0] = (wrong[0][0] + 1) % 10
    return answer_reply(task, grid_text(wrong))


ULTRA_ARC_CONTROLS = Controls(golden=ultra_arc_golden, negative=ultra_arc_negative, memory_mb=GRADER_MEMORY_MB)


def pipelines() -> list[RlDataPipeline]:
    return [
        RlDataPipeline(
            name="tasktrove-arc_inductive",
            source=tasktrove_source(INDUCTIVE_CONFIG),
            convert=convert_tasktrove_inductive,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=TASKTROVE_INDUCTIVE_RUBRIC,
            controls=Controls(negative=tasktrove_inductive_negative, memory_mb=GRADER_MEMORY_MB),
            atlas_id=f"Task Trove:{INDUCTIVE_CONFIG}",
        ),
        RlDataPipeline(
            name="tasktrove-arc_transductive",
            source=tasktrove_source(TRANSDUCTIVE_CONFIG),
            convert=convert_tasktrove_transductive,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=TASKTROVE_TRANSDUCTIVE_RUBRIC,
            controls=Controls(negative=tasktrove_transductive_negative, memory_mb=GRADER_MEMORY_MB),
            atlas_id=f"Task Trove:{TRANSDUCTIVE_CONFIG}",
        ),
    ]
