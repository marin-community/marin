# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A script package runs its grade script beside the row's config and the scorer files it ships."""

from pathlib import Path

import pytest
from shellbox.machine import DockerImage, MachineSpec

from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    CommandSemantics,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.task_grading import grade_task

from .test_runtime import LocalGradingMachines

IMAGE = "grader@sha256:" + "c" * 64
ROW = RawRow("task", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})
SCORER = b"""
def compute_score(answer, words):
    found = [word in answer.split() for word in words]
    return sum(found) / len(found)
"""
# The grader runs from /, so these relative paths are /tests and /app/answer.txt.
GRADE = b"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "tests")
from colors.scoring import compute_score

config = json.loads(Path("tests/config.json").read_text())
print("scorer output before the reward")
print(compute_score(Path("app/answer.txt").read_text(), config["words"]))
"""


@pytest.fixture
def task(tmp_path) -> TaskSpec:
    scorers = tmp_path / "scorers"
    (scorers / "colors").mkdir(parents=True)
    (scorers / "colors/__init__.py").write_bytes(b"")
    (scorers / "colors/scoring.py").write_bytes(SCORER)
    script = tmp_path / "colors_grade.py"
    script.write_bytes(GRADE)
    package = script_package(
        grade_script(script, *shipped_files(scorers, "colors/__init__.py", "colors/scoring.py")),
        {"words": ["red", "blue"]},
        environment=EnvironmentRequirements(docker_image=IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS),
        timeout=30,
        answer_path="/app/answer.txt",
    )
    return conversation_task(ROW, events=(TextMessage(role="user", content="Name the flag's colors."),), package=package)


@pytest.mark.parametrize("answer,expected", [("red blue", 1.0), ("red green", 0.5), ("green", 0.0)])
def test_shipped_scorer_reward_is_the_grade(tmp_path: Path, task: TaskSpec, answer: str, expected: float):
    trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
    result = grade_task(
        task,
        GradingAttempt(trace),
        machine_factory=LocalGradingMachines(tmp_path / "machine"),
        machine_spec=MachineSpec(DockerImage(IMAGE)),
    )
    assert (result.status, result.reward) == (Outcome.GRADED, expected), result.error
