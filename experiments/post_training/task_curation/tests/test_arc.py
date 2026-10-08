# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC tasks, from TaskTrove archives and Nemotron Ultra rows, ship the NVARC scorer and its grade script."""

import json

import pytest
from taskcompendium.grader import grader_config
from taskcompendium.models import AnswerType, ScriptGrader, Source, StdoutReward, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow, Reply, WorkspaceFiles

from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.tests.conversion import (
    FIXTURE_GRADER_ENVIRONMENT,
    FIXTURE_GRADER_IMAGE,
    convert_row,
    converted_task,
    tasktrove_row,
)

PIPELINES = {pipeline.name: pipeline for pipeline in arc.pipelines()}
# The Nemotron Ultra declarations that use convert_ultra_arc name the grader image.
ULTRA_CONTEXT = ConversionContext({}, FIXTURE_GRADER_ENVIRONMENT)
GRID = [[0, 1], [2, 9]]
ARCHIVE_FILES = {
    "task.toml": b"[verifier]\ntimeout_sec = 600.0\n",
    "tests/test.sh": b"#!/bin/bash\npython3 /tests/verifier.py > /logs/verifier/reward.txt\n",
    "tests/verifier.py": b"print(1)\n",
    "environment/Dockerfile": b"FROM python:3.11\nRUN pip install numpy scipy\n",
}
SHIPPED = {
    "grade.py",
    "config.json",
    "local_sandbox.py",
    "skyrl_gym/__init__.py",
    "skyrl_gym/envs/__init__.py",
    "skyrl_gym/envs/aime/utils.py",
    "skyrl_gym/envs/nemotron_ultra/__init__.py",
    "skyrl_gym/envs/nemotron_ultra/answer_extraction.py",
    "skyrl_gym/envs/nemotron_ultra/nvarc.py",
    "skyrl_gym/envs/nemotron_ultra/sandbox.py",
}
"""The grade script, the row's record, and the NVARC scorer with the modules and package markers it imports."""


def archive(instruction: str, verifier_data: dict) -> dict:
    return tasktrove_row(
        {
            "instruction.md": instruction.encode(),
            "tests/verifier_data.json": json.dumps(verifier_data).encode(),
            **ARCHIVE_FILES,
        }
    )


ROWS: dict[str, dict] = {
    "tasktrove-arc_inductive": archive(
        "Write transform(grid) in /app/solution.py.", {"test_cases": [{"input": GRID, "output": GRID}]}
    ),
    "tasktrove-arc_transductive": archive("Write the output grid to /app/answer.txt.", {"expected_output": GRID}),
}
TASKTROVE = {
    "tasktrove-arc_inductive": (
        ("/app/solution.py", "/app/answer.txt"),
        {"mode": "inductive", "contract": {"test_input": GRID, "expected_output": GRID}},
        {"/app/solution.py": arc.literal_transform(GRID).encode()},
    ),
    "tasktrove-arc_transductive": (
        ("/app/answer.txt",),
        {"mode": "transductive", "contract": {"expected_output": GRID}},
        {"/app/answer.txt": b"0 1\n2 9\n"},
    ),
}


@pytest.mark.parametrize("name", sorted(ROWS))
def test_tasktrove_arc_grades_the_agents_files_with_nvarc(name):
    output_paths, config, golden = TASKTROVE[name]
    task = converted_task(PIPELINES[name], ROWS[name])
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert (grader.argv, grader.cwd, grader.answer_path, grader.reward) == (
        ("python3", "/tests/grade.py"),
        "/",
        None,
        StdoutReward(),
    )
    assert grader.environment.docker_image == FIXTURE_GRADER_IMAGE
    assert (task.answer_type, task.output_paths) == (AnswerType.FILE, output_paths)
    assert {resource.path for resource in task.resources.verifier} == SHIPPED
    assert grader_config(task) == config
    controls = PIPELINES[name].controls
    assert controls is not None and controls.golden is not None
    assert controls.golden(task) == WorkspaceFiles(golden)


@pytest.mark.parametrize(
    ("name", "row", "kind", "reason"),
    [
        (
            "tasktrove-arc_transductive",
            tasktrove_row({"instruction.md": b"Solve.", **ARCHIVE_FILES}),
            ImportFailureKind.SOURCE_DEFECT,
            "missing_input",
        ),
        (
            "tasktrove-arc_transductive",
            archive("Solve.", {"expected_output": [[0, 10]]}),
            ImportFailureKind.SOURCE_DEFECT,
            "reference_conflict",
        ),
        (
            "tasktrove-arc_inductive",
            archive("Solve.", {"test_cases": [{"input": GRID, "output": [[1], [1, 2]]}]}),
            ImportFailureKind.SOURCE_DEFECT,
            "reference_conflict",
        ),
        (
            "tasktrove-arc_inductive",
            archive("Solve.", {"test_cases": [{"input": GRID}]}),
            ImportFailureKind.SOURCE_DEFECT,
            "invalid_verifier_data",
        ),
        (
            "tasktrove-arc_inductive",
            archive("Solve.", {"test_cases": [{"input": GRID, "output": GRID}] * 2}),
            ImportFailureKind.UNSUPPORTED,
            "multiple_test_cases",
        ),
    ],
)
def test_tasktrove_arc_rejects_rows_nvarc_cannot_score(name, row, kind, reason):
    result = convert_row(PIPELINES[name], row)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (kind, reason)


def ultra_row(agent: str, **fields) -> RawRow:
    data = {
        "dataset": "ultra_sft_step3200_nvarc",
        "agent_ref": {"name": agent},
        "responses_create_params": {"input": [{"role": "user", "content": "Solve the puzzle."}]},
        **fields,
    }
    return RawRow("nvarc", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), data)


def reply(content: str) -> Reply:
    return Reply(TextMessage(role="assistant", content=content))


def test_ultra_transductive_golden_submits_the_expected_grid():
    result = arc.convert_ultra_arc(ultra_row(arc.TRANSDUCTIVE_AGENT, expected_output=GRID), ULTRA_CONTEXT)
    assert not isinstance(result, ImportRejection)
    task = result.task
    assert isinstance(task.grader, ScriptGrader) and task.grader.answer_path == "/app/answer.txt"
    assert grader_config(task)["mode"] == "transductive"
    assert arc.ultra_arc_golden(task) == reply("0 1\n2 9")


def test_ultra_inductive_golden_submits_a_literal_transform():
    row = ultra_row(arc.INDUCTIVE_AGENT, test_input=GRID, expected_output=GRID)
    result = arc.convert_ultra_arc(row, ULTRA_CONTEXT)
    assert not isinstance(result, ImportRejection)
    task = result.task
    assert {resource.path for resource in task.resources.verifier} == SHIPPED
    assert arc.ultra_arc_golden(task) == reply(f"```python\n{arc.literal_transform(GRID)}```")


def test_ultra_inductive_row_without_a_test_input_is_rejected():
    result = arc.convert_ultra_arc(ultra_row(arc.INDUCTIVE_AGENT, expected_output=GRID), ULTRA_CONTEXT)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.SOURCE_DEFECT, "invalid_verifier_data")
