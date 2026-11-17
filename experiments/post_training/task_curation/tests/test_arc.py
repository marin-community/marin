# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC tasks, from TaskTrove archives and Nemotron Ultra rows, ship the NVARC scorer and its grade script."""

import json
from typing import cast

import pytest
from taskcompendium.models import (
    Source,
    VerifyitGrader,
    verifyit_spec,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow, WorkspaceFiles
from verifyit.grade import grade

from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.tests.conversion import (
    FIXTURE_GRADER_ENVIRONMENT,
    convert_row,
    converted_task,
    tasktrove_row,
)

RECIPES = {source.name: cast(CurationRecipe, source.config) for source in arc.sources()}
# The Nemotron Ultra declarations that use convert_ultra_arc name the grader packages.
ULTRA_CONTEXT = ConversionContext({}, FIXTURE_GRADER_ENVIRONMENT)
GRID = [[0, 1], [2, 9]]
ARCHIVE_FILES = {
    "task.toml": b"[verifier]\ntimeout_sec = 600.0\n",
    "tests/test.sh": b"#!/bin/bash\npython3 /tests/verifier.py > /logs/verifier/reward.txt\n",
    "tests/verifier.py": b"print(1)\n",
    "environment/Dockerfile": b"FROM python:3.11\nRUN pip install numpy scipy\n",
}


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


@pytest.mark.parametrize(
    ("answer", "reward"),
    [
        ("0 1\n2 9\n", 1.0),
        (" 0\t1  2 9 ", 1.0),
        ("\\boxed{0 1\n2 9}", 1.0),
        ("[[0, 1], [2, 9]]", 0.0),
        ("0 1\n2 8", 0.0),
    ],
)
def test_tasktrove_transductive_preserves_the_release_grid_comparison(answer, reward, tmp_path):
    task = converted_task(RECIPES["tasktrove-arc_transductive"], ROWS["tasktrove-arc_transductive"])
    (tmp_path / "answer.txt").write_text(answer)
    result = grade(verifyit_spec(cast(VerifyitGrader, task.grader)), tmp_path, tmp_path)
    assert result.reward == reward
    assert arc.tasktrove_golden(task) == WorkspaceFiles({"/app/answer.txt": b"0 1\n2 9\n"})


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
    result = convert_row(RECIPES[name], row)
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


def test_ultra_inductive_row_without_a_test_input_is_rejected():
    result = arc.convert_ultra_arc(ultra_row(arc.INDUCTIVE_AGENT, expected_output=GRID), ULTRA_CONTEXT)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.SOURCE_DEFECT, "invalid_verifier_data")
