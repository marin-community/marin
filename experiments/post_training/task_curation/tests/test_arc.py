# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC tasks keep their archive's grader (TaskTrove) or the NVARC scorer (Nemotron Ultra)."""

import json

import pytest
from taskcompendium.grader import grader_config
from taskcompendium.models import AnswerType, ScriptGrader, Source, TextMessage
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow, Reply, WorkspaceFiles
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import arc
from experiments.post_training.task_curation.images import ARC_IMAGE
from experiments.post_training.task_curation.tests.conversion import convert_row, converted_task, tasktrove_row

PIPELINES = {pipeline.name: pipeline for pipeline in arc.pipelines()}
GRID = [[0, 1], [2, 9]]
TASK_TOML = b"[verifier]\ntimeout_sec = 600.0\n"
GRADER_FILES = {
    "task.toml": TASK_TOML,
    "tests/test.sh": b"#!/bin/bash\npython3 /tests/verifier.py > /logs/verifier/reward.txt\n",
    "tests/verifier.py": b"print(1)\n",
    "environment/Dockerfile": b"FROM python:3.11\nRUN pip install numpy scipy\n",
}


def archive(instruction: str, verifier_data: dict, **files: bytes) -> dict:
    return tasktrove_row(
        {
            "instruction.md": instruction.encode(),
            "tests/verifier_data.json": json.dumps(verifier_data).encode(),
            **GRADER_FILES,
            **files,
        }
    )


ROWS: dict[str, dict] = {
    "tasktrove-arc_inductive": archive(
        "Write transform(grid) in /app/solution.py.", {"test_cases": [{"input": GRID, "output": GRID}]}
    ),
    "tasktrove-arc_transductive": archive("Write the output grid to /app/answer.txt.", {"expected_output": GRID}),
}
OUTPUT_PATHS = {
    "tasktrove-arc_inductive": ("/app/solution.py", "/app/answer.txt"),
    "tasktrove-arc_transductive": ("/app/answer.txt",),
}
NEGATIVES = {
    "tasktrove-arc_inductive": "/app/solution.py",
    "tasktrove-arc_transductive": "/app/answer.txt",
}


@pytest.mark.parametrize("name", sorted(ROWS))
def test_tasktrove_arc_runs_the_archive_grader_on_the_agents_files(name):
    task = converted_task(PIPELINES[name], ROWS[name])
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert (grader.argv, grader.cwd, grader.answer_path, grader.timeout) == (
        ("bash", "/tests/test.sh"),
        "/",
        None,
        600.0,
    )
    assert grader.environment.docker_image == ARC_IMAGE.reference
    assert (task.answer_type, task.output_paths) == (AnswerType.FILE, OUTPUT_PATHS[name])
    assert {"test.sh", "verifier.py", "verifier_data.json"} <= {resource.path for resource in task.resources.verifier}
    controls = PIPELINES[name].controls
    assert controls is not None and controls.negative is not None and controls.golden is None
    negative = controls.negative(task)
    assert isinstance(negative, WorkspaceFiles) and set(negative.files) == {NEGATIVES[name]}


@pytest.mark.parametrize(
    ("name", "row", "kind", "reason"),
    [
        (
            "tasktrove-arc_transductive",
            tasktrove_row({"instruction.md": b"Solve.", **GRADER_FILES}),
            ImportFailureKind.SOURCE_DEFECT,
            "missing_input",
        ),
        (
            "tasktrove-arc_transductive",
            archive("Solve.", {"expected_output": [[0, 10]]}),
            ImportFailureKind.SOURCE_DEFECT,
            "invalid_verifier_data",
        ),
        (
            "tasktrove-arc_inductive",
            archive("Solve.", {"test_cases": [{"input": GRID, "output": [[1], [1, 2]]}]}),
            ImportFailureKind.SOURCE_DEFECT,
            "invalid_verifier_data",
        ),
        (
            "tasktrove-arc_inductive",
            tasktrove_row(
                {
                    "instruction.md": b"Solve.",
                    "tests/verifier_data.json": json.dumps({"test_cases": [{"input": GRID, "output": GRID}]}).encode(),
                    "task.toml": TASK_TOML,
                }
            ),
            ImportFailureKind.UNSUPPORTED,
            "missing_archive_grader",
        ),
    ],
)
def test_tasktrove_arc_rejects_rows_without_a_valid_reference_or_grader(name, row, kind, reason):
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


def test_ultra_transductive_controls_submit_the_expected_grid_and_a_changed_grid():
    result = arc.convert_ultra_arc(ultra_row(arc.TRANSDUCTIVE_AGENT, expected_output=GRID))
    assert not isinstance(result, ImportRejection)
    task = result.task
    assert grader_config(task)["contract"]["expected_output"] == GRID
    assert arc.ultra_arc_golden(task) == Reply(TextMessage(role="assistant", content="0 1\n2 9"))
    assert arc.ultra_arc_negative(task) == Reply(TextMessage(role="assistant", content="1 1\n2 9"))


def test_ultra_inductive_ships_the_grade_script_and_has_no_known_program():
    result = arc.convert_ultra_arc(ultra_row(arc.INDUCTIVE_AGENT, test_cases=[{"input": GRID, "output": GRID}]))
    assert not isinstance(result, ImportRejection)
    task = result.task
    assert isinstance(task.grader, ScriptGrader) and task.grader.argv == (
        "python3",
        f"/tests/{arc.ARC_GRADE}",
        "/tests/config.json",
        "/app/answer.txt",
        "/logs/verifier/score.json",
    )
    scripts = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    assert scripts[arc.ARC_GRADE] == arc.ARC_GRADE_BYTES
    assert arc.ultra_arc_golden(task) is None
    negative = arc.ultra_arc_negative(task)
    assert isinstance(negative.event, TextMessage) and "raise RuntimeError" in negative.event.content
