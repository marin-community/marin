# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest
from verifyit.grade import Status, grade
from verifyit.spec import parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.task_format import VERIFIER_TOML
from experiments.post_training.tasktrove.taskbinary import read_task_binary

FIXTURE = Path(__file__).parents[1] / "fixtures/e2egit_todo.tar.gz"
CORRECT_JS = """const tasks = [];
export function addTask(task) { if (!tasks.includes(task)) tasks.push(task); }
export function removeTask(task) {
    const index = tasks.indexOf(task);
    if (index < 0) return `Task '${task}' not found.`;
    tasks.splice(index, 1);
    return `Task '${task}' removed.`;
}
export function listTasks() { return [...tasks]; }
"""


@pytest.mark.parametrize("variant", ["correct", "duplicates", "resets_between_calls", "does_not_remove"])
def test_javascript_grader_isolates_scenarios_and_rejects_broken_operations(tmp_path, variant):
    info = SourceInfo("DCAgent__exp_rpt_e2egit-v2", SourceVerdict.KEEP, "unit-test-gen", "")
    record = convert_one(info, "e2egit-0433", FIXTURE.read_bytes(), converter_index(), "test-ref")
    assert record.status == ConvertStatus.CONVERTED
    task = read_task_binary(record.task_binary)
    task.write_to(tmp_path)
    workspace = tmp_path / "app"
    workspace.mkdir()
    solution = CORRECT_JS
    if variant == "duplicates":
        solution = solution.replace("if (!tasks.includes(task)) tasks.push(task)", "tasks.push(task)")
    elif variant == "resets_between_calls":
        solution = solution.replace("if (!tasks.includes(task))", "tasks.length = 0; if (!tasks.includes(task))")
    elif variant == "does_not_remove":
        solution = solution.replace("tasks.splice(index, 1);", "")
    (workspace / "todo_list.js").write_text(solution)
    spec = parse_spec(task.text(VERIFIER_TOML))
    spec = replace(spec, command=f"bash {tmp_path}/tests/todo_cases/run.sh")
    verdict = grade(spec, tmp_path / "tests", workspace)
    assert (verdict.status, verdict.reward) == (Status.SCORED, float(variant == "correct"))
    assert verdict.detail["total"] == 4
