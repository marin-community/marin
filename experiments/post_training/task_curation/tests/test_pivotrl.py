# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PivotRL candidates, as PivotRL's own preparation lays them out, become tasks whose expert turn scores 1."""

import json
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from marin.rl.nemotron_pivot import CANDIDATES_FILENAME, Holdout, prepare_swe_candidates, prepare_terminal_candidates
from marin.rl.openhands_pivot import candidate_rows
from rolloutengine.spec import LoweredTaskSpec
from taskcompendium.models import AnswerType, AssistantToolCalls, TaskSpec, TextMessage
from taskcompendium.pipeline.models import CheckStatus, ImportRejection, NormalizedTask
from taskcompendium.pipeline.verification import verify_task

from experiments.post_training.pivotrl.compendium import (
    CANDIDATE_TAG_PREFIX,
    LOWERED_TASK_COLUMN,
    TASKS_FILENAME,
    PivotSession,
    pivot_tasks,
)
from experiments.post_training.pivotrl.selection import TRAIN_FILENAME
from experiments.post_training.task_curation.datasets.pivotrl.tasks import sources
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.tests.conversion import convert_row

RECIPES = {source.name: cast(CurationRecipe, source.config) for source in sources()}
TOOLS = [
    {"type": "function", "name": name, "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}
    for name in ("list_dir", "edit_file")
]
TRAIN = Holdout(buckets=0, turns_per_trajectory=1)


def _converted(name: str, row: dict) -> NormalizedTask:
    result = convert_row(RECIPES[name], row)
    assert isinstance(result, NormalizedTask), result
    return result


def _checks(task: TaskSpec) -> dict[str, CheckStatus]:
    return {check.check: check.status for check in verify_task(task)}


def _swe_candidates(tmp_path, tools: list[dict]) -> list[dict]:
    history = [
        {"role": "system", "content": "You fix bugs."},
        {"role": "user", "content": "Fix the bug."},
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "look around"}]},
        {"type": "function_call", "name": "list_dir", "arguments": json.dumps({"id": "a"}), "call_id": "a"},
        {"type": "function_call_output", "call_id": "a", "output": "core.py"},
    ]
    row = {
        "trajectory_id": 7,
        "responses_create_params": {"input": history, "tools": tools, "parallel_tool_calls": False},
        "expected_action": {"type": "function_call", "name": "edit_file", "arguments": json.dumps({"id": "core.py"})},
        "metadata": {"instance_id": "repo__7"},
        "agent_ref": {"name": "swe"},
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    prepare_swe_candidates(
        release_path=str(tmp_path), release_filename="train.jsonl", output_path=str(tmp_path / "out"), holdout=TRAIN
    )
    return pq.read_table(tmp_path / "out" / CANDIDATES_FILENAME).to_pylist()


def test_swe_candidates_become_native_action_tasks_without_earlier_reasoning(tmp_path):
    first, last = _swe_candidates(tmp_path, TOOLS)
    converted = _converted("pivotrl-swe", last)
    task = converted.task
    assert task.answer_type == AnswerType.NATIVE_ACTION
    assert [tool.name for tool in task.final_tools] == ["list_dir", "edit_file"]
    assert CANDIDATE_TAG_PREFIX + last["extra_info"]["source_id"] in task.tags
    assert isinstance(task.context.events[2], AssistantToolCalls)
    # The candidate prompt keeps the teacher's reasoning; the task records dropping it.
    assert last["prompt"][2]["reasoning_content"] == "look around"
    assert [change.field for change in converted.changes] == ["prompt[*].reasoning_content"]
    assert _checks(task) == {"empty": CheckStatus.PASS, "reference": CheckStatus.PASS}
    assert _checks(_converted("pivotrl-swe", first).task) == {"empty": CheckStatus.PASS, "reference": CheckStatus.PASS}


def test_swe_candidate_without_tools_is_a_source_defect(tmp_path):
    rows = _swe_candidates(tmp_path, [])
    result = convert_row(RECIPES["pivotrl-swe"], rows[0])
    assert isinstance(result, ImportRejection) and result.reason == "missing_tools"


def test_terminal_candidates_become_text_tasks_graded_by_keystroke_similarity(tmp_path):
    def action(keystrokes: str) -> str:
        return json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": keystrokes, "duration": 0.1}]})

    row = {
        "schema_version": "1",
        "task_name": "count-files",
        "metadata": {"source_trajectory_uid": "t1", "pivot_agent_turn_index": 1},
        "responses_create_params": {
            "input": [
                {"role": "user", "content": "Count the files."},
                {"role": "assistant", "content": action("ls\n")},
                {"role": "user", "content": "a b c"},
            ]
        },
        "expected_answer": action("ls | wc -l\n"),
        "agent_ref": {"name": "terminus"},
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    prepare_terminal_candidates(
        release_path=str(tmp_path), release_filename="train.jsonl", output_path=str(tmp_path / "out"), holdout=TRAIN
    )
    tasks = [
        _converted("pivotrl-terminal", candidate).task
        for candidate in pq.read_table(tmp_path / "out" / CANDIDATES_FILENAME).to_pylist()
    ]
    assert [task.answer_type for task in tasks] == [AnswerType.TEXT, AnswerType.TEXT]
    assert isinstance(tasks[1].context.events[1], TextMessage)
    assert all(_checks(task) == {"empty": CheckStatus.PASS, "reference": CheckStatus.PASS} for task in tasks)


def test_openhands_candidates_become_next_action_tasks():
    root = "/workspace/acme__widgets"

    def bash(call_id: str, command: str) -> dict:
        function = {"name": "execute_bash", "arguments": json.dumps({"command": command})}
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": function}],
        }

    trajectory = [
        {"role": "user", "content": f"<uploaded_files>\n{root}\n</uploaded_files>\nFix it.", "tool_calls": None},
        bash("1", f"cd {root}/src && ls"),
        {
            "role": "tool",
            "content": f"core.py\n[Current working directory: {root}/src]",
            "tool_call_id": "1",
            "tool_calls": None,
        },
        bash("2", "cat core.py"),
    ]
    tools = json.dumps([{"type": "function", "function": {"name": "execute_bash", "parameters": {"type": "object"}}}])
    record = {"trajectory_id": "t", "instance_id": "i", "repo": "acme/widgets", "trajectory": trajectory}
    rows = candidate_rows(record, "train", tools, max_prompt_chars=100_000)
    task = _converted("pivotrl-openhands", rows[-1]).task
    assert task.grader.parameters["cwd"] == f"{root}/src"
    assert _checks(task) == {"empty": CheckStatus.PASS, "reference": CheckStatus.PASS}


def test_selected_pivots_become_lowered_tasks(tmp_path):
    candidates = _swe_candidates(tmp_path, TOOLS)
    tasks = [_converted("pivotrl-swe", candidate).task for candidate in candidates]
    curated = tmp_path / "curated" / "final"
    curated.mkdir(parents=True)
    pq.write_table(pa.table({"task_json": [task.model_dump_json() for task in tasks]}), curated / "part-00000.parquet")
    # The selection keeps the last candidate, in the candidate schema.
    pivots = tmp_path / "pivots"
    pivots.mkdir()
    pq.write_table(pq.read_table(tmp_path / "out" / CANDIDATES_FILENAME).slice(1), pivots / TRAIN_FILENAME)

    pivot_tasks(
        curated_path=str(tmp_path / "curated"),
        pivots_path=str(pivots),
        output_path=str(tmp_path / "tasks"),
        session=PivotSession(model_turn_timeout=None, cleanup_timeout=30.0),
    )

    rows = pq.read_table(tmp_path / "tasks" / TASKS_FILENAME).column(LOWERED_TASK_COLUMN).to_pylist()
    (lowered,) = [LoweredTaskSpec.model_validate_json(row) for row in rows]
    assert lowered.task == tasks[1]
    assert (lowered.session.task_session, lowered.session.max_turns) == ("shellbox", 1)
    manifest = json.loads((tmp_path / "tasks" / "manifest.json").read_text())
    assert manifest == {"selected_pivots": 1, "admitted_tasks": 2, "tasks": 1, "selected_not_admitted": 0}
