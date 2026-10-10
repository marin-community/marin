# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow.parquet as pq
from marin.rl.nemotron_pivot import (
    CANDIDATES_FILENAME,
    PARTITION_BUCKETS,
    VALIDATION_FILENAME,
    Holdout,
    prepare_swe_candidates,
    prepare_terminal_candidates,
)


def _reasoning(text: str) -> dict:
    return {"type": "reasoning", "id": f"rs_{text}", "summary": [{"type": "summary_text", "text": text}]}


def _call(name: str, call_id: str) -> dict:
    return {"type": "function_call", "name": name, "arguments": json.dumps({"id": call_id}), "call_id": call_id}


def _output(call_id: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": f"ran {call_id}"}


def _expected(name: str, call_id: str) -> dict:
    return {"type": "function_call", "name": name, "arguments": json.dumps({"id": call_id})}


TASK = [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Fix the bug."}]
# Turn 0 narrates and calls; turn 1 calls two tools in parallel; turn 2 only narrates.
SWE_HISTORY = [
    *TASK,
    _reasoning("look around"),
    {"type": "message", "role": "assistant", "content": "Let me look."},
    _call("list_dir", "a"),
    _output("a"),
    _reasoning("read both"),
    _call("read_file", "b"),
    _call("read_file", "c"),
    _output("b"),
    _output("c"),
    _reasoning("summarize"),
    {"type": "message", "role": "assistant", "content": "I see the bug."},
    {"role": "user", "content": "Continue."},
]


def _swe_row(history: list[dict], expected: dict, *, trajectory_id: int = 7, step: int = 0) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "info": {"turn": 1, "step": step, "depth": step},
        "responses_create_params": {"input": history, "tools": [], "parallel_tool_calls": False},
        "expected_action": expected,
        "metadata": {"agent_cls": "CodeActAgent", "instance_id": f"repo__{trajectory_id}"},
        "agent_ref": {"name": "swe"},
        "pass_rate": 0.25,
        "pass_rate_total": 8,
        "pass_rate_passed": 2,
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _rows(path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def test_swe_trajectory_becomes_one_candidate_per_expert_turn_from_its_longest_copy(tmp_path):
    final = _expected("edit_file", "d")
    # The release repeats a trajectory at several depths; only the longest copy is laid out.
    shorter = _swe_row(SWE_HISTORY[:6], _expected("read_file", "b"), step=1)
    longest = _swe_row(SWE_HISTORY, final, step=3)
    _write_jsonl(tmp_path / "train.jsonl", [shorter, longest, shorter, longest])

    prepare_swe_candidates(
        release_path=str(tmp_path),
        release_filename="train.jsonl",
        output_path=str(tmp_path / "out"),
        holdout=Holdout(buckets=0, turns_per_trajectory=1),
    )

    rows = _rows(tmp_path / "out" / CANDIDATES_FILENAME)
    records = [json.loads(row["extra_info"]["nemotron_ultra"]["record_json"]) for row in rows]
    # Parallel calls are graded against the first call, as in the release; narration-only turns are skipped.
    assert [record["expected_action"] for record in records] == [
        _expected("list_dir", "a"),
        _expected("read_file", "b"),
        final,
    ]
    assert [len(row["prompt"]) for row in rows] == [2, 5, 10]
    # A turn's own reasoning is never in its prompt; earlier turns' reasoning is.
    second_prompt = rows[1]["prompt"]
    assert "read both" not in json.dumps(second_prompt)
    assert second_prompt[2]["reasoning_content"] == "look around"
    assert not any(key.startswith("pass_rate") for record in records for key in record)
    assert _rows(tmp_path / "out" / VALIDATION_FILENAME) == []


def test_validation_holds_out_whole_instances_and_spreads_its_turns(tmp_path):
    rows = [
        _swe_row(SWE_HISTORY, _expected("edit_file", "d"), trajectory_id=trajectory_id) for trajectory_id in range(40)
    ]
    _write_jsonl(tmp_path / "train.jsonl", rows)

    prepare_swe_candidates(
        release_path=str(tmp_path),
        release_filename="train.jsonl",
        output_path=str(tmp_path / "out"),
        holdout=Holdout(buckets=PARTITION_BUCKETS // 2, turns_per_trajectory=2),
    )

    train = _rows(tmp_path / "out" / CANDIDATES_FILENAME)
    validation = _rows(tmp_path / "out" / VALIDATION_FILENAME)
    train_instances = {row["extra_info"]["group"] for row in train}
    validation_instances = {row["extra_info"]["group"] for row in validation}
    assert train_instances and validation_instances
    assert not train_instances & validation_instances
    assert len(train_instances | validation_instances) == 40
    # Training keeps all three turns of a trajectory; validation keeps two, spread across it.
    assert len(train) == 3 * len(train_instances)
    assert {row["extra_info"]["index"] for row in validation} == {0, 2}
    assert len(validation) == 2 * len(validation_instances)


def test_terminal_reply_that_is_not_a_terminus_action_is_not_a_candidate(tmp_path):
    def action(keystrokes: str) -> str:
        return json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": keystrokes, "duration": 0.1}]})

    history = [
        {"role": "user", "content": "Solve the task."},
        {"role": "assistant", "content": action("ls\n")},
        {"role": "user", "content": "output"},
        {"role": "assistant", "content": '{"analysis": "unterminated'},
        {"role": "user", "content": "Previous response had parsing errors."},
    ]
    row = {
        "schema_version": "v1",
        "task_name": "task",
        "tool_name": "bash_command",
        "responses_create_params": {"input": history},
        "expected_answer": action("cat log\n"),
        "agent_ref": {"name": "terminus"},
        "metadata": {"source_trajectory_uid": "t", "pivot_agent_turn_index": 2, "total_source_agent_turns": 4},
    }
    _write_jsonl(tmp_path / "terminal.jsonl", [row])

    prepare_terminal_candidates(
        release_path=str(tmp_path),
        release_filename="terminal.jsonl",
        output_path=str(tmp_path / "out"),
        holdout=Holdout(buckets=0, turns_per_trajectory=1),
    )

    rows = _rows(tmp_path / "out" / CANDIDATES_FILENAME)
    answers = [json.loads(row["extra_info"]["nemotron_ultra"]["record_json"])["expected_answer"] for row in rows]
    assert answers == [action("ls\n"), action("cat log\n")]
    assert [len(row["prompt"]) for row in rows] == [1, 5]
