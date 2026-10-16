# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from verifyit.candidate import candidate_spec, grade_candidate
from verifyit.grade import Status, run
from verifyit.spec import Mode, NextActionSpec, PivotToolCallSpec, TerminusSpec, parse_spec, render_spec

ROOT = "/workspace/acme__widgets__1.2"
GREP = {"type": "function_call", "name": "bash", "arguments": json.dumps({"command": "grep -rn foo src"})}
VIEW_CORE = {
    "name": "str_replace_editor",
    "arguments": json.dumps({"command": "view", "path": f"{ROOT}/widgets/core.py"}),
}


def _reply(name: str, arguments: dict) -> dict:
    call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


def _terminus(keystrokes: str) -> str:
    return json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": keystrokes, "duration": 0.1}]})


@pytest.mark.parametrize(
    "mode,parameters,candidate,reward",
    [
        # NeMo's word-overlap match accepts the same tool with overlapping arguments.
        (Mode.PIVOT_TOOL_CALL, {"expected_action": json.dumps(GREP)}, _reply("bash", {"command": "grep foo"}), 1.0),
        (Mode.PIVOT_TOOL_CALL, {"expected_action": json.dumps(GREP)}, _reply("edit", {"command": "grep foo"}), 0.0),
        (Mode.TERMINUS, {"expected_answer": _terminus("ls -la\n")}, _terminus("ls -la\n"), 1.0),
        (Mode.TERMINUS, {"expected_answer": _terminus("ls -la\n")}, _terminus("rm -rf build\n"), 0.0),
        # A cat of the file the expert viewed is the same view.
        (
            Mode.NEXT_ACTION,
            {"expected_call": json.dumps(VIEW_CORE), "repo_root": ROOT, "cwd": ROOT},
            _reply("execute_bash", {"command": f"cat {ROOT}/widgets/core.py"}),
            1.0,
        ),
        (
            Mode.NEXT_ACTION,
            {"expected_call": json.dumps(VIEW_CORE), "repo_root": ROOT, "cwd": ROOT},
            _reply("execute_bash", {"command": f"cat {ROOT}/README.md"}),
            0.0,
        ),
    ],
)
def test_pivot_modes_grade_in_process_with_the_adapters(mode, parameters, candidate, reward):
    verdict = grade_candidate(candidate_spec(mode, parameters), candidate, {})
    assert (verdict.status, verdict.reward) == (Status.SCORED, reward)


@pytest.mark.parametrize(
    "spec",
    [
        PivotToolCallSpec(expected_action=json.dumps({"type": "function_call", "name": "bash", "arguments": "[}"})),
        TerminusSpec(expected_answer=json.dumps({"commands": "ls"})),
        NextActionSpec(expected_call=json.dumps(VIEW_CORE), repo_root="relative", cwd=ROOT),
    ],
)
def test_pivot_modes_reject_unusable_expert_turns(spec):
    candidate = "x" if isinstance(spec, TerminusSpec) else _reply("bash", {})
    assert grade_candidate(spec, candidate, {}).status == Status.INVALID_TASK


def test_message_modes_require_an_assistant_message():
    spec = PivotToolCallSpec(expected_action=json.dumps(GREP))
    with pytest.raises(TypeError, match="assistant message"):
        grade_candidate(spec, "grep foo", {})


def test_file_grading_reads_the_message_from_the_answer_file(tmp_path):
    spec = PivotToolCallSpec(expected_action=json.dumps(GREP), output=str(tmp_path / "answer.txt"))
    assert parse_spec(render_spec(spec)) == spec
    (tmp_path / "verifier.toml").write_text(render_spec(spec))
    (tmp_path / "answer.txt").write_text(json.dumps(_reply("bash", {"command": "grep -n foo"})))
    assert run(tmp_path / "verifier.toml", tmp_path).reward == 1.0
    (tmp_path / "answer.txt").write_text("grep -n foo")
    assert run(tmp_path / "verifier.toml", tmp_path).detail["reason"] == "invalid_message"
