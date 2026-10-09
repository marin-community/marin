# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from verifyit.adapters.nemotron_pivot import grade_terminus, grade_tool_call


def _call(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": name, "arguments": json.dumps(arguments)}}],
    }


def _text(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": []}


RUN = {"type": "function_call", "name": "run", "arguments": json.dumps({"n": 3, "scale": 0.5, "ok": True})}
GREP = {"type": "function_call", "name": "bash", "arguments": json.dumps({"command": "grep -rn foo src"})}
ONE = {"type": "function_call", "name": "f", "arguments": json.dumps({"n": 1})}
MESSAGE = {"type": "message"}


@pytest.mark.parametrize(
    "expected,reply,scores",
    [
        (RUN, _call("run", {"n": 3, "scale": 0.5, "ok": True}), (1, 1, 1)),
        (RUN, _call("run", {"n": 3, "scale": 0.5000001, "ok": True}), (1, 1, 1)),
        (RUN, _call("run", {"n": 3, "scale": 0.51, "ok": True}), (1, 0, 0)),
        # NeMo Gym type-checks with isinstance: an int fails an expected float, a bool passes an int.
        (RUN, _call("run", {"n": 3, "scale": 1, "ok": True}), (1, 0, 0)),
        (ONE, _call("f", {"n": True}), (1, 1, 1)),
        (RUN, _call("other", {"n": 3, "scale": 0.5, "ok": True}), (0, 0, 0)),
        (RUN, {**_call("run", {"n": 3, "scale": 0.5, "ok": True}), "tool_calls": [{}, {}]}, (0, 0, 0)),
        (RUN, _text("done"), (0, 0, 0)),
        # Strings of two or more words match on any shared word.
        (GREP, _call("bash", {"command": "grep -n foo src/core.py"}), (1, 1, 0)),
        (MESSAGE, _text("All done."), (1, 1, 1)),
        (MESSAGE, _text("<think>never finished"), (1, 1, 0)),
    ],
)
def test_grade_tool_call_scores_tool_name_nemo_and_exact(expected, reply, scores):
    verdict = grade_tool_call(expected, reply)
    tool_name, nemo, exact = scores
    assert verdict.reward == nemo
    assert verdict.detail["components"] == {"tool_name": tool_name, "nemo": nemo, "exact": exact}


def _terminus(keystrokes: str, *, task_complete: bool | None = None) -> str:
    action = {"analysis": "a", "plan": "p", "commands": [{"keystrokes": keystrokes, "duration": 0.1}]}
    if task_complete is not None:
        action["task_complete"] = task_complete
    return json.dumps(action)


LS = _terminus("ls -la /workspace/project\n")
DONE = _terminus("echo done\n", task_complete=True)


@pytest.mark.parametrize(
    "expected,reply,threshold,scores",
    [
        (LS, LS, None, (1, 1, 1)),
        # Grug closes its reasoning with a special token, not </think>.
        (LS, "Let me look at the project first.<|end_think|>" + LS, None, (1, 1, 1)),
        (LS, _terminus("ls -la /workspace/project \n"), None, (1, 0, 1)),
        (LS, _terminus("ls -la /workspace/proj\n"), None, (1, 0, 1)),
        (LS, _terminus("ls -la /workspace/proj\n"), 0.99, (1, 0, 0)),
        (LS, _terminus("rm -rf /tmp/scratch\n"), None, (1, 0, 0)),
        (LS, LS[:-5], None, (0, 0, 0)),
        (LS, json.dumps({"analysis": "a", "plan": "p", "commands": [], "extra": 1}), None, (0, 0, 0)),
        (DONE, _terminus("echo done\n"), None, (0, 0, 0)),
        (DONE, DONE, None, (1, 1, 1)),
    ],
)
def test_grade_terminus_scores_schema_commands_and_similarity(expected, reply, threshold, scores):
    verdict = grade_terminus(expected, reply, threshold)
    schema_completion, exact_commands, string_90 = scores
    assert verdict.reward == string_90
    assert verdict.detail["components"] == {
        "schema_completion": schema_completion,
        "exact_commands": exact_commands,
        "string_90": string_90,
    }
