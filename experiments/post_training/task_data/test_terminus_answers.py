# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from experiments.post_training.chat_template import GRUG_TOKENS
from experiments.post_training.task_data.code_x import grade_x_task
from experiments.post_training.task_data.terminus_answers import AnswerFormat, answer_text, solution_writes

THINK_CLOSE = GRUG_TOKENS.think_close
# Two of Snowball B's real P_X replies (the part after its thinking): an input task answered
# with a heredoc plus `echo 'f(1000)' >> solution.txt`, an output task with `echo 51 > /app/solution.txt`.
B_REPLIES = json.loads((Path(__file__).parent / "testdata" / "b_terminus_replies.json").read_text())


def turn(analysis: str, *keystrokes: str) -> str:
    commands = ", ".join(f'{{"keystrokes": "{keys}", "duration": 0.1}}' for keys in keystrokes)
    return f'{{"analysis": "{analysis}", "plan": "write it", "commands": [{commands}], "task_complete": true}}'


@pytest.mark.parametrize("reply", B_REPLIES, ids=[reply["task"]["direction"] for reply in B_REPLIES])
def test_real_b_terminus_replies_grade_through_the_x_grader(reply):
    answer = answer_text(f"<|start_think|>work{THINK_CLOSE}{reply['reply_after_thinking']}", THINK_CLOSE)
    assert answer.format is AnswerFormat.TERMINUS
    assert grade_x_task(reply["task"], answer.text)
    if reply["task"]["direction"] == "output":  # the echoed value has no \boxed until it is extracted
        assert not grade_x_task(reply["task"], reply["reply_after_thinking"])


def test_a_single_backslash_boxed_survives_and_the_written_solution_comes_last():
    reply = f"thinking{THINK_CLOSE}" + turn(r"the sum is \boxed{6}", r"echo 7 > /app/solution.txt\n")
    answer = answer_text(reply, THINK_CLOSE)
    assert r"\boxed{6}" in answer.text and answer.text.rstrip().endswith(r"\boxed{7}")


def test_heredoc_bodies_are_written_solutions():
    keys = "cat > solution.txt << 'EOF'\nThe answer is \\boxed{12}.\nEOF\n"
    assert solution_writes(keys) == [r"The answer is \boxed{12}."]
    assert solution_writes("echo '\\boxed{3}' >> solution.txt") == [r"\boxed{3}"]  # already boxed: not wrapped


def test_a_plain_reply_is_used_as_it_is():
    answer = answer_text(f"thinking{THINK_CLOSE}So the answer is \\boxed{{4}}.", THINK_CLOSE)
    assert answer == answer_text("So the answer is \\boxed{4}.", THINK_CLOSE)
    assert answer.format is AnswerFormat.PLAIN and answer.text == "So the answer is \\boxed{4}."
