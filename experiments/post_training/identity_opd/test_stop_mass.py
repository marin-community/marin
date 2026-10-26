# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import itertools
import math

import pytest
from marinskyrl.distillation import TeacherPromptInsertion

from experiments.post_training.identity_opd.dry_run_report import END_OF_TURN
from experiments.post_training.identity_opd.stop_mass import stop_mass

RULE = TeacherPromptInsertion(after_prefix=(1, 2), tokens=(77, 78), after_marker=(9,))
TOOL_OPEN = [50, 51]


def _record(route, text, prompt, response):
    return {
        "trajectory": {"instance_id": text, "environment_extras": {"teacher_route": route}},
        "prompt": {"token_ids": prompt},
        "response": {"text": text, "token_ids": response},
    }


class Teacher:
    """Log-prob -1 for end of turn after token 5, -3 otherwise; -0.5 per tool-open token. Records each call."""

    def __init__(self):
        self.calls = []

    def __call__(self, token_ids):
        self.calls.append(token_ids)
        values = [None]
        for previous, token in itertools.pairwise(token_ids):
            if token == END_OF_TURN:
                values.append(-1.0 if previous == 5 else -3.0)
            elif token in TOOL_OPEN:
                values.append(-0.5)
            else:
                values.append(-9.0)
        return values


def test_stop_mass_scores_the_teacher_at_the_student_stop():
    records = [
        _record("tool_thinking", "plain answer", [1, 2, 3, 9, 4], [5, END_OF_TURN]),
        _record("tool_nonthinking", "other answer", [1, 2, 9, 4], [6, END_OF_TURN]),
        _record("tool_thinking", "<tool_call>call</tool_call>", [1, 2, 9], [7, END_OF_TURN]),
        _record("tool_thinking", "cut off", [1, 2, 9], [8, 8]),
        _record("identity_thinking", "not tool", [1, 2, 9], [5, END_OF_TURN]),
    ]
    teacher = Teacher()
    result = stop_mass(records, RULE, TOOL_OPEN, teacher)
    assert result["tool_samples"] == 4
    assert result["stopped_without_call"] == 2
    assert result["no_tool_call_fraction"] == 0.5
    first, second = result["samples"]
    assert first["end_of_turn_logprob"] == -1.0 and second["end_of_turn_logprob"] == -3.0
    assert first["tool_open_logprob"] == -1.0
    assert result["medians"]["end_of_turn_probability"] == pytest.approx((math.exp(-1) + math.exp(-3)) / 2)
    # The teacher sees the identity insertion after the header marker, then the response without its stop token.
    assert teacher.calls[0] == [1, 2, 3, 9, 77, 78, 4, 5, END_OF_TURN]
    assert teacher.calls[1] == [1, 2, 3, 9, 77, 78, 4, 5, *TOOL_OPEN]
    assert not result["flag"]


def test_stop_mass_flags_a_student_that_never_calls_when_the_teacher_would_stop_anyway():
    records = [_record("tool_thinking", f"answer {i}", [1, 2, 9], [5, END_OF_TURN]) for i in range(30)]
    assert stop_mass(records, RULE, TOOL_OPEN, Teacher())["flag"]


def test_stop_mass_needs_tool_rollouts():
    with pytest.raises(ValueError, match="no tool-family"):
        stop_mass([_record("identity_thinking", "x", [1, 2, 9], [5, END_OF_TURN])], RULE, TOOL_OPEN, Teacher())
