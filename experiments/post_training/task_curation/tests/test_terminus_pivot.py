# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shipped NeMo string-only grading contract."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "datasets"))
    return importlib.import_module("terminus_grade").score


@pytest.mark.parametrize(
    "candidate,expected_score",
    [
        (
            {
                "analysis": "Different reasoning",
                "plan": "Different plan",
                "commands": [{"keystrokes": "echo one\n"}, {"keystrokes": "echo two\n"}],
            },
            1.0,
        ),
        ({"analysis": "", "plan": "", "commands": [{"keystrokes": "rm -rf /tmp\n"}]}, 0.0),
        ({"analysis": "", "plan": "", "commands": [{"keystrokes": "echo one\necho two\n"}], "unexpected": True}, 0.0),
    ],
)
def test_string_only_scores_keystrokes_and_schema_instead_of_reasoning(scorer, candidate, expected_score):
    expected = {
        "analysis": "Reference reasoning",
        "plan": "Run both",
        "commands": [{"keystrokes": "echo one\necho two\n"}],
    }
    assert scorer(expected, json.dumps(candidate), "terminus_2", 0.9) == expected_score


def test_reference_completion_cannot_be_replaced_by_noncompletion(scorer):
    expected = {"analysis": "", "plan": "Done", "commands": [], "task_complete": True}
    candidate = {**expected, "task_complete": False}
    assert scorer(expected, json.dumps(candidate), "terminus_2", 0.9) == 0.0
    assert scorer(expected, json.dumps(expected), "terminus_2", 0.9) == 1.0
