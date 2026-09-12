# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from tasktrove_verify.grade import Status, main
from tasktrove_verify.grade import grade as dispatch
from tasktrove_verify.modes import grade_mcq
from tasktrove_verify.spec import McqSpec


def _answer(workspace: Path, text: str) -> None:
    (workspace / "answer.txt").write_text(text)


@pytest.mark.parametrize(
    "text, reward",
    [
        ("The third option fits.\nAnswer: C\n", 1.0),
        ("Answer: c", 1.0),
        ("Answer:C", 1.0),
        ("Answer : C", 1.0),
        ("Answer: B\n", 0.0),
        # Half of the Nemotron prompts ask for a boxed letter; models also decorate the line.
        ("Answer: \\boxed{C}\n", 1.0),
        ("**Answer:** C\n", 1.0),
        ("Answer: (C)\n", 1.0),
        ("Answer: `C`\n", 1.0),
        ("Answer: \\boxed{B}\n", 0.0),
        # The last stated answer wins: a model may revise itself.
        ("Answer: A\nOn reflection that is wrong.\nAnswer: C\n", 1.0),
        # Letters outside A..D cannot be the answer to a four-option question.
        ("Answer: E\n", 0.0),
        ("Answer: 3\n", 0.0),
        # Prose alone is not an answer.
        ("The answer is obviously C\n", 0.0),
        ("I worked through every option and settled on the third one.\nC\n", 0.0),
        ("", 0.0),
        ("   \n\n", 0.0),
    ],
)
def test_mcq_scores_only_a_stated_answer_line(tmp_path, text, reward):
    _answer(tmp_path, text)
    assert grade_mcq.grade(McqSpec(expected="C"), tmp_path, tmp_path).reward == reward


def test_mcq_missing_answer_file_scores_zero(tmp_path):
    result = grade_mcq.grade(McqSpec(expected="C"), tmp_path, tmp_path)
    assert result.reward == 0.0
    assert result.detail["reason"] == "no_output"


def test_mcq_reward_detail_carries_the_extracted_letter(tmp_path):
    _answer(tmp_path, "Answer: b\n")
    assert grade_mcq.grade(McqSpec(expected="C"), tmp_path, tmp_path).detail["extracted"] == "B"


def test_mcq_letter_beyond_the_option_count_is_wrong_not_a_task_defect(tmp_path):
    _answer(tmp_path, "Answer: E\n")
    result = grade_mcq.grade(McqSpec(expected="C", options=4), tmp_path, tmp_path)
    assert (result.status, result.reward, result.detail["extracted"]) == (Status.SCORED, 0.0, "E")


@pytest.mark.parametrize("spec", [McqSpec(expected="E", options=4), McqSpec(expected="A", options=0)])
def test_mcq_expected_outside_the_options_is_an_invalid_task(tmp_path, spec):
    _answer(tmp_path, "Answer: A\n")
    assert dispatch(spec, tmp_path, tmp_path).status == Status.INVALID_TASK


def test_mcq_fifth_option_is_gradable_when_declared(tmp_path):
    _answer(tmp_path, "Answer: E\n")
    assert grade_mcq.grade(McqSpec(expected="E", options=5), tmp_path, tmp_path).reward == 1.0


def test_cli_grades_an_mcq_task_and_writes_the_verdict(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "verifier.toml").write_text('mode = "mcq"\nexpected = "C"\noptions = 4\n')
    _answer(tmp_path, "Working through the options...\nAnswer: C\n")
    logs = tmp_path / "logs"

    assert main([str(tests_dir / "verifier.toml"), "--logs-dir", str(logs), "--workspace", str(tmp_path)]) == 0
    assert json.loads((logs / "verdict.json").read_text()) == {
        "reward": 1.0,
        "status": "scored",
        "detail": {"extracted": "C", "expected": "C"},
    }
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert (logs / "reward.txt").read_text() == "1.0\n"
