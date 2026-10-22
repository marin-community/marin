# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest
from verifyit.adapters.harbor_answers import grade_answer, grade_files, main
from verifyit.grade import Status


@pytest.mark.parametrize(
    ("mode", "expected", "candidate", "score"),
    [
        ("aime", "42", " 42\n", 1.0),
        ("aime", "42", "\\boxed{42}", 0.0),
        ("gaia", "The\nAnswer", " theanSwer  ", 1.0),
        ("gaia", "A B", "AB", 0.0),
        ("satbench", "SAT", "[UNSAT] then [SAT]", 1.0),
        ("satbench", "SAT", "[SAT] then [UNSAT]", 0.0),
        ("gpqa-diamond", "B", " b \n", 1.0),
        ("gpqa-diamond", "B", "Answer: B", 0.0),
    ],
)
def test_harbor_answer_routes_preserve_source_extraction(mode, expected, candidate, score):
    reward = grade_answer(mode, expected, candidate)
    assert (reward.status, reward.reward) == (Status.SCORED, score)


def test_harbor_answer_cli_distinguishes_missing_candidate_from_invalid_reference(tmp_path):
    expected = tmp_path / "expected_answer.txt"
    candidate = tmp_path / "answer.txt"
    logs = tmp_path / "logs"
    expected.write_text("answer")
    assert main(["gaia", str(expected), str(candidate), "--logs-dir", str(logs)]) == 0
    assert json.loads((logs / "verdict.json").read_text())["status"] == "scored"
    assert (logs / "reward.txt").read_text().strip() == "0.0"
    expected.unlink()
    assert main(["gaia", str(expected), str(candidate), "--logs-dir", str(logs)]) == 0
    assert json.loads((logs / "verdict.json").read_text())["status"] == "invalid_task"
    assert not (logs / "reward.txt").exists()


def test_harbor_answer_cli_removes_prior_reward_after_candidate_read_failure(tmp_path):
    expected = tmp_path / "expected_answer.txt"
    candidate = tmp_path / "answer.txt"
    logs = tmp_path / "logs"
    expected.write_text("42")
    candidate.write_text("42")
    assert main(["aime", str(expected), str(candidate), "--logs-dir", str(logs)]) == 0
    assert (logs / "reward.txt").read_text().strip() == "1.0"
    candidate.write_bytes(b"\xff")
    assert main(["aime", str(expected), str(candidate), "--logs-dir", str(logs)]) == 0
    assert json.loads((logs / "verdict.json").read_text())["status"] == "infra_error"
    assert not (logs / "reward.txt").exists()


def test_harbor_answer_cli_rejects_candidate_alias_to_protected_reference(tmp_path: Path) -> None:
    expected = tmp_path / "expected_answer.txt"
    candidate = tmp_path / "answer.txt"
    logs = tmp_path / "logs"
    expected.write_text("New York")
    candidate.symlink_to(expected)

    assert main(["gaia", str(expected), str(candidate), "--logs-dir", str(logs)]) == 0
    verdict = json.loads((logs / "verdict.json").read_text())
    assert (verdict["status"], verdict["reward"]) == ("scored", 0.0)
    assert (logs / "reward.txt").read_text().strip() == "0.0"

    candidate.unlink()
    os.mkfifo(candidate)
    assert main(["gaia", str(expected), str(candidate), "--logs-dir", str(logs)]) == 0
    verdict = json.loads((logs / "verdict.json").read_text())
    assert (verdict["status"], verdict["reward"]) == ("scored", 0.0)

    expected.unlink()
    expected.symlink_to(candidate)
    assert main(["gaia", str(expected), str(candidate), "--logs-dir", str(logs)]) == 0
    assert json.loads((logs / "verdict.json").read_text())["status"] == "invalid_task"
    assert not (logs / "reward.txt").exists()


@pytest.mark.parametrize(
    ("mode", "reference"),
    [
        ("aime", "forty-two"),
        ("gaia", " \n "),
        ("gpqa-diamond", "E"),
        ("satbench", '{"expected_answer":"MAYBE"}'),
        ("satbench", "not JSON"),
        ("satbench", '{"expected_answer":"UNSAT","expected_answer":"SAT"}'),
        ("unknown", "42"),
    ],
)
def test_invalid_reference_precedes_redirected_candidate(tmp_path: Path, mode: str, reference: str) -> None:
    expected = tmp_path / "expected"
    candidate = tmp_path / "candidate"
    expected.write_text(reference)
    candidate.symlink_to(expected)

    reward = grade_files(mode, expected, candidate)
    assert reward.status == Status.INVALID_TASK
    assert reward.reward == 0.0


def test_answer_size_limit_applies_before_matching(tmp_path):
    expected = tmp_path / "expected"
    candidate = tmp_path / "candidate"
    expected.write_text("42")
    candidate.write_bytes(b"42" + b" " * 999_998)
    assert grade_files("gaia", expected, candidate).reward == 1.0
    with candidate.open("ab") as output:
        output.write(b" ")
    assert grade_files("gaia", expected, candidate).reward == 0.0
    expected.write_bytes(candidate.read_bytes())
    assert grade_files("gaia", expected, candidate).status == Status.INVALID_TASK


def test_answer_parent_link_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    (real / "expected").write_text("42")
    (real / "candidate").write_text("42")
    assert grade_files("gaia", real / "expected", alias / "candidate").reward == 0.0
    assert grade_files("gaia", alias / "expected", real / "candidate").status == Status.INVALID_TASK
