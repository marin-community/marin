# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI-2 grid grading at the file and public reward boundaries."""

import json
import os
from pathlib import Path

import pytest
from tasktrove_verify.adapters.harbor_arc_agi_2 import grade_files, grade_grids, main
from tasktrove_verify.grade import InvalidTask


@pytest.mark.parametrize(
    ("candidate", "expected_reward"),
    [
        ([[0, 1], [9, 2]], 1.0),
        ([[0, 1], [8, 2]], 0.0),
        ([[0, 1]], 0.0),
        ([[0, 1], [9, 0]], 0.0),
        ([[False, 1], [9, 2]], 0.0),
        ([[0.0, 1], [9, 2]], 0.0),
        ([[float("nan"), 1], [9, 2]], 0.0),
        ([[float("inf"), 1], [9, 2]], 0.0),
        ([[10, 1], [9, 2]], 0.0),
    ],
)
def test_grid_equality_and_cell_types(candidate: object, expected_reward: float) -> None:
    assert grade_grids([[0, 1], [9, 2]], candidate).reward == expected_reward


def test_empty_and_ragged_trusted_grids_are_invalid() -> None:
    for expected in ([], [[]], [[1], [1, 2]]):
        with pytest.raises(InvalidTask):
            grade_grids(expected, expected)


def test_python_bool_equality_would_false_positive_in_source() -> None:
    expected = [[0, 1], [9, 2]]
    candidate = [[False, 1], [9, 2]]
    assert candidate == expected  # The source compares rows using Python equality.
    assert grade_grids(expected, candidate).reward == 0.0


@pytest.mark.parametrize("expected", [None, [[True]], [[1.0]], [[-1]], [[float("nan")]]])
def test_bad_protected_grid_is_invalid_task(expected: object) -> None:
    with pytest.raises(InvalidTask):
        grade_grids(expected, [[1]])


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_file_errors_and_stale_reward_are_fail_closed(tmp_path: Path, encoding: str) -> None:
    reference = tmp_path / "expected.json"
    candidate = tmp_path / "output.json"
    logs = tmp_path / "logs"
    reference.write_text("[[1]]", encoding="utf-8")
    candidate.write_text("[[1]]", encoding=encoding)
    assert main([str(reference), str(candidate), "--logs-dir", str(logs)]) == 0
    assert float((logs / "reward.txt").read_text()) == 1.0

    candidate.write_text("not JSON", encoding="utf-8")
    assert main([str(reference), str(candidate), "--logs-dir", str(logs)]) == 0
    assert grade_files(reference, candidate).reward == 0.0
    assert float((logs / "reward.txt").read_text()) == 0.0

    reference.write_text("[[true]]", encoding="utf-8")
    assert main([str(reference), str(candidate), "--logs-dir", str(logs)]) == 0
    verdict = json.loads((logs / "verdict.json").read_text())
    assert verdict["status"] == "invalid_task"
    assert not (logs / "reward.txt").exists()

    candidate.unlink()
    assert grade_files(reference, candidate).status == "invalid_task"


def test_missing_candidate_scores_zero_and_missing_reference_is_invalid(tmp_path: Path) -> None:
    reference = tmp_path / "expected.json"
    candidate = tmp_path / "output.json"
    reference.write_text("[[1]]", encoding="utf-8")
    assert grade_files(reference, candidate).reward == 0.0
    reference.unlink()
    assert grade_files(reference, candidate).status == "invalid_task"


def test_candidate_symlink_cannot_reuse_protected_reference(tmp_path: Path) -> None:
    reference = tmp_path / "expected.json"
    candidate = tmp_path / "output.json"
    reference.write_text("[[1]]", encoding="utf-8")
    candidate.symlink_to(reference)
    assert grade_files(reference, candidate).reward == 0.0


def test_protected_link_or_fifo_is_invalid_before_candidate_read(tmp_path: Path) -> None:
    actual = tmp_path / "actual.json"
    reference = tmp_path / "expected.json"
    candidate = tmp_path / "output.json"
    actual.write_text("[[1]]", encoding="utf-8")
    reference.symlink_to(actual)
    assert grade_files(reference, candidate).status == "invalid_task"
    reference.unlink()
    os.mkfifo(reference)
    assert grade_files(reference, candidate).status == "invalid_task"
    reference.unlink()
    reference.write_text("[[1]]", encoding="utf-8")
    os.mkfifo(candidate)
    assert grade_files(reference, candidate).reward == 0.0


def test_oversized_grid_file_is_rejected_before_json_parse(tmp_path: Path) -> None:
    reference = tmp_path / "expected.json"
    candidate = tmp_path / "output.json"
    reference.write_text("[[1]]", encoding="utf-8")
    candidate.write_bytes(b" " * 1_000_001)
    assert grade_files(reference, candidate).reward == 0.0
    reference.write_bytes(b" " * 1_000_001)
    assert grade_files(reference, candidate).status == "invalid_task"


def test_grid_parent_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    (real / "expected.json").write_text("[[1]]", encoding="utf-8")
    (real / "output.json").write_text("[[1]]", encoding="utf-8")
    assert grade_files(real / "expected.json", alias / "output.json").reward == 0.0
    assert grade_files(alias / "expected.json", real / "output.json").status == "invalid_task"
