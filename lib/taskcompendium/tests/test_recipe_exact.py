# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from verifyit.grade import grade as dispatch

from taskcompendium.pipeline.datasets.grader_scripts.puzzle import puzzle_spec


def _answer(path, text):
    (path / "answer.txt").write_text(text)


def test_puzzle_ordered_list_accepts_line_breaks_but_rejects_reordering(tmp_path):
    spec = puzzle_spec("Defect, Salt, chair, donate", "ordered_list")
    _answer(tmp_path, "Defect\nSalt\nchair\ndonate")
    assert dispatch(spec, tmp_path, tmp_path).reward == 1.0
    _answer(tmp_path, "chair, Defect, donate, Salt")
    assert dispatch(spec, tmp_path, tmp_path).reward == 0.0
