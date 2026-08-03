# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents/projects/luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import label_frozen_hierarchy as frozen  # noqa: E402
import label_frozen_hierarchy_training as training  # noqa: E402
from glm_semantic_labels import SampleDocument  # noqa: E402


def document(index: int, source: str = "source") -> SampleDocument:
    return SampleDocument(index, f"sha-{index}", source, "standard", index, f"text-{index}")


def test_documents_excluding_remove_all_specified_rows_and_reindex(monkeypatch: pytest.MonkeyPatch) -> None:
    pilot = [document(0), document(1)]
    candidates = [document(9), document(0), document(8), document(1), document(7)]
    requested_sizes = []

    def select_sample(_manifest, sample_size):
        requested_sizes.append(sample_size)
        return candidates

    monkeypatch.setattr(frozen, "select_sample", select_sample)

    selected = frozen.documents_excluding({"sources": {}}, pilot, sample_size=3)

    assert requested_sizes == [5]
    assert [row.eval_rank for row in selected] == [9, 8, 7]
    assert [row.sample_index for row in selected] == [0, 1, 2]


def test_documents_excluding_reject_missing_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frozen, "select_sample", lambda _manifest, _sample_size: [document(2), document(3)])

    with pytest.raises(ValueError, match="missing 1 excluded rows"):
        frozen.documents_excluding({"sources": {}}, [document(0)], sample_size=1)


def test_projection_training_documents_exclude_pilot_and_fixed_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [document(index) for index in range(4)]
    monkeypatch.setattr(frozen, "select_sample", lambda _manifest, _sample_size: candidates)

    selected = training.projection_training_documents(
        {"sources": {}},
        pilot_documents=[document(0)],
        evaluation_documents=[document(1)],
        training_size=2,
    )

    assert [row.eval_rank for row in selected] == [2, 3]
    assert [row.sample_index for row in selected] == [0, 1]
